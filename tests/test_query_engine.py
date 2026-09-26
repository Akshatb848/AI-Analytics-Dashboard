"""Ask Data: question parsing, plan validation and query results."""
import pandas as pd
import pytest

from analytics.data_utils import generate_sample_data
from analytics.query_engine import SmartQueryEngine
from analytics.query_parser import apply_filters, normalize_intent


@pytest.fixture(scope="module")
def engine():
    return SmartQueryEngine(generate_sample_data(2000))


def plan(engine, query):
    return engine.process_query(query)["intent"]


@pytest.mark.parametrize("query, expected", [
    ("Total sales by region", dict(type="aggregate", aggregation="sum", metric="sales", groupby="region")),
    ("Top 5 category by revenue", dict(metric="revenue", groupby="category", limit=5, sort_order="desc")),
    ("Top 3 regions by profit", dict(metric="profit", groupby="region", limit=3, sort_order="desc")),
    ("Bottom 2 channels by sales", dict(metric="sales", groupby="channel", limit=2, sort_order="asc")),
    ("which region has the highest profit", dict(metric="profit", groupby="region", aggregation="sum")),
    ("average satisfaction score by segment", dict(aggregation="average", metric="satisfaction_score",
                                                   groupby="segment")),
    ("median cost per category", dict(aggregation="median", metric="cost", groupby="category")),
    ("total sales by region and channel", dict(groupby="region", groupby2="channel")),
    ("Compare sales across channels", dict(type="aggregate", metric="sales", groupby="channel")),
    ("how many orders by region", dict(aggregation="count", groupby="region")),
    ("how many records", dict(aggregation="count", groupby=None)),
    ("profit margin by channel", dict(aggregation="average", metric="profit_margin")),
    ("Show top performers and bottom performers", dict(type="aggregate", groupby="category")),
    ("Monthly growth rate of sales", dict(type="trend", metric="sales", time_grain="MS")),
    ("sales by month", dict(type="trend", metric="sales", time_grain="MS")),
    ("monthly sales trend by region", dict(type="trend", groupby="region", time_grain="MS")),
    ("Show profit trend over time", dict(type="trend", metric="profit", time_grain=None)),
    ("Correlation between sales and profit", dict(type="correlation", metric="sales", metric2="profit")),
    ("correlation between discount and satisfaction score",
     dict(type="correlation", metric="discount", metric2="satisfaction_score")),
    ("distribution of revenue", dict(type="distribution", metric="revenue")),
    ("anomalies in profit", dict(type="anomaly", metric="profit")),
    ("What are the key insights from this data?", dict(type="insight")),
])
def test_question_is_parsed_into_the_right_plan(engine, query, expected):
    intent = plan(engine, query)
    for key, value in expected.items():
        assert intent[key] == value, (query, key, intent)


@pytest.mark.parametrize("query, expected_filters", [
    ("sales where region is North", [("region", "==", "North")]),
    ("total revenue in North", [("region", "==", "North")]),
    ("average profit for Electronics", [("category", "==", "Electronics")]),
    ("sales where quantity > 40", [("quantity", ">", 40.0)]),
    ("average profit where discount >= 0.2 and region is West",
     [("region", "==", "West"), ("discount", ">=", 0.2)]),
    ("total sales in 2025", [("date", "year", 2025)]),
])
def test_filters_are_extracted(engine, query, expected_filters):
    intent = plan(engine, query)
    assert [(f["column"], f["operator"], f["value"]) for f in intent["filters"]] == expected_filters


def test_filters_change_the_answer(engine):
    df = engine.df
    result = engine.process_query("total sales in North")
    assert result["data"] == pytest.approx(df.loc[df["region"] == "North", "sales"].sum())
    assert "Region = North" in result["text"]


def test_top_n_returns_n_ranked_rows(engine):
    data = engine.process_query("Top 3 regions by profit")["data"]
    expected = engine.df.groupby("region")["profit"].sum().nlargest(3)
    assert list(data["region"]) == list(expected.index)


def test_unknown_filter_column_is_reported_not_silently_ignored(engine):
    result = engine.process_query("sales where city is Pune")
    assert "couldn't match 'city'" in result["narrative"]


def test_no_matching_rows_is_reported(engine):
    result = engine.process_query("sales where quantity > 100000")
    assert result["text"].startswith("No rows match")


def test_trend_with_too_few_complete_periods_says_so(engine):
    narrative = engine.process_query("yearly revenue")["narrative"]
    assert "not enough to measure growth" in narrative


@pytest.mark.parametrize("query", [
    "Top 5 category by revenue", "Bottom 2 channels by sales", "total sales by region and channel",
    "monthly sales trend by region", "Correlation between sales and profit", "sales where quantity > 40",
    "anomalies in profit where region is North", "how many orders by region", "weekly sales in 2025",
])
def test_queries_run_without_errors(engine, query):
    result = engine.process_query(query)
    assert result["success"], result["text"]
    assert "Error" not in result["text"]


# -----------------------------------------------------------------------------
# Plans from outside (e.g. an LLM) are validated before they run
# -----------------------------------------------------------------------------

def normalize(engine, raw):
    return normalize_intent(raw, engine.df, engine.numeric_cols, engine.categorical_cols, engine.date_col)


def test_normalize_drops_unknown_columns_and_operations(engine):
    intent = normalize(engine, {
        "type": "drop_table", "aggregation": "exec", "metric": "password", "groupby": "region",
        "filters": [
            {"column": "region", "operator": "==", "value": "North"},
            {"column": "nope", "operator": "==", "value": "x"},
            {"column": "sales", "operator": "__import__", "value": 1},
            {"column": "region", "operator": ">", "value": "abc"},
            {"column": "sales", "operator": ">", "value": {"$gt": 1}},
        ],
        "limit": 10 ** 9,
    })
    assert intent["type"] is None and intent["aggregation"] is None
    assert intent["metric"] is None
    assert intent["groupby"] == "region"
    assert intent["filters"] == [{"column": "region", "operator": "==", "value": "North"}]
    assert intent["limit"] is None


def test_normalize_matches_column_names_case_insensitively(engine):
    intent = normalize(engine, {"type": "aggregate", "metric": "SALES", "groupby": "Region"})
    assert intent["metric"] == "sales" and intent["groupby"] == "region"
    assert intent["aggregation"] == "sum"


def test_normalize_rejects_non_dict_input(engine):
    assert normalize(engine, "DROP TABLE")["type"] is None


def test_metric_must_be_numeric(engine):
    assert normalize(engine, {"type": "aggregate", "metric": "region"})["metric"] is None


def test_apply_filters_ignores_case_and_compares_numbers():
    df = pd.DataFrame({"city": ["Delhi", "delhi ", "Pune"], "sales": [1.0, 5.0, 9.0]})
    assert len(apply_filters(df, [{"column": "city", "operator": "==", "value": "DELHI"}])) == 2
    assert len(apply_filters(df, [{"column": "sales", "operator": ">=", "value": 5.0}])) == 2


def test_common_words_as_column_names_or_values_do_not_hijack_questions():
    df = pd.DataFrame({
        "a": ["x", "y"] * 10, "time": [1.0, 2.0] * 10,
        "status": ["in", "out"] * 10, "sales": range(20), "region": ["North", "South"] * 10,
    })
    engine = SmartQueryEngine(df)
    intent = engine.process_query("what is the total sales in a region")["intent"]
    assert intent["groupby"] == "region"
    assert intent["metric"] == "sales"
    assert intent["filters"] == []
