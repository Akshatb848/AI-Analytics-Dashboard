"""Smoke tests: run the Streamlit app headlessly and exercise the core engines."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app as dashboard

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 180

warnings.filterwarnings("ignore")


def run_app(**session_state) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in session_state.items():
        at.session_state[key] = value
    at.run()
    return at


def assert_no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


# -----------------------------------------------------------------------------
# End-to-end app runs
# -----------------------------------------------------------------------------

def test_app_runs_on_sample_data():
    at = run_app()
    assert_no_errors(at)
    warnings_shown = [w.value for w in at.warning]
    assert not any("No date column" in w for w in warnings_shown)


@pytest.mark.parametrize("query", [
    "Total sales by region",
    "Show sales trend over time",
    "Correlation between sales and profit",
    "Identify anomalies and outliers",
    "distribution of profit",
    "What are the key insights from this data?",
])
def test_ask_data_queries(query):
    at = run_app()
    at.text_input(key="nl_query").input(query).run()
    assert_no_errors(at)
    rendered = " ".join(m.value for m in at.markdown)
    assert "Error processing query" not in rendered
    assert "Cannot perform trend analysis" not in rendered


def test_forecast_generates():
    at = run_app()
    button(at, "🔮 Generate Forecast").click().run()
    assert_no_errors(at)
    assert any(m.label == "Predicted Avg" for m in at.metric)
    assert any(m.label == "Typical error (MAPE)" for m in at.metric)


def test_dashboard_card_can_be_removed():
    at = run_app()
    at.text_input(key="nl_query").input("Total sales by region").run()
    button(at, "Save card").click().run()
    assert len(at.session_state["saved_dashboards"]) == 1

    button(at, "Remove card").click().run()
    assert_no_errors(at)
    assert at.session_state["saved_dashboards"] == []


def test_data_tools_changes_are_used_by_the_app():
    at = run_app()
    button(at, "Generate Date Features").click().run()
    assert_no_errors(at)

    processed = at.session_state["processed_data"]["sample:2000"]
    assert {"year", "month", "quarter"} <= set(processed.columns)
    assert any("Using cleaned data" in c.value for c in at.caption)

    button(at, "↩️ Reset to original data").click().run()
    assert_no_errors(at)
    assert "sample:2000" not in at.session_state["processed_data"]


def test_uploaded_dataset_without_numeric_columns():
    df = pd.DataFrame({"city": ["Delhi", "Pune", "Mumbai"] * 5, "team": ["A", "B", "C"] * 5})
    at = run_app(datasets={"text_only.csv": df}, active_dataset="text_only.csv")
    at.radio[0].set_value("Upload File").run()
    assert_no_errors(at)
    assert any("No numeric KPIs" in i.value for i in at.info)


# -----------------------------------------------------------------------------
# Engine / helper units
# -----------------------------------------------------------------------------

def messy_upload() -> pd.DataFrame:
    # read_csv on pandas 3 gives StringDtype text columns, like a real upload
    return pd.DataFrame({
        "order_date": ["2024-01-01", "2024-01-02", None, "2024-01-05"] * 10,
        "city": ["Delhi", "Mumbai", None, "Pune"] * 10,
        "amount": ["1,200", "₹3,400", "", "55%"] * 10,
        "notes": ["fast, cheap", "late", "ok", "n/a"] * 10,
    }).astype("string")


def test_sanitize_converts_numbers_and_dates():
    df = dashboard.sanitize_dataframe(messy_upload())
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert pd.api.types.is_float_dtype(df["amount"])
    assert df["amount"].iloc[:2].tolist() == [1200.0, 3400.0]
    # Free text is left untouched (commas are not stripped)
    assert df["notes"].iloc[0] == "fast, cheap"
    assert dashboard.detect_categorical_columns(df) == ["city", "notes"]


def test_detects_non_nanosecond_datetime_columns():
    df = dashboard.generate_sample_data(200)
    df["date"] = df["date"].astype("datetime64[us]")
    assert dashboard.detect_date_column(df) == "date"


def test_numeric_time_column_is_not_a_date():
    df = pd.DataFrame({"delivery_time": [1.5, 2.0, 3.25], "city": ["a", "b", "c"]})
    assert dashboard.detect_date_column(df) is None
    assert dashboard.detect_numeric_columns(df) == ["delivery_time"]


@pytest.mark.parametrize("method", ["iqr", "zscore"])
def test_outlier_removal_keeps_rows_with_missing_values(method):
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 2.5, 1.5, 1000.0], "y": list("abcdef")})
    result = dashboard.DataPreprocessor(df).remove_outliers(method=method, threshold=1.5)
    out = result.get_transformed_data()
    assert 1000.0 not in out["x"].tolist()
    assert out["x"].isna().sum() == 1


@pytest.mark.parametrize("strategy", ["auto", "mean", "median", "mode", "zero"])
def test_missing_value_handling_fills_nulls(strategy):
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": pd.Series(["a", None, "a"], dtype="string")})
    out = dashboard.DataPreprocessor(df).handle_missing_values(strategy).get_transformed_data()
    assert out["x"].isna().sum() == 0
    # mean/median only apply to numeric columns; text columns are left as-is
    expected_text_nulls = 1 if strategy in ("mean", "median") else 0
    assert out["y"].isna().sum() == expected_text_nulls


# -----------------------------------------------------------------------------
# Security, upload limits and caching
# -----------------------------------------------------------------------------

XSS = "<img src=x onerror=alert(1)>"


def dataset_with_malicious_category() -> pd.DataFrame:
    # A large gap between the two groups produces a "performance gap" insight
    # whose title, description and narrative all include the category value
    return pd.DataFrame({
        "region": [XSS] * 60 + ["North"] * 60,
        "sales": [1000.0 + i for i in range(60)] + [10.0 + i for i in range(60)],
    })


def test_safe_html_escapes_markup_but_keeps_highlight_spans():
    narrative = f"<span class='narrative-highlight'>{XSS}</span> & more"
    assert dashboard.safe_html(narrative) == (
        "<span class='narrative-highlight'>&lt;img src=x onerror=alert(1)&gt;</span> &amp; more"
    )


def test_uploaded_values_are_escaped_in_the_app():
    at = run_app(datasets={"evil.csv": dataset_with_malicious_category()}, active_dataset="evil.csv")
    at.radio[0].set_value("Upload File").run()
    assert_no_errors(at)
    raw_html = [m.value for m in at.markdown if m.allow_html]
    assert not any("<img" in block for block in raw_html)
    assert any("&lt;img src=x onerror=alert(1)&gt;" in block for block in raw_html)


def test_html_report_escapes_uploaded_values():
    df = dataset_with_malicious_category()
    insights, recommendations = dashboard.EnhancedInsightsEngine(df).generate_all_insights()
    assert any(XSS in i["description"] for i in insights)
    report = dashboard.ReportGenerator(df, insights, recommendations).generate_html_report()
    assert "<img" not in report
    assert "&lt;img src=x onerror=alert(1)&gt;" in report


@pytest.mark.parametrize("payload, message", [
    ([1, 2], "must be a JSON object"),
    ({"metrics": "revenue"}, "'metrics' must be a list"),
    ({"metrics": ["revenue"]}, "'metrics' must be a list"),
    ({"dimensions": "region"}, "'dimensions' must be a list"),
    ({"time_column": ["date"]}, "'time_column' must be a column name"),
    ({"hierarchies": []}, "'hierarchies' must be an object"),
])
def test_semantic_catalog_rejects_malformed_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        dashboard.SemanticCatalog.from_dict(payload)


def test_upload_is_capped_at_max_rows(monkeypatch):
    monkeypatch.setattr(dashboard, "MAX_UPLOAD_ROWS", 10)
    dashboard.load_uploaded_file.clear()
    csv = "date,amount\n" + "\n".join(f"2024-01-{d:02d},\"1,{d:03d}\"" for d in range(1, 26))
    loaded = dashboard.load_uploaded_file(csv.encode(), "orders.csv")
    assert loaded["truncated"]
    assert len(loaded["df"]) == 10
    assert pd.api.types.is_float_dtype(loaded["df"]["amount"])


def test_excel_upload_loads():
    buffer = pd.io.common.BytesIO()
    pd.DataFrame({"city": ["Delhi", "Pune"], "sales": [1.0, 2.0]}).to_excel(buffer, index=False)
    loaded = dashboard.load_uploaded_file(buffer.getvalue(), "sales.xlsx")
    assert not loaded["truncated"]
    assert loaded["df"]["sales"].tolist() == [1.0, 2.0]


def test_fingerprint_detects_a_single_changed_value_in_a_large_frame():
    # Streamlit's own hashing samples rows of frames this size
    df = pd.DataFrame({"x": np.arange(150_000, dtype="float64")})
    changed = df.copy()
    changed.loc[123_456, "x"] = -1.0
    assert dashboard.data_fingerprint(df) != dashboard.data_fingerprint(changed)
    assert dashboard.data_fingerprint(df) == dashboard.data_fingerprint(df.copy())
