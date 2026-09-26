"""Turn a plain-English question into a structured query plan.

A plan ("intent") is a small dict that SmartQueryEngine knows how to run:

    {
        "type": "aggregate" | "trend" | "correlation" | "distribution"
                | "anomaly" | "forecast" | "insight" | None,
        "aggregation": "sum" | "average" | "max" | "min" | "count" | "median" | None,
        "metric": column | None,        # main numeric column
        "metric2": column | None,       # second numeric column (correlation)
        "groupby": column | None,
        "groupby2": column | None,      # optional second grouping column
        "filters": [{"column": ..., "operator": "==" | ">" | "<" | ">=" | "<=" | "year",
                     "value": ...}],
        "limit": int | None,            # top/bottom N
        "sort_order": "desc" | "asc",
        "time_grain": "D" | "W" | "MS" | "QS" | "YS" | None,
        "unmatched": [str],             # condition terms that match no column
    }

The rule-based parser below builds plans from keywords and the dataset's own
column names and category values. Plans from any other source (such as an
LLM) go through `normalize_intent`, which drops anything that does not match
the dataset, so a plan can never reference a missing column or an unknown
operation.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

INTENT_TYPES = {"aggregate", "trend", "correlation", "distribution", "anomaly", "forecast", "insight"}
AGGREGATIONS = {"sum", "average", "max", "min", "count", "median"}
OPERATORS = {"==", ">", "<", ">=", "<=", "year"}
TIME_GRAINS = {"D", "W", "MS", "QS", "YS"}

# Checked in order; the first match wins
_AGGREGATION_WORDS = [
    ("median", r"\bmedian\b"),
    ("average", r"\b(?:average|avg|mean|typical)\b"),
    ("count", r"\b(?:count|how many|number of)\b"),
    ("max", r"\b(?:max|maximum|peak)\b"),
    ("min", r"\b(?:min|minimum)\b"),
    ("sum", r"\b(?:total|sum|overall|combined)\b"),
]
# Ranking words sort grouped results rather than changing the aggregation
_RANK_DESC = r"\b(?:highest|largest|biggest|most|best|top)\b"
_RANK_ASC = r"\b(?:lowest|smallest|least|worst|bottom)\b"

_GRAIN_WORDS = {
    "daily": "D", "day": "D", "days": "D",
    "weekly": "W", "week": "W", "weeks": "W",
    "monthly": "MS", "month": "MS", "months": "MS",
    "quarterly": "QS", "quarter": "QS", "quarters": "QS",
    "yearly": "YS", "annual": "YS", "annually": "YS", "year": "YS", "years": "YS",
}
_GROUP_WORDS = r"(?:by|per|for each|each|across|grouped by|broken down by|split by)"

_COMPARATORS = [
    (">=", r">=|at least|no less than"),
    ("<=", r"<=|at most|no more than"),
    (">", r">|greater than|more than|higher than|above|over|exceeds?"),
    ("<", r"<|less than|lower than|below|under"),
    ("==", r"==|=|equals?|is"),
]

# Extra words people use for common columns: word -> part of the column name
_SYNONYMS = {
    "income": "revenue", "turnover": "revenue",
    "earnings": "profit",
    "expense": "cost", "expenses": "cost", "spend": "cost", "spending": "cost",
    "qty": "quantity", "units": "quantity", "volume": "quantity",
}

# Columns that hold rates or scores are averaged rather than summed by default
_RATE_COLUMN = re.compile(r"margin|rate|ratio|score|percent|pct|%|avg|average")

# Words that are never treated as column names or category values, so a column
# called "a" or a category "in" can't hijack ordinary questions
_STOPWORDS = {
    "a", "an", "the", "by", "in", "of", "is", "to", "for", "and", "or", "per", "on", "at",
    "top", "all", "each", "show", "what", "which", "how", "many", "much", "me", "with", "where",
    "vs", "over", "total", "sum", "average", "mean", "count", "trend", "time", "data",
}

# Categorical columns with more distinct values than this are not scanned for value mentions
_MAX_VALUES_TO_MATCH = 50


def empty_intent() -> Dict[str, Any]:
    return {
        "type": None,
        "aggregation": None,
        "metric": None,
        "metric2": None,
        "groupby": None,
        "groupby2": None,
        "filters": [],
        "limit": None,
        "sort_order": "desc",
        "time_grain": None,
        "unmatched": [],
    }


def _word_forms(word: str) -> set:
    forms = {word}
    if word.endswith("y") and len(word) > 2:
        forms.add(word[:-1] + "ies")
    elif word.endswith("s"):
        forms.add(word[:-1])
    else:
        forms.update({word + "s", word + "es"})
    return forms


def _to_number(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class QueryParser:
    """Rule-based parser that reads column names and category values from the data."""

    def __init__(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                 date_col: Optional[str]):
        self.df = df
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.date_col = date_col
        self._aliases = self._build_aliases()
        self._values = self._build_values()

    # ------------------------------------------------------------------ lookup tables

    def _build_aliases(self) -> List[Tuple[str, str]]:
        """(alias, column) pairs, longest alias first so 'profit margin' beats 'profit'."""
        aliases: Dict[str, str] = {}
        for col in self.df.columns:
            name = str(col).lower().strip()
            spaced = name.replace("_", " ")
            for form in _word_forms(spaced) | _word_forms(name):
                if len(form) >= 2 and form not in _STOPWORDS:
                    aliases.setdefault(form, col)
        for word, fragment in _SYNONYMS.items():
            target = next((c for c in self.df.columns if fragment in str(c).lower()), None)
            if target is not None:
                aliases.setdefault(word, target)
        return sorted(aliases.items(), key=lambda kv: -len(kv[0]))

    def _build_values(self) -> List[Tuple[str, str, Any]]:
        """(lowercase value, column, original value) for low-cardinality categorical columns."""
        values = []
        for col in self.categorical_cols:
            if col not in self.df.columns:
                continue
            uniques = self.df[col].dropna().unique()
            if len(uniques) > _MAX_VALUES_TO_MATCH:
                continue
            for value in uniques:
                text = str(value).strip().lower()
                if len(text) >= 2 and text not in _STOPWORDS and _to_number(text) is None:
                    values.append((text, col, value))
        return sorted(values, key=lambda v: -len(v[0]))

    # ------------------------------------------------------------------ parsing

    def parse(self, query: str) -> Dict[str, Any]:
        text = " " + query.lower().strip() + " "
        intent = empty_intent()
        taken = [False] * len(text)

        def claim(start: int, end: int) -> bool:
            if any(taken[start:end]):
                return False
            for i in range(start, end):
                taken[i] = True
            return True

        # Column mentions: (start, end, column)
        mentions = []
        for alias, col in self._aliases:
            for m in re.finditer(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", text):
                if claim(m.start(), m.end()):
                    mentions.append((m.start(), m.end(), col))
        mentions.sort()

        # Category values mentioned anywhere, e.g. "in North", "for Electronics"
        filters: List[Dict[str, Any]] = []
        for value_text, col, value in self._values:
            for m in re.finditer(r"(?<![\w])" + re.escape(value_text) + r"(?![\w])", text):
                if claim(m.start(), m.end()):
                    filters.append({"column": col, "operator": "==", "value": value})

        # Comparisons right after a column mention: "quantity > 40", "region is North"
        filter_cols = set()
        for start, end, col in mentions:
            after = text[end:]
            for op, words in _COMPARATORS:
                m = re.match(r"\s*(?:is\s+)?(?:" + words + r")\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w])", after)
                if m and col in self.numeric_cols:
                    filters.append({"column": col, "operator": op, "value": _to_number(m.group(1))})
                    filter_cols.add(col)
                    break
            else:
                if col in self.categorical_cols:
                    m = re.match(r"\s*(?:is|=|==|equals?)\s+['\"]?([\w .&/-]+?)['\"]?"
                                 r"(?=\s+(?:and|by|per|with|where|in|for)\b|[?.!,]|\s*$)", after)
                    if m and not any(f["column"] == col for f in filters):
                        filters.append({"column": col, "operator": "==", "value": m.group(1).strip()})
                        filter_cols.add(col)
        filter_cols |= {f["column"] for f in filters}

        # Year filter on the date column: "in 2024", "during 2025"
        year = re.search(r"\b(?:in|for|during|of)\s+((?:19|20)\d{2})\b", text)
        if year and self.date_col:
            filters.append({"column": self.date_col, "operator": "year", "value": int(year.group(1))})
        intent["filters"] = filters

        # Conditions on names that are not columns ("where city is Pune") are reported, not ignored
        for m in re.finditer(r"\b(?:where|with)\s+([a-z][\w ]*?)\s*(?:is|=|==|equals?|>=?|<=?)\s", text):
            if not any(taken[m.start(1):m.end(1)]):
                intent["unmatched"].append(m.group(1).strip())

        # Time grain: "monthly", "by month", "per week"
        for word, grain in _GRAIN_WORDS.items():
            if re.search(r"\b" + word + r"\b", text) and (
                word.endswith("ly") or re.search(_GROUP_WORDS + r"\s+(?:the\s+)?" + word + r"\b", text)
            ):
                intent["time_grain"] = grain
                break

        # Grouping: columns right after "by/per/across/each", chained with "and"
        groupbys: List[str] = []
        sort_metrics: List[str] = []
        previous_was_group = False
        for start, end, col in mentions:
            before = text[:start]
            is_group = re.search(_GROUP_WORDS + r"\s+(?:the\s+|all\s+)?$", before) or (
                previous_was_group and re.search(r"\b(?:and|,)\s*$", before)
            )
            if is_group:
                if col in self.numeric_cols:
                    sort_metrics.append(col)  # "top 5 category by revenue": revenue is the metric
                elif col not in groupbys:
                    groupbys.append(col)
            previous_was_group = bool(is_group)

        # Top/bottom N
        top = re.search(r"\b(?:top|best|highest|first)\s+(\d+)\b", text)
        bottom = re.search(r"\b(?:bottom|worst|lowest|last)\s+(\d+)\b", text)
        if top:
            intent["limit"], intent["sort_order"] = int(top.group(1)), "desc"
        elif bottom:
            intent["limit"], intent["sort_order"] = int(bottom.group(1)), "asc"

        # Categorical columns mentioned without a grouping word still group ("top 3 regions by profit")
        for _, _, col in mentions:
            if col in self.categorical_cols and col not in groupbys and col not in filter_cols:
                groupbys.append(col)
        groupbys = [g for g in groupbys if g != self.date_col]
        # "top performers" means ranking the main category
        if re.search(r"\bperformers?\b", text) and not groupbys and self.categorical_cols:
            groupbys = [self.categorical_cols[0]]
        intent["groupby"] = groupbys[0] if groupbys else None
        intent["groupby2"] = groupbys[1] if len(groupbys) > 1 else None

        # Metrics: numeric columns mentioned that are not only used as filters
        metrics = [col for _, _, col in mentions
                   if col in self.numeric_cols and col not in filter_cols and col not in groupbys]
        for col in sort_metrics:
            if col not in metrics:
                metrics.insert(0, col)
        metrics = list(dict.fromkeys(metrics))
        intent["metric"] = metrics[0] if metrics else None
        intent["metric2"] = metrics[1] if len(metrics) > 1 else None

        # Aggregation
        for agg, pattern in _AGGREGATION_WORDS:
            if re.search(pattern, text):
                intent["aggregation"] = agg
                break
        if intent["limit"] is None:
            if re.search(_RANK_ASC, text) and not re.search(_RANK_DESC, text):
                intent["sort_order"] = "asc"
        if intent["aggregation"] is None and re.search(_RANK_DESC + "|" + _RANK_ASC, text) and not groupbys:
            intent["aggregation"] = "max" if re.search(_RANK_DESC, text) else "min"

        intent["type"] = self._classify(text, intent)
        if intent["type"] == "aggregate" and intent["aggregation"] is None:
            rate_like = intent["metric"] and _RATE_COLUMN.search(str(intent["metric"]).lower())
            intent["aggregation"] = "average" if rate_like else "sum"
        return intent

    def _classify(self, text: str, intent: Dict[str, Any]) -> Optional[str]:
        if re.search(r"\b(?:correlat\w*|relationship|related|relation)\b", text):
            return "correlation"
        if re.search(r"\b(?:forecast\w*|predict\w*|projection|future)\b", text):
            return "forecast"
        if re.search(r"\b(?:anomal\w*|outliers?|unusual|abnormal)\b", text):
            return "anomaly"
        if re.search(r"\b(?:distribution|histogram|spread)\b", text):
            return "distribution"
        if self.date_col and (
            intent["time_grain"]
            or re.search(r"\b(?:trends?|over time|time series|growth|evolution|timeline)\b", text)
        ):
            return "trend"
        has_target = intent["metric"] or intent["groupby"] or intent["filters"]
        if re.search(r"\bperformers?\b", text):
            return "aggregate"
        if not has_target and re.search(
            r"\b(?:insights?|findings?|summar\w*|overview|key|important|interesting|explain|why|"
            r"tell me about|stands? out)\b", text
        ):
            return "insight"
        if has_target or intent["aggregation"] or re.search(r"\bcompare\b", text):
            return "aggregate"
        return None


def normalize_intent(raw: Any, df: pd.DataFrame, numeric_cols: List[str],
                     categorical_cols: List[str], date_col: Optional[str]) -> Dict[str, Any]:
    """Validate a plan against the dataset, dropping anything that doesn't fit.

    Column names are matched case-insensitively; unknown columns, operators and
    values of the wrong type are discarded rather than trusted.
    """
    intent = empty_intent()
    if not isinstance(raw, dict):
        return intent
    by_lower = {str(c).lower(): c for c in df.columns}

    def column(value: Any, allowed: Optional[List[str]] = None) -> Optional[str]:
        if not isinstance(value, str):
            return None
        col = by_lower.get(value.strip().lower())
        if col is None or (allowed is not None and col not in allowed):
            return None
        return col

    if raw.get("type") in INTENT_TYPES:
        intent["type"] = raw["type"]
    if raw.get("aggregation") in AGGREGATIONS:
        intent["aggregation"] = raw["aggregation"]
    intent["metric"] = column(raw.get("metric"), numeric_cols)
    intent["metric2"] = column(raw.get("metric2"), numeric_cols)
    groupable = list(categorical_cols) + [c for c in df.columns if c not in numeric_cols and c != date_col]
    intent["groupby"] = column(raw.get("groupby"), groupable)
    intent["groupby2"] = column(raw.get("groupby2"), groupable)
    if intent["groupby2"] == intent["groupby"]:
        intent["groupby2"] = None
    if raw.get("sort_order") in ("asc", "desc"):
        intent["sort_order"] = raw["sort_order"]
    limit = raw.get("limit")
    if isinstance(limit, (int, float)) and not isinstance(limit, bool) and 0 < limit <= 1000:
        intent["limit"] = int(limit)
    if raw.get("time_grain") in TIME_GRAINS:
        intent["time_grain"] = raw["time_grain"]

    filters = raw.get("filters") or []
    if isinstance(filters, list):
        for f in filters[:10]:
            if not isinstance(f, dict):
                continue
            col = column(f.get("column"))
            op = f.get("operator", "==")
            value = f.get("value")
            if col is None or op not in OPERATORS or isinstance(value, (list, dict)) or value is None:
                continue
            if op == "year":
                year = _to_number(value)
                if col != date_col or year is None:
                    continue
                value = int(year)
            elif op != "==":
                value = _to_number(value)
                if col not in numeric_cols or value is None:
                    continue
            intent["filters"].append({"column": col, "operator": op, "value": value})

    unmatched = raw.get("unmatched") or []
    if isinstance(unmatched, list):
        intent["unmatched"] = [str(u)[:50] for u in unmatched[:5] if isinstance(u, str)]

    if intent["type"] == "aggregate" and intent["aggregation"] is None:
        intent["aggregation"] = "sum"
    return intent


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply validated plan filters; text comparisons ignore case."""
    for f in filters:
        col, op, value = f["column"], f["operator"], f["value"]
        if col not in df.columns:
            continue
        series = df[col]
        if op == "year":
            if pd.api.types.is_datetime64_any_dtype(series):
                df = df[series.dt.year == int(value)]
            continue
        if op == "==":
            number = _to_number(value)
            if pd.api.types.is_numeric_dtype(series) and number is not None:
                df = df[series == number]
            else:
                df = df[series.astype(str).str.strip().str.lower() == str(value).strip().lower()]
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        df = df[{">": series > value, "<": series < value,
                 ">=": series >= value, "<=": series <= value}[op]]
    return df


def describe_filters(filters: List[Dict[str, Any]]) -> str:
    parts = []
    for f in filters:
        col = str(f["column"]).replace("_", " ").title()
        if f["operator"] == "year":
            parts.append(f"{col} in {f['value']}")
        elif f["operator"] == "==":
            parts.append(f"{col} = {f['value']}")
        else:
            parts.append(f"{col} {f['operator']} {f['value']:,g}")
    return ", ".join(parts)
