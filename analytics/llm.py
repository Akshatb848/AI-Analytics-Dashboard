"""Optional LLM query planning for Ask Data, using Z.ai's GLM models.

The model never runs code or sees raw rows. It receives the dataset's schema
(column names, types, a few category values, the date range) and the user's
question, and replies with a JSON query plan in the format described in
`analytics.query_parser`. The plan is validated by `normalize_intent` before
anything runs, and the built-in parser is used whenever the model is not
configured or its reply can't be used.

Configuration (environment variables or Streamlit secrets):
    ZAI_API_KEY    required to enable the model
    ZAI_BASE_URL   default https://api.z.ai/api/paas/v4
    ZAI_MODEL      default glm-4.5-flash (free tier)
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_MODEL = "glm-4.5-flash"
# Category values shown to the model per column, and their maximum length
_MAX_VALUES_PER_COLUMN = 15
_MAX_VALUE_LENGTH = 40


class LLMError(Exception):
    """The model could not be reached or its reply could not be used."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = 30.0

    @classmethod
    def from_env(cls, secrets: Optional[Callable[[str], Optional[str]]] = None) -> Optional["LLMConfig"]:
        """Read settings from the environment, then from `secrets` (e.g. st.secrets.get).

        Returns None when no API key is configured.
        """
        def setting(name: str) -> Optional[str]:
            value = os.environ.get(name)
            if not value and secrets is not None:
                try:
                    value = secrets(name)
                except Exception:  # no secrets file, or an unreadable one
                    value = None
            return str(value).strip() if value else None

        api_key = setting("ZAI_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            base_url=(setting("ZAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            model=setting("ZAI_MODEL") or DEFAULT_MODEL,
        )

    def __repr__(self) -> str:  # never print the key
        return f"LLMConfig(base_url={self.base_url!r}, model={self.model!r})"


class GLMClient:
    """Minimal client for Z.ai's OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 600) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Planning is a short structured task; skipping chain-of-thought keeps it fast
            "thinking": {"type": "disabled"},
        }
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"could not reach the model API ({type(e).__name__})") from None

        if response.status_code != 200:
            detail = ""
            try:
                error = response.json().get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else str(error)
            except ValueError:
                pass
            raise LLMError(f"model API returned HTTP {response.status_code}"
                           + (f": {detail[:200]}" if detail else ""))
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("unexpected response format from the model API") from None


def describe_schema(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str],
                    date_col: Optional[str]) -> Dict[str, Any]:
    """The only information about the data that is sent to the model (no raw rows)."""
    columns = []
    for col in df.columns:
        entry: Dict[str, Any] = {"name": str(col)}
        if col == date_col:
            entry["kind"] = "date"
            dates = df[col].dropna()
            if len(dates):
                entry["range"] = [str(dates.min())[:10], str(dates.max())[:10]]
        elif col in numeric_cols:
            entry["kind"] = "numeric"
        elif col in categorical_cols:
            entry["kind"] = "category"
            values = df[col].dropna().astype(str).unique()[:_MAX_VALUES_PER_COLUMN]
            entry["values"] = [v[:_MAX_VALUE_LENGTH] for v in values]
        else:
            entry["kind"] = "other"
        columns.append(entry)
    return {"row_count": int(len(df)), "columns": columns}


_SYSTEM_PROMPT = """You convert questions about a table into a JSON query plan.
Reply with one JSON object and nothing else. Use only column names from the schema.

Plan fields:
- "type": one of "aggregate", "trend", "correlation", "distribution", "anomaly", "forecast", "insight".
  Use "insight" for open-ended requests like "what stands out". Use null if the question can't be answered.
- "aggregation": one of "sum", "average", "max", "min", "count", "median", or null.
  Use "count" for "how many". Average rates, margins and scores rather than summing them.
- "metric": a numeric column, or null. "metric2": a second numeric column (correlation only).
- "groupby", "groupby2": category columns to group by, or null.
- "filters": list of {"column", "operator", "value"}; operator is one of "==", ">", "<", ">=", "<=",
  or "year" (on the date column, value is a year like 2025). Use category values exactly as listed.
- "limit": integer for top/bottom N, or null. "sort_order": "desc" (top/highest) or "asc" (bottom/lowest).
- "time_grain": for trends, one of "D", "W", "MS" (month), "QS" (quarter), "YS" (year), or null.
- "unmatched": list of words from the question that refer to columns not in the schema.

Example: "top 3 regions by average profit in 2025" ->
{"type": "aggregate", "aggregation": "average", "metric": "profit", "groupby": "region",
 "filters": [{"column": "date", "operator": "year", "value": 2025}], "limit": 3, "sort_order": "desc"}"""


def extract_json(text: str) -> Dict[str, Any]:
    """Pull the JSON object out of a reply, tolerating code fences and surrounding text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        raise LLMError("the model's reply was not valid JSON") from None
    if not isinstance(parsed, dict):
        raise LLMError("the model's reply was not a JSON object")
    return parsed


class LLMQueryPlanner:
    """Ask the model for a query plan. The caller validates and runs it."""

    def __init__(self, client: GLMClient):
        self.client = client

    def plan(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                                        f"Question: {question[:500]}"},
        ]
        return extract_json(self.client.chat(messages))
