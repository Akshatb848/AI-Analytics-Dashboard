"""Save and restore dashboard cards as a JSON file, so dashboards outlive the browser session.

A card is {"title", "query", "summary", "text", "figure", "data"}: the figure is a Plotly
figure (or None) and the data a DataFrame (or None). Imported files are untrusted: only
known fields are read, text is kept as text, and figures are rebuilt through Plotly's
validators, which drop anything that isn't a valid figure property.
"""
import json
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List

import pandas as pd
import plotly.io as pio

FILE_FORMAT = "ai-analytics-dashboard"
FILE_VERSION = 1
MAX_CARDS = 200
_TEXT_FIELDS = ("title", "query", "summary", "text")
_MAX_TEXT = 10_000


class DashboardFileError(ValueError):
    """The uploaded file is not a dashboard export this app can read."""


def _export_table(data: pd.DataFrame) -> Dict[str, Any]:
    table = json.loads(data.to_json(orient="split", date_format="iso", index=False))
    # JSON has no date type, so record which columns to turn back into dates
    table["datetime_columns"] = [str(c) for c in data.columns
                                 if pd.api.types.is_datetime64_any_dtype(data[c])]
    return table


def _import_table(raw: Dict[str, Any]) -> pd.DataFrame:
    table = {k: raw[k] for k in ("columns", "data") if k in raw}
    frame = pd.read_json(StringIO(json.dumps(table)), orient="split", convert_dates=False)
    for col in raw.get("datetime_columns") or []:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def export_dashboards(cards: List[Dict[str, Any]]) -> str:
    exported = []
    for card in cards:
        figure, data = card.get("figure"), card.get("data")
        exported.append({
            **{field: str(card.get(field) or "") for field in _TEXT_FIELDS},
            "figure": json.loads(figure.to_json()) if figure is not None else None,
            "data": _export_table(data) if isinstance(data, pd.DataFrame) else None,
        })
    return json.dumps({
        "format": FILE_FORMAT,
        "version": FILE_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cards": exported,
    }, indent=2)


def import_dashboards(content: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        raise DashboardFileError("the file is not valid JSON") from None
    if not isinstance(payload, dict) or payload.get("format") != FILE_FORMAT:
        raise DashboardFileError("the file is not a dashboard export from this app")
    if payload.get("version") != FILE_VERSION:
        raise DashboardFileError(f"unsupported dashboard file version {payload.get('version')!r}")
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise DashboardFileError("the file has no list of cards")
    if len(raw_cards) > MAX_CARDS:
        raise DashboardFileError(f"the file has more than {MAX_CARDS} cards")

    cards = []
    for number, raw in enumerate(raw_cards, 1):
        if not isinstance(raw, dict):
            raise DashboardFileError(f"card {number} is not an object")
        card: Dict[str, Any] = {field: str(raw.get(field) or "")[:_MAX_TEXT] for field in _TEXT_FIELDS}
        card["title"] = card["title"] or f"Imported card {number}"
        try:
            card["figure"] = (pio.from_json(json.dumps(raw["figure"]), skip_invalid=True)
                              if isinstance(raw.get("figure"), dict) else None)
            card["data"] = _import_table(raw["data"]) if isinstance(raw.get("data"), dict) else None
        except (ValueError, TypeError, KeyError):
            raise DashboardFileError(f"card {number} has an unreadable chart or table") from None
        cards.append(card)
    return cards
