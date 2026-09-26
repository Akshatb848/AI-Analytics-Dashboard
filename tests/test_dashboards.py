"""Saving and restoring dashboard cards as a file."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import pytest
from streamlit.testing.v1 import AppTest

from analytics.dashboards import DashboardFileError, export_dashboards, import_dashboards

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def card():
    data = pd.DataFrame({"region": ["North", "South"], "sales": [120.5, 80.0],
                         "month": pd.to_datetime(["2025-01-01", "2025-02-01"])})
    return {"title": "Sales by region", "query": "total sales by region", "summary": "**North** leads",
            "text": "## 📊 Sum of Sales by Region", "figure": px.bar(data, x="region", y="sales"),
            "data": data}


def test_round_trip_keeps_text_chart_and_table():
    restored = import_dashboards(export_dashboards([card(), {**card(), "figure": None, "data": None}]))
    assert len(restored) == 2
    first = restored[0]
    assert first["title"] == "Sales by region" and first["summary"] == "**North** leads"
    assert list(first["figure"].data[0].x) == ["North", "South"]
    assert first["data"]["sales"].tolist() == [120.5, 80.0]
    assert pd.api.types.is_datetime64_any_dtype(first["data"]["month"])
    assert restored[1]["figure"] is None and restored[1]["data"] is None


def test_export_is_versioned_json():
    payload = json.loads(export_dashboards([card()]))
    assert payload["format"] == "ai-analytics-dashboard" and payload["version"] == 1
    assert payload["exported_at"]


@pytest.mark.parametrize("content, message", [
    ("not json", "not valid JSON"),
    ("[]", "not a dashboard export"),
    ('{"format": "something-else", "version": 1, "cards": []}', "not a dashboard export"),
    ('{"format": "ai-analytics-dashboard", "version": 99, "cards": []}', "unsupported"),
    ('{"format": "ai-analytics-dashboard", "version": 1, "cards": "x"}', "no list of cards"),
    ('{"format": "ai-analytics-dashboard", "version": 1, "cards": [1]}', "card 1 is not an object"),
    ('{"format": "ai-analytics-dashboard", "version": 1, "cards": [{"data": {"columns": 5}}]}',
     "unreadable chart or table"),
])
def test_bad_files_are_rejected_with_a_reason(content, message):
    with pytest.raises(DashboardFileError, match=message):
        import_dashboards(content)


def test_too_many_cards_is_rejected():
    payload = {"format": "ai-analytics-dashboard", "version": 1, "cards": [{}] * 201}
    with pytest.raises(DashboardFileError, match="more than 200"):
        import_dashboards(json.dumps(payload))


def test_unknown_fields_are_dropped_and_values_kept_as_text():
    payload = {"format": "ai-analytics-dashboard", "version": 1, "cards": [{
        "title": ["not", "text"], "summary": "<img src=x onerror=alert(1)>", "run": "os.system('x')",
        "figure": {"data": [{"type": "bar", "x": [1], "y": [2], "onclick": "alert(1)"}],
                   "layout": {"title": {"text": "t"}, "evil": 1}},
    }]}
    restored = import_dashboards(json.dumps(payload))[0]
    assert set(restored) == {"title", "query", "summary", "text", "figure", "data"}
    assert restored["title"] == "['not', 'text']"
    assert restored["summary"] == "<img src=x onerror=alert(1)>"  # escaped when rendered
    figure_json = restored["figure"].to_json()
    assert "onclick" not in figure_json and "evil" not in figure_json


def test_untitled_cards_get_a_title():
    payload = {"format": "ai-analytics-dashboard", "version": 1, "cards": [{}]}
    assert import_dashboards(json.dumps(payload))[0]["title"] == "Imported card 1"


def test_dashboards_tab_offers_a_download_once_cards_exist():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any(b.label == "Save card" for b in at.button)
    at.text_input(key="nl_query").input("Total sales by region").run()
    next(b for b in at.button if b.label == "Save card").click().run()
    assert not at.exception
    downloads = [d for d in at.get("download_button") if "Download dashboard" in d.proto.label]
    assert len(downloads) == 1
