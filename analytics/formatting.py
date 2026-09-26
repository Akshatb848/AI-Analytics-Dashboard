"""HTML escaping and number formatting shared by the UI and reports."""
import html
import re
from typing import Any

import pandas as pd

_ALLOWED_NARRATIVE_TAGS = {
    html.escape("<span class='narrative-highlight'>"): "<span class='narrative-highlight'>",
    html.escape("</span>"): "</span>",
}


def safe_html(value: Any) -> str:
    """Escape text for raw-HTML blocks so column names, category values and
    queries from uploaded data can't inject markup or scripts."""
    text = html.escape(str(value))
    for escaped, raw in _ALLOWED_NARRATIVE_TAGS.items():
        text = text.replace(escaped, raw)
    return text


def _markdown_bold_to_html(text: str) -> str:
    """Render **bold** markdown in already-escaped report text."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def narrative_html(text: Any) -> str:
    """Escape a narrative for a raw-HTML card, then render its **bold** and line breaks."""
    return _markdown_bold_to_html(safe_html(text)).replace("\n", "<br>")


def format_number(num: float) -> str:
    """Format numbers for display."""
    if pd.isna(num):
        return "N/A"
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:,.2f}"
