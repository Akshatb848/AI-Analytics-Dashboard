"""Small reusable Streamlit UI pieces."""

import pandas as pd
import streamlit as st

from analytics.data_utils import (
    get_suggested_queries,
)
from analytics.formatting import safe_html


def render_tutorial():
    """Render onboarding tutorial for first-time users."""
    st.markdown("""
    <div class="tutorial-card">
        <h4>🎓 Welcome to AI Analytics Dashboard!</h4>
        <p style="color: #94a3b8; margin-bottom: 15px;">Here's a quick guide to get you started:</p>
        
        <div class="tutorial-step">
            <div class="tutorial-step-number">1</div>
            <div class="tutorial-step-content">
                <strong>Upload Your Data</strong><br>
                Use the sidebar to upload CSV or Excel files. Multiple sheets are supported.
            </div>
        </div>
        
        <div class="tutorial-step">
            <div class="tutorial-step-number">2</div>
            <div class="tutorial-step-content">
                <strong>Explore Insights</strong><br>
                The Insights tab shows AI-generated findings with business narratives.
            </div>
        </div>
        
        <div class="tutorial-step">
            <div class="tutorial-step-number">3</div>
            <div class="tutorial-step-content">
                <strong>Ask Questions</strong><br>
                Use natural language queries like "Total sales by region" or "Show profit trend".
            </div>
        </div>
        
        <div class="tutorial-step">
            <div class="tutorial-step-number">4</div>
            <div class="tutorial-step-content">
                <strong>Generate Forecasts</strong><br>
                The Predictions tab uses Prophet for time series forecasting.
            </div>
        </div>
        
        <div class="tutorial-step">
            <div class="tutorial-step-number">5</div>
            <div class="tutorial-step-content">
                <strong>Export Reports</strong><br>
                Download insights as HTML reports, CSV, or markdown summaries.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_suggested_queries(df: pd.DataFrame):
    """Render smart query suggestions based on data."""
    suggestions = get_suggested_queries(df)
    
    st.markdown("**💡 Suggested Queries:**")
    cols = st.columns(4)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 4]:
            st.markdown(f"<span class='suggested-query'>{safe_html(suggestion)}</span>", unsafe_allow_html=True)
