"""
AI Analytics Dashboard - Alternative to Tableau AI
A comprehensive analytics platform with automated insights, predictive analytics, and NL queries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta
import json
import re
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# JSON SAFETY FIX
# =============================================================================

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    else:
        return obj


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def generate_sample_data(rows: int = 1000) -> pd.DataFrame:
    np.random.seed(42)

    date_range = pd.date_range(
        start=datetime.now() - timedelta(days=365*2),
        end=datetime.now(),
        freq='D'
    )

    base_sales = 1000
    seasonal_pattern = np.sin(np.arange(len(date_range)) * 2 * np.pi / 365) * 200
    trend = np.linspace(0, 300, len(date_range))
    noise = np.random.normal(0, 100, len(date_range))

    sales = base_sales + seasonal_pattern + trend + noise
    sales = np.maximum(sales, 100)

    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']

    data = pd.DataFrame({
        'date': np.random.choice(date_range, rows),
        'sales': np.random.choice(sales, rows) * np.random.uniform(0.5, 1.5, rows),
        'quantity': np.random.randint(1, 50, rows),
        'category': np.random.choice(categories, rows),
        'region': np.random.choice(regions, rows),
        'customer_id': np.random.randint(1000, 9999, rows),
        'profit_margin': np.random.uniform(0.1, 0.4, rows)
    })

    data['profit'] = data['sales'] * data['profit_margin']
    data['date'] = pd.to_datetime(data['date'])

    return data.sort_values('date').reset_index(drop=True)


def detect_date_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col].head(50))
            return col
        except:
            pass
    return None


def detect_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():

    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Analytics Dashboard</h1>
        <p>Automated insights, predictive analytics, and natural language queries</p>
    </div>
    """, unsafe_allow_html=True)

    # ================= SIDEBAR =================
    with st.sidebar:
        source = st.radio("Data Source", ["Sample Data", "Upload CSV"])

        if source == "Upload CSV":
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)
                for col in df.columns:
                    if 'date' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            pass
            else:
                df = generate_sample_data()
        else:
            df = generate_sample_data()

        date_col = detect_date_column(df)
        numeric_cols = detect_numeric_columns(df)
        categorical_cols = detect_categorical_columns(df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Insights",
        "📈 Predictions",
        "💬 Ask Data",
        "📋 Data Explorer"
    ])

    # ================= TAB 5 FIX =================
    with tab5:
        st.subheader("📋 Data Explorer")

        st.dataframe(df.head(100), use_container_width=True)

        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "date_column": date_col if date_col else "Not detected",
            "missing_values": df.isnull().sum().sum()
        }

        safe_summary = make_json_safe(summary)
        summary_json = json.dumps(safe_summary, indent=4)

        st.download_button(
            label="📥 Download Summary Report (JSON)",
            data=summary_json,
            file_name="summary_report.json",
            mime="application/json"
        )

    st.markdown("""
    <div class="footer">
        <p>AI Analytics Dashboard | Streamlit • Prophet • Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
