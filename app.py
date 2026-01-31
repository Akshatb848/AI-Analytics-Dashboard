"""
AI Analytics Dashboard v2.0 - Enterprise Edition
Alternative to Tableau AI / Power BI Copilot
Features: LLM-powered queries, narrative insights, drill-down dashboards, multi-dataset support
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
import hashlib
import base64
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Analytics Dashboard v2.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None
if 'filters' not in st.session_state:
    st.session_state.filters = {}
if 'saved_dashboards' not in st.session_state:
    st.session_state.saved_dashboards = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True
if 'chart_theme' not in st.session_state:
    st.session_state.chart_theme = 'default'
if 'color_palette' not in st.session_state:
    st.session_state.color_palette = 'viridis'

# =============================================================================
# ENHANCED CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --secondary-light: #34d399;
        --accent: #f59e0b;
        --accent-light: #fbbf24;
        --danger: #ef4444;
        --warning: #f97316;
        --info: #0ea5e9;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --surface-lighter: #475569;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --text-dim: #64748b;
        --border: rgba(99, 102, 241, 0.2);
        --border-hover: rgba(99, 102, 241, 0.5);
        --glow: rgba(99, 102, 241, 0.4);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 30%, #0f172a 70%, #1e1b4b 100%);
        font-family: 'DM Sans', 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.1) 50%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #10b981 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Version badge */
    .version-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 10px;
        vertical-align: middle;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #10b981);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.25);
    }
    
    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta-positive {
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-delta-negative {
        color: #ef4444;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Narrative card */
    .narrative-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    
    .narrative-card h4 {
        color: #10b981;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .narrative-card p {
        color: #e2e8f0;
        font-size: 1rem;
    }
    
    .narrative-highlight {
        background: rgba(99, 102, 241, 0.2);
        padding: 2px 6px;
        border-radius: 4px;
        color: #818cf8;
        font-weight: 600;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.04));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .insight-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(4px);
    }
    
    .insight-card.high-priority {
        border-left: 4px solid #ef4444;
    }
    
    .insight-card.medium-priority {
        border-left: 4px solid #f59e0b;
    }
    
    .insight-card.low-priority {
        border-left: 4px solid #10b981;
    }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    
    .insight-title {
        color: #f1f5f9;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .insight-description {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    .insight-narrative {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-top: 0.75rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        line-height: 1.6;
        border-left: 3px solid #6366f1;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.1), rgba(52, 211, 153, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        border-color: rgba(16, 185, 129, 0.6);
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15);
    }
    
    .recommendation-title {
        color: #10b981;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .recommendation-description {
        color: #d1d5db;
        font-size: 0.95rem;
        margin-top: 0.75rem;
        line-height: 1.6;
    }
    
    .recommendation-action {
        color: #34d399;
        font-size: 0.85rem;
        margin-top: 0.75rem;
        font-weight: 500;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 8px;
        display: inline-block;
    }
    
    /* Query container */
    .query-container {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
    }
    
    .query-container::before {
        content: '🤖 AI Query Engine';
        position: absolute;
        top: -12px;
        left: 20px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        color: white;
        font-weight: 600;
    }
    
    /* Tutorial/Help card */
    .tutorial-card {
        background: linear-gradient(145deg, rgba(14, 165, 233, 0.1), rgba(30, 41, 59, 0.9));
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .tutorial-card h4 {
        color: #0ea5e9;
        margin-bottom: 1rem;
    }
    
    .tutorial-step {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    }
    
    .tutorial-step-number {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .tutorial-step-content {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    
    /* Data quality indicator */
    .data-quality {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(51, 65, 85, 0.7));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .quality-score {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
    }
    
    .quality-score.excellent { color: #10b981; }
    .quality-score.good { color: #f59e0b; }
    .quality-score.poor { color: #ef4444; }
    
    /* Filter tags */
    .filter-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #818cf8;
    }
    
    .filter-tag-remove {
        cursor: pointer;
        color: #ef4444;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 500;
        padding: 0.75rem 1.25rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background: rgba(99, 102, 241, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #10b981, #f59e0b) !important;
        border-radius: 10px !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .animate-slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 3rem;
        background: linear-gradient(180deg, transparent, rgba(99, 102, 241, 0.05));
    }
    
    /* Suggested queries */
    .suggested-query {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        font-size: 0.85rem;
        color: #818cf8;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .suggested-query:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }
    
    /* Dataset tabs */
    .dataset-tab {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .dataset-tab.active {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        border-color: #6366f1;
        color: white;
    }
    
    .dataset-tab:hover:not(.active) {
        border-color: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop empty rows & columns
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("—", "", regex=False)
                .str.strip()
            )

            # Try numeric conversion
            df[col] = pd.to_numeric(df[col], errors="ignore")

            # Try datetime conversion
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore")
            except:
                pass

    return df

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


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect date column in dataframe."""
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            return col
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(100))
                return col
            except:
                continue
    return None


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns suitable for analysis."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive data quality metrics."""
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = len(df) - len(df.drop_duplicates())
    
    completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
    uniqueness = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
    
    # Check for consistent data types
    consistency = 100
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col], errors='raise')
                consistency -= 5  # Penalty for numeric stored as string
            except:
                pass
    
    overall_score = (completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3)
    
    return {
        'overall': overall_score,
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency,
        'missing_cells': int(missing_cells),
        'duplicate_rows': int(duplicate_rows),
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }


def get_suggested_queries(df: pd.DataFrame) -> List[str]:
    """Generate smart query suggestions based on data schema."""
    suggestions = []
    numeric_cols = detect_numeric_columns(df)
    categorical_cols = detect_categorical_columns(df)
    date_col = detect_date_column(df)
    
    if numeric_cols and categorical_cols:
        suggestions.append(f"Total {numeric_cols[0]} by {categorical_cols[0]}")
        suggestions.append(f"Average {numeric_cols[0]} by {categorical_cols[0]}")
        if len(categorical_cols) > 1:
            suggestions.append(f"Compare {categorical_cols[0]} performance across {categorical_cols[1]}")
    
    if date_col and numeric_cols:
        suggestions.append(f"Show {numeric_cols[0]} trend over time")
        suggestions.append(f"Monthly growth rate of {numeric_cols[0]}")
    
    if len(numeric_cols) >= 2:
        suggestions.append(f"Correlation between {numeric_cols[0]} and {numeric_cols[1]}")
    
    suggestions.extend([
        "What are the key insights from this data?",
        "Identify anomalies and outliers",
        "Show top performers and bottom performers"
    ])
    
    return suggestions[:8]


@st.cache_data
def generate_sample_data(rows: int = 2000) -> pd.DataFrame:
    """Generate comprehensive sample sales data."""
    np.random.seed(42)
    
    date_range = pd.date_range(
        start=datetime.now() - timedelta(days=365*2),
        end=datetime.now(),
        freq='D'
    )
    
    # Create seasonal patterns
    base_sales = 1000
    seasonal_pattern = np.sin(np.arange(len(date_range)) * 2 * np.pi / 365) * 200
    trend = np.linspace(0, 300, len(date_range))
    noise = np.random.normal(0, 100, len(date_range))
    
    sales = base_sales + seasonal_pattern + trend + noise
    sales = np.maximum(sales, 100)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    channels = ['Online', 'Retail', 'Wholesale', 'Direct']
    customer_segments = ['Enterprise', 'SMB', 'Consumer']
    
    data = pd.DataFrame({
        'date': np.random.choice(date_range, rows),
        'sales': np.random.choice(sales, rows) * np.random.uniform(0.5, 1.5, rows),
        'quantity': np.random.randint(1, 50, rows),
        'category': np.random.choice(categories, rows),
        'region': np.random.choice(regions, rows),
        'channel': np.random.choice(channels, rows),
        'segment': np.random.choice(customer_segments, rows),
        'customer_id': np.random.randint(1000, 9999, rows),
        'profit_margin': np.random.uniform(0.1, 0.4, rows),
        'discount': np.random.uniform(0, 0.3, rows),
        'returns': np.random.choice([0, 1], rows, p=[0.95, 0.05]),
        'satisfaction_score': np.random.uniform(3.0, 5.0, rows)
    })
    
    data['profit'] = data['sales'] * data['profit_margin']
    data['revenue'] = data['sales'] * (1 - data['discount'])
    data['cost'] = data['sales'] - data['profit']
    data['date'] = pd.to_datetime(data['date'])
    
    return data.sort_values('date').reset_index(drop=True)


# =============================================================================
# DATA PREPROCESSING CLASS
# =============================================================================

class DataPreprocessor:
    """Comprehensive data preprocessing with UI feedback."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.transformations = []
        self.issues_found = []
    
    def profile_data(self) -> Dict[str, Any]:
        """Generate data profiling report."""
        profile = {
            'shape': self.df.shape,
            'columns': {},
            'issues': []
        }
        
        for col in self.df.columns:
            col_profile = {
                'dtype': str(self.df[col].dtype),
                'null_count': int(self.df[col].isnull().sum()),
                'null_pct': float(self.df[col].isnull().sum() / len(self.df) * 100),
                'unique_count': int(self.df[col].nunique()),
                'unique_pct': float(self.df[col].nunique() / len(self.df) * 100)
            }
            
            if self.df[col].dtype in ['float64', 'int64']:
                col_profile.update({
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'skewness': float(self.df[col].skew()),
                    'kurtosis': float(self.df[col].kurtosis())
                })
                
                # Detect issues
                if abs(col_profile['skewness']) > 2:
                    profile['issues'].append(f"High skewness in {col}")
                
                # Detect outliers
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = len(self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)])
                if outliers > len(self.df) * 0.05:
                    profile['issues'].append(f"{outliers} outliers in {col} ({outliers/len(self.df)*100:.1f}%)")
            
            if col_profile['null_pct'] > 5:
                profile['issues'].append(f"High missing values in {col} ({col_profile['null_pct']:.1f}%)")
            
            profile['columns'][col] = col_profile
        
        return profile
    
    def handle_missing_values(self, strategy: str = 'auto', columns: List[str] = None) -> 'DataPreprocessor':
        """Handle missing values with various strategies."""
        cols = columns or self.df.columns
        
        for col in cols:
            if self.df[col].isnull().sum() > 0:
                original_nulls = self.df[col].isnull().sum()
                
                if strategy == 'auto':
                    if self.df[col].dtype in ['float64', 'int64']:
                        if abs(self.df[col].skew()) > 1:
                            self.df[col].fillna(self.df[col].median(), inplace=True)
                            method = 'median'
                        else:
                            self.df[col].fillna(self.df[col].mean(), inplace=True)
                            method = 'mean'
                    else:
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col].fillna(mode_val[0], inplace=True)
                            method = 'mode'
                        else:
                            self.df[col].fillna('Unknown', inplace=True)
                            method = 'constant'
                elif strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                    method = 'mean'
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    method = 'median'
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    method = 'mode'
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                    method = 'drop rows'
                elif strategy == 'zero':
                    self.df[col].fillna(0, inplace=True)
                    method = 'zero'
                else:
                    method = 'none'
                
                self.transformations.append(f"Filled {original_nulls} nulls in '{col}' using {method}")
        
        return self
    
    def remove_outliers(self, columns: List[str] = None, method: str = 'iqr', threshold: float = 1.5) -> 'DataPreprocessor':
        """Remove outliers using IQR or Z-score method."""
        cols = columns or detect_numeric_columns(self.df)
        original_len = len(self.df)
        
        for col in cols:
            if col not in self.df.columns or self.df[col].dtype not in ['float64', 'int64']:
                continue
                
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - threshold * IQR) & 
                                  (self.df[col] <= Q3 + threshold * IQR)]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                self.df = self.df.iloc[z_scores < threshold]
        
        removed = original_len - len(self.df)
        if removed > 0:
            self.transformations.append(f"Removed {removed} outlier rows using {method} method")
        
        return self
    
    def create_date_features(self) -> 'DataPreprocessor':
        """Auto-generate useful features from datetime columns."""
        date_col = detect_date_column(self.df)
        
        if date_col:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df['year'] = self.df[date_col].dt.year
            self.df['month'] = self.df[date_col].dt.month
            self.df['month_name'] = self.df[date_col].dt.month_name()
            self.df['day_of_week'] = self.df[date_col].dt.dayofweek
            self.df['day_name'] = self.df[date_col].dt.day_name()
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
            self.df['quarter'] = self.df[date_col].dt.quarter
            self.df['week_of_year'] = self.df[date_col].dt.isocalendar().week
            
            self.transformations.append(f"Created 8 date features from '{date_col}'")
        
        return self
    
    def get_transformed_data(self) -> pd.DataFrame:
        return self.df
    
    def get_transformation_log(self) -> List[str]:
        return self.transformations
    
    def reset(self) -> 'DataPreprocessor':
        self.df = self.original_df.copy()
        self.transformations = []
        return self


# =============================================================================
# NARRATIVE INSIGHTS ENGINE - Generate Business Narratives
# =============================================================================

class NarrativeEngine:
    """Generate human-readable narrative explanations for insights."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.date_col = detect_date_column(df)
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
    
    def generate_trend_narrative(self, col: str, change_pct: float, period: str = "recent period") -> str:
        """Generate narrative for trend insights."""
        direction = "increased" if change_pct > 0 else "decreased"
        magnitude = "significantly" if abs(change_pct) > 25 else "moderately" if abs(change_pct) > 10 else "slightly"
        
        # Find potential causes
        causes = self._identify_potential_causes(col, change_pct)
        
        narrative = f"**{col.replace('_', ' ').title()}** has {magnitude} {direction} by "
        narrative += f"<span class='narrative-highlight'>{abs(change_pct):.1f}%</span> during the {period}. "
        
        if causes:
            narrative += f"This change appears to be driven by {causes}. "
        
        # Add recommendation
        if change_pct < -15:
            narrative += "This decline warrants immediate investigation to identify root causes and implement corrective measures."
        elif change_pct > 20:
            narrative += "This positive momentum should be analyzed to identify success factors that can be replicated across other areas."
        
        return narrative
    
    def _identify_potential_causes(self, col: str, change_pct: float) -> str:
        """Identify potential causes for changes."""
        causes = []
        
        if not self.categorical_cols:
            return ""
        
        for cat_col in self.categorical_cols[:2]:
            try:
                # Compare category performance
                df_sorted = self.df.sort_values(self.date_col) if self.date_col else self.df
                half = len(df_sorted) // 2
                
                recent = df_sorted.iloc[half:].groupby(cat_col)[col].mean()
                previous = df_sorted.iloc[:half].groupby(cat_col)[col].mean()
                
                changes = ((recent - previous) / previous * 100).dropna()
                
                if len(changes) > 0:
                    top_change = changes.idxmax() if change_pct > 0 else changes.idxmin()
                    top_change_val = changes.max() if change_pct > 0 else changes.min()
                    
                    if abs(top_change_val) > 15:
                        causes.append(f"{'strong performance' if top_change_val > 0 else 'underperformance'} in {cat_col} '{top_change}'")
            except:
                continue
        
        return ", ".join(causes[:2]) if causes else ""
    
    def generate_correlation_narrative(self, col1: str, col2: str, corr_value: float) -> str:
        """Generate narrative for correlation insights."""
        strength = "very strong" if abs(corr_value) > 0.8 else "strong" if abs(corr_value) > 0.6 else "moderate"
        direction = "positive" if corr_value > 0 else "negative"
        
        narrative = f"There is a {strength} {direction} correlation "
        narrative += f"(<span class='narrative-highlight'>r = {corr_value:.3f}</span>) between "
        narrative += f"**{col1.replace('_', ' ').title()}** and **{col2.replace('_', ' ').title()}**. "
        
        if corr_value > 0:
            narrative += f"This means that as {col1.replace('_', ' ')} increases, {col2.replace('_', ' ')} tends to increase as well. "
        else:
            narrative += f"This means that as {col1.replace('_', ' ')} increases, {col2.replace('_', ' ')} tends to decrease. "
        
        # Business implication
        narrative += "This relationship can be leveraged for predictive modeling and strategic planning."
        
        return narrative
    
    def generate_anomaly_narrative(self, col: str, anomaly_count: int, anomaly_pct: float) -> str:
        """Generate narrative for anomaly insights."""
        severity = "concerning" if anomaly_pct > 10 else "notable" if anomaly_pct > 5 else "minor"
        
        narrative = f"A {severity} number of anomalies have been detected in **{col.replace('_', ' ').title()}**: "
        narrative += f"<span class='narrative-highlight'>{anomaly_count:,} records ({anomaly_pct:.1f}%)</span> "
        narrative += "fall outside the expected range. "
        
        narrative += "These outliers may represent data quality issues, exceptional business events, "
        narrative += "or fraudulent activity. A detailed review of these records is recommended "
        narrative += "to determine appropriate action—whether correction, exclusion, or further investigation."
        
        return narrative
    
    def generate_performance_narrative(self, cat_col: str, metric_col: str, 
                                       top_performer: str, bottom_performer: str,
                                       gap_pct: float) -> str:
        """Generate narrative for performance gap insights."""
        narrative = f"Significant performance variation exists across **{cat_col.replace('_', ' ').title()}**. "
        narrative += f"<span class='narrative-highlight'>{top_performer}</span> leads with the highest {metric_col.replace('_', ' ')}, "
        narrative += f"while <span class='narrative-highlight'>{bottom_performer}</span> shows the lowest performance—"
        narrative += f"a gap of <span class='narrative-highlight'>{gap_pct:.1f}%</span>. "
        
        narrative += "\n\nThis disparity suggests opportunities for: (1) analyzing success factors from top performers, "
        narrative += "(2) implementing targeted improvement initiatives for underperformers, and "
        narrative += "(3) reallocating resources to maximize overall returns."
        
        return narrative
    
    def generate_executive_summary(self, insights: List[Dict]) -> str:
        """Generate executive summary narrative from all insights."""
        high_priority = [i for i in insights if i.get('priority') == 'high']
        
        summary = "## 📊 Executive Summary\n\n"
        summary += f"Analysis of **{len(self.df):,}** records reveals "
        summary += f"**{len(high_priority)}** critical findings requiring attention.\n\n"
        
        if high_priority:
            summary += "### Key Findings:\n\n"
            for i, insight in enumerate(high_priority[:3], 1):
                summary += f"{i}. **{insight['title']}**: {insight['description']}\n\n"
        
        # Add time context
        if self.date_col:
            date_range = f"{self.df[self.date_col].min().strftime('%B %d, %Y')} to {self.df[self.date_col].max().strftime('%B %d, %Y')}"
            summary += f"\n*Analysis period: {date_range}*"
        
        return summary


# =============================================================================
# ENHANCED INSIGHTS ENGINE with Narratives
# =============================================================================

class EnhancedInsightsEngine:
    """Generate automated insights with narrative explanations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
        self.recommendations = []
        self.date_col = detect_date_column(df)
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
        self.narrative_engine = NarrativeEngine(df)
    
    def generate_all_insights(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate comprehensive insights with narratives."""
        self.insights = []
        self.recommendations = []
        
        # Run all analysis methods
        self._analyze_trends()
        self._analyze_correlations()
        self._detect_anomalies()
        self._analyze_categorical_performance()
        self._analyze_distributions()
        self._perform_statistical_tests()
        self._detect_seasonality()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.insights, self.recommendations
    
    def _analyze_trends(self):
        """Analyze time-based trends with narratives."""
        if not self.date_col or not self.numeric_cols:
            return
        
        for col in self.numeric_cols[:4]:
            try:
                df_sorted = self.df.sort_values(self.date_col)
                
                # Overall trend
                recent = df_sorted[col].tail(int(len(df_sorted) * 0.2)).mean()
                previous = df_sorted[col].head(int(len(df_sorted) * 0.2)).mean()
                
                if previous > 0:
                    change_pct = ((recent - previous) / previous) * 100
                    
                    if abs(change_pct) > 8:
                        narrative = self.narrative_engine.generate_trend_narrative(col, change_pct)
                        
                        self.insights.append({
                            'type': 'trend',
                            'icon': '📈' if change_pct > 0 else '📉',
                            'title': f'{col.replace("_", " ").title()} {"Increased" if change_pct > 0 else "Decreased"} by {abs(change_pct):.1f}%',
                            'description': f'Comparing recent 20% of data vs earliest 20%',
                            'narrative': narrative,
                            'priority': 'high' if abs(change_pct) > 20 else 'medium',
                            'metric': col,
                            'value': change_pct
                        })
                
                # Week-over-week for recent data
                if len(df_sorted) >= 14:
                    last_week = df_sorted[col].tail(7).mean()
                    prev_week = df_sorted[col].iloc[-14:-7].mean()
                    
                    if prev_week > 0:
                        wow_change = ((last_week - prev_week) / prev_week) * 100
                        
                        if abs(wow_change) > 12:
                            self.insights.append({
                                'type': 'trend_weekly',
                                'icon': '📅',
                                'title': f'Week-over-Week: {col.replace("_", " ").title()} {"Up" if wow_change > 0 else "Down"} {abs(wow_change):.1f}%',
                                'description': f'Comparing last 7 days vs previous 7 days',
                                'narrative': f'Short-term momentum shows {col.replace("_", " ")} {"accelerating" if wow_change > 0 else "decelerating"} with a {abs(wow_change):.1f}% {"gain" if wow_change > 0 else "decline"} week-over-week.',
                                'priority': 'medium',
                                'metric': col,
                                'value': wow_change
                            })
            except Exception as e:
                continue
    
    def _analyze_correlations(self):
        """Find and explain significant correlations."""
        if len(self.numeric_cols) < 2:
            return
        
        try:
            corr_matrix = self.df[self.numeric_cols].corr()
            
            for i, col1 in enumerate(self.numeric_cols):
                for col2 in self.numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_val) > 0.6:
                        narrative = self.narrative_engine.generate_correlation_narrative(col1, col2, corr_val)
                        
                        self.insights.append({
                            'type': 'correlation',
                            'icon': '🔗',
                            'title': f'{"Strong" if abs(corr_val) > 0.8 else "Moderate"} {"Positive" if corr_val > 0 else "Negative"} Correlation',
                            'description': f'{col1.replace("_", " ").title()} ↔ {col2.replace("_", " ").title()} (r={corr_val:.3f})',
                            'narrative': narrative,
                            'priority': 'high' if abs(corr_val) > 0.8 else 'medium',
                            'metric': f'{col1}_vs_{col2}',
                            'value': corr_val
                        })
        except:
            pass
    
    def _detect_anomalies(self):
        """Detect and explain anomalies."""
        for col in self.numeric_cols[:4]:
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                anomalies = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
                anomaly_pct = (len(anomalies) / len(self.df)) * 100
                
                if anomaly_pct > 3:
                    narrative = self.narrative_engine.generate_anomaly_narrative(col, len(anomalies), anomaly_pct)
                    
                    self.insights.append({
                        'type': 'anomaly',
                        'icon': '⚠️',
                        'title': f'Anomalies Detected in {col.replace("_", " ").title()}',
                        'description': f'{len(anomalies):,} outliers ({anomaly_pct:.1f}% of data)',
                        'narrative': narrative,
                        'priority': 'high' if anomaly_pct > 8 else 'medium',
                        'metric': col,
                        'value': anomaly_pct
                    })
            except:
                continue
    
    def _analyze_categorical_performance(self):
        """Analyze performance across categories with narratives."""
        if not self.categorical_cols or not self.numeric_cols:
            return
        
        for cat_col in self.categorical_cols[:2]:
            for num_col in self.numeric_cols[:2]:
                try:
                    perf = self.df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count'])
                    
                    if len(perf) >= 2:
                        top_performer = perf['mean'].idxmax()
                        bottom_performer = perf['mean'].idxmin()
                        
                        top_val = perf.loc[top_performer, 'mean']
                        bottom_val = perf.loc[bottom_performer, 'mean']
                        
                        if bottom_val > 0:
                            gap_pct = ((top_val - bottom_val) / bottom_val) * 100
                            
                            if gap_pct > 25:
                                narrative = self.narrative_engine.generate_performance_narrative(
                                    cat_col, num_col, top_performer, bottom_performer, gap_pct
                                )
                                
                                self.insights.append({
                                    'type': 'performance_gap',
                                    'icon': '⚡',
                                    'title': f'Performance Gap: {cat_col.replace("_", " ").title()}',
                                    'description': f'{top_performer} outperforms {bottom_performer} by {gap_pct:.1f}% in {num_col.replace("_", " ")}',
                                    'narrative': narrative,
                                    'priority': 'high' if gap_pct > 50 else 'medium',
                                    'metric': f'{cat_col}_{num_col}',
                                    'value': gap_pct
                                })
                except:
                    continue
    
    def _analyze_distributions(self):
        """Analyze data distributions."""
        for col in self.numeric_cols[:3]:
            try:
                skewness = self.df[col].skew()
                kurtosis = self.df[col].kurtosis()
                
                if abs(skewness) > 1.5:
                    direction = "right (positive)" if skewness > 0 else "left (negative)"
                    
                    self.insights.append({
                        'type': 'distribution',
                        'icon': '📊',
                        'title': f'{col.replace("_", " ").title()} is Heavily Skewed',
                        'description': f'Skewness: {skewness:.2f} ({direction})',
                        'narrative': f'The distribution of **{col.replace("_", " ").title()}** is significantly skewed {direction}. This indicates {"a concentration of lower values with some high outliers" if skewness > 0 else "a concentration of higher values with some low outliers"}. Consider using median instead of mean for central tendency, or apply log transformation for analysis.',
                        'priority': 'low',
                        'metric': col,
                        'value': skewness
                    })
            except:
                continue
    
    def _perform_statistical_tests(self):
        """Run statistical significance tests."""
        if not self.categorical_cols or not self.numeric_cols:
            return
        
        for cat_col in self.categorical_cols[:1]:
            for num_col in self.numeric_cols[:2]:
                try:
                    groups = self.df.groupby(cat_col)[num_col].apply(list).to_dict()
                    group_names = list(groups.keys())
                    
                    if len(group_names) >= 2:
                        # Compare top 2 groups
                        group1 = groups[group_names[0]]
                        group2 = groups[group_names[1]]
                        
                        if len(group1) >= 30 and len(group2) >= 30:
                            t_stat, p_value = stats.ttest_ind(group1, group2)
                            
                            if p_value < 0.05:
                                mean1, mean2 = np.mean(group1), np.mean(group2)
                                diff_pct = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
                                
                                self.insights.append({
                                    'type': 'statistical_test',
                                    'icon': '🔬',
                                    'title': f'Statistically Significant Difference in {num_col.replace("_", " ").title()}',
                                    'description': f'{group_names[1]} vs {group_names[0]}: {abs(diff_pct):.1f}% difference (p={p_value:.4f})',
                                    'narrative': f'Statistical analysis confirms that the difference in **{num_col.replace("_", " ")}** between **{group_names[1]}** and **{group_names[0]}** is statistically significant (p-value: {p_value:.4f}). This means the observed {abs(diff_pct):.1f}% difference is unlikely to be due to random chance.',
                                    'priority': 'high' if p_value < 0.01 else 'medium',
                                    'metric': f'{cat_col}_{num_col}_ttest',
                                    'value': p_value
                                })
                except:
                    continue
    
    def _detect_seasonality(self):
        """Detect seasonal patterns."""
        if not self.date_col:
            return
        
        for col in self.numeric_cols[:2]:
            try:
                df_ts = self.df.groupby(self.date_col)[col].sum().reset_index()
                df_ts = df_ts.sort_values(self.date_col)
                
                if len(df_ts) >= 30:
                    values = df_ts[col].values
                    
                    # Weekly pattern check
                    if len(values) >= 14:
                        autocorr_7 = np.corrcoef(values[:-7], values[7:])[0, 1]
                        
                        if abs(autocorr_7) > 0.4:
                            self.insights.append({
                                'type': 'seasonality',
                                'icon': '🔄',
                                'title': f'Weekly Seasonality in {col.replace("_", " ").title()}',
                                'description': f'Autocorrelation at lag 7: {autocorr_7:.3f}',
                                'narrative': f'A **weekly seasonal pattern** has been detected in {col.replace("_", " ")}. The autocorrelation coefficient of {autocorr_7:.3f} suggests that values tend to repeat on a 7-day cycle. This pattern should be accounted for in forecasting models and can inform staffing/inventory decisions.',
                                'priority': 'medium',
                                'metric': col,
                                'value': autocorr_7
                            })
            except:
                continue
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        high_priority = [i for i in self.insights if i['priority'] == 'high']
        
        # Recommendations based on insight types
        for insight in high_priority[:5]:
            if insight['type'] == 'trend' and insight['value'] < -15:
                self.recommendations.append({
                    'icon': '🎯',
                    'title': 'Investigate Declining Trend',
                    'description': f"The {abs(insight['value']):.1f}% decline in {insight['metric'].replace('_', ' ')} requires immediate attention. Conduct root cause analysis focusing on recent operational changes, market conditions, and competitive factors.",
                    'action': 'Schedule stakeholder meeting to review decline drivers',
                    'priority': 'high'
                })
            
            elif insight['type'] == 'correlation' and abs(insight['value']) > 0.7:
                self.recommendations.append({
                    'icon': '📊',
                    'title': 'Leverage Correlation for Prediction',
                    'description': f"The strong correlation identified can be used to build predictive models. When one variable changes, you can anticipate changes in the correlated variable.",
                    'action': 'Develop regression model for forecasting',
                    'priority': 'medium'
                })
            
            elif insight['type'] == 'anomaly':
                self.recommendations.append({
                    'icon': '🔍',
                    'title': 'Audit Anomalous Data',
                    'description': f"Review the {insight['value']:.1f}% of outlier records to distinguish between data errors and genuine exceptional cases.",
                    'action': 'Export anomalies for manual review',
                    'priority': 'high'
                })
            
            elif insight['type'] == 'performance_gap':
                self.recommendations.append({
                    'icon': '⚡',
                    'title': 'Close Performance Gap',
                    'description': 'Analyze what differentiates top performers and create an improvement playbook for underperformers.',
                    'action': 'Conduct best practices analysis',
                    'priority': 'high'
                })
        
        # General recommendations
        if len(self.df) > 1000 and self.date_col:
            self.recommendations.append({
                'icon': '📈',
                'title': 'Enable Time Series Forecasting',
                'description': 'Your dataset has sufficient history for accurate forecasting. Use the Predictions tab to generate forecasts with confidence intervals.',
                'action': 'Navigate to Predictions tab',
                'priority': 'medium'
            })
        
        if len(self.categorical_cols) >= 2:
            self.recommendations.append({
                'icon': '🔀',
                'title': 'Perform Cross-Segment Analysis',
                'description': 'Multiple categorical dimensions allow for drill-down analysis. Examine how metrics vary across different segment combinations.',
                'action': 'Use filters to compare segments',
                'priority': 'low'
            })


# =============================================================================
# ENHANCED NL QUERY ENGINE - LLM-Ready Architecture
# =============================================================================

class SmartQueryEngine:
    """
    Intelligent query processing with semantic understanding.
    Architecture ready for LLM integration (OpenAI/Anthropic API).
    Currently uses advanced pattern matching with fallback to smart defaults.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
        self.date_col = detect_date_column(df)
        self.schema = self._build_schema()
        
        # Enhanced patterns for better query understanding
        self.intent_patterns = {
            'aggregate': {
                'sum': r'(?:total|sum|aggregate|combined|overall)\s+(?:of\s+)?(\w+)',
                'average': r'(?:average|mean|avg|typical)\s+(?:of\s+)?(\w+)',
                'max': r'(?:max(?:imum)?|highest|largest|best|top|peak)\s+(\w+)',
                'min': r'(?:min(?:imum)?|lowest|smallest|worst|bottom)\s+(\w+)',
                'count': r'(?:count|number|how many|quantity)\s+(?:of\s+)?(\w+)?',
                'median': r'(?:median|middle)\s+(?:of\s+)?(\w+)',
            },
            'grouping': {
                'by': r'(?:by|per|for each|grouped by|across|segmented by)\s+(\w+)',
                'compare': r'compare\s+(\w+)\s+(?:and|vs|versus|with|to)\s+(\w+)',
            },
            'filtering': {
                'where': r'(?:where|when|if|for|in|with)\s+(\w+)\s*(?:is|=|equals?|==)\s*["\']?([^"\']+)["\']?',
                'top_n': r'(?:top|first|best)\s+(\d+)',
                'bottom_n': r'(?:bottom|last|worst)\s+(\d+)',
                'greater': r'(\w+)\s*(?:>|greater than|more than|above|over)\s*(\d+(?:\.\d+)?)',
                'less': r'(\w+)\s*(?:<|less than|under|below)\s*(\d+(?:\.\d+)?)',
            },
            'analysis': {
                'trend': r'(?:trend|over time|time series|growth|change|evolution)\s+(?:of\s+)?(\w+)?',
                'correlation': r'(?:correlation|relationship|connection|link)\s+(?:between\s+)?(\w+)\s+(?:and|with)\s+(\w+)',
                'distribution': r'(?:distribution|spread|histogram|breakdown)\s+(?:of\s+)?(\w+)',
                'forecast': r'(?:forecast|predict|projection|future)\s+(?:of\s+)?(\w+)',
                'anomaly': r'(?:anomal(?:y|ies)|outlier|unusual|abnormal)\s+(?:in\s+)?(\w+)?',
            },
            'insights': {
                'insight': r'(?:insight|finding|discover|analyze|tell me about|explain|why)',
                'summary': r'(?:summary|overview|describe|summarize|recap)',
                'key': r'(?:key|important|significant|main|critical)',
            }
        }
    
    def _build_schema(self) -> Dict:
        """Build data schema for context."""
        schema = {
            'columns': {},
            'row_count': len(self.df),
            'date_range': None
        }
        
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'null_pct': self.df[col].isnull().sum() / len(self.df) * 100
            }
            
            if col in self.numeric_cols:
                col_info.update({
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'mean': float(self.df[col].mean())
                })
            elif col in self.categorical_cols:
                col_info['categories'] = self.df[col].unique()[:10].tolist()
            
            schema['columns'][col] = col_info
        
        if self.date_col:
            schema['date_range'] = {
                'start': self.df[self.date_col].min().isoformat(),
                'end': self.df[self.date_col].max().isoformat()
            }
        
        return schema
    
    def _find_column(self, term: str) -> Optional[str]:
        """Smart column matching with fuzzy logic."""
        if not term:
            return None
        
        term_lower = term.lower().strip()
        
        # Exact match
        for col in self.df.columns:
            if col.lower() == term_lower:
                return col
        
        # Partial match
        for col in self.df.columns:
            if term_lower in col.lower() or col.lower() in term_lower:
                return col
        
        # Word-level match
        for col in self.df.columns:
            col_words = set(col.lower().replace('_', ' ').split())
            if term_lower in col_words:
                return col
        
        # Synonym matching
        synonyms = {
            'revenue': ['sales', 'income', 'amount'],
            'profit': ['margin', 'earnings', 'gain'],
            'cost': ['expense', 'spending', 'price'],
            'quantity': ['count', 'number', 'volume', 'qty'],
            'date': ['time', 'day', 'period', 'when'],
            'category': ['type', 'group', 'segment', 'class'],
            'region': ['area', 'location', 'territory', 'zone']
        }
        
        for col, syns in synonyms.items():
            if term_lower in syns:
                for df_col in self.df.columns:
                    if col in df_col.lower():
                        return df_col
        
        return None
    
    def _detect_intent(self, query: str) -> Dict:
        """Detect query intent and extract parameters."""
        query_lower = query.lower()
        intent = {
            'type': None,
            'aggregation': None,
            'metric': None,
            'groupby': None,
            'filters': [],
            'limit': None,
            'sort_order': 'desc'
        }
        
        # Check for insight/summary requests
        for pattern in self.intent_patterns['insights'].values():
            if re.search(pattern, query_lower):
                intent['type'] = 'insight'
                break
        
        # Check for analysis types
        for analysis_type, pattern in self.intent_patterns['analysis'].items():
            match = re.search(pattern, query_lower)
            if match:
                intent['type'] = analysis_type
                if match.groups():
                    intent['metric'] = self._find_column(match.group(1))
                break
        
        # Check for aggregations
        for agg_type, pattern in self.intent_patterns['aggregate'].items():
            match = re.search(pattern, query_lower)
            if match:
                intent['aggregation'] = agg_type
                if match.group(1):
                    intent['metric'] = self._find_column(match.group(1))
                if not intent['type']:
                    intent['type'] = 'aggregate'
                break
        
        # Check for grouping
        for group_type, pattern in self.intent_patterns['grouping'].items():
            match = re.search(pattern, query_lower)
            if match:
                intent['groupby'] = self._find_column(match.group(1))
                break
        
        # Check for filters
        where_match = re.search(self.intent_patterns['filtering']['where'], query_lower)
        if where_match:
            filter_col = self._find_column(where_match.group(1))
            if filter_col:
                intent['filters'].append({
                    'column': filter_col,
                    'operator': '==',
                    'value': where_match.group(2).strip()
                })
        
        # Check for top/bottom N
        top_match = re.search(self.intent_patterns['filtering']['top_n'], query_lower)
        if top_match:
            intent['limit'] = int(top_match.group(1))
            intent['sort_order'] = 'desc'
        
        bottom_match = re.search(self.intent_patterns['filtering']['bottom_n'], query_lower)
        if bottom_match:
            intent['limit'] = int(bottom_match.group(1))
            intent['sort_order'] = 'asc'
        
        return intent
    
    def process_query(self, query: str) -> Dict:
        """Process query and return comprehensive results."""
        intent = self._detect_intent(query)
        
        result = {
            'success': True,
            'query': query,
            'intent': intent,
            'text': '',
            'data': None,
            'figure': None,
            'narrative': '',
            'follow_up_suggestions': []
        }
        
        try:
            # Route to appropriate handler
            if intent['type'] == 'insight':
                return self._handle_insight_query(query, result)
            elif intent['type'] == 'trend':
                return self._handle_trend_query(intent, result)
            elif intent['type'] == 'correlation':
                return self._handle_correlation_query(intent, result)
            elif intent['type'] == 'distribution':
                return self._handle_distribution_query(intent, result)
            elif intent['type'] == 'anomaly':
                return self._handle_anomaly_query(intent, result)
            elif intent['type'] == 'forecast':
                return self._handle_forecast_query(intent, result)
            elif intent['aggregation']:
                return self._handle_aggregation_query(intent, result)
            else:
                return self._handle_default_query(query, result)
                
        except Exception as e:
            result['success'] = False
            result['text'] = f"Error processing query: {str(e)}"
            result['follow_up_suggestions'] = get_suggested_queries(self.df)
            return result
    
    def _handle_insight_query(self, query: str, result: Dict) -> Dict:
        """Handle requests for insights."""
        engine = EnhancedInsightsEngine(self.df)
        insights, recommendations = engine.generate_all_insights()
        
        high_priority = [i for i in insights if i['priority'] == 'high'][:3]
        
        result['text'] = f"## 🔍 Key Insights from Your Data\n\nAnalyzed **{len(self.df):,}** records and found **{len(insights)}** insights."
        
        narrative_parts = []
        for insight in high_priority:
            narrative_parts.append(f"**{insight['title']}**: {insight.get('narrative', insight['description'])}")
        
        result['narrative'] = "\n\n".join(narrative_parts)
        result['data'] = pd.DataFrame([{
            'Insight': i['title'],
            'Priority': i['priority'].title(),
            'Details': i['description']
        } for i in insights[:10]])
        
        result['follow_up_suggestions'] = [
            f"Show trend of {self.numeric_cols[0]}" if self.numeric_cols else "Show data summary",
            f"Compare {self.categorical_cols[0]} performance" if self.categorical_cols else "Show distribution",
            "What anomalies exist in the data?"
        ]
        
        return result
    
    def _handle_trend_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle trend analysis queries."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
        
        if not metric or not self.date_col:
            result['text'] = "Cannot perform trend analysis. Need a date column and numeric metric."
            return result
        
        # Calculate trend data
        trend_data = self.df.groupby(self.date_col)[metric].sum().reset_index()
        trend_data = trend_data.sort_values(self.date_col)
        trend_data['ma_7'] = trend_data[metric].rolling(window=7, min_periods=1).mean()
        
        # Calculate growth
        first_val = trend_data[metric].iloc[:7].mean()
        last_val = trend_data[metric].iloc[-7:].mean()
        growth = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data[self.date_col], y=trend_data[metric],
            mode='lines', name='Daily', line=dict(color='#6366f1', width=1), opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=trend_data[self.date_col], y=trend_data['ma_7'],
            mode='lines', name='7-Day MA', line=dict(color='#10b981', width=2)
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{metric.replace("_", " ").title()} Trend Over Time',
            hovermode='x unified'
        )
        
        result['figure'] = fig
        result['data'] = trend_data
        result['text'] = f"## 📈 Trend Analysis: {metric.replace('_', ' ').title()}"
        result['narrative'] = f"**{metric.replace('_', ' ').title()}** shows an overall {'increase' if growth > 0 else 'decrease'} of **{abs(growth):.1f}%** over the analysis period. "
        
        if abs(growth) > 20:
            result['narrative'] += f"This is a significant {'upward' if growth > 0 else 'downward'} trend that warrants attention."
        
        result['follow_up_suggestions'] = [
            f"Forecast {metric} for next 30 days",
            f"Show {metric} by {self.categorical_cols[0]}" if self.categorical_cols else f"Distribution of {metric}",
            "What's driving this trend?"
        ]
        
        return result
    
    def _handle_correlation_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle correlation analysis queries."""
        if len(self.numeric_cols) < 2:
            result['text'] = "Need at least 2 numeric columns for correlation analysis."
            return result
        
        col1 = intent.get('metric') or self.numeric_cols[0]
        col2 = self.numeric_cols[1] if len(self.numeric_cols) > 1 else col1
        
        correlation = self.df[col1].corr(self.df[col2])
        
        fig = px.scatter(
            self.df, x=col1, y=col2,
            trendline='ols',
            title=f'Correlation: {col1} vs {col2} (r={correlation:.3f})',
            color_discrete_sequence=['#6366f1']
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        result['figure'] = fig
        result['text'] = f"## 🔗 Correlation Analysis"
        
        strength = "very strong" if abs(correlation) > 0.8 else "strong" if abs(correlation) > 0.6 else "moderate" if abs(correlation) > 0.4 else "weak"
        direction = "positive" if correlation > 0 else "negative"
        
        result['narrative'] = f"The correlation between **{col1.replace('_', ' ').title()}** and **{col2.replace('_', ' ').title()}** is **{strength} {direction}** (r = {correlation:.3f}). "
        result['narrative'] += f"This means {'when one increases, the other tends to increase' if correlation > 0 else 'when one increases, the other tends to decrease'}."
        
        return result
    
    def _handle_distribution_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle distribution analysis queries."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
        
        if not metric:
            result['text'] = "No numeric column found for distribution analysis."
            return result
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.df[metric], nbinsx=50,
            marker=dict(color='#6366f1', line=dict(color='#818cf8', width=1))
        ))
        
        mean_val = self.df[metric].mean()
        median_val = self.df[metric].median()
        
        fig.add_vline(x=mean_val, line_dash="solid", line_color="#10b981",
                     annotation_text=f"Mean: {mean_val:,.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#f59e0b",
                     annotation_text=f"Median: {median_val:,.2f}")
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'Distribution of {metric.replace("_", " ").title()}'
        )
        
        result['figure'] = fig
        result['text'] = f"## 📊 Distribution Analysis: {metric.replace('_', ' ').title()}"
        
        skewness = self.df[metric].skew()
        result['narrative'] = f"The distribution of **{metric.replace('_', ' ')}** ranges from {self.df[metric].min():,.2f} to {self.df[metric].max():,.2f}. "
        result['narrative'] += f"Mean: {mean_val:,.2f}, Median: {median_val:,.2f}. "
        result['narrative'] += f"The distribution is {'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'approximately normal'} (skewness: {skewness:.2f})."
        
        return result
    
    def _handle_anomaly_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle anomaly detection queries."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
        
        if not metric:
            result['text'] = "No numeric column found for anomaly detection."
            return result
        
        Q1 = self.df[metric].quantile(0.25)
        Q3 = self.df[metric].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        anomalies = self.df[(self.df[metric] < lower) | (self.df[metric] > upper)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(self.df))), y=self.df[metric],
            mode='markers', name='Normal',
            marker=dict(color='#6366f1', size=4)
        ))
        
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies.index.tolist(), y=anomalies[metric],
                mode='markers', name='Anomalies',
                marker=dict(color='#ef4444', size=8, symbol='x')
            ))
        
        fig.add_hline(y=upper, line_dash="dash", line_color="#f59e0b", annotation_text="Upper bound")
        fig.add_hline(y=lower, line_dash="dash", line_color="#f59e0b", annotation_text="Lower bound")
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'Anomaly Detection: {metric.replace("_", " ").title()}'
        )
        
        result['figure'] = fig
        result['data'] = anomalies.head(20)
        result['text'] = f"## ⚠️ Anomaly Detection: {metric.replace('_', ' ').title()}"
        result['narrative'] = f"Found **{len(anomalies):,}** anomalies ({len(anomalies)/len(self.df)*100:.1f}% of data) in **{metric.replace('_', ' ')}**. "
        result['narrative'] += f"Values outside the range [{lower:,.2f}, {upper:,.2f}] are flagged as outliers."
        
        return result
    
    def _handle_forecast_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle forecast queries."""
        result['text'] = "## 📈 Forecasting"
        result['narrative'] = "For detailed forecasting with Prophet, please use the **Predictions** tab. It provides interactive controls for forecast horizon and seasonality settings."
        result['follow_up_suggestions'] = ["Navigate to Predictions tab for forecasting"]
        return result
    
    def _handle_aggregation_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle aggregation queries (sum, avg, etc.)."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
        agg_type = intent['aggregation'] or 'sum'
        groupby = intent['groupby']
        
        if not metric:
            # Count query
            if agg_type == 'count':
                if groupby:
                    data = self.df.groupby(groupby).size().reset_index(name='count')
                    data = data.sort_values('count', ascending=False)
                    
                    fig = px.bar(data, x=groupby, y='count', color='count',
                               color_continuous_scale='Viridis',
                               title=f'Count by {groupby.replace("_", " ").title()}')
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    
                    result['figure'] = fig
                    result['data'] = data
                    result['text'] = f"## 📊 Count by {groupby.replace('_', ' ').title()}"
                else:
                    result['text'] = f"Total count: **{len(self.df):,}** records"
                return result
            else:
                result['text'] = "Please specify a numeric column for aggregation."
                return result
        
        # Apply filters
        df_filtered = self.df.copy()
        for f in intent['filters']:
            if f['column'] in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[f['column']].astype(str).str.lower() == f['value'].lower()]
        
        if groupby:
            # Grouped aggregation
            agg_func = {'sum': 'sum', 'average': 'mean', 'mean': 'mean', 'max': 'max', 'min': 'min', 'count': 'count', 'median': 'median'}
            
            data = df_filtered.groupby(groupby)[metric].agg(agg_func.get(agg_type, 'sum')).reset_index()
            data.columns = [groupby, metric]
            data = data.sort_values(metric, ascending=(intent['sort_order'] == 'asc'))
            
            if intent['limit']:
                data = data.head(intent['limit'])
            
            fig = px.bar(data, x=groupby, y=metric, color=metric,
                        color_continuous_scale='Viridis',
                        title=f'{agg_type.title()} of {metric.replace("_", " ").title()} by {groupby.replace("_", " ").title()}')
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            
            result['figure'] = fig
            result['data'] = data
            result['text'] = f"## 📊 {agg_type.title()} of {metric.replace('_', ' ').title()} by {groupby.replace('_', ' ').title()}"
            
            # Narrative
            top_row = data.iloc[0] if intent['sort_order'] == 'desc' else data.iloc[-1]
            bottom_row = data.iloc[-1] if intent['sort_order'] == 'desc' else data.iloc[0]
            
            result['narrative'] = f"**{top_row[groupby]}** leads with {format_number(top_row[metric])} in {metric.replace('_', ' ')}, "
            result['narrative'] += f"while **{bottom_row[groupby]}** has the lowest at {format_number(bottom_row[metric])}."
            
        else:
            # Simple aggregation
            agg_funcs = {
                'sum': df_filtered[metric].sum(),
                'average': df_filtered[metric].mean(),
                'mean': df_filtered[metric].mean(),
                'max': df_filtered[metric].max(),
                'min': df_filtered[metric].min(),
                'count': df_filtered[metric].count(),
                'median': df_filtered[metric].median()
            }
            
            value = agg_funcs.get(agg_type, df_filtered[metric].sum())
            result['text'] = f"## 📊 {agg_type.title()} of {metric.replace('_', ' ').title()}: **{format_number(value)}**"
            result['data'] = value
        
        return result
    
    def _handle_default_query(self, query: str, result: Dict) -> Dict:
        """Handle queries that don't match specific patterns."""
        result['text'] = "## 🤔 Query Not Recognized"
        result['narrative'] = f"I couldn't understand the specific request: '{query}'. Here are some things I can help with:"
        
        suggestions = get_suggested_queries(self.df)
        result['follow_up_suggestions'] = suggestions
        
        # Provide schema context
        result['narrative'] += f"\n\n**Available columns:**\n"
        result['narrative'] += f"- Numeric: {', '.join(self.numeric_cols[:5])}\n"
        result['narrative'] += f"- Categorical: {', '.join(self.categorical_cols[:5])}\n"
        
        if self.date_col:
            result['narrative'] += f"- Date: {self.date_col}"
        
        return result


# =============================================================================
# PREDICTIVE ANALYTICS ENGINE
# =============================================================================

class PredictiveEngine:
    """Time series forecasting using Prophet with enhanced visualization."""
    
    def __init__(self, df: pd.DataFrame, date_col: str, target_col: str):
        self.df = df
        self.date_col = date_col
        self.target_col = target_col
        self.model = None
        self.forecast = None
    
    def prepare_data(self) -> pd.DataFrame:
        prophet_df = self.df[[self.date_col, self.target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        return prophet_df.sort_values('ds')
    
    def train_model(self, yearly_seasonality: bool = True, 
                   weekly_seasonality: bool = True,
                   daily_seasonality: bool = False) -> None:
        prophet_df = self.prepare_data()
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )
        self.model.fit(prophet_df)
    
    def make_forecast(self, periods: int = 30, freq: str = 'D') -> pd.DataFrame:
        if self.model is None:
            self.train_model()
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)
        return self.forecast
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        if self.forecast is None:
            return {}
        future_forecast = self.forecast[self.forecast['ds'] > self.df[self.date_col].max()]
        return {
            'predicted_mean': float(future_forecast['yhat'].mean()),
            'predicted_min': float(future_forecast['yhat_lower'].min()),
            'predicted_max': float(future_forecast['yhat_upper'].max()),
            'trend_direction': 'increasing' if future_forecast['trend'].iloc[-1] > future_forecast['trend'].iloc[0] else 'decreasing',
            'confidence_interval': float((future_forecast['yhat_upper'].mean() - future_forecast['yhat_lower'].mean()) / 2)
        }
    
    def plot_forecast(self) -> go.Figure:
        if self.forecast is None:
            self.make_forecast()
        prophet_df = self.prepare_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'], y=prophet_df['y'],
            mode='markers', name='Historical',
            marker=dict(color='#6366f1', size=5, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=self.forecast['ds'], y=self.forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='#10b981', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([self.forecast['ds'], self.forecast['ds'][::-1]]),
            y=pd.concat([self.forecast['yhat_upper'], self.forecast['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='95% CI'
        ))
        
        last_historical = prophet_df['ds'].max()
        fig.add_vline(x=last_historical, line_dash="dash", line_color="rgba(245, 158, 11, 0.5)")
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{self.target_col.replace("_", " ").title()} Forecast',
            hovermode='x unified',
            height=500
        )
        return fig
    
    def plot_components(self) -> go.Figure:
        if self.forecast is None:
            self.make_forecast()
        
        fig = make_subplots(rows=3, cols=1,
            subplot_titles=('Trend', 'Weekly Pattern', 'Yearly Pattern'),
            vertical_spacing=0.12)
        
        fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['trend'],
            mode='lines', line=dict(color='#6366f1', width=2)), row=1, col=1)
        
        if 'weekly' in self.forecast.columns:
            fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['weekly'],
                mode='lines', line=dict(color='#10b981', width=2)), row=2, col=1)
        
        if 'yearly' in self.forecast.columns:
            fig.add_trace(go.Scatter(x=self.forecast['ds'], y=self.forecast['yearly'],
                mode='lines', line=dict(color='#f59e0b', width=2)), row=3, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600, showlegend=False
        )
        return fig


# =============================================================================
# VISUALIZATION BUILDER with Chart Customization
# =============================================================================

class VisualizationBuilder:
    """Build customizable interactive visualizations."""
    
    COLOR_PALETTES = {
        'viridis': px.colors.sequential.Viridis,
        'plasma': px.colors.sequential.Plasma,
        'blues': px.colors.sequential.Blues,
        'greens': px.colors.sequential.Greens,
        'reds': px.colors.sequential.Reds,
        'purples': px.colors.sequential.Purples,
        'sunset': px.colors.sequential.Sunset,
        'turbo': px.colors.sequential.Turbo
    }
    
    def __init__(self, df: pd.DataFrame, color_palette: str = 'viridis'):
        self.df = df
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
        self.date_col = detect_date_column(df)
        self.color_palette = color_palette
    
    def create_overview_metrics(self) -> List[Dict[str, Any]]:
        metrics = []
        priority_cols = ['sales', 'revenue', 'profit', 'quantity', 'amount']
        cols_to_use = [c for c in priority_cols if c in self.numeric_cols]
        cols_to_use.extend([c for c in self.numeric_cols if c not in cols_to_use])
        
        for col in cols_to_use[:4]:
            is_cumulative = any(k in col.lower() for k in ['sales', 'revenue', 'profit', 'quantity', 'amount', 'total', 'cost'])
            current = self.df[col].sum() if is_cumulative else self.df[col].mean()
            
            half = len(self.df) // 2
            prev = self.df[col].iloc[:half].sum() if is_cumulative else self.df[col].iloc[:half].mean()
            delta = ((current - prev) / abs(prev)) * 100 if prev != 0 else 0
            
            metrics.append({
                'label': col.replace('_', ' ').title(),
                'value': current,
                'delta': delta
            })
        return metrics
    
    def plot_time_series(self, value_col: str, agg: str = 'sum') -> go.Figure:
        if not self.date_col:
            return None
        
        if agg == 'sum':
            ts_data = self.df.groupby(self.date_col)[value_col].sum().reset_index()
        else:
            ts_data = self.df.groupby(self.date_col)[value_col].mean().reset_index()
        ts_data = ts_data.sort_values(self.date_col)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data[self.date_col], y=ts_data[value_col],
            mode='lines', line=dict(color='#6366f1', width=2),
            fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.15)',
            name=value_col.replace("_", " ").title()
        ))
        
        window = min(7, len(ts_data) // 4)
        if window > 1:
            ts_data['ma'] = ts_data[value_col].rolling(window=window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=ts_data[self.date_col], y=ts_data['ma'],
                mode='lines', line=dict(color='#f59e0b', width=2, dash='dash'),
                name=f'{window}-Day MA'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{value_col.replace("_", " ").title()} Over Time',
            hovermode='x unified', height=400
        )
        return fig
    
    def plot_categorical_breakdown(self, category_col: str, value_col: str, chart_type: str = 'bar') -> go.Figure:
        data = self.df.groupby(category_col)[value_col].sum().sort_values(ascending=True).reset_index()
        
        if chart_type == 'bar':
            fig = px.bar(data, x=value_col, y=category_col, orientation='h',
                        color=value_col, color_continuous_scale=self.color_palette,
                        title=f'{value_col.replace("_", " ").title()} by {category_col.replace("_", " ").title()}')
        elif chart_type == 'pie':
            fig = px.pie(data, values=value_col, names=category_col,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title=f'{value_col.replace("_", " ").title()} Distribution')
            fig.update_traces(textposition='inside', textinfo='percent+label')
        elif chart_type == 'treemap':
            fig = px.treemap(data, path=[category_col], values=value_col,
                           color=value_col, color_continuous_scale=self.color_palette,
                           title=f'{value_col.replace("_", " ").title()} Treemap')
        else:
            fig = px.bar(data, x=category_col, y=value_col,
                        color=value_col, color_continuous_scale=self.color_palette)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    def plot_distribution(self, column: str) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.df[column], nbinsx=50,
            marker=dict(color='#6366f1', line=dict(color='#818cf8', width=1))
        ))
        
        mean_val = self.df[column].mean()
        median_val = self.df[column].median()
        fig.add_vline(x=mean_val, line_dash="solid", line_color="#10b981",
                     annotation_text=f"Mean: {mean_val:,.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#f59e0b",
                     annotation_text=f"Median: {median_val:,.2f}")
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'Distribution of {column.replace("_", " ").title()}',
            height=400
        )
        return fig
    
    def plot_correlation_matrix(self) -> go.Figure:
        corr = self.df[self.numeric_cols].corr()
        
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=[c.replace("_", " ").title() for c in corr.columns],
            y=[c.replace("_", " ").title() for c in corr.index],
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10)
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title='Correlation Matrix',
            height=450
        )
        return fig
    
    def plot_scatter(self, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        fig = px.scatter(
            self.df, x=x_col, y=y_col,
            color=color_col if color_col else None,
            trendline='ols',
            title=f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    def plot_heatmap(self, x_col: str, y_col: str, value_col: str) -> go.Figure:
        pivot = self.df.pivot_table(values=value_col, index=y_col, columns=x_col, aggfunc='mean')
        
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=self.color_palette,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{value_col.replace("_", " ").title()} Heatmap',
            height=450
        )
        return fig


# =============================================================================
# REPORT GENERATOR - Export to PDF/PowerPoint
# =============================================================================

class ReportGenerator:
    """Generate exportable reports in multiple formats."""
    
    def __init__(self, df: pd.DataFrame, insights: List[Dict], recommendations: List[Dict]):
        self.df = df
        self.insights = insights
        self.recommendations = recommendations
        self.date_col = detect_date_column(df)
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary markdown."""
        high_priority = [i for i in self.insights if i.get('priority') == 'high']
        
        date_range = ""
        if self.date_col:
            date_range = f"**Analysis Period:** {self.df[self.date_col].min().strftime('%B %d, %Y')} to {self.df[self.date_col].max().strftime('%B %d, %Y')}"
        
        summary = f"""# 📊 AI Analytics Executive Report
*Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*

---

## Overview

- **Total Records Analyzed:** {len(self.df):,}
- **Data Columns:** {len(self.df.columns)}
- **Critical Findings:** {len(high_priority)}
- {date_range}

---

## 🔴 Key Findings

"""
        for i, insight in enumerate(high_priority[:5], 1):
            summary += f"### {i}. {insight['title']}\n\n"
            summary += f"{insight.get('narrative', insight['description'])}\n\n"
        
        summary += "\n---\n\n## 💡 Recommendations\n\n"
        
        for rec in self.recommendations[:5]:
            summary += f"### {rec['icon']} {rec['title']}\n\n"
            summary += f"{rec['description']}\n\n"
            summary += f"**Action:** {rec['action']}\n\n"
        
        return summary
    
    def generate_html_report(self) -> str:
        """Generate full HTML report."""
        high_priority = [i for i in self.insights if i.get('priority') == 'high']
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI Analytics Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #0f172a, #1e1b4b); 
            color: #f1f5f9; 
            padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ 
            font-size: 2.5rem; 
            background: linear-gradient(135deg, #6366f1, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        h2 {{ color: #818cf8; margin: 30px 0 15px; border-bottom: 2px solid #6366f1; padding-bottom: 10px; }}
        h3 {{ color: #10b981; margin: 20px 0 10px; }}
        .meta {{ color: #94a3b8; margin-bottom: 30px; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ 
            background: rgba(30, 41, 59, 0.8); 
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid rgba(99, 102, 241, 0.3);
            flex: 1;
            min-width: 150px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2rem; color: #6366f1; font-weight: bold; }}
        .stat-label {{ color: #94a3b8; font-size: 0.9rem; }}
        .insight {{ 
            background: rgba(99, 102, 241, 0.1); 
            border-left: 4px solid #6366f1;
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0;
        }}
        .insight-high {{ border-left-color: #ef4444; }}
        .insight-medium {{ border-left-color: #f59e0b; }}
        .recommendation {{ 
            background: rgba(16, 185, 129, 0.1); 
            border-left: 4px solid #10b981;
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0;
        }}
        .action {{ 
            background: rgba(16, 185, 129, 0.2); 
            padding: 8px 15px; 
            border-radius: 6px; 
            display: inline-block;
            margin-top: 10px;
            color: #34d399;
        }}
        p {{ margin: 10px 0; color: #e2e8f0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AI Analytics Report</h1>
        <p class="meta">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(self.df):,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.df.columns)}</div>
                <div class="stat-label">Data Columns</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(high_priority)}</div>
                <div class="stat-label">Critical Findings</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(self.insights)}</div>
                <div class="stat-label">Total Insights</div>
            </div>
        </div>
        
        <h2>🔍 Key Insights</h2>
"""
        
        for insight in self.insights[:10]:
            priority_class = f"insight-{insight.get('priority', 'medium')}"
            html += f"""
        <div class="insight {priority_class}">
            <h3>{insight['icon']} {insight['title']}</h3>
            <p>{insight.get('narrative', insight['description'])}</p>
        </div>
"""
        
        html += "\n        <h2>💡 Recommendations</h2>\n"
        
        for rec in self.recommendations[:5]:
            html += f"""
        <div class="recommendation">
            <h3>{rec['icon']} {rec['title']}</h3>
            <p>{rec['description']}</p>
            <div class="action">→ {rec['action']}</div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>"""
        
        return html
    
    def generate_csv_summary(self) -> str:
        """Generate insights as CSV."""
        rows = []
        for insight in self.insights:
            rows.append({
                'Type': insight['type'],
                'Priority': insight.get('priority', 'medium'),
                'Title': insight['title'],
                'Description': insight['description'],
                'Metric': insight.get('metric', ''),
                'Value': insight.get('value', '')
            })
        return pd.DataFrame(rows).to_csv(index=False)


# =============================================================================
# TUTORIAL/HELP SYSTEM
# =============================================================================

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
            st.markdown(f"<span class='suggested-query'>{suggestion}</span>", unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>🚀 AI Analytics Dashboard <span class="version-badge">v2.0</span></h1>
        <p>Enterprise-grade analytics with AI-powered insights, narrative explanations, and predictive intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Management")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload File"],
            label_visibility="collapsed"
        )
        
        df = None
        
        if data_source == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload CSV or Excel files",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload one or more data files"
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    try:
                        if file.name.endswith('.csv'):
                            temp_df = pd.read_csv(file)
                            temp_df = sanitize_dataframe(temp_df)
                        else:
                            # Excel with multiple sheets
                            excel_file = pd.ExcelFile(file)
                            if len(excel_file.sheet_names) > 1:
                                sheet = st.selectbox(
                                    f"Select sheet from {file.name}",
                                    excel_file.sheet_names,
                                    key=f"sheet_{file.name}"
                                )
                                temp_df = pd.read_excel(file, sheet_name=sheet)
                            else:
                                temp_df = pd.read_excel(file)
                        
                        # Store in session state
                        dataset_name = file.name.rsplit('.', 1)[0]
                        st.session_state.datasets[dataset_name] = temp_df
                        
                        if st.session_state.active_dataset is None:
                            st.session_state.active_dataset = dataset_name
                        
                        st.success(f"✅ Loaded {dataset_name}: {len(temp_df):,} rows")
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
            
            # Dataset selector if multiple datasets
            if len(st.session_state.datasets) > 1:
                st.markdown("---")
                st.markdown("**Active Dataset:**")
                st.session_state.active_dataset = st.selectbox(
                    "Select dataset",
                    list(st.session_state.datasets.keys()),
                    label_visibility="collapsed"
                )
            
            if st.session_state.active_dataset:
                df = st.session_state.datasets[st.session_state.active_dataset]
        else:
            sample_size = st.slider("Sample size", 500, 5000, 2000, 500)
            df = generate_sample_data(sample_size)
        
        if df is None:
            df = generate_sample_data(2000)
        
        # Parse date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        st.markdown("---")
        
        # Data Quality Score
        quality = calculate_data_quality_score(df)
        quality_class = 'excellent' if quality['overall'] >= 90 else 'good' if quality['overall'] >= 70 else 'poor'
        
        st.markdown(f"""
        <div class="data-quality">
            <div class="quality-score {quality_class}">{quality['overall']:.0f}%</div>
            <div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">Data Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Quality Details"):
            st.write(f"Completeness: {quality['completeness']:.1f}%")
            st.write(f"Uniqueness: {quality['uniqueness']:.1f}%")
            st.write(f"Missing Cells: {quality['missing_cells']:,}")
            st.write(f"Duplicate Rows: {quality['duplicate_rows']:,}")
        
        st.markdown("---")
        
        # Quick Stats
        date_col = detect_date_column(df)
        numeric_cols = detect_numeric_columns(df)
        categorical_cols = detect_categorical_columns(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Numeric", len(numeric_cols))
        with col2:
            st.metric("Columns", len(df.columns))
            st.metric("Categories", len(categorical_cols))
        
        if date_col:
            st.caption(f"📅 Date: {date_col}")
            st.caption(f"{df[date_col].min().strftime('%Y-%m-%d')} → {df[date_col].max().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        # Chart Customization
        st.markdown("### 🎨 Chart Settings")
        st.session_state.color_palette = st.selectbox(
            "Color Palette",
            list(VisualizationBuilder.COLOR_PALETTES.keys()),
            index=0
        )
        
        st.markdown("---")
        
        # Help toggle
        st.session_state.show_tutorial = st.checkbox("Show Tutorial", value=False)
    
    # Tutorial
    if st.session_state.show_tutorial:
        render_tutorial()
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🔍 Insights",
        "📈 Predictions",
        "💬 Ask Data",
        "🔧 Data Tools",
        "📄 Reports"
    ])
    
    # Initialize engines
    viz_builder = VisualizationBuilder(df, st.session_state.color_palette)
    
    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown("### 📊 Key Performance Metrics")
        
        metrics = viz_builder.create_overview_metrics()
        if len(metrics) == 0:
            st.info("ℹ️ No numeric KPIs detected in this dataset.")
        else:
            cols = st.columns(min(len(metrics), 4))

        
        for col, metric in zip(cols, metrics):
            with col:
                delta_class = "metric-delta-positive" if metric['delta'] >= 0 else "metric-delta-negative"
                delta_symbol = "↑" if metric['delta'] >= 0 else "↓"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metric['value']:,.0f}</div>
                    <div class="metric-label">{metric['label']}</div>
                    <div class="{delta_class}">{delta_symbol} {abs(metric['delta']):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Filters
        if categorical_cols:
            st.markdown("### 🔍 Quick Filters")
            filter_cols = st.columns(min(len(categorical_cols), 4))
            
            active_filters = {}
            for i, cat_col in enumerate(categorical_cols[:4]):
                with filter_cols[i]:
                    selected = st.multiselect(
                        cat_col.replace("_", " ").title(),
                        options=df[cat_col].unique().tolist(),
                        default=None,
                        key=f"filter_{cat_col}"
                    )
                    if selected:
                        active_filters[cat_col] = selected
            
            # Apply filters
            df_filtered = df.copy()
            for col, values in active_filters.items():
                df_filtered = df_filtered[df_filtered[col].isin(values)]
            
            if active_filters:
                st.info(f"Showing {len(df_filtered):,} of {len(df):,} records")
                viz_builder = VisualizationBuilder(df_filtered, st.session_state.color_palette)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if date_col and numeric_cols:
                selected_metric = st.selectbox("Time Series Metric", numeric_cols, key="ts_metric")
                fig = viz_builder.plot_time_series(selected_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="ts_chart")
        
        with col2:
            if categorical_cols and numeric_cols:
                c1, c2, c3 = st.columns(3)
                with c1:
                    selected_cat = st.selectbox("Category", categorical_cols, key="cat_select")
                with c2:
                    selected_val = st.selectbox("Metric", numeric_cols, key="val_select")
                with c3:
                    chart_type = st.selectbox("Chart", ["bar", "pie", "treemap"], key="chart_type")
                
                fig = viz_builder.plot_categorical_breakdown(selected_cat, selected_val, chart_type)
                st.plotly_chart(fig, use_container_width=True, key="cat_chart")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if numeric_cols:
                dist_col = st.selectbox("Distribution", numeric_cols, key="dist_col")
                fig = viz_builder.plot_distribution(dist_col)
                st.plotly_chart(fig, use_container_width=True, key="dist_chart")
        
        with col4:
            if len(numeric_cols) >= 2:
                fig = viz_builder.plot_correlation_matrix()
                st.plotly_chart(fig, use_container_width=True, key="corr_chart")
    
    # =========================================================================
    # TAB 2: INSIGHTS with Narratives
    # =========================================================================
    with tab2:
        st.markdown("### 🔍 AI-Powered Insights with Business Narratives")
        
        with st.spinner("🤖 Analyzing your data..."):
            insights_engine = EnhancedInsightsEngine(df)
            insights, recommendations = insights_engine.generate_all_insights()
        
        # Executive Summary
        narrative_engine = NarrativeEngine(df)
        exec_summary = narrative_engine.generate_executive_summary(insights)
        
        st.markdown(f"""
        <div class="narrative-card">
            <h4>📋 Executive Summary</h4>
            <p>Analysis of <strong>{len(df):,}</strong> records reveals <strong>{len([i for i in insights if i['priority'] == 'high'])}</strong> critical findings requiring attention.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Insights with narratives
            high_priority = [i for i in insights if i['priority'] == 'high']
            medium_priority = [i for i in insights if i['priority'] == 'medium']
            
            if high_priority:
                st.markdown("#### 🔴 Critical Findings")
                for insight in high_priority:
                    st.markdown(f"""
                    <div class="insight-card high-priority animate-slide-in">
                        <span class="insight-icon">{insight['icon']}</span>
                        <span class="insight-title">{insight['title']}</span>
                        <div class="insight-description">{insight['description']}</div>
                        <div class="insight-narrative">{insight.get('narrative', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if medium_priority:
                with st.expander(f"🟡 Additional Insights ({len(medium_priority)})"):
                    for insight in medium_priority:
                        st.markdown(f"""
                        <div class="insight-card medium-priority">
                            <span class="insight-icon">{insight['icon']}</span>
                            <span class="insight-title">{insight['title']}</span>
                            <div class="insight-description">{insight['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            for rec in recommendations[:5]:
                st.markdown(f"""
                <div class="recommendation-card animate-fade-in">
                    <div class="recommendation-title">{rec['icon']} {rec['title']}</div>
                    <div class="recommendation-description">{rec['description']}</div>
                    <div class="recommendation-action">→ {rec['action']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 3: PREDICTIONS
    # =========================================================================
    with tab3:
        st.markdown("### 📈 Predictive Analytics")
        
        if not date_col:
            st.warning("⚠️ No date column detected. Forecasting requires time series data.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_col = st.selectbox("Target Metric", numeric_cols, key="pred_target")
            with col2:
                forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
            with col3:
                yearly = st.checkbox("Yearly Seasonality", value=True)
                weekly = st.checkbox("Weekly Seasonality", value=True)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        pred_engine = PredictiveEngine(df, date_col, target_col)
                        pred_engine.train_model(yearly_seasonality=yearly, weekly_seasonality=weekly)
                        forecast = pred_engine.make_forecast(periods=forecast_days)
                        
                        fig = pred_engine.plot_forecast()
                        st.plotly_chart(fig, use_container_width=True)
                        
                        summary = pred_engine.get_forecast_summary()
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Predicted Avg", format_number(summary['predicted_mean']))
                        with cols[1]:
                            st.metric("Range", f"{format_number(summary['predicted_min'])} - {format_number(summary['predicted_max'])}")
                        with cols[2]:
                            st.metric("Trend", summary['trend_direction'].title())
                        with cols[3]:
                            st.metric("Confidence ±", format_number(summary['confidence_interval']))
                        
                        with st.expander("📊 Forecast Components"):
                            fig_comp = pred_engine.plot_components()
                            st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Download forecast
                        forecast_dl = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_dl.columns = ['Date', 'Forecast', 'Lower', 'Upper']
                        st.download_button("📥 Download Forecast", forecast_dl.to_csv(index=False),
                                          "forecast.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Forecasting error: {e}")
    
    # =========================================================================
    # TAB 4: ASK DATA (Smart NL Queries)
    # =========================================================================
    with tab4:
        st.markdown("### 💬 Ask Your Data")
        
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., 'Total sales by region' or 'What insights can you find?'",
            label_visibility="collapsed",
            key="nl_query"
        )
        
        # Suggested queries
        suggestions = get_suggested_queries(df)
        st.markdown("**💡 Try these:**")
        suggestion_cols = st.columns(4)
        for i, sugg in enumerate(suggestions[:8]):
            with suggestion_cols[i % 4]:
                if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                    query = sugg
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if query:
            with st.spinner("🔍 Processing..."):
                query_engine = SmartQueryEngine(df)
                result = query_engine.process_query(query)
                
                st.markdown(result['text'])
                
                if result.get('narrative'):
                    st.markdown(f"""
                    <div class="narrative-card">
                        <h4>💡 Analysis</h4>
                        <p>{result['narrative']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if result.get('figure'):
                    st.plotly_chart(result['figure'], use_container_width=True)
                
                if isinstance(result.get('data'), pd.DataFrame) and len(result['data']) > 0:
                    st.dataframe(result['data'], use_container_width=True, hide_index=True)
                
                if result.get('follow_up_suggestions'):
                    st.markdown("**Follow-up questions:**")
                    for sugg in result['follow_up_suggestions'][:3]:
                        st.markdown(f"• {sugg}")
    
    # =========================================================================
    # TAB 5: DATA TOOLS
    # =========================================================================
    with tab5:
        st.markdown("### 🔧 Data Preprocessing & Cleaning")
        
        preprocessor = DataPreprocessor(df)
        profile = preprocessor.profile_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Profile")
            
            if profile['issues']:
                st.warning(f"⚠️ {len(profile['issues'])} issues detected")
                for issue in profile['issues'][:5]:
                    st.write(f"• {issue}")
            else:
                st.success("✅ No major issues detected")
            
            st.markdown("#### Column Statistics")
            col_stats = []
            for col, info in profile['columns'].items():
                col_stats.append({
                    'Column': col,
                    'Type': info['dtype'],
                    'Null %': f"{info['null_pct']:.1f}%",
                    'Unique': info['unique_count']
                })
            st.dataframe(pd.DataFrame(col_stats), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 🛠️ Preprocessing Options")
            
            with st.expander("Handle Missing Values"):
                missing_strategy = st.selectbox(
                    "Strategy",
                    ["None", "Auto (Smart)", "Mean", "Median", "Mode", "Drop Rows", "Fill with Zero"]
                )
                
                if st.button("Apply Missing Value Handling"):
                    if missing_strategy != "None":
                        strategy_map = {
                            "Auto (Smart)": "auto",
                            "Mean": "mean",
                            "Median": "median",
                            "Mode": "mode",
                            "Drop Rows": "drop",
                            "Fill with Zero": "zero"
                        }
                        preprocessor.handle_missing_values(strategy_map[missing_strategy])
                        st.success(f"✅ Applied: {', '.join(preprocessor.get_transformation_log())}")
            
            with st.expander("Remove Outliers"):
                outlier_method = st.selectbox("Method", ["IQR", "Z-Score"])
                outlier_threshold = st.slider("Threshold", 1.0, 3.0, 1.5, 0.1)
                
                if st.button("Remove Outliers"):
                    preprocessor.remove_outliers(method=outlier_method.lower(), threshold=outlier_threshold)
                    st.success(f"✅ {', '.join(preprocessor.get_transformation_log())}")
            
            with st.expander("Create Date Features"):
                if date_col:
                    if st.button("Generate Date Features"):
                        preprocessor.create_date_features()
                        st.success("✅ Created year, month, day_of_week, quarter, etc.")
                else:
                    st.info("No date column detected")
        
        # Data Preview
        st.markdown("#### 📄 Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=300)
    
    # =========================================================================
    # TAB 6: REPORTS
    # =========================================================================
    with tab6:
        st.markdown("### 📄 Export Reports")
        
        # Generate insights if not already done
        if 'insights' not in dir() or not insights:
            insights_engine = EnhancedInsightsEngine(df)
            insights, recommendations = insights_engine.generate_all_insights()
        
        report_gen = ReportGenerator(df, insights, recommendations)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📝 Markdown Report")
            md_report = report_gen.generate_executive_summary()
            st.download_button(
                "📥 Download Markdown",
                md_report,
                "analytics_report.md",
                "text/markdown",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🌐 HTML Report")
            html_report = report_gen.generate_html_report()
            st.download_button(
                "📥 Download HTML",
                html_report,
                "analytics_report.html",
                "text/html",
                use_container_width=True
            )
        
        with col3:
            st.markdown("#### 📊 Insights CSV")
            csv_report = report_gen.generate_csv_summary()
            st.download_button(
                "📥 Download CSV",
                csv_report,
                "insights.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Data Export
        st.markdown("#### 💾 Data Export")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Full Dataset (CSV)",
                df.to_csv(index=False),
                "data_export.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            summary = {
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'columns': list(df.columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'date_column': date_col,
                'data_quality': quality
            }
            st.download_button(
                "📥 Download Schema (JSON)",
                json.dumps(summary, indent=2, default=str),
                "data_schema.json",
                "application/json",
                use_container_width=True
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>AI Analytics Dashboard v2.0</strong> | Enterprise Edition</p>
        <p>Built with Streamlit, Prophet, Plotly | Open Source Alternative to Tableau AI</p>
        <p style="margin-top: 10px; font-size: 0.85rem;">
            📊 Automated Insights • 📝 Narrative Explanations • 📈 Predictive Analytics • 💬 Smart Queries
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
