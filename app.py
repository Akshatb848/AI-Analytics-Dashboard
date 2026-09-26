"""
AI Analytics Dashboard v2.0 - Enterprise Edition
Alternative to Tableau AI / Power BI Copilot

This module is the Streamlit page: layout, session state and caching. The
analysis itself lives in the `analytics` package and the styling in `ui`.
"""
from semantic_engine import (
    DatasetProfiler,
    SemanticClassifier,
    MetricIntelligenceEngine,
    SemanticCatalog
)
from analytics import data_utils
from analytics.data_utils import (  # noqa: F401  (re-exported for tests and callers)
    TYPE_CONVERSION_THRESHOLD,
    _is_numeric_dtype,
    _is_text_dtype,
    _parse_datetime,
    calculate_data_quality_score,
    detect_categorical_columns,
    detect_date_column,
    detect_numeric_columns,
    get_suggested_queries,
    sanitize_dataframe,
    semantic_catalog_template,
)
from analytics.dashboards import DashboardFileError, export_dashboards, import_dashboards
from analytics.forecasting import PredictiveEngine
from analytics.formatting import format_number, narrative_html, safe_html
from analytics.insights import EnhancedInsightsEngine, NarrativeEngine  # noqa: F401
from analytics.llm import GLMClient, LLMConfig, LLMError, LLMQueryPlanner, describe_schema
from analytics.query_parser import normalize_intent
from analytics.preprocessing import DataPreprocessor
from analytics.query_engine import SmartQueryEngine
from analytics.reports import ReportGenerator
from analytics.visualization import VisualizationBuilder
from ui import auth
from ui.components import render_tutorial
from ui.styles import CUSTOM_CSS
import logging
import streamlit as st
import pandas as pd
import json
import hashlib
from datetime import datetime
import os
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

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
    st.session_state.saved_dashboards = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True
if 'chart_theme' not in st.session_state:
    st.session_state.chart_theme = 'default'
if 'color_palette' not in st.session_state:
    st.session_state.color_palette = 'viridis'
if 'semantic_catalog' not in st.session_state:
    st.session_state.semantic_catalog = None
if 'semantic_issues' not in st.session_state:
    st.session_state.semantic_issues = []
if 'semantic_schema' not in st.session_state:
    st.session_state.semantic_schema = None
if 'processed_data' not in st.session_state:
    # Cleaned DataFrames from the Data Tools tab, keyed by dataset
    st.session_state.processed_data = {}
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = {}

# =============================================================================
# ENHANCED CUSTOM STYLING
# =============================================================================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================







# The only markup the app itself puts in narratives; restored after escaping




# Share of non-empty values that must convert for a text column to be retyped















def load_semantic_catalog(payload: Dict[str, Any], df: pd.DataFrame) -> None:
    catalog = SemanticCatalog.from_dict(payload)
    st.session_state.semantic_catalog = catalog
    st.session_state.semantic_issues = catalog.validate(df)


def resolve_semantic_schema(df: pd.DataFrame) -> Dict[str, Any]:
    catalog = st.session_state.semantic_catalog
    if not catalog:
        return {
            "numeric_cols": detect_numeric_columns(df),
            "categorical_cols": detect_categorical_columns(df),
            "date_col": detect_date_column(df),
            "metric_definitions": []
        }

    metric_columns = [col for col in catalog.metric_columns() if col in df.columns]
    numeric_cols = metric_columns if metric_columns else detect_numeric_columns(df)
    dimensions = [col for col in catalog.dimensions if col in df.columns]
    categorical_cols = dimensions if dimensions else detect_categorical_columns(df)
    date_col = catalog.time_column if catalog.time_column in df.columns else detect_date_column(df)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_col": date_col,
        "metric_definitions": catalog.metrics
    }






# Sample data is deterministic, so it is cached across reruns and sessions
generate_sample_data = st.cache_data(data_utils.generate_sample_data)


# Largest number of rows read from an uploaded file (override with MAX_UPLOAD_ROWS)
MAX_UPLOAD_ROWS = int(os.environ.get("MAX_UPLOAD_ROWS", 200_000))


@st.cache_data(show_spinner=False, max_entries=16)
def list_excel_sheets(content: bytes) -> List[str]:
    return pd.ExcelFile(BytesIO(content)).sheet_names


@st.cache_data(show_spinner="Loading file...", max_entries=16)
def load_uploaded_file(content: bytes, file_name: str, sheet: Optional[str] = None) -> Dict[str, Any]:
    """Read, sanitize and profile an uploaded CSV/Excel file, capped at MAX_UPLOAD_ROWS."""
    buffer = BytesIO(content)
    if file_name.lower().endswith(".csv"):
        raw = pd.read_csv(buffer, nrows=MAX_UPLOAD_ROWS + 1)
    else:
        raw = pd.read_excel(buffer, sheet_name=sheet or 0, nrows=MAX_UPLOAD_ROWS + 1)

    df = sanitize_dataframe(raw.head(MAX_UPLOAD_ROWS))
    profiles = DatasetProfiler(df).profile()
    semantic = SemanticClassifier(profiles).classify()
    return {
        "df": df,
        "truncated": len(raw) > MAX_UPLOAD_ROWS,
        "profiles": profiles,
        "semantic": semantic,
        "kpis": MetricIntelligenceEngine(df, semantic).discover_kpis(),
    }


# -----------------------------------------------------------------------------
# Cached analysis. Streamlit reruns the whole script on every click, so the
# expensive steps are cached per dataset. They take the DataFrame as `_df`
# (not hashed) plus an exact content fingerprint: Streamlit's own DataFrame
# hashing only samples rows of large frames and could miss a cleaning step.
# -----------------------------------------------------------------------------
ANALYSIS_CACHE = dict(show_spinner=False, max_entries=32, ttl=3600)


def data_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update(repr((list(df.columns), [str(dtype) for dtype in df.dtypes])).encode())
    return digest.hexdigest()


@st.cache_data(**ANALYSIS_CACHE)
def compute_quality(_df: pd.DataFrame, fingerprint: str) -> Dict[str, Any]:
    return calculate_data_quality_score(_df)


@st.cache_data(**ANALYSIS_CACHE)
def compute_insights(_df: pd.DataFrame, fingerprint: str) -> Tuple[List[Dict], List[Dict]]:
    return EnhancedInsightsEngine(_df).generate_all_insights()


@st.cache_data(**ANALYSIS_CACHE)
def compute_profile(_df: pd.DataFrame, fingerprint: str) -> Dict[str, Any]:
    return DataPreprocessor(_df).profile_data()


@st.cache_data(**ANALYSIS_CACHE)
def dataframe_to_csv(_df: pd.DataFrame, fingerprint: str) -> str:
    return _df.to_csv(index=False)


@st.cache_data(**ANALYSIS_CACHE)
def run_query(_df: pd.DataFrame, fingerprint: str, query: str, numeric_cols: List[str],
              categorical_cols: List[str], date_col: Optional[str],
              plan_json: Optional[str] = None) -> Dict[str, Any]:
    """Answer a question with the built-in parser, or run a plan from the LLM when given."""
    engine = SmartQueryEngine(_df, numeric_cols=numeric_cols,
                              categorical_cols=categorical_cols, date_col=date_col)
    return engine.process_query(query, intent=json.loads(plan_json) if plan_json else None)


# -----------------------------------------------------------------------------
# Optional GLM query planning for Ask Data (see analytics/llm.py)
# -----------------------------------------------------------------------------

def llm_config() -> Optional[LLMConfig]:
    """GLM settings from the environment or .streamlit/secrets.toml; None without a key."""
    return LLMConfig.from_env(secrets=lambda name: st.secrets.get(name))


@st.cache_data(show_spinner=False, max_entries=256, ttl=3600)
def plan_with_llm(_schema: Dict[str, Any], fingerprint: str, query: str, model: str) -> Dict[str, Any]:
    """Ask the model for a query plan. Errors are raised, so failed calls are not cached."""
    config = llm_config()
    if config is None:
        raise LLMError("no API key configured")
    return LLMQueryPlanner(GLMClient(config)).plan(query, _schema)


def answer_question(df: pd.DataFrame, fingerprint: str, query: str, numeric_cols: List[str],
                    categorical_cols: List[str], date_col: Optional[str], use_llm: bool) -> Dict[str, Any]:
    """Answer with a GLM-planned query when enabled, falling back to the built-in parser."""
    source, note, plan_json = "built-in parser", None, None
    config = llm_config() if use_llm else None
    if config is not None:
        try:
            schema = describe_schema(df, numeric_cols, categorical_cols, date_col)
            raw = plan_with_llm(schema, fingerprint, query, config.model)
            plan = normalize_intent(raw, df, numeric_cols, categorical_cols, date_col)
            if plan["type"] is None and plan["aggregation"] is None:
                note = f"{config.model} couldn't map this question to the data; answered with the built-in parser."
            else:
                plan_json = json.dumps(raw, sort_keys=True, default=str)
                source = config.model
        except LLMError as e:
            logger.warning("LLM planning failed, using the built-in parser: %s", e)
            note = f"{config.model} was unavailable ({e}); answered with the built-in parser."

    result = run_query(df, fingerprint, query, numeric_cols, categorical_cols, date_col, plan_json)
    result['source'] = source
    result['llm_note'] = note
    return result


# =============================================================================
# DATA PREPROCESSING CLASS
# =============================================================================



# =============================================================================
# NARRATIVE INSIGHTS ENGINE - Generate Business Narratives
# =============================================================================



# =============================================================================
# ENHANCED INSIGHTS ENGINE with Narratives
# =============================================================================



# =============================================================================
# ENHANCED NL QUERY ENGINE - LLM-Ready Architecture
# =============================================================================



# =============================================================================
# PREDICTIVE ANALYTICS ENGINE
# =============================================================================



# =============================================================================
# VISUALIZATION BUILDER with Chart Customization
# =============================================================================



# =============================================================================
# REPORT GENERATOR - Export to PDF/PowerPoint
# =============================================================================





# =============================================================================
# TUTORIAL/HELP SYSTEM
# =============================================================================





# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Sign-in gate; does nothing unless [auth] is configured in secrets
    auth.require_login()
    
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
        dataset_key = "sample:2000"
        
        if data_source == "Upload File":
            uploaded_files = st.file_uploader(
                "Upload CSV or Excel files",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload one or more data files"
            )
            
            if uploaded_files:
                all_datasets = []

                for file in uploaded_files:
                    try:
                        # ==============================
                        # LOAD FILE
                        # ==============================
                        content = file.getvalue()
                        sheet = None
                        if not file.name.lower().endswith(".csv"):
                            sheet_names = list_excel_sheets(content)
                            if len(sheet_names) > 1:
                                sheet = st.selectbox(
                                    f"Select sheet from {file.name}",
                                    sheet_names,
                                    key=f"sheet_{file.name}"
                                )

                        # Read, sanitize and profile once per file (cached across reruns)
                        loaded = load_uploaded_file(content, file.name, sheet)
                        temp_df = loaded["df"]
                        profiles = loaded["profiles"]
                        semantic = loaded["semantic"]
                        kpis = loaded["kpis"]

                        if loaded["truncated"]:
                            st.warning(
                                f"⚠️ {file.name} has more than {MAX_UPLOAD_ROWS:,} rows; "
                                f"only the first {MAX_UPLOAD_ROWS:,} were loaded."
                            )

                        # ==============================
                        # STORE DATASET
                        # ==============================
                        all_datasets.append({
                            "file_name": file.name,
                            "df": temp_df,
                            "semantic": semantic,
                            "kpis": kpis,
                            "profiles": profiles
                        })

                        st.success(
                            f"✅ {file.name} loaded successfully | "
                            f"KPIs: {len(kpis)} | "
                            f"Time columns: {len(semantic.time_columns)}"
                        )

                    except Exception as e:
                        logger.exception("Failed to load upload %s", file.name)
                        st.error(f"❌ Failed to load {file.name}: {str(e)}")

                # ==============================
                # SELECT ACTIVE DATASET
                # ==============================
                if all_datasets:
                    selected_dataset = st.selectbox(
                        "Select dataset",
                        [d["file_name"] for d in all_datasets]
                    )

                    dataset = next(
                        d for d in all_datasets if d["file_name"] == selected_dataset
                    )

                    df = dataset["df"]
                    semantic = dataset["semantic"]
                    kpis = dataset["kpis"]
                    profiles = dataset["profiles"]
                    
                    st.success(f"✅ Loaded {selected_dataset}: {len(dataset['df']):,} rows")
                    
                    # Store in session state
                    st.session_state.datasets[selected_dataset] = dataset["df"]
                    if st.session_state.active_dataset is None:
                        st.session_state.active_dataset = selected_dataset
            
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
                dataset_key = f"upload:{st.session_state.active_dataset}"
        else:
            sample_size = st.slider("Sample size", 500, 5000, 2000, 500)
            df = generate_sample_data(sample_size)
            dataset_key = f"sample:{sample_size}"
        
        if df is None:
            df = generate_sample_data(2000)
            dataset_key = "sample:2000"

        # Use the cleaned version from the Data Tools tab when one exists
        if dataset_key in st.session_state.processed_data:
            df = st.session_state.processed_data[dataset_key]
            steps = len(st.session_state.processing_log.get(dataset_key, []))
            st.caption(f"🔧 Using cleaned data ({steps} step{'s' if steps != 1 else ''} applied in Data Tools)")
        
        # Parse text date columns (numeric columns like "delivery_time" are left alone)
        df = df.copy()
        for col in df.columns:
            if ('date' in col.lower() or 'time' in col.lower()) and _is_text_dtype(df[col]):
                parsed = _parse_datetime(df[col])
                if parsed.notna().sum() >= TYPE_CONVERSION_THRESHOLD * df[col].notna().sum():
                    df[col] = parsed

        st.markdown("---")
        st.markdown("### 🧠 Semantic Catalog")
        with st.expander("Load governed metrics & dimensions"):
            st.caption("Upload a semantic catalog to standardize metrics, dimensions, and time grains.")
            catalog_file = st.file_uploader(
                "Semantic catalog (JSON)",
                type=["json"],
                key="semantic_catalog_uploader"
            )
            template_payload = semantic_catalog_template()
            st.download_button(
                "⬇️ Download template",
                data=json.dumps(template_payload, indent=2),
                file_name="semantic_catalog_template.json",
                mime="application/json"
            )

            if catalog_file is not None:
                try:
                    payload = json.load(catalog_file)
                    load_semantic_catalog(payload, df)
                    if st.session_state.semantic_issues:
                        st.warning("⚠️ Semantic catalog loaded with issues:")
                        for issue in st.session_state.semantic_issues:
                            st.write(f"• {issue}")
                    else:
                        st.success("✅ Semantic catalog applied successfully.")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    st.error("❌ Invalid JSON. Please check the catalog format.")
                except ValueError as e:
                    st.error(f"❌ Invalid semantic catalog: {e}")

        st.markdown("---")
        
        # Resolve the schema first: it may convert the date column, after
        # which the data is final for this run and can be fingerprinted
        semantic_schema = resolve_semantic_schema(df)
        date_col = semantic_schema["date_col"]
        if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = _parse_datetime(df[date_col])
        fingerprint = data_fingerprint(df)

        # Data Quality Score
        quality = compute_quality(df, fingerprint)
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
        numeric_cols = semantic_schema["numeric_cols"]
        categorical_cols = semantic_schema["categorical_cols"]
        metric_definitions = semantic_schema["metric_definitions"]
        st.session_state.semantic_schema = semantic_schema
        
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

    semantic_schema = st.session_state.semantic_schema or resolve_semantic_schema(df)
    date_col = semantic_schema["date_col"]
    numeric_cols = semantic_schema["numeric_cols"]
    categorical_cols = semantic_schema["categorical_cols"]
    metric_definitions = semantic_schema["metric_definitions"]
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🔍 Insights",
        "📈 Predictions",
        "💬 Ask Data",
        "🔧 Data Tools",
        "📄 Reports",
        "🗂️ Dashboards"
    ])
    
    # Initialize engines
    viz_builder = VisualizationBuilder(
        df,
        st.session_state.color_palette,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        date_col=date_col,
        metric_definitions=metric_definitions
    )
    
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
                        <div class="metric-label">{safe_html(metric['label'])}</div>
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
                viz_builder = VisualizationBuilder(
                    df_filtered,
                    st.session_state.color_palette,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    date_col=date_col,
                    metric_definitions=metric_definitions
                )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if date_col and numeric_cols:
                selected_metric = st.selectbox("Time Series Metric", numeric_cols, key="ts_metric")
                fig = viz_builder.plot_time_series(selected_metric)
                if fig:
                    st.plotly_chart(fig, width="stretch", key="ts_chart")
        
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
                st.plotly_chart(fig, width="stretch", key="cat_chart")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if numeric_cols:
                dist_col = st.selectbox("Distribution", numeric_cols, key="dist_col")
                fig = viz_builder.plot_distribution(dist_col)
                st.plotly_chart(fig, width="stretch", key="dist_chart")
        
        with col4:
            if len(numeric_cols) >= 2:
                fig = viz_builder.plot_correlation_matrix()
                st.plotly_chart(fig, width="stretch", key="corr_chart")
    
    # =========================================================================
    # TAB 2: INSIGHTS with Narratives
    # =========================================================================
    with tab2:
        st.markdown("### 🔍 AI-Powered Insights with Business Narratives")
        
        with st.spinner("🤖 Analyzing your data..."):
            insights, recommendations = compute_insights(df, fingerprint)
        
        # Executive Summary
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
                        <span class="insight-icon">{safe_html(insight['icon'])}</span>
                        <span class="insight-title">{safe_html(insight['title'])}</span>
                        <div class="insight-description">{safe_html(insight['description'])}</div>
                        <div class="insight-narrative">{narrative_html(insight.get('narrative', ''))}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if medium_priority:
                with st.expander(f"🟡 Additional Insights ({len(medium_priority)})"):
                    for insight in medium_priority:
                        st.markdown(f"""
                        <div class="insight-card medium-priority">
                            <span class="insight-icon">{safe_html(insight['icon'])}</span>
                            <span class="insight-title">{safe_html(insight['title'])}</span>
                            <div class="insight-description">{safe_html(insight['description'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            for rec in recommendations[:5]:
                st.markdown(f"""
                <div class="recommendation-card animate-fade-in">
                    <div class="recommendation-title">{safe_html(rec['icon'])} {safe_html(rec['title'])}</div>
                    <div class="recommendation-description">{safe_html(rec['description'])}</div>
                    <div class="recommendation-action">→ {safe_html(rec['action'])}</div>
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
                        st.plotly_chart(fig, width="stretch")
                        
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
                        
                        with st.spinner("Checking accuracy on held-back data..."):
                            accuracy = pred_engine.backtest(forecast_days, yearly_seasonality=yearly,
                                                            weekly_seasonality=weekly)
                        st.markdown("#### 🎯 How reliable is this forecast?")
                        if accuracy is None:
                            st.info("Not enough history to test accuracy (needs at least ~40 dates).")
                        else:
                            verdict = PredictiveEngine.rate_accuracy(accuracy)
                            (st.success if "good" in verdict else st.warning)(verdict)
                            acc_cols = st.columns(3)
                            with acc_cols[0]:
                                st.metric("Typical error (MAPE)",
                                          "n/a" if accuracy['mape'] is None else f"{accuracy['mape']:.1f}%")
                            with acc_cols[1]:
                                st.metric("Inside 95% band", f"{accuracy['interval_coverage']:.0f}%",
                                          help="Share of held-back days whose actual value fell inside the forecast band.")
                            with acc_cols[2]:
                                gain = accuracy['improvement_vs_baseline']
                                st.metric("vs. naive guess", "n/a" if gain is None else f"{gain:+.0f}%",
                                          help="Error reduction compared with predicting the recent average.")
                            with st.expander("See the backtest"):
                                st.caption(f"Trained on data up to {accuracy['holdout_days']} days before the "
                                           f"end, then compared with the {accuracy['test_points']} held-back dates.")
                                st.plotly_chart(PredictiveEngine.plot_backtest(accuracy), width="stretch",
                                                key="backtest_chart")

                        with st.expander("📊 Forecast Components"):
                            fig_comp = pred_engine.plot_components()
                            st.plotly_chart(fig_comp, width="stretch")
                        
                        # Download forecast
                        forecast_dl = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_dl.columns = ['Date', 'Forecast', 'Lower', 'Upper']
                        st.download_button("📥 Download Forecast", forecast_dl.to_csv(index=False),
                                          "forecast.csv", "text/csv")
                    except Exception as e:
                        logger.exception("Forecast failed for %s", target_col)
                        st.error(f"Forecasting error: {e}")
    
    # =========================================================================
    # TAB 4: ASK DATA (Smart NL Queries)
    # =========================================================================
    with tab4:
        st.markdown("### 💬 Ask Your Data")
        
        glm = llm_config()
        if glm is not None:
            use_llm = st.toggle(f"Use {glm.model} to interpret questions", value=True, key="use_llm")
            st.caption("Sends column names, types, up to 15 category values per column and the date "
                       "range to Z.ai — never the data rows. The reply is checked against your columns "
                       "before anything runs.")
        else:
            use_llm = False
            st.caption("Using the built-in question parser. Set `ZAI_API_KEY` to let GLM-4.5-Flash "
                       "interpret questions (see README).")

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
                if st.button(sugg, key=f"sugg_{i}", width="stretch"):
                    query = sugg
        
        
        if query:
            with st.spinner("🔍 Processing..."):
                result = answer_question(df, fingerprint, query, numeric_cols, categorical_cols,
                                         date_col, use_llm)

                st.markdown(result['text'])
                if result.get('llm_note'):
                    st.info(result['llm_note'])
                
                if result.get('narrative'):
                    st.markdown(f"""
                    <div class="narrative-card">
                        <h4>💡 Analysis</h4>
                        <p>{narrative_html(result['narrative'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if result.get('figure'):
                    st.plotly_chart(result['figure'], width="stretch", key="ask_chart")
                
                if isinstance(result.get('data'), pd.DataFrame) and len(result['data']) > 0:
                    st.dataframe(result['data'], width="stretch", hide_index=True)
                
                if result.get('follow_up_suggestions'):
                    st.markdown("**Follow-up questions:**")
                    for sugg in result['follow_up_suggestions'][:3]:
                        st.markdown(f"• {sugg}")

                with st.expander(f"How this was answered: {result['source']}"):
                    st.json(result['intent'])

                if result.get('figure') is not None or isinstance(result.get('data'), pd.DataFrame):
                    st.markdown("---")
                    st.markdown("### 🗂️ Save to Dashboard")
                    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
                    default_title = f"Insight: {query[:45]}{'...' if len(query) > 45 else ''}"
                    card_title = st.text_input(
                        "Dashboard card title",
                        value=default_title,
                        key=f"dash_title_{query_hash}"
                    )
                    if st.button("Save card", key=f"save_card_{query_hash}"):
                        st.session_state.saved_dashboards.append({
                            "title": card_title,
                            "query": query,
                            "summary": result.get("narrative", ""),
                            "text": result.get("text", ""),
                            "figure": result.get("figure"),
                            "data": result.get("data")
                        })
                        st.success("✅ Added to Dashboards tab.")
    
    # =========================================================================
    # TAB 5: DATA TOOLS
    # =========================================================================
    with tab5:
        st.markdown("### 🔧 Data Preprocessing & Cleaning")
        
        preprocessor = DataPreprocessor(df)
        profile = compute_profile(df, fingerprint)

        def apply_preprocessing(message: str) -> None:
            """Keep the cleaned data for this dataset so every tab uses it."""
            st.session_state.processed_data[dataset_key] = preprocessor.get_transformed_data()
            st.session_state.processing_log.setdefault(dataset_key, []).extend(
                preprocessor.get_transformation_log() or [message]
            )
            st.session_state.processing_notice = message
            st.rerun()

        notice = st.session_state.pop('processing_notice', None)
        if notice:
            st.success(f"✅ {notice}")

        applied_steps = st.session_state.processing_log.get(dataset_key, [])
        if applied_steps:
            with st.expander(f"🧾 Applied cleaning steps ({len(applied_steps)})"):
                for step in applied_steps:
                    st.write(f"• {step}")
                if st.button("↩️ Reset to original data", key="reset_preprocessing"):
                    st.session_state.processed_data.pop(dataset_key, None)
                    st.session_state.processing_log.pop(dataset_key, None)
                    st.session_state.processing_notice = "Restored the original data"
                    st.rerun()
        
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
            st.dataframe(pd.DataFrame(col_stats), width="stretch", hide_index=True)
        
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
                        apply_preprocessing(f"Missing values handled ({missing_strategy})")
            
            with st.expander("Remove Outliers"):
                outlier_method = st.selectbox("Method", ["IQR", "Z-Score"])
                outlier_threshold = st.slider("Threshold", 1.0, 3.0, 1.5, 0.1)
                
                if st.button("Remove Outliers"):
                    method = {"IQR": "iqr", "Z-Score": "zscore"}[outlier_method]
                    preprocessor.remove_outliers(method=method, threshold=outlier_threshold)
                    apply_preprocessing(f"Outlier removal ({outlier_method}, threshold {outlier_threshold}): "
                                        f"{len(df) - len(preprocessor.get_transformed_data()):,} rows removed")
            
            with st.expander("Create Date Features"):
                if date_col:
                    if st.button("Generate Date Features"):
                        preprocessor.create_date_features()
                        apply_preprocessing("Created year, month, day_of_week, quarter, etc.")
                else:
                    st.info("No date column detected")
        
        # Data Preview
        st.markdown("#### 📄 Data Preview")
        st.dataframe(df.head(100), width="stretch", height=300)
    
    # =========================================================================
    # TAB 6: REPORTS
    # =========================================================================
    with tab6:
        st.markdown("### 📄 Export Reports")
        
        insights, recommendations = compute_insights(df, fingerprint)
        
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
                width="stretch"
            )
        
        with col2:
            st.markdown("#### 🌐 HTML Report")
            html_report = report_gen.generate_html_report()
            st.download_button(
                "📥 Download HTML",
                html_report,
                "analytics_report.html",
                "text/html",
                width="stretch"
            )
        
        with col3:
            st.markdown("#### 📊 Insights CSV")
            csv_report = report_gen.generate_csv_summary()
            st.download_button(
                "📥 Download CSV",
                csv_report,
                "insights.csv",
                "text/csv",
                width="stretch"
            )
        
        st.markdown("---")
        
        # Data Export
        st.markdown("#### 💾 Data Export")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Full Dataset (CSV)",
                dataframe_to_csv(df, fingerprint),
                "data_export.csv",
                "text/csv",
                width="stretch"
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
                width="stretch"
            )

    # =========================================================================
    # TAB 7: DASHBOARDS
    # =========================================================================
    with tab7:
        st.markdown("### 🗂️ Executive Dashboards")
        st.caption("Curate AI-generated cards for stakeholder-ready dashboards.")

        # Cards live in the browser session; a file keeps them across sessions and machines
        save_col, restore_col = st.columns(2)
        with save_col:
            if st.session_state.saved_dashboards:
                st.download_button(
                    "💾 Download dashboard (.json)",
                    export_dashboards(st.session_state.saved_dashboards),
                    f"dashboard_{datetime.now():%Y%m%d_%H%M}.json",
                    "application/json",
                    width="stretch",
                    help="Save all cards to a file you can restore later.",
                )
        with restore_col:
            restore_file = st.file_uploader("Restore a saved dashboard", type=["json"],
                                            key="dashboard_restore")
            if restore_file is not None:
                content = restore_file.getvalue()
                file_id = hashlib.sha256(content).hexdigest()
                # The uploader keeps the file across reruns; import each file only once
                if st.session_state.get("restored_dashboard_file") != file_id:
                    try:
                        restored = import_dashboards(content.decode("utf-8"))
                        st.session_state.saved_dashboards.extend(restored)
                        st.session_state.restored_dashboard_file = file_id
                        st.success(f"✅ Restored {len(restored)} card{'s' if len(restored) != 1 else ''}.")
                    except (DashboardFileError, UnicodeDecodeError) as e:
                        st.error(f"❌ Couldn't restore this file: {e}")

        if not st.session_state.saved_dashboards:
            st.info("No dashboard cards yet. Save insights from the Ask Data tab.")
        else:
            for idx, card in enumerate(st.session_state.saved_dashboards):
                with st.expander(f"{idx + 1}. {card['title']}"):
                    st.caption(f"Query: {card['query']}")
                    if card.get("text"):
                        st.markdown(card["text"])
                    if card.get("summary"):
                        st.markdown(f"""
                        <div class="narrative-card">
                            <h4>💡 Business Insight</h4>
                            <p>{narrative_html(card['summary'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    if card.get("figure") is not None:
                        st.plotly_chart(card["figure"], width="stretch", key=f"dash_chart_{idx}")
                    if isinstance(card.get("data"), pd.DataFrame) and len(card["data"]) > 0:
                        st.dataframe(card["data"], width="stretch", hide_index=True)

                    if st.button("Remove card", key=f"remove_card_{idx}"):
                        st.session_state.saved_dashboards.pop(idx)
                        st.rerun()
    
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
