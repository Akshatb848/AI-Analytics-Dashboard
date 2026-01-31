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
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #10b981;
        --accent: #f59e0b;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .metric-delta-positive {
        color: #10b981;
        font-size: 0.85rem;
    }
    
    .metric-delta-negative {
        color: #ef4444;
        font-size: 0.85rem;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        border-color: rgba(99, 102, 241, 0.6);
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
    }
    
    /* Query input */
    .query-container {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .stTextInput > div > div > input {
        background-color: #334155 !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #818cf8, #6366f1) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #f1f5f9 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #10b981) !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def generate_sample_data(rows: int = 1000) -> pd.DataFrame:
    """Generate sample sales data for demonstration."""
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
    sales = np.maximum(sales, 100)  # Ensure no negative sales
    
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


# =============================================================================
# AUTOMATED INSIGHTS ENGINE
# =============================================================================

class InsightsEngine:
    """Generate automated insights from data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
        self.date_col = detect_date_column(df)
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
    
    def generate_all_insights(self) -> List[Dict[str, Any]]:
        """Generate all types of insights."""
        self.insights = []
        
        # Statistical insights
        self._generate_statistical_insights()
        
        # Trend insights
        if self.date_col:
            self._generate_trend_insights()
        
        # Distribution insights
        self._generate_distribution_insights()
        
        # Correlation insights
        self._generate_correlation_insights()
        
        # Categorical insights
        self._generate_categorical_insights()
        
        # Anomaly detection
        self._detect_anomalies()
        
        return self.insights
    
    def _generate_statistical_insights(self):
        """Generate basic statistical insights."""
        for col in self.numeric_cols[:3]:  # Limit to first 3 numeric columns
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0
            
            if cv > 50:
                self.insights.append({
                    'type': 'variability',
                    'icon': '📊',
                    'title': f'High Variability in {col}',
                    'description': f'{col} shows high variability (CV: {cv:.1f}%). Consider investigating the factors causing this variation.',
                    'priority': 'medium'
                })
    
    def _generate_trend_insights(self):
        """Generate time-based trend insights."""
        if not self.date_col or not self.numeric_cols:
            return
            
        for col in self.numeric_cols[:2]:
            try:
                df_sorted = self.df.sort_values(self.date_col)
                recent = df_sorted[col].tail(30).mean()
                previous = df_sorted[col].head(30).mean()
                
                if previous > 0:
                    change_pct = ((recent - previous) / previous) * 100
                    
                    if abs(change_pct) > 10:
                        direction = "increased" if change_pct > 0 else "decreased"
                        icon = "📈" if change_pct > 0 else "📉"
                        
                        self.insights.append({
                            'type': 'trend',
                            'icon': icon,
                            'title': f'{col} has {direction}',
                            'description': f'{col} has {direction} by {abs(change_pct):.1f}% comparing recent vs earlier periods.',
                            'priority': 'high' if abs(change_pct) > 25 else 'medium'
                        })
            except Exception:
                continue
    
    def _generate_distribution_insights(self):
        """Analyze data distributions."""
        for col in self.numeric_cols[:2]:
            skewness = self.df[col].skew()
            
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                self.insights.append({
                    'type': 'distribution',
                    'icon': '📊',
                    'title': f'{col} is skewed {direction}',
                    'description': f'The distribution of {col} is significantly skewed {direction} (skewness: {skewness:.2f}). This may indicate outliers or a non-normal distribution.',
                    'priority': 'low'
                })
    
    def _generate_correlation_insights(self):
        """Find significant correlations."""
        if len(self.numeric_cols) < 2:
            return
            
        try:
            corr_matrix = self.df[self.numeric_cols].corr()
            
            for i, col1 in enumerate(self.numeric_cols):
                for col2 in self.numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_val) > 0.7:
                        relationship = "positive" if corr_val > 0 else "negative"
                        self.insights.append({
                            'type': 'correlation',
                            'icon': '🔗',
                            'title': f'Strong {relationship} correlation found',
                            'description': f'{col1} and {col2} show a strong {relationship} correlation ({corr_val:.2f}). Changes in one may predict changes in the other.',
                            'priority': 'high'
                        })
        except Exception:
            pass
    
    def _generate_categorical_insights(self):
        """Analyze categorical columns."""
        for col in self.categorical_cols[:2]:
            value_counts = self.df[col].value_counts()
            
            if len(value_counts) > 0:
                top_category = value_counts.index[0]
                top_pct = (value_counts.iloc[0] / len(self.df)) * 100
                
                if top_pct > 40:
                    self.insights.append({
                        'type': 'concentration',
                        'icon': '🎯',
                        'title': f'{col} is concentrated',
                        'description': f'"{top_category}" represents {top_pct:.1f}% of all {col} values. This concentration may indicate market dominance or data bias.',
                        'priority': 'medium'
                    })
    
    def _detect_anomalies(self):
        """Detect anomalies using IQR method."""
        for col in self.numeric_cols[:2]:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            anomaly_pct = (len(anomalies) / len(self.df)) * 100
            
            if anomaly_pct > 5:
                self.insights.append({
                    'type': 'anomaly',
                    'icon': '⚠️',
                    'title': f'Anomalies detected in {col}',
                    'description': f'{anomaly_pct:.1f}% of {col} values are outliers. These may represent data quality issues or exceptional cases worth investigating.',
                    'priority': 'high'
                })


# =============================================================================
# PREDICTIVE ANALYTICS ENGINE
# =============================================================================

class PredictiveEngine:
    """Time series forecasting using Prophet."""
    
    def __init__(self, df: pd.DataFrame, date_col: str, target_col: str):
        self.df = df
        self.date_col = date_col
        self.target_col = target_col
        self.model = None
        self.forecast = None
    
    def prepare_data(self) -> pd.DataFrame:
        """Prepare data for Prophet."""
        prophet_df = self.df[[self.date_col, self.target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        return prophet_df.sort_values('ds')
    
    def train_model(self, yearly_seasonality: bool = True, 
                   weekly_seasonality: bool = True,
                   daily_seasonality: bool = False) -> None:
        """Train Prophet model."""
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
        """Generate forecast."""
        if self.model is None:
            self.train_model()
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the forecast."""
        if self.forecast is None:
            return {}
        
        future_forecast = self.forecast[self.forecast['ds'] > self.df[self.date_col].max()]
        
        return {
            'predicted_mean': future_forecast['yhat'].mean(),
            'predicted_min': future_forecast['yhat_lower'].min(),
            'predicted_max': future_forecast['yhat_upper'].max(),
            'trend_direction': 'increasing' if future_forecast['trend'].iloc[-1] > future_forecast['trend'].iloc[0] else 'decreasing',
            'confidence_interval': (future_forecast['yhat_upper'].mean() - future_forecast['yhat_lower'].mean()) / 2
        }
    
    def plot_forecast(self) -> go.Figure:
        """Create interactive forecast plot."""
        if self.forecast is None:
            self.make_forecast()
        
        prophet_df = self.prepare_data()
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='markers',
            name='Historical',
            marker=dict(color='#6366f1', size=4, opacity=0.6)
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=self.forecast['ds'],
            y=self.forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#10b981', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([self.forecast['ds'], self.forecast['ds'][::-1]]),
            y=pd.concat([self.forecast['yhat_upper'], self.forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(
                text=f'{self.target_col} Forecast',
                font=dict(size=20, color='#f1f5f9')
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(99, 102, 241, 0.1)',
                title='Date'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(99, 102, 241, 0.1)',
                title=self.target_col
            ),
            legend=dict(
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(99, 102, 241, 0.3)'
            ),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_components(self) -> go.Figure:
        """Plot forecast components."""
        if self.forecast is None:
            self.make_forecast()
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Trend', 'Weekly Seasonality', 'Yearly Seasonality'),
            vertical_spacing=0.12
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=self.forecast['ds'],
                y=self.forecast['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='#6366f1', width=2)
            ),
            row=1, col=1
        )
        
        # Weekly seasonality
        if 'weekly' in self.forecast.columns:
            weekly = self.forecast[['ds', 'weekly']].drop_duplicates()
            fig.add_trace(
                go.Scatter(
                    x=weekly['ds'],
                    y=weekly['weekly'],
                    mode='lines',
                    name='Weekly',
                    line=dict(color='#10b981', width=2)
                ),
                row=2, col=1
            )
        
        # Yearly seasonality
        if 'yearly' in self.forecast.columns:
            yearly = self.forecast[['ds', 'yearly']].drop_duplicates()
            fig.add_trace(
                go.Scatter(
                    x=yearly['ds'],
                    y=yearly['yearly'],
                    mode='lines',
                    name='Yearly',
                    line=dict(color='#f59e0b', width=2)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            showlegend=False
        )
        
        for i in range(1, 4):
            fig.update_xaxes(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)', row=i, col=1)
        
        return fig


# =============================================================================
# NATURAL LANGUAGE QUERY ENGINE
# =============================================================================

class NLQueryEngine:
    """Process natural language queries on data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
        self.date_col = detect_date_column(df)
        
        # Query patterns
        self.patterns = {
            'sum': r'(?:total|sum|aggregate)\s+(?:of\s+)?(\w+)',
            'average': r'(?:average|mean|avg)\s+(?:of\s+)?(\w+)',
            'max': r'(?:maximum|max|highest|largest)\s+(\w+)',
            'min': r'(?:minimum|min|lowest|smallest)\s+(\w+)',
            'count': r'(?:count|number|how many)\s+(?:of\s+)?(\w+)',
            'group_by': r'(?:by|per|for each|grouped by)\s+(\w+)',
            'filter': r'(?:where|when|if|for)\s+(\w+)\s*(?:is|=|equals?)\s*["\']?(\w+)["\']?',
            'top': r'(?:top|first)\s+(\d+)',
            'bottom': r'(?:bottom|last)\s+(\d+)',
            'trend': r'(?:trend|over time|time series)\s+(?:of\s+)?(\w+)',
            'compare': r'compare\s+(\w+)\s+(?:and|vs|versus)\s+(\w+)',
            'correlation': r'correlation\s+(?:between\s+)?(\w+)\s+(?:and|with)\s+(\w+)',
        }
    
    def _find_column(self, term: str) -> Optional[str]:
        """Find the closest matching column name."""
        term_lower = term.lower()
        
        # Exact match
        for col in self.df.columns:
            if col.lower() == term_lower:
                return col
        
        # Partial match
        for col in self.df.columns:
            if term_lower in col.lower() or col.lower() in term_lower:
                return col
        
        return None
    
    def process_query(self, query: str) -> Tuple[str, Optional[Any], Optional[go.Figure]]:
        """Process a natural language query and return results."""
        query_lower = query.lower()
        result_text = ""
        result_data = None
        result_fig = None
        
        try:
            # Detect aggregation type
            agg_type = None
            agg_col = None
            group_col = None
            filter_col = None
            filter_val = None
            limit = None
            
            # Check for sum
            sum_match = re.search(self.patterns['sum'], query_lower)
            if sum_match:
                agg_type = 'sum'
                agg_col = self._find_column(sum_match.group(1))
            
            # Check for average
            avg_match = re.search(self.patterns['average'], query_lower)
            if avg_match:
                agg_type = 'mean'
                agg_col = self._find_column(avg_match.group(1))
            
            # Check for max
            max_match = re.search(self.patterns['max'], query_lower)
            if max_match:
                agg_type = 'max'
                agg_col = self._find_column(max_match.group(1))
            
            # Check for min
            min_match = re.search(self.patterns['min'], query_lower)
            if min_match:
                agg_type = 'min'
                agg_col = self._find_column(min_match.group(1))
            
            # Check for count
            count_match = re.search(self.patterns['count'], query_lower)
            if count_match:
                agg_type = 'count'
                agg_col = self._find_column(count_match.group(1))
            
            # Check for group by
            group_match = re.search(self.patterns['group_by'], query_lower)
            if group_match:
                group_col = self._find_column(group_match.group(1))
            
            # Check for filter
            filter_match = re.search(self.patterns['filter'], query_lower)
            if filter_match:
                filter_col = self._find_column(filter_match.group(1))
                filter_val = filter_match.group(2)
            
            # Check for top/bottom
            top_match = re.search(self.patterns['top'], query_lower)
            if top_match:
                limit = int(top_match.group(1))
            
            bottom_match = re.search(self.patterns['bottom'], query_lower)
            if bottom_match:
                limit = -int(bottom_match.group(1))
            
            # Check for trend analysis
            trend_match = re.search(self.patterns['trend'], query_lower)
            if trend_match and self.date_col:
                trend_col = self._find_column(trend_match.group(1))
                if trend_col and trend_col in self.numeric_cols:
                    trend_data = self.df.groupby(self.date_col)[trend_col].sum().reset_index()
                    
                    result_fig = px.line(
                        trend_data, x=self.date_col, y=trend_col,
                        title=f'{trend_col} Over Time'
                    )
                    result_fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    result_text = f"Showing trend of {trend_col} over time"
                    result_data = trend_data
                    return result_text, result_data, result_fig
            
            # Check for correlation
            corr_match = re.search(self.patterns['correlation'], query_lower)
            if corr_match:
                col1 = self._find_column(corr_match.group(1))
                col2 = self._find_column(corr_match.group(2))
                
                if col1 and col2 and col1 in self.numeric_cols and col2 in self.numeric_cols:
                    correlation = self.df[col1].corr(self.df[col2])
                    
                    result_fig = px.scatter(
                        self.df, x=col1, y=col2,
                        title=f'Correlation: {col1} vs {col2} (r={correlation:.3f})',
                        trendline='ols'
                    )
                    result_fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    result_text = f"Correlation between {col1} and {col2}: {correlation:.3f}"
                    return result_text, correlation, result_fig
            
            # Process aggregation query
            if agg_col or agg_type == 'count':
                df_filtered = self.df.copy()
                
                # Apply filter if specified
                if filter_col and filter_val:
                    if filter_col in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered[filter_col].astype(str).str.lower() == filter_val.lower()]
                
                # If no specific column for count, use the dataframe length
                if agg_type == 'count' and not agg_col:
                    if group_col:
                        result_data = df_filtered.groupby(group_col).size().reset_index(name='count')
                        result_text = f"Count by {group_col}"
                    else:
                        result_data = len(df_filtered)
                        result_text = f"Total count: {result_data:,}"
                        return result_text, result_data, None
                
                elif agg_col and agg_col in self.numeric_cols:
                    if group_col:
                        # Aggregation with grouping
                        if agg_type == 'sum':
                            result_data = df_filtered.groupby(group_col)[agg_col].sum().reset_index()
                        elif agg_type == 'mean':
                            result_data = df_filtered.groupby(group_col)[agg_col].mean().reset_index()
                        elif agg_type == 'max':
                            result_data = df_filtered.groupby(group_col)[agg_col].max().reset_index()
                        elif agg_type == 'min':
                            result_data = df_filtered.groupby(group_col)[agg_col].min().reset_index()
                        elif agg_type == 'count':
                            result_data = df_filtered.groupby(group_col)[agg_col].count().reset_index()
                        
                        # Apply limit
                        if limit:
                            if limit > 0:
                                result_data = result_data.nlargest(limit, agg_col)
                            else:
                                result_data = result_data.nsmallest(abs(limit), agg_col)
                        
                        result_text = f"{agg_type.capitalize()} of {agg_col} by {group_col}"
                        
                        # Create visualization
                        result_fig = px.bar(
                            result_data, x=group_col, y=agg_col,
                            title=result_text,
                            color=agg_col,
                            color_continuous_scale='Viridis'
                        )
                        result_fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                    else:
                        # Simple aggregation
                        if agg_type == 'sum':
                            result_data = df_filtered[agg_col].sum()
                        elif agg_type == 'mean':
                            result_data = df_filtered[agg_col].mean()
                        elif agg_type == 'max':
                            result_data = df_filtered[agg_col].max()
                        elif agg_type == 'min':
                            result_data = df_filtered[agg_col].min()
                        elif agg_type == 'count':
                            result_data = df_filtered[agg_col].count()
                        
                        result_text = f"{agg_type.capitalize()} of {agg_col}: {result_data:,.2f}"
                
                return result_text, result_data, result_fig
            
            # Default: show data summary
            result_text = "I understood your query but couldn't find a specific action. Here's a summary of available columns:\n"
            result_text += f"- Numeric columns: {', '.join(self.numeric_cols)}\n"
            result_text += f"- Categorical columns: {', '.join(self.categorical_cols)}\n"
            result_text += f"\nTry queries like:\n- 'Total sales by region'\n- 'Average profit by category'\n- 'Trend of sales over time'\n- 'Top 5 categories by sales'"
            
            return result_text, None, None
            
        except Exception as e:
            return f"Error processing query: {str(e)}", None, None


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================

class VisualizationBuilder:
    """Build various interactive visualizations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = detect_numeric_columns(df)
        self.categorical_cols = detect_categorical_columns(df)
        self.date_col = detect_date_column(df)
    
    def create_overview_metrics(self) -> List[Dict[str, Any]]:
        """Create overview metric cards."""
        metrics = []
        
        for col in self.numeric_cols[:4]:
            current = self.df[col].sum() if col in ['sales', 'revenue', 'profit'] else self.df[col].mean()
            
            # Calculate delta (compare to first half)
            half = len(self.df) // 2
            prev = self.df[col].iloc[:half].sum() if col in ['sales', 'revenue', 'profit'] else self.df[col].iloc[:half].mean()
            
            if prev != 0:
                delta = ((current - prev) / prev) * 100
            else:
                delta = 0
            
            metrics.append({
                'label': col.replace('_', ' ').title(),
                'value': current,
                'delta': delta,
                'format': ',.2f' if current < 1000000 else ',.0f'
            })
        
        return metrics
    
    def plot_distribution(self, column: str) -> go.Figure:
        """Create distribution plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df[column],
            nbinsx=50,
            marker=dict(
                color='#6366f1',
                line=dict(color='#818cf8', width=1)
            ),
            opacity=0.8
        ))
        
        # Add KDE line
        from scipy import stats
        kde_x = np.linspace(self.df[column].min(), self.df[column].max(), 100)
        kde = stats.gaussian_kde(self.df[column].dropna())
        kde_y = kde(kde_x) * len(self.df) * (self.df[column].max() - self.df[column].min()) / 50
        
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode='lines',
            line=dict(color='#10b981', width=2),
            name='Density'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'Distribution of {column}',
            xaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'),
            showlegend=False
        )
        
        return fig
    
    def plot_time_series(self, value_col: str, agg: str = 'sum') -> go.Figure:
        """Create time series plot."""
        if not self.date_col:
            return None
        
        if agg == 'sum':
            ts_data = self.df.groupby(self.date_col)[value_col].sum().reset_index()
        else:
            ts_data = self.df.groupby(self.date_col)[value_col].mean().reset_index()
        
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=ts_data[self.date_col],
            y=ts_data[value_col],
            mode='lines',
            line=dict(color='#6366f1', width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)',
            name=value_col
        ))
        
        # Add moving average
        window = min(7, len(ts_data) // 4)
        if window > 1:
            ts_data['ma'] = ts_data[value_col].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=ts_data[self.date_col],
                y=ts_data['ma'],
                mode='lines',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                name=f'{window}-day MA'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{value_col} Over Time',
            xaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_categorical_breakdown(self, category_col: str, value_col: str) -> go.Figure:
        """Create categorical breakdown chart."""
        breakdown = self.df.groupby(category_col)[value_col].sum().sort_values(ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=breakdown.values,
            y=breakdown.index,
            orientation='h',
            marker=dict(
                color=breakdown.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=value_col)
            )
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{value_col} by {category_col}',
            xaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(99, 102, 241, 0.1)')
        )
        
        return fig
    
    def plot_correlation_matrix(self) -> go.Figure:
        """Create correlation heatmap."""
        corr_matrix = self.df[self.numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hoverongaps=False
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title='Correlation Matrix',
            height=500
        )
        
        return fig
    
    def plot_scatter_matrix(self, columns: List[str]) -> go.Figure:
        """Create scatter matrix."""
        fig = px.scatter_matrix(
            self.df,
            dimensions=columns[:4],
            color=self.categorical_cols[0] if self.categorical_cols else None,
            opacity=0.6
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600
        )
        
        return fig
    
    def plot_pie_chart(self, category_col: str, value_col: str) -> go.Figure:
        """Create pie chart."""
        pie_data = self.df.groupby(category_col)[value_col].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=pie_data.index,
            values=pie_data.values,
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set2
            )
        )])
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=f'{value_col} Distribution by {category_col}'
        )
        
        return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>🚀 AI Analytics Dashboard</h1>
        <p>Automated insights, predictive analytics, and natural language queries powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Data Source")
        
        data_source = st.radio(
            "Choose data source",
            ["Sample Data", "Upload CSV"],
            label_visibility="collapsed"
        )
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload your CSV file",
                type=['csv'],
                help="Upload a CSV file with your data"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Try to parse date columns
                    for col in df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            try:
                                df[col] = pd.to_datetime(df[col])
                            except:
                                pass
                    st.success(f"✅ Loaded {len(df):,} rows")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    df = generate_sample_data()
            else:
                st.info("Using sample data until you upload a file")
                df = generate_sample_data()
        else:
            sample_size = st.slider("Sample size", 500, 5000, 2000, 500)
            df = generate_sample_data(sample_size)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        date_col = detect_date_column(df)
        numeric_cols = detect_numeric_columns(df)
        categorical_cols = detect_categorical_columns(df)
        
        if date_col:
            st.info(f"📅 Date column: **{date_col}**")
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        st.metric("Numeric", len(numeric_cols))
        st.metric("Categorical", len(categorical_cols))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Insights", 
        "📈 Predictions", 
        "💬 Ask Data",
        "📋 Data Explorer"
    ])
    
    # =============================================================================
    # TAB 1: OVERVIEW
    # =============================================================================
    with tab1:
        st.markdown("### Key Metrics")
        
        viz_builder = VisualizationBuilder(df)
        metrics = viz_builder.create_overview_metrics()
        
        # Metric cards
        cols = st.columns(len(metrics))
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
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            if date_col and numeric_cols:
                selected_metric = st.selectbox(
                    "Select metric for time series",
                    numeric_cols,
                    key="ts_metric"
                )
                fig = viz_builder.plot_time_series(selected_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if categorical_cols and numeric_cols:
                selected_cat = st.selectbox(
                    "Select category",
                    categorical_cols,
                    key="cat_breakdown"
                )
                selected_val = st.selectbox(
                    "Select value",
                    numeric_cols,
                    key="val_breakdown"
                )
                fig = viz_builder.plot_categorical_breakdown(selected_cat, selected_val)
                st.plotly_chart(fig, use_container_width=True)
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Distribution Analysis")
            dist_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            fig = viz_builder.plot_distribution(dist_col)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("#### Correlation Matrix")
            fig = viz_builder.plot_correlation_matrix()
            st.plotly_chart(fig, use_container_width=True)
    
    # =============================================================================
    # TAB 2: INSIGHTS
    # =============================================================================
    with tab2:
        st.markdown("### 🔍 Automated Data Insights")
        st.markdown("AI-powered analysis of your data to uncover patterns, anomalies, and opportunities.")
        
        with st.spinner("Generating insights..."):
            insights_engine = InsightsEngine(df)
            insights = insights_engine.generate_all_insights()
        
        if insights:
            # Group insights by priority
            high_priority = [i for i in insights if i['priority'] == 'high']
            medium_priority = [i for i in insights if i['priority'] == 'medium']
            low_priority = [i for i in insights if i['priority'] == 'low']
            
            if high_priority:
                st.markdown("#### 🔴 High Priority Insights")
                for insight in high_priority:
                    st.markdown(f"""
                    <div class="insight-card">
                        <span class="insight-icon">{insight['icon']}</span>
                        <span class="insight-title">{insight['title']}</span>
                        <div class="insight-description">{insight['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if medium_priority:
                st.markdown("#### 🟡 Medium Priority Insights")
                for insight in medium_priority:
                    st.markdown(f"""
                    <div class="insight-card">
                        <span class="insight-icon">{insight['icon']}</span>
                        <span class="insight-title">{insight['title']}</span>
                        <div class="insight-description">{insight['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if low_priority:
                with st.expander("📊 Additional Insights"):
                    for insight in low_priority:
                        st.markdown(f"""
                        <div class="insight-card">
                            <span class="insight-icon">{insight['icon']}</span>
                            <span class="insight-title">{insight['title']}</span>
                            <div class="insight-description">{insight['description']}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("No significant insights detected. Try uploading more data or data with more variation.")
    
    # =============================================================================
    # TAB 3: PREDICTIONS
    # =============================================================================
    with tab3:
        st.markdown("### 📈 Predictive Analytics")
        st.markdown("Forecast future values using Prophet time series analysis.")
        
        if not date_col:
            st.warning("⚠️ No date column detected. Predictive analytics requires time-series data.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_col = st.selectbox(
                    "Target variable",
                    numeric_cols,
                    key="pred_target"
                )
            
            with col2:
                forecast_periods = st.slider(
                    "Forecast periods (days)",
                    7, 90, 30,
                    key="pred_periods"
                )
            
            with col3:
                yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
                weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
            
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Training model and generating forecast..."):
                    try:
                        pred_engine = PredictiveEngine(df, date_col, target_col)
                        pred_engine.train_model(
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality
                        )
                        forecast = pred_engine.make_forecast(periods=forecast_periods)
                        
                        # Display forecast chart
                        fig = pred_engine.plot_forecast()
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast summary
                        summary = pred_engine.get_forecast_summary()
                        
                        st.markdown("#### Forecast Summary")
                        cols = st.columns(4)
                        
                        with cols[0]:
                            st.metric(
                                "Predicted Average",
                                f"{summary['predicted_mean']:,.2f}"
                            )
                        
                        with cols[1]:
                            st.metric(
                                "Predicted Range",
                                f"{summary['predicted_min']:,.0f} - {summary['predicted_max']:,.0f}"
                            )
                        
                        with cols[2]:
                            st.metric(
                                "Trend",
                                summary['trend_direction'].title()
                            )
                        
                        with cols[3]:
                            st.metric(
                                "Confidence ±",
                                f"{summary['confidence_interval']:,.2f}"
                            )
                        
                        # Components plot
                        with st.expander("📊 Forecast Components"):
                            fig_components = pred_engine.plot_components()
                            st.plotly_chart(fig_components, use_container_width=True)
                        
                        # Download forecast
                        forecast_download = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_download.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                        
                        st.download_button(
                            "📥 Download Forecast",
                            forecast_download.to_csv(index=False),
                            "forecast.csv",
                            "text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
    
    # =============================================================================
    # TAB 4: ASK DATA
    # =============================================================================
    with tab4:
        st.markdown("### 💬 Ask Your Data")
        st.markdown("Use natural language to query and analyze your data.")
        
        # Query input
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., 'What is the total sales by region?' or 'Show me the trend of profit over time'",
            label_visibility="collapsed"
        )
        
        # Example queries
        st.markdown("**Example queries:**")
        example_cols = st.columns(3)
        
        examples = [
            "Total sales by category",
            "Average profit by region",
            "Trend of sales over time",
            "Top 5 categories by sales",
            "Correlation between sales and profit",
            "Count of orders by region"
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(example, key=f"example_{i}"):
                    query = example
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if query:
            with st.spinner("Analyzing query..."):
                nl_engine = NLQueryEngine(df)
                result_text, result_data, result_fig = nl_engine.process_query(query)
                
                st.markdown("#### Result")
                st.markdown(result_text)
                
                if result_fig:
                    st.plotly_chart(result_fig, use_container_width=True)
                
                if isinstance(result_data, pd.DataFrame):
                    st.dataframe(result_data, use_container_width=True)
    
    # =============================================================================
    # TAB 5: DATA EXPLORER
    # =============================================================================
    with tab5:
        st.markdown("### 📋 Data Explorer")
        
        # Data preview
        st.markdown("#### Data Preview")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400
        )
        
        # Column statistics
        st.markdown("#### Column Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Columns**")
            st.dataframe(
                df[numeric_cols].describe().T.round(2),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Categorical Columns**")
            cat_stats = []
            for col in categorical_cols:
                cat_stats.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Null Count': df[col].isnull().sum()
                })
            st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
        
        # Download data
        st.markdown("#### Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📥 Download Full Dataset (CSV)",
                df.to_csv(index=False),
                "analytics_data.csv",
                "text/csv"
            )
        
        with col2:
            # Summary report - convert to native Python types for JSON serialization
            summary = {
                'Total Rows': int(len(df)),
                'Total Columns': int(len(df.columns)),
                'Numeric Columns': int(len(numeric_cols)),
                'Categorical Columns': int(len(categorical_cols)),
                'Date Column': date_col if date_col else 'Not detected',
                'Missing Values': int(df.isnull().sum().sum())
            }
            
            st.download_button(
                "📥 Download Summary Report (JSON)",
                json.dumps(summary, indent=2, default=str),
                "summary_report.json",
                "application/json"
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>AI Analytics Dashboard v1.0 | Built with Streamlit, Prophet, and Plotly</p>
        <p>Alternative to Tableau AI - Open Source Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
