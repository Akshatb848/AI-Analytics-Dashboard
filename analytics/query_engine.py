"""Natural-language questions answered with charts and tables."""
import logging
import re
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.data_utils import (
    detect_categorical_columns,
    detect_date_column,
    detect_numeric_columns,
    get_suggested_queries,
)
from analytics.formatting import format_number
from analytics.insights import EnhancedInsightsEngine


logger = logging.getLogger(__name__)


class SmartQueryEngine:
    """
    Intelligent query processing with semantic understanding.
    Architecture ready for LLM integration (OpenAI/Anthropic API).
    Currently uses advanced pattern matching with fallback to smart defaults.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        date_col: Optional[str] = None
    ):
        self.df = df
        self.numeric_cols = numeric_cols or detect_numeric_columns(df)
        self.categorical_cols = categorical_cols or detect_categorical_columns(df)
        self.date_col = date_col or detect_date_column(df)
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
            logger.exception("Query failed: %r", query)
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
        result['text'] = "## 🔗 Correlation Analysis"
        
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
        result['narrative'] += "\n\n**Available columns:**\n"
        result['narrative'] += f"- Numeric: {', '.join(self.numeric_cols[:5])}\n"
        result['narrative'] += f"- Categorical: {', '.join(self.categorical_cols[:5])}\n"
        
        if self.date_col:
            result['narrative'] += f"- Date: {self.date_col}"
        
        return result
