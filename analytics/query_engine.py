"""Natural-language questions answered with charts and tables."""
import logging
from typing import Any, Dict, List, Optional

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
from analytics.query_parser import QueryParser, apply_filters, describe_filters, normalize_intent


logger = logging.getLogger(__name__)


class SmartQueryEngine:
    """
    Answers plain-English questions about a DataFrame.

    Questions are turned into a validated query plan, either by the built-in
    rule-based parser or by a plan supplied to `process_query` (for example
    from an LLM), and the plan is then run with pandas.
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
        self.parser = QueryParser(df, self.numeric_cols, self.categorical_cols, self.date_col)
        # Rows the current question applies to (after its filters)
        self.data = df
    
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
    
    
    def _detect_intent(self, query: str) -> Dict:
        """Parse a question into a validated query plan (see analytics.query_parser)."""
        return self.normalize(self.parser.parse(query))

    def normalize(self, raw_intent: Any) -> Dict:
        """Validate a plan from any source against this dataset."""
        return normalize_intent(raw_intent, self.df, self.numeric_cols, self.categorical_cols, self.date_col)
    
    def process_query(self, query: str, intent: Optional[Dict] = None) -> Dict:
        """Answer a question, using a plan supplied by the caller (e.g. an LLM) when given."""
        intent = self.normalize(intent) if intent is not None else self._detect_intent(query)

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
            self.data = apply_filters(self.df, intent['filters'])
            if intent['filters'] and self.data.empty:
                result['text'] = f"No rows match {describe_filters(intent['filters'])}."
                return result

            result = self._route(query, intent, result)
            if intent['unmatched'] and result['text']:
                terms = ", ".join(f"'{t}'" for t in intent['unmatched'])
                note = (f"⚠️ I couldn't match {terms} to a column, so that condition was ignored. "
                        f"Available columns: {', '.join(map(str, self.df.columns[:12]))}.")
                result['narrative'] = f"{note}\n\n{result['narrative']}".strip()
            return result

        except Exception as e:
            logger.exception("Query failed: %r", query)
            result['success'] = False
            result['text'] = f"Error processing query: {str(e)}"
            result['follow_up_suggestions'] = get_suggested_queries(self.df)
            return result

    def _route(self, query: str, intent: Dict, result: Dict) -> Dict:
        """Send a validated plan to the handler for its type."""
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
        elif intent['type'] == 'aggregate' or intent['aggregation']:
            return self._handle_aggregation_query(intent, result)
        else:
            return self._handle_default_query(query, result)

    def _filter_suffix(self, intent: Dict) -> str:
        return f" ({describe_filters(intent['filters'])})" if intent['filters'] else ""
    
    def _handle_insight_query(self, query: str, result: Dict) -> Dict:
        """Handle requests for insights."""
        engine = EnhancedInsightsEngine(self.data)
        insights, recommendations = engine.generate_all_insights()
        
        high_priority = [i for i in insights if i['priority'] == 'high'][:3]
        
        result['text'] = f"## 🔍 Key Insights from Your Data\n\nAnalyzed **{len(self.data):,}** records and found **{len(insights)}** insights."
        
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
        """Handle trend analysis queries, optionally per time grain and per group."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)

        if not metric or not self.date_col:
            result['text'] = "Cannot perform trend analysis. Need a date column and numeric metric."
            return result

        grain = intent['time_grain'] or 'D'
        grain_name = {'D': 'Daily', 'W': 'Weekly', 'MS': 'Monthly', 'QS': 'Quarterly', 'YS': 'Yearly'}[grain]
        how = 'mean' if intent['aggregation'] in ('average', 'median') else 'sum'
        group = intent['groupby']
        keys = [pd.Grouper(key=self.date_col, freq=grain)] + ([group] if group else [])
        trend_data = self.data.groupby(keys)[metric].agg(how).reset_index().sort_values(self.date_col)
        label = metric.replace("_", " ").title()

        if group:
            fig = px.line(trend_data, x=self.date_col, y=metric, color=group,
                          title=f'{grain_name} {label} by {group.replace("_", " ").title()}')
        else:
            trend_data['change_pct'] = trend_data[metric].pct_change() * 100
            window = 7 if grain == 'D' else 3
            trend_data['moving_avg'] = trend_data[metric].rolling(window=window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data[self.date_col], y=trend_data[metric],
                mode='lines' if grain == 'D' else 'lines+markers', name=grain_name,
                line=dict(color='#6366f1', width=1 if grain == 'D' else 2), opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                x=trend_data[self.date_col], y=trend_data['moving_avg'],
                mode='lines', name=f'{window}-period average', line=dict(color='#10b981', width=2)
            ))
            fig.update_layout(title=f'{grain_name} {label}')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified')

        # Growth: first vs last week of days, or first vs last complete period.
        # Periods the data only partly covers (e.g. a month starting mid-way) are skipped.
        totals = trend_data.groupby(self.date_col)[metric].sum() if group else trend_data.set_index(self.date_col)[metric]
        if grain != 'D' and len(totals) > 2:
            dates = self.data[self.date_col]
            if dates.min() > totals.index[0]:
                totals = totals.iloc[1:]
            next_start = totals.index[-1] + pd.tseries.frequencies.to_offset(grain)
            if dates.max() < next_start - pd.Timedelta(days=1):
                totals = totals.iloc[:-1]
        span = 7 if grain == 'D' else 1
        period = {'D': 'week', 'W': 'complete week', 'MS': 'complete month',
                  'QS': 'complete quarter', 'YS': 'complete year'}[grain]

        result['figure'] = fig
        result['data'] = trend_data
        by_text = f" by {group.replace('_', ' ').title()}" if group else ""
        result['text'] = f"## 📈 {grain_name} Trend: {label}{by_text}{self._filter_suffix(intent)}"

        if len(totals) < 2 * span:
            unit = 'day' if grain == 'D' else period
            result['narrative'] = (
                f"The data has only {len(totals)} {unit}{'s' if len(totals) != 1 else ''} of "
                f"{label.lower()}, not enough to measure growth. Try a finer time grain."
            )
        else:
            first_val, last_val = totals.iloc[:span].mean(), totals.iloc[-span:].mean()
            growth = ((last_val - first_val) / abs(first_val) * 100) if first_val else 0
            result['narrative'] = (
                f"**{label}** shows an overall {'increase' if growth > 0 else 'decrease'} of "
                f"**{abs(growth):.1f}%** from the first to the last {period}. "
            )
            if grain != 'D' and len(totals) >= 3:
                result['narrative'] += (
                    f"Average change from one {period.replace('complete ', '')} to the next: "
                    f"**{totals.pct_change().dropna().mean() * 100:+.1f}%**. "
                )
            if abs(growth) > 20:
                result['narrative'] += f"This is a significant {'upward' if growth > 0 else 'downward'} trend that warrants attention."

        result['follow_up_suggestions'] = [
            f"Forecast {metric} for next 30 days",
            f"Show {metric} by {self.categorical_cols[0]}" if self.categorical_cols else f"Distribution of {metric}",
            f"Monthly {metric} trend by {self.categorical_cols[0]}" if self.categorical_cols else f"Weekly {metric} trend",
        ]

        return result
    
    def _handle_correlation_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle correlation analysis between the two columns asked about."""
        if len(self.numeric_cols) < 2:
            result['text'] = "Need at least 2 numeric columns for correlation analysis."
            return result

        col1 = intent.get('metric') or self.numeric_cols[0]
        col2 = intent.get('metric2') or next(c for c in self.numeric_cols if c != col1)
        correlation = self.data[col1].corr(self.data[col2])

        fig = px.scatter(
            self.data, x=col1, y=col2,
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
        result['text'] = f"## 🔗 Correlation Analysis{self._filter_suffix(intent)}"

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
            x=self.data[metric], nbinsx=50,
            marker=dict(color='#6366f1', line=dict(color='#818cf8', width=1))
        ))
        
        mean_val = self.data[metric].mean()
        median_val = self.data[metric].median()
        
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
        
        skewness = self.data[metric].skew()
        result['narrative'] = f"The distribution of **{metric.replace('_', ' ')}** ranges from {self.data[metric].min():,.2f} to {self.data[metric].max():,.2f}. "
        result['narrative'] += f"Mean: {mean_val:,.2f}, Median: {median_val:,.2f}. "
        result['narrative'] += f"The distribution is {'right-skewed' if skewness > 0.5 else 'left-skewed' if skewness < -0.5 else 'approximately normal'} (skewness: {skewness:.2f})."
        
        return result
    
    def _handle_anomaly_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle anomaly detection queries."""
        metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
        
        if not metric:
            result['text'] = "No numeric column found for anomaly detection."
            return result
        
        Q1 = self.data[metric].quantile(0.25)
        Q3 = self.data[metric].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        anomalies = self.data[(self.data[metric] < lower) | (self.data[metric] > upper)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data.index, y=self.data[metric],
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
        result['narrative'] = f"Found **{len(anomalies):,}** anomalies ({len(anomalies)/len(self.data)*100:.1f}% of data) in **{metric.replace('_', ' ')}**. "
        result['narrative'] += f"Values outside the range [{lower:,.2f}, {upper:,.2f}] are flagged as outliers."
        
        return result
    
    def _handle_forecast_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle forecast queries."""
        result['text'] = "## 📈 Forecasting"
        result['narrative'] = "For detailed forecasting with Prophet, please use the **Predictions** tab. It provides interactive controls for forecast horizon and seasonality settings."
        result['follow_up_suggestions'] = ["Navigate to Predictions tab for forecasting"]
        return result
    
    def _handle_aggregation_query(self, intent: Dict, result: Dict) -> Dict:
        """Handle aggregations: totals, averages, counts, top/bottom N, one or two groupings."""
        agg_type = intent['aggregation'] or 'sum'
        groups = [g for g in (intent['groupby'], intent['groupby2']) if g]
        if not groups and intent['limit'] and self.categorical_cols:
            groups = [self.categorical_cols[0]]  # "top 5 by sales" needs something to rank
        suffix = self._filter_suffix(intent)
        data_rows = self.data

        if agg_type == 'count':
            metric, value_col = None, 'count'
            if not groups:
                result['text'] = f"## 📊 Count{suffix}: **{len(data_rows):,}** records"
                result['data'] = len(data_rows)
                return result
            data = data_rows.groupby(groups).size().reset_index(name='count')
            title = f"Count by {' and '.join(g.replace('_', ' ').title() for g in groups)}"
        else:
            metric = intent['metric'] or (self.numeric_cols[0] if self.numeric_cols else None)
            if not metric:
                result['text'] = "Please specify a numeric column for aggregation."
                return result
            value_col = metric
            func = {'sum': 'sum', 'average': 'mean', 'max': 'max', 'min': 'min', 'median': 'median'}[agg_type]
            label = f"{agg_type.title()} of {metric.replace('_', ' ').title()}"
            if not groups:
                value = data_rows[metric].agg(func)
                result['text'] = f"## 📊 {label}{suffix}: **{format_number(value)}**"
                result['data'] = value
                return result
            data = data_rows.groupby(groups)[metric].agg(func).reset_index()
            title = f"{label} by {' and '.join(g.replace('_', ' ').title() for g in groups)}"

        data = data.sort_values(value_col, ascending=(intent['sort_order'] == 'asc'))
        if intent['limit']:
            if len(groups) > 1:
                keep = (data.groupby(groups[0])[value_col].sum()
                        .sort_values(ascending=(intent['sort_order'] == 'asc'))
                        .head(intent['limit']).index)
                data = data[data[groups[0]].isin(keep)]
            else:
                data = data.head(intent['limit'])
            ranking = 'Top' if intent['sort_order'] == 'desc' else 'Bottom'
            title = f"{ranking} {intent['limit']}: {title}"

        if len(groups) > 1:
            fig = px.bar(data, x=groups[0], y=value_col, color=groups[1], barmode='group', title=title)
        else:
            fig = px.bar(data, x=groups[0], y=value_col, color=value_col,
                         color_continuous_scale='Viridis', title=title)
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        result['figure'] = fig
        result['data'] = data
        result['text'] = f"## 📊 {title}{suffix}"

        if len(groups) == 1 and len(data) > 0:
            ordered = data.sort_values(value_col, ascending=False)
            top_row, bottom_row = ordered.iloc[0], ordered.iloc[-1]
            what = 'records' if agg_type == 'count' else metric.replace('_', ' ')
            result['narrative'] = f"**{top_row[groups[0]]}** leads with {format_number(top_row[value_col])} {what}"
            if len(ordered) > 1:
                result['narrative'] += f", while **{bottom_row[groups[0]]}** has the lowest at {format_number(bottom_row[value_col])}."
            else:
                result['narrative'] += "."

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
