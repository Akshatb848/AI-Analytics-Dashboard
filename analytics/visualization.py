"""Plotly chart builders for the Overview tab."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics.data_utils import (
    detect_categorical_columns,
    detect_date_column,
    detect_numeric_columns,
)


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
    
    def __init__(
        self,
        df: pd.DataFrame,
        color_palette: str = 'viridis',
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        date_col: Optional[str] = None,
        metric_definitions: Optional[List[Any]] = None
    ):
        self.df = df
        self.numeric_cols = numeric_cols or detect_numeric_columns(df)
        self.categorical_cols = categorical_cols or detect_categorical_columns(df)
        self.date_col = date_col or detect_date_column(df)
        self.color_palette = color_palette
        self.metric_definitions = metric_definitions or []
    
    def create_overview_metrics(self) -> List[Dict[str, Any]]:
        metrics = []
        if self.metric_definitions:
            cols_to_use = [m.column for m in self.metric_definitions if m.column in self.numeric_cols]
        else:
            priority_cols = ['sales', 'revenue', 'profit', 'quantity', 'amount']
            cols_to_use = [c for c in priority_cols if c in self.numeric_cols]
            cols_to_use.extend([c for c in self.numeric_cols if c not in cols_to_use])
        
        for col in cols_to_use[:4]:
            definition = next((m for m in self.metric_definitions if m.column == col), None)
            agg = definition.aggregation if definition else 'sum'
            is_cumulative = agg in ['sum', 'total']
            current = self.df[col].sum() if is_cumulative else self.df[col].mean()
            
            half = len(self.df) // 2
            prev = self.df[col].iloc[:half].sum() if is_cumulative else self.df[col].iloc[:half].mean()
            delta = ((current - prev) / abs(prev)) * 100 if prev != 0 else 0
            
            metrics.append({
                'label': definition.name if definition else col.replace('_', ' ').title(),
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
