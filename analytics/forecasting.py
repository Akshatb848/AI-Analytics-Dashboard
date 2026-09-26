"""Time series forecasting with Prophet."""
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet


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
