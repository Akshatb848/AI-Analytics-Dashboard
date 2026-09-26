"""Time series forecasting with Prophet."""
from typing import Any, Dict, Optional

import numpy as np

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
    
    @staticmethod
    def _new_model(yearly_seasonality: bool, weekly_seasonality: bool, daily_seasonality: bool) -> Prophet:
        return Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )

    def train_model(self, yearly_seasonality: bool = True, 
                   weekly_seasonality: bool = True,
                   daily_seasonality: bool = False) -> None:
        prophet_df = self.prepare_data()
        self.model = self._new_model(yearly_seasonality, weekly_seasonality, daily_seasonality)
        self.model.fit(prophet_df)

    def backtest(self, horizon_days: int = 30, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True) -> Optional[Dict[str, Any]]:
        """Hold back the most recent stretch of history, forecast it, and score the forecast.

        The held-back window is the forecast horizon, capped at a quarter of the history.
        Returns None when there is too little data to test meaningfully.
        """
        data = self.prepare_data()
        if len(data) < 40:
            return None
        span_days = (data['ds'].max() - data['ds'].min()).days
        holdout_days = max(7, min(horizon_days, span_days // 4))
        cutoff = data['ds'].max() - pd.Timedelta(days=holdout_days)
        train, test = data[data['ds'] <= cutoff], data[data['ds'] > cutoff]
        if len(train) < 30 or len(test) < 5:
            return None

        model = self._new_model(yearly_seasonality, weekly_seasonality, False)
        model.fit(train)
        predicted = model.predict(test[['ds']])

        actual = test['y'].to_numpy()
        yhat = predicted['yhat'].to_numpy()
        errors = np.abs(actual - yhat)
        nonzero = actual != 0
        mape = float(np.mean(errors[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else None
        coverage = float(np.mean((actual >= predicted['yhat_lower'].to_numpy()) &
                                 (actual <= predicted['yhat_upper'].to_numpy())) * 100)
        # Naive baseline: predict the average of the last `holdout_days` of training data
        baseline = train[train['ds'] > cutoff - pd.Timedelta(days=holdout_days)]['y'].mean()
        baseline_mae = float(np.mean(np.abs(actual - baseline)))
        mae = float(errors.mean())

        return {
            'holdout_days': int(holdout_days),
            'train_points': int(len(train)),
            'test_points': int(len(test)),
            'mae': mae,
            'mape': mape,
            'interval_coverage': coverage,
            'baseline_mae': baseline_mae,
            'improvement_vs_baseline': float((1 - mae / baseline_mae) * 100) if baseline_mae else None,
            'comparison': pd.DataFrame({
                'Date': test['ds'].to_numpy(), 'Actual': actual, 'Forecast': yhat,
                'Lower': predicted['yhat_lower'].to_numpy(), 'Upper': predicted['yhat_upper'].to_numpy(),
            }),
        }

    @staticmethod
    def rate_accuracy(result: Dict[str, Any]) -> str:
        """Plain-language verdict on a backtest."""
        mape, gain = result['mape'], result['improvement_vs_baseline']
        if gain is not None and gain <= 0:
            return ("No better than a naive guess (the recent average) on held-back data. "
                    "Treat this forecast as rough.")
        if mape is None:
            return "Accuracy percentage can't be computed because the actual values include zeros."
        level = "good" if mape < 10 else "fair" if mape < 25 else "low"
        return (f"Accuracy is {level}: typical error {mape:.1f}% on the last {result['holdout_days']} days, "
                f"{gain:.0f}% better than a naive guess.")

    @staticmethod
    def plot_backtest(result: Dict[str, Any]) -> go.Figure:
        data = result['comparison']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([data['Date'], data['Date'][::-1]]),
            y=pd.concat([data['Upper'], data['Lower'][::-1]]),
            fill='toself', fillcolor='rgba(16, 185, 129, 0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='95% interval'
        ))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Actual'], mode='lines+markers',
                                 name='Actual', line=dict(color='#6366f1')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Forecast'], mode='lines',
                                 name='Forecast (trained without these days)', line=dict(color='#10b981')))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', hovermode='x unified', height=400,
                          title='Backtest: forecast vs. what actually happened')
        return fig
    
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
