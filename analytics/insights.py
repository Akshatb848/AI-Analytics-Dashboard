"""Automated insights with plain-language narratives."""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analytics.data_utils import (
    detect_categorical_columns,
    detect_date_column,
    detect_numeric_columns,
)


logger = logging.getLogger(__name__)


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
            except Exception:
                logger.debug("%s skipped a column", "_identify_potential_causes", exc_info=True)
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
                            'description': 'Comparing recent 20% of data vs earliest 20%',
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
                                'description': 'Comparing last 7 days vs previous 7 days',
                                'narrative': f'Short-term momentum shows {col.replace("_", " ")} {"accelerating" if wow_change > 0 else "decelerating"} with a {abs(wow_change):.1f}% {"gain" if wow_change > 0 else "decline"} week-over-week.',
                                'priority': 'medium',
                                'metric': col,
                                'value': wow_change
                            })
            except Exception:
                logger.debug("%s skipped a column", "_analyze_trends", exc_info=True)
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
        except Exception:
            logger.debug("%s skipped a column", "_analyze_correlations", exc_info=True)
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
            except Exception:
                logger.debug("%s skipped a column", "_detect_anomalies", exc_info=True)
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
                except Exception:
                    logger.debug("%s skipped a column", "_analyze_categorical_performance", exc_info=True)
                    continue
    
    def _analyze_distributions(self):
        """Analyze data distributions."""
        for col in self.numeric_cols[:3]:
            try:
                skewness = self.df[col].skew()
                
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
            except Exception:
                logger.debug("%s skipped a column", "_analyze_distributions", exc_info=True)
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
                except Exception:
                    logger.debug("%s skipped a column", "_perform_statistical_tests", exc_info=True)
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
            except Exception:
                logger.debug("%s skipped a column", "_detect_seasonality", exc_info=True)
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
                    'description': "The strong correlation identified can be used to build predictive models. When one variable changes, you can anticipate changes in the correlated variable.",
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
