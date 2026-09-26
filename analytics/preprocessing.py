"""Data cleaning operations used by the Data Tools tab."""
from typing import Any, Dict, List

import pandas as pd

from analytics.data_utils import (
    _is_numeric_dtype,
    detect_date_column,
    detect_numeric_columns,
)


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
            
            if _is_numeric_dtype(self.df[col]):
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
                    if _is_numeric_dtype(self.df[col]):
                        if abs(self.df[col].skew()) > 1:
                            self.df[col] = self.df[col].fillna(self.df[col].median())
                            method = 'median'
                        else:
                            self.df[col] = self.df[col].fillna(self.df[col].mean())
                            method = 'mean'
                    else:
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col] = self.df[col].fillna(mode_val[0])
                            method = 'mode'
                        else:
                            self.df[col] = self.df[col].fillna('Unknown')
                            method = 'constant'
                elif strategy in ('mean', 'median') and not _is_numeric_dtype(self.df[col]):
                    continue  # mean/median only apply to numeric columns
                elif strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    method = 'mean'
                elif strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    method = 'median'
                elif strategy == 'mode':
                    mode_val = self.df[col].mode()
                    if len(mode_val) == 0:
                        continue
                    self.df[col] = self.df[col].fillna(mode_val[0])
                    method = 'mode'
                elif strategy == 'drop':
                    self.df.dropna(subset=[col], inplace=True)
                    method = 'drop rows'
                elif strategy == 'zero':
                    self.df[col] = self.df[col].fillna(0)
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
            if col not in self.df.columns or not _is_numeric_dtype(self.df[col]):
                continue
                
            # Rows with missing values are kept; only measured outliers are dropped
            values = self.df[col]
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                keep = values.between(Q1 - threshold * IQR, Q3 + threshold * IQR)
            elif method == 'zscore':
                std = values.std()
                if not std or pd.isna(std):
                    continue
                keep = ((values - values.mean()) / std).abs() < threshold
            else:
                continue
            self.df = self.df[keep | values.isna()]
        
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
