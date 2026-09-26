"""Column type detection, data cleaning, quality scoring and sample data."""
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _is_text_dtype(series: pd.Series) -> bool:
    """True for object and string columns (pandas 3 stores text as StringDtype)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _is_numeric_dtype(series: pd.Series) -> bool:
    """True for numeric columns, excluding booleans."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Parse text to datetimes, returning NaT for values that don't parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", format="mixed")


TYPE_CONVERSION_THRESHOLD = 0.9


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop empty rows & columns
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    for col in df.columns:
        if not _is_text_dtype(df[col]):
            continue

        stripped = df[col].astype("string").str.strip().replace("", pd.NA)
        present = stripped.notna().sum()
        if present == 0:
            continue

        # Numeric conversion: drop thousands separators, currency and percent signs
        cleaned = (
            stripped
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("—", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() / present >= TYPE_CONVERSION_THRESHOLD:
            df[col] = numeric.astype("float64")
            continue

        # Datetime conversion
        parsed = _parse_datetime(stripped)
        if parsed.notna().sum() / present >= TYPE_CONVERSION_THRESHOLD:
            df[col] = parsed

    return df


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect date column in dataframe."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if _is_text_dtype(df[col]):
            sample = df[col].dropna().head(100)
            if len(sample) and _parse_datetime(sample).notna().mean() >= TYPE_CONVERSION_THRESHOLD:
                return col
    return None


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect numeric columns suitable for analysis."""
    return [col for col in df.columns if _is_numeric_dtype(df[col])]


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Detect categorical columns."""
    return [
        col for col in df.columns
        if _is_text_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]


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
        if _is_text_dtype(df[col]):
            try:
                pd.to_numeric(df[col], errors='raise')
                consistency -= 5  # Penalty for numeric stored as string
            except (ValueError, TypeError):
                pass  # not numeric text, which is what we want
    
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


def semantic_catalog_template() -> Dict[str, Any]:
    return {
        "time_column": "date",
        "dimensions": ["region", "category", "segment"],
        "metrics": [
            {
                "name": "Revenue",
                "column": "revenue",
                "aggregation": "sum",
                "format": "currency",
                "description": "Net revenue after discounts",
                "unit": "USD"
            },
            {
                "name": "Profit",
                "column": "profit",
                "aggregation": "sum",
                "format": "currency",
                "description": "Gross profit",
                "unit": "USD"
            }
        ],
        "hierarchies": {
            "region": ["country", "state", "city"]
        }
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
