import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ======================================================
# COLUMN PROFILE
# ======================================================

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_ratio: float
    unique_ratio: float
    is_numeric: bool
    is_datetime: bool
    variance: Optional[float]
    monotonic: bool


class DatasetProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def profile(self) -> Dict[str, ColumnProfile]:
        profiles = {}

        for col in self.df.columns:
            series = self.df[col]

            profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_ratio=series.isnull().mean(),
                unique_ratio=series.nunique() / max(len(series), 1),
                is_numeric=pd.api.types.is_numeric_dtype(series),
                is_datetime=pd.api.types.is_datetime64_any_dtype(series),
                variance=series.var() if pd.api.types.is_numeric_dtype(series) else None,
                monotonic=series.is_monotonic_increasing if pd.api.types.is_numeric_dtype(series) else False,
            )

        return profiles


# ======================================================
# SEMANTIC CLASSIFIER
# ======================================================

class SemanticModel:
    def __init__(self):
        self.time_columns: List[str] = []
        self.measures: List[str] = []
        self.dimensions: List[str] = []
        self.identifiers: List[str] = []
        self.rates: List[str] = []


class SemanticClassifier:
    def __init__(self, profiles: Dict[str, ColumnProfile]):
        self.profiles = profiles
        self.model = SemanticModel()

    def classify(self) -> SemanticModel:
        for col, p in self.profiles.items():

            name = col.lower()

            if p.is_datetime or "date" in name or "time" in name:
                self.model.time_columns.append(col)

            elif p.unique_ratio > 0.9 and not p.is_numeric:
                self.model.identifiers.append(col)

            elif "%" in name or "margin" in name or "rate" in name:
                self.model.rates.append(col)

            elif p.is_numeric and p.variance and p.variance > 0:
                self.model.measures.append(col)

            else:
                self.model.dimensions.append(col)

        return self.model


# ======================================================
# METRIC INTELLIGENCE
# ======================================================

class MetricIntelligenceEngine:
    def __init__(self, df: pd.DataFrame, semantic: SemanticModel):
        self.df = df
        self.semantic = semantic

    def discover_kpis(self) -> List[str]:
        kpis = []

        for col in self.semantic.measures:
            series = self.df[col]

            if (
                series.var() > 0
                and series.nunique() > 10
                and series.isnull().mean() < 0.5
            ):
                kpis.append(col)

        return kpis


# ======================================================
# FORECAST ELIGIBILITY ENGINE
# ======================================================

class ForecastEligibilityEngine:
    def __init__(self, df, time_col, metric):
        self.df = df
        self.time_col = time_col
        self.metric = metric

    def check(self) -> Tuple[bool, str]:
        ts = self.df[[self.time_col, self.metric]].dropna()

        if len(ts) < 15:
            return False, "Insufficient historical data"

        mean = ts[self.metric].mean()
        std = ts[self.metric].std()

        if mean == 0:
            return False, "Zero mean series"

        if std / mean > 2.5:
            return False, "Highly volatile series"

        return True, "Forecast eligible"
