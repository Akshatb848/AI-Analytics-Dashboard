"""Forecast accuracy checks (backtests)."""
import logging

import numpy as np
import pandas as pd
import pytest

from analytics.forecasting import PredictiveEngine

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def series(values):
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "sales": values})


def test_predictable_series_scores_well_and_beats_the_naive_guess():
    days = np.arange(400)
    values = 1000 + 2 * days + 150 * np.sin(2 * np.pi * days / 7)
    result = PredictiveEngine(series(values), "date", "sales").backtest(30, yearly_seasonality=False)
    assert result["holdout_days"] == 30 and result["test_points"] == 30
    assert result["mape"] < 5
    assert result["improvement_vs_baseline"] > 50
    assert "Accuracy is good" in PredictiveEngine.rate_accuracy(result)


def test_pure_noise_is_flagged_as_no_better_than_naive():
    rng = np.random.default_rng(0)
    values = 1000 + rng.normal(0, 200, 400)
    result = PredictiveEngine(series(values), "date", "sales").backtest(30, yearly_seasonality=False)
    assert result["improvement_vs_baseline"] < 10
    verdict = PredictiveEngine.rate_accuracy(result)
    assert "No better than a naive guess" in verdict or "Accuracy is" in verdict


def test_holdout_is_capped_at_a_quarter_of_history():
    values = 500 + np.arange(80, dtype=float)
    result = PredictiveEngine(series(values), "date", "sales").backtest(90, yearly_seasonality=False)
    assert result["holdout_days"] == 79 // 4


def test_too_little_history_returns_none():
    assert PredictiveEngine(series(np.arange(20.0)), "date", "sales").backtest(7) is None


def test_zero_actuals_do_not_break_the_error_percentage():
    values = np.where(np.arange(200) % 2 == 0, 0.0, 100.0)
    result = PredictiveEngine(series(values), "date", "sales").backtest(14, yearly_seasonality=False)
    assert result["mape"] is not None and np.isfinite(result["mape"])


@pytest.mark.parametrize("gain, mape, expected", [
    (-5.0, 30.0, "No better than a naive guess"),
    (40.0, 8.0, "Accuracy is good"),
    (40.0, 18.0, "Accuracy is fair"),
    (40.0, 60.0, "Accuracy is low"),
    (40.0, None, "can't be computed"),
])
def test_rating_wording(gain, mape, expected):
    result = {"mape": mape, "improvement_vs_baseline": gain, "holdout_days": 30}
    assert expected in PredictiveEngine.rate_accuracy(result)
