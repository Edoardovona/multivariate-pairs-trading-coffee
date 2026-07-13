"""Unit tests for z-score signals and threshold calibration."""

import numpy as np
import pandas as pd
import pytest

from src.signals import compute_time_to_mean, compute_zscore, grid_search_thresholds


@pytest.fixture
def mean_reverting_spread() -> pd.Series:
    """Synthetic Ornstein-Uhlenbeck-like spread with a fixed seed."""
    rng = np.random.default_rng(42)
    n = 500
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.05)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(spread, index=index)


def test_zscore_is_standardized(mean_reverting_spread):
    z = compute_zscore(mean_reverting_spread, window=22).dropna()
    # Rolling standardization keeps the series centered with unit-order scale
    assert abs(z.mean()) < 0.5
    assert 0.5 < z.std() < 2.0


def test_zscore_leading_window_is_nan(mean_reverting_spread):
    z = compute_zscore(mean_reverting_spread, window=22)
    assert z.iloc[:21].isna().all()
    assert z.iloc[22:].notna().all()


def test_zscore_of_constant_series_is_nan():
    constant = pd.Series(np.ones(100))
    z = compute_zscore(constant, window=22)
    # Zero rolling std -> division yields NaN, never a false signal
    assert z.iloc[22:].isna().all()


def test_grid_search_respects_open_greater_than_close(mean_reverting_spread):
    log_prices = pd.DataFrame(
        {"A": mean_reverting_spread + 5.0, "KC1": pd.Series(5.0, index=mean_reverting_spread.index)}
    )
    results, heatmap = grid_search_thresholds(
        log_prices, ["A"], "KC1",
        open_grid=np.array([1.0, 1.5]), close_grid=np.array([0.5, 1.0, 1.5]),
    )
    assert (results["close"] < results["open"]).all()
    assert heatmap.shape == (2, 3)


def test_time_to_mean_positive(mean_reverting_spread):
    log_prices = pd.DataFrame(
        {"A": mean_reverting_spread + 5.0, "KC1": pd.Series(5.0, index=mean_reverting_spread.index)}
    )
    ttm = compute_time_to_mean(log_prices, ["A"], "KC1", 1.0, 0.5)
    assert ttm["A"] > 0