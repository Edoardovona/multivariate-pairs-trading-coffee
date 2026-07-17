"""Unit tests for the daily simulation engine (leverage handling)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_strategy


def _tiny_problem():
    rng = np.random.default_rng(11)
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    anchor = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    log_prices = pd.DataFrame(
        {"KC1": anchor, "EQ": anchor + np.sin(np.arange(n) / 5) * 0.05},
        index=idx,
    )
    return dict(
        log_prices=log_prices,
        equities=["EQ"],
        anchor="KC1",
        split_date=idx[100],
        rolling_betas=pd.DataFrame({"EQ": 0.3}, index=idx),
        covariances={"EQ": pd.DataFrame([[2e-4, 1e-5], [1e-5, 1e-4]])},
        time_to_mean={"EQ": 5.0},
        avg_return_anchor=1e-4,
        avg_return_equity={"EQ": 2e-4},
        open_threshold=1.0,
        close_threshold=0.5,
        solver="scipy",
    )


def test_unrealistic_leverage_is_rejected():
    with pytest.raises(ValueError, match="max_gross_exposure"):
        run_strategy(**_tiny_problem(), max_gross_exposure=10.0)


def test_no_leverage_default_accepted_and_runs():
    result = run_strategy(**_tiny_problem())
    assert len(result.pnl) == 100


def test_leverage_scales_pnl_magnitude():
    base = run_strategy(**_tiny_problem(), max_gross_exposure=1.0)
    levered = run_strategy(**_tiny_problem(), max_gross_exposure=2.0)
    # a larger capital budget cannot shrink the traded book
    assert levered.pnl.abs().sum() >= base.pnl.abs().sum()
