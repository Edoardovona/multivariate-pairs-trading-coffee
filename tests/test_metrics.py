"""Unit tests for performance and trade metrics."""

import numpy as np
import pandas as pd

from src.metrics import max_drawdown, portfolio_metrics, sharpe_ratio, trade_metrics


def test_sharpe_positive_for_high_positive_returns():
    returns = pd.Series([0.01] * 252)  # 1% every day, far above the risk-free rate
    assert sharpe_ratio(returns, risk_free_rate=0.04) > 0


def test_sharpe_zero_for_constant_zero_returns():
    returns = pd.Series(np.zeros(252))
    assert sharpe_ratio(returns) == 0.0


def test_max_drawdown_is_zero_for_monotone_increase():
    values = pd.Series(np.linspace(100, 200, 252))
    assert max_drawdown(values) == 0.0


def test_max_drawdown_known_value():
    values = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert np.isclose(max_drawdown(values), -0.5)  # 120 -> 60


def test_portfolio_metrics_keys_and_signs():
    values = pd.Series(np.linspace(10_000, 12_000, 504))
    result = portfolio_metrics(values)
    assert set(result) == {
        "sharpe", "annualized_return", "total_return",
        "annualized_volatility", "max_drawdown",
    }
    assert result["total_return"] > 0
    assert result["max_drawdown"] <= 0
    assert result["annualized_volatility"] >= 0


def test_trade_metrics_basic_counts():
    trades = [
        {"pnl": 0.02, "direction": 1, "holding_days": 5},
        {"pnl": -0.01, "direction": -1, "holding_days": 3},
        {"pnl": 0.03, "direction": -1, "holding_days": 7},
    ]
    result = trade_metrics(trades)
    assert result["n_trades"] == 3
    assert np.isclose(result["pct_winning"], 2 / 3)
    assert result["largest_win"] == 0.03
    assert result["largest_loss"] == -0.01
    assert np.isclose(result["avg_holding_days"], 5.0)


def test_trade_metrics_empty_log():
    assert trade_metrics([]) == {}
