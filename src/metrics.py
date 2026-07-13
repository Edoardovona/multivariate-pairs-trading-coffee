"""Performance and trade-level metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _daily_risk_free(risk_free_rate: float) -> float:
    return (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio of a daily return series."""
    returns = returns.dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns - _daily_risk_free(risk_free_rate)
    return float(excess.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(values: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a portfolio value series (<= 0)."""
    return float((values / values.cummax() - 1).min())


def portfolio_metrics(values: pd.Series, risk_free_rate: float = 0.04) -> dict[str, float]:
    """Summary metrics from a portfolio value series.

    Returns Sharpe ratio, annualized/total return, annualized volatility
    and maximum drawdown.
    """
    returns = values.pct_change().dropna()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    total_return = float(values.iloc[-1] / values.iloc[0] - 1)
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    return {
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "annualized_return": annualized_return,
        "total_return": total_return,
        "annualized_volatility": float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "max_drawdown": max_drawdown(values),
    }


def trade_metrics(trades: list[dict]) -> dict[str, float]:
    """Trade-level statistics from the backtest trade log.

    Each trade record holds ``pnl``, ``direction`` (+1/-1) and
    ``holding_days``.
    """
    if not trades:
        return {}
    pnls = [t["pnl"] for t in trades]
    directions = [t["direction"] for t in trades]
    holdings = [t["holding_days"] for t in trades]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    longs = [p for p, d in zip(pnls, directions) if d == 1]
    shorts = [p for p, d in zip(pnls, directions) if d == -1]
    long_wins = [p for p in longs if p > 0]
    short_wins = [p for p in shorts if p > 0]

    return {
        "n_trades": len(pnls),
        "pct_winning": len(wins) / len(pnls),
        "pct_losing": len(losses) / len(pnls),
        "pct_long_win": len(long_wins) / len(longs) if longs else 0.0,
        "pct_short_win": len(short_wins) / len(shorts) if shorts else 0.0,
        "win_loss_ratio": (
            abs(np.mean(wins) / np.mean(losses)) if wins and losses else np.nan
        ),
        "average_win": float(np.mean(wins)) if wins else 0.0,
        "average_loss": float(np.mean(losses)) if losses else 0.0,
        "largest_win": max(pnls),
        "largest_loss": min(pnls),
        "avg_holding_days": float(np.mean(holdings)),
    }
