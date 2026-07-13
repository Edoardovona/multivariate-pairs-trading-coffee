"""Out-of-sample backtest engine for the multivariate pair trading strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio
from .optimizer import optimize_allocations
from .signals import compute_zscore


@dataclass
class BacktestResult:
    """Container for the daily PnL series, Sharpe ratio and trade log."""

    pnl: pd.Series
    sharpe: float
    trades: list[dict] = field(default_factory=list)

    def portfolio_value(self, initial_capital: float = 10_000.0) -> pd.Series:
        """Cumulative portfolio value from the daily PnL (returns in weight space)."""
        return initial_capital * (1 + self.pnl).cumprod()


def run_strategy(
    log_prices: pd.DataFrame,
    equities: list[str],
    anchor: str,
    split_date: pd.Timestamp,
    rolling_betas: pd.DataFrame,
    covariances: dict[str, pd.DataFrame],
    time_to_mean: dict[str, float],
    avg_return_anchor: float,
    avg_return_equity: dict[str, float],
    open_threshold: float,
    close_threshold: float,
    window: int = 22,
    risk_aversion: float = 1.0,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.04,
    solver: str = "auto",
    min_free_capital: float = 0.01,
    end_date: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run the daily out-of-sample simulation.

    Each day, in order: mark open positions to market (equity leg plus the
    beta-scaled anchor hedge), close positions whose z-score reverted
    inside the close band (deducting exit costs), then hand newly
    triggered signals to the optimizer, which sizes them subject to the
    no-leverage and beta-neutrality constraints. The beta active at entry
    is locked for the life of the position.

    The simulation runs from ``split_date`` to ``end_date`` (inclusive;
    defaults to the last observation). Positions still open on the final
    day are force-closed at that day's price, paying exit costs, so that
    walk-forward folds do not leak positions into each other.

    All prices are log-prices; position weights are fractions of the
    portfolio, so the daily PnL is expressed in return space.
    """
    zscores = {
        stock: compute_zscore(log_prices[stock] - log_prices[anchor], window)
        for stock in equities
    }

    pnl = pd.Series(0.0, index=log_prices.index)
    positions = {s: 0.0 for s in equities}      # signed equity-leg weight
    active_betas = {s: 0.0 for s in equities}   # beta locked at entry
    entry_day = {s: 0 for s in equities}
    entry_dir = {s: 0 for s in equities}
    trades: list[dict] = []

    start_idx = log_prices.index.get_loc(split_date)
    end_idx = (
        len(log_prices)
        if end_date is None
        else log_prices.index.get_loc(end_date) + 1
    )

    for t in range(start_idx, end_idx):
        daily_pnl = 0.0

        # 1. Mark-to-market of open positions (t-1 -> t)
        for stock in equities:
            if positions[stock] != 0:
                ret_equity = log_prices[stock].iloc[t] - log_prices[stock].iloc[t - 1]
                ret_anchor = log_prices[anchor].iloc[t] - log_prices[anchor].iloc[t - 1]
                w_equity = positions[stock]
                w_anchor = -w_equity * active_betas[stock]
                daily_pnl += w_equity * ret_equity + w_anchor * ret_anchor
        pnl.iloc[t] = daily_pnl

        # 2. Exits: z-score reverted inside the close band
        for stock in equities:
            if positions[stock] != 0:
                z = zscores[stock].iloc[t]
                if (positions[stock] > 0 and z > -close_threshold) or (
                    positions[stock] < 0 and z < close_threshold
                ):
                    exit_volume = abs(positions[stock]) * (1 + abs(active_betas[stock]))
                    pnl.iloc[t] -= exit_volume * transaction_cost
                    trades.append(
                        {
                            "stock": stock,
                            "direction": entry_dir[stock],
                            "pnl": pnl.iloc[entry_day[stock] : t + 1].sum(),
                            "holding_days": t - entry_day[stock],
                        }
                    )
                    positions[stock] = 0.0

        # 3. Entries: z-score breached the open band
        triggered = {}
        for stock in equities:
            if positions[stock] == 0.0:
                z = zscores[stock].iloc[t]
                if z > open_threshold:
                    triggered[stock] = -1
                elif z < -open_threshold:
                    triggered[stock] = 1

        # 4. Optimal sizing of the new positions
        if triggered:
            used_capital = sum(
                abs(positions[s]) * (1 + abs(active_betas[s])) for s in equities
            )
            available_capital = max(0.0, 1.0 - used_capital)
            if available_capital > min_free_capital:
                allocations = optimize_allocations(
                    triggered,
                    available_capital,
                    rolling_betas.iloc[t],
                    covariances,
                    time_to_mean,
                    avg_return_anchor,
                    avg_return_equity,
                    risk_aversion=risk_aversion,
                    solver=solver,
                )
                for stock, direction in triggered.items():
                    weight = allocations.get(stock, 0.0)
                    if weight > 0:
                        positions[stock] = weight * direction
                        active_betas[stock] = rolling_betas.iloc[t][stock]
                        entry_volume = weight * (1 + abs(active_betas[stock]))
                        pnl.iloc[t] -= entry_volume * transaction_cost
                        entry_day[stock] = t
                        entry_dir[stock] = direction

    # Force-close whatever is still open on the final day (fold boundary)
    last_t = end_idx - 1
    for stock in equities:
        if positions[stock] != 0:
            exit_volume = abs(positions[stock]) * (1 + abs(active_betas[stock]))
            pnl.iloc[last_t] -= exit_volume * transaction_cost
            trades.append(
                {
                    "stock": stock,
                    "direction": entry_dir[stock],
                    "pnl": pnl.iloc[entry_day[stock] : last_t + 1].sum(),
                    "holding_days": last_t - entry_day[stock],
                }
            )
            positions[stock] = 0.0

    pnl_oos = pnl.iloc[start_idx:end_idx]
    return BacktestResult(
        pnl=pnl_oos,
        sharpe=sharpe_ratio(pnl_oos, risk_free_rate=risk_free_rate),
        trades=trades,
    )


def buy_and_hold_benchmark(
    log_prices_oos: pd.DataFrame,
    assets: list[str],
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.Series:
    """Equally weighted buy-and-hold benchmark over the OOS window.

    A single entry cost per asset is deducted at inception; no rebalancing
    afterwards.
    """
    log_returns = log_prices_oos[assets].diff().fillna(0)
    portfolio_returns = log_returns.mean(axis=1)
    start_value = initial_capital * (1 - len(assets) * transaction_cost)
    return start_value * np.exp(portfolio_returns.cumsum())