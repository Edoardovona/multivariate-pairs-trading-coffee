"""Expanding-window walk-forward validation.

Instead of a single chronological train/test split — which calibrates the
thresholds once, on one market regime, and judges them on another — the
sample is swept by expanding folds:

    fold 1:  train [0 ....... T1) -> embargo -> test [T1 .. T2)
    fold 2:  train [0 ............. T2) -> embargo -> test [T2 .. T3)
    ...

Every fold re-calibrates the signal thresholds, the mean-reversion speed,
the expected returns and the pair covariances on its own (growing) train
window, then trades the following test window out-of-sample. The stitched
test-fold PnL is the only performance ever reported.

To reduce selection bias in the per-fold grid search, thresholds are taken
as the median of the top decile of the grid (the plateau) rather than the
single argmax cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import backtest, hedging, signals
from .metrics import sharpe_ratio

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21


@dataclass
class WalkForwardResult:
    """Stitched out-of-sample PnL, per-fold diagnostics and trade log."""

    pnl: pd.Series
    sharpe: float
    folds: pd.DataFrame
    trades: list[dict] = field(default_factory=list)

    def portfolio_value(self, initial_capital: float = 10_000.0) -> pd.Series:
        """Cumulative portfolio value over the stitched test folds."""
        return initial_capital * (1 + self.pnl).cumprod()


def walk_forward_folds(
    index: pd.DatetimeIndex,
    min_train_size: int,
    test_size: int,
    embargo: int = 0,
    min_test_size: int = 60,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Build expanding (anchored) train/test folds over a trading calendar.

    Parameters
    ----------
    index:
        Full trading-day index.
    min_train_size:
        Trading days in the first train window; every later fold's train
        window is the entire history up to its test start (expanding).
    test_size:
        Trading days per test window. Folds step forward by this amount,
        so test windows are contiguous and non-overlapping.
    embargo:
        Trading days skipped between train end and test start, so that
        rolling statistics computed at the start of the test window do not
        lean on the very last training days.
    min_test_size:
        Discard a final stub fold shorter than this.
    """
    folds = []
    start = min_train_size
    while start + embargo < len(index):
        train = index[:start]
        test = index[start + embargo : start + test_size]
        if len(test) >= min_test_size:
            folds.append((train, test))
        start += test_size
    return folds


def calibrate_thresholds(
    log_train: pd.DataFrame,
    equities: list[str],
    anchor: str,
    window: int = 22,
    risk_free_rate: float = 0.04,
    top_fraction: float = 0.1,
) -> tuple[float, float]:
    """Grid-search the thresholds on the train window, then pick the plateau.

    Returns the median (open, close) of the top ``top_fraction`` of grid
    cells by average Sharpe. Using the plateau centre instead of the argmax
    cell reduces the selection bias of running one grid search per fold.
    """
    grid, _ = signals.grid_search_thresholds(
        log_train, equities, anchor, window=window, risk_free_rate=risk_free_rate
    )
    top = grid.head(max(1, int(len(grid) * top_fraction)))
    open_th = round(float(top["open"].median()), 1)
    close_th = round(float(top["close"].median()), 1)
    if close_th >= open_th:  # medians of a ragged plateau can cross; fall back
        open_th = float(grid.iloc[0]["open"])
        close_th = float(grid.iloc[0]["close"])
    return open_th, close_th


def run_walk_forward(
    log_prices: pd.DataFrame,
    equities: list[str],
    anchor: str,
    features: list[str],
    min_train_years: float = 4.0,
    test_months: int = 12,
    embargo_days: int = 21,
    window: int = 22,
    risk_aversion: float = 1.0,
    transaction_cost: float = 0.001,
    risk_free_rate: float = 0.04,
    rolling_beta_window: int = 252,
    solver: str = "auto",
    verbose: bool = True,
) -> WalkForwardResult:
    """Run the full walk-forward evaluation.

    Per fold, estimated on the train window only: signal thresholds
    (plateau of the grid search), mean-reversion speed, average returns
    and pair covariance matrices. The rolling betas are computed once —
    they only ever use the trailing ``rolling_beta_window`` days, so they
    are causal by construction.
    """
    folds = walk_forward_folds(
        log_prices.index,
        min_train_size=int(min_train_years * TRADING_DAYS_PER_YEAR),
        test_size=test_months * TRADING_DAYS_PER_MONTH,
        embargo=embargo_days,
    )
    if not folds:
        raise ValueError("Sample too short for the requested fold structure")

    rolling = hedging.rolling_betas(
        log_prices, equities, features, anchor, window=rolling_beta_window
    )

    segments: list[pd.Series] = []
    records: list[dict] = []
    all_trades: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        log_train = log_prices.loc[train_idx]

        open_th, close_th = calibrate_thresholds(
            log_train, equities, anchor, window=window,
            risk_free_rate=risk_free_rate,
        )
        _, residuals = hedging.estimate_static_betas(
            log_train, equities, features, anchor
        )
        covariances = hedging.pair_covariances(
            log_train[anchor].diff().dropna(),
            residuals.diff().dropna(),
            equities,
        )
        time_to_mean = signals.compute_time_to_mean(
            log_train, equities, anchor, open_th, close_th, window=window
        )
        avg_ret_anchor = log_train[anchor].diff().dropna().mean()
        avg_ret_equity = {
            s: log_train[s].diff().dropna().mean() for s in equities
        }

        result = backtest.run_strategy(
            log_prices, equities, anchor,
            split_date=test_idx[0], end_date=test_idx[-1],
            rolling_betas=rolling, covariances=covariances,
            time_to_mean=time_to_mean,
            avg_return_anchor=avg_ret_anchor,
            avg_return_equity=avg_ret_equity,
            open_threshold=open_th, close_threshold=close_th,
            window=window, risk_aversion=risk_aversion,
            transaction_cost=transaction_cost,
            risk_free_rate=risk_free_rate, solver=solver,
        )
        segments.append(result.pnl)
        all_trades.extend(result.trades)
        records.append(
            {
                "fold": i,
                "train_start": train_idx[0].date(),
                "test_start": test_idx[0].date(),
                "test_end": test_idx[-1].date(),
                "open_th": open_th,
                "close_th": close_th,
                "n_trades": len(result.trades),
                "sharpe": result.sharpe,
                "total_return": float((1 + result.pnl).prod() - 1),
            }
        )
        if verbose:
            print(
                f"fold {i}: train {train_idx[0].date()} -> {train_idx[-1].date()}"
                f" | test {test_idx[0].date()} -> {test_idx[-1].date()}"
                f" | thresholds +/-{open_th}/+/-{close_th}"
                f" | {len(result.trades)} trades | Sharpe {result.sharpe:.3f}"
            )

    pnl = pd.concat(segments)
    return WalkForwardResult(
        pnl=pnl,
        sharpe=sharpe_ratio(pnl, risk_free_rate=risk_free_rate),
        folds=pd.DataFrame(records),
        trades=all_trades,
    )
