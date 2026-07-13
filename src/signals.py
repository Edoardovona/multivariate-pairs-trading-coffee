"""Z-score trading signals and in-sample threshold calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio


def compute_zscore(spread: pd.Series, window: int = 22) -> pd.Series:
    """Rolling z-score of a spread series.

    A 22-day window corresponds to roughly one trading month, following
    the calibration used in the report.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / rolling_std


def _position_path(
    zscore: np.ndarray, open_threshold: float, close_threshold: float
) -> np.ndarray:
    """Signed position (+1/0/-1) held at the close of each day.

    Opens short (long) when the z-score exceeds +open_threshold (falls
    below -open_threshold) and closes when it reverts inside the close band.
    """
    positions = np.zeros(len(zscore))
    position = 0
    for t, z in enumerate(zscore):
        if position == 0:
            if z > open_threshold:
                position = -1
            elif z < -open_threshold:
                position = 1
        elif position == 1 and z > -close_threshold:
            position = 0
        elif position == -1 and z < close_threshold:
            position = 0
        positions[t] = position
    return positions


def threshold_backtest(
    spread: pd.Series,
    open_threshold: float,
    close_threshold: float,
    window: int = 22,
    risk_free_rate: float = 0.04,
) -> float:
    """Sharpe ratio of a single-spread threshold strategy.

    Used only for the calibration grid search; the full out-of-sample
    engine lives in :mod:`src.backtest`. The day-t spread move is booked
    with the position carried from t-1 — booking it against the position
    updated on the day-t signal would charge every entry with the very
    move that triggered it.
    """
    zscore = compute_zscore(spread, window).dropna()
    values = spread.loc[zscore.index].to_numpy()
    positions = _position_path(zscore.to_numpy(), open_threshold, close_threshold)
    returns = positions[:-1] * np.diff(values)
    return sharpe_ratio(pd.Series(returns), risk_free_rate=risk_free_rate)


def grid_search_thresholds(
    log_prices: pd.DataFrame,
    equities: list[str],
    anchor: str,
    open_grid: np.ndarray | None = None,
    close_grid: np.ndarray | None = None,
    window: int = 22,
    risk_free_rate: float = 0.04,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Grid search of (open, close) thresholds maximizing the average Sharpe.

    Only combinations with ``close < open`` are evaluated, preventing
    degenerate bands. Returns the ranked results and the Sharpe matrix
    (open x close) for heatmap plotting.
    """
    if open_grid is None:
        open_grid = np.round(np.arange(1.0, 2.5, 0.1), 1)
    if close_grid is None:
        close_grid = np.round(np.arange(0.5, 2.0, 0.1), 1)

    heatmap = np.full((len(open_grid), len(close_grid)), np.nan)
    rows = []
    for i, open_th in enumerate(open_grid):
        for j, close_th in enumerate(close_grid):
            if close_th >= open_th:
                continue
            sharpes = [
                threshold_backtest(
                    log_prices[stock] - log_prices[anchor],
                    open_th,
                    close_th,
                    window=window,
                    risk_free_rate=risk_free_rate,
                )
                for stock in equities
            ]
            avg_sharpe = float(np.mean(sharpes))
            rows.append({"open": open_th, "close": close_th, "sharpe": avg_sharpe})
            heatmap[i, j] = avg_sharpe

    results = (
        pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    )
    return results, heatmap


def compute_time_to_mean(
    log_prices: pd.DataFrame,
    equities: list[str],
    anchor: str,
    open_threshold: float,
    close_threshold: float,
    window: int = 22,
) -> dict[str, float]:
    """Average holding period (days) of threshold round-trips per spread.

    Serves as the mean-reversion speed proxy ``mr`` in the expected-profit
    term of the optimizer. Defaults to 1.0 when a spread never triggers.
    """
    time_to_mean: dict[str, float] = {}
    for stock in equities:
        spread = log_prices[stock] - log_prices[anchor]
        zscore = compute_zscore(spread, window).dropna().abs().to_numpy()
        holding_periods = []
        entry_t = -1
        for t, z in enumerate(zscore):
            if entry_t < 0:
                if z > open_threshold:
                    entry_t = t
            elif z < close_threshold:
                holding_periods.append(t - entry_t)
                entry_t = -1
        time_to_mean[stock] = float(np.mean(holding_periods)) if holding_periods else 1.0
    return time_to_mean