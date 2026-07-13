"""Data loading, cleaning and train/test splitting utilities.

Two data sources are supported:

1. **Bloomberg Excel export** (used in the original study, not distributed
   with this repository). Expected layout: a date column followed by
   ``<TICKER>_LAST`` close-price columns.
2. **Yahoo Finance CSV** produced by ``scripts/download_data.py``. Free and
   fully reproducible, but note that Yahoo series differ from Bloomberg
   continuous futures (roll methodology, listing history), so results will
   not match the report exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_bloomberg_excel(path: str | Path) -> pd.DataFrame:
    """Load close prices from a Bloomberg OHLCV Excel export.

    Keeps only the ``*_LAST`` columns, strips the suffix and indexes the
    frame by date.

    Parameters
    ----------
    path:
        Path to the Excel file (first column: dates, remaining columns:
        Bloomberg fields such as ``KC1_LAST``).

    Returns
    -------
    pd.DataFrame
        Close prices indexed by ``DatetimeIndex``, one column per asset.
    """
    raw = pd.read_excel(path)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    numeric = numeric[[c for c in numeric.columns if not c.startswith("Unnamed")]]

    last_cols = [c for c in numeric.columns if c.endswith("_LAST")]
    prices = numeric[last_cols].copy()
    prices.columns = prices.columns.str.replace("_LAST", "", regex=False)
    prices.index = pd.DatetimeIndex(dates)
    prices = prices.sort_index()
    prices.index.name = "date"
    return prices


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a close-price CSV (as written by ``scripts/download_data.py``)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices.index.name = "date"
    return prices


def load_prices(path: str | Path) -> pd.DataFrame:
    """Dispatch to the Excel or CSV loader based on the file extension."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return load_bloomberg_excel(path)
    return load_prices_csv(path)


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Count NaNs, zeros and infinities per column."""
    return pd.DataFrame(
        {
            "nans": prices.isna().sum(),
            "zeros": (prices == 0).sum(),
            "infs": prices.isin([np.inf, -np.inf]).sum(),
        }
    ).sort_values(by=["nans", "zeros", "infs"], ascending=False)


def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Natural-log transform of price levels."""
    return np.log(prices)


def train_test_split(
    log_prices: pd.DataFrame, target: str, split_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Chronological in-sample / out-of-sample split.

    The split date is chosen so that ``split_ratio`` of the target asset's
    observations fall in-sample. All screening and calibration must use
    the in-sample window only, to avoid look-ahead bias.

    Returns
    -------
    (in_sample, out_of_sample, split_date)
    """
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be in (0, 1)")
    idx = log_prices[target].index
    split_date = idx[int(len(idx) * split_ratio)]
    in_sample = log_prices.loc[log_prices.index < split_date]
    out_of_sample = log_prices.loc[log_prices.index >= split_date]
    return in_sample, out_of_sample, split_date
