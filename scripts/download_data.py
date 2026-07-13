"""Download free daily close prices from Yahoo Finance.

The original study used a Bloomberg export that cannot be redistributed.
This script rebuilds a comparable universe from Yahoo Finance so the whole
pipeline can run end-to-end with free data. Series are not identical to
the Bloomberg ones (different futures roll methodology, corporate-action
adjustments and listing coverage), so results will differ from the report.

Usage:
    python scripts/download_data.py [--start 2016-02-12] [--end 2026-02-12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# Internal name -> Yahoo Finance ticker.
# KC=F is the ICE Arabica coffee front-month future (Bloomberg: KC1).
TICKER_MAP = {
    "KC1": "KC=F",
    "XLP": "XLP",
    "SJM": "SJM",
    "KO": "KO",
    "MDLZ": "MDLZ",
    "JVA": "JVA",
    "FARM": "FARM",
    "NSRGY": "NSRGY",
    "MCD": "MCD",
    "SBUX": "SBUX",
    "QSR": "QSR",
    "KDP": "KDP",
}


def download_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for the full universe."""
    data = yf.download(
        list(TICKER_MAP.values()), start=start, end=end, auto_adjust=True
    )["Close"]
    reverse_map = {v: k for k, v in TICKER_MAP.items()}
    prices = data.rename(columns=reverse_map)
    prices.index.name = "date"
    return prices[list(TICKER_MAP.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-02-12")
    parser.add_argument("--end", default="2026-02-12")
    parser.add_argument("--output", default="data/raw/prices.csv")
    args = parser.parse_args()

    prices = download_prices(args.start, args.end)
    if prices.dropna(how="all").empty:
        raise SystemExit(
            "Download returned no data (Yahoo Finance may be rate-limiting; "
            "retry in a few minutes). Existing file left untouched."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} assets to {output}")
    print(prices.tail())


if __name__ == "__main__":
    main()