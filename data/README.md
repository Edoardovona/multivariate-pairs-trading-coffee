# Data

## Original study (results in the report and README)

The results reported in [`reports/Multivariate_Pair_Trading.pdf`](../reports/Multivariate_Pair_Trading.pdf)
were produced with **daily Bloomberg data** (OHLCV, 2016-02-12 to 2026-02-12):
`KC1` Arabica front-month continuous future, `DF1` Robusta, six sector ETFs/ETNs
and 17 coffee-value-chain equities. Bloomberg data it is not distributed with this repository
(everything under `data/raw/` is git-ignored by design).

The author's local copy lives at `data/raw/CoffeeData.xlsx` and is used via

```bash
python scripts/run_backtest.py --config configs/bloomberg.yaml
```

If you have Bloomberg access, export close prices as an Excel file with a date
column followed by `<TICKER>_LAST` columns, place it at that same path and use
the same config. The loader (`src/data.py`) auto-detects the format from the
file extension.

## Free reproduction path (Yahoo Finance)

```bash
python scripts/download_data.py
```

downloads a comparable universe (KC=F for Arabica, XLP, and the candidate
equities) to `data/raw/prices.csv`.

**Note that Yahoo series are not identical to Bloomberg's**. The futures roll
methodology differs, prices are adjusted differently for corporate actions, and
some tickers from the original universe (COFF ETN, NESN on SIX, late-listed
names) are unavailable or partial. Expect the pipeline to run end-to-end and
produce qualitatively similar behaviour, but not the exact figures in the report.
This is the source `configs/default.yaml` points at.

## Canonical format after loading

Whatever the source, the loaders in `src/data.py` normalize the data into a
single close-price DataFrame that the rest of the pipeline consumes:

| column | description |
|--------|-------------|
| `date` (index) | trading day |
| `KC1`  | Arabica coffee front-month future close |
| `XLP`  | consumer staples sector ETF close |
| `SJM`, `KO`, `MDLZ`, `JVA`, ... | equity closes |

The Yahoo CSV already has this layout on disk. The Bloomberg Excel does not —
it holds a date column followed by `<TICKER>_LAST` close columns (plus other
OHLCV fields) — so `load_bloomberg_excel` keeps only the `*_LAST` columns,
strips the suffix and indexes by date, producing the same DataFrame. To plug in
any other data source, just match this format.
