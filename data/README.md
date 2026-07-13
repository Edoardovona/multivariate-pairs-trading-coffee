# Data

## Original study (results in the report and README)

The results reported in [`reports/Multivariate_Pair_Trading.pdf`](../reports/Multivariate_Pair_Trading.pdf)
were produced with **daily Bloomberg data** (OHLCV, 2016-02-12 to 2026-02-12):
`KC1` Arabica front-month continuous future, `DF1` Robusta, six sector ETFs/ETNs
and 17 coffee-value-chain equities. Bloomberg data is proprietary and licensed —
**it is not and will never be distributed with this repository.**

If you have Bloomberg access, export close prices as an Excel file with a date
column followed by `<TICKER>_LAST` columns, place it under `data/raw/` and point
`configs/default.yaml` at it. The loader (`src/data.py`) handles both formats.

## Free reproduction path (Yahoo Finance)

```bash
python scripts/download_data.py
```

downloads a comparable universe (KC=F for Arabica, XLP, and the candidate
equities) to `data/raw/prices.csv`.

**Caveat:** Yahoo series are *not* identical to Bloomberg's. The futures roll
methodology differs, prices are adjusted differently for corporate actions, and
some tickers from the original universe (COFF ETN, NESN on SIX, late-listed
names) are unavailable or partial. Expect the pipeline to run end-to-end and
produce qualitatively similar behaviour, but not the exact figures in the report.

## Expected schema

| column | description |
|--------|-------------|
| `date` (index) | trading day |
| `KC1`  | Arabica coffee front-month future close |
| `XLP`  | consumer staples sector ETF close |
| `SJM`, `KO`, `MDLZ`, `JVA`, ... | equity closes |

Everything under `data/raw/` is git-ignored by design.
