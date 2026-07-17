# Multivariate Pairs Trading on the Coffee Market

A Python implementation and extension of:

> Yang, H., & Malik, A. (2024). Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency Platform. 
> International Journal of Financial Studies, 12(3), 77.

This repository transposes the multivariate pair-trading framework above from its native cryptocurrency/fiat setting to the coffee market: a basket of spreads between the Arabica front-month future (`KC1`) and coffee value-chain equities, on ten years of daily Bloomberg data (2016–2026). The transposition is not a mechanical one — where crypto/fiat pairs are near-perfectly cointegrated by construction, the long-run link between a soft commodity and listed equities must be established statistically, and much of the project revolves around what happens when that foundation is weak.

Relative to the reference implementation, the pipeline adds a rolling walk-forward validation in place of a single chronological split, threshold calibration to limit selection bias, a dual-backend convex optimizer (Gurobi, with a SciPy fallback so the results are reproducible), and an alternative data path via Yahoo Finance.

The project was developed for the *Commodities Markets and Models* course, where each student replicated an trading strategy on an assigned commodity.

---

## Main Results

**Rolling walk-forward validation** on the Bloomberg dataset: 7 folds, each calibrated on a fixed 4-year window and traded on the following 12 months, stitched out-of-sample window Jan 2020 → Feb 2026 (~6 years of test data), $10,000 initial capital, 10 bps transaction costs per leg, Gurobi solver. The equity basket itself is re-selected on every calibration window (ADF + Engle-Granger screen, top-4 positive-hedge-ratio names), so nothing in the pipeline is frozen on information the fold could not have had.

Two dials control aggressiveness, both reported below. The risk-aversion coefficient **$\lambda$** sizes positions: a lower $\lambda$ weakens the variance penalty and pushes allocations toward the gross-exposure budget; it does not change entry/exit timing, which is threshold-driven. **Gross exposure** extends the paper's no-leverage constraint to a levered book: the runs below use **2× gross** (long + short legs, in units of NAV) with the borrowed portion **financed daily at the risk-free rate**, so leverage is not free. Values above 4× are rejected as unrealistic.

| Metric | MPTS $\lambda$=0.25 | MPTS $\lambda$=1 | MPTS $\lambda$=5 | B&H KC1 | B&H EW basket |
|---|---|---|---|---|---|
| Sharpe ratio | **+0.15** | +0.05 | −0.13 | 0.53 | −0.03 |
| Annualized return | 4.8% | 2.6% | 1.8% | 18.5% | 0.7% |
| Total return | 30.9% | 16.1% | 10.7% | 166% | 3.9% |
| Annualized volatility | 22.5% | 22.4% | **11.6%** | 37.7% | 22.8% |
| Max drawdown | −29.8% | −31.1% | **−27.2%** | −44.3% | −46.1% |
| Trades | — | 174 (57% winners) | — | — | — |

![Walk-forward equity curves per lambda vs benchmarks](reports/figures/walk_forward_equity.png)

Per-fold diagnostics at $\lambda$=1 (basket and thresholds re-estimated on each rolling calibration window):

| Fold | Test window | Basket | Thresholds (open/close) | Trades | Sharpe |
|---|---|---|---|---|---|
| 1 | Jan 2020 – Dec 2020 | SJM, FARM | ±1.6 / ±0.8 | 26 | **+0.74** |
| 2 | Jan 2021 – Nov 2021 | JVA, FARM | ±1.7 / ±0.8 | 25 | +0.17 |
| 3 | Dec 2021 – Nov 2022 | SJM, MCD, NSGRY, KO | ±2.2 / ±0.8 | 29 | −2.23 |
| 4 | Dec 2022 – Nov 2023 | MDLZ, MCD, KO, SJM | ±2.4 / ±1.1 | 23 | −1.86 |
| 5 | Dec 2023 – Oct 2024 | KO, NESN, NSGRY, MCD | ±2.3 / ±1.0 | 14 | −0.92 |
| 6 | Nov 2024 – Oct 2025 | KO, MCD, NESN, MDLZ | ±1.7 / ±0.7 | 48 | **+1.58** |
| 7 | Nov 2025 – Feb 2026 (short) | JVA, MCD | ±1.3 / ±1.0 | 9 | −2.93 |

Median fold Sharpe **−0.92**, range **[−2.93, +1.58]**.

**Reading the fold table.** The *median* fold Sharpe is reported next to the aggregate because with only 7 folds the mean is dominated by outliers (fold 7 is a 3-month stub); the median describes the typical fold, the range describes the dispersion, and a strategy whose fold Sharpe flips sign this often is regime-dependent regardless of the aggregate. Note also how the selected basket rotates — SJM/FARM in the calm regime, the large-cap staples later — and that some folds retain fewer than 4 names because the positive-hedge-ratio and unit-root filters leave fewer eligible candidates.

The walk-forward machinery behaves as designed: baskets and thresholds adapt fold by fold, volatility stays well below the KC1 benchmark's at every setting, and the aggressive configuration ($\lambda$=0.25, 2× gross) ends mildly positive net of transaction and financing costs. The honest caveat is that the aggressiveness dials change the *magnitude* of the P&L, not its quality: fold-level Sharpe still flips sign across regimes, so the mild aggregate edge is fragile rather than structural. That is the expected outcome given [the statistical limitation below](#the-main-statistical-limitation) — mean reversion cannot be traded reliably where cointegration is weak and unstable — and it is precisely the conclusion a single favorable split can hide.

---

## Methodology

**Pipeline** (every estimate is computed on data strictly prior to the window it is used in):

1. **Per-fold universe screening** — daily closes 2016–2026 for KC1, 6 sector ETFs and 17 coffee value-chain equities → log-prices → on *each* calibration window: ADF unit-root test (require $I(1)$) → Engle-Granger cointegration against KC1 → discard candidates with a negative hedge ratio (an inverted long-run relationship is not the co-movement a spread trades) → the top-4 names form that fold's basket. Re-screening per fold is consistent with the rolling philosophy: a basket frozen on one historical window would leak that window's regime into every fold. KDP is excluded outright — its Bloomberg back-history is unadjusted for the $103.75 special dividend of the 2018 Keurig–Dr Pepper merger, so the series is not economically continuous.
2. **Rolling walk-forward validation** — each fold estimates all parameters on a fixed-length 4-year calibration window, trades the following 12 months out-of-sample (1-month embargo in between, forced liquidation at fold boundaries), then the window slides forward by one test window, dropping the oldest data:

```text
time ──────────────────────────────────────────────────────────────────▶

fold 1   [======== calibration (4y) ========] e [--- test (12m) ---]
fold 2        [======== calibration (4y) ========] e [--- test (12m) ---]
fold 3             [======== calibration (4y) ========] e [--- test (12m) ---]
   ⋮                                  ⋮
         e = 1-month embargo · window slides forward by one test length
         re-estimated per fold: basket, thresholds, expected profits, covariances
         reported performance: the stitched test windows only
```

   *Why rolling rather than expanding:* the sample splits into structurally different regimes (range-bound 2016–2019, supply-shock trends from 2021 on); an expanding window would let stale early-regime threshold economics dominate every later calibration — the regime-mismatch failure the original study diagnosed. Rolling keeps the estimation sample representative of current dynamics and its size constant across folds; the trade-off is fewer observations per calibration (noisier estimates), mitigated by the plateau selection in step 3. *Why the embargo:* rolling statistics at the start of a test window (the 22-day $z$-score in particular) would otherwise be computed largely from the last calibration days — information the calibration already consumed; skipping one month removes that overlap at negligible cost (the purging/embargo idea of combinatorial cross-validation). Only stitched test-fold performance is ever reported.
3. **Signals** — 22-day rolling $z$-score of each log-spread; open when $|z|$ breaches the entry band, close on reversion. Thresholds come from a per-fold grid search (open $\in \[1, 2.5)$, close $\in \[0.5, 2.0)$, close < open enforced) maximizing the average Sharpe across the basket; to curb per-fold selection bias, the strategy uses the **median of the top decile of the grid** (the plateau) rather than the argmax cell.
4. **Hedging** — each equity's coffee sensitivity β<sub>i,KC1</sub> is isolated via multivariate OLS on KC1 **and** the consumer-staples ETF (XLP), estimated on a 252-day rolling window (causal by construction) and locked at trade entry.
5. **Position sizing** — every day with triggered signals, solve:

$$\max_{W}\; \sum_n W_n \cdot (EP_n \odot [1,-1])^{\prime} \;-\; \lambda \sum_n W_n\, \tilde{\Sigma}_n\, W_n^{\prime}$$

$$\text{s.t.}\quad 0 \le W_{long} \le 1,\quad -1 \le W_{short} \le 0,\quad \sum_n (W_{long} - W_{short}) \le \text{capital},\quad \sum_e \beta_{e}\,(W_{long}+W_{short}) = 0$$

   where the expected profit *EP* combines train-window mean returns with a mean-reversion-speed proxy (average round-trip holding time), and Σ̃ is the pair covariance with sign-flipped off-diagonals. Solved with **Gurobi** (original study) or an open-source **SciPy SLSQP** fallback so anyone can run it.
6. **Backtest** — daily simulation over each test fold with mark-to-market, per-leg transaction costs, capital recycling, trade logging and forced liquidation at fold boundaries; benchmarked against KC1 buy-and-hold and an equally weighted equity basket over the same stitched period. The capital budget is set by `max_gross_exposure`: 1.0 reproduces the paper's no-leverage constraint, values up to 4× allow a levered market-neutral book with the borrowed portion financed daily at the risk-free rate, and anything beyond 4× is rejected as outside industry practice.

<p align="center">
<img src="reports/figures/zscore_signals_oos.png" width="70%" alt="Z-score signals over the OOS period"/>
</p>

---

## Repository structure

```
multivariate-pairs-coffee/
├── README.md
├── configs/
│   ├── default.yaml            # every parameter of the pipeline (Yahoo data)
│   └── bloomberg.yaml          # same pipeline on the original Bloomberg export
├── data/
│   ├── README.md               
│   └── raw/                    # git-ignored 
├── notebooks/
│   ├── 01_statistical_screening.ipynb   # EDA, ADF, Engle-Granger, basket selection
│   └── 02_backtest_analysis.ipynb       # calibration, backtest, benchmarks
├── scripts/
│   ├── download_data.py        # Yahoo Finance universe builder
│   └── run_backtest.py         # one-command end-to-end pipeline
├── src/
│   ├── data.py                 # loaders (Bloomberg xlsx / Yahoo csv), screening window
│   ├── stat_tests.py           # ADF and Engle-Granger screening
│   ├── signals.py              # z-scores, threshold grid search, reversion speed
│   ├── hedging.py              # static & rolling multivariate betas, covariances
│   ├── optimizer.py            # bi-objective allocation (Gurobi + SciPy backends)
│   ├── backtest.py             # daily simulation engine, trade log, benchmarks
│   ├── walk_forward.py         # rolling folds, plateau calibration, WF runner
│   └── metrics.py              # Sharpe, drawdown, trade-level statistics
├── tests/                      # pytest unit tests (no license, no network needed)
├── reports/
├   ├── YangMalik2024.pdf       # main reference paper 
│   ├── figures/                # committed figures embedded in this README
│   ├── equity_curves.csv       # generated by run_backtest.py (git-ignored)
│   └── walk_forward_folds.csv  # generated by run_backtest.py (git-ignored)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
git clone https://github.com/Edoardovona/multivariate-pairs-coffee.git
cd multivariate-pairs-coffee
pip install -r requirements.txt

python scripts/download_data.py      # free Yahoo Finance data --> data/raw/prices.csv
python scripts/run_backtest.py       # screening --> calibration --> backtest 
```

With the original Bloomberg export (see [A note on data](#a-note-on-data)):

```bash
# place the export at data/raw/CoffeeData.xlsx, then
python scripts/run_backtest.py --config configs/bloomberg.yaml
```

Run the tests:

```bash
pytest tests/ -v
```

Every parameter (universe, walk-forward fold structure, $\lambda$, transaction costs, solver) lives in [`configs/default.yaml`](configs/default.yaml).

## A note on data

The study was conducted on a **Bloomberg** OHLCV daily dataset. The file is **git-ignored and not distributed** — it lives only in a local copy at `data/raw/CoffeeData.xlsx` and is consumed via [`configs/bloomberg.yaml`](configs/bloomberg.yaml).

So that anyone cloning this repository can still run the pipeline end-to-end, `scripts/download_data.py` rebuilds a comparable universe from **free Yahoo Finance data** (`KC=F` for the Arabica future), which the default config points at. Details in [`data/README.md`](data/README.md).


## The main statistical limitation

**There is essentially no cointegration within the assigned equity basket — and this is a constraint of the assigned universe, not of the methodology.** In the screening window only SJM is cointegrated with KC1 at the 5% level; the other candidates show Engle-Granger p-values between 0.22 and 0.53. The economics are intuitive: large consumer-staples companies (Coca-Cola, Mondelēz) have the pricing power to pass coffee input costs through to consumers, which severs the long-run equilibrium between their equity prices and the commodity. Pairs trading exploits mean reversion, and mean reversion requires a stable long-run relationship — no optimizer, hedge or calibration scheme can manufacture one that is statistically absent.

**The acknowledged simple alternative:** trading the **Arabica–Robusta spread (KC1 vs DF1)** directly. The two coffee qualities are economically bound by substitution in roasters' blends, making them a far more natural cointegrated pair than any coffee-adjacent equity. The course assignment was to replicate the multivariate equity-basket framework of the reference paper on the assigned commodity, which is what this repository does — the inter-commodity spread is the obvious next experiment and is listed in the extensions below.

## Other findings

- **Threshold economics are regime-dependent.** The per-fold calibrations drift as the sample grows and the volatility regime shifts — visible directly in the fold table above. This is precisely what a single-split design cannot reveal, and why the walk-forward structure replaced it.
- **Risk control works.** The beta-neutrality constraint and variance penalty compress volatility to a fraction of the KC1 benchmark's and roughly halve the drawdown, across data sources and configurations.
- **Transaction-cost sensitivity is moderate** at daily frequency (multi-day average holding), unlike the intraday original study where costs accumulate over thousands of trades.

## Possible extensions

- **Arabica–Robusta (KC1/DF1) inter-commodity spread** — the natural cointegrated pair, per the limitation above.
- Kalman-filter time-varying hedge ratio instead of static Engle-Granger betas.
- Johansen cointegration on the full basket rather than pairwise tests.
- Volatility-adaptive thresholds (bands scaled by short- vs long-run spread volatility) to react to regime shifts between fold recalibrations.
- Broader universe: FX of producer countries (BRL, VND), cross-exchange spreads.

## References

- Yang, H. & Malik, A. (2024). *Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency Platform*. International Journal of Financial Studies, 12(3):77.
- Silveira, Mattos & Saes (2017). *The Reaction of Coffee Futures Price Volatility to Crop Reports*. Emerging Markets Finance and Trade, 53(10).
- Geman, H. (2014). *Agricultural Finance: from Crops to Land, Water and Infrastructure*. Wiley.