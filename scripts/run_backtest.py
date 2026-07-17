"""End-to-end pipeline: screening, walk-forward validation, reporting.

Usage:
    python scripts/run_backtest.py [--config configs/default.yaml]

Steps:
    1. Load close prices (Yahoo CSV or Bloomberg Excel) and log-transform.
    2. ADF + Engle-Granger screening on the first calibration window only.
    3. Rolling-window walk-forward: every fold re-calibrates thresholds,
       mean-reversion speed, expected returns and covariances on its own
       fixed-length calibration window, then trades the next test window
       out-of-sample.
    4. Performance report vs. buy-and-hold benchmarks over the stitched
       test folds, plus per-fold and trade-level diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, data, metrics, stat_tests, walk_forward


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def plot_walk_forward(
    results: dict[float, walk_forward.WalkForwardResult],
    benchmarks: dict[str, pd.Series],
    initial_capital: float,
    output_path: Path,
) -> None:
    """Stitched OOS equity curves (one per lambda) vs. benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for lam, result in results.items():
        ax.plot(result.portfolio_value(initial_capital), lw=1.4,
                label=f"MPTS λ={lam} (Sharpe {result.sharpe:.2f})")
    for name, series in benchmarks.items():
        ax.plot(series, ls=":", lw=1.3, label=name)
    reference = next(iter(results.values()))
    for test_start in pd.to_datetime(reference.folds["test_start"]).iloc[1:]:
        ax.axvline(test_start, color="grey", ls="--", lw=0.7, alpha=0.6)
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    anchor = cfg["universe"]["anchor"]
    equities = cfg["universe"]["equities"]
    features = [anchor, cfg["universe"]["sector_proxy"]]
    wf_cfg = cfg["walk_forward"]
    rf = cfg["strategy"]["risk_free_rate"]
    tc = cfg["strategy"]["transaction_cost"]
    initial_capital = cfg["backtest"]["initial_capital"]

    # 1. Data
    prices = data.load_prices(cfg["data"]["path"])
    log_prices = data.to_log_prices(prices)
    print(f"Loaded {prices.shape[0]} rows x {prices.shape[1]} assets")

    # 2. Screening on the first calibration window only (no look-ahead)
    log_screen = data.screening_window(log_prices, wf_cfg["calibration_years"])
    exclude = set(cfg["universe"]["exclude"]) | {anchor}
    adf_results = stat_tests.adf_screen(log_screen, exclude=exclude)
    eg_results = stat_tests.engle_granger_screen(log_screen, anchor, exclude=exclude)
    print(f"\n=== Screening on the first calibration window "
          f"({log_screen.index[0].date()} -> {log_screen.index[-1].date()}) ===")
    print(stat_tests.summary_table(log_screen, anchor, adf_results, eg_results)
          .head(6).round(4).to_string())

    # 3. Walk-forward validation. When reselect_basket is on, the ADF +
    # Engle-Granger screen is re-run on every calibration window and the
    # basket is re-selected from the whole eligible universe.
    candidates = (
        [c for c in log_prices.columns if c not in exclude and c != anchor]
        if wf_cfg.get("reselect_basket", False)
        else None
    )
    # The lambda sweep is the aggressiveness dial (report, Section 4.2):
    # lower lambda -> weaker variance penalty -> positions sized up to the
    # capital bound. It scales position size, not signal timing.
    base_lambda = cfg["strategy"]["risk_aversion"]
    sweep = cfg["strategy"].get("risk_aversion_sweep", [base_lambda])
    if base_lambda not in sweep:
        sweep = sorted({base_lambda, *sweep})

    results: dict[float, walk_forward.WalkForwardResult] = {}
    for lam in sweep:
        print(f"\n=== Rolling-window walk-forward (lambda={lam}) ===")
        results[lam] = walk_forward.run_walk_forward(
            log_prices, equities, anchor, features,
            calibration_years=wf_cfg["calibration_years"],
            test_months=wf_cfg["test_months"],
            embargo_days=wf_cfg["embargo_days"],
            candidates=candidates,
            n_pairs=cfg["universe"].get("n_pairs", 4),
            window=cfg["signals"]["zscore_window"],
            risk_aversion=lam,
            transaction_cost=tc,
            risk_free_rate=rf,
            rolling_beta_window=cfg["strategy"]["rolling_beta_window"],
            solver=cfg["strategy"]["solver"],
            max_gross_exposure=cfg["strategy"].get("max_gross_exposure", 1.0),
            verbose=(lam == base_lambda),
        )
    base = results[base_lambda]

    print(f"\n=== Per-fold diagnostics (lambda={base_lambda}) ===")
    print(base.folds.round(3).to_string(index=False))
    print(f"\nFold Sharpe: median {base.folds['sharpe'].median():.3f}, "
          f"range [{base.folds['sharpe'].min():.3f}, "
          f"{base.folds['sharpe'].max():.3f}]")

    # 4. Aggregate performance vs. benchmarks over the stitched test folds
    log_oos = log_prices.loc[base.pnl.index]
    kc1_value = backtest.buy_and_hold_benchmark(log_oos, [anchor], initial_capital, tc)
    ew_value = backtest.buy_and_hold_benchmark(log_oos, equities, initial_capital, tc)

    print("\n=== Aggregate out-of-sample performance (stitched folds) ===")
    report = pd.DataFrame(
        {f"MPTS lam={lam}": metrics.portfolio_metrics(
            res.portfolio_value(initial_capital), rf)
         for lam, res in results.items()}
    )
    report[f"B&H {anchor}"] = pd.Series(metrics.portfolio_metrics(kc1_value, rf))
    report["B&H EW basket"] = pd.Series(metrics.portfolio_metrics(ew_value, rf))
    print(report.round(4).to_string())

    print(f"\n=== Trade-level statistics (lambda={base_lambda}) ===")
    for key, value in metrics.trade_metrics(base.trades).items():
        print(f"{key:>20}: {value:.4f}" if isinstance(value, float)
              else f"{key:>20}: {value}")

    # 5. Outputs
    output_dir = Path("reports")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    equity_curves = pd.DataFrame(
        {f"MPTS_lambda_{lam}": res.portfolio_value(initial_capital)
         for lam, res in results.items()}
    )
    equity_curves[f"BH_{anchor}"] = kc1_value
    equity_curves["BH_EW"] = ew_value
    equity_curves.to_csv(output_dir / "equity_curves.csv")
    base.folds.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    plot_walk_forward(
        results,
        {f"B&H {anchor}": kc1_value, "B&H EW basket": ew_value},
        initial_capital,
        output_dir / "figures" / "walk_forward_equity.png",
    )
    print(f"\nSaved equity curves, fold table and figure under {output_dir}/")


if __name__ == "__main__":
    main()
