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
    result: walk_forward.WalkForwardResult,
    benchmarks: dict[str, pd.Series],
    initial_capital: float,
    output_path: Path,
) -> None:
    """Stitched OOS equity curve vs. benchmarks, with fold boundaries."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(result.portfolio_value(initial_capital), lw=1.4,
            label=f"MPTS walk-forward (Sharpe {result.sharpe:.2f})")
    for name, series in benchmarks.items():
        ax.plot(series, ls=":", lw=1.3, label=name)
    for test_start in pd.to_datetime(result.folds["test_start"]).iloc[1:]:
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

    # 3. Walk-forward validation
    print("\n=== Rolling-window walk-forward ===")
    result = walk_forward.run_walk_forward(
        log_prices, equities, anchor, features,
        calibration_years=wf_cfg["calibration_years"],
        test_months=wf_cfg["test_months"],
        embargo_days=wf_cfg["embargo_days"],
        window=cfg["signals"]["zscore_window"],
        risk_aversion=cfg["strategy"]["risk_aversion"],
        transaction_cost=tc,
        risk_free_rate=rf,
        rolling_beta_window=cfg["strategy"]["rolling_beta_window"],
        solver=cfg["strategy"]["solver"],
    )

    print("\n=== Per-fold diagnostics ===")
    print(result.folds.round(3).to_string(index=False))
    print(f"\nFold Sharpe: median {result.folds['sharpe'].median():.3f}, "
          f"range [{result.folds['sharpe'].min():.3f}, "
          f"{result.folds['sharpe'].max():.3f}]")

    # 4. Aggregate performance vs. benchmarks over the stitched test folds
    log_oos = log_prices.loc[result.pnl.index]
    kc1_value = backtest.buy_and_hold_benchmark(log_oos, [anchor], initial_capital, tc)
    ew_value = backtest.buy_and_hold_benchmark(log_oos, equities, initial_capital, tc)

    print("\n=== Aggregate out-of-sample performance (stitched folds) ===")
    report = pd.DataFrame({
        "MPTS walk-forward": metrics.portfolio_metrics(
            result.portfolio_value(initial_capital), rf),
        f"B&H {anchor}": metrics.portfolio_metrics(kc1_value, rf),
        "B&H EW basket": metrics.portfolio_metrics(ew_value, rf),
    })
    print(report.round(4).to_string())

    print("\n=== Trade-level statistics ===")
    for key, value in metrics.trade_metrics(result.trades).items():
        print(f"{key:>20}: {value:.4f}" if isinstance(value, float)
              else f"{key:>20}: {value}")

    # 5. Outputs
    output_dir = Path("reports")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    equity_curves = pd.DataFrame({
        "MPTS_walk_forward": result.portfolio_value(initial_capital),
        f"BH_{anchor}": kc1_value,
        "BH_EW": ew_value,
    })
    equity_curves.to_csv(output_dir / "equity_curves.csv")
    result.folds.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    plot_walk_forward(
        result,
        {f"B&H {anchor}": kc1_value, "B&H EW basket": ew_value},
        initial_capital,
        output_dir / "figures" / "walk_forward_equity.png",
    )
    print(f"\nSaved equity curves, fold table and figure under {output_dir}/")


if __name__ == "__main__":
    main()
