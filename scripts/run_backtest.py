"""End-to-end pipeline: screening, calibration, backtest, reporting.

Usage:
    python scripts/run_backtest.py [--config configs/default.yaml]

Steps:
    1. Load close prices (Yahoo CSV or Bloomberg Excel) and log-transform.
    2. Chronological in-sample / out-of-sample split (no look-ahead).
    3. ADF + Engle-Granger screening on the in-sample window.
    4. Threshold calibration (in-sample grid search or manual override).
    5. Rolling beta estimation and pair covariance matrices.
    6. Daily out-of-sample backtest with optimal position sizing.
    7. Performance report vs. buy-and-hold benchmarks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, data, hedging, metrics, signals, stat_tests


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    anchor = cfg["universe"]["anchor"]
    sector_proxy = cfg["universe"]["sector_proxy"]
    equities = cfg["universe"]["equities"]
    features = [anchor, sector_proxy]
    window = cfg["signals"]["zscore_window"]
    tc = cfg["strategy"]["transaction_cost"]
    rf = cfg["strategy"]["risk_free_rate"]
    initial_capital = cfg["backtest"]["initial_capital"]

    # 1. Data
    prices = data.load_prices(cfg["data"]["path"])
    log_prices = data.to_log_prices(prices)
    print(f"Loaded {prices.shape[0]} rows x {prices.shape[1]} assets")

    # 2. Split
    log_is, log_oos, split_date = data.train_test_split(
        log_prices, anchor, cfg["split"]["ratio"]
    )
    print(f"In-sample: {log_is.index[0].date()} -> {split_date.date()} "
          f"({len(log_is)} obs) | Out-of-sample: {len(log_oos)} obs")

    # 3. Screening (in-sample only)
    exclude = set(cfg["universe"]["exclude"]) | {anchor}
    adf_results = stat_tests.adf_screen(log_is, exclude=exclude)
    eg_results = stat_tests.engle_granger_screen(log_is, anchor, exclude=exclude)
    print("\n=== In-sample screening (top 6 by Engle-Granger p-value) ===")
    print(stat_tests.summary_table(log_is, anchor, adf_results, eg_results)
          .head(6).round(4).to_string())

    # 4. Threshold calibration. Two configurations are evaluated, mirroring
    # the report: the in-sample-optimized thresholds (methodologically clean)
    # and a tighter manual band (exploratory, embeds look-ahead) that shows
    # the strategy's sensitivity to threshold regime mismatch.
    if cfg["signals"]["threshold_mode"] == "grid":
        grid_results, _ = signals.grid_search_thresholds(
            log_is, equities, anchor, window=window, risk_free_rate=rf
        )
        open_th = float(grid_results.iloc[0]["open"])
        close_th = float(grid_results.iloc[0]["close"])
        print(f"\nIS-optimized thresholds: open=+/-{open_th}, close=+/-{close_th}")
    else:
        open_th = cfg["signals"]["manual_open"]
        close_th = cfg["signals"]["manual_close"]
        print(f"\nManual thresholds: open=+/-{open_th}, close=+/-{close_th}")

    configurations = {
        f"MPTS ({open_th}/{close_th})": (open_th, close_th),
        "MPTS (manual 1.0/0.6)": (
            cfg["signals"]["manual_open"],
            cfg["signals"]["manual_close"],
        ),
    }

    # 5. Hedging inputs
    _, residuals = hedging.estimate_static_betas(log_prices, equities, features, anchor)
    anchor_returns = log_prices[anchor].diff().dropna()
    covariances = hedging.pair_covariances(
        anchor_returns, residuals.diff().dropna(), equities
    )
    rolling = hedging.rolling_betas(
        log_prices, equities, features, anchor,
        window=cfg["strategy"]["rolling_beta_window"],
    )
    avg_ret_anchor = log_is[anchor].diff().dropna().mean()
    avg_ret_equity = {s: log_is[s].diff().dropna().mean() for s in equities}

    # 6. Backtest each threshold configuration
    results = {}
    for label, (o_th, c_th) in configurations.items():
        time_to_mean = signals.compute_time_to_mean(
            log_is, equities, anchor, o_th, c_th, window=window
        )
        results[label] = backtest.run_strategy(
            log_prices, equities, anchor, split_date,
            rolling, covariances, time_to_mean, avg_ret_anchor, avg_ret_equity,
            open_threshold=o_th, close_threshold=c_th, window=window,
            risk_aversion=cfg["strategy"]["risk_aversion"],
            transaction_cost=tc, risk_free_rate=rf,
            solver=cfg["strategy"]["solver"],
        )
        print(f"{label}: {len(results[label].trades)} trades over the OOS window")

    # 7. Reporting
    kc1_value = backtest.buy_and_hold_benchmark(log_oos, [anchor], initial_capital, tc)
    ew_value = backtest.buy_and_hold_benchmark(log_oos, equities, initial_capital, tc)

    print("\n=== Out-of-sample performance ===")
    columns = {
        label: metrics.portfolio_metrics(res.portfolio_value(initial_capital), rf)
        for label, res in results.items()
    }
    columns[f"B&H {anchor}"] = metrics.portfolio_metrics(kc1_value, rf)
    columns["B&H EW basket"] = metrics.portfolio_metrics(ew_value, rf)
    print(pd.DataFrame(columns).round(4).to_string())

    print("\n=== Trade-level statistics ===")
    trade_report = pd.DataFrame(
        {label: metrics.trade_metrics(res.trades) for label, res in results.items()}
    )
    print(trade_report.round(4).to_string() if not trade_report.empty
          else "No trades triggered.")

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    equity_curves = pd.DataFrame(
        {label: res.portfolio_value(initial_capital) for label, res in results.items()}
        | {f"BH_{anchor}": kc1_value, "BH_EW": ew_value}
    )
    output_path = output_dir / "equity_curves.csv"
    equity_curves.to_csv(output_path)
    print(f"\nEquity curves saved to {output_path}")


if __name__ == "__main__":
    main()
