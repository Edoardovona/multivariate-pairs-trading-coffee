"""Stationarity and cointegration screening.

Pairs trading rests on two statistical prerequisites, both tested on the in-sample window only:

* individual log-price series must be non-stationary, i.e. I(1) (Augmented Dickey-Fuller test);
* the spread between the anchor and a candidate must be stationary (Engle-Granger two-step cointegration test).
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def adf_screen(
    log_prices: pd.DataFrame,
    exclude: set[str] | None = None,
    min_obs: int = 100,
) -> pd.DataFrame:
    """Run the ADF unit-root test on each eligible log-price series.

    Returns a DataFrame sorted by p-value (asset, n_obs, adf_stat, p_value).
    High p-values fail to reject the unit root — the desired outcome for pairs-trading candidates.
    """
    exclude = exclude or set()
    rows = []
    for col in log_prices.columns:
        if col in exclude:
            continue
        series = log_prices[col].dropna()
        if len(series) < min_obs:
            continue
        stat, pval, *_ = adfuller(series)
        rows.append(
            {"asset": col, "n_obs": len(series), "adf_stat": stat, "p_value": pval}
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def engle_granger_screen(
    log_prices: pd.DataFrame,
    target: str,
    exclude: set[str] | None = None,
    min_obs: int = 200,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Engle-Granger cointegration test of every candidate against ``target``.

    Step 1: OLS of the target log-price on the candidate log-price.
    Step 2: ADF test on the regression residuals.

    Returns a DataFrame sorted by p-value with the hedge ratio (beta),
    intercept, R-squared and a boolean significance flag.
    """
    exclude = exclude or set()
    rows = []
    for col in log_prices.columns:
        if col == target or col in exclude:
            continue
        pair = pd.concat([log_prices[target], log_prices[col]], axis=1).dropna()
        if len(pair) < min_obs:
            continue
        y = pair[target]
        x = sm.add_constant(pair[col])
        model = sm.OLS(y, x).fit()
        stat, pval, *_ = adfuller(model.resid)
        rows.append(
            {
                "asset": col,
                "n_obs": len(pair),
                "beta": model.params[col],
                "alpha": model.params["const"],
                "r_squared": model.rsquared,
                "p_value": pval,
                "cointegrated": pval < significance,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def summary_table(
    log_prices: pd.DataFrame,
    target: str,
    adf_results: pd.DataFrame,
    coint_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine descriptive statistics with ADF and Engle-Granger p-values."""
    eligible = coint_results["asset"].tolist()
    subset = log_prices[eligible]
    stats = pd.DataFrame(
        {
            "n_obs": subset.count().astype(int),
            "mean": subset.mean(),
            "std": subset.std(),
            "skew": subset.skew(),
            "kurtosis": subset.kurtosis(),
            "corr_with_target": subset.corrwith(log_prices[target]),
        }
    )
    adf_p = adf_results.set_index("asset")["p_value"].rename("adf_p_value")
    eg_p = coint_results.set_index("asset")["p_value"].rename("eg_p_value")
    return stats.join(adf_p).join(eg_p).sort_values("eg_p_value")