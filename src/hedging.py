"""Commodity-sensitivity estimation for the beta-neutrality constraint.

Each equity's log-price is regressed on the coffee anchor (KC1) and a
consumer-staples sector proxy (XLP):

    p_i = alpha_i + beta_{i,KC1} * p_KC1 + beta_{i,XLP} * p_XLP + eps_i

Controlling for the sector isolates the residual coffee sensitivity
beta_{i,KC1}, which the optimizer uses to immunize the portfolio against
commodity price swings.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def estimate_static_betas(
    log_prices: pd.DataFrame,
    equities: list[str],
    features: list[str],
    anchor: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample multivariate OLS per equity (descriptive purpose).

    Returns
    -------
    (coefficients, residuals)
        ``coefficients`` has one row per equity (alpha, one beta per
        feature, adj. R-squared); ``residuals`` holds the regression
        residuals used for the pair covariance matrices.
    """
    rows = []
    residuals = pd.DataFrame(index=log_prices.index)
    for stock in equities:
        frame = log_prices[[stock] + features].dropna()
        y = frame[stock]
        x = sm.add_constant(frame[features])
        model = sm.OLS(y, x).fit()
        residuals[stock] = model.resid
        row = {"asset": stock, "alpha": model.params["const"]}
        row.update({f"beta_{f}": model.params[f] for f in features})
        row["adj_r_squared"] = model.rsquared_adj
        rows.append(row)
    coefficients = pd.DataFrame(rows).set_index("asset")
    return coefficients, residuals


def rolling_betas(
    log_prices: pd.DataFrame,
    equities: list[str],
    features: list[str],
    anchor: str,
    window: int = 252,
) -> pd.DataFrame:
    """One-year rolling OLS betas w.r.t. the anchor, used in the backtest.

    The beta active at entry is locked for the life of each position so
    that the hedge reflects only information available at trade time.
    """
    betas = pd.DataFrame(index=log_prices.index, columns=equities, dtype=float)
    for t in range(window, len(log_prices)):
        for stock in equities:
            # Subset before dropna so unrelated assets with missing history
            # cannot empty the estimation window.
            window_data = (
                log_prices[[stock] + features].iloc[t - window : t].dropna()
            )
            if len(window_data) < window // 2:
                continue
            model = sm.OLS(
                window_data[stock], sm.add_constant(window_data[features])
            ).fit()
            betas.iat[t, betas.columns.get_loc(stock)] = model.params[anchor]
    return betas.ffill().fillna(0.0)


def pair_covariances(
    anchor_returns: pd.Series,
    residual_returns: pd.DataFrame,
    equities: list[str],
) -> dict[str, pd.DataFrame]:
    """2x2 covariance matrix [anchor, equity residual] per pair."""
    covariances = {}
    for stock in equities:
        frame = pd.concat([anchor_returns, residual_returns[stock]], axis=1).dropna()
        covariances[stock] = frame.cov()
    return covariances