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

    Computed in closed form from rolling variances/covariances (normal
    equations of the two-regressor OLS), which is vectorised and orders of
    magnitude faster than per-day statsmodels fits. Windows end at t-1, so
    the beta used on day t only reflects information available at trade
    time; the beta active at entry is locked for the life of each position.
    """
    proxy = next(f for f in features if f != anchor)
    x1, x2 = log_prices[anchor], log_prices[proxy]
    var1 = x1.rolling(window).var()
    var2 = x2.rolling(window).var()
    cov12 = x1.rolling(window).cov(x2)
    determinant = var1 * var2 - cov12**2

    betas = pd.DataFrame(index=log_prices.index, columns=equities, dtype=float)
    for stock in equities:
        y = log_prices[stock]
        cov1y = x1.rolling(window).cov(y)
        cov2y = x2.rolling(window).cov(y)
        betas[stock] = (var2 * cov1y - cov12 * cov2y) / determinant
    return betas.shift(1).ffill().fillna(0.0)


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