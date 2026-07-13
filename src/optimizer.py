"""Bi-objective portfolio optimizer (expected profit vs. lambda-weighted risk).

Solves, each day a set of signals is triggered:

    max_W  sum_n W_n . (EP_n * [1, -1])  -  lambda * sum_n W_n . Sigma~_n . W_n'

    s.t.   0 <= W_long <= 1,   -1 <= W_short <= 0        (per pair)
           sum_n (W_long - W_short) <= available capital  (no leverage)
           sum_e beta_e (W_long + W_short) = 0            (beta neutrality)

where Sigma~ is the pair covariance with sign-flipped off-diagonal terms so
long and short legs contribute with opposite signs.

Two solver backends are provided:

* **Gurobi** (``gurobipy``) — used in the original study; requires a license.
* **SciPy SLSQP** — open-source fallback so the project is fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gurobipy as gp

    HAS_GUROBI = True
except ImportError: 
    HAS_GUROBI = False

from scipy.optimize import minimize


def _flip_off_diagonal(cov: pd.DataFrame) -> np.ndarray:
    flipped = cov.values.copy()
    flipped[0, 1] = -flipped[0, 1]
    flipped[1, 0] = -flipped[1, 0]
    return flipped


def optimize_allocations(
    triggered_signals: dict[str, int],
    available_capital: float,
    betas: dict[str, float] | pd.Series,
    covariances: dict[str, pd.DataFrame],
    time_to_mean: dict[str, float],
    avg_return_anchor: float,
    avg_return_equity: dict[str, float],
    risk_aversion: float = 1.0,
    solver: str = "auto",
) -> dict[str, float]:
    """Solve the daily allocation problem for the triggered pairs.

    Parameters
    ----------
    triggered_signals:
        Mapping ``equity -> direction`` (+1 long spread, -1 short spread).
    available_capital:
        Fraction of the portfolio not already committed (0 to 1).
    betas:
        Current rolling betas of each equity w.r.t. the anchor.
    covariances:
        Per-pair 2x2 covariance matrices ``[anchor, equity residual]``.
    time_to_mean:
        Mean-reversion speed proxy per pair (average holding days).
    avg_return_anchor, avg_return_equity:
        In-sample average daily log-returns entering the expected-profit term.
    risk_aversion:
        The lambda coefficient weighting the variance penalty.
    solver:
        ``"gurobi"``, ``"scipy"`` or ``"auto"`` (Gurobi when available).

    Returns
    -------
    dict
        Long-leg weight per triggered equity (0 when the solver abstains).
    """
    if not triggered_signals:
        return {}
    if solver == "auto":
        solver = "gurobi" if HAS_GUROBI else "scipy"
    if solver == "gurobi" and not HAS_GUROBI:
        raise RuntimeError("gurobipy is not installed; use solver='scipy'")

    if solver == "gurobi":
        return _solve_gurobi(
            triggered_signals, available_capital, betas, covariances,
            time_to_mean, avg_return_anchor, avg_return_equity, risk_aversion,
        )
    return _solve_scipy(
        triggered_signals, available_capital, betas, covariances,
        time_to_mean, avg_return_anchor, avg_return_equity, risk_aversion,
    )


def _solve_gurobi(
    triggered_signals, available_capital, betas, covariances,
    time_to_mean, avg_return_anchor, avg_return_equity, risk_aversion,
) -> dict[str, float]:
    model = gp.Model("mpts_allocation")
    model.setParam("OutputFlag", 0)

    w_long = {s: model.addVar(lb=0, ub=1, name=f"long_{s}") for s in triggered_signals}
    w_short = {s: model.addVar(lb=-1, ub=0, name=f"short_{s}") for s in triggered_signals}
    model.update()

    profit = gp.LinExpr()
    for stock, direction in triggered_signals.items():
        mr = time_to_mean[stock]
        profit += direction * w_long[stock] * (mr * avg_return_equity[stock])
        profit += direction * w_short[stock] * (mr * avg_return_anchor)

    risk = gp.QuadExpr()
    for stock in triggered_signals:
        cov = _flip_off_diagonal(covariances[stock])
        legs = [w_short[stock], w_long[stock]]  # column order [anchor, equity]
        for r in range(2):
            for c in range(2):
                risk += legs[r] * cov[r, c] * legs[c]

    model.setObjective(profit - risk_aversion * risk, gp.GRB.MAXIMIZE)
    model.addConstr(
        gp.quicksum(w_long[s] - w_short[s] for s in triggered_signals)
        <= available_capital,
        name="no_leverage",
    )
    model.addConstr(
        gp.quicksum(betas[s] * (w_long[s] + w_short[s]) for s in triggered_signals)
        == 0,
        name="beta_neutrality",
    )
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        return {s: w_long[s].X for s in triggered_signals}
    return {s: 0.0 for s in triggered_signals}


def _solve_scipy(
    triggered_signals, available_capital, betas, covariances,
    time_to_mean, avg_return_anchor, avg_return_equity, risk_aversion,
) -> dict[str, float]:
    stocks = list(triggered_signals)
    n = len(stocks)
    # decision vector: [w_long_1..n, w_short_1..n]

    # Daily log-returns put the raw objective on a ~1e-4 scale, below
    # SLSQP's default step tolerance — rescale so the solver actually moves.
    scale = 1e4

    def objective(x: np.ndarray) -> float:
        w_long, w_short = x[:n], x[n:]
        profit = 0.0
        risk = 0.0
        for i, stock in enumerate(stocks):
            direction = triggered_signals[stock]
            mr = time_to_mean[stock]
            profit += direction * w_long[i] * (mr * avg_return_equity[stock])
            profit += direction * w_short[i] * (mr * avg_return_anchor)
            cov = _flip_off_diagonal(covariances[stock])
            legs = np.array([w_short[i], w_long[i]])
            risk += legs @ cov @ legs
        return -scale * (profit - risk_aversion * risk)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: available_capital - np.sum(x[:n]) + np.sum(x[n:]),
        },
    ]
    # A beta-neutrality constraint with an all-zero gradient is degenerate
    # (always satisfied) and makes SLSQP's LSQ subproblem singular — skip it.
    if any(abs(betas[s]) > 1e-12 for s in stocks):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x: sum(
                    betas[stocks[i]] * (x[i] + x[n + i]) for i in range(n)
                ),
            }
        )
    bounds = [(0.0, 1.0)] * n + [(-1.0, 0.0)] * n
    result = minimize(
        objective,
        x0=np.zeros(2 * n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if not result.success:
        return {s: 0.0 for s in stocks}
    return {s: float(result.x[i]) for i, s in enumerate(stocks)}