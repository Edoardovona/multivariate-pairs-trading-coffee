"""Unit tests for the daily allocation optimizer (SciPy backend).

The tests use the open-source SLSQP backend so they run without a Gurobi
license; the Gurobi path solves the identical formulation.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import optimize_allocations


@pytest.fixture
def problem():
    equities = ["SJM", "KO"]
    triggered = {"SJM": 1, "KO": -1}
    betas = pd.Series({"SJM": 0.28, "KO": -0.04})
    cov = pd.DataFrame([[4e-4, 1e-4], [1e-4, 2e-4]])
    covariances = {s: cov.copy() for s in equities}
    time_to_mean = {"SJM": 5.0, "KO": 4.0}
    avg_return_equity = {"SJM": 2e-4, "KO": 1e-4}
    return triggered, betas, covariances, time_to_mean, avg_return_equity


def test_empty_signals_return_empty():
    assert optimize_allocations({}, 1.0, pd.Series(dtype=float), {}, {}, 0.0, {}) == {}


def test_weights_within_bounds(problem):
    triggered, betas, covariances, ttm, avg_eq = problem
    weights = optimize_allocations(
        triggered, 1.0, betas, covariances, ttm, 1e-4, avg_eq, solver="scipy"
    )
    for stock in triggered:
        assert 0.0 <= weights[stock] <= 1.0


def test_no_leverage_constraint(problem):
    triggered, betas, covariances, ttm, avg_eq = problem
    available = 0.3
    weights = optimize_allocations(
        triggered, available, betas, covariances, ttm, 1e-4, avg_eq, solver="scipy"
    )
    # Long legs alone must not exceed the available capital
    assert sum(weights.values()) <= available + 1e-6


def test_high_risk_aversion_shrinks_positions(problem):
    triggered, betas, covariances, ttm, avg_eq = problem
    small = optimize_allocations(
        triggered, 1.0, betas, covariances, ttm, 1e-4, avg_eq,
        risk_aversion=100.0, solver="scipy",
    )
    large = optimize_allocations(
        triggered, 1.0, betas, covariances, ttm, 1e-4, avg_eq,
        risk_aversion=0.1, solver="scipy",
    )
    assert sum(small.values()) <= sum(large.values()) + 1e-6