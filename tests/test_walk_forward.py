"""Unit tests for the rolling-window walk-forward fold structure."""

import pandas as pd

from src.walk_forward import walk_forward_folds

INDEX = pd.bdate_range("2016-01-01", periods=1000)


def test_calibration_windows_have_fixed_length_and_roll_forward():
    folds = walk_forward_folds(INDEX, calibration_size=400, test_size=100)
    assert len(folds) >= 2
    for cal, _ in folds:
        assert len(cal) == 400  # fixed-length rolling window
    starts = [cal[0] for cal, _ in folds]
    assert starts == sorted(starts) and starts[0] == INDEX[0]
    # window slides by exactly one test window per fold
    assert INDEX.get_loc(folds[1][0][0]) - INDEX.get_loc(folds[0][0][0]) == 100


def test_test_windows_are_contiguous_and_disjoint():
    folds = walk_forward_folds(INDEX, calibration_size=400, test_size=100)
    for (_, test_a), (_, test_b) in zip(folds, folds[1:]):
        assert test_a[-1] < test_b[0]
    covered = sum(len(test) for _, test in folds)
    assert covered <= len(INDEX) - 400


def test_no_overlap_between_calibration_and_test():
    folds = walk_forward_folds(INDEX, calibration_size=400, test_size=100, embargo=10)
    for cal, test in folds:
        assert cal[-1] < test[0]
        # embargo: at least 10 trading days between calibration end and test start
        gap = INDEX.get_loc(test[0]) - INDEX.get_loc(cal[-1]) - 1
        assert gap >= 10


def test_short_final_stub_is_discarded():
    folds = walk_forward_folds(
        INDEX, calibration_size=900, test_size=200, min_test_size=150
    )
    # only 100 days remain after the calibration window -> below min_test_size
    assert folds == []


def test_select_basket_prefers_cointegrated_positive_beta():
    import numpy as np
    from src.walk_forward import select_basket

    rng = np.random.default_rng(7)
    n = 600
    anchor = np.cumsum(rng.normal(0, 0.01, n)) + 5.0
    idx = pd.bdate_range("2016-01-01", periods=n)
    log_prices = pd.DataFrame(
        {
            "KC1": anchor,
            "COINT": anchor + rng.normal(0, 0.02, n),        # cointegrated, beta > 0
            "INVERSE": -anchor + rng.normal(0, 0.02, n) + 12,  # cointegrated, beta < 0
            "WALK": np.cumsum(rng.normal(0, 0.01, n)) + 3.0,   # independent random walk
        },
        index=idx,
    )
    basket = select_basket(log_prices, "KC1", ["COINT", "INVERSE", "WALK"], n_pairs=2)
    assert basket[0] == "COINT"
    assert "INVERSE" not in basket  # negative hedge ratio discarded
