"""Unit tests for the expanding-window walk-forward fold structure."""

import pandas as pd

from src.walk_forward import walk_forward_folds

INDEX = pd.bdate_range("2016-01-01", periods=1000)


def test_train_windows_expand_from_origin():
    folds = walk_forward_folds(INDEX, min_train_size=400, test_size=100)
    assert len(folds) >= 2
    train_lengths = [len(train) for train, _ in folds]
    for train, _ in folds:
        assert train[0] == INDEX[0]  # anchored at the sample start
    assert train_lengths == sorted(train_lengths)
    assert train_lengths[1] - train_lengths[0] == 100  # grows by one test window


def test_test_windows_are_contiguous_and_disjoint():
    folds = walk_forward_folds(INDEX, min_train_size=400, test_size=100)
    for (_, test_a), (_, test_b) in zip(folds, folds[1:]):
        assert test_a[-1] < test_b[0]
    covered = sum(len(test) for _, test in folds)
    assert covered <= len(INDEX) - 400


def test_no_overlap_between_train_and_test():
    folds = walk_forward_folds(INDEX, min_train_size=400, test_size=100, embargo=10)
    for train, test in folds:
        assert train[-1] < test[0]
        # embargo: at least 10 trading days between train end and test start
        gap = INDEX.get_loc(test[0]) - INDEX.get_loc(train[-1]) - 1
        assert gap >= 10


def test_short_final_stub_is_discarded():
    folds = walk_forward_folds(
        INDEX, min_train_size=900, test_size=200, min_test_size=150
    )
    # only 100 days remain after the train window -> below min_test_size
    assert folds == []
