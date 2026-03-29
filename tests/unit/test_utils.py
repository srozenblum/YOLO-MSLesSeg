"""Unit tests for pure utility functions in utils.py."""

import argparse

import numpy as np
import pytest

from yolo_mslesseg.utils.constants import StageResult
from yolo_mslesseg.utils.utils import (
    AUC,
    DSC,
    compute_fold,
    evaluate_results,
    int_or_percentile,
    precision,
    recall,
)


# ---------------------------------------------------------------------------
# compute_fold
# ---------------------------------------------------------------------------

def test_compute_fold_first_patient_k5():
    # P1 should fall in fold 1 with k=5 (first group of ~10 patients)
    assert compute_fold("P1", k_folds=5) == 1


def test_compute_fold_last_patient_k5():
    # P53 should fall in fold 5 with k=5 (last group)
    assert compute_fold("P53", k_folds=5) == 5


def test_compute_fold_k1():
    # With k=1 every patient falls in fold 1
    assert compute_fold("P25", k_folds=1) == 1


def test_compute_fold_k2_symmetry():
    fold_p1 = compute_fold("P1", k_folds=2)
    fold_p53 = compute_fold("P53", k_folds=2)
    assert fold_p1 == 1
    assert fold_p53 == 2


def test_compute_fold_returns_int():
    result = compute_fold("P10", k_folds=5)
    assert isinstance(result, int)


def test_compute_fold_range_k5():
    # All folds must be in [1, k_folds]
    for i in range(1, 54):
        fold = compute_fold(f"P{i}", k_folds=5)
        assert 1 <= fold <= 5


# ---------------------------------------------------------------------------
# int_or_percentile
# ---------------------------------------------------------------------------

def test_int_or_percentile_integer_string():
    assert int_or_percentile("50") == 50


def test_int_or_percentile_integer():
    assert int_or_percentile(50) == 50


def test_int_or_percentile_percentil_uppercase():
    assert int_or_percentile("P50") == "P50"


def test_int_or_percentile_percentil_lowercase():
    assert int_or_percentile("p50") == "P50"


def test_int_or_percentile_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        int_or_percentile("abc")


def test_int_or_percentile_invalid_p_prefix():
    with pytest.raises(argparse.ArgumentTypeError):
        int_or_percentile("Pabc")


# ---------------------------------------------------------------------------
# DSC
# ---------------------------------------------------------------------------

def test_dsc_perfect_overlap():
    y = np.array([1, 1, 0, 0])
    assert DSC(y, y) == pytest.approx(1.0, abs=1e-3)


def test_dsc_no_overlap():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    assert DSC(y_true, y_pred) == pytest.approx(0.0, abs=1e-2)


def test_dsc_partial_overlap():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    # intersection=2, sum=4+2=6 → DSC = 4/6 ≈ 0.667
    assert DSC(y_true, y_pred) == pytest.approx(0.667, abs=1e-2)


def test_dsc_all_zeros():
    y = np.zeros(10)
    # Both empty: numerator=0, denominator≈1e-8 → DSC≈0
    assert DSC(y, y) == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------

def test_precision_perfect():
    y = np.array([1, 1, 0, 0])
    assert precision(y, y) == pytest.approx(1.0, abs=1e-3)


def test_precision_all_false_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert precision(y_true, y_pred) == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

def test_recall_perfect():
    y = np.array([1, 1, 0, 0])
    assert recall(y, y) == pytest.approx(1.0, abs=1e-3)


def test_recall_all_false_negatives():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    assert recall(y_true, y_pred) == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def test_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert AUC(y_true, y_pred) == pytest.approx(1.0, abs=1e-3)


def test_auc_single_class_returns_nan():
    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    result = AUC(y_true, y_pred)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# evaluate_results
# ---------------------------------------------------------------------------

class TestEvaluateResults:
    def test_empty_list_returns_skipped(self):
        assert evaluate_results([]) is StageResult.SKIPPED

    def test_all_skipped_returns_skipped(self):
        assert evaluate_results([StageResult.SKIPPED, StageResult.SKIPPED, StageResult.SKIPPED]) is StageResult.SKIPPED

    def test_all_completed_returns_completed(self):
        assert evaluate_results([StageResult.COMPLETED, StageResult.COMPLETED, StageResult.COMPLETED]) is StageResult.COMPLETED

    def test_mix_completed_and_skipped_returns_partial(self):
        assert evaluate_results([StageResult.COMPLETED, StageResult.SKIPPED, StageResult.COMPLETED]) is StageResult.PARTIAL

    def test_single_completed_returns_completed(self):
        assert evaluate_results([StageResult.COMPLETED]) is StageResult.COMPLETED

    def test_single_skipped_returns_skipped(self):
        assert evaluate_results([StageResult.SKIPPED]) is StageResult.SKIPPED
