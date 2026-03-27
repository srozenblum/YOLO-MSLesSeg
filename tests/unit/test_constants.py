"""Sanity checks for constants.py."""

from yolo_mslesseg.utils.constants import (
    ENHANCEMENTS,
    MODALITIES,
    N_TRAIN_PATIENTS,
    PLANES,
    ANATOMICAL_PLANES,
    RESULTS_GLOBAL_PREFIX,
    RESULTS_SUFFIX,
    SPLIT_TEST,
    SPLIT_TRAIN,
    TIMEPOINTS,
)


def test_anatomical_planes_subset_of_planes():
    assert all(p in PLANES for p in ANATOMICAL_PLANES)


def test_consenso_not_in_anatomical_planes():
    assert "consenso" not in ANATOMICAL_PLANES


def test_consenso_in_planes():
    assert "consenso" in PLANES


def test_enhancements_no_none():
    # None is handled explicitly in code; it must not be in the tuple
    assert None not in ENHANCEMENTS


def test_enhancements_expected_values():
    assert set(ENHANCEMENTS) == {"HE", "CLAHE", "GC", "LT"}


def test_modalities_expected_values():
    assert set(MODALITIES) == {"T1", "T2", "FLAIR"}


def test_splits_distinct():
    assert SPLIT_TRAIN != SPLIT_TEST


def test_n_train_patients():
    assert N_TRAIN_PATIENTS == 53


def test_timepoints_contains_t1():
    assert "T1" in TIMEPOINTS


def test_results_suffix():
    assert RESULTS_SUFFIX == "_results"


def test_results_global_prefix():
    assert RESULTS_GLOBAL_PREFIX == "global_"
