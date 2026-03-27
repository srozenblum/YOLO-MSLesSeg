"""
Integration tests for Config classes.

Focuses on:
- Validation errors (no disk access needed)
- Correct path construction for different k_folds / fold_test combinations
"""

import pytest

from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.constants import (
    EXT_JSON,
    RESULTS_GLOBAL_PREFIX,
    RESULTS_SUFFIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_cv():
    """Model with k_folds=5 (cross-validation mode)."""
    return Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5, enhancement="GC")


@pytest.fixture
def model_single():
    """Model with k_folds=1 (fixed train/test split mode)."""
    return Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=1, enhancement="GC")


# ---------------------------------------------------------------------------
# ConfigEval — validation
# ---------------------------------------------------------------------------

def test_config_eval_invalid_forced_plane(model_cv):
    with pytest.raises(ValueError, match="forced_plane"):
        ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1, forced_plane="diagonal")


def test_config_eval_valid_forced_plane(model_cv):
    # Should not raise
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1, forced_plane="coronal")
    assert cfg.plane == "coronal"


def test_config_eval_plane_defaults_to_model(model_cv):
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1)
    assert cfg.plane == model_cv.plane


# ---------------------------------------------------------------------------
# ConfigEval — path construction (CV mode)
# ---------------------------------------------------------------------------

def test_config_eval_fold_json_name_cv(model_cv):
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1)
    expected_name = f"fold1_axial{RESULTS_SUFFIX}{EXT_JSON}"
    assert cfg.results_fold_json.name == expected_name


def test_config_eval_global_json_name_cv(model_cv):
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1)
    expected_name = f"{RESULTS_GLOBAL_PREFIX}axial{RESULTS_SUFFIX}{EXT_JSON}"
    assert cfg.results_experiment_json.name == expected_name


# ---------------------------------------------------------------------------
# ConfigEval — path construction (k_folds=1 mode)
# ---------------------------------------------------------------------------

def test_config_eval_fold_json_name_single(model_single):
    cfg = ConfigEval(model=model_single, epochs=50, k_folds=1)
    # With k_folds=1, results_fold_json points to the global results file
    expected_name = f"{RESULTS_GLOBAL_PREFIX}axial{RESULTS_SUFFIX}{EXT_JSON}"
    assert cfg.results_fold_json.name == expected_name


def test_config_eval_results_base_dir_contains_epochs(model_cv):
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1)
    assert "50epochs" in str(cfg.results_base_dir)


def test_config_eval_results_base_dir_contains_enhancement(model_cv):
    cfg = ConfigEval(model=model_cv, epochs=50, k_folds=5, fold_test=1)
    assert "GC" in str(cfg.results_base_dir)
