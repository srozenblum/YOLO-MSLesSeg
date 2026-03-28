"""Unit tests for ConfigTrain validation and path construction."""

import re

import pytest

from yolo_mslesseg.configs.ConfigTrain import ConfigTrain
from yolo_mslesseg.utils.Model import Model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_cv() -> Model:
    """Model with k_folds=5 (cross-validation mode)."""
    return Model(plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=5)


@pytest.fixture(scope="module")
def model_single() -> Model:
    """Model with k_folds=1 (fixed train/test split mode)."""
    return Model(plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=1)


# ---------------------------------------------------------------------------
# TestConfigTrainValidation
# ---------------------------------------------------------------------------

class TestConfigTrainValidation:
    def test_cv_mode_fold_test_none_raises(self, model_cv):
        with pytest.raises(ValueError):
            ConfigTrain(model=model_cv, epochs=50, fold_test=None)

    def test_single_fold_fold_test_set_raises(self, model_single):
        with pytest.raises(ValueError):
            ConfigTrain(model=model_single, epochs=50, fold_test=1)

    def test_cv_mode_fold_test_set_does_not_raise(self, model_cv):
        ConfigTrain(model=model_cv, epochs=50, fold_test=1)

    def test_single_fold_fold_test_none_does_not_raise(self, model_single):
        ConfigTrain(model=model_single, epochs=50, fold_test=None)


# ---------------------------------------------------------------------------
# TestConfigTrainPaths
# ---------------------------------------------------------------------------

class TestConfigTrainPaths:
    def test_cv_model_path_contains_fold_weights(self, model_cv):
        cfg = ConfigTrain(model=model_cv, epochs=50, fold_test=1)
        assert "fold1/weights/best.pt" in str(cfg.model_path)

    def test_cv_yaml_path_filename_contains_fold(self, model_cv):
        cfg = ConfigTrain(model=model_cv, epochs=50, fold_test=1)
        assert "fold1" in cfg.yaml_path.name

    def test_cv_fold_train_dir_contains_train_fold(self, model_cv):
        cfg = ConfigTrain(model=model_cv, epochs=50, fold_test=1)
        assert "train_fold1" in str(cfg.fold_train_dir)

    def test_cv_fold_test_dir_contains_test_fold(self, model_cv):
        cfg = ConfigTrain(model=model_cv, epochs=50, fold_test=1)
        assert "test_fold1" in str(cfg.fold_test_dir)

    def test_single_model_path_has_weights_but_no_fold_subdir(self, model_single):
        cfg = ConfigTrain(model=model_single, epochs=50, fold_test=None)
        assert "weights/best.pt" in str(cfg.model_path)
        # No fold<N> subdirectory — none of the path parts should match "fold<digit(s)>"
        assert not any(re.fullmatch(r"fold\d+", part) for part in cfg.model_path.parts)

    def test_single_yaml_path_filename_has_no_fold_suffix(self, model_single):
        cfg = ConfigTrain(model=model_single, epochs=50, fold_test=None)
        # No "_fold<N>" suffix in the filename (unlike CV mode which appends _fold1, etc.)
        assert not re.search(r"_fold\d+", cfg.yaml_path.name)

    def test_cv_train_output_dir_contains_plane(self, model_cv):
        cfg = ConfigTrain(model=model_cv, epochs=50, fold_test=1)
        assert model_cv.plane in str(cfg.train_output_dir)

    def test_single_train_output_dir_contains_plane(self, model_single):
        cfg = ConfigTrain(model=model_single, epochs=50, fold_test=None)
        assert model_single.plane in str(cfg.train_output_dir)
