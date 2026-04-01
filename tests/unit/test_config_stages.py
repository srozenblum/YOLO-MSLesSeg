"""Unit tests for ConfigPred, ConfigReconstruction, and ConfigConsensus path construction."""

import re

import pytest

from yolo_mslesseg.configs.ConfigConsensus import ConfigConsensus
from yolo_mslesseg.configs.ConfigPred import ConfigPred
from yolo_mslesseg.configs.ConfigReconstruction import ConfigReconstruction
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient


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
# TestConfigPredPaths
# ---------------------------------------------------------------------------

class TestConfigPredPaths:
    def test_cv_dataset_fold_dir_contains_fold1(self, model_cv):
        cfg = ConfigPred(model=model_cv, epochs=50, fold_test=1)
        assert "fold1" in str(cfg.dataset_fold_dir)

    def test_cv_model_path_contains_fold_weights(self, model_cv):
        cfg = ConfigPred(model=model_cv, epochs=50, fold_test=1)
        assert "fold1/weights/best.pt" in str(cfg.model_path)

    def test_single_dataset_fold_dir_contains_test(self, model_single):
        cfg = ConfigPred(model=model_single, epochs=50)
        assert "test" in str(cfg.dataset_fold_dir)

    def test_single_model_path_has_no_fold_subdir(self, model_single):
        cfg = ConfigPred(model=model_single, epochs=50)
        # No fold<N> subdirectory in the path (model string may contain "1fold")
        assert not any(re.fullmatch(r"fold\d+", part) for part in cfg.model_path.parts)

    def test_cv_no_patient_no_fold_test_raises(self, model_cv):
        with pytest.raises(ValueError):
            ConfigPred(model=model_cv, epochs=50)


# ---------------------------------------------------------------------------
# TestConfigReconstructionPaths
# ---------------------------------------------------------------------------

class TestConfigReconstructionPaths:
    def test_cv_dataset_fold_dir_contains_fold2(self, model_cv):
        cfg = ConfigReconstruction(model=model_cv, epochs=50, fold_test=2)
        assert "fold2" in str(cfg.dataset_fold_dir)

    def test_cv_pred_vols_fold_dir_contains_fold2(self, model_cv):
        cfg = ConfigReconstruction(model=model_cv, epochs=50, fold_test=2)
        assert "fold2" in str(cfg.pred_vols_fold_dir)

    def test_cv_gt_dir_ends_with_train(self, model_cv):
        cfg = ConfigReconstruction(model=model_cv, epochs=50, fold_test=2)
        assert cfg.gt_dir.name == "train"

    def test_single_gt_dir_ends_with_test(self, model_single):
        cfg = ConfigReconstruction(model=model_single, epochs=50)
        assert cfg.gt_dir.name == "test"

    def test_single_pred_vols_fold_dir_contains_test(self, model_single):
        cfg = ConfigReconstruction(model=model_single, epochs=50)
        assert "test" in str(cfg.pred_vols_fold_dir)


# ---------------------------------------------------------------------------
# TestConfigConsensusPaths
# ---------------------------------------------------------------------------

class TestConfigConsensusPaths:
    def test_cv_plane_is_consenso(self, model_cv):
        cfg = ConfigConsensus(model=model_cv, epochs=50, fold_test=1)
        assert cfg.plane == "consensus"

    def test_cv_pred_vols_fold_dir_contains_fold1(self, model_cv):
        cfg = ConfigConsensus(model=model_cv, epochs=50, fold_test=1)
        assert "fold1" in str(cfg.pred_vols_fold_dir)

    def test_single_pred_vols_fold_dir_contains_test(self, model_single):
        cfg = ConfigConsensus(model=model_single, epochs=50)
        assert "test" in str(cfg.pred_vols_fold_dir)

    def test_single_gt_dir_ends_with_test(self, model_single):
        cfg = ConfigConsensus(model=model_single, epochs=50)
        assert cfg.gt_dir.name == "test"

    def test_cv_no_patient_no_fold_test_raises(self, model_cv):
        with pytest.raises(ValueError):
            ConfigConsensus(model=model_cv, epochs=50)
