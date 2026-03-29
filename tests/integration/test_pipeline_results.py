"""
Integration tests that verify real artifacts produced by the full pipeline.

Configuration under test: FLAIR, P50slices, 5folds, 50epochs, no enhancement (Base).
All paths are relative to the repository root; the session fixture in conftest.py
ensures the working directory is set correctly before any test runs.
"""

from pathlib import Path

import pytest

import nibabel as nib
import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.scripts.eval import compute_metrics
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.utils import (
    predicted_volumes_complete,
    read_json,
    trained_model_exists,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_base() -> Model:
    """Base FLAIR model with k_folds=5, no enhancement."""
    return Model(plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=5)


# ---------------------------------------------------------------------------
# TestTrainedModelExists
# ---------------------------------------------------------------------------

class TestTrainedModelExists:
    def test_base_axial_fold1_exists(self, model_base):
        assert trained_model_exists(model_base, epochs=50, fold_test=1) is True

    def test_base_coronal_fold1_exists(self):
        model = Model(plane="coronal", num_slices="P50", modality=["FLAIR"], k_folds=5)
        assert trained_model_exists(model, epochs=50, fold_test=1) is True

    def test_base_sagittal_fold1_exists(self):
        model = Model(plane="sagittal", num_slices="P50", modality=["FLAIR"], k_folds=5)
        assert trained_model_exists(model, epochs=50, fold_test=1) is True

    def test_nonexistent_config_returns_false(self, model_base):
        # epochs=99 were never trained → weights file does not exist
        assert trained_model_exists(model_base, epochs=99, fold_test=1) is False

    def test_fold_test_none_with_cv_model_returns_false(self, model_base):
        # fold_test=None targets the flat (k_folds==1) weight path, absent here
        assert trained_model_exists(model_base, epochs=50, fold_test=None) is False


# ---------------------------------------------------------------------------
# TestPredictedVolumesOnDisk
# ---------------------------------------------------------------------------

PRED_VOLS_FOLD1_P1 = Path("pred_vols/Base/FLAIR_P50slices_5folds_50epochs/fold1/P1")


class TestPredictedVolumesOnDisk:
    def test_p1_fold1_axial_volume_exists(self):
        assert (PRED_VOLS_FOLD1_P1 / "P1_axial.nii.gz").exists()

    def test_p1_fold1_coronal_volume_exists(self):
        assert (PRED_VOLS_FOLD1_P1 / "P1_coronal.nii.gz").exists()

    def test_p1_fold1_sagittal_volume_exists(self):
        assert (PRED_VOLS_FOLD1_P1 / "P1_sagittal.nii.gz").exists()

    def test_p1_fold1_consenso_volume_exists(self):
        assert (PRED_VOLS_FOLD1_P1 / "P1_consenso.nii.gz").exists()

    def test_predicted_volumes_complete_for_p1_fold1(self):
        assert predicted_volumes_complete(PRED_VOLS_FOLD1_P1) is True


# ---------------------------------------------------------------------------
# TestFoldMetricsOnDisk
# ---------------------------------------------------------------------------

RESULTS_BASE_DIR = Path("results/Base/FLAIR_P50slices_5folds_50epochs")


class TestFoldMetricsOnDisk:
    def test_global_axial_results_json_exists(self):
        assert (RESULTS_BASE_DIR / "global_axial_results.json").exists()

    def test_global_coronal_results_json_exists(self):
        assert (RESULTS_BASE_DIR / "global_coronal_results.json").exists()

    def test_global_sagittal_results_json_exists(self):
        assert (RESULTS_BASE_DIR / "global_sagittal_results.json").exists()

    def test_global_consenso_results_json_exists(self):
        assert (RESULTS_BASE_DIR / "global_consenso_results.json").exists()

    def test_global_axial_results_has_expected_metric_keys(self):
        data = read_json(RESULTS_BASE_DIR / "global_axial_results.json")
        assert set(data.keys()) == {"DSC", "AUC", "Precision", "Recall"}

    def test_global_axial_dsc_has_media_and_std(self):
        data = read_json(RESULTS_BASE_DIR / "global_axial_results.json")
        assert "media" in data["DSC"]
        assert "std" in data["DSC"]


# ---------------------------------------------------------------------------
# TestComputeMetricsWithRealData
# ---------------------------------------------------------------------------

_PRED_VOL = Path("pred_vols/Base/FLAIR_P50slices_5folds_50epochs/fold1/P1/P1_axial.nii.gz")
_GT_VOL = Path("GT/train/P1/P1_MASK.nii.gz")


class TestComputeMetricsWithRealData:
    @pytest.fixture(scope="class")
    def metrics(self):
        return compute_metrics(gt_vol_path=_GT_VOL, pred_vol_path=_PRED_VOL)

    def test_returns_non_empty_dict(self, metrics):
        assert isinstance(metrics, dict) and len(metrics) > 0

    def test_output_contains_exactly_expected_keys(self, metrics):
        assert set(metrics.keys()) == {"DSC", "AUC", "Precision", "Recall"}

    def test_dsc_is_float_in_unit_range(self, metrics):
        assert isinstance(metrics["DSC"], float)
        assert 0.0 <= metrics["DSC"] <= 1.0

    def test_precision_is_float_in_unit_range(self, metrics):
        assert isinstance(metrics["Precision"], float)
        assert 0.0 <= metrics["Precision"] <= 1.0

    def test_recall_is_float_in_unit_range(self, metrics):
        assert isinstance(metrics["Recall"], float)
        assert 0.0 <= metrics["Recall"] <= 1.0

    def test_mismatched_shapes_returns_empty_dict(self, tmp_path):
        wrong_vol = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), np.eye(4))
        wrong_path = tmp_path / "wrong.nii.gz"
        nib.save(wrong_vol, wrong_path)
        result = compute_metrics(gt_vol_path=_GT_VOL, pred_vol_path=wrong_path)
        assert result == {}
