"""Unit tests for pure utility functions in utils.py."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import yolo_mslesseg.utils.utils as utils_module
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.constants import StageResult
from yolo_mslesseg.utils.utils import (
    AUC,
    DSC,
    build_config_name,
    compute_fold,
    evaluate_results,
    get_patient_slices,
    int_or_percentile,
    list_patients,
    log_fold_status,
    patient_paths,
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


# ---------------------------------------------------------------------------
# build_config_name
# ---------------------------------------------------------------------------

class TestBuildConfigName:
    def test_single_modality_cv_includes_all_components(self):
        """Verifies the canonical name for a single-modality CV model."""
        m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
        assert build_config_name(m, epochs=50) == "FLAIR_50slices_5folds_50epochs"

    def test_single_fold_uses_1fold_label(self):
        """Verifies that k_folds=1 produces '1fold' (not '1folds') in the name."""
        m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=1)
        result = build_config_name(m, epochs=100)
        assert "1fold" in result
        assert "1folds" not in result

    def test_multiple_modalities_are_concatenated(self):
        """Verifies that multiple modalities are joined without a separator."""
        m = Model(plane="axial", num_slices=50, modality=["T1", "T2", "FLAIR"], k_folds=5)
        result = build_config_name(m, epochs=50)
        assert result.startswith("T1T2FLAIR_")

    def test_percentile_num_slices_preserved_as_string(self):
        """Verifies that a percentile num_slices value appears literally in the name."""
        m = Model(plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=5)
        result = build_config_name(m, epochs=50)
        assert "P50slices" in result

    def test_epochs_value_appears_in_result(self):
        """Verifies that the epochs argument is reflected in the output string."""
        m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
        assert "75epochs" in build_config_name(m, epochs=75)


# ---------------------------------------------------------------------------
# compute_fold — edge cases
# ---------------------------------------------------------------------------

class TestComputeFoldEdgeCases:
    def test_patient_beyond_train_range_raises(self):
        """Verifies that a patient ID beyond P53 raises ValueError (not a CV patient)."""
        with pytest.raises(ValueError):
            compute_fold("P54", k_folds=5)

    def test_patient_far_beyond_train_range_raises(self):
        """Verifies that any patient ID above N_TRAIN_PATIENTS raises ValueError."""
        with pytest.raises(ValueError):
            compute_fold("P99", k_folds=5)


# ---------------------------------------------------------------------------
# int_or_percentile — additional edge cases
# ---------------------------------------------------------------------------

class TestIntOrPercentileEdgeCases:
    def test_zero_integer_string(self):
        """Verifies that '0' is parsed as integer 0."""
        assert int_or_percentile("0") == 0

    def test_percentile_p0_is_valid(self):
        """Verifies that 'P0' is accepted as a valid percentile string."""
        assert int_or_percentile("P0") == "P0"

    def test_p_with_no_digits_raises(self):
        """Verifies that 'P' alone (no digit after prefix) raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            int_or_percentile("P")

    def test_empty_string_raises(self):
        """Verifies that an empty string raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            int_or_percentile("")


# ---------------------------------------------------------------------------
# log_fold_status
# ---------------------------------------------------------------------------

class TestLogFoldStatus:
    def test_completed_calls_info_with_correct_message(self):
        """Verifies that COMPLETED logs an info message containing the fold number."""
        mock_logger = MagicMock()
        log_fold_status(mock_logger, StageResult.COMPLETED, fold=3)
        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "3" in message
        assert "🆗" in message

    def test_skipped_calls_skip_with_correct_message(self):
        """Verifies that SKIPPED calls the custom skip level with the fold number."""
        mock_logger = MagicMock()
        log_fold_status(mock_logger, StageResult.SKIPPED, fold=2)
        mock_logger.skip.assert_called_once()
        message = mock_logger.skip.call_args[0][0]
        assert "2" in message
        assert "⏩" in message

    def test_partial_calls_info_with_correct_message(self):
        """Verifies that PARTIAL logs an info message containing the fold number."""
        mock_logger = MagicMock()
        log_fold_status(mock_logger, StageResult.PARTIAL, fold=1)
        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "1" in message
        assert "🔁" in message

    def test_unknown_result_calls_warning(self):
        """Verifies that an unrecognised result value falls through to logger.warning."""
        mock_logger = MagicMock()
        log_fold_status(mock_logger, result=None, fold=4)
        mock_logger.warning.assert_called_once()
        message = mock_logger.warning.call_args[0][0]
        assert "4" in message
        assert "⚠️" in message


# ---------------------------------------------------------------------------
# patient_paths
# ---------------------------------------------------------------------------

class TestPatientPaths:
    """Verifies path composition in patient_paths using a lightweight stub patient.

    k_folds > 1 is used so that patient_base_dir only requires patient.id
    and patient.plane (no split attribute or filesystem access needed).
    """

    @pytest.fixture(scope="class")
    def stub_patient(self):
        """Minimal patient stub with the attributes consumed by patient_paths."""
        return SimpleNamespace(id="P14", plane="axial", modality_str="FLAIR")

    @pytest.fixture(scope="class")
    def model_cv(self):
        """CV model — patient_base_dir resolves fold from patient ID only."""
        return Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)

    @pytest.fixture(scope="class")
    def paths(self, stub_patient, model_cv):
        """Computed paths dict for slice 44 of P14."""
        return patient_paths(patient=stub_patient, model=model_cv, slice_idx=44)

    def test_keys_are_img_pred_gt(self, paths):
        """Verifies that the returned dict has exactly the three expected keys."""
        assert set(paths.keys()) == {"img", "pred", "gt"}

    def test_img_filename_contains_patient_id(self, paths, stub_patient):
        """Verifies that the image filename includes the patient ID."""
        assert stub_patient.id in paths["img"].name

    def test_img_filename_contains_modality_str(self, paths, stub_patient):
        """Verifies that the image filename includes the modality string."""
        assert stub_patient.modality_str in paths["img"].name

    def test_img_filename_contains_slice_idx(self, paths):
        """Verifies that the image filename includes the slice index."""
        assert "44" in paths["img"].name

    def test_pred_filename_contains_patient_id_and_slice(self, paths, stub_patient):
        """Verifies that the prediction mask filename contains patient ID and slice."""
        assert stub_patient.id in paths["pred"].name
        assert "44" in paths["pred"].name

    def test_gt_filename_omits_modality_str(self, paths, stub_patient):
        """Verifies that the GT filename does NOT include the modality string."""
        assert stub_patient.modality_str not in paths["gt"].name

    def test_gt_filename_contains_patient_id_and_slice(self, paths, stub_patient):
        """Verifies that the GT filename contains patient ID and slice index."""
        assert stub_patient.id in paths["gt"].name
        assert "44" in paths["gt"].name

    def test_img_is_under_images_subdir(self, paths):
        """Verifies that the image path sits inside an images/ subdirectory."""
        assert paths["img"].parent.name == "images"

    def test_pred_is_under_pred_masks_subdir(self, paths):
        """Verifies that the prediction path sits inside a pred_masks/ subdirectory."""
        assert paths["pred"].parent.name == "pred_masks"

    def test_gt_is_under_gt_masks_subdir(self, paths):
        """Verifies that the GT path sits inside a GT_masks/ subdirectory."""
        assert paths["gt"].parent.name == "GT_masks"


# ---------------------------------------------------------------------------
# get_patient_slices
# ---------------------------------------------------------------------------

class TestGetPatientSlices:
    """Verifies slice index extraction in get_patient_slices.

    patient_base_dir is monkeypatched so the function operates on a
    controlled tmp_path directory without touching the real dataset.
    """

    @pytest.fixture()
    def stub_patient(self):
        """Minimal patient stub — only id and plane are read before monkeypatching."""
        return SimpleNamespace(id="P1", plane="axial")

    @pytest.fixture()
    def stub_model(self):
        return Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)

    def test_returns_sorted_slice_indices(self, monkeypatch, tmp_path, stub_patient, stub_model):
        """Verifies that slice indices are extracted and returned in ascending order."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        for idx in [12, 3, 5]:
            (images_dir / f"P1_FLAIR_{idx}.png").touch()

        monkeypatch.setattr(utils_module, "patient_base_dir", lambda patient, model: tmp_path)
        result = get_patient_slices(stub_patient, stub_model)
        assert result == [3, 5, 12]

    def test_ignores_non_png_files(self, monkeypatch, tmp_path, stub_patient, stub_model):
        """Verifies that non-PNG files in images/ do not appear in the output."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "P1_FLAIR_7.png").touch()
        (images_dir / "P1_FLAIR_8.txt").touch()
        (images_dir / "P1_FLAIR_9.nii").touch()

        monkeypatch.setattr(utils_module, "patient_base_dir", lambda patient, model: tmp_path)
        result = get_patient_slices(stub_patient, stub_model)
        assert result == [7]

    def test_ignores_filenames_without_integer_suffix(self, monkeypatch, tmp_path, stub_patient, stub_model):
        """Verifies that PNG files whose stem does not end in an integer are skipped."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "P1_FLAIR_10.png").touch()
        (images_dir / "P1_FLAIR_abc.png").touch()

        monkeypatch.setattr(utils_module, "patient_base_dir", lambda patient, model: tmp_path)
        result = get_patient_slices(stub_patient, stub_model)
        assert result == [10]

    def test_empty_images_dir_returns_empty_list(self, monkeypatch, tmp_path, stub_patient, stub_model):
        """Verifies that an images/ directory with no PNG files returns an empty list."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        monkeypatch.setattr(utils_module, "patient_base_dir", lambda patient, model: tmp_path)
        result = get_patient_slices(stub_patient, stub_model)
        assert result == []


# ---------------------------------------------------------------------------
# list_patients
# ---------------------------------------------------------------------------

class TestListPatients:
    def test_returns_patients_in_numeric_order(self, tmp_path):
        """Verifies that patients are sorted by their numeric ID, not lexicographically."""
        for pid in ["P10", "P2", "P1"]:
            (tmp_path / pid).mkdir()
        assert list_patients(tmp_path) == ["P1", "P2", "P10"]

    def test_ignores_dot_prefixed_files(self, tmp_path):
        """Verifies that hidden files such as .DS_Store are excluded from results."""
        (tmp_path / "P1").mkdir()
        (tmp_path / ".DS_Store").touch()
        result = list_patients(tmp_path)
        assert result == ["P1"]
        assert ".DS_Store" not in result

    def test_ignores_tilde_prefixed_files(self, tmp_path):
        """Verifies that temporary files starting with '~' are excluded."""
        (tmp_path / "P1").mkdir()
        (tmp_path / "~lock").touch()
        result = list_patients(tmp_path)
        assert result == ["P1"]

    def test_ignores_tmp_files(self, tmp_path):
        """Verifies that files ending in '.tmp' are excluded."""
        (tmp_path / "P1").mkdir()
        (tmp_path / "session.tmp").touch()
        result = list_patients(tmp_path)
        assert result == ["P1"]

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        """Verifies that a directory with no valid patient entries raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            list_patients(tmp_path)

    def test_only_ignorable_files_raises_file_not_found(self, tmp_path):
        """Verifies that a directory containing only hidden files raises FileNotFoundError."""
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "~lock").touch()
        with pytest.raises(FileNotFoundError):
            list_patients(tmp_path)
