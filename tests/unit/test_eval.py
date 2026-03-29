"""Unit tests for pure helper functions in eval.py and average_folds.py."""

import json

import numpy as np
import pytest

from yolo_mslesseg.scripts.eval import compute_averages
from yolo_mslesseg.scripts.average_folds import aggregate_fold_metrics, compute_experiment_summary


# ---------------------------------------------------------------------------
# compute_averages
# ---------------------------------------------------------------------------

class TestComputeAverages:
    def test_empty_dict_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_averages({})

    def test_single_metric_single_value_has_media_and_std_keys(self):
        result = compute_averages({"DSC": [0.8]})
        assert "media" in result["DSC"]
        assert "std" in result["DSC"]

    def test_single_metric_multiple_values_correct_stats(self):
        values = [0.6, 0.7, 0.8]
        result = compute_averages({"DSC": values})
        expected_media = float(np.round(np.mean(values), 3))
        expected_std = float(np.round(np.std(values, ddof=1), 3))
        assert result["DSC"]["media"] == pytest.approx(expected_media, abs=1e-3)
        assert result["DSC"]["std"] == pytest.approx(expected_std, abs=1e-3)

    def test_multiple_metrics_all_keys_present(self):
        metrics = {"DSC": [0.8], "AUC": [0.9], "Precision": [0.7], "Recall": [0.75]}
        result = compute_averages(metrics)
        assert set(result.keys()) == {"DSC", "AUC", "Precision", "Recall"}

    def test_identical_values_std_is_zero(self):
        result = compute_averages({"DSC": [0.5, 0.5, 0.5]})
        assert result["DSC"]["std"] == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# aggregate_fold_metrics
# ---------------------------------------------------------------------------

class TestAggregateFoldMetrics:
    def test_fold_format_appends_media_value(self, tmp_path):
        f = tmp_path / "fold1.json"
        f.write_text(json.dumps({"DSC": {"media": 0.8, "std": 0.05}}))
        total = {}
        aggregate_fold_metrics(total, f)
        assert total == {"DSC": [0.8]}

    def test_patient_format_appends_scalar(self, tmp_path):
        f = tmp_path / "patient.json"
        f.write_text(json.dumps({"DSC": 0.75}))
        total = {}
        aggregate_fold_metrics(total, f)
        assert total == {"DSC": [0.75]}

    def test_multiple_metrics_all_accumulated(self, tmp_path):
        f = tmp_path / "fold1.json"
        f.write_text(json.dumps({"DSC": {"media": 0.8, "std": 0.0}, "AUC": {"media": 0.9, "std": 0.0}}))
        total = {}
        aggregate_fold_metrics(total, f)
        assert set(total.keys()) == {"DSC", "AUC"}

    def test_calling_twice_grows_list_to_length_two(self, tmp_path):
        f = tmp_path / "fold1.json"
        f.write_text(json.dumps({"DSC": {"media": 0.8, "std": 0.0}}))
        total = {}
        aggregate_fold_metrics(total, f)
        aggregate_fold_metrics(total, f)
        assert len(total["DSC"]) == 2


# ---------------------------------------------------------------------------
# compute_experiment_summary
# ---------------------------------------------------------------------------

class TestComputeExperimentSummary:
    def test_single_fold_single_metric_has_media_and_std_keys(self):
        result = compute_experiment_summary({"DSC": [0.8]})
        assert "media" in result["DSC"]
        assert "std" in result["DSC"]

    def test_two_folds_single_metric_media_is_mean(self):
        values = [0.6, 0.8]
        result = compute_experiment_summary({"DSC": values})
        expected_media = float(np.round(np.mean(values), 3))
        assert result["DSC"]["media"] == pytest.approx(expected_media, abs=1e-3)

    def test_multiple_metrics_all_keys_present(self):
        fold_metrics = {"DSC": [0.8, 0.9], "AUC": [0.7, 0.75], "Precision": [0.6, 0.65], "Recall": [0.5, 0.55]}
        result = compute_experiment_summary(fold_metrics)
        assert set(result.keys()) == {"DSC", "AUC", "Precision", "Recall"}

    def test_identical_values_std_is_zero(self):
        result = compute_experiment_summary({"DSC": [0.5, 0.5, 0.5]})
        assert result["DSC"]["std"] == pytest.approx(0.0, abs=1e-3)
