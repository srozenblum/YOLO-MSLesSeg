"""
Unit tests for pure functions in generate_predictions.py.

Covers: combine_predictions, normalize_prediction.
"""

import numpy as np
import pytest

from yolo_mslesseg.scripts.generate_predictions import (
    combine_predictions,
    normalize_prediction,
)


# ---------------------------------------------------------------------------
# normalize_prediction
# ---------------------------------------------------------------------------


class TestNormalizePrediction:
    def test_transposes_shape(self):
        pred = np.zeros((6, 4), dtype=np.uint8)
        result = normalize_prediction(pred)
        assert result.shape == (4, 6)

    def test_zeros_map_to_zero(self):
        pred = np.zeros((5, 5), dtype=np.uint8)
        assert normalize_prediction(pred).max() == 0

    def test_ones_map_to_255(self):
        pred = np.ones((5, 5), dtype=np.uint8)
        assert normalize_prediction(pred).min() == 255

    def test_output_dtype_is_uint8(self):
        pred = np.zeros((4, 4), dtype=np.uint8)
        assert normalize_prediction(pred).dtype == np.uint8

    def test_binary_values_only(self):
        pred = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = normalize_prediction(pred)
        assert set(np.unique(result)).issubset({0, 255})

    def test_does_not_mutate_input(self):
        pred = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        original = pred.copy()
        normalize_prediction(pred)
        np.testing.assert_array_equal(pred, original)

    def test_non_square_shape(self):
        pred = np.ones((3, 7), dtype=np.uint8)
        result = normalize_prediction(pred)
        assert result.shape == (7, 3)


# ---------------------------------------------------------------------------
# combine_predictions
# ---------------------------------------------------------------------------


class TestCombinePredictions:
    def test_empty_list_returns_zeros(self):
        result = combine_predictions([], shape=(4, 6))
        np.testing.assert_array_equal(result, np.zeros((4, 6), dtype=np.uint8))

    def test_single_mask_above_threshold_sets_pixel(self):
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[2, 2] = 0.9
        result = combine_predictions([mask], shape=(4, 4))
        assert result[2, 2] == 1

    def test_single_mask_below_threshold_leaves_zero(self):
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[2, 2] = 0.3
        result = combine_predictions([mask], shape=(4, 4))
        assert result[2, 2] == 0

    def test_threshold_is_exclusive_at_05(self):
        mask = np.full((2, 2), 0.5, dtype=np.float32)
        result = combine_predictions([mask], shape=(2, 2))
        assert result.sum() == 0  # 0.5 is not > 0.5

    def test_union_of_two_disjoint_masks(self):
        mask1 = np.zeros((4, 4), dtype=np.float32)
        mask1[0, 0] = 1.0
        mask2 = np.zeros((4, 4), dtype=np.float32)
        mask2[3, 3] = 1.0
        result = combine_predictions([mask1, mask2], shape=(4, 4))
        assert result[0, 0] == 1
        assert result[3, 3] == 1

    def test_output_is_binary(self):
        mask = np.random.rand(8, 8).astype(np.float32)
        result = combine_predictions([mask], shape=(8, 8))
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_shape_matches_target(self):
        mask = np.ones((3, 3), dtype=np.float32)
        result = combine_predictions([mask], shape=(6, 8))
        assert result.shape == (6, 8)

    def test_output_dtype_is_uint8(self):
        result = combine_predictions([], shape=(4, 4))
        assert result.dtype == np.uint8

    def test_multiple_masks_same_pixel_stays_one(self):
        masks = [np.ones((4, 4), dtype=np.float32) for _ in range(5)]
        result = combine_predictions(masks, shape=(4, 4))
        assert result.max() == 1  # OR — no value exceeds 1
