"""
Unit tests for pure functions in reconstruct_volume.py.

Covers: validate_slice, insert_slice, extract_png_indices,
and load_and_preprocess_image.
"""

import numpy as np
import pytest
from PIL import Image

from yolo_mslesseg.scripts.reconstruct_volume import (
    load_and_preprocess_image,
    extract_png_indices,
    insert_slice,
    needs_reconstruction,
    validate_slice,
)


# Synthetic volume shape: (x=10, y=12, z=14)
SHAPE = (10, 12, 14)


# ---------------------------------------------------------------------------
# validate_slice
# ---------------------------------------------------------------------------


class TestValidateSlice:
    # --- axial ---

    def test_axial_valid_first_index(self):
        validate_slice(0, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "axial")

    def test_axial_valid_last_index(self):
        validate_slice(SHAPE[2] - 1, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "axial")

    def test_axial_index_at_bound_raises(self):
        with pytest.raises(ValueError):
            validate_slice(SHAPE[2], np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "axial")

    def test_axial_negative_index_raises(self):
        with pytest.raises(ValueError):
            validate_slice(-1, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "axial")

    def test_axial_correct_shape_passes(self):
        validate_slice(0, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "axial")

    def test_axial_transposed_shape_raises(self):
        # (y, x) instead of (x, y)
        with pytest.raises(ValueError):
            validate_slice(0, np.zeros((SHAPE[1], SHAPE[0])), SHAPE, "axial")

    # --- coronal ---

    def test_coronal_valid_index_and_shape(self):
        validate_slice(0, np.zeros((SHAPE[0], SHAPE[2])), SHAPE, "coronal")

    def test_coronal_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_slice(SHAPE[1], np.zeros((SHAPE[0], SHAPE[2])), SHAPE, "coronal")

    def test_coronal_wrong_shape_raises(self):
        # axial shape passed for coronal
        with pytest.raises(ValueError):
            validate_slice(0, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "coronal")

    # --- sagital ---

    def test_sagital_valid_index_and_shape(self):
        validate_slice(0, np.zeros((SHAPE[1], SHAPE[2])), SHAPE, "sagital")

    def test_sagital_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            validate_slice(SHAPE[0], np.zeros((SHAPE[1], SHAPE[2])), SHAPE, "sagital")

    def test_sagital_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            validate_slice(0, np.zeros((SHAPE[0], SHAPE[1])), SHAPE, "sagital")


# ---------------------------------------------------------------------------
# insert_slice
# ---------------------------------------------------------------------------


class TestInsertSlice:
    def test_axial_inserts_at_correct_z(self):
        vol = np.zeros(SHAPE)
        slice_data = np.ones((SHAPE[0], SHAPE[1]))
        insert_slice(vol, slice_data, 7, "axial")
        np.testing.assert_array_equal(vol[:, :, 7], slice_data)

    def test_axial_does_not_affect_adjacent_slices(self):
        vol = np.zeros(SHAPE)
        insert_slice(vol, np.ones((SHAPE[0], SHAPE[1])), 0, "axial")
        assert vol[:, :, 1].sum() == 0

    def test_coronal_inserts_at_correct_y(self):
        vol = np.zeros(SHAPE)
        slice_data = np.ones((SHAPE[0], SHAPE[2]))
        insert_slice(vol, slice_data, 5, "coronal")
        np.testing.assert_array_equal(vol[:, 5, :], slice_data)

    def test_sagital_inserts_at_correct_x(self):
        vol = np.zeros(SHAPE)
        slice_data = np.ones((SHAPE[1], SHAPE[2]))
        insert_slice(vol, slice_data, 3, "sagital")
        np.testing.assert_array_equal(vol[3, :, :], slice_data)

    def test_multiple_axial_insertions_are_independent(self):
        vol = np.zeros(SHAPE)
        insert_slice(vol, np.ones((SHAPE[0], SHAPE[1])), 2, "axial")
        insert_slice(vol, np.full((SHAPE[0], SHAPE[1]), 2.0), 9, "axial")
        assert vol[:, :, 2].sum() == SHAPE[0] * SHAPE[1]
        assert vol[:, :, 9].sum() == SHAPE[0] * SHAPE[1] * 2.0

    def test_insert_does_not_touch_other_planes(self):
        vol = np.zeros(SHAPE)
        insert_slice(vol, np.ones((SHAPE[0], SHAPE[1])), 0, "axial")
        # Only the inserted axial slice should be fully ones
        assert vol[:, :, 0].sum() == SHAPE[0] * SHAPE[1]
        assert vol[:, :, 1].sum() == 0


# ---------------------------------------------------------------------------
# extract_png_indices
# ---------------------------------------------------------------------------


class TestExtractPngIndices:
    def test_returns_list_sorted_by_index(self, tmp_path):
        for i in [10, 3, 77]:
            (tmp_path / f"P1_{i}.png").touch()
        result = extract_png_indices(tmp_path)
        indices = [idx for _, idx in result]
        assert indices == sorted(indices)

    def test_extracts_correct_index_value(self, tmp_path):
        (tmp_path / "P1_42.png").touch()
        result = extract_png_indices(tmp_path)
        assert result[0][1] == 42

    def test_returns_all_valid_png_files(self, tmp_path):
        for i in range(5):
            (tmp_path / f"P1_{i}.png").touch()
        result = extract_png_indices(tmp_path)
        assert len(result) == 5

    def test_nonexistent_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_png_indices(tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_png_indices(tmp_path)

    def test_no_valid_pngs_raises(self, tmp_path):
        (tmp_path / "not_a_mask.txt").touch()
        with pytest.raises(FileNotFoundError):
            extract_png_indices(tmp_path)

    def test_index_zero_extracted_correctly(self, tmp_path):
        (tmp_path / "P1_0.png").touch()
        result = extract_png_indices(tmp_path)
        assert result[0][1] == 0

    def test_large_index_extracted_correctly(self, tmp_path):
        (tmp_path / "P1_999.png").touch()
        result = extract_png_indices(tmp_path)
        assert result[0][1] == 999


# ---------------------------------------------------------------------------
# load_and_preprocess_image
# ---------------------------------------------------------------------------


class TestLoadAndPreprocessImage:
    def test_grayscale_255_binarized_to_float(self, tmp_path):
        img_path = tmp_path / "mask.png"
        arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(img_path)
        result = load_and_preprocess_image(img_path)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_all_zeros_stays_zero(self, tmp_path):
        img_path = tmp_path / "zeros.png"
        arr = np.zeros((4, 4), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(img_path)
        result = load_and_preprocess_image(img_path)
        assert result.sum() == 0

    def test_rgb_converted_to_2d(self, tmp_path):
        img_path = tmp_path / "rgb.png"
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        Image.fromarray(arr, mode="RGB").save(img_path)
        result = load_and_preprocess_image(img_path)
        assert result.ndim == 2

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_image(tmp_path / "missing.png")

    def test_result_is_float32(self, tmp_path):
        img_path = tmp_path / "mask.png"
        arr = np.array([[0, 255]], dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(img_path)
        result = load_and_preprocess_image(img_path)
        # After binarization (max > 1), dtype is float32
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# needs_reconstruction
# ---------------------------------------------------------------------------


class TestNeedsReconstruction:
    def test_returns_true_when_file_does_not_exist(self, tmp_path):
        assert needs_reconstruction(tmp_path / "missing.nii.gz") is True

    def test_returns_true_when_file_is_empty(self, tmp_path):
        empty_file = tmp_path / "empty.nii.gz"
        empty_file.touch()
        assert needs_reconstruction(empty_file) is True

    def test_returns_false_when_file_has_content(self, tmp_path):
        existing_file = tmp_path / "volume.nii.gz"
        existing_file.write_bytes(b"data")
        assert needs_reconstruction(existing_file) is False
