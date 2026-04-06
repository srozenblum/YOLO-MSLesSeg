"""
Unit tests for combine_volumes in generate_consensus.py.

The function implements voxel-wise majority voting across three binary volumes
(axial, coronal, sagittal) with a fixed threshold of 2 (majority voting).
"""

import numpy as np
import pytest

from yolo_mslesseg.scripts.generate_consensus import combine_volumes


SHAPE = (4, 5, 6)


# ---------------------------------------------------------------------------
# Majority vote (threshold = 2, fixed)
# ---------------------------------------------------------------------------


def test_all_active_gives_ones():
    a = np.ones(SHAPE, dtype=np.uint8)
    np.testing.assert_array_equal(combine_volumes(a, a, a), a)


def test_two_active_gives_ones():
    active = np.ones(SHAPE, dtype=np.uint8)
    inactive = np.zeros(SHAPE, dtype=np.uint8)
    result = combine_volumes(active, active, inactive)
    np.testing.assert_array_equal(result, active)


def test_one_active_gives_zeros():
    active = np.ones(SHAPE, dtype=np.uint8)
    inactive = np.zeros(SHAPE, dtype=np.uint8)
    result = combine_volumes(active, inactive, inactive)
    np.testing.assert_array_equal(result, inactive)


def test_all_inactive_gives_zeros():
    z = np.zeros(SHAPE, dtype=np.uint8)
    np.testing.assert_array_equal(combine_volumes(z, z, z), z)


# ---------------------------------------------------------------------------
# Per-voxel correctness
# ---------------------------------------------------------------------------


def test_per_voxel_voting():
    """Only the voxel with two active planes should be 1."""
    axial = np.zeros(SHAPE, dtype=np.uint8)
    coronal = np.zeros(SHAPE, dtype=np.uint8)
    sagittal = np.zeros(SHAPE, dtype=np.uint8)
    axial[1, 2, 3] = 1
    coronal[1, 2, 3] = 1
    result = combine_volumes(axial, coronal, sagittal)
    assert result[1, 2, 3] == 1
    assert result[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# Output properties
# ---------------------------------------------------------------------------


def test_output_is_binary():
    a = np.ones(SHAPE, dtype=np.uint8)
    b = np.zeros(SHAPE, dtype=np.uint8)
    result = combine_volumes(a, b, a)
    assert set(np.unique(result)).issubset({0, 1})


def test_output_dtype_is_uint8():
    a = np.ones(SHAPE, dtype=np.uint8)
    assert combine_volumes(a, a, a).dtype == np.uint8


def test_output_shape_preserved():
    a = np.zeros(SHAPE, dtype=np.uint8)
    assert combine_volumes(a, a, a).shape == SHAPE
