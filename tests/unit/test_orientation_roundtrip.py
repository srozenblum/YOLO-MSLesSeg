"""
Unit tests for the orientation round-trip through the pipeline:
  extraction (.T) → YOLO prediction → normalize_prediction (.T) → insert_slice

These tests verify that a slice extracted from a 3D volume, passed through YOLO
in image coordinates, and then renormalized back into voxel coordinates can be
inserted into a reconstructed volume without any spatial distortion (no spurious
flips or transpositions). All three anatomical planes are covered.
"""

import numpy as np
import pytest

from yolo_mslesseg.scripts.generate_predictions import normalize_prediction
from yolo_mslesseg.scripts.reconstruct_volume import insert_slice


@pytest.fixture
def volume():
    """Synthetic binary volume with a reproducible random pattern."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=(10, 12, 14), dtype=np.uint8)


# ---------------------------------------------------------------------------
# normalize_prediction round-trip
# ---------------------------------------------------------------------------


def test_axial_normalize_roundtrip(volume):
    """Prediction mask for an axial slice reconstructs the original slice."""
    original = volume[:, :, 5].copy()          # (x=10, y=12)
    yolo_input = original.T                     # (12, 10) — image axes
    pred = (yolo_input > 0).astype(np.uint8)   # simulate perfect YOLO mask
    result = normalize_prediction(pred)        # back to (10, 12)
    np.testing.assert_array_equal((result > 0).astype(np.uint8), original)


def test_coronal_normalize_roundtrip(volume):
    """Prediction mask for a coronal slice reconstructs the original slice."""
    original = volume[:, 4, :].copy()          # (x=10, z=14)
    yolo_input = original.T                     # (14, 10)
    pred = (yolo_input > 0).astype(np.uint8)
    result = normalize_prediction(pred)        # back to (10, 14)
    np.testing.assert_array_equal((result > 0).astype(np.uint8), original)


def test_sagital_normalize_roundtrip(volume):
    """Prediction mask for a sagital slice reconstructs the original slice."""
    original = volume[3, :, :].copy()          # (y=12, z=14)
    yolo_input = original.T                     # (14, 12)
    pred = (yolo_input > 0).astype(np.uint8)
    result = normalize_prediction(pred)        # back to (12, 14)
    np.testing.assert_array_equal((result > 0).astype(np.uint8), original)


# ---------------------------------------------------------------------------
# insert_slice round-trip
# ---------------------------------------------------------------------------


def test_axial_insert_roundtrip(volume):
    """A slice inserted at the correct axial index matches the original."""
    i = 7
    original = volume[:, :, i].copy()
    new_vol = np.zeros_like(volume, dtype=np.float32)
    insert_slice(new_vol, original.astype(np.float32), i, "axial")
    np.testing.assert_array_equal(new_vol[:, :, i], original)


def test_coronal_insert_roundtrip(volume):
    """A slice inserted at the correct coronal index matches the original."""
    j = 6
    original = volume[:, j, :].copy()
    new_vol = np.zeros_like(volume, dtype=np.float32)
    insert_slice(new_vol, original.astype(np.float32), j, "coronal")
    np.testing.assert_array_equal(new_vol[:, j, :], original)


def test_sagital_insert_roundtrip(volume):
    """A slice inserted at the correct sagital index matches the original."""
    k = 2
    original = volume[k, :, :].copy()
    new_vol = np.zeros_like(volume, dtype=np.float32)
    insert_slice(new_vol, original.astype(np.float32), k, "sagital")
    np.testing.assert_array_equal(new_vol[k, :, :], original)


# ---------------------------------------------------------------------------
# Full end-to-end round-trip (extract → normalize → insert → compare)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plane,idx", [("axial", 5), ("coronal", 4), ("sagital", 3)])
def test_full_roundtrip_all_planes(volume, plane, idx):
    """
    For every plane: extract a slice, simulate a perfect YOLO prediction,
    normalize it back, and insert it — the reconstructed volume slice must
    match the original NIfTI slice exactly.
    """
    if plane == "axial":
        original = volume[:, :, idx]
    elif plane == "coronal":
        original = volume[:, idx, :]
    else:
        original = volume[idx, :, :]

    pred = (original.T > 0).astype(np.uint8)  # image-space mask
    reconstructed = (normalize_prediction(pred) > 0).astype(np.uint8)

    new_vol = np.zeros_like(volume)
    insert_slice(new_vol, reconstructed, idx, plane)

    if plane == "axial":
        np.testing.assert_array_equal(new_vol[:, :, idx], original)
    elif plane == "coronal":
        np.testing.assert_array_equal(new_vol[:, idx, :], original)
    else:
        np.testing.assert_array_equal(new_vol[idx, :, :], original)
