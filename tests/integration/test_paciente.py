"""
Integration tests for the Patient class using real NIfTI files from the dataset.

Requires MSLesSeg-Dataset/ to be present at the repository root.
Uses P1 (train split) and P54 (test split) as representative patients.
"""

import numpy as np
import pytest

from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import SPLIT_TEST, SPLIT_TRAIN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patient_train():
    return Patient(id="P1", plane="axial", modality=["FLAIR"])


@pytest.fixture(scope="module")
def patient_test():
    return Patient(id="P54", plane="axial", modality=["FLAIR"])


# ---------------------------------------------------------------------------
# Split resolution
# ---------------------------------------------------------------------------

def test_train_patient_resolves_to_train_split(patient_train):
    assert patient_train.split == SPLIT_TRAIN


def test_test_patient_resolves_to_test_split(patient_test):
    assert patient_test.split == SPLIT_TEST


def test_nonexistent_patient_raises():
    with pytest.raises(FileNotFoundError):
        Patient(id="P999", plane="axial")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_id_format_raises():
    with pytest.raises(ValueError, match="ID"):
        Patient(id="999", plane="axial")


def test_invalid_plane_raises():
    with pytest.raises(ValueError, match="[Pp]lane|[Pp]lano"):
        Patient(id="P1", plane="diagonal")


def test_invalid_enhancement_raises():
    with pytest.raises(ValueError, match="[Ee]nhancement|[Mm]ejora"):
        Patient(id="P1", plane="axial", enhancement="INVALID")


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def test_volume_is_3d(patient_train):
    vol = patient_train.load_volume("FLAIR")
    assert vol.ndim == 3


def test_volume_has_positive_dimensions(patient_train):
    vol = patient_train.load_volume("FLAIR")
    assert all(d > 0 for d in vol.shape)


def test_gt_mask_same_shape_as_volume(patient_train):
    vol = patient_train.load_volume("FLAIR")
    gt = patient_train.gt_mask
    assert gt.shape == vol.shape


def test_gt_mask_is_binary(patient_train):
    gt = patient_train.gt_mask
    unique_values = np.unique(gt)
    assert set(unique_values).issubset({0, 1})


# ---------------------------------------------------------------------------
# Slice extraction
# ---------------------------------------------------------------------------

def test_get_image_slice_returns_2d(patient_train):
    vol = patient_train.load_volume("FLAIR")
    mid = vol.shape[2] // 2
    slice_2d = patient_train.get_image_slice(mid, "FLAIR")
    assert slice_2d.ndim == 2


def test_num_slices_positive(patient_train):
    assert patient_train.num_slices > 0


# ---------------------------------------------------------------------------
# Multi-channel slice
# ---------------------------------------------------------------------------

def test_get_multichannel_slice_shape(patient_train):
    vol = patient_train.load_volume("FLAIR")
    mid = vol.shape[2] // 2
    img = patient_train.get_multichannel_slice(mid)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_get_multichannel_slice_dtype(patient_train):
    vol = patient_train.load_volume("FLAIR")
    mid = vol.shape[2] // 2
    img = patient_train.get_multichannel_slice(mid)
    assert img.dtype == np.uint8


def test_lesion_slices_multichannel_returns_list(patient_train):
    slices = patient_train.lesion_slices_multichannel(num_slices=5)
    assert isinstance(slices, list)
    assert len(slices) > 0
    i, img = slices[0]
    assert isinstance(i, int)
    assert img.shape[2] == 3


# ---------------------------------------------------------------------------
# plane_index — plane-to-axis mapping
# ---------------------------------------------------------------------------


def test_plane_index_axial_maps_to_z_axis(patient_train):
    assert patient_train.plane_index(5) == (slice(None), slice(None), 5)


def test_plane_index_coronal_maps_to_y_axis():
    p = Patient(id="P1", plane="coronal", modality=["FLAIR"])
    assert p.plane_index(3) == (slice(None), 3, slice(None))


def test_plane_index_sagittal_maps_to_x_axis():
    p = Patient(id="P1", plane="sagittal", modality=["FLAIR"])
    assert p.plane_index(2) == (2, slice(None), slice(None))


def test_plane_index_consenso_raises():
    p = Patient(id="P1", plane="consenso", modality=["FLAIR"])
    with pytest.raises(ValueError):
        p.plane_index(0)


# ---------------------------------------------------------------------------
# Coronal and sagittal slice shapes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def patient_coronal():
    return Patient(id="P1", plane="coronal", modality=["FLAIR"])


@pytest.fixture(scope="module")
def patient_sagittal():
    return Patient(id="P1", plane="sagittal", modality=["FLAIR"])


def test_coronal_slice_is_2d(patient_coronal):
    vol = patient_coronal.load_volume("FLAIR")
    slice_2d = patient_coronal.get_image_slice(vol.shape[1] // 2, "FLAIR")
    assert slice_2d.ndim == 2


def test_coronal_slice_shape_matches_xz(patient_coronal):
    vol = patient_coronal.load_volume("FLAIR")
    slice_2d = patient_coronal.get_image_slice(vol.shape[1] // 2, "FLAIR")
    assert slice_2d.shape == (vol.shape[0], vol.shape[2])


def test_sagittal_slice_is_2d(patient_sagittal):
    vol = patient_sagittal.load_volume("FLAIR")
    slice_2d = patient_sagittal.get_image_slice(vol.shape[0] // 2, "FLAIR")
    assert slice_2d.ndim == 2


def test_sagittal_slice_shape_matches_yz(patient_sagittal):
    vol = patient_sagittal.load_volume("FLAIR")
    slice_2d = patient_sagittal.get_image_slice(vol.shape[0] // 2, "FLAIR")
    assert slice_2d.shape == (vol.shape[1], vol.shape[2])


# ---------------------------------------------------------------------------
# num_slices — plane-to-dimension mapping
# ---------------------------------------------------------------------------


def test_axial_num_slices_equals_z_dim(patient_train):
    assert patient_train.num_slices == patient_train.gt_mask.shape[2]


def test_coronal_num_slices_equals_y_dim(patient_coronal):
    assert patient_coronal.num_slices == patient_coronal.gt_mask.shape[1]


def test_sagittal_num_slices_equals_x_dim(patient_sagittal):
    assert patient_sagittal.num_slices == patient_sagittal.gt_mask.shape[0]


# ---------------------------------------------------------------------------
# get_mask_slice
# ---------------------------------------------------------------------------


def test_get_mask_slice_is_2d(patient_train):
    mid = patient_train.num_slices // 2
    mask_slice = patient_train.get_mask_slice(mid)
    assert mask_slice.ndim == 2


def test_get_mask_slice_shape_matches_image(patient_train):
    mid = patient_train.num_slices // 2
    img_slice = patient_train.get_image_slice(mid, "FLAIR")
    mask_slice = patient_train.get_mask_slice(mid)
    assert img_slice.shape == mask_slice.shape


def test_get_mask_slice_is_binary(patient_train):
    mid = patient_train.num_slices // 2
    mask_slice = patient_train.get_mask_slice(mid)
    assert set(np.unique(mask_slice)).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# slices_to_use — central-window selection
# ---------------------------------------------------------------------------


def test_slices_to_use_none_returns_all_lesion_slices(patient_train):
    all_indices = patient_train.lesion_slice_indices()
    assert patient_train.slices_to_use(num_slices=None) == all_indices


def test_slices_to_use_large_count_returns_all(patient_train):
    all_indices = patient_train.lesion_slice_indices()
    assert patient_train.slices_to_use(num_slices=10_000) == all_indices


def test_slices_to_use_limited_count_returns_exact_count(patient_train):
    all_indices = patient_train.lesion_slice_indices()
    n = max(1, len(all_indices) - 2)
    if n < len(all_indices):
        assert len(patient_train.slices_to_use(num_slices=n)) == n


def test_slices_to_use_subset_is_from_lesion_slices(patient_train):
    all_indices = patient_train.lesion_slice_indices()
    used = patient_train.slices_to_use(num_slices=2)
    assert all(i in all_indices for i in used)


def test_lesion_slice_indices_nonempty(patient_train):
    assert len(patient_train.lesion_slice_indices()) > 0


def test_lesion_slice_indices_sorted(patient_train):
    indices = patient_train.lesion_slice_indices()
    assert indices == sorted(indices)
