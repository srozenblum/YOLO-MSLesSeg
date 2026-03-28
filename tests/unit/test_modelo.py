"""Unit tests for the Model class."""

from pathlib import Path

import pytest

from yolo_mslesseg.utils.Model import Model


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------
def test_init_minimal():
    m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
    assert m.plane == "axial"
    assert m.num_slices == 50
    assert m.enhancement is None


def test_init_with_enhancement():
    m = Model(
        plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5, enhancement="gc"
    )
    assert m.enhancement == "GC"  # uppercased


def test_init_case_insensitive_plane():
    m = Model(plane="AXIAL", num_slices=50, modality=["T1"], k_folds=1)
    assert m.plane == "axial"


def test_init_percentile_num_slices():
    m = Model(plane="axial", num_slices="P50", modality=["T1"], k_folds=5)
    assert m.num_slices == "P50"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_plane():
    with pytest.raises(ValueError, match="Invalid plane"):
        Model(plane="diagonal", num_slices=50, modality=["T1"], k_folds=5)


def test_invalid_enhancement():
    with pytest.raises(ValueError, match="enhancement"):
        Model(
            plane="axial",
            num_slices=50,
            modality=["T1"],
            k_folds=5,
            enhancement="UNKNOWN",
        )


def test_invalid_num_slices_zero():
    with pytest.raises(ValueError, match="num_slices"):
        Model(plane="axial", num_slices=0, modality=["T1"], k_folds=5)


def test_invalid_num_slices_negative():
    with pytest.raises(ValueError, match="num_slices"):
        Model(plane="axial", num_slices=-10, modality=["T1"], k_folds=5)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_folds_string_single():
    m = Model(plane="axial", num_slices=50, modality=["T1"], k_folds=1)
    assert m.folds_string == "1fold"


def test_folds_string_cv():
    m = Model(plane="axial", num_slices=50, modality=["T1"], k_folds=5)
    assert m.folds_string == "5folds"


def test_exp_string_no_enhancement():
    m = Model(plane="axial", num_slices=50, modality=["T1"], k_folds=5)
    assert m.exp_string == "Base"


def test_exp_string_with_enhancement():
    m = Model(
        plane="axial", num_slices=50, modality=["T1"], k_folds=5, enhancement="HE"
    )
    assert m.exp_string == "HE"


def test_modality_str_single():
    m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
    assert m.modality_str == "FLAIR"


def test_modality_str_multiple():
    m = Model(plane="axial", num_slices=50, modality=["T1", "FLAIR"], k_folds=5)
    assert m.modality_str == "T1FLAIR"


def test_base_path_no_enhancement():
    m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
    assert m.base_path == Path("Base") / "FLAIR_50slices_5folds"


def test_base_path_with_enhancement():
    m = Model(
        plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5, enhancement="GC"
    )
    assert m.base_path == Path("GC") / "FLAIR_50slices_5folds"


def test_base_path_percentile():
    m = Model(
        plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=5, enhancement="GC"
    )
    assert m.base_path == Path("GC") / "FLAIR_P50slices_5folds"


def test_model_string_no_enhancement():
    m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
    assert m.model_string == "axial_FLAIR_50slices_5folds"


def test_model_string_with_enhancement():
    m = Model(
        plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5, enhancement="GC"
    )
    assert m.model_string == "axial_FLAIR_GC_50slices_5folds"


def test_repr():
    m = Model(plane="axial", num_slices=50, modality=["FLAIR"], k_folds=5)
    assert "Model(" in repr(m)
