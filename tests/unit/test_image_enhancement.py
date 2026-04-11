"""Unit tests for image enhancement algorithms in image_enhancement.py."""

import numpy as np
import pytest

from yolo_mslesseg.utils.image_enhancement import (
    CLAHE,
    GC,
    HE,
    LT,
    get_algorithm,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    """Synthetic 2D grayscale float32 image with values in [0, 1000]."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1000, (64, 64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers shared across algorithm test classes
# ---------------------------------------------------------------------------

def _assert_output_contract(output: np.ndarray, input_image: np.ndarray) -> None:
    """Assert spatial dims match, dtype is uint8, and values are in [0, 255].

    Args:
        output: Enhanced image array produced by an algorithm.
        input_image: Original input image passed to the algorithm.
    """
    # All algorithms convert greyscale → BGR, so output is (H, W, 3).
    assert output.shape[:2] == input_image.shape[:2]
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255


# ---------------------------------------------------------------------------
# TestHE
# ---------------------------------------------------------------------------

class TestHE:
    def test_output_shape_equals_input_shape(self, sample_image):
        out = HE().apply(sample_image)
        assert out.shape[:2] == sample_image.shape[:2]

    def test_output_dtype_is_uint8(self, sample_image):
        out = HE().apply(sample_image)
        assert out.dtype == np.uint8

    def test_output_values_in_range(self, sample_image):
        out = HE().apply(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_input_not_mutated(self, sample_image):
        original = sample_image.copy()
        HE().apply(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ---------------------------------------------------------------------------
# TestCLAHE
# ---------------------------------------------------------------------------

class TestCLAHE:
    def test_output_shape_equals_input_shape(self, sample_image):
        out = CLAHE().apply(sample_image)
        assert out.shape[:2] == sample_image.shape[:2]

    def test_output_dtype_is_uint8(self, sample_image):
        out = CLAHE().apply(sample_image)
        assert out.dtype == np.uint8

    def test_output_values_in_range(self, sample_image):
        out = CLAHE().apply(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_input_not_mutated(self, sample_image):
        original = sample_image.copy()
        CLAHE().apply(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ---------------------------------------------------------------------------
# TestGC
# ---------------------------------------------------------------------------

class TestGC:
    def test_output_shape_equals_input_shape(self, sample_image):
        out = GC().apply(sample_image)
        assert out.shape[:2] == sample_image.shape[:2]

    def test_output_dtype_is_uint8(self, sample_image):
        out = GC().apply(sample_image)
        assert out.dtype == np.uint8

    def test_output_values_in_range(self, sample_image):
        out = GC().apply(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_input_not_mutated(self, sample_image):
        original = sample_image.copy()
        GC().apply(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ---------------------------------------------------------------------------
# TestLT
# ---------------------------------------------------------------------------

class TestLT:
    def test_output_shape_equals_input_shape(self, sample_image):
        out = LT().apply(sample_image)
        assert out.shape[:2] == sample_image.shape[:2]

    def test_output_dtype_is_uint8(self, sample_image):
        out = LT().apply(sample_image)
        assert out.dtype == np.uint8

    def test_output_values_in_range(self, sample_image):
        out = LT().apply(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_input_not_mutated(self, sample_image):
        original = sample_image.copy()
        LT().apply(sample_image)
        np.testing.assert_array_equal(sample_image, original)


# ---------------------------------------------------------------------------
# TestGetAlgorithm
# ---------------------------------------------------------------------------

class TestGetAlgorithm:
    def test_he_returns_he_instance(self):
        assert isinstance(get_algorithm("HE"), HE)

    def test_clahe_returns_clahe_instance(self):
        assert isinstance(get_algorithm("CLAHE"), CLAHE)

    def test_gc_returns_gc_instance(self):
        assert isinstance(get_algorithm("GC"), GC)

    def test_lt_returns_lt_instance(self):
        assert isinstance(get_algorithm("LT"), LT)

    def test_invalid_name_raises_value_error(self):
        with pytest.raises(ValueError):
            get_algorithm("UNKNOWN")

    def test_gc_default_gamma_is_two(self):
        algo = get_algorithm("GC")
        assert algo.gamma == 2.0

    def test_gc_custom_gamma_is_forwarded(self):
        algo = get_algorithm("GC", gamma=1.5)
        assert isinstance(algo, GC)
        assert algo.gamma == 1.5

    def test_non_gc_algorithm_ignores_gamma_kwarg(self):
        algo = get_algorithm("HE", gamma=1.5)
        assert isinstance(algo, HE)
