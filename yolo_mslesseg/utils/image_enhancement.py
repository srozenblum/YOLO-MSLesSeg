"""
Module: image_enhancement.py

Description:
    Defines an object-oriented structure for applying different 2D image
    enhancement techniques. Includes an abstract base class 'Algorithm' and
    four concrete implementations: HE, CLAHE, GC, and LT. Each class
    implements its own 'apply' method, which executes the corresponding
    technique on an input image.

Usage:
    from yolo_mslesseg.utils.image_enhancement import get_algorithm
    algo = get_algorithm("HE")
    enhanced = algo.apply(image)

Inputs:
    - 2D or 3D float NumPy image arrays.

Outputs:
    - Enhanced uint8 NumPy image arrays.

Relationships:
    - Used by Patient (utils/Patient.py) to apply enhancements per slice.
    - Enhancement names must match constants.ENHANCEMENTS.
"""

from abc import ABC, abstractmethod
from typing import Any

import cv2

from yolo_mslesseg.utils.constants import DEFAULT_GAMMA
import numpy as np

from yolo_mslesseg.utils.utils import convert_to_bgr


# ======================================
#             BASE CLASS
# ======================================


class Algorithm(ABC):
    """
    Class: Algorithm

    Description:
        Abstract base class for 2D image enhancement techniques.
        Defines the common interface that all subclasses must implement.

    Attributes:
        None. Subclasses define their own algorithm-specific parameters.
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the enhancement algorithm to a 2D image and return the result.

        Args:
            image: 2D or 3D float image array to enhance.

        Returns:
            Enhanced uint8 NumPy image array.
        """

    def __repr__(self) -> str:
        """Returns the class name as the string representation.

        Returns:
            Class name string of the concrete subclass.
        """
        return type(self).__name__


# ======================================
#        ENHANCEMENT TECHNIQUES
# ======================================


class HE(Algorithm):
    """
    Class: HE

    Description:
        Implements Histogram Equalisation (HE), improving global contrast by
        redistributing pixel intensity uniformly across the histogram.

    Attributes:
        None. This algorithm has no configurable parameters.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies HE to the image.

        Args:
            image: 2D or 3D float image array to enhance.

        Returns:
            Enhanced image as a uint8 RGB NumPy array.
        """

        # Convert the image to BGR if it is RGB or greyscale
        img_bgr = convert_to_bgr(image)

        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

        # Convert to YUV to isolate luminance (Y). Equalising Y while leaving U and V
        # unchanged prevents colour distortion in multi-channel (multi-modality) images.
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        return img_rgb


class CLAHE(Algorithm):
    """
    Class: CLAHE

    Description:
        Implements Contrast Limited Adaptive Histogram Equalisation (CLAHE),
        which adjusts contrast locally while avoiding over-amplification of noise.

    Attributes:
        clip_limit (float): contrast limit for equalisation (default 2.0).
        tile_grid_size (tuple[int, int]): grid size for local processing (default (8, 8)).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        """Initialises CLAHE with configurable clip limit and tile grid size.

        Args:
            clip_limit: Contrast limit for histogram equalisation. Defaults to 2.0.
            tile_grid_size: Grid size for local adaptive processing. Defaults to (8, 8).
        """
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE to the L channel of the image.

        Args:
            image: 2D or 3D float image array to enhance.

        Returns:
            Enhanced image as a uint8 BGR NumPy array.
        """

        # Convert the image to BGR if it is RGB or greyscale
        img_bgr = convert_to_bgr(image)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

        # Apply CLAHE only to the L (luminance) channel
        l_clahe = clahe.apply(l)

        img_merge = cv2.merge((l_clahe, a, b))
        img_result = cv2.cvtColor(img_merge, cv2.COLOR_LAB2BGR)

        return img_result

    def __repr__(self) -> str:
        """Returns the class name with clip limit and tile grid size.

        Returns:
            String of the form 'CLAHE(clip_limit=<value>, tile_grid_size=<value>)'.
        """
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


class GC(Algorithm):
    """
    Class: GC

    Description:
        Implements Gamma Correction (GC), adjusting brightness and contrast
        through a non-linear transformation of pixel intensity values.

    Attributes:
        gamma (float): gamma correction factor (default 2.0).
    """

    def __init__(self, gamma: float = DEFAULT_GAMMA) -> None:
        """Initialises GC with a configurable gamma correction factor.

        Args:
            gamma: Gamma correction exponent applied to pixel intensities. Defaults to DEFAULT_GAMMA.
        """
        super().__init__()
        self.gamma = gamma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies GC to the image.

        Args:
            image: 2D or 3D float image array to enhance.

        Returns:
            Enhanced image as a uint8 BGR NumPy array.
        """

        # Convert the image to BGR if it is RGB or greyscale
        img_bgr = convert_to_bgr(image)

        # Build a 256-entry lookup table for gamma correction. Applying via cv2.LUT
        # is significantly faster than per-pixel exponentiation on large images.
        table = np.array((np.linspace(0, 1, 256) ** self.gamma) * 255, dtype=np.uint8)

        img_bgr = cv2.LUT(img_bgr, table)

        return img_bgr

    def __repr__(self) -> str:
        """Returns the class name and gamma value.

        Returns:
            String of the form 'GC(gamma=<value>)'.
        """
        return f"GC(gamma={self.gamma})"


class LT(Algorithm):
    """
    Class: LT

    Description:
        Implements Logarithmic Transformation (LT), which enhances details in
        dark regions by compressing the dynamic intensity range.

    Attributes:
        None. This algorithm has no configurable parameters.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Applies LT to the image.

        Args:
            image: 2D or 3D float image array to enhance.

        Returns:
            Enhanced image as a uint8 BGR NumPy array.
        """

        # Convert the image to BGR if it is RGB or greyscale
        img_bgr = convert_to_bgr(image)

        # Upcast to uint16 before the log transform: c*log(1+x)
        # can exceed 255 for uint8 inputs and would wrap around silently.
        img_bgr = img_bgr.astype(np.uint16)

        # Compute the scaling constant c
        c = 255 / np.log(1 + img_bgr.max())

        img_log = c * np.log(1 + img_bgr)

        # Clip and convert the result to uint8 for compatibility with OpenCV
        img_bgr = np.clip(img_log, 0, 255).astype(np.uint8)

        return img_bgr


# ======================================
#              REGISTRY
# ======================================

_REGISTRY: dict[str, type[Algorithm]] = {
    "HE": HE,
    "CLAHE": CLAHE,
    "GC": GC,
    "LT": LT,
}


def get_algorithm(name: str, **kwargs: Any) -> Algorithm:
    """Returns a new instance of the enhancement algorithm identified by name.

    For the 'GC' algorithm, keyword arguments are forwarded to the constructor
    (e.g. ``get_algorithm("GC", gamma=1.5)`` returns ``GC(gamma=1.5)``).
    For all other algorithms, keyword arguments are silently ignored.

    Args:
        name: Algorithm key as stored in constants.ENHANCEMENTS ('HE', 'CLAHE', 'GC', 'LT').
        **kwargs: Optional keyword arguments forwarded to the GC constructor.

    Returns:
        New instance of the requested Algorithm subclass.

    Raises:
        ValueError: If name does not match any registered algorithm.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown enhancement algorithm: '{name}'. "
            f"Valid options: {list(_REGISTRY)}"
        )
    if name == "GC":
        return cls(**kwargs)
    return cls()
