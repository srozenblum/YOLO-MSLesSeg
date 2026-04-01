from pathlib import Path

from yolo_mslesseg.utils.constants import ENHANCEMENTS, PLANES


class Model:
    """
    Class: Model

    Description:
        Represents the structural configuration of a YOLO model used in the
        YOLO-MSLesSeg pipeline. Manages the identification and naming of the
        model according to the anatomical plane, the MRI modalities, the number
        of slices, the number of cross-validation folds, and the image
        enhancement technique applied.

        Used in all pipeline stages to ensure consistent naming across
        different experimental configurations.

    Attributes:
        plane (str):
            Anatomical plane used ('axial', 'coronal', 'sagittal') or 'consensus'.

        num_slices (int | str):
            Number of slices used (integer value or percentile string 'PX').

        modality (list[str]):
            MRI image modality or modalities considered (T1, T2, FLAIR).

        k_folds (int):
            Number of folds used for cross-validation.

        enhancement (str | None):
            Enhancement algorithm applied. Must be one of the values defined
            in constants.ENHANCEMENTS ('HE', 'CLAHE', 'GC', 'LT'), or None.
            Defaults to None.
    """

    def __init__(
        self,
        plane: str,
        num_slices: int | str,
        modality: list[str],
        k_folds: int,
        enhancement: str | None = None,
    ) -> None:
        """Initialises a Model instance with the given experimental configuration.

        Args:
            plane: Anatomical plane ('axial', 'coronal', 'sagittal', or 'consensus').
            num_slices: Number of slices as an integer or percentile string (e.g. 'P50').
            modality: List of MRI modalities to include (e.g. ['T1', 'FLAIR']).
            k_folds: Number of cross-validation folds (1 for fixed split).
            enhancement: Image enhancement algorithm name, or None for no enhancement.

        Raises:
            ValueError: If plane, num_slices, or enhancement values are invalid.
        """
        # --- Argument validation ---
        self._validate_args(plane, num_slices, enhancement)

        # --- Core attributes ---
        self._set_core_attributes(
            plane,
            num_slices,
            modality,
            k_folds,
            enhancement,
        )

    # ======================================
    #        CONSTRUCTOR HELPERS
    # ======================================

    def _validate_args(self, plane: str, num_slices: int | str, enhancement: str | None) -> None:
        """Validates the constructor arguments before setting any attributes.

        Args:
            plane: Anatomical plane string to validate.
            num_slices: Slice count to validate (must be a positive integer if numeric).
            enhancement: Enhancement algorithm name to validate, or None.

        Raises:
            ValueError: If any argument is outside the set of accepted values.
        """
        if plane.lower() not in PLANES:
            raise ValueError(f"Invalid plane '{plane}'. Must be one of {PLANES}.")
        if isinstance(num_slices, int) and num_slices <= 0:
            raise ValueError(f"num_slices must be a positive integer, got {num_slices}.")
        if enhancement is not None and enhancement.upper() not in ENHANCEMENTS:
            raise ValueError(
                f"Invalid enhancement '{enhancement}'. Must be one of {ENHANCEMENTS} or None."
            )

    def _set_core_attributes(
        self,
        plane: str,
        num_slices: int | str,
        modality: list[str],
        k_folds: int,
        enhancement: str | None,
    ) -> None:
        """Sets the core attributes of the model after validation.

        Args:
            plane: Anatomical plane string (stored in lowercase).
            num_slices: Slice count as an integer or percentile string.
            modality: List of MRI modality strings.
            k_folds: Number of cross-validation folds.
            enhancement: Enhancement algorithm name (stored in uppercase), or None.
        """
        self.plane = plane.lower()
        self.num_slices = num_slices
        self.modality = modality
        self.k_folds = k_folds
        self.enhancement = enhancement.upper() if enhancement else None

    # ======================================
    #           IDENTIFIERS
    # ======================================

    @property
    def modality_str(self) -> str:
        """Concatenated representation of the image modalities (e.g. 'T1T2FLAIR')."""
        return "".join(self.modality)

    @property
    def exp_string(self) -> str:
        """Short experiment name ('Base' or enhancement type)."""
        return self.enhancement if self.enhancement else "Base"

    @property
    def folds_string(self) -> str:
        """String representation of the number of folds ('1fold' or '<k>folds')."""
        if self.k_folds == 1:
            return "1fold"
        return f"{self.k_folds}folds"

    @property
    def base_path(self) -> Path:
        """Base path for the model."""
        return (
            Path(self.exp_string)
            / f"{self.modality_str}_{self.num_slices}slices_{self.folds_string}"
        )

    @property
    def model_string(self) -> str:
        """Unique, human-readable model identifier based on plane, modality, and slice count."""
        if not self.enhancement:
            return f"{self.plane}_{self.modality_str}_{self.num_slices}slices_{self.folds_string}"
        else:
            return f"{self.plane}_{self.modality_str}_{self.enhancement}_{self.num_slices}slices_{self.folds_string}"

    # ======================================
    #           REPRESENTATION
    # ======================================

    def __repr__(self) -> str:
        """Internal representation of the Model instance."""
        return f"Model({self.model_string})"

    def __str__(self) -> str:
        """Human-readable representation of the Model instance."""
        return f"{self.model_string}"
