"""
Module: Model.py

Description:
    Defines the Model class, which represents the structural configuration
    of a YOLO model used in the YOLO-MSLesSeg pipeline. The Model object is
    the central configuration carrier passed to all pipeline stages.

Usage:
    from yolo_mslesseg.utils.Model import Model
    model = Model(plane="axial", num_slices="P50", modality=["FLAIR"], k_folds=5)

Inputs:
    None. Provides the Model class definition.

Outputs:
    None. Provides the Model class definition.

Relationships:
    - Used by all Config classes (ConfigDataset, ConfigTrain, ConfigPred,
      ConfigReconstruction, ConfigEval, ConfigConsensus).
    - Used by all pipeline scripts as the primary configuration object.
"""

from pathlib import Path

from yolo_mslesseg.utils.constants import ENHANCEMENTS, MODALITIES, PLANES, DEFAULT_GAMMA, ENHANCEMENT_BASE


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

        gamma (float):
            Gamma correction factor used when enhancement is 'GC'. Must be
            positive. Defaults to 2.0. Ignored when enhancement is not 'GC'.
    """

    def __init__(
        self,
        plane: str,
        num_slices: int | str,
        modality: list[str],
        k_folds: int,
        enhancement: str | None = None,
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        """Initialises a Model instance with the given experimental configuration.

        Args:
            plane: Anatomical plane ('axial', 'coronal', 'sagittal', or 'consensus').
            num_slices: Number of slices as an integer or percentile string (e.g. 'P50').
            modality: List of MRI modalities to include (e.g. ['T1', 'FLAIR']).
            k_folds: Number of cross-validation folds (1 for fixed split).
            enhancement: Image enhancement algorithm name, or None for no enhancement.
            gamma: Gamma correction factor used when enhancement is 'GC'. Must be
                positive. Defaults to DEFAULT_GAMMA.

        Raises:
            ValueError: If plane, num_slices, enhancement, or gamma values are invalid.
        """
        # --- Argument validation ---
        self._validate_args(plane, num_slices, enhancement, gamma)

        # --- Core attributes ---
        self._set_core_attributes(
            plane,
            num_slices,
            modality,
            k_folds,
            enhancement,
            gamma,
        )

    # ======================================
    #        CONSTRUCTOR HELPERS
    # ======================================

    def _validate_args(
        self,
        plane: str,
        num_slices: int | str,
        enhancement: str | None,
        gamma: float,
    ) -> None:
        """Validates the constructor arguments before setting any attributes.

        Args:
            plane: Anatomical plane string to validate.
            num_slices: Slice count to validate (must be a positive integer if numeric).
            enhancement: Enhancement algorithm name to validate, or None.
            gamma: Gamma correction factor to validate (must be positive).

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
        if gamma <= 0:
            raise ValueError(f"gamma must be a positive number, got {gamma}.")

    def _set_core_attributes(
        self,
        plane: str,
        num_slices: int | str,
        modality: list[str],
        k_folds: int,
        enhancement: str | None,
        gamma: float,
    ) -> None:
        """Sets the core attributes of the model after validation.

        Args:
            plane: Anatomical plane string (stored in lowercase).
            num_slices: Slice count as an integer or percentile string.
            modality: List of MRI modality strings.
            k_folds: Number of cross-validation folds.
            enhancement: Enhancement algorithm name (stored in uppercase), or None.
            gamma: Gamma correction factor (used only when enhancement is 'GC').
        """
        self.plane = plane.lower()
        self.num_slices = num_slices
        self.modality = list(dict.fromkeys(modality))
        self.k_folds = k_folds
        self.enhancement = enhancement.upper() if enhancement else None
        self.gamma = gamma

    # ======================================
    #           IDENTIFIERS
    # ======================================

    @property
    def modality_str(self) -> str:
        """Concatenated representation of the image modalities (e.g. 'T1T2FLAIR').

        Returns:
            String of modality names joined without separators.
        """
        return "".join(m for m in MODALITIES if m in self.modality)  # Iterate over MODALITIES (canonical order) to produce a deterministic string regardless of the order the user specified.

    @property
    def exp_string(self) -> str:
        """Short experiment name ('Base', enhancement type, or 'GC/g<gamma>' for GC).

        Returns:
            'GC/g<gamma>' when enhancement is 'GC', enhancement algorithm name
            for other enhancements, or 'Base' when no enhancement is set.
        """
        if self.enhancement == "GC":
            return f"GC/g{self.gamma}"
        return self.enhancement if self.enhancement else ENHANCEMENT_BASE

    @property
    def folds_string(self) -> str:
        """String representation of the number of folds ('1fold' or '<k>folds').

        Returns:
            '1fold' when k_folds == 1, or '<k>folds' for cross-validation.
        """
        if self.k_folds == 1:
            return "1fold"
        return f"{self.k_folds}folds"

    @property
    def base_path(self) -> Path:
        """Root directory segment shared by all pipeline output directories (datasets/,
        trains/, pred_vols/, results/) for this experiment configuration.

        Returns:
            Path combining the experiment string and the modality/slices/folds identifier.
        """
        return (
            Path(self.exp_string)
            / f"{self.modality_str}_{self.num_slices}slices_{self.folds_string}"
        )

    @property
    def model_string(self) -> str:
        """Unique, human-readable model identifier based on plane, modality, and slice count.

        When enhancement is 'GC', the gamma value is included in the identifier
        (e.g. 'axial_FLAIR_GC_g2.0_P50slices_5folds').

        Returns:
            Canonical model identifier string including plane, modalities, enhancement
            and gamma (if GC), slice count, and fold scheme.
        """
        if self.enhancement == "GC":
            return f"{self.plane}_{self.modality_str}_GC_g{self.gamma}_{self.num_slices}slices_{self.folds_string}"
        elif self.enhancement:
            return f"{self.plane}_{self.modality_str}_{self.enhancement}_{self.num_slices}slices_{self.folds_string}"
        else:
            return f"{self.plane}_{self.modality_str}_{self.num_slices}slices_{self.folds_string}"

    # ======================================
    #           REPRESENTATION
    # ======================================

    def __repr__(self) -> str:
        """String representation of the Model instance.

        Returns:
            String of the form 'Model(<model_string>)'.
        """
        return f"Model({self.model_string})"
