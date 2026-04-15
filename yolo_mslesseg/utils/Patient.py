"""
Module: Patient.py

Description:
    Defines the Patient class, which represents a single patient from the
    MSLesSeg dataset. Provides access to MRI volumes, ground truth masks,
    and lesion-containing slices for a given anatomical plane, timepoint,
    and set of modalities.

Usage:
    from yolo_mslesseg.utils.Patient import Patient
    patient = Patient(id="P12", plane="axial", modality=["FLAIR"])

Inputs:
    - MSLesSeg-Dataset/ directory must exist with the expected structure.

Outputs:
    None. Provides the Patient class definition.

Relationships:
    - Used by extract_dataset.py for slice extraction.
    - Used by eval.py and run_pipeline.py for individual patient evaluation.
    - Used by ConfigBase subclasses (ConfigPred, ConfigEval, etc.).
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from yolo_mslesseg.utils.constants import (
    PLANES,
    TIMEPOINTS,
    ENHANCEMENTS,
    MODALITIES,
    DATASET_DIR,
    SPLIT_TRAIN,
    SPLIT_TEST,
    EXT_NIFTI,
    MASK_SUFFIX,
    DEFAULT_GAMMA,
    PLANE_CONSENSUS,
)
from yolo_mslesseg.utils.image_enhancement import get_algorithm
from yolo_mslesseg.utils.utils import normalize_to_uint8, path_exists


class Patient:
    """
    Class: Patient

    Description:
        Represents a patient from the MSLesSeg dataset, with their modalities,
        timepoints (if present), anatomical planes, and associated masks. Provides
        access to, enhancement of, and slice extraction from MRI volumes (T1, T2,
        FLAIR) and the ground truth mask across different anatomical planes.

        Used in the slice extraction, training, and evaluation stages to provide
        centralised management of volumes, masks, and enhancement algorithms.

    Directory convention:
        MSLesSeg-Dataset/
            └── train/
                └── PX/
                    ├── T1/
                    │   ├── PX_T1_T1.nii.gz
                    │   ├── PX_T1_T2.nii.gz
                    │   ├── PX_T1_FLAIR.nii.gz
                    │   └── PX_T1_MASK.nii.gz
                    ├── T2/
                    └── ...

    Attributes:
        id (str):
            Patient identifier (format 'PX').

        plane (str):
            Anatomical orientation ('axial', 'coronal', 'sagittal') or 'consensus'.

        timepoint (str, optional):
            MRI acquisition timepoint. Defaults to 'T1'.
            The pipeline currently always uses 'T1'; other timepoints are
            supported by the class but not exercised by any pipeline stage.

        modality (list[str], optional):
            MRI image modalities considered (T1, T2, FLAIR). Defaults to all.

        modality_str (str):
            Concatenated representation of the image modalities (e.g. 'T1T2FLAIR').

        enhancement (str | None, optional):
            Image enhancement algorithm to apply ('HE', 'CLAHE', 'GC', 'LT', or None).
            Defaults to None.

        gamma (float):
            Gamma correction factor used when enhancement is 'GC'. Defaults to 2.0.

        base_dir (Path):
            Base path of the patient in the MSLesSeg dataset.

        gt_mask (np.ndarray):
            Ground truth volume loaded in memory.

        split (str):
            Dataset split the patient belongs to ('train' or 'test').

        no_timepoints (bool):
            True if the patient directory does not contain timepoint subdirectories.

        _volumes (dict[str, np.ndarray]):
            Dictionary of volumes loaded per modality.
    """

    def __init__(
        self,
        id: str,
        plane: str,
        timepoint: str = "T1",
        modality: list[str] | None = None,
        enhancement: str | None = None,
        gamma: float = DEFAULT_GAMMA,
        gt_mask: np.ndarray | None = None,
    ) -> None:
        """Initialises a Patient instance for the given MRI configuration.

        Args:
            id: Patient identifier string (e.g. 'P12').
            plane: Anatomical plane ('axial', 'coronal', 'sagittal', or 'consensus').
            timepoint: MRI acquisition timepoint. Defaults to 'T1'.
            modality: List of MRI modalities to use. Defaults to all modalities.
            enhancement: Enhancement algorithm name, or None for no enhancement.
            gamma: Gamma correction factor forwarded to GC when enhancement is 'GC'.
                Defaults to DEFAULT_GAMMA.
            gt_mask: Pre-loaded ground truth mask array. Loaded lazily if None.

        Raises:
            ValueError: If any argument is outside the set of accepted values.
            FileNotFoundError: If the patient directory does not exist in the dataset.
        """
        modality = modality or list(MODALITIES)

        # --- Argument validation ---
        self._validate_args(id, plane, timepoint, enhancement, modality)

        # --- Core attributes ---
        self._set_core_attributes(
            id, plane, timepoint, modality, enhancement, gamma, gt_mask
        )

    # ======================================
    #        CONSTRUCTOR HELPERS
    # ======================================

    def _validate_args(
        self,
        id: str,
        plane: str,
        timepoint: str,
        enhancement: str | None,
        modality: list[str] | None,
    ) -> None:
        """Validates the constructor arguments before setting any attributes.

        Args:
            id: Patient identifier to validate.
            plane: Anatomical plane string to validate.
            timepoint: Timepoint string to validate.
            enhancement: Enhancement algorithm name to validate, or None.
            modality: List of modality strings to validate.

        Raises:
            ValueError: If any argument is outside the set of accepted values.
        """
        if not id.startswith("P"):
            raise ValueError(
                f"Invalid patient ID: '{id}'. "
                "Must follow the format 'P#' (e.g. P1, P12, P53)."
            )

        if plane not in PLANES:
            raise ValueError(f"Invalid plane: {plane}.")

        if timepoint not in TIMEPOINTS:
            raise ValueError(f"Invalid timepoint: {timepoint}.")

        if enhancement is not None and enhancement not in ENHANCEMENTS:
            raise ValueError(
                f"Invalid enhancement algorithm: '{enhancement}'. Options: {ENHANCEMENTS}"
            )

        if modality is None:
            modality = list(MODALITIES)

        invalid = [m for m in modality if m not in MODALITIES]
        if invalid:
            raise ValueError(f"Unrecognised modalities: {invalid}")

    def _resolve_split(self) -> tuple[str, Path]:
        """Resolves which dataset split (train or test) the patient belongs to.

        Returns:
            Tuple of (split_name, base_dir) where split_name is 'train' or 'test'
            and base_dir is the absolute path to the patient's directory in the
            MSLesSeg dataset split.

        Raises:
            FileNotFoundError: If the patient does not exist in either split.
        """
        train_dir = DATASET_DIR / SPLIT_TRAIN / self.id
        test_dir = DATASET_DIR / SPLIT_TEST / self.id

        if train_dir.is_dir():
            return SPLIT_TRAIN, train_dir

        if test_dir.is_dir():
            return SPLIT_TEST, test_dir

        raise FileNotFoundError(
            f"Patient {self.id} does not exist in either train or test split of the MSLesSeg dataset."
        )

    def _set_core_attributes(
        self,
        id: str,
        plane: str,
        timepoint: str,
        modality: list[str],
        enhancement: str | None,
        gamma: float,
        gt_mask: np.ndarray | None,
    ) -> None:
        """Sets the core attributes of the patient after validation.

        Args:
            id: Patient identifier string.
            plane: Anatomical plane string.
            timepoint: MRI acquisition timepoint string.
            modality: List of MRI modality strings.
            enhancement: Enhancement algorithm name (stored in uppercase), or None.
            gamma: Gamma correction factor used when enhancement is 'GC'.
            gt_mask: Pre-loaded ground truth mask array, or None for lazy loading.
        """
        self.id = id
        self.split, self.base_dir = self._resolve_split()
        self.plane = plane
        self.timepoint = timepoint
        self.no_timepoints = not any(
            (self.base_dir / tp).exists() for tp in TIMEPOINTS
        )
        self.enhancement = enhancement.upper() if enhancement else None
        self.gamma = gamma
        self._gt_mask = gt_mask
        self._volumes = {}  # Images by modality

        # Normalise modalities and generate string
        self.modality = list(dict.fromkeys(modality))
        self.modality_str = "".join(
            [m for m in MODALITIES if m in set(self.modality)]
        )

    # ======================================
    #               PATHS
    # ======================================

    @property
    def is_train(self) -> bool:
        """True if the patient belongs to the training split ('train'), False for 'test'."""
        return self.split == SPLIT_TRAIN

    @property
    def is_test(self) -> bool:
        """True if the patient belongs to the test split ('test'), False for 'train'."""
        return self.split == SPLIT_TEST

    def volume_path(self, modality: str) -> Path:
        """Returns the path to the NIfTI volume file for the given modality.

        Args:
            modality: MRI modality string (e.g. 'T1', 'T2', 'FLAIR').

        Returns:
            Path to the corresponding NIfTI volume file.
        """
        if self.no_timepoints:
            return self.base_dir / f"{self.id}_{modality}{EXT_NIFTI}"
        return (
            self.base_dir
            / self.timepoint
            / f"{self.id}_{self.timepoint}_{modality}{EXT_NIFTI}"
        )

    @property
    def gt_mask_path(self) -> Path:
        """Returns the path to the ground truth mask.

        Returns:
            Path to the patient's ground truth NIfTI mask file.
        """
        if self.no_timepoints:
            return self.base_dir / f"{self.id}{MASK_SUFFIX}{EXT_NIFTI}"
        return (
            self.base_dir
            / self.timepoint
            / f"{self.id}_{self.timepoint}{MASK_SUFFIX}{EXT_NIFTI}"
        )

    # ======================================
    #            DATA LOADING
    # ======================================

    def load_volume(self, modality: str) -> np.ndarray:
        """Returns the 3D volume for the given modality, loading and caching it if needed.

        The volume is loaded from disk on the first call and cached in memory;
        subsequent calls for the same modality return the cached array.

        Args:
            modality: MRI modality string (e.g. 'T1', 'T2', 'FLAIR').

        Returns:
            3D NumPy array with the volume data.

        Raises:
            FileNotFoundError: If the volume file does not exist on disk.
        """
        if modality not in self._volumes:
            vol_path = self.volume_path(modality)
            if not path_exists(vol_path):
                raise FileNotFoundError(f"Volume not found: {vol_path}")
            self._volumes[modality] = nib.load(vol_path).get_fdata()
        return self._volumes[modality]

    @property
    def gt_mask(self) -> np.ndarray:
        """Returns the binary ground truth mask, loading and caching it if needed.

        Returns:
            3D NumPy array with the ground truth mask data.
        """
        if self._gt_mask is None:
            if not path_exists(self.gt_mask_path):
                raise FileNotFoundError(
                    f"Ground truth mask not found at {self.gt_mask_path}"
                )
            self._gt_mask = nib.load(self.gt_mask_path).get_fdata()
        return self._gt_mask

    @property
    def num_slices(self) -> int:
        """Returns the total number of slices in the mask for the current plane.

        Returns:
            Integer count of slices along the axis corresponding to the current plane.

        Raises:
            ValueError: If the current plane is not 'axial', 'coronal', or 'sagittal'.
        """
        mapping = {"axial": 2, "coronal": 1, "sagittal": 0}  # Axis 0=x (sagittal), 1=y (coronal), 2=z (axial) — NIfTI convention.
        if self.plane not in mapping:
            raise ValueError(f"Unrecognised plane: {self.plane}")
        return self.gt_mask.shape[mapping[self.plane]]

    # ======================================
    #             PROCESSING
    # ======================================

    def apply_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Applies the configured enhancement algorithm to an image slice.

        When enhancement is 'GC', the gamma factor stored in self.gamma is
        forwarded to the algorithm.

        Args:
            image: 2D image array to enhance.

        Returns:
            Enhanced image array, or the original image if no enhancement is set.
        """
        if self.enhancement is None:
            return image
        if self.enhancement == "GC":
            return get_algorithm(self.enhancement, gamma=self.gamma).apply(image)
        return get_algorithm(self.enhancement).apply(image)

    # ======================================
    #             EXTRACTION
    # ======================================

    def get_image_slice(self, i: int, modality: str) -> np.ndarray:
        """Extracts the i-th slice of the volume and applies the configured enhancement.

        Args:
            i: Slice index within the current anatomical plane.
            modality: MRI modality string (e.g. 'T1', 'T2', 'FLAIR').

        Returns:
            2D image array after enhancement (if any).
        """
        img_slice = self.load_volume(modality)[self.plane_index(i)]
        return self.apply_enhancement(image=img_slice)

    def get_multichannel_slice(self, i: int) -> np.ndarray:
        """Returns a 3-channel uint8 image for slice i by stacking one channel per modality.

        If fewer than 3 modalities are configured, the last channel is repeated
        to fill all 3 slots. This format is directly compatible with YOLO's
        3-channel (RGB) input, allowing the model to learn from all modalities.

        Args:
            i: Slice index within the current anatomical plane.

        Returns:
            HxWx3 uint8 NumPy array suitable for YOLO input.
        """
        channels = []
        for m in self.modality:
            img_slice = self.get_image_slice(i, m)
            if img_slice.ndim == 3:
                img_slice = img_slice[:, :, 0]  # extract grayscale channel (all channels identical)
            # NIfTI voxel data axes are (x, y, z); slicing a plane gives a 2D array whose
            # first axis is not the image row. Transpose to (row, col) convention for OpenCV.
            img_slice = img_slice.T
            channels.append(normalize_to_uint8(img_slice))

        # Pad to exactly 3 channels by repeating the last one
        while len(channels) < 3:
            channels.append(channels[-1])

        return np.stack(channels[:3], axis=-1)  # HxWx3 uint8

    def lesion_slices_multichannel(self, num_slices: int | None = None) -> list[tuple[int, np.ndarray]]:
        """Returns lesion-containing slices as multi-channel images.

        Note:
            Each HxWx3 array holds three modality channels in memory simultaneously.
            For large volumes with many lesion slices, memory usage can be significant.

        Args:
            num_slices: Maximum number of slices to return. If None or fewer
                lesion slices exist, all lesion slices are returned.

        Returns:
            List of (slice_index, image) tuples where image is HxWx3 uint8.
        """
        indices = self.slices_to_use(num_slices)
        return [(i, self.get_multichannel_slice(i)) for i in indices]

    def get_mask_slice(self, i: int) -> np.ndarray:
        """Extracts the i-th slice of the ground truth mask for the patient's plane.

        Args:
            i: Slice index within the current anatomical plane.

        Returns:
            2D binary mask array for the given slice.
        """
        return self.gt_mask[self.plane_index(i)]

    def plane_index(self, i: int) -> tuple[slice | int, slice | int, slice | int]:
        """Returns NumPy slicing indices for the current plane at position i.

        Args:
            i: Slice index within the current anatomical plane.

        Returns:
            3-tuple of slice or int objects for indexing a 3D volume array.

        Raises:
            ValueError: If the current plane is 'consensus', which does not
                support index extraction.
        """
        if self.plane == PLANE_CONSENSUS:
            raise ValueError(
                "'consensus' is not an anatomical plane and does not support index extraction."
            )

        # NIfTI volume axes: (x=left-right/sagittal, y=front-back/coronal, z=inf-sup/axial).
        mapping = {
            "axial": (slice(None), slice(None), i),
            "coronal": (slice(None), i, slice(None)),
            "sagittal": (i, slice(None), slice(None)),
        }

        return mapping[self.plane]

    # ======================================
    #          LESION DETECTION
    # ======================================

    def lesion_slice_indices(self) -> list[int]:
        """Returns the indices of all slices that contain at least one lesion voxel.

        Returns:
            Sorted list of slice indices where the ground truth mask is non-zero.
        """
        indices = [
            i
            for i in range(self.num_slices)
            if np.any(self.get_mask_slice(i) > 0)
        ]
        return indices

    def slices_to_use(self, num_slices: int | None = None) -> list[int]:
        """Returns the lesion slice indices to extract, respecting the slice budget.

        If num_slices is None or the patient has fewer lesion slices than
        num_slices, all lesion slices are returned. Otherwise, the central
        num_slices slices are selected from the sorted lesion slice list.

        Args:
            num_slices: Maximum number of slices to return. None means no limit.

        Returns:
            List of selected lesion slice indices.
        """
        valid_indices = self.lesion_slice_indices()
        if num_slices is None or len(valid_indices) <= num_slices:
            return valid_indices

        center = len(valid_indices) // 2
        half = num_slices // 2
        start = max(0, center - half)
        end = start + num_slices
        return valid_indices[start:end]

    # ======================================
    #        LESION-BASED EXTRACTION
    # ======================================

    def lesion_slices_by_modality(self, num_slices: int | None = None) -> dict[str, list[tuple[int, np.ndarray]]]:
        """Returns lesion-containing slices organised by modality.

        Args:
            num_slices: Maximum number of slices to return per modality.
                None means no limit.

        Returns:
            Dictionary mapping each modality string to a list of
            (slice_index, slice_array) tuples.
        """
        slices_dict = {}
        indices = self.slices_to_use(num_slices)
        for m in self.modality:
            slice_list = []
            for i in indices:
                img_slice = self.get_image_slice(i, modality=m)
                slice_list.append((i, img_slice))
            slices_dict[m] = slice_list
        return slices_dict

    def lesion_mask_slices(self, num_slices: int | None = None) -> list[tuple[int, np.ndarray]]:
        """Returns lesion-containing ground truth mask slices.

        Args:
            num_slices: Maximum number of slices to return. None means no limit.

        Returns:
            List of (slice_index, mask_array) tuples for selected lesion slices.
        """
        indices = self.slices_to_use(num_slices)
        return [(i, self.get_mask_slice(i)) for i in indices]

    def __repr__(self) -> str:
        """Internal representation of the Patient instance.

        Returns:
            String of the form 'Patient(<id>)'.
        """
        return f"Patient({self.id})"

    def __str__(self) -> str:
        """Human-readable representation of the Patient instance.

        Returns:
            The patient identifier string.
        """
        return self.id
