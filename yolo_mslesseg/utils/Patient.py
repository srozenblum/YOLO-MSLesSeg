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
            Anatomical orientation ('axial', 'coronal', 'sagital') or 'consenso'.

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

        base_dir (Path):
            Base path of the patient in the MSLesSeg dataset.

        gt_mask (np.ndarray):
            Ground truth volume loaded in memory.

        _volumes (dict[str, np.ndarray]):
            Dictionary of volumes loaded per modality.
    """

    def __init__(
        self, id, plane, timepoint="T1", modality=None, enhancement=None, gt_mask=None
    ):
        modality = modality or list(MODALITIES)

        # --- Argument validation ---
        self._validate_args(id, plane, timepoint, enhancement, modality)

        # --- Core attributes ---
        self._set_core_attributes(
            id, plane, timepoint, modality, enhancement, gt_mask
        )

    # ======================================
    #        CONSTRUCTOR HELPERS
    # ======================================

    def _validate_args(self, id, plane, timepoint, enhancement, modality):
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

    def _resolve_split(self):
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
        self, id, plane, timepoint, modality, enhancement, gt_mask
    ):
        self.id = id
        self.split, self.base_dir = self._resolve_split()
        self.plane = plane
        self.timepoint = timepoint
        self.no_timepoints = not any(
            (self.base_dir / tp).exists() for tp in TIMEPOINTS
        )
        self.enhancement = enhancement.upper() if enhancement else None
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
    def is_train(self):
        return self.split == SPLIT_TRAIN

    @property
    def is_test(self):
        return self.split == SPLIT_TEST

    def volume_path(self, modality):
        if self.no_timepoints:
            return self.base_dir / f"{self.id}_{modality}{EXT_NIFTI}"
        return (
            self.base_dir
            / self.timepoint
            / f"{self.id}_{self.timepoint}_{modality}{EXT_NIFTI}"
        )

    @property
    def gt_mask_path(self):
        """Returns the path to the ground truth mask."""
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

    def load_volume(self, modality):
        """
        Returns the 3D volume for the given modality and caches it in the
        internal _volumes dictionary if not already loaded.
        """
        if modality not in self._volumes:
            vol_path = self.volume_path(modality)
            if not path_exists(vol_path):
                raise FileNotFoundError(f"Volume not found for modality {modality}.")
            self._volumes[modality] = nib.load(vol_path).get_fdata()
        return self._volumes[modality]

    @property
    def gt_mask(self):
        """Returns the binary ground truth mask."""
        if self._gt_mask is None:
            if not path_exists(self.gt_mask_path):
                raise FileNotFoundError(
                    f"Ground truth mask not found at {self.gt_mask_path}"
                )
            self._gt_mask = nib.load(self.gt_mask_path).get_fdata()
        return self._gt_mask

    @property
    def num_slices(self):
        """Returns the total number of slices in the mask for the current plane."""
        mapping = {"axial": 2, "coronal": 1, "sagital": 0}
        if self.plane not in mapping:
            raise ValueError(f"Unrecognised plane: {self.plane}")
        return self.gt_mask.shape[mapping[self.plane]]

    # ======================================
    #             PROCESSING
    # ======================================

    def apply_enhancement(self, image):
        """Applies the configured enhancement algorithm, if any."""
        if self.enhancement is None:
            return image
        return get_algorithm(self.enhancement).apply(image)

    # ======================================
    #             EXTRACTION
    # ======================================

    def get_image_slice(self, i, modality):
        """
        Extracts the i-th slice of the volume for the patient's plane and modality,
        and applies the corresponding enhancement algorithm.
        """
        img_slice = self.load_volume(modality)[self.plane_index(i)]
        return self.apply_enhancement(image=img_slice)

    def get_multichannel_slice(self, i):
        """
        Returns a 3-channel uint8 image for slice i by stacking one channel
        per modality. If fewer than 3 modalities are configured, the last
        channel is repeated to fill all 3 slots.

        This format is directly compatible with YOLO's 3-channel (RGB) input,
        allowing the model to jointly learn from all available modalities.
        """
        channels = []
        for m in self.modality:
            img_slice = self.get_image_slice(i, m).T  # transpose to match image axes
            channels.append(normalize_to_uint8(img_slice))

        # Pad to exactly 3 channels by repeating the last one
        while len(channels) < 3:
            channels.append(channels[-1])

        return np.stack(channels[:3], axis=-1)  # HxWx3 uint8

    def lesion_slices_multichannel(self, num_slices=None):
        """
        Returns lesion-containing slices as multi-channel images.
        Each entry is a tuple (slice_index, image) where image is HxWx3 uint8.
        """
        indices = self.slices_to_use(num_slices)
        return [(i, self.get_multichannel_slice(i)) for i in indices]

    def get_mask_slice(self, i):
        """Extracts the i-th slice of the mask for the patient's plane."""
        return self.gt_mask[self.plane_index(i)]

    def plane_index(self, i):
        """
        Returns a tuple of slicing indices corresponding to the current plane
        and the given position index.
        """
        if self.plane == "consenso":
            raise ValueError(
                "'consenso' is not an anatomical plane and does not support index extraction."
            )

        mapping = {
            "axial": (slice(None), slice(None), i),
            "coronal": (slice(None), i, slice(None)),
            "sagital": (i, slice(None), slice(None)),
        }

        return mapping[self.plane]

    # ======================================
    #          LESION DETECTION
    # ======================================

    def lesion_slice_indices(self):
        """Returns the indices of slices that contain lesion in the mask."""
        indices = [
            i
            for i in range(self.num_slices)
            if np.any(self.get_mask_slice(i) > 0)
        ]
        return indices

    def slices_to_use(self, num_slices=None):
        """
        Returns the lesion slice indices to use.
        - If num_slices is None or there are fewer lesion slices than num_slices,
          all lesion slices are returned.
        - If there are more lesion slices than num_slices, the central ones are returned.
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

    def lesion_slices_by_modality(self, num_slices=None):
        """
        Returns all lesion-containing volume slices organised by modality as a
        dictionary. Each key is a modality (T1, T2, FLAIR) and the value is a
        list of tuples (volume_index, slice):

        {"T1": [(i0, slice0), (i1, slice1), ...],
         "T2": [(i0, slice0), ...],
         "FLAIR": [...]}
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

    def lesion_mask_slices(self, num_slices=None):
        """
        Returns all lesion-containing mask slices as a list of tuples
        [(index, slice), ...].
        """
        indices = self.slices_to_use(num_slices)
        return [(i, self.get_mask_slice(i)) for i in indices]

    def __repr__(self):
        """Internal representation of the Patient instance."""
        return f"Patient({self.id})"

    def __str__(self):
        """Human-readable representation of the Patient instance."""
        return self.id
