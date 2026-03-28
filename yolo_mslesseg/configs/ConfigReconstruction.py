from yolo_mslesseg.configs.ConfigBase import ConfigBase
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import (
    GT_DIR,
    SPLIT_TEST,
    SPLIT_TRAIN,
    EXT_NIFTI,
    MASK_SUFFIX,
    PRED_VOLS_DIR,
    DATASETS_DIR,
)
from yolo_mslesseg.utils.utils import (
    path_exists,
    create_directory,
    list_patients,
    compute_fold,
)

# Configure logger
logger = get_logger(__file__)


class ConfigReconstruction(ConfigBase):
    """
    Class: ConfigReconstruction

    Description:
        Configuration and path management for the volume reconstruction stage,
        implemented in `reconstruct_volume.py`. Handles verification, creation,
        and cleanup of directories required for both fold-level and individual
        patient execution, ensuring that the predicted 2D masks and ground truth
        volumes are correctly located for generating the final 3D volumes.

    Execution modes:
        1. Individual patient mode (`patient` != None)
           Reconstructs volumes only for the specified patient.

        2. Fold mode (`patient` = None, `fold_test` != None)
           Reconstructs volumes for all patients in the indicated fold.

    Directory conventions:
        datasets/: YOLO dataset split by folds and patients
        └── <enhancement>/
             └── <modality>_<num_slices>slices_<k_folds>folds/
                 ├── <fold_test>/
                 │   ├── PX/
                 │   │   └── <plane>/
                 │   │        ├── images/: input images
                 │   │        ├── labels/: YOLO annotations
                 │   │        ├── GT_masks/: ground truth masks per image
                 │   │        └── pred_masks/: 2D masks predicted by the model
                 │   └── ...
                 └── ...

        pred_vols/: model-predicted volumes (reconstructed from 2D masks)
        └── <enhancement>/
             └── <modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/
                 ├── <fold_test>/
                 │   ├── PX/
                 │   │   └── PX_<plane>.nii.gz
                 │   └── ...
                 └── ...

        GT/: ground truth volumes

    Attributes:
        model (Model):
            Model instance defining the plane, modalities, enhancement, and base_path.

        plane (str):
            Anatomical processing plane ('axial', 'coronal', 'sagital').
            Matches `model.plane`.

        epochs (int):
            Number of epochs of the trained YOLO model.

        k_folds (int):
            Number of cross-validation folds.

        patient (Patient | None, optional):
            Patient instance for individual execution.
            None for fold-level execution.

        fold_test (int | None, optional):
            Test fold number (1, ..., `k_folds`).
            For individual patient execution, it is computed automatically.

        dataset_fold_dir (Path):
            Directory of the corresponding fold inside the dataset.

        pred_vols_base_dir (Path):
            Base directory of the reconstructed volumes for the experiment.

        pred_vols_fold_dir (Path):
            Directory of the reconstructed volumes for the fold.

        gt_dir (Path):
            Base directory of the ground truth volumes.

        # --- Attributes valid only in patient mode ---
        patient_pred_masks (Path | None):
            Directory containing the patient's predicted 2D masks.

        patient_vol_root (Path | None):
            Base directory where the patient's reconstructed volume will be stored.

        patient_pred_vol (Path | None):
            Path to the NIfTI file of the patient's reconstructed volume
            for the model plane.

        patient_gt_vol (Path | None):
            Path to the patient's ground truth NIfTI file.
    """

    def __init__(
        self,
        model: Model,
        epochs: int,
        patient: Patient | None = None,
        fold_test: int | None = None,
    ) -> None:
        """Initialises a ConfigReconstruction instance for the volume reconstruction stage.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            epochs: Number of training epochs of the YOLO model.
            patient: Patient instance for individual execution, or None for fold-level.
            fold_test: Test fold index when using cross-validation, or None.

        Raises:
            ValueError: If the execution mode cannot be determined from the arguments.
        """
        # --- Shared base attributes ---
        super().__init__(
            model=model,
            epochs=epochs,
            patient=patient,
            fold_test=fold_test,
        )

        # GT directory (depends on single_fold, already initialised in base)
        self.gt_dir = GT_DIR / (SPLIT_TEST if self.single_fold else SPLIT_TRAIN)

        # --- Determine execution mode ---
        self._resolve_execution_mode()

        # --- Dataset directories (input) ---
        self._resolve_dataset_paths()

        # --- Volume directories (output) ---
        self._resolve_pred_vols_paths()

        # --- Patient-specific paths (if applicable) ---
        self._resolve_patient_paths()

    # ======================================
    #          CONSTRUCTOR HELPERS
    # ======================================

    def _resolve_execution_mode(self) -> None:
        """Resolves the execution mode and configures the ground truth directory.

        Raises:
            ValueError: If a patient belongs to an incompatible split for the
                chosen k_folds value, or if no valid mode can be determined.
        """
        self.is_individual_patient = self.patient is not None
        self.is_fold = not self.is_individual_patient and self.fold_test is not None

        if self.is_individual_patient:
            # k_folds == 1 → only test patients are allowed
            if self.single_fold:
                self.group = self.patient.split

                if self.group == "train":
                    raise ValueError(
                        f"Patient {self.patient.id} belongs to 'train'. "
                        "With k_folds == 1, individual reconstruction is only allowed "
                        "for patients from 'test'."
                    )

                self.gt_dir = GT_DIR / SPLIT_TEST
                return

            # Test patient → not valid in CV mode
            if getattr(self.patient, "split", None) == "test":
                raise ValueError(
                    f"Patient {self.patient.id} belongs to 'test'. "
                    "With k_folds > 1, only patients from the 'train' split (P1-P53) are allowed."
                )
            # Train patient → compute their fold
            self.fold_test = compute_fold(
                patient_id=self.patient.id,
                k_folds=self.k_folds,
            )
            return

        elif self.single_fold:
            # Fixed train/test split → resolved by group
            self.gt_dir = GT_DIR / SPLIT_TEST
            return

        elif self.is_fold:
            self.gt_dir = GT_DIR / SPLIT_TRAIN
            return

        else:
            raise ValueError(
                "An execution mode must be specified: test fold or individual patient."
            )

    def _resolve_dataset_paths(self) -> None:
        """Resolves the base dataset directory and the active fold subdirectory."""
        self.dataset_base_dir = DATASETS_DIR / f"{self.model.base_path}"
        self.dataset_fold_dir = self.dataset_base_dir / self.fold_subdir

    def _resolve_pred_vols_paths(self) -> None:
        """Resolves the base and fold-specific predicted volumes directories."""
        self.pred_vols_base_dir = (
            PRED_VOLS_DIR / f"{self.model.base_path}_{self.epochs}epochs"
        )
        self.pred_vols_fold_dir = self.pred_vols_base_dir / self.fold_subdir

    def _resolve_patient_paths(self) -> None:
        """Resolves the predicted masks, volume output, and GT paths for an individual patient.

        Has no effect when running in fold-level mode.
        """
        if not self.is_individual_patient:
            return

        self.patient_pred_masks = (
            self.dataset_base_dir
            / self.fold_subdir
            / self.patient.id
            / self.plane
            / "pred_masks"
        )
        self.patient_vol_root = self.pred_vols_base_dir / self.fold_subdir / self.patient.id

        # NIfTI files (predicted volume and ground truth)
        self.patient_pred_vol = (
            self.patient_vol_root / f"{self.patient.id}_{self.plane}{EXT_NIFTI}"
        )
        self.patient_gt_vol = (
            self.gt_dir
            / self.patient.id
            / f"{self.patient.id}{MASK_SUFFIX}{EXT_NIFTI}"
        )

    # ======================================
    #               CLEANUP
    # ======================================

    def _clean_fold_volumes(self) -> None:
        """Cleans the reconstructed NIfTI volumes for the current plane across all patients in the fold."""
        if path_exists(self.pred_vols_fold_dir):
            patients = list_patients(self.pred_vols_fold_dir)

            for patient_id in patients:
                patient_pred_vols_dir = self.pred_vols_fold_dir / patient_id
                if not patient_pred_vols_dir.is_dir():
                    continue

                # Delete NIfTI file for the model plane
                for file in patient_pred_vols_dir.iterdir():
                    if self.plane.lower() in file.name.lower() and file.suffixes[
                        -2:
                    ] == [".nii", ".gz"]:
                        try:
                            file.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {file}: {e}")

    def _clean_patient_volume(self) -> None:
        """Cleans the reconstructed NIfTI volume for an individual patient."""
        if path_exists(self.patient_vol_root):
            try:
                self.patient_pred_vol.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Could not delete volume: {e}")

    def clean(self) -> None:
        """Cleans reconstructed volumes for the active execution mode.

        Raises:
            ValueError: If neither a fold nor a patient is specified.
        """
        if self.is_individual_patient:
            self._clean_patient_volume()
            return

        if self.is_fold or self.single_fold:
            self._clean_fold_volumes()
            return

        raise ValueError("A fold or a patient must be specified.")

    # ======================================
    #            VERIFICATION
    # ======================================

    def _verify_fold_paths(self) -> None:
        """Verifies input and output paths for all patients in the active fold.

        Raises:
            FileNotFoundError: If the dataset fold directory or a patient's
                pred_masks directory does not exist.
        """
        if not path_exists(self.dataset_fold_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_fold_dir}."
            )

        patients = list_patients(self.dataset_fold_dir)

        for patient_id in patients:
            patient_dir_path = self.dataset_fold_dir / patient_id / self.plane
            patient_pred_masks_subdir = patient_dir_path / "pred_masks"
            patient_pred_vols_fold_dir = self.pred_vols_fold_dir / patient_id

            # pred_masks_dir
            if not path_exists(patient_pred_masks_subdir):
                raise FileNotFoundError(
                    f"pred_masks directory not found for patient {patient_id}: {patient_pred_masks_subdir}."
                )

            # pred_vols_fold_dir
            create_directory(patient_pred_vols_fold_dir)  # Ensure output exists

    def _verify_patient_paths(self) -> None:
        """Verifies input and output paths for an individual patient.

        Raises:
            FileNotFoundError: If the patient's pred_masks directory does not exist.
        """
        if not path_exists(self.patient_pred_masks):  # Raises exception if not found
            raise FileNotFoundError(
                f"pred_masks not found for patient: {self.patient_pred_masks}"
            )
        create_directory(self.patient_vol_root)  # Ensure output exists

    def _verify_gt_paths(self) -> None:
        """Verifies that the ground truth directory exists.

        Raises:
            FileNotFoundError: If the GT directory does not exist.
        """
        if not path_exists(self.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

    def verify_paths(self) -> None:
        """Verifies that all required paths exist for volume reconstruction.

        Always checks the GT directory. Then delegates path verification to
        _verify_patient_paths or _verify_fold_paths based on the active mode.
        """
        self._verify_gt_paths()

        if self.is_individual_patient:
            self._verify_patient_paths()

        elif self.is_fold or self.single_fold:
            self._verify_fold_paths()
