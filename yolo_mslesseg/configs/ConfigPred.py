"""
Module: ConfigPred

Description:
    Defines the ConfigPred class, which manages configuration and path
    resolution for the prediction generation stage (generate_predictions.py).

Usage:
    from yolo_mslesseg.configs.ConfigPred import ConfigPred
    config = ConfigPred(model=model, epochs=50, fold_test=1)

Inputs:
    None. Provides the ConfigPred class definition.

Outputs:
    None. Provides the ConfigPred class definition.

Relationships:
    - Used exclusively by generate_predictions.py.
    - Subclasses ConfigBase.
    - Depends on Model and Patient from utils/.
    - Depends on constants.py for directory paths.
"""

from pathlib import Path

from yolo_mslesseg.configs.ConfigBase import ConfigBase
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import DATASETS_DIR, TRAINS_DIR, WEIGHTS_FILE, SPLIT_TRAIN, WEIGHTS_SUBDIR
from yolo_mslesseg.utils.utils import (
    path_exists,
    create_directory,
    delete_directory,
    list_patients,
    trained_model_exists,
    compute_fold,
)

logger = get_logger(__file__)


class ConfigPred(ConfigBase):
    """
    Class: ConfigPred

    Description:
        Configuration and path management for the prediction generation stage,
        implemented in `generate_predictions.py`. Handles verification, creation,
        and cleanup of directories required for both fold-level and individual
        patient execution, ensuring that the input images are correctly located
        for generating the predicted masks.

    Execution modes:
        1. Individual patient mode (`patient` != None):
           Generates predictions only for the specified patient.

        2. Fold mode (`patient` = None, `fold_test` != None):
           Generates predictions for all patients in the indicated fold.

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

        trains/: YOLO training results (model weights)

    Attributes:
        model (Model):
            Model instance defining the plane, modalities, enhancement, and base_path.

        plane (str):
            Anatomical processing plane ('axial', 'coronal', 'sagittal').
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

        train_base_dir (Path):
            Base training output directory for the corresponding fold.

        # --- Attributes valid only in patient mode ---
        patient_root (Path | None):
            Base directory of the patient within their fold.

        patient_dir (dict[str, Path] | None):
            Dictionary of patient subdirectories:
                - images/
                - pred_masks/
    """

    def __init__(
        self,
        model: Model,
        epochs: int,
        fold_test: int | None = None,
        patient: Patient | None = None,
    ) -> None:
        """Initialises a ConfigPred instance for the prediction generation stage.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            epochs: Number of training epochs of the YOLO model.
            fold_test: Test fold index when using cross-validation, or None.
            patient: Patient instance for individual execution, or None for fold-level.

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

        # --- Determine execution mode ---
        self._resolve_execution_mode()

        # --- Dataset directories (input and output) ---
        self._resolve_dataset_paths()

        # --- Training directories (input) ---
        self._resolve_training_paths()

        # --- Patient-specific paths (if applicable) ---
        self._resolve_patient_paths()

    # ======================================
    #          CONSTRUCTOR HELPERS
    # ======================================

    def _resolve_execution_mode(self) -> None:
        """Resolves the execution mode and sets internal flags for path construction.

        Supports four modes: fold-level CV (k_folds > 1), patient-level CV,
        fixed-split group (k_folds == 1), and fixed-split patient. Sets
        is_individual_patient, is_fold, fold_test, and group accordingly.

        Raises:
            ValueError: If a test or train patient is used in an incompatible mode,
                or if no valid execution mode can be determined.
        """
        self.is_individual_patient = self.patient is not None
        self.is_fold = not self.is_individual_patient and self.fold_test is not None

        if self.is_individual_patient:

            if self.k_folds > 1:
                if self.patient.split == SPLIT_TRAIN:
                    # Train patient → belongs to a fold
                    self.fold_test = compute_fold(
                        patient_id=self.patient.id,
                        k_folds=self.k_folds,
                    )
                else:
                    # Test patient → not valid in CV mode
                    raise ValueError(
                        f"Patient {self.patient.id} belongs to 'test'. "
                        "With k_folds > 1, only patients from the 'train' split (P1-P53) are allowed."
                    )

            else:
                # k_folds == 1 → direct train/test split
                self.group = self.patient.split

                if self.group == SPLIT_TRAIN:
                    raise ValueError(
                        f"Cannot generate predictions for patient {self.patient.id} with "
                        "k_folds == 1 if they belong to 'train'. The model was trained on that subset."
                    )

            return

        elif self.single_fold:
            # k_folds == 1 and full execution → use test group by default
            return

        elif self.is_fold:
            return

        else:
            raise ValueError(
                "An execution mode must be specified: test fold or individual patient."
            )

    def _resolve_dataset_paths(self) -> None:
        """Resolves the base dataset directory and the active fold subdirectory."""
        self.dataset_base_dir = DATASETS_DIR / f"{self.model.base_path}"
        self.dataset_fold_dir = self.dataset_base_dir / self.fold_subdir

    def _resolve_training_paths(self) -> None:
        """Resolves the training output directory and the YOLO model weights path."""
        # Base training output directory
        if self.single_fold:
            self.train_base_dir = (
                TRAINS_DIR / f"{self.model.base_path}_{self.epochs}epochs" / self.plane
            )

            self.model_path = self.train_base_dir / WEIGHTS_SUBDIR / WEIGHTS_FILE
            return

        self.train_base_dir = (
            TRAINS_DIR
            / f"{self.model.base_path}_{self.epochs}epochs"
            / self.plane
            / f"fold{self.fold_test}"
        )

        self.model_path = self.train_base_dir / WEIGHTS_SUBDIR / WEIGHTS_FILE

    def _resolve_patient_paths(self) -> None:
        """Resolves the input images and output pred_masks paths for an individual patient.

        Has no effect when running in fold-level mode.
        """
        if not self.is_individual_patient:
            return

        self.patient_root = (
            self.dataset_base_dir / self.fold_subdir / self.patient.id / self.plane
        )

        self.patient_dir = {
            subdir: self.patient_root / subdir for subdir in ["images", "pred_masks"]
        }

    # ======================================
    #               CLEANUP
    # ======================================

    def _clean_fold_predictions(self) -> None:
        """Cleans the pred_masks/ directory for all patients in the active fold."""
        if path_exists(self.dataset_fold_dir):
            patients = list_patients(self.dataset_fold_dir)

            for patient_id in patients:
                patient_pred_masks_subdir = (
                    self.dataset_fold_dir / patient_id / self.plane / "pred_masks"
                )
                if path_exists(patient_pred_masks_subdir):
                    try:
                        delete_directory(patient_pred_masks_subdir)
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Could not delete {patient_pred_masks_subdir}: {e}"
                        )

    def _clean_patient_predictions(self) -> None:
        """Cleans the predicted 2D mask slices for an individual patient."""
        if path_exists(self.patient_dir["pred_masks"]):
            try:
                delete_directory(self.patient_dir["pred_masks"])
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not delete {self.patient_dir['pred_masks']}: {e}"
                )

    def clean(self) -> None:
        """Cleans predicted 2D mask slices for the active execution mode.

        Raises:
            ValueError: If neither a fold nor a patient is specified.
        """
        if self.is_individual_patient:
            self._clean_patient_predictions()

        elif self.is_fold or self.single_fold:
            self._clean_fold_predictions()

        else:
            raise ValueError("A fold or a patient must be specified.")

    # ======================================
    #            VERIFICATION
    # ======================================

    def _verify_fold_paths(self) -> None:
        """Verifies input and output paths for all patients in the active fold.

        Raises:
            FileNotFoundError: If the dataset fold directory or an images
                subdirectory does not exist.
        """
        if not path_exists(self.dataset_fold_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_fold_dir}."
            )

        patients = list_patients(self.dataset_fold_dir)

        for patient_id in patients:
            patient_dir_path = self.dataset_fold_dir / patient_id / self.plane
            if patient_dir_path.is_dir():
                patient_images_subdir = patient_dir_path / "images"
                patient_pred_masks_subdir = patient_dir_path / "pred_masks"

                if not path_exists(patient_images_subdir):
                    raise FileNotFoundError(
                        f"Images directory not found for patient {patient_id}: {patient_images_subdir}."
                    )

                create_directory(patient_pred_masks_subdir)

    def _verify_patient_paths(self) -> None:
        """Verifies input and output paths for an individual patient.

        Raises:
            FileNotFoundError: If the patient's images directory does not exist.
        """
        if not path_exists(self.patient_dir["images"]):
            raise FileNotFoundError(
                f"Images directory not found for patient {self.patient.id}: {self.patient_dir['images']}."
            )

        create_directory(self.patient_dir["pred_masks"])

    def _verify_model_path(self) -> None:
        """Verifies that the trained YOLO model weights file exists.

        Raises:
            FileNotFoundError: If the model weights file does not exist.
        """
        if self.single_fold:
            if not path_exists(self.model_path):
                raise FileNotFoundError(
                    f"Trained model not found at {self.model_path}."
                )
            return

        if not trained_model_exists(
            model=self.model, epochs=self.epochs, fold_test=self.fold_test
        ):  # Raises exception if not found
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}."
            )

    def verify_paths(self) -> None:
        """Verifies that all required paths exist for prediction generation.

        Always checks the model weights file. Then delegates path verification
        to _verify_patient_paths or _verify_fold_paths based on the active mode.
        """
        self._verify_model_path()

        if self.is_individual_patient:
            self._verify_patient_paths()

        elif self.is_fold or self.single_fold:
            self._verify_fold_paths()
