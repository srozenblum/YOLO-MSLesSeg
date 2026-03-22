from yolo_mslesseg.configs.ConfigBase import ConfigBase
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import DATASETS_DIR, TRAINS_DIR, WEIGHTS_FILE
from yolo_mslesseg.utils.utils import (
    path_exists,
    create_directory,
    delete_directory,
    list_patients,
    trained_model_exists,
    compute_fold,
)

# Configure logger
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
        model,
        epochs: int,
        k_folds: int = 5,
        fold_test=None,
        patient=None,
    ) -> None:
        # --- Shared base attributes ---
        super().__init__(
            model=model,
            epochs=epochs,
            k_folds=k_folds,
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

    def _resolve_execution_mode(self):
        """
        Resolves the execution mode based on the received parameters.

        Supported modes:
            1) k_folds > 1 (cross-validation)
               - Fold mode:    requires fold_test (fold to process).
               - Patient mode: patient is provided and their fold is computed.

            2) k_folds == 1 (train/test split)
               - Group mode:   processes the 'test' group (no folds).
               - Patient mode: patient is provided and must belong to 'test'.

        This method sets internal flags and auxiliary fields used later for
        path construction and path verification.
        """
        self.is_individual_patient = self.patient is not None
        self.is_fold = not self.is_individual_patient and self.fold_test is not None

        if self.is_individual_patient:

            if self.k_folds > 1:
                if self.patient.split == "train":
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

                if self.group == "train":
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

    def _resolve_dataset_paths(self):
        self.dataset_base_dir = DATASETS_DIR / f"{self.model.base_path}"
        self.dataset_fold_dir = self.dataset_base_dir / self.fold_subdir

    def _resolve_training_paths(self):
        # Base training output directory
        if self.single_fold:
            self.train_base_dir = (
                TRAINS_DIR / f"{self.model.base_path}_{self.epochs}epochs" / self.plane
            )

            # Trained YOLO model weights file
            self.model_path = self.train_base_dir / "weights" / WEIGHTS_FILE
            return

        self.train_base_dir = (
            TRAINS_DIR
            / f"{self.model.base_path}_{self.epochs}epochs"
            / self.plane
            / f"fold{self.fold_test}"
        )

        # Trained YOLO model weights file
        self.model_path = self.train_base_dir / "weights" / WEIGHTS_FILE

    def _resolve_patient_paths(self):
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

    def _clean_fold_predictions(self):
        """
        Cleans the pred_masks/ directory for the corresponding plane
        across all patients in the fold.
        """
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

    def _clean_patient_predictions(self):
        """
        Cleans the predicted slices for the corresponding plane for an
        individual patient.
        """
        if path_exists(self.patient_dir["pred_masks"]):
            try:
                delete_directory(self.patient_dir["pred_masks"])
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not delete {self.patient_dir['pred_masks']}: {e}"
                )

    def clean(self) -> None:
        """
        Cleans the predicted slices for the model plane and the active
        execution mode.
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

    def _verify_fold_paths(self):
        """
        Verifies that the input files and output directory exist for the
        patients in the fold.
        - Input:  images directory per patient (images_dir).
        - Output: predicted 2D masks directory per patient (pred_masks_dir).
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

                # images_dir
                if not path_exists(
                    patient_images_subdir
                ):  # Raises exception if not found
                    raise FileNotFoundError(
                        f"Images directory not found for patient {patient_id}: {patient_images_subdir}."
                    )

                # pred_masks_dir
                create_directory(patient_pred_masks_subdir)  # Ensure output exists

    def _verify_patient_paths(self):
        """
        Verifies that the input and output directories exist for an individual patient.
        - Input:  images directory (patient_dir["images"]).
        - Output: predicted 2D masks directory (patient_dir["pred_masks"]).
        """
        # patient_dir["images"]
        if not path_exists(
            self.patient_dir["images"]
        ):  # Raises exception if not found
            raise FileNotFoundError(
                f"Images directory not found for patient {self.patient.id}: {self.patient_dir['images']}."
            )

        # patient_dir["pred_masks"]
        create_directory(self.patient_dir["pred_masks"])  # Ensure output exists

    def _verify_model_path(self):
        """
        Verifies that the trained YOLO model weights file exists.
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

    def verify_paths(self):
        """
        Verifies that the input and output directories exist for prediction generation.

        - Always verifies the existence of the trained model weights file.

        - Fold mode:
            * Verifies paths for all patients in the fold.

        - Individual patient mode:
            * Verifies paths only for the specified patient.
        """
        self._verify_model_path()

        if self.is_individual_patient:
            self._verify_patient_paths()

        elif self.is_fold or self.single_fold:
            self._verify_fold_paths()
