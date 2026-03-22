from yolo_mslesseg.configs.ConfigBase import ConfigBase
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import (
    GT_DIR,
    SPLIT_TEST,
    SPLIT_TRAIN,
    PRED_VOLS_DIR,
    ANATOMICAL_PLANES,
    PLANES,
    EXT_NIFTI,
    MASK_SUFFIX,
)
from yolo_mslesseg.utils.utils import path_exists, list_patients, compute_fold

# Configure logger
logger = get_logger(__file__)


class ConfigConsensus(ConfigBase):
    """
    Class: ConfigConsensus

    Description:
        Configuration and path management for the consensus volume generation stage,
        implemented in `generate_consensus.py`. Handles verification, creation, and
        cleanup of the directories required for both fold-level and individual patient
        execution, ensuring that the predicted volumes needed to generate the consensus
        masks are correctly available.

    Execution modes:
        1. Individual patient mode (`patient` != None)
           Generates the consensus only for the specified patient.

        2. Fold mode (`patient` = None, `fold_test` != None)
           Generates the consensus for all patients in the indicated fold.

    Directory conventions:
        pred_vols/: model-predicted volumes
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
            Fixed anatomical plane label ('consenso').

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

        pred_vols_base_dir (Path):
            Base directory of the reconstructed volumes for the experiment.

        pred_vols_fold_dir (Path):
            Directory of the reconstructed volumes for the fold.

        gt_dir (Path):
            Base directory of the ground truth volumes.

        # --- Attributes valid only in patient mode ---
        patient_vol_root (Path | None):
            Directory containing the patient's reconstructed volumes.

        patient_pred_vols (dict[str, Path] | None):
            Dictionary of paths to the patient's predicted volumes per plane.

        patient_gt_vol (Path | None):
            Path to the patient's ground truth NIfTI file.
    """

    def __init__(
        self,
        model,
        epochs: int,
        k_folds: int = 5,
        patient=None,
        fold_test=None,
    ) -> None:
        # --- Shared base attributes ---
        super().__init__(
            model=model,
            epochs=epochs,
            k_folds=k_folds,
            patient=patient,
            fold_test=fold_test,
        )

        # Override: consenso stage always uses the 'consenso' plane label
        self.plane = "consenso"

        # GT directory (depends on single_fold, already initialised in base)
        self.gt_dir = GT_DIR / (SPLIT_TEST if self.single_fold else SPLIT_TRAIN)

        # --- Determine execution mode ---
        self._resolve_execution_mode()

        # --- Volume directories (input and output) ---
        self._resolve_pred_vols_paths()

        # --- Patient-specific paths (if applicable) ---
        self._resolve_patient_paths()

    # ======================================
    #          CONSTRUCTOR HELPERS
    # ======================================

    def _resolve_execution_mode(self):
        """
        Resolves the consensus execution mode based on the received parameters.

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

        # k_folds > 1 → fold mode requires fold_test
        self.is_fold = (
            (not self.single_fold)
            and (not self.is_individual_patient)
            and (self.fold_test is not None)
        )

        # k_folds == 1 → group mode (no patient) processes the test group
        self.is_group = self.single_fold and (not self.is_individual_patient)

        if self.is_individual_patient:
            if self.single_fold:
                # In single-fold mode, the patient must belong to 'test'
                if getattr(self.patient, "split", None) != "test":
                    raise ValueError(
                        f"Cannot generate consensus for patient {self.patient.id} with k_folds == 1 "
                        "if they belong to 'train'. The model was trained on that subset."
                    )
                self.group = "test"
                self.fold_test = None
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

        if self.is_fold:
            return

        if self.is_group:
            self.group = "test"
            self.fold_test = None
            return

        raise ValueError(
            "An execution mode must be specified: test fold or individual patient."
        )

    def _resolve_pred_vols_paths(self):
        self.pred_vols_base_dir = (
            PRED_VOLS_DIR / f"{self.model.base_path}_{self.epochs}epochs"
        )
        self.pred_vols_fold_dir = self.pred_vols_base_dir / self.fold_subdir

    def _resolve_patient_paths(self):
        if not self.is_individual_patient:
            return

        self.patient_vol_root = self.pred_vols_fold_dir / self.patient.id

        # Dictionary of predicted volumes per anatomical plane
        self.patient_pred_vols = {
            plane: self.patient_vol_root / f"{self.patient.id}_{plane}{EXT_NIFTI}"
            for plane in PLANES
        }

        # Ground truth
        self.patient_gt_vol = (
            self.gt_dir
            / self.patient.id
            / f"{self.patient.id}{MASK_SUFFIX}{EXT_NIFTI}"
        )

    # ======================================
    #               CLEANUP
    # ======================================

    def _clean_fold_consensus_volumes(self):
        """
        Cleans the consensus files for the corresponding plane across all
        patients in the fold.
        """
        if path_exists(self.pred_vols_fold_dir):
            patients = list_patients(self.pred_vols_fold_dir)

            for patient_id in patients:
                patient_dir = self.pred_vols_fold_dir / patient_id
                if not patient_dir.is_dir():
                    continue

                # Delete only consensus NIfTI files
                for file in patient_dir.iterdir():
                    if "consenso" in file.name.lower() and file.name.endswith(
                        f"{EXT_NIFTI}"
                    ):
                        try:
                            file.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {file}: {e}")

    def _clean_patient_consensus_volume(self):
        """
        Cleans the consensus file for an individual patient.
        """
        consensus_path = self.patient_pred_vols["consenso"]
        if path_exists(consensus_path):
            try:
                consensus_path.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Could not delete consensus volume: {e}")

    def clean(self) -> None:
        """
        Cleans the consensus volumes for the model plane and the active
        execution mode.

        - Fold mode:
          Cleans the consensus volumes for all patients in the fold.

        - Individual patient mode:
          Cleans only the consensus volume for the specified patient,
          without affecting the rest of the fold.
        """
        if self.is_individual_patient:
            self._clean_patient_consensus_volume()
            return

        elif self.is_fold or self.single_fold:
            self._clean_fold_consensus_volumes()
            return

        raise ValueError("A fold or a patient must be specified.")

    # ======================================
    #            VERIFICATION
    # ======================================

    def _verify_fold_paths(self):
        """
        Verifies that the input files and output directory exist for the
        patients in the fold.
        - Input:  predicted volumes for each of the 3 anatomical planes.
        - Output: same directory as input (verified implicitly).
        """
        patients = list_patients(self.pred_vols_fold_dir)

        for patient_id in patients:
            patient_root = self.pred_vols_fold_dir / patient_id

            # Volume per anatomical plane
            for plane in ANATOMICAL_PLANES:
                vol_path = patient_root / f"{patient_id}_{plane}{EXT_NIFTI}"
                if not path_exists(vol_path):  # Raises exception if not found
                    raise FileNotFoundError(
                        f"Missing {plane} volume for patient {patient_id}: {vol_path}."
                    )

    def _verify_patient_paths(self):
        """
        Verifies that the input files and output directory exist for an individual patient.
        - Input:  predicted volumes for each of the 3 anatomical planes (patient_pred_vols).
        - Output: same directory as input (verified implicitly).
        """
        # patient_pred_vols per anatomical plane
        for plane in ANATOMICAL_PLANES:
            vol_path = self.patient_vol_root / f"{self.patient.id}_{plane}{EXT_NIFTI}"
            if not path_exists(vol_path):  # Raises exception if not found
                raise FileNotFoundError(
                    f"Missing predicted {plane} volume for patient {self.patient.id}: {vol_path}."
                )

    def _verify_gt_paths(self):
        """
        Verifies that the ground truth volumes directory exists.
        """
        if not path_exists(self.gt_dir):  # Raises exception if not found
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

    def verify_paths(self):
        """
        Verifies that the input and output directories exist for consensus generation.

        - Always verifies the existence of the ground truth volumes directory.

        - Fold mode:
            * Verifies paths for all patients in the fold.

        - Individual patient mode:
            * Verifies paths only for the specified patient.
        """

        self._verify_gt_paths()

        if self.is_individual_patient:
            self._verify_patient_paths()

        elif self.is_fold or self.single_fold:
            self._verify_fold_paths()
