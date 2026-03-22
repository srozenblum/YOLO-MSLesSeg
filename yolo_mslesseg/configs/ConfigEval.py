from yolo_mslesseg.configs.ConfigBase import ConfigBase
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import (
    GT_DIR,
    PLANES,
    PRED_VOLS_DIR,
    EXT_JSON,
    EXT_NIFTI,
    MASK_SUFFIX,
    RESULTS_DIR,
    RESULTS_SUFFIX,
    RESULTS_GLOBAL_PREFIX,
    SPLIT_TEST,
    SPLIT_TRAIN,
)
from yolo_mslesseg.utils.utils import (
    path_exists,
    create_directory,
    list_patients,
    compute_fold,
)

# Configure logger
logger = get_logger(__file__)


class ConfigEval(ConfigBase):
    """
    Class: ConfigEval

    Description:
        Configuration and path management for the evaluation stages, implemented
        in `eval.py` (per-patient or per-fold metrics) and `average_folds.py`
        (global experiment metrics). Handles verification, creation, and cleanup
        of directory structures for both individual patients and folds, ensuring
        correct localisation of predicted and ground truth volumes for metric
        computation.

    Execution modes:
        1. Individual patient mode (`patient` ≠ None)
           Computes metrics only for the specified patient.

        2. Fold mode (`patient` = None, `fold_test` ≠ None)
           Computes metrics for all patients in the indicated fold.

        3. Experiment mode (`patient` = None and `fold_test` = None)
           Computes global experiment metrics (fold average).
           Requires individual fold results to already exist.

    Directory conventions:
        results/: evaluation metrics
        └── <enhancement>/
             └── <modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/
                 ├── global_<plane>_results.json
                 ├── <fold_test>/
                 │   ├── <fold_test>_<plane>_results.json
                 │   ├── PX/
                 │   │   └── PX_<plane>_results.json
                 │   └── ...
                 └── ...

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
            Anatomical processing plane ('axial', 'coronal', 'sagital').
            If `forced_plane` is provided, it overrides the model's plane.

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

        results_fold_json (Path):
            Path to the JSON file with the fold's metrics.

        results_experiment_json (Path):
            Path to the JSON file with the global experiment metrics.

        # --- Attributes valid only in patient mode ---
        patient_vol_root (Path | None):
            Base directory of the reconstructed volumes for the patient.

        patient_results_root (Path | None):
            Base directory of the patient's metrics.

        patient_pred_vol (Path | None):
            Path to the NIfTI file of the patient's reconstructed volume
            for the model plane.

        patient_gt_vol (Path | None):
            Path to the patient's ground truth NIfTI file.

        patient_results_json (Path | None):
            Path to the JSON file with the patient's metrics.
    """

    def __init__(
        self,
        model: Model,
        epochs: int,
        k_folds: int = 5,
        patient: Patient | None = None,
        fold_test: int | None = None,
        forced_plane: str | None = None,
    ) -> None:
        """Initialises a ConfigEval instance for the evaluation stage.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            epochs: Number of training epochs of the YOLO model.
            k_folds: Number of cross-validation folds (1 for a fixed split).
            patient: Patient instance for individual execution, or None for fold-level.
            fold_test: Test fold index when using cross-validation, or None.
            forced_plane: Plane label overriding the model's plane, or None.

        Raises:
            ValueError: If forced_plane is not a valid plane identifier.
        """
        # --- Validate forced_plane ---
        if forced_plane is not None and forced_plane not in PLANES:
            raise ValueError(
                f"forced_plane '{forced_plane}' is not valid. Must be one of {PLANES}."
            )

        # --- Shared base attributes ---
        super().__init__(
            model=model,
            epochs=epochs,
            k_folds=k_folds,
            patient=patient,
            fold_test=fold_test,
        )

        # Override plane if forced_plane is provided
        if forced_plane is not None:
            self.plane = forced_plane

        # GT directory (depends on single_fold, already initialised in base)
        self.gt_dir = GT_DIR / (SPLIT_TEST if self.single_fold else SPLIT_TRAIN)

        # --- Determine execution mode ---
        self._resolve_execution_mode()

        # --- Volume directories (input) ---
        self._resolve_pred_vols_paths()

        # --- Results directories (output) ---
        self._resolve_results_paths()

        # --- Patient-specific paths (if applicable) ---
        self._resolve_patient_paths()

    # ======================================
    #          CONSTRUCTOR HELPERS
    # ======================================

    def _resolve_execution_mode(self) -> None:
        """Resolves the execution mode and sets internal flags for path construction.

        Raises:
            ValueError: If a patient belongs to an incompatible split for the
                chosen k_folds value.
        """
        self.is_individual_patient = self.patient is not None

        # k_folds == 1: no folds. Evaluates the full test/ split or a single test patient.
        if self.single_fold:
            self.is_fold = not self.is_individual_patient
            self.is_experiment = False
            self.fold_test = None

            if self.is_individual_patient:
                if getattr(self.patient, "split", None) != "test":
                    raise ValueError(
                        f"Cannot evaluate patient {self.patient.id} with k_folds == 1 "
                        "if they belong to 'train'. The model was trained on that subset."
                    )
            return

        # k_folds > 1: standard behaviour (fold/patient/experiment)
        self.is_fold = not self.is_individual_patient and self.fold_test is not None
        self.is_experiment = not self.is_individual_patient and self.fold_test is None

        if self.is_individual_patient:
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

    def _resolve_pred_vols_paths(self) -> None:
        """Resolves the base and fold-specific predicted volumes directories."""
        self.pred_vols_base_dir = (
            PRED_VOLS_DIR / f"{self.model.base_path}_{self.epochs}epochs"
        )
        self.pred_vols_fold_dir = self.pred_vols_base_dir / self.fold_subdir

    def _resolve_results_paths(self) -> None:
        """Resolves the base, fold-specific, and global results directories and JSON paths."""
        self.results_base_dir = (
            RESULTS_DIR / f"{self.model.base_path}_{self.epochs}epochs"
        )
        self.results_fold_dir = self.results_base_dir / self.fold_subdir

        # Fold metrics JSON
        if self.single_fold:
            self.results_fold_json = (
                self.results_base_dir / f"{RESULTS_GLOBAL_PREFIX}{self.plane}{RESULTS_SUFFIX}{EXT_JSON}"
            )
        else:
            self.results_fold_json = (
                self.results_fold_dir
                / f"fold{self.fold_test}_{self.plane}{RESULTS_SUFFIX}{EXT_JSON}"
            )

        # Global experiment metrics JSON (fold average)
        self.results_experiment_json = (
            self.results_base_dir / f"{RESULTS_GLOBAL_PREFIX}{self.plane}{RESULTS_SUFFIX}{EXT_JSON}"
        )

    def _resolve_patient_paths(self) -> None:
        """Resolves the predicted volume, GT, and results paths for an individual patient.

        Has no effect when running in fold-level or experiment mode.
        """
        if not self.is_individual_patient:
            return

        self.patient_vol_root = self.pred_vols_base_dir / self.fold_subdir / self.patient.id
        self.patient_results_root = self.results_base_dir / self.fold_subdir / self.patient.id

        # NIfTI files (predicted and ground truth volumes)
        self.patient_pred_vol = (
            self.patient_vol_root / f"{self.patient.id}_{self.plane}{EXT_NIFTI}"
        )
        self.patient_gt_vol = (
            self.gt_dir
            / self.patient.id
            / f"{self.patient.id}{MASK_SUFFIX}{EXT_NIFTI}"
        )

        # Patient metrics JSON
        self.patient_results_json = (
            self.patient_results_root
            / f"{self.patient.id}_{self.plane}{RESULTS_SUFFIX}{EXT_JSON}"
        )

    # ======================================
    #               CLEANUP
    # ======================================

    def _clean_fold_results(self) -> None:
        """
        Cleans the fold metrics JSON and the individual JSON files for all patients.
        """
        # Fold metrics JSON
        if path_exists(self.results_fold_json):
            try:
                self.results_fold_json.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Could not delete {self.results_fold_json}: {e}")

        if path_exists(self.results_fold_dir):
            patients = list_patients(self.results_fold_dir)

            # Individual per-patient JSON files
            for patient_id in patients:
                patient_result_dir = self.results_fold_dir / patient_id
                if not patient_result_dir.is_dir():
                    continue

                for file in patient_result_dir.iterdir():
                    if (
                        file.is_file()
                        and file.suffix.lower() == EXT_JSON
                        and self.plane.lower() in file.name.lower()
                    ):
                        try:
                            file.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {file}: {e}")

    def _clean_experiment_results(self) -> None:
        """
        Cleans the global experiment metrics JSON.
        """
        if path_exists(self.results_experiment_json):
            try:
                self.results_experiment_json.unlink()
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not delete {self.results_experiment_json}: {e}"
                )

    def clean(self) -> None:
        """Cleans metrics JSON files for the model plane and the active execution mode."""
        if self.is_fold:
            self._clean_fold_results()

        elif self.is_individual_patient:
            if path_exists(self.patient_results_json):
                self.patient_results_json.unlink()

        else:  # experiment mode
            self._clean_experiment_results()

    # ======================================
    #            VERIFICATION
    # ======================================

    def _verify_fold_paths(self) -> None:
        """Verifies input and output paths for all patients in the active fold.

        Raises:
            FileNotFoundError: If a patient's GT or predicted volume does not exist.
        """
        patients = list_patients(self.pred_vols_fold_dir)

        for patient_id in patients:
            patient_gt_dir = (
                self.gt_dir / patient_id / f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"
            )
            patient_pred_vol_dir = (
                self.pred_vols_fold_dir
                / patient_id
                / f"{patient_id}_{self.plane}{EXT_NIFTI}"
            )
            results_root_patient = self.results_fold_dir / patient_id

            # gt_dir
            if not path_exists(patient_gt_dir):  # Raises exception if not found
                raise FileNotFoundError(
                    f"Ground truth volume not found for patient {patient_id}: {patient_gt_dir}."
                )

            # pred_vol_dir
            if not path_exists(
                patient_pred_vol_dir
            ):  # Raises exception if not found
                raise FileNotFoundError(
                    f"Prediction not found for patient {patient_id}: {patient_pred_vol_dir}."
                )

            # results_root
            create_directory(results_root_patient)  # Ensure output directory exists

    def _verify_patient_paths(self) -> None:
        """Verifies input and output paths for an individual patient.

        Raises:
            FileNotFoundError: If the patient's GT or predicted volume does not exist.
        """
        # patient_gt_vol
        if not path_exists(self.patient_gt_vol):  # Raises exception if not found
            raise FileNotFoundError(
                f"GT not found for patient {self.patient.id}: {self.patient_gt_vol}."
            )

        # patient_pred_vol
        if not path_exists(self.patient_pred_vol):  # Raises exception if not found
            raise FileNotFoundError(
                f"Prediction not found for patient {self.patient.id}: {self.patient_pred_vol}."
            )

        # patient_results_root
        create_directory(self.patient_results_root)  # Ensure output directory exists

    def _verify_experiment_paths(self) -> None:
        """Verifies that per-fold metrics JSON files exist for the experiment-level average.

        Raises:
            FileNotFoundError: If the results directory or any fold's metrics JSON is missing.
        """
        if not path_exists(self.results_base_dir):
            raise FileNotFoundError(
                f"Results directory not found: {self.results_base_dir}"
            )

        # Expected input files → `k_folds` individual results files per fold
        expected_folds = [f"fold{i}_{self.plane}" for i in range(1, self.k_folds + 1)]
        found_folds = set()

        # Search all subdirectories of the experiment
        for fold_dir in self.results_base_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
                continue
            for file in fold_dir.iterdir():
                if file.is_file() and file.suffix.lower() == f"{EXT_JSON}":
                    for expected in expected_folds:
                        if file.name.startswith(expected):
                            found_folds.add(expected)

        missing = [f for f in expected_folds if f not in found_folds]

        if missing:  # Raises exception if any fold is missing
            raise FileNotFoundError(
                f"❌ Results JSON not found for the following folds: {missing}"
            )

    def verify_paths(self) -> None:
        """Verifies that all required paths exist for metric computation.

        Delegates path verification to the appropriate method based on the active mode.
        """

        if self.is_fold:
            self._verify_fold_paths()

        elif self.is_individual_patient:
            self._verify_patient_paths()

        else:  # experiment mode
            self._verify_experiment_paths()
