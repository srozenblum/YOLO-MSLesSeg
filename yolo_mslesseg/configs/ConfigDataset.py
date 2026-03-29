from pathlib import Path

from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import (
    DATASET_DIR,
    SPLIT_TRAIN,
    SPLIT_TEST,
    DATASETS_DIR,
)
from yolo_mslesseg.utils.utils import (
    path_exists,
    delete_directory,
    create_directory,
    list_patients,
    compute_fold,
)

# Configure logger
logger = get_logger(__file__)


class ConfigDataset:
    """
    Class: ConfigDataset

    Description:
        Configuration and path management for the YOLO dataset extraction stage,
        implemented in `extract_dataset.py`. Handles verification, creation, and
        cleanup of directory structures for both individual patients and the full
        patient set, ensuring correct organisation of images, masks, and annotations.

        Supports two output schemes:
        - k_folds > 1 → fold-based partition: fold1/, fold2/, ...
        - k_folds == 1 → simple partition: train/ and test/ (no fold subdirectories)

    Execution modes:
        1. Individual patient mode (`patient` ≠ None):
           Builds the directory structure only for the specified patient.

        2. Full mode (`patient` = None, `full` = True):
           Builds the directory structure for all patients.

    Directory conventions:
        MSLesSeg-Dataset/: original input dataset
            ├── train/
            └── test/

        datasets/: YOLO datasets split by group and patient
        └── <enhancement>/
             └── <modality>_<num_slices>slices_<k_folds>folds/
                 - If k_folds > 1:
                    ├── fold1/PX/<plane>/(images|labels|GT_masks)/
                    ├── fold2/...
                    └── ...
                 - If k_folds == 1:
                    ├── train/PX/<plane>/(images|labels|GT_masks)/
                    └── test/PX/<plane>/(images|labels|GT_masks)/

    Attributes:
        model (Model):
            Model instance defining the plane, modalities, enhancement, and base_path.

        plane (str):
            Anatomical processing plane ('axial', 'coronal', or 'sagittal').

        k_folds (int):
            Number of cross-validation folds.

        full (bool):
            Indicates whether the full patient set will be processed.

        patient (Patient | None, optional):
            Patient instance for individual execution.

        mslesseg_root (Path):
            Root directory of the original MSLesSeg dataset.

        mslesseg_train_dir (Path):
            MSLesSeg-Dataset/train directory.

        mslesseg_test_dir (Path):
            MSLesSeg-Dataset/test directory.

        input_dir (Path):
            Effective input directory for the current execution.
            - Defaults to MSLesSeg-Dataset/train.
            - In patient mode, may be train or test depending on the patient's split.

        output_dir (Path):
            Base output directory: datasets/<model.base_path>.

        # --- Attributes valid only in patient mode ---
        patient_fold (int | None, optional):
            Patient's fold (only when k_folds > 1).

        patient_group (str | None, optional):
            'train' or 'test' (only when k_folds == 1).

        patient_root (Path | None, optional):
            Base directory of the patient in the YOLO dataset.

        patient_dir (dict[str, Path] | None, optional):
            Dictionary of patient subdirectories: images/, GT_masks/, labels/.

        patient_is_train (bool):
            True if the patient belongs to the train split (CV mode only).
    """

    def __init__(
        self,
        model: Model,
        input_dir: str | Path | None = None,
        full: bool = False,
        patient: Patient | None = None,
    ) -> None:
        """Initialises a ConfigDataset instance for the dataset extraction stage.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            input_dir: Override for the input dataset directory. Defaults
                to the MSLesSeg train directory if None.
            full: If True, processes all patients (ignored in patient mode).
            patient: Patient instance for individual execution, or None for full mode.

        Raises:
            ValueError: If neither full mode nor a patient instance is specified.
        """
        # --- Main attributes ---
        self._set_main_attributes(
            model=model,
            input_dir=input_dir,
            full=full,
            patient=patient,
        )

        # --- Determine execution mode ---
        self._resolve_execution_mode()

        # --- Patient-specific paths (if applicable) ---
        self._resolve_patient_paths()

    # ======================================
    #          CONSTRUCTOR HELPERS
    # ======================================

    def _set_main_attributes(
        self,
        model: Model,
        input_dir: str | Path | None,
        full: bool,
        patient: Patient | None,
    ) -> None:
        """Sets the core attributes and resolves the input/output directories.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            input_dir: Override for the input directory, or None for default.
            full: Whether to process the full dataset.
            patient: Patient instance for individual execution, or None.
        """
        self.model = model
        self.plane = model.plane
        self.k_folds = model.k_folds
        self.patient = patient
        self.full = full

        # MSLesSeg dataset (root + subdirs)
        self.mslesseg_root = DATASET_DIR
        self.mslesseg_train_dir = self.mslesseg_root / SPLIT_TRAIN
        self.mslesseg_test_dir = self.mslesseg_root / SPLIT_TEST

        # Effective input dataset directory
        if input_dir is None:
            self.input_dir = self.mslesseg_train_dir
        else:
            self.input_dir = Path(input_dir)

        # Output directory
        self.output_dir = DATASETS_DIR / f"{self.model.base_path}"

    def _resolve_execution_mode(self) -> None:
        """Resolves the execution mode and configures patient-related attributes.

        Determines whether this is an individual patient run or a full dataset
        run, and sets the input directory and fold/group accordingly.

        Raises:
            ValueError: If neither full mode nor patient mode is specified.
        """
        self.is_individual_patient = self.patient is not None
        self.is_full = (not self.is_individual_patient) and self.full

        if self.is_individual_patient:
            original_input_dir = self.input_dir

            if self.k_folds > 1:
                # Cross-validation:
                # - Train patients → assigned to their corresponding fold
                # - Test patients  → do not belong to any fold
                if self.patient.split == "train":
                    self.patient_fold = compute_fold(
                        patient_id=self.patient.id,
                        k_folds=self.k_folds,
                    )
                    self.input_dir = self.mslesseg_train_dir
                else:
                    self.patient_group = "test"
                    self.input_dir = self.mslesseg_test_dir
                self.patient_is_train: bool = self.patient.split == "train"

            else:
                # k_folds == 1:
                # The patient's original group (train/test) is used directly
                self.patient_group = self.patient.split
                self.input_dir = self.get_input_dir(self.patient_group)

            if self.input_dir != original_input_dir:
                logger.warning(
                    f"⚠️ --input_dir was overridden: '{original_input_dir}' → "
                    f"'{self.input_dir}' (derived from patient split)."
                )

        elif self.is_full:
            pass  # nothing extra

        else:
            raise ValueError(
                "An execution mode must be specified: full dataset or individual patient."
            )

    def _resolve_patient_paths(self) -> None:
        """Resolves and stores the output paths for an individual patient.

        Has no effect when running in full dataset mode.
        """
        if not self.is_individual_patient:
            return

        # --- k_folds > 1 ---
        # Train patients → inside foldX/
        # Test patients  → inside test/
        if self.k_folds > 1:
            if self.patient_is_train:
                self.patient_root = (
                    self.output_dir
                    / f"fold{self.patient_fold}"
                    / self.patient.id
                    / self.plane
                )
            else:
                self.patient_root = (
                    self.output_dir / "test" / self.patient.id / self.plane
                )

        # --- k_folds == 1 ---
        # Patient's group (train/test) is used directly
        else:
            self.input_dir = self.get_input_dir(self.patient_group)

            self.patient_root = (
                self.output_dir / self.patient_group / self.patient.id / self.plane
            )

        # Standard patient subdirectories
        self.patient_dir = {
            subdir: self.patient_root / subdir
            for subdir in ["images", "GT_masks", "labels"]
        }

    # ======================================
    #               INPUTS
    # ======================================

    def get_input_dir(self, group: str = "train") -> Path:
        """Returns the MSLesSeg input directory for the given group.

        When k_folds > 1, always returns the train directory (the group
        argument is ignored, as CV test patients come from the train set).
        When k_folds == 1, returns the train or test directory based on group.

        Args:
            group: Dataset group — either 'train' or 'test'.

        Returns:
            Path to the corresponding MSLesSeg input directory.

        Raises:
            ValueError: If group is not 'train' or 'test'.
        """
        if self.k_folds > 1:
            return self.mslesseg_train_dir

        if group == "train":
            return self.mslesseg_train_dir
        if group == "test":
            return self.mslesseg_test_dir

        raise ValueError("group must be 'train' or 'test'.")

    # ======================================
    #               CLEANUP
    # ======================================

    def _clean_patients_root(self, root_dir: Path) -> None:
        """Cleans the plane subdirectories for all patients under a root directory.

        Removes images/, GT_masks/, and labels/ for the current plane across
        all patients under root_dir (e.g. a foldX/ or train/test directory).

        Args:
            root_dir: Root directory containing patient subdirectories.
        """
        if not path_exists(root_dir):
            return

        patients = list_patients(root_dir)

        for patient_id in patients:
            patient_path = root_dir / patient_id
            if not patient_path.is_dir():
                continue

            for plane_dir in patient_path.iterdir():
                if not plane_dir.is_dir():
                    continue
                if self.plane.lower() not in plane_dir.name.lower():
                    continue

                for subdir in plane_dir.iterdir():
                    if subdir.is_dir():
                        try:
                            delete_directory(subdir)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not delete {subdir}: {e}")

    def _clean_full_dataset(self) -> None:
        """Cleans patient subdirectories for the current plane across the entire output structure."""
        if not path_exists(self.output_dir):
            return

        if self.k_folds > 1:
            for fold_dir in self.output_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.lower().startswith("fold"):
                    self._clean_patients_root(fold_dir)
        else:
            for group in ["train", "test"]:
                group_dir = self.output_dir / group
                if path_exists(group_dir):
                    self._clean_patients_root(group_dir)

    def _clean_patient_dataset(self) -> None:
        """Cleans the plane subdirectories for a single patient."""
        if not path_exists(self.patient_root):
            return

        for name, subdir_path in self.patient_dir.items():
            if not path_exists(subdir_path):
                continue
            try:
                delete_directory(subdir_path)
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not delete {name} for {self.patient.id}: {e}"
                )

    def clean_dataset(self) -> None:
        """Cleans output files according to the active execution mode.

        Delegates to _clean_full_dataset for full mode, or to
        _clean_patient_dataset for individual patient mode.
        """
        if self.is_full:
            self._clean_full_dataset()
        else:
            self._clean_patient_dataset()

    # ======================================
    #            VERIFICATION
    # ======================================

    def _create_output_structure(self, patients: list[str], root_output: Path) -> None:
        """Creates the YOLO output directory structure for a list of patients.

        Creates images/, GT_masks/, and labels/ subdirectories for each
        patient under root_output.

        Args:
            patients: List of patient ID strings to create directories for.
            root_output: Root directory under which patient directories are created.
        """
        create_directory(root_output)

        for patient_id in patients:
            patient_dir = root_output / patient_id / self.plane
            create_directory(patient_dir)
            for subdir in ["images", "GT_masks", "labels"]:
                create_directory(patient_dir / subdir)

    def _verify_full_paths(self) -> None:
        """Verifies that the input dataset exists and builds the output directory structure.

        When k_folds > 1, creates fold1..foldK using patients from the train split.
        When k_folds == 1, creates train/ and test/ directories.

        Raises:
            FileNotFoundError: If the MSLesSeg train directory does not exist.
        """
        if not self.mslesseg_train_dir.is_dir():
            raise FileNotFoundError(
                f"Input dataset directory not found: {self.mslesseg_train_dir}"
            )

        create_directory(self.output_dir)

        # k_folds > 1 → folds
        if self.k_folds > 1:
            train_patients = list_patients(self.mslesseg_train_dir)

            # Create fold directories
            for i in range(1, self.k_folds + 1):
                create_directory(self.output_dir / f"fold{i}")

            # Create per-patient substructure according to fold
            for patient_id in train_patients:
                patient_fold = compute_fold(
                    patient_id=patient_id, k_folds=self.k_folds
                )
                root_fold = self.output_dir / f"fold{patient_fold}"
                self._create_output_structure([patient_id], root_fold)

        # k_folds == 1 → train/test
        else:
            # train
            train_patients = list_patients(self.mslesseg_train_dir)
            self._create_output_structure(train_patients, self.output_dir / "train")

            # test (optional: only created if the test split exists in the dataset)
            if self.mslesseg_test_dir.is_dir():
                test_patients = list_patients(self.mslesseg_test_dir)
                self._create_output_structure(test_patients, self.output_dir / "test")

    def _verify_patient_paths(self) -> None:
        """Verifies input and output paths for an individual patient.

        Checks that the patient's input directory exists, and creates the
        images/, GT_masks/, and labels/ output subdirectories.

        Raises:
            FileNotFoundError: If the patient's input directory does not exist.
        """
        patient_input_dir = self.input_dir / self.patient.id
        if not patient_input_dir.is_dir():
            raise FileNotFoundError(
                f"Patient input directory not found for {self.patient.id}: {patient_input_dir}"
            )

        for subdir in self.patient_dir.values():
            create_directory(subdir)

    def verify_paths(self) -> None:
        """Verifies that the input and output directories exist for dataset extraction.

        Delegates to _verify_full_paths for full mode, or to
        _verify_patient_paths for individual patient mode.
        """
        if self.is_full:
            self._verify_full_paths()
        else:
            self._verify_patient_paths()

    # ======================================
    #            REPRESENTATION
    # ======================================

    def __repr__(self) -> str:
        """String representation of this ConfigDataset instance."""
        if self.is_full:
            return f"{self.__class__.__name__}(model={self.model.model_string}, full={self.full}, k_folds={self.k_folds})"
        return f"{self.__class__.__name__}(model={self.model.model_string}, patient={self.patient.id}, k_folds={self.k_folds})"
