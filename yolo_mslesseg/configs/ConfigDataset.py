from pathlib import Path

from yolo_mslesseg.utils.logging_config import get_logger
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
            Anatomical processing plane ('axial', 'coronal', or 'sagital').

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

        dataset_entrada (Path):
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
    """

    def __init__(
        self,
        model,
        dataset_entrada=None,
        k_folds=5,
        full=False,
        patient=None,
    ):
        # --- Main attributes ---
        self._set_main_attributes(
            model=model,
            dataset_entrada=dataset_entrada,
            k_folds=k_folds,
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
        model,
        dataset_entrada,
        k_folds,
        full,
        patient,
    ):
        self.model = model
        self.plane = model.plane
        self.k_folds = k_folds
        self.patient = patient
        self.full = full

        # MSLesSeg dataset (root + subdirs)
        self.mslesseg_root = DATASET_DIR
        self.mslesseg_train_dir = self.mslesseg_root / SPLIT_TRAIN
        self.mslesseg_test_dir = self.mslesseg_root / SPLIT_TEST

        # Effective input dataset directory
        if dataset_entrada is None:
            self.dataset_entrada = self.mslesseg_train_dir
        else:
            self.dataset_entrada = Path(dataset_entrada)

        # Output directory
        self.output_dir = DATASETS_DIR / f"{self.model.base_path}"

    def _resolve_execution_mode(self):
        self.is_individual_patient = self.patient is not None
        self.is_full = (not self.is_individual_patient) and self.full

        if self.is_individual_patient:
            if self.k_folds > 1:
                # Cross-validation:
                # - Train patients → assigned to their corresponding fold
                # - Test patients  → do not belong to any fold
                if self.patient.split == "train":
                    self.patient_fold = compute_fold(
                        patient_id=self.patient.id,
                        k_folds=self.k_folds,
                    )
                    self.dataset_entrada = self.mslesseg_train_dir
                else:
                    self.patient_group = "test"
                    self.dataset_entrada = self.mslesseg_test_dir

            else:
                # k_folds == 1:
                # The patient's original group (train/test) is used directly
                self.patient_group = self.patient.split
                self.dataset_entrada = self.input_dir(self.patient_group)

        elif self.is_full:
            pass  # nothing extra

        else:
            raise ValueError(
                "An execution mode must be specified: full dataset or individual patient."
            )

    def _resolve_patient_paths(self):
        if not self.is_individual_patient:
            return

        # --- k_folds > 1 ---
        # Train patients → inside foldX/
        # Test patients  → inside test/
        if self.k_folds > 1:
            if hasattr(self, "patient_fold"):
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
            self.dataset_entrada = self.input_dir(self.patient_group)

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

    def input_dir(self, group="train"):
        """
        Returns the MSLesSeg input directory for the given group.

        - If k_folds > 1: always returns MSLesSeg-Dataset/train.
          (the `group` parameter is ignored, as in CV mode test patients
          come from the same train set).
        - If k_folds == 1:
            * group='train' → MSLesSeg-Dataset/train
            * group='test'  → MSLesSeg-Dataset/test
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

    def _clean_patients_root(self, root_dir):
        """
        Cleans the images/, GT_masks/, and labels/ subdirectories for the
        current plane across all patients under root_dir (foldX/ or train/test).
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

    def _clean_full_dataset(self):
        """
        Cleans patient subdirectories for the current plane across the entire
        output structure.
        """
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

    def _clean_patient_dataset(self):
        """
        Cleans the plane subdirectories for a single patient.
        """
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

    def clean_dataset(self):
        """
        Cleans files and directories in the output directory according to
        the execution mode.
        - Full dataset: cleans all patients.
        - Individual patient: cleans only that patient.
        """
        if self.is_full:
            self._clean_full_dataset()
        else:
            self._clean_patient_dataset()

    # ======================================
    #            VERIFICATION
    # ======================================

    def _create_output_structure(self, patients, root_output):
        """
        Creates the images/, GT_masks/, and labels/ structure for a list of
        patients under root_output (foldX/ or train/test).
        """
        create_directory(root_output)

        for patient_id in patients:
            patient_dir = root_output / patient_id / self.plane
            create_directory(patient_dir)
            for subdir in ["images", "GT_masks", "labels"]:
                create_directory(patient_dir / subdir)

    def _verify_full_paths(self):
        """
        Verifies that the input dataset exists and builds the output structure.
        - k_folds > 1: creates fold1..foldK using patients from MSLesSeg-Dataset/train.
        - k_folds == 1: creates train/ and test/.
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

    def _verify_patient_paths(self):
        """
        Verifies input and output paths for an individual patient.
        - Input:  <dataset_entrada>/<patient_id>
        - Output: images/, GT_masks/, labels/ inside the patient's destination
                  (foldX/ or train/test).
        """
        patient_input_dir = self.dataset_entrada / self.patient.id
        if not patient_input_dir.is_dir():
            raise FileNotFoundError(
                f"Patient input directory not found for {self.patient.id}: {patient_input_dir}"
            )

        for subdir in self.patient_dir.values():
            create_directory(subdir)

    def verify_paths(self):
        """
        Verifies that the input and output directories exist for dataset extraction.
        - Full dataset mode: builds the global structure.
        - Individual patient mode: builds the structure only for that patient.
        """
        if self.is_full:
            self._verify_full_paths()
        else:
            self._verify_patient_paths()

    # ======================================
    #            REPRESENTATION
    # ======================================

    def __repr__(self):
        if self.is_full:
            return f"{self.__class__.__name__}(model={self.model.model_string}, full={self.full}, k_folds={self.k_folds})"
        return f"{self.__class__.__name__}(model={self.model.model_string}, patient={self.patient.id}, k_folds={self.k_folds})"
