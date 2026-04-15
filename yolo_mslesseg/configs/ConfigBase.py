"""
Module: ConfigBase

Description:
    Defines the abstract base class shared by ConfigPred, ConfigReconstruction,
    ConfigEval, and ConfigConsensus. Centralises the constructor arguments that
    are common to all four pipeline stage configurations and exposes the
    fold_subdir helper property that eliminates repeated single_fold branching
    throughout path-resolution methods.

Usage:
    Not instantiated directly. Used as the base class for all stage-specific
    Config classes that handle folds and patients.

Inputs:
    None. Provides the ConfigBase abstract class definition.

Outputs:
    None. Provides the ConfigBase abstract class definition.

Relationships:
    - Subclassed by ConfigPred, ConfigReconstruction, ConfigEval, and ConfigConsensus.
    - Depends on Model and Patient from utils/.
"""

from abc import ABC, abstractmethod

from yolo_mslesseg.utils.constants import SPLIT_TEST
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient

logger = get_logger(__file__)


class ConfigBase(ABC):
    """
    Class: ConfigBase

    Description:
        Abstract base class for pipeline stage configurations that manage
        folds and patients: ConfigPred, ConfigReconstruction, ConfigEval,
        and ConfigConsensus. Defines the common attribute contract, the
        fold_subdir property, and the abstract methods verify_paths() and clean().
        Contract: every subclass must implement verify_paths() (validate paths and
        create output directories) and clean() (delete stage outputs for re-runs).
        Callers always invoke them in order: clean() → verify_paths() → run stage.

    Attributes:
        model (Model):
            Model instance defining the plane, modalities, enhancement, and base_path.

        plane (str):
            Anatomical processing plane ('axial', 'coronal', 'sagittal').
            Subclasses may override this (e.g. 'consensus' in ConfigConsensus).

        epochs (int):
            Number of epochs of the YOLO model.

        k_folds (int):
            Number of cross-validation folds. Derived from model.k_folds.

        patient (Patient | None, optional):
            Patient instance for individual execution. None for fold or
            experiment-level execution.

        fold_test (int | None, optional):
            Index of the test fold (1, ..., k_folds). None when k_folds == 1.

        single_fold (bool):
            True if k_folds == 1 (fixed train/test split, no cross-validation).

        group (str | None, optional):
            Name of the active group when single_fold == True ('test').
            None when cross-validation is used.
    """

    def __init__(
        self,
        model: Model,
        epochs: int,
        patient: Patient | None = None,
        fold_test: int | None = None,
    ) -> None:
        """Initialises the shared attributes for all stage configurations.

        Args:
            model: Model instance defining the plane, modalities, and base_path.
            epochs: Number of training epochs of the YOLO model.
            patient: Patient instance for individual execution, or None for fold-level.
            fold_test: Test fold index when using cross-validation, or None.
        """
        self.model = model
        self.plane: str = model.plane
        self.epochs: int = epochs
        self.k_folds: int = model.k_folds
        self.patient = patient
        self.fold_test = fold_test
        self.single_fold: bool = model.k_folds == 1
        self.group: str | None = SPLIT_TEST if self.single_fold else None

    # ======================================
    #          FOLD SUBDIRECTORY
    # ======================================

    @property
    def fold_subdir(self) -> str:
        """Returns the fold subdirectory name for path construction.

        Returns:
            'test' when k_folds == 1, or 'fold<fold_test>' when k_folds > 1.

        Raises:
            ValueError: If fold_test is None in cross-validation mode (experiment mode).
        """
        if self.single_fold:
            return self.group
        if self.fold_test is None:
            raise ValueError(
                "fold_subdir is not available in experiment mode (fold_test is None)."
            )
        return f"fold{self.fold_test}"

    # ======================================
    #             INTERFACE
    # ======================================

    @abstractmethod
    def verify_paths(self) -> None:
        """Verify that all required input and output directories exist for this stage."""

    @abstractmethod
    def clean(self) -> None:
        """Clean the output files produced by this stage for the active execution mode."""

    # ======================================
    #            REPRESENTATION
    # ======================================

    def __repr__(self) -> str:
        """String representation of this configuration instance.

        Returns:
            String including the class name, model identifier, epochs, and the
            active execution scope (patient, group, fold, or experiment mode).
        """
        if self.patient is not None:
            return (
                f"{self.__class__.__name__}("
                f"model={self.model.model_string}, "
                f"epochs={self.epochs}, "
                f"patient={self.patient.id})"
            )
        if self.single_fold:
            return (
                f"{self.__class__.__name__}("
                f"model={self.model.model_string}, "
                f"epochs={self.epochs}, "
                f"group={self.group})"
            )
        if self.fold_test is not None:
            return (
                f"{self.__class__.__name__}("
                f"model={self.model.model_string}, "
                f"epochs={self.epochs}, "
                f"fold={self.fold_test})"
            )
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.model_string}, "
            f"epochs={self.epochs}, "
            f"mode=experiment)"
        )
