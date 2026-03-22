"""
Module: ConfigBase

Description:
    Defines the abstract base class shared by ConfigPred, ConfigReconstruction,
    ConfigEval, and ConfigConsensus. Centralises the constructor arguments that
    are common to all four pipeline stage configurations and exposes the
    fold_subdir helper property that eliminates repeated single_fold branching
    throughout path-resolution methods.
"""

from abc import ABC, abstractmethod

from yolo_mslesseg.utils.logging_config import get_logger

logger = get_logger(__file__)


class ConfigBase(ABC):
    """
    Class: ConfigBase

    Description:
        Abstract base class for pipeline stage configurations that manage
        folds and patients: ConfigPred, ConfigReconstruction, ConfigEval,
        and ConfigConsensus. Defines the common attribute contract, the
        fold_subdir property, and the abstract methods verify_paths() and clean().

    Attributes:
        model (Model):
            Model instance defining the plane, modalities, enhancement, and base_path.

        plane (str):
            Anatomical processing plane ('axial', 'coronal', 'sagital').
            Subclasses may override this (e.g. 'consenso' in ConfigConsensus).

        epochs (int):
            Number of epochs of the YOLO model.

        k_folds (int):
            Number of cross-validation folds.

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
        model,
        epochs: int,
        k_folds: int = 5,
        patient=None,
        fold_test=None,
    ) -> None:
        self.model = model
        self.plane: str = model.plane
        self.epochs: int = epochs
        self.k_folds: int = k_folds
        self.patient = patient
        self.fold_test = fold_test
        self.single_fold: bool = k_folds == 1
        self.group: str | None = "test" if self.single_fold else None

    # ======================================
    #          FOLD SUBDIRECTORY
    # ======================================

    @property
    def fold_subdir(self) -> str:
        """
        Returns the fold subdirectory name for path construction.

        - k_folds == 1: returns self.group ('test').
        - k_folds > 1:  returns 'fold<fold_test>'.
        """
        return self.group if self.single_fold else f"fold{self.fold_test}"

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
        """String representation of this configuration instance."""
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
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.model_string}, "
            f"epochs={self.epochs}, "
            f"fold={self.fold_test})"
        )
