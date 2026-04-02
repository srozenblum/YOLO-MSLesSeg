"""
Module: constants.py

Description:
    Defines global constants for the YOLO-MSLesSeg project, including base
    directories, data splits, image modalities, anatomical planes, timepoints,
    image enhancement techniques, file extensions, and standard names used
    throughout the pipeline.

    This module centralises the static configuration of the system to ensure
    structural consistency and ease repository maintenance. All Config classes
    and pipeline scripts must reference these constants to keep names and
    directory organisation consistent.

Usage:
    - Internal import from any module in the package.
    - Does not expose a CLI interface.
    - Does not perform any I/O operations.

Inputs:
    None. All definitions are static.

Outputs:
    None. Provides reusable constants.

Relationships:
    - Used by ConfigDataset, ConfigTrain, ConfigPred,
      ConfigReconstruction, ConfigConsensus, and ConfigEval.
    - Referenced by scripts such as setup, extract_dataset, train,
      generate_predictions, reconstruct_volume, generate_consensus, and eval.
    - Ensures consistency with the official pipeline directory structure.
"""

from enum import Enum
from pathlib import Path

# Base directories
DATASET_DIR = Path("MSLesSeg-Dataset")
DATASETS_DIR = Path("datasets")
TRAINS_DIR = Path("trains")
PRED_VOLS_DIR = Path("pred_vols")
RESULTS_DIR = Path("results")
GT_DIR = Path("GT")
VISUALIZATIONS_DIR = Path("visualizations")

# Splits
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"

# Dataset split sizes (MSLesSeg-Dataset)
# In CV mode (k_folds > 1), only the train split (P1–P53) is used.
N_TRAIN_PATIENTS = 53

# Modalities
MODALITIES = ("T1", "T2", "FLAIR")

# Planes
PLANES = ("axial", "coronal", "sagittal", "consensus")
ANATOMICAL_PLANES = ("axial", "coronal", "sagittal")

# Timepoints present in the MSLesSeg dataset directory structure.
# Used to detect whether a patient directory uses timepoint subdirectories.
# The pipeline currently always uses timepoint "T1".
TIMEPOINTS = ("T1", "T2", "T3", "T4")

# Enhancement algorithms
ENHANCEMENTS = ("HE", "CLAHE", "GC", "LT")

# File extensions
EXT_NIFTI = ".nii.gz"
EXT_PNG = ".png"
EXT_JSON = ".json"
EXT_YAML = ".yaml"
EXT_CSV = ".csv"
EXT_TXT = ".txt"

# File names
MASK_SUFFIX = "_MASK"
WEIGHTS_FILE = "best.pt"

# Results JSON naming conventions
# File names follow these patterns:
#   patient:    {patient_id}_{plane}_results.json
#   fold:       fold{N}_{plane}_results.json
#   global:     global_{plane}_results.json
RESULTS_SUFFIX = "_results"
RESULTS_GLOBAL_PREFIX = "global_"


class StageResult(Enum):
    """Enum representing the execution status of a pipeline stage.

    Attributes:
        COMPLETED: The stage ran and produced new output.
        SKIPPED: The stage was skipped because output already existed.
        PARTIAL: Some patients were processed and some were skipped.
    """

    COMPLETED = "completed"
    SKIPPED = "skipped"
    PARTIAL = "partial"
