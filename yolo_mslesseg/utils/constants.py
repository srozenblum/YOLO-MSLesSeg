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
# Fixed MSLesSeg competition split: 53 training patients (P1–P53), 22 test patients (P54–P75).
N_TRAIN_PATIENTS = 53

# Modalities
MODALITIES = ("T1", "T2", "FLAIR")

# Planes
PLANES = ("axial", "coronal", "sagittal", "consensus")
ANATOMICAL_PLANES = ("axial", "coronal", "sagittal")

# Timepoints present in the MSLesSeg dataset directory structure.
# Used to detect whether a patient directory uses timepoint subdirectories.
# The pipeline currently always uses timepoint "T1".
# "T1" here refers to the first acquisition timepoint, not the T1-weighted MRI
# modality. Multi-timepoint support is not exercised by the current pipeline.
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

# Pipeline defaults
# Majority-vote threshold: a voxel is classified as lesion if at least this many
# of the 3 anatomical planes predict it as positive. Set to 3 for unanimity voting.
CONSENSUS_THRESHOLD = 2
# Gamma > 1 darkens the image non-linearly, increasing contrast in bright lesion
# regions relative to surrounding white matter. 2.0 was chosen empirically.
DEFAULT_GAMMA = 2.0
METRIC_DECIMAL_PLACES = 3  # Matches the MSLesSeg competition reporting convention.
WEIGHTS_SUBDIR = "weights"
ENHANCEMENT_BASE = "Base"
PLANE_CONSENSUS = "consensus"


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
