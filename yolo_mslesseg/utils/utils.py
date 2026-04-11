"""
Module: utils.py

Description:
    Collection of common utilities used as support for the main pipeline
    scripts, providing reusable and consistent functions that avoid
    duplicating logic across different stages.

Functional blocks:
    - Directories and files:
        Creation, deletion, and validation of paths; filtering of
        irrelevant files.

    - NIfTI volumes:
        Loading, saving, dimensional validation, and reconstruction of 3D volumes.

    - YOLO models:
        Safe model loading and basic error handling.

    - JSON:
        Reading and writing dictionaries in JSON format.

    - Patients and folds:
        ID retrieval, listing, and fold assignment.

    - Percentile computation:
        Custom int_or_percentile type and percentile calculation.

    - Image processing:
        Binary mask normalisation, uint8 conversion, RGB/BGR conversion,
        and greyscale normalisation.

    - Metrics and evaluation:
        Performance metric computation and partial result evaluation.

    - Fold status logging:
        Helper functions to log the execution status of a fold within
        pipeline stages (e.g. prediction, reconstruction, or evaluation),
        including compatibility with custom levels such as SKIP.

Usage:
    - Internal: imported from any pipeline stage.
    - Not designed for direct CLI execution.

Conventions:
    - All paths are handled with pathlib.Path.
    - Functions never silently swallow critical errors: exceptions are re-raised.
    - Binary masks are normalised to the range {0, 1}.
    - The `int_or_percentile` type accepts integer values or strings "P<n>".

Inputs:
    None. Provides reusable utility functions.

Outputs:
    None. Provides reusable utility functions.

Relationships:
    - Used by all pipeline scripts and Config classes.
    - Depends on constants.py for StageResult and other constants.
    - Depends on logging_config.py for the logger.
"""

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from yolo_mslesseg.utils.constants import (
    EXT_PNG,
    ANATOMICAL_PLANES,
    EXT_NIFTI,
    WEIGHTS_FILE,
    DATASETS_DIR,
    N_TRAIN_PATIENTS,
    StageResult,
)
from yolo_mslesseg.utils.logging_config import get_logger

if TYPE_CHECKING:
    from yolo_mslesseg.utils.Model import Model
    from yolo_mslesseg.utils.Patient import Patient

logger = get_logger(__file__)


# ======================================
#       DIRECTORIES AND FILES
# ======================================


def path_exists(path: str | Path) -> bool:
    """Returns True if the given path exists.

    Args:
        path: Filesystem path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    return Path(path).exists()


def create_directory(path: str | Path) -> None:
    """Creates the directory if it does not exist.

    Args:
        path: Directory path to create, including all intermediate parents.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def delete_directory(input_dir: str | Path) -> None:
    """Recursively deletes the given directory.

    Args:
        input_dir: Directory to delete. Does nothing if it does not exist.
    """
    path = Path(input_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def is_ignorable_file(name: str) -> bool:
    """Returns True if the file appears to be a system or hidden file.

    Args:
        name: Filename to check.

    Returns:
        True if the file starts with '.' or '~', or ends with '.tmp'.
    """
    name_lower = name.lower()
    return name.startswith(".") or name.startswith("~") or name_lower.endswith(".tmp")


def build_config_name(model: "Model", epochs: int) -> str:
    """Builds the global configuration folder name for the model.

    Combines modality, slice count, k_folds, and epochs into a canonical
    folder name. Uses '1fold' when k_folds == 1 and '<k>folds' otherwise.

    Args:
        model: Model instance providing modality, num_slices, and folds_string.
        epochs: Number of training epochs.

    Returns:
        Configuration folder name string.
    """
    modalities = "".join(model.modality)

    return f"{modalities}_{model.num_slices}slices_{model.folds_string}_{epochs}epochs"


def patient_base_dir(patient: "Patient", model: "Model") -> Path:
    """Returns the base directory for a patient's images and masks within the YOLO dataset.

    Args:
        patient: Patient instance providing the ID, plane, and split group.
        model: Model instance defining k_folds and base path.

    Returns:
        Absolute path to the patient's dataset directory.

    Raises:
        ValueError: If k_folds == 1 and the patient does not belong to the
            test split.
    """
    base_root = Path.cwd()

    patient_id = patient.id
    plane = patient.plane

    # k_folds > 1 → fold
    if model.k_folds > 1:
        fold = compute_fold(patient_id, model.k_folds)
        return (
            base_root
            / DATASETS_DIR
            / model.base_path
            / f"fold{fold}"
            / patient_id
            / plane
        )

    # k_folds == 1 → group (test/train)
    group = patient.split

    if patient.split != "test":
        raise ValueError(
            f"Patient {patient_id} belongs to 'train'. "
            "With k_folds == 1, only visualizations for 'test' patients are allowed."
        )

    return base_root / DATASETS_DIR / model.base_path / group / patient_id / plane


def patient_paths(
    patient: "Patient", model: "Model", slice_idx: int
) -> dict[str, Path]:
    """Builds a dictionary of paths for a specific slice of the patient.

    Args:
        patient: Patient instance providing ID and modality string.
        model: Model instance used to resolve the base directory.
        slice_idx: Slice index within the anatomical plane.

    Returns:
        Dictionary with keys 'img', 'pred', and 'gt' mapping to their
        respective file paths.
    """
    patient_id = patient.id

    base_dir = patient_base_dir(patient=patient, model=model)

    return {
        "img": base_dir
        / "images"
        / f"{patient_id}_{patient.modality_str}_{slice_idx}{EXT_PNG}",
        "pred": base_dir
        / "pred_masks"
        / f"{patient_id}_{patient.modality_str}_{slice_idx}{EXT_PNG}",
        "gt": base_dir / "GT_masks" / f"{patient_id}_{slice_idx}{EXT_PNG}",
    }


# ======================================
#         NIFTI VOLUME HANDLING
# ======================================


def load_volume(vol_path: str | Path) -> np.ndarray:
    """Loads a NIfTI file and returns its data array.

    Args:
        vol_path: Path to the NIfTI file.

    Returns:
        NumPy array containing the volume data.

    Raises:
        Exception: Re-raises any exception thrown by nibabel after logging it.
    """
    try:
        return nib.load(vol_path).get_fdata()
    except Exception as e:
        logger.error(f"❌ Error loading volume from {vol_path}: {e}")
        raise


def load_nifti_reference(
    reference_path: str | Path,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Loads a NIfTI file and returns its shape and affine transform.

    Args:
        reference_path: Path to the reference NIfTI file.

    Returns:
        Tuple of (shape, affine) where shape is the voxel dimensions tuple
        and affine is the 4x4 transformation matrix.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid NIfTI image.
    """
    if not path_exists(reference_path):
        raise FileNotFoundError(f"File not found: {reference_path}")
    try:
        nifti = nib.load(reference_path)
        return nifti.shape, nifti.affine
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"Invalid file: {reference_path}") from e


def save_volume(
    volume: np.ndarray, affine: np.ndarray, output_path: str | Path
) -> None:
    """Saves a NIfTI volume to the given output path.

    Args:
        volume: Volume data as a NumPy array.
        affine: 4x4 affine transformation matrix.
        output_path: Destination path for the NIfTI file.

    Raises:
        Exception: Re-raises any exception thrown by nibabel after logging it.
    """
    try:
        nifti_out = nib.Nifti1Image(volume, affine)
        nib.save(nifti_out, output_path)
    except Exception as e:
        logger.error(f"❌ Error saving volume to {output_path}: {e}")
        raise


def is_valid_reconstruction(pred_vol_path: str | Path, gt_vol_path: str | Path) -> bool:
    """Validates that the reconstructed volume is consistent with the ground truth.

    Compares the shapes of the predicted and ground truth volumes.

    Args:
        pred_vol_path: Path to the predicted NIfTI volume.
        gt_vol_path: Path to the ground truth NIfTI volume.

    Returns:
        True if the shapes match, False otherwise.
    """
    pred_vol = load_volume(pred_vol_path)
    gt_vol = load_volume(gt_vol_path)

    if pred_vol.shape != gt_vol.shape:
        logger.warning(f"⚠️ Shape mismatch: {pred_vol.shape} vs {gt_vol.shape}")
        return False

    return True


def predicted_volumes_complete(patient_dir: str | Path) -> bool:
    """Checks that all three predicted volumes exist for a patient.

    Verifies that axial, coronal, and sagittal volumes are present in the
    patient's prediction directory.

    Args:
        patient_dir: Path to the patient's prediction directory.

    Returns:
        True if all three plane volumes exist, False otherwise.
    """
    patient_id = Path(patient_dir).name
    return all(
        (Path(patient_dir) / f"{patient_id}_{plane}{EXT_NIFTI}").exists()
        for plane in ANATOMICAL_PLANES
    )


def verify_group_volumes(root_dir: Path) -> bool:
    """Verifies that all patients in a directory have complete predicted volumes.

    Checks that every patient has predicted volumes for all three anatomical
    planes: axial, coronal, and sagittal.

    Args:
        root_dir: Directory containing patient subdirectories.

    Returns:
        True if all patients have complete volumes, False otherwise.
    """
    patients = list_patients(root_dir)
    incomplete_patients = []

    for patient_id in patients:
        patient_pred_root_dir = root_dir / patient_id
        if not predicted_volumes_complete(patient_pred_root_dir):
            incomplete_patients.append(patient_id)

    return incomplete_patients == []


# ======================================
#          YOLO MODEL HANDLING
# ======================================


def load_model(model_path: str | Path) -> YOLO:
    """Loads a YOLO model from the given path.

    Args:
        model_path: Path to the YOLO model weights file.

    Returns:
        Loaded YOLO model instance.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    try:
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")


def trained_model_exists(model: "Model", epochs: int, fold_test: int | None) -> bool:
    """Checks whether the trained model weights file exists and is non-empty.

    With fold_test == None (k_folds == 1), looks in
    trains/.../plane/weights/best.pt. With fold_test set (k_folds > 1),
    looks in trains/.../plane/foldN/weights/best.pt.

    Args:
        model: Model instance providing the base path and plane.
        epochs: Number of training epochs used to locate the weights directory.
        fold_test: Fold index for cross-validation, or None for a fixed split.

    Returns:
        True if the weights file exists and has a non-zero size, False otherwise.
    """
    train_base = Path("trains") / f"{model.base_path}_{epochs}epochs" / model.plane

    if fold_test is None:
        # k_folds == 1: flat structure without fold subdirectory
        model_path = train_base / "weights" / WEIGHTS_FILE
    else:
        # k_folds > 1: structure with fold subdirectory
        model_path = train_base / f"fold{fold_test}" / "weights" / WEIGHTS_FILE

    return model_path.exists() and model_path.stat().st_size > 0


# ======================================
#                JSON
# ======================================


def write_json(dic: dict, json_path: str | Path) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        dic: Dictionary to serialise.
        json_path: Destination file path.
    """
    with open(json_path, "w") as f:
        json.dump(dic, f)


def read_json(json_path: str | Path) -> dict:
    """Reads a JSON file and returns its contents as a dictionary.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dictionary with the file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"File not found: {json_path}")


# ======================================
#       PATIENT AND FOLD UTILITIES
# ======================================


def get_id(patient: str) -> int | float:
    """Extracts the numeric ID from a patient name string.

    Args:
        patient: Patient identifier string (e.g. 'P12').

    Returns:
        Integer ID if a number is found, or infinity if the pattern does
        not match.
    """
    match = re.search(r"P(\d+)", patient)
    return (
        int(match.group(1)) if match else float("inf")
    )


def list_patients(input_dir: str | Path) -> list[str]:
    """Returns a sorted list of patient IDs found in a directory.

    Args:
        input_dir: Directory containing patient subdirectories.

    Returns:
        Sorted list of patient ID strings.

    Raises:
        FileNotFoundError: If no patients are found in the directory.
    """
    input_path = Path(input_dir)

    patients = [d.name for d in input_path.iterdir() if not is_ignorable_file(d.name)]
    if not patients:
        raise FileNotFoundError(f"No patients found in {input_dir}.")

    return sorted(patients, key=lambda p: int(p[1:]) if p[1:].isdigit() else 1_000_000)


def compute_fold(patient_id: str, k_folds: int = 5) -> int:
    """Assigns a patient to their corresponding cross-validation fold.

    Args:
        patient_id: Patient identifier string (e.g. 'P12').
        k_folds: Total number of cross-validation folds.

    Returns:
        1-based fold index for the patient.

    Raises:
        ValueError: If the patient ID cannot be assigned to any fold.
    """

    number = int(patient_id[1:])

    # Only train-split patients (P1–P53) are used in CV mode
    all_ids = list(range(1, N_TRAIN_PATIENTS + 1))

    folds = np.array_split(all_ids, k_folds)

    # Find which fold the patient belongs to
    for i, fold in enumerate(folds, 1):
        if number in fold:
            return i

    raise ValueError(f"Cannot compute fold for patient {patient_id}.")


def get_patient_slices(patient: "Patient", model: "Model") -> list[int]:
    """Returns a sorted list of available slice indices for a patient.

    Indices are extracted from PNG filenames in the images/ subdirectory of
    the patient's YOLO dataset directory.

    Args:
        patient: Patient instance defining the ID and plane.
        model: Model instance used to resolve the dataset directory.

    Returns:
        Sorted list of integer slice indices.
    """
    base_dir = patient_base_dir(patient=patient, model=model)
    images_dir = base_dir / "images"

    slices = []
    for fname in images_dir.glob(f"*{EXT_PNG}"):
        try:
            slice_num = int(fname.stem.split("_")[-1])
            slices.append(slice_num)
        except ValueError:
            continue  # Ignore files that do not follow the naming convention

    return sorted(slices)


# ======================================
#         PERCENTILE HANDLING
# ======================================


def int_or_percentile(value: str | int) -> int | str:
    """Parses a value as an integer or a percentile string.

    Accepts plain integer values or strings of the form 'P<n>'
    (e.g. 'P50' for the 50th percentile).

    Args:
        value: Value to parse, either an integer or a 'P<n>' string.

    Returns:
        Integer if value is a plain number, or the uppercase percentile
        string if it matches the 'P<n>' pattern.

    Raises:
        argparse.ArgumentTypeError: If the value matches neither format.
    """
    try:
        return int(value)
    except ValueError:
        if (
            isinstance(value, str)
            and value.upper().startswith("P")
            and value[1:].isdigit()
        ):
            return value.upper()
        raise argparse.ArgumentTypeError(
            "Value must be an integer or a string of the form 'PX' (e.g. P10 for the 10th percentile)."
        )


# ======================================
#          IMAGE PROCESSING
# ======================================


def load_png(path: str | Path) -> np.ndarray:
    """Loads a PNG file in greyscale and returns it as a NumPy array.

    Args:
        path: Path to the PNG file.

    Returns:
        2D NumPy array with pixel values.
    """
    return np.array(Image.open(path).convert("L"))


def prepare_pred_gt_slices(
    img_path: str | Path,
    pred_path: str | Path,
    gt_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads and prepares the image, prediction mask, and ground truth for a slice.

    Applies a corrective 90-degree rotation to the prediction mask to align
    it with NIfTI voxel coordinates.

    Args:
        img_path: Path to the input image PNG.
        pred_path: Path to the prediction mask PNG.
        gt_path: Path to the ground truth mask PNG.

    Returns:
        Tuple of (img, pred, gt) as float NumPy arrays with binary values.
    """
    img = load_png(img_path)
    pred = (load_png(pred_path) > 0).astype(float)
    gt = (load_png(gt_path) > 0).astype(float)

    pred = np.rot90(pred, 1)  # Corrective rotation

    return img, pred, gt


def normalize_binary_mask(mask_path: str | Path) -> None:
    """Normalises and saves a binary mask to values 0 (background) and 1 (object).

    Args:
        mask_path: Path to the mask file to normalise in place.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_bin)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalises an image to the range 0–255 and converts it to uint8.

    Args:
        image: Input image array of any numeric dtype.

    Returns:
        Image array with dtype uint8 and values in [0, 255].
    """
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image -= np.min(image)
        if np.ptp(image) > 0:
            image = 255 * (image / np.ptp(image))
        image = image.astype(np.uint8)
    return image


def convert_to_bgr(image: np.ndarray) -> np.ndarray:
    """Converts a 2D or RGB image to BGR format.

    Args:
        image: Input image array (greyscale or RGB).

    Returns:
        BGR image array with dtype uint8.
    """
    image_uint8 = normalize_to_uint8(image)
    if len(image_uint8.shape) == 2:  # Greyscale image
        img_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:  # RGB image
        img_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    return img_bgr


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Returns the image in greyscale, converting from BGR if necessary.

    Args:
        image: Input image array (greyscale or 3-channel BGR).

    Returns:
        2D greyscale NumPy array.
    """
    if image.ndim == 3 and image.shape[2] == 3:  # Colour image (3 channels)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already greyscale


# ======================================
#          RESULT EVALUATION
# ======================================


def evaluate_results(results: list) -> StageResult:
    """Evaluates the combined status of a list of partial pipeline results.

    Each element must be a StageResult value: COMPLETED or SKIPPED.

    Args:
        results: List of per-patient or per-fold stage results.

    Returns:
        StageResult.SKIPPED if all results are SKIPPED or the list is empty,
        StageResult.COMPLETED if all results are COMPLETED, or
        StageResult.PARTIAL if there is a mix of both.
    """
    if not results:
        return StageResult.SKIPPED

    if all(r is StageResult.SKIPPED for r in results):
        return StageResult.SKIPPED
    elif all(r is StageResult.COMPLETED for r in results):
        return StageResult.COMPLETED
    else:
        return StageResult.PARTIAL


# ======================================
#              METRICS
# ======================================


def DSC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Dice Similarity Coefficient (DSC).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.

    Returns:
        DSC score rounded to 3 decimal places.
    """
    intersection = np.sum(y_true * y_pred)
    dsc = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

    return float(np.round(dsc, 3))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes precision (positive predictive value).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.

    Returns:
        Precision score rounded to 3 decimal places.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    prec = tp / (tp + fp + 1e-8)

    return float(np.round(prec, 3))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes recall (sensitivity).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.

    Returns:
        Recall score rounded to 3 decimal places.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    rec = tp / (tp + fn + 1e-8)

    return float(np.round(rec, 3))


def AUC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.

    Returns:
        AUC score rounded to 3 decimal places, or NaN if undefined.
    """
    try:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        if len(np.unique(y_true)) < 2:
            logger.warning("⚠️ AUC undefined: y_true contains only one class.")
            return np.nan
        auc = float(np.round(roc_auc_score(y_true, y_pred), 3))
        return auc

    except Exception as e:
        logger.warning(f"⚠️ Could not compute AUC: {e}")
        return np.nan


# ======================================
#        FOLD STATUS LOGGING
# ======================================


def log_fold_status(logger: logging.Logger, result: StageResult, fold: int) -> None:
    """Logs the execution status of a fold for a specific pipeline stage.

    Args:
        logger: Logger instance with custom level support (skip, info, warning).
        result: Stage result — StageResult.COMPLETED, StageResult.SKIPPED,
            StageResult.PARTIAL, or any other value (unknown status).
        fold: Fold index to include in the log message.
    """
    if result is StageResult.SKIPPED:
        logger.skip(f"⏩ Fold {fold} already exists.")
    elif result is StageResult.COMPLETED:
        logger.info(f"🆗 Fold {fold} completed.")
    elif result is StageResult.PARTIAL:
        logger.info(f"🔁 Fold {fold} partially updated.")
    else:
        logger.warning(f"⚠️ Fold {fold}: unknown status.")
