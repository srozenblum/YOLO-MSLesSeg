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
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import (
    EXT_PNG,
    ANATOMICAL_PLANES,
    EXT_NIFTI,
    WEIGHTS_FILE,
    DATASETS_DIR,
    N_TRAIN_PATIENTS,
)

# Configure logger
logger = get_logger(__file__)


# ======================================
#       DIRECTORIES AND FILES
# ======================================


def path_exists(path):
    """Returns True if the given path exists."""
    return Path(path).exists()


def create_directory(path):
    """Creates the directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def delete_directory(input_dir):
    """Recursively deletes the given directory."""
    path = Path(input_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def is_ignorable_file(name):
    """Returns True if the file appears to be a system or hidden file."""
    name_lower = name.lower()
    return (
        name.startswith(".")
        or name.startswith("~")
        or name_lower.endswith(".tmp")
    )


def build_config_name(model, epochs):
    """
    Builds the global configuration folder name for the model
    (modality, slice count, k_folds, and epochs).

    - If k_folds == 1 → '1fold'
    - If k_folds > 1  → '<k>folds'
    """
    modalities = "".join(model.modality)

    return f"{modalities}_{model.num_slices}slices_{model.folds_string}_{epochs}epochs"


def patient_base_dir(patient, model):
    """
    Returns the base directory where the images, predictions, and ground truth
    masks of a patient are stored within the YOLO dataset.
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


def patient_paths(patient, model, slice_idx):
    """
    Builds and returns a dictionary of paths to the image, prediction, and
    ground truth mask for a specific slice of the patient.
    """
    patient_id = patient.id
    modality = patient.modality_str

    base_dir = patient_base_dir(patient=patient, model=model)

    return {
        "img": base_dir / "images" / f"{patient_id}_{modality}_{slice_idx}{EXT_PNG}",
        "pred": base_dir / "pred_masks" / f"{patient_id}_{modality}_{slice_idx}{EXT_PNG}",
        "gt": base_dir / "GT_masks" / f"{patient_id}_{slice_idx}{EXT_PNG}",
    }


# ======================================
#         NIFTI VOLUME HANDLING
# ======================================


def load_volume(vol_path):
    """Loads a NIfTI file and returns its data array."""
    try:
        return nib.load(vol_path).get_fdata()
    except Exception as e:
        logger.error(f"❌ Error loading volume from {vol_path}: {e}")
        raise


def load_nifti_reference(reference_path):
    """Loads a NIfTI file and returns its shape and affine."""
    if not path_exists(reference_path):
        raise FileNotFoundError(f"File not found: {reference_path}")
    try:
        nifti = nib.load(reference_path)
        return nifti.shape, nifti.affine
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"Invalid file: {reference_path}") from e


def save_volume(volume, affine, output_path):
    """Saves a NIfTI volume to the given output path."""
    try:
        nifti_out = nib.Nifti1Image(volume, affine)
        nib.save(nifti_out, output_path)
    except Exception as e:
        logger.error(f"❌ Error saving volume to {output_path}: {e}")
        raise


def is_valid_reconstruction(pred_vol_path, gt_vol_path):
    """
    Validates that the reconstructed volume is consistent with the ground truth
    by comparing their shapes.
    """
    pred_vol = load_volume(pred_vol_path)
    gt_vol = load_volume(gt_vol_path)

    if pred_vol.shape != gt_vol.shape:
        logger.warning(f"⚠️ Shape mismatch: {pred_vol.shape} vs {gt_vol.shape}")
        return False

    return True


def predicted_volumes_complete(patient_dir):
    """
    Checks that all three predicted volumes (axial, coronal, sagital) exist
    for a patient within their prediction directory.
    """
    patient_id = Path(patient_dir).name
    return all(
        (Path(patient_dir) / f"{patient_id}_{plane}{EXT_NIFTI}").exists()
        for plane in ANATOMICAL_PLANES
    )


def verify_group_volumes(root_dir):
    """
    Verifies that all patients within root_dir have predicted volumes for all
    three anatomical planes (axial, coronal, and sagital).
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


def load_model(model_path):
    """Loads a YOLO model from the given path."""
    try:
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")


def trained_model_exists(model, epochs, fold_test):
    """
    Checks whether the trained model weights exist.

    - If fold_test is None (k_folds == 1): looks in trains/.../plane/weights/best.pt
    - If fold_test is a number (k_folds > 1): looks in trains/.../plane/foldN/weights/best.pt
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


def write_json(dic, json_path):
    """Saves a dictionary as a JSON file."""
    with open(json_path, "w") as f:
        json.dump(dic, f)


def read_json(json_path):
    """Reads a JSON file and returns its contents as a dictionary."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"File not found: {json_path}")


# ======================================
#       PATIENT AND FOLD UTILITIES
# ======================================


def get_id(patient):
    """Extracts the numeric ID from a patient name (P12 → 12)."""
    match = re.search(r"P(\d+)", patient)
    return (
        int(match.group(1)) if match else float("inf")
    )  # Returns a very large value if no number is found


def list_patients(input_dir):
    """
    Returns a sorted list of patient IDs in a directory.
    """
    input_path = Path(input_dir)

    patients = [d.name for d in input_path.iterdir() if not is_ignorable_file(d.name)]
    if not patients:
        raise FileNotFoundError(f"No patients found in {input_dir}.")

    return sorted(patients, key=lambda p: int(p[1:]) if p[1:].isdigit() else 1_000_000)


def compute_fold(patient_id, k_folds=5):
    """Assigns a patient to their corresponding cross-validation fold."""

    # Convert patient ID to number
    numero = int(patient_id[1:])

    # Only train-split patients (P1–P53) are used in CV mode
    all_ids = list(range(1, N_TRAIN_PATIENTS + 1))

    # Split consecutively into k_folds
    folds = np.array_split(all_ids, k_folds)

    # Find which fold the patient belongs to
    for i, fold in enumerate(folds, 1):
        if numero in fold:
            return i

    raise ValueError(f"Cannot compute fold for patient {patient_id}.")


def get_patient_slices(patient, model):
    """
    Returns a sorted list of available slice indices for a patient in a given
    plane, extracted from the images/ subdirectory of the YOLO dataset.
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


def int_or_percentile(value):
    """Accepts integer values or percentile strings ('P<n>')."""
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


def load_png(path):
    """
    Loads a PNG file in greyscale and returns it as a NumPy array.
    """
    return np.array(Image.open(path).convert("L"))


def prepare_pred_gt_slices(img_path, pred_path, gt_path):
    """
    Loads and prepares the image, prediction mask, and GT mask for the same
    slice, applying the geometric correction required for the prediction.
    """
    img = load_png(img_path)
    pred = (load_png(pred_path) > 0).astype(float)
    gt = (load_png(gt_path) > 0).astype(float)

    pred = np.rot90(pred, 1)  # Corrective rotation

    return img, pred, gt


def normalize_binary_mask(mask_path):
    """
    Normalises and saves a binary mask to values 0 (background) and 1 (object).
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_bin)


def normalize_to_uint8(image):
    """
    Normalises an image (float32/64) to the range 0–255 and type uint8.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image -= np.min(image)
        if np.ptp(image) > 0:
            image = 255 * (image / np.ptp(image))
        image = image.astype(np.uint8)
    return image


def convert_to_bgr(image):
    """
    Converts a 2D or RGB image to BGR format.
    """
    image_uint8 = normalize_to_uint8(image)
    if len(image_uint8.shape) == 2:  # Greyscale image
        img_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    else:  # RGB image
        img_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    return img_bgr


def ensure_grayscale(image):
    """
    Returns the image in greyscale, converting if necessary.
    """
    if image.ndim == 3 and image.shape[2] == 3:  # Colour image (3 channels)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already greyscale


# ======================================
#          RESULT EVALUATION
# ======================================


def evaluate_results(results):
    """
    Evaluates the global status of a list of partial results from different
    pipeline stages.

    Each element of the list can be:
    - True  → the stage executed successfully.
    - None  → the stage was skipped or produced no result.

    Returns:
    - True if all stages were successful.
    - None if no stage produced a result.
    - 'partial' if there is a mix of states (some successful, some not).
    """
    if not results:
        return None  # Avoid failure if the list is empty

    if all(r is None for r in results):
        return None
    elif all(r is True for r in results):
        return True
    else:
        return "partial"


# ======================================
#              METRICS
# ======================================


def DSC(y_true, y_pred):
    """Computes the Dice Similarity Coefficient (DSC)."""
    intersection = np.sum(y_true * y_pred)
    dsc = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

    return float(np.round(dsc, 3))


def precision(y_true, y_pred):
    """Computes precision."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    prec = tp / (tp + fp + 1e-8)

    return float(np.round(prec, 3))


def recall(y_true, y_pred):
    """Computes recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    rec = tp / (tp + fn + 1e-8)

    return float(np.round(rec, 3))


def AUC(y_true, y_pred):
    """Computes the Area Under the ROC Curve (AUC)."""
    try:
        # Flatten arrays
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


def log_fold_status(logger, result, fold):
    """
    Logs the execution status of a fold for a specific pipeline stage.
    """
    if result is None:
        logger.skip(f"⏩ Fold {fold} already exists.")
    elif result is True or isinstance(result, (dict, list)):
        logger.info(f"🆗 Fold {fold} completed.")
    elif result == "partial":
        logger.info(f"🔁 Fold {fold} partially updated.")
    else:
        logger.warning(f"⚠️ Fold {fold}: unknown status.")
