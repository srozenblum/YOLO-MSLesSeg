"""
Script: generate_predictions.py

Description:
    Applies a trained YOLO model to generate 2D prediction masks for each
    slice in the test set, either for an individual patient or for all
    patients in a fold. Can be executed both as a standalone script (from
    the CLI) and internally from within the pipeline (`run_pipeline.py`).

    The generated masks are stored in the pred_masks/ subdirectory within
    each patient directory, maintaining the original dataset structure.

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.
       - Model and (optionally) Patient instances are created.

    2. Internal (from `run_pipeline.py`):
       - Pre-built Model and (optionally) Patient instances are received,
         along with the remaining parameters.
       - The argument parser is not used.

CLI Arguments:
    --plane (str, required)
        Anatomical extraction plane ('axial', 'coronal', 'sagital').

    --modality (list[str], optional)
        MRI modality or modalities ('T1', 'T2', 'FLAIR').
        Defaults to all.

    --num_slices (int_or_percentile, required)
        Number of extracted slices (integer value or percentile, e.g. 50 or 'P75').

    --enhancement (str, optional)
        Image enhancement algorithm applied ('HE', 'CLAHE', 'GC', 'LT', or None).
        Defaults to None.

    --epochs (int, required)
        Number of epochs of the trained model.

    --k_folds (int, optional)
        Number of folds for cross-validation.
        Defaults to 5.

    --fold_test (int, mutually exclusive with --patient_id)
        Generate 2D predictions for all patients in the indicated fold,
        used as the test set.

    --patient_id (str, mutually exclusive with --fold_test)
        Generate 2D predictions only for the specified patient.

    --clean (flag, optional)
        Clean the directory with binary 2D predictions before generating new ones.

CLI Usage:
    python -m yolo_mslesseg.scripts.generate_predictions \\
        --plane coronal \\
        --modality FLAIR \\
        --num_slices 50 \\
        --epochs 100 \\
        --fold_test 3

Inputs:
    - Dataset: generated previously with extract_dataset.py and trained with train.py.
        Contains the input images and their YOLO annotations split by folds.

    - Model weights: best.pt file.
        Trained YOLO model weights file, located in
        trains/<enhancement>/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/
        <plane>/<fold_test>/weights/best.pt.

    - Classes:
        * ConfigPred → manages directories and global variables for prediction generation.
        * Model      → defines the plane, modalities, enhancement, and num_slices.

Outputs:
    - 2D prediction masks (.png) in the patient's pred_masks/ subdirectory.
"""

import argparse
import sys

import cv2
import numpy as np
from tqdm import tqdm

from yolo_mslesseg.configs.ConfigPred import ConfigPred
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_PNG, ENHANCEMENTS
from yolo_mslesseg.utils.utils import (
    list_patients,
    int_or_percentile,
    path_exists,
    evaluate_results,
    create_directory,
    load_model,
    log_fold_status,
)

# Configure logger
logger = get_logger(__file__)


# ======================================
#              BASE FUNCTIONS
# ======================================


def run_prediction(model, img_array):
    """Runs the YOLO model on an image and returns the raw masks."""
    try:
        pred = model(img_array, verbose=False)[0]
    except Exception as e:
        raise RuntimeError(f"Error running model prediction: {e}.")

    if pred.masks is None:
        return []
    return pred.masks.data.cpu().numpy()


def combine_predictions(predictions, shape):
    """Combines a list of predictions (binary masks) into a single 2D mask."""
    height, width = shape
    combined = np.zeros((height, width), dtype=np.uint8)

    for pred in predictions:
        binary = (pred > 0.5).astype(np.uint8)
        resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, resized)

    return combined


def normalize_prediction(pred):
    """Converts the predicted mask from image coordinates to NIfTI voxel coordinates (0-255)."""
    pred_normalised = pred.T.copy()
    pred_normalised *= 255
    return pred_normalised


def save_prediction(pred, image_filename, output_dir):
    """Saves the binary prediction in PNG format."""
    if pred is None or pred.size == 0:
        logger.warning(f"⚠️ Empty prediction for {image_filename}, nothing saved.")
        return None

    create_directory(output_dir)
    output_path = output_dir / f"{image_filename}{EXT_PNG}"

    if not path_exists(output_path):
        cv2.imwrite(output_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return output_path


def fold_predictions_complete(fold_dir, plane, single_fold=False):
    """
    Returns True if all patients in fold_dir have non-empty pred_masks directories.
    """

    for patient_id in list_patients(fold_dir):
        patient_pred_masks_dir = fold_dir / patient_id / plane / "pred_masks"

        if not patient_pred_masks_dir.exists() or not any(
            patient_pred_masks_dir.glob(f"*{EXT_PNG}")
        ):
            return False
    return True


# ======================================
#          PER-IMAGE PREDICTION
# ======================================


def generate_2d_prediction(model, img_array, image_filename, output_dir):
    """Applies the YOLO model to an image and saves the resulting binary prediction."""
    raw_predictions = run_prediction(model=model, img_array=img_array)
    combined = combine_predictions(
        predictions=raw_predictions, shape=img_array.shape[:2]
    )
    normalised = normalize_prediction(pred=combined)
    save_prediction(
        pred=normalised,
        image_filename=image_filename,
        output_dir=output_dir,
    )


def get_patient_images(patient_id, images_dir):
    """Returns a sorted list of full paths to the patient's PNG images."""
    if not path_exists(images_dir):
        raise FileNotFoundError(f"Directory {images_dir} does not exist.")

    images = sorted(
        [img for img in images_dir.glob(f"{patient_id}_*{EXT_PNG}") if img.is_file()]
    )

    if not images:
        raise FileNotFoundError(
            f"No PNG images found for {patient_id} in {images_dir}."
        )
    return images


def generate_predictions(model, image_list, output_dir):
    """Applies a YOLO model to all images of a patient."""
    for image_path in image_list:
        image_filename = image_path.stem
        img_array = cv2.imread(str(image_path))

        if img_array is None:
            logger.warning(f"⚠️ Could not load image {image_path}.")
            continue

        generate_2d_prediction(
            model=model,
            img_array=img_array,
            image_filename=image_filename,
            output_dir=output_dir,
        )


# ======================================
#              PROCESSING
# ======================================


def process_patient_predictions(
    patient_id, config, paths_dir=None, yolo_model=None
):
    """
    Executes the full prediction process
    (model loading → predictions → mask saving)
    for an individual patient.
    """
    # If no model is provided → load it
    if yolo_model is None:
        yolo_model = load_model(config.model_path)

    # If no directories are provided → patient mode → use config directories
    if paths_dir is None:
        paths_dir = config.patient_dir

    patient_images_dir = paths_dir["images"]
    patient_pred_masks_dir = paths_dir["pred_masks"]

    # Skip if results already exist
    if path_exists(patient_pred_masks_dir) and any(
        patient_pred_masks_dir.glob(f"*{EXT_PNG}")
    ):
        return

    # Get patient images
    image_list = get_patient_images(
        patient_id=patient_id, images_dir=patient_images_dir
    )

    if not image_list:
        raise RuntimeError(f"No valid images found in {patient_images_dir}.")

    generate_predictions(
        model=yolo_model,
        image_list=image_list,
        output_dir=patient_pred_masks_dir,
    )
    return True


def build_paths(patient_id, config):
    """
    Builds a dictionary of paths (images, pred_masks) for an individual patient.
    """
    root = config.dataset_fold_dir / patient_id / config.plane
    return {
        "images": root / "images",
        "pred_masks": root / "pred_masks",
    }


def generate_predictions_for_patients(input_dir, config):
    """
    Executes the prediction generation process for all patients in input_dir.
    """
    patients = list_patients(input_dir)
    yolo_model = load_model(config.model_path)
    results = []

    for patient_id in tqdm(patients, desc=f"Patients {input_dir.name}", unit="pat"):
        patient_paths_dir = build_paths(patient_id, config)
        try:
            pred_result = process_patient_predictions(
                patient_id=patient_id,
                config=config,
                paths_dir=patient_paths_dir,
                yolo_model=yolo_model,
            )
            results.append(pred_result)
        except Exception as e:
            logger.warning(
                f"⚠️ Error generating predictions for {patient_id}, skipping: {e}."
            )
            continue

    return evaluate_results(results)


# ======================================
#               MAIN FLOW
# ======================================


def run_prediction_flow(config, clean, verbose=False):
    """Executes the main prediction flow, over a fold or an individual patient."""
    if verbose:
        if config.is_individual_patient:
            str_header = f"patient {config.patient}"
        elif config.single_fold:
            str_header = f"group {config.group}"
        else:
            str_header = f"fold {config.fold_test}"

        logger.header(f"\n🎯 Generating predictions for {str_header}.")

    # Clean if requested
    if clean:
        if verbose:
            logger.info(f"♻️ Cleaning previous predictions.")
        config.clean()

    # Verify paths
    config.verify_paths()

    # Patient execution
    if config.is_individual_patient:
        predictions_generated = process_patient_predictions(
            patient_id=config.patient.id, config=config
        )
        if predictions_generated is None:
            logger.skip(f"⏩ Predictions already exist.")
        elif predictions_generated is True:
            logger.info(f"✅ Predictions generated successfully.")
        else:
            logger.warning(f"⚠️ Unknown status when generating predictions.")

    # Fold / group execution
    else:
        # Check if the full set already has predictions
        if fold_predictions_complete(
            config.dataset_fold_dir,
            config.plane,
            single_fold=config.single_fold,
        ):
            if config.single_fold:
                logger.skip(f"⏩ Predictions for {config.group} already exist.")
            else:
                logger.skip(f"⏩ Fold {config.fold_test} already exists.")
            return

        processed = generate_predictions_for_patients(
            input_dir=config.dataset_fold_dir, config=config
        )

        if config.single_fold:
            if processed is None:
                logger.skip(f"⏩ Predictions for {config.group} already exist.")
            elif processed is True:
                logger.info(f"🆗 Predictions for {config.group} generated successfully.")
            elif processed == "partial":
                logger.info(
                    f"🔁 Predictions for {config.group} partially updated."
                )
            else:
                logger.warning("⚠️ Unknown status when generating predictions.")
        else:
            log_fold_status(logger=logger, result=processed, fold=config.fold_test)


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args(argv=None):
    """
    Parses the script arguments.
    If no argument list is provided, reads from the command line.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Apply a trained YOLO model to generate 2D prediction masks.",
    )
    parser.add_argument(
        "--plane",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Anatomical extraction plane.",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=["T1", "T2", "FLAIR"],
        default=["T1", "T2", "FLAIR"],
        metavar="[T1, T2, FLAIR]",
        help="MRI modality or modalities. Defaults to all.",
    )
    parser.add_argument(
        "--num_slices",
        type=int_or_percentile,
        required=True,
        metavar="<num_slices>",
        help="Number of extracted slices (fixed value or percentile).",
    )
    parser.add_argument(
        "--enhancement",
        type=str,
        default=None,
        choices=list(ENHANCEMENTS),
        metavar="<enhancement>",
        help="Image enhancement algorithm applied. Defaults to None.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 5.",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--fold_test",
        type=int,
        metavar="<fold_test>",
        help="Generate 2D predictions for the indicated fold, used as the test set.",
    )
    group.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="Generate 2D predictions only for the specified patient.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean previously generated binary 2D predictions.",
    )

    args = parser.parse_args(argv)

    # Validations
    if args.k_folds == 1 and args.fold_test is not None:
        parser.error("--fold_test must not be specified when --k_folds == 1.")
    if args.k_folds > 1 and args.fold_test is None and args.patient_id is None:
        parser.error(
            "--fold_test or --patient_id must be specified when k_folds > 1."
        )

    return args


def main(argv=None):
    """
    CLI entry point: parses arguments, builds Model/Patient/ConfigPred instances,
    and executes the full flow.
    """
    args = parse_args(argv)

    model = Model(
        plane=args.plane,
        num_slices=args.num_slices,
        modality=args.modality,
        k_folds=args.k_folds,
        enhancement=args.enhancement,
    )

    patient = None
    if args.patient_id is not None:
        patient = Patient(
            id=args.patient_id,
            plane=model.plane,
            modality=model.modality,
            enhancement=model.enhancement,
        )
    config = ConfigPred(
        model=model,
        epochs=args.epochs,
        k_folds=args.k_folds,
        patient=patient,
        fold_test=args.fold_test,
    )
    run_prediction_flow(config=config, clean=args.clean, verbose=True)


def run_predictions_pipeline(
    model, patient=None, fold_test=None, epochs=50, k_folds=5, clean=False
):
    """
    Internal pipeline entry point: receives pre-built objects and executes
    the flow without using the CLI parser.
    """
    config = ConfigPred(
        model=model,
        epochs=epochs,
        k_folds=k_folds,
        patient=patient,
        fold_test=fold_test,
    )
    run_prediction_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
