"""
Script: extract_dataset.py

Description:
    Generates the annotated YOLO dataset for the workflow from the MSLesSeg
    input dataset, either for an individual patient or for all patients.
    Can be executed both as a standalone script (from the CLI) and internally
    from within the pipeline (`run_pipeline.py`).

    Extracts 2D slices from MRI volumes using the selected modalities
    (T1, T2, FLAIR) and the specified anatomical plane, automatically
    generating the directory structure expected by YOLO
    (images/, GT_masks/, labels/).

    Supported output schemes:
    - k_folds > 1: creates fold1/, ..., foldK/ using patients from the train set.
    - k_folds == 1: creates train/ and test/ without fold subdirectories.

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.
       - Model and (optionally) Patient instances are created.

    2. Internal (from `run_pipeline.py`):
       - Pre-built Model and (optionally) Patient instances are received.
       - The argument parser is not used.

CLI Arguments:
    --plane (str, required)
        Anatomical extraction plane ('axial', 'coronal', 'sagital').

    --modality (list[str], optional)
        MRI modality or modalities ('T1', 'T2', 'FLAIR').
        Defaults to all.

    --num_slices (int_or_percentile, required)
        Number of slices to extract (integer value or percentile, e.g. 50 or 'P75').

    --enhancement (str, optional)
        Image enhancement algorithm ('HE', 'CLAHE', 'GC', 'LT', or None).
        Defaults to None.

    --k_folds (int, optional)
        Number of folds for cross-validation.
        - If k_folds > 1: output in fold1..foldK.
        - If k_folds == 1: output in train/ and test/.
        Defaults to 5.

    --full (flag, mutually exclusive with --patient_id)
        Generate the YOLO dataset for all patients.

    --patient_id (str, mutually exclusive with --full)
        Generate the YOLO dataset only for the specified patient (e.g. 'P12').

    --dataset_entrada (str, optional)
        MSLesSeg input dataset directory.
        Defaults to MSLesSeg-Dataset/train.
        Example: MSLesSeg-Dataset/test.

    --clean (flag, optional)
        Clean the previous YOLO dataset before extracting a new one.

CLI Usage:
    python -m yolo_mslesseg.scripts.extract_dataset \\
        --plane "sagital" \\
        --modality "T1" \\
        --num_slices 100 \\
        --k_folds 4 \\
        --full

Inputs:
    - MSLesSeg dataset:
        * MSLesSeg-Dataset/train/
        * MSLesSeg-Dataset/test/ (if k_folds == 1)

    - Classes:
        * ConfigDataset → manages directories and global variables.
        * Model         → defines the plane, modalities, enhancement, and base_path.

Outputs:
    - YOLO structure with images, GT masks, and annotations in .txt format.
"""

import argparse
import logging
import sys

import numpy as np
from matplotlib import pyplot as plt
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigDataset import ConfigDataset
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_PNG, ENHANCEMENTS
from yolo_mslesseg.utils.utils import (
    list_patients,
    normalize_binary_mask,
    evaluate_results,
    int_or_percentile,
    compute_fold,
)

# Configure logger
logger = get_logger(__file__)

# Suppress ultralytics logging
LOGGER.setLevel(logging.WARNING)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def compute_percentile_slice_count(input_dir, plane, modality, percentil=50):
    """
    Computes the number of slices to use based on the global percentile
    of the distribution of lesion-containing slices across all patients.
    """
    patients = list_patients(input_dir)
    slice_counts = []

    for patient_id in patients:
        patient = Patient(id=patient_id, plane=plane, modality=modality)
        indices = patient.slices_to_use()  # All lesion-containing slices
        slice_counts.append(len(indices))

    if not slice_counts:
        raise ValueError(
            f"No valid lesion slices found to compute the percentile in {input_dir}."
        )

    try:
        num_slices = int(np.percentile(slice_counts, percentil))
    except Exception as e:
        raise ValueError(f"Invalid percentile ({percentil}): {e}")

    return num_slices


def resolve_num_slices(num_slices, input_dir, plane, modality):
    """
    Resolves the number of slices to use based on a fixed value or a percentile,
    returning a tuple (num_slices, percentile).
    """
    if isinstance(num_slices, int) or num_slices is None:
        return num_slices, None

    if isinstance(num_slices, str) and num_slices.startswith("P"):
        percentil = int(num_slices[1:])
        num_slices_percentile = compute_percentile_slice_count(
            input_dir=input_dir,
            plane=plane,
            modality=modality,
            percentil=percentil,
        )
        return num_slices_percentile, percentil

    raise ValueError(f"Invalid num_slices format: {num_slices}.")


def build_paths(patient, config, group=None):
    """
    Builds a dictionary of paths (images, GT_masks, labels) for a patient.

    - k_folds > 1:
        datasets/<base_path>/fold{fold}/PX/<plane>/(images|GT_masks|labels)
    - k_folds == 1:
        datasets/<base_path>/{train|test}/PX/<plane>/(images|GT_masks|labels)
        * group is inferred from config.dataset_entrada if not passed explicitly.
    """
    if config.k_folds > 1:
        fold = compute_fold(patient_id=patient.id, k_folds=config.k_folds)
        root = config.output_dir / f"fold{fold}" / patient.id / patient.plane
    else:
        if group is None:
            entrada_norm = str(config.dataset_entrada).replace("\\", "/").lower()
            group = "test" if entrada_norm.endswith("/test") else "train"
        root = config.output_dir / group / patient.id / patient.plane

    return {
        "images": root / "images",
        "GT_masks": root / "GT_masks",
        "labels": root / "labels",
    }


def save_slices(patient, images_dir, gt_masks_dir, num_slices):
    """
    Saves image and mask slices for a patient to the given directories.

    Each image is saved as a 3-channel (RGB) PNG with one channel per modality,
    allowing YOLO to jointly process all modalities. If fewer than 3 modalities
    are configured, the last channel is repeated. Filename: {patient_id}_{i}.png.
    """
    slice_images = patient.lesion_slices_multichannel(num_slices=num_slices)
    slice_masks = patient.lesion_mask_slices(num_slices=num_slices)

    if not slice_images or not slice_masks:
        raise ValueError(
            f"No valid slices found for patient {patient.id}."
        )

    for i, image in slice_images:
        img_path = images_dir / f"{patient.id}_{i}{EXT_PNG}"
        plt.imsave(img_path, image)

    for i, mask in slice_masks:
        gt_mask_path = gt_masks_dir / f"{patient.id}_{i}{EXT_PNG}"
        plt.imsave(gt_mask_path, mask.T, cmap="gray")


def normalize_masks(gt_masks_dir):
    """
    Normalises all masks in gt_masks_dir to binary values (0 and 1).
    """
    files = list(gt_masks_dir.glob(f"*{EXT_PNG}"))
    if not files:
        raise FileNotFoundError(
            f"No {EXT_PNG} masks found in {gt_masks_dir}"
        )

    for path in files:
        try:
            normalize_binary_mask(path)
        except Exception as e:
            raise OSError(f"Error normalising {path.name}: {e}")


def annotate_masks(gt_masks_dir, labels_dir):
    """
    Converts a patient's GT masks to YOLO annotation format.
    """
    normalize_masks(gt_masks_dir)

    convert_segment_masks_to_yolo_seg(
        masks_dir=gt_masks_dir,
        output_dir=labels_dir,
        classes=1,
    )


# ======================================
#             PROCESSING
# ======================================


def process_patient_dataset(patient, config, paths_dir=None, num_slices="P50"):
    """
    Executes the slice extraction process for an individual patient.
    """
    if paths_dir is None:
        paths_dir = config.patient_dir

    if all(path.is_dir() and any(path.iterdir()) for path in paths_dir.values()):
        return

    save_slices(
        patient=patient,
        images_dir=paths_dir["images"],
        gt_masks_dir=paths_dir["GT_masks"],
        num_slices=num_slices,
    )

    annotate_masks(gt_masks_dir=paths_dir["GT_masks"], labels_dir=paths_dir["labels"])
    return True


def save_patient_slices(input_dir, config, num_slices, group=None):
    """
    Executes the slice extraction process for all patients in input_dir.
    """
    patients = list_patients(input_dir)

    results = []
    for patient_id in patients:
        patient = Patient(
            id=patient_id,
            plane=config.model.plane,
            modality=config.model.modality,
            enhancement=config.model.enhancement,
        )

        paths_dir = build_paths(patient=patient, config=config, group=group)

        try:
            patient_processed = process_patient_dataset(
                patient=patient,
                config=config,
                paths_dir=paths_dir,
                num_slices=num_slices,
            )
            results.append(patient_processed)
        except Exception as e:
            logger.warning(
                f"⚠️ Error extracting YOLO dataset for {patient_id}, skipping: {e}."
            )
            continue

    return evaluate_results(results)


# ======================================
#             MAIN FLOW
# ======================================


def run_dataset_flow(config, clean, verbose=False):
    """
    Executes the main YOLO dataset extraction flow.
    """
    if verbose:
        str_full = "full patient set"
        str_patient = f"patient {config.patient}"
        logger.header(
            f"\n🧩 Preparing YOLO dataset for the {str_patient if config.is_individual_patient else str_full}."
        )

    if clean:
        if verbose:
            logger.info("♻️ Cleaning previous YOLO dataset.")
        config.clean_dataset()

    config.verify_paths()

    num_slices, percentil = resolve_num_slices(
        num_slices=config.model.num_slices,
        input_dir=config.dataset_entrada,
        plane=config.model.plane,
        modality=config.model.modality,
    )

    if percentil is None:
        logger.info(f"📊 Number of slices to extract: {num_slices}.")
    else:
        logger.info(f"📊 Number of slices to extract: {num_slices} (P{percentil}).")

    # =========================
    #   INDIVIDUAL PATIENT MODE
    # =========================
    if config.is_individual_patient:
        dataset_extracted = process_patient_dataset(
            patient=config.patient,
            config=config,
            num_slices=num_slices,
        )
        if dataset_extracted is None:
            logger.skip("⏩ YOLO dataset already exists.")
        elif dataset_extracted is True:
            logger.info("✅ Slice extraction completed.")
            logger.info("📝 Annotations completed.")
        else:
            logger.warning("⚠️ Unknown status when extracting the YOLO dataset.")
        return

    # =========================
    #       FULL DATASET MODE
    # =========================

    # k_folds > 1 → train only (folds)
    if config.k_folds > 1:
        processed = save_patient_slices(
            input_dir=config.input_dir("train"),
            config=config,
            num_slices=num_slices,
        )

        if processed is None:
            logger.skip("⏩ YOLO dataset already exists.")
        elif processed is True:
            logger.info("🆗 YOLO dataset extracted successfully.")
        elif processed == "partial":
            logger.info("🔁 YOLO dataset partially updated.")
        else:
            logger.warning("⚠️ Unknown status when extracting the YOLO dataset.")
        return

    # k_folds == 1 → train + test
    processed_train = save_patient_slices(
        input_dir=config.input_dir("train"),
        config=config,
        num_slices=num_slices,
        group="train",
    )

    processed_test = save_patient_slices(
        input_dir=config.input_dir("test"),
        config=config,
        num_slices=num_slices,
        group="test",
    )

    if processed_train is None and processed_test is None:
        logger.skip("⏩ YOLO dataset already exists.")
    elif processed_train is True and processed_test is True:
        logger.info("🆗 YOLO dataset extracted successfully.")
    elif processed_train == "partial" or processed_test == "partial":
        logger.info("🔁 YOLO dataset partially updated.")
    else:
        logger.warning("⚠️ Unknown status when extracting YOLO train/ dataset.")


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args(argv=None):
    """
    Parses the script arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Extract a YOLO dataset for the YOLO-MSLesSeg workflow.",
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
        help="Number of slices to extract (integer value or percentile).",
    )
    parser.add_argument(
        "--enhancement",
        type=str,
        default=None,
        choices=list(ENHANCEMENTS),
        metavar="[HE, CLAHE, GC, LT]",
        help="Image enhancement algorithm to apply. Defaults to None.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 5.",
    )
    parser.add_argument(
        "--dataset_entrada",
        type=str,
        default=None,
        metavar="<dataset_entrada>",
        help="MSLesSeg input dataset directory. Defaults to MSLesSeg-Dataset/train.",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--full",
        action="store_true",
        help="Extract the YOLO dataset for all patients.",
    )
    group.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="Extract the YOLO dataset only for the specified patient.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the previously extracted YOLO dataset.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    CLI entry point: parses arguments, builds Model/Patient/ConfigDataset
    instances, and executes the full flow.
    """
    args = parse_args(argv)

    model = Model(
        plane=args.plane,
        num_slices=args.num_slices,
        modality=args.modality,
        k_folds=args.k_folds,
        enhancement=args.enhancement,
    )

    if args.patient_id is not None:
        patient = Patient(
            id=args.patient_id,
            plane=model.plane,
            modality=model.modality,
            enhancement=model.enhancement,
        )
        config = ConfigDataset(
            model=model,
            dataset_entrada=args.dataset_entrada,
            k_folds=args.k_folds,
            patient=patient,
        )
    else:
        config = ConfigDataset(
            model=model,
            dataset_entrada=args.dataset_entrada,
            k_folds=args.k_folds,
            full=True,
        )

    run_dataset_flow(
        config=config,
        clean=args.clean,
        verbose=True,
    )


def run_dataset_pipeline(model, patient=None, k_folds=5, clean=False):
    """
    Internal pipeline entry point: receives pre-built objects and executes
    the flow without using the CLI parser.
    """
    if patient is not None:
        config = ConfigDataset(
            model=model,
            k_folds=k_folds,
            patient=patient,
        )
    else:
        config = ConfigDataset(
            model=model,
            k_folds=k_folds,
            full=True,
        )

    run_dataset_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
