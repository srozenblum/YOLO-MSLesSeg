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

CLI Usage:
    python -m yolo_mslesseg.scripts.extract_dataset \
        --plane "sagittal" \
        --modality "T1" \
        --num_slices 100 \
        --k_folds 4 \
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
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigDataset import ConfigDataset
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_PNG, ENHANCEMENTS, SPLIT_TRAIN, SPLIT_TEST, StageResult
from yolo_mslesseg.utils.utils import (
    list_patients,
    normalize_binary_mask,
    evaluate_results,
    int_or_percentile,
    compute_fold,
)

logger = get_logger(__file__)

# Suppress ultralytics logging
LOGGER.setLevel(logging.WARNING)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def compute_percentile_slice_count(input_dir: Path, plane: str, modality: list[str], percentile: int = 50) -> int:
    """Computes the number of slices based on the global percentile of lesion-containing slices.

    Args:
        input_dir: Directory containing patient subdirectories.
        plane: Anatomical plane to use ('axial', 'coronal', 'sagittal').
        modality: List of MRI modalities to include.
        percentile: Percentile to compute over the distribution of slice counts.

    Returns:
        Number of slices corresponding to the given percentile across all patients.

    Raises:
        ValueError: If no valid lesion slices are found or the percentile is invalid.
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
        num_slices = int(np.percentile(slice_counts, percentile))
    except Exception as e:
        raise ValueError(f"Invalid percentile ({percentile}): {e}")

    return num_slices


def resolve_num_slices(num_slices: int | str | None, input_dir: Path, plane: str, modality: list[str]) -> tuple[int | None, int | None]:
    """Resolves the number of slices to extract from a fixed value or a percentile string.

    Args:
        num_slices: Fixed integer count, a percentile string (e.g. 'P50'), or None.
        input_dir: Directory containing patient subdirectories (used for percentile computation).
        plane: Anatomical plane ('axial', 'coronal', 'sagittal').
        modality: List of MRI modalities to include.

    Returns:
        Tuple of (num_slices, percentile) where percentile is None for fixed values.

    Raises:
        ValueError: If num_slices has an unrecognised format.
    """
    if isinstance(num_slices, int) or num_slices is None:
        return num_slices, None

    if isinstance(num_slices, str) and num_slices.startswith("P"):
        percentile = int(num_slices[1:])
        num_slices_percentile = compute_percentile_slice_count(
            input_dir=input_dir,
            plane=plane,
            modality=modality,
            percentile=percentile,
        )
        return num_slices_percentile, percentile

    raise ValueError(f"Invalid num_slices format: {num_slices}.")


def build_paths(patient: Patient, config: ConfigDataset, group: str | None = None) -> dict[str, Path]:
    """Builds the output directory paths for a patient's images, GT masks, and labels.

    For k_folds > 1, paths follow datasets/<base_path>/fold{fold}/PX/<plane>/...
    For k_folds == 1, paths follow datasets/<base_path>/{train|test}/PX/<plane>/...
    The group is inferred from config.input_dir if not provided.

    Args:
        patient: Patient instance whose paths are being resolved.
        config: ConfigDataset instance providing output directory and fold settings.
        group: Dataset group ('train' or 'test'). Inferred from config.input_dir if None (k_folds == 1 only).

    Returns:
        Dictionary with keys 'images', 'GT_masks', and 'labels' mapping to Path objects.
    """
    if config.k_folds > 1:
        fold = compute_fold(patient_id=patient.id, k_folds=config.k_folds)
        root = config.output_dir / f"fold{fold}" / patient.id / patient.plane
    else:
        if group is None:
            path_norm = str(config.input_dir).replace("\\", "/").lower()
            group = SPLIT_TEST if path_norm.endswith("/test") else SPLIT_TRAIN
        root = config.output_dir / group / patient.id / patient.plane

    return {
        "images": root / "images",
        "GT_masks": root / "GT_masks",
        "labels": root / "labels",
    }


def save_slices(patient: Patient, images_dir: Path, gt_masks_dir: Path, num_slices: int | None) -> None:
    """Saves image and mask slices for a patient to the output directories.

    Each image is saved as a 3-channel (RGB) PNG with one channel per modality.
    If fewer than 3 modalities are configured, the last channel is repeated.
    Filenames follow the pattern {patient_id}_{slice_index}.png, where
    slice_index is the original slice position within the volume (not a
    sequential counter), so that the index can be recovered during reconstruction.

    Args:
        patient: Patient instance providing the slice data.
        images_dir: Directory where image PNGs will be saved.
        gt_masks_dir: Directory where ground truth mask PNGs will be saved.
        num_slices: Maximum number of slices to extract, or None for all lesion slices.

    Raises:
        ValueError: If no valid slices are found for the patient.
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


def normalize_masks(gt_masks_dir: Path) -> None:
    """Normalises all PNG masks in a directory to binary values (0 and 1).

    Args:
        gt_masks_dir: Directory containing the PNG mask files to normalise.

    Raises:
        FileNotFoundError: If no PNG masks are found in gt_masks_dir.
        OSError: If a mask file cannot be normalised.
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


def annotate_masks(gt_masks_dir: Path, labels_dir: Path) -> None:
    """Converts ground truth masks to YOLO segmentation annotation format.

    Normalises the masks before conversion.

    Note:
        Existing .txt files in labels_dir are overwritten without warning.
        If the set of masks has changed since the last run, remove stale label
        files manually before calling this function.

    Args:
        gt_masks_dir: Directory containing the PNG ground truth masks.
        labels_dir: Directory where the YOLO .txt annotation files will be written.
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


def process_patient_dataset(patient: Patient, config: ConfigDataset, paths_dir: dict[str, Path] | None = None, num_slices: int | None = None) -> StageResult:
    """Executes the slice extraction and annotation process for an individual patient.

    Skips extraction if all output directories already exist and are non-empty.

    Args:
        patient: Patient instance to process.
        config: ConfigDataset instance providing the patient output paths.
        paths_dir: Dictionary of output paths (images, GT_masks, labels). Defaults to
            config.patient_dir if None.
        num_slices: Maximum number of slices to extract, or None for all lesion slices.

    Returns:
        StageResult.COMPLETED if extraction was performed, StageResult.SKIPPED if
        skipped (already exists).
    """
    if paths_dir is None:
        paths_dir = config.patient_dir

    if all(path.is_dir() and any(path.iterdir()) for path in paths_dir.values()):
        return StageResult.SKIPPED

    save_slices(
        patient=patient,
        images_dir=paths_dir["images"],
        gt_masks_dir=paths_dir["GT_masks"],
        num_slices=num_slices,
    )

    annotate_masks(gt_masks_dir=paths_dir["GT_masks"], labels_dir=paths_dir["labels"])
    return StageResult.COMPLETED


def save_patient_slices(input_dir: Path, config: ConfigDataset, num_slices: int | None, group: str | None = None) -> StageResult:
    """Executes the slice extraction process for all patients in a directory.

    Args:
        input_dir: Directory containing patient subdirectories to process.
        config: ConfigDataset instance providing output paths and model settings.
        num_slices: Maximum number of slices to extract per patient, or None for all.
        group: Dataset group ('train' or 'test'). Inferred from config if None.

    Returns:
        StageResult.COMPLETED if all patients were processed, StageResult.SKIPPED
        if all were skipped, or StageResult.PARTIAL if there was a mix.
    """
    patients = list_patients(input_dir)

    results = []
    for patient_id in patients:
        patient = Patient(
            id=patient_id,
            plane=config.model.plane,
            modality=config.model.modality,
            enhancement=config.model.enhancement,
            gamma=config.model.gamma,
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


def run_dataset_flow(config: ConfigDataset, clean: bool, verbose: bool = False) -> None:
    """Executes the main YOLO dataset extraction flow.

    Args:
        config: ConfigDataset instance defining the extraction configuration.
        clean: If True, deletes existing dataset outputs before extracting.
        verbose: If True, logs a header message at the start of execution.
    """
    if verbose:
        str_header = f"patient {config.patient}" if config.is_individual_patient else "full patient set"
        logger.header(
            f"\n🧩 Preparing YOLO dataset for the {str_header}."
        )

    # --- Setup: clean, verify paths, resolve slice count ---
    if clean:
        if verbose:
            logger.info("♻️ Cleaning previous YOLO dataset.")
        config.clean_dataset()

    config.verify_paths()

    num_slices, percentile = resolve_num_slices(
        num_slices=config.model.num_slices,
        input_dir=config.input_dir,
        plane=config.model.plane,
        modality=config.model.modality,
    )

    if config.is_individual_patient:
        slices_str = "all lesion slices" if num_slices is None else str(num_slices)
        logger.info(f"📊 Slices to extract: {slices_str} — 1 patient.")
    else:
        n_train = len(list_patients(config.get_input_dir(SPLIT_TRAIN)))
        has_test = config.k_folds == 1 and config.mslesseg_test_dir.is_dir()
        n_test = len(list_patients(config.get_input_dir(SPLIT_TEST))) if has_test else 0
        n_total = n_train + n_test
        percentile_str = f" (P{percentile})" if percentile is not None else ""

        if num_slices is None:
            logger.info(f"📊 Slices to extract: all lesion slices — {n_total} patients.")
        elif has_test:
            logger.info(
                f"📊 Slices to extract: {num_slices * n_total}{percentile_str} total "
                f"({num_slices * n_train} train / {num_slices * n_test} test) — {num_slices} slices × {n_total} patients."
            )
        else:
            logger.info(
                f"📊 Slices to extract: {num_slices * n_total}{percentile_str} total — "
                f"{num_slices} slices × {n_total} patients."
            )

    # --- Process patients ---
    # =========================
    #   INDIVIDUAL PATIENT MODE
    # =========================
    if config.is_individual_patient:
        dataset_extracted = process_patient_dataset(
            patient=config.patient,
            config=config,
            num_slices=num_slices,
        )
        if dataset_extracted is StageResult.SKIPPED:
            logger.skip("⏩ YOLO dataset already exists.")
        elif dataset_extracted is StageResult.COMPLETED:
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
            input_dir=config.get_input_dir(SPLIT_TRAIN),
            config=config,
            num_slices=num_slices,
        )

        if processed is StageResult.SKIPPED:
            logger.skip("⏩ YOLO dataset already exists.")
        elif processed is StageResult.COMPLETED:
            logger.info("🆗 YOLO dataset extracted successfully.")
        elif processed is StageResult.PARTIAL:
            logger.info("🔁 YOLO dataset partially updated.")
        else:
            logger.warning("⚠️ Unknown status when extracting the YOLO dataset.")
        return

    # k_folds == 1 → train + test
    processed_train = save_patient_slices(
        input_dir=config.get_input_dir(SPLIT_TRAIN),
        config=config,
        num_slices=num_slices,
        group=SPLIT_TRAIN,
    )

    processed_test = save_patient_slices(
        input_dir=config.get_input_dir(SPLIT_TEST),
        config=config,
        num_slices=num_slices,
        group=SPLIT_TEST,
    )

    if processed_train is StageResult.SKIPPED and processed_test is StageResult.SKIPPED:
        logger.skip("⏩ YOLO dataset already exists.")
    elif processed_train is StageResult.COMPLETED and processed_test is StageResult.COMPLETED:
        logger.info("🆗 YOLO dataset extracted successfully.")
    elif processed_train is StageResult.PARTIAL or processed_test is StageResult.PARTIAL:
        logger.info("🔁 YOLO dataset partially updated.")
    else:
        logger.info("🔁 YOLO dataset partially updated.")


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the dataset extraction script.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
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
        choices=["axial", "coronal", "sagittal"],
        metavar="[axial, coronal, sagittal]",
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
        default=1,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 1.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        metavar="<input_dir>",
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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parses arguments and executes the dataset extraction flow.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.
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
            input_dir=args.input_dir,
            patient=patient,
        )
    else:
        config = ConfigDataset(
            model=model,
            input_dir=args.input_dir,
            full=True,
        )

    run_dataset_flow(
        config=config,
        clean=args.clean,
        verbose=True,
    )


def run_dataset_pipeline(model: Model, patient: Patient | None = None, clean: bool = False) -> None:
    """Internal pipeline entry point: executes the dataset extraction flow programmatically.

    Args:
        model: Model instance defining the extraction configuration.
        patient: Patient instance for individual execution, or None for full mode.
        clean: If True, deletes existing dataset outputs before extracting.
    """
    if patient is not None:
        config = ConfigDataset(
            model=model,
            patient=patient,
        )
    else:
        config = ConfigDataset(
            model=model,
            full=True,
        )

    run_dataset_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
