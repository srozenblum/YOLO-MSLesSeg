"""
Script: generate_predictions_gif.py

Description:
    Generates an animated GIF that iterates over all available slices of a patient
    and overlays the prediction mask (red) and the ground truth mask (green).

CLI Arguments:
    --patient_id (str, required)
        ID of the patient to visualise.

    --plane (str, required)
        Anatomical extraction plane ('axial', 'coronal', 'sagital').

    --modality (list[str], optional)
        MRI modality or modalities ('T1', 'T2', 'FLAIR').
        Defaults to all.

    --num_slices (int_or_percentile, required)
        Number of extracted slices (integer value or percentile, e.g. 50 or 'P50').

    --enhancement (str, optional)
        Image enhancement algorithm applied ('HE', 'CLAHE', 'GC', 'LT', or None).
        Defaults to None.

    --epochs (int, required)
        Number of epochs of the trained model.

    --k_folds (int, optional)
        Number of folds for cross-validation.
        Defaults to 5.

    --clean (flag, optional)
        If a previous GIF exists, deletes it before generating a new one.

CLI Usage:
    python -m yolo_mslesseg.extras.generate_predictions_gif \\
        --patient_id P14 \\
        --plane sagital \\
        --modality FLAIR \\
        --num_slices P50 \\
        --enhancement HE \\
        --epochs 50 \\
        --k_folds 5

Inputs:
    - Images, predicted masks, and ground truth masks in PNG format.

Outputs:
    - Animated GIF with all predictions for the patient.
"""

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    compute_fold,
    path_exists,
    build_config_name,
    create_directory,
    patient_paths,
    get_patient_slices,
    prepare_pred_gt_slices,
)

logger = get_logger(__file__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ======================================
#           BASE FUNCTIONS
# ======================================


def normalize_img_global(img_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """
    Normalises an image to the [0, 1] range using global minimum and maximum
    values (shared across slices), avoiding division by zero.
    """
    img_array = img_array.astype(float)
    denom = vmax - vmin
    if denom <= 0:
        denom = 1.0
    return (img_array - vmin) / (denom + 1e-8)


def load_slice_series(patient: Patient, model: Model) -> tuple[list, list, list]:
    """
    Loads and validates all available slices of the patient for the given model.
    Returns three parallel lists:
        - images: list of (slice_num, img_array) tuples
        - preds:  list of prediction masks
        - gts:    list of ground truth masks
    """
    slices = get_patient_slices(patient, model)

    if not slices:
        raise RuntimeError(
            f"No PNG slices found for patient {patient.id}."
        )

    images, preds, gts = [], [], []

    for slice_num in slices:
        paths = patient_paths(patient, model, slice_num)

        # Validate that all files exist
        for file_type, path in paths.items():
            if not path_exists(path):
                raise RuntimeError(
                    f"Missing '{file_type}' file for slice {slice_num} of patient "
                    f"{patient.id}: {path}"
                )

        img_array, pred_array, gt_array = prepare_pred_gt_slices(
            img_path=paths["img"],
            pred_path=paths["pred"],
            gt_path=paths["gt"],
        )

        images.append((slice_num, img_array))
        preds.append(pred_array)
        gts.append(gt_array)

    return images, preds, gts


def compute_global_range(images: list) -> tuple[float, float]:
    """
    Computes the global minimum and maximum intensity from
    the list of images [(slice_num, img_array), ...].
    """
    global_min = min(img.min() for _, img in images)
    global_max = max(img.max() for _, img in images)
    return global_min, global_max


# ======================================
#          FRAME GENERATION
# ======================================


def create_frame(
    img_array: np.ndarray,
    pred_array: np.ndarray,
    gt_array: np.ndarray,
    slice_num: int,
    patient: Patient,
    enhancement: str | None,
    vmin: float,
    vmax: float,
) -> Image.Image:
    """
    Generates a GIF frame overlaying:
        - Ground Truth (blue)
        - False Positives (orange)
        - True Positives (green)
    on the original image.
    """
    str_enhancement = enhancement if enhancement is not None else "Base"

    # Global normalisation
    norm = (img_array - vmin) / (vmax - vmin + 1e-8)

    # Square figure without borders
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.margins(0, 0)
    ax.set_position([0, 0, 1, 1])

    # Base image
    ax.imshow(norm, cmap="gray", vmin=0, vmax=1)

    # TP / FP / FN masks
    tp = (pred_array == 1) & (gt_array == 1)  # True positives
    fp = (pred_array == 1) & (gt_array == 0)  # False positives
    fn = (pred_array == 0) & (gt_array == 1)  # False negatives

    # Overlay masks in correct order: FN → FP → TP
    ax.imshow(
        np.ma.masked_where(fn == 0, fn), cmap=ListedColormap(["#0099FF"]), alpha=0.5
    )  # Blue
    ax.imshow(
        np.ma.masked_where(fp == 0, fp), cmap=ListedColormap(["#FF4500"]), alpha=0.5
    )  # Orange
    ax.imshow(
        np.ma.masked_where(tp == 0, tp), cmap=ListedColormap(["#00CC66"]), alpha=0.7
    )  # Green

    # Title (top, centred)
    ax.text(
        0.5,
        0.985,
        f"{patient.id} – {str_enhancement} – {patient.plane.capitalize()}",
        ha="center",
        va="top",
        color="white",
        fontsize=22,
        fontweight="bold",
        family="Arial",
        transform=ax.transAxes,
    )

    # Slice number (bottom left)
    ax.text(
        0.01,
        0.005,
        f"Slice {slice_num}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=16,
        fontweight="bold",
        family="Arial",
        transform=ax.transAxes,
    )

    # Legend (bottom right)
    ax.legend(
        handles=[
            mpatches.Patch(color="#00CC66", label="True positives"),
            mpatches.Patch(color="#FF4500", label="False positives"),
            mpatches.Patch(color="#0099FF", label="False negatives"),
        ],
        loc="lower right",
        prop={"family": "Arial", "weight": "bold", "size": 10},
        frameon=True,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        framealpha=0.6,
    )

    # Save frame
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=120, pad_inches=0, facecolor="black", bbox_inches="tight"
    )
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)


def build_gif_frames(
    patient: Patient,
    images: list,
    preds: list,
    gts: list,
    vmin: float,
    vmax: float,
) -> list:
    """
    Builds the list of frames for the GIF from the globally normalised
    images and their associated masks.
    """
    frames = []

    for (slice_num, img), pred, gt in zip(images, preds, gts):
        frame = create_frame(
            img_array=img,
            pred_array=pred,
            gt_array=gt,
            slice_num=slice_num,
            patient=patient,
            enhancement=patient.enhancement,
            vmin=vmin,
            vmax=vmax,
        )
        frames.append(frame)

    return frames


# ======================================
#              PROCESSING
# ======================================


def generate_gif(patient: Patient, model: Model, output_path: Path) -> None:
    """
    Generates a GIF iterating over all slices of the patient, overlaying
    the prediction and ground truth, with global intensity normalisation
    and FPS adjusted to the number of slices.
    """
    # Load and validate all slices
    images, preds, gts = load_slice_series(patient, model)

    # Global normalisation
    global_min, global_max = compute_global_range(images)

    # Build frames
    frames = build_gif_frames(
        patient=patient,
        images=images,
        preds=preds,
        gts=gts,
        vmin=global_min,
        vmax=global_max,
    )

    if not frames:
        raise RuntimeError(
            f"Could not generate frames for patient {patient.id}."
        )

    # Set duration
    fps = max(3, min(12, len(frames) // 4))
    duration_ms = int(1000 / fps)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


# ======================================
#               MAIN FLOW
# ======================================


def run_flow(patient: Patient, model: Model, epochs: int, clean: bool) -> None:
    """Generates the full GIF combining all slices of the patient."""
    logger.header("\n🎥️ Generating predictions GIF")

    root = Path.cwd()  # respects demo/ if called from a demo script
    global_config = build_config_name(model, epochs)

    patient_id = patient.id
    plane = patient.plane
    enhancement = patient.enhancement if patient.enhancement else "Base"

    # k_folds == 1 → save to test/ (only test patients allowed)
    if model.k_folds == 1:
        if getattr(patient, "split", None) != "test":
            raise ValueError(
                f"Patient {patient_id} belongs to 'train'. "
                "With k_folds == 1, GIFs can only be generated for 'test' patients."
            )

        output_dir = (
            root
            / "visualizaciones"
            / enhancement
            / global_config
            / "test"
            / patient_id
            / plane
        )

    # k_folds > 1 → save to foldX/
    else:
        patient_fold = compute_fold(patient_id, model.k_folds)

        output_dir = (
            root
            / "visualizaciones"
            / enhancement
            / global_config
            / f"fold{patient_fold}"
            / patient_id
            / plane
        )

    create_directory(output_dir)

    output_path = output_dir / f"{patient.id}_{model.modality_str}.gif"

    # Clean if requested
    if clean and path_exists(output_path):
        logger.info(f"♻️ Deleting previous GIF.")
        output_path.unlink()

    generate_gif(patient=patient, model=model, output_path=output_path)

    logger.info("✅ GIF generated successfully.")


# ======================================
#          CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parses the script arguments.
    If no argument list is provided, reads from the command line.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=(
            "Generate a GIF with all slices of a patient, "
            "overlaying the prediction and the ground truth "
            "on the original image."
        )
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        metavar="<patient_id>",
        help="ID of the patient to visualise.",
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
        choices=["HE", "CLAHE", "GC", "LT"],
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
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete the previous GIF before generating a new one.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point: parses arguments, builds the Model and Patient instances,
    and executes the GIF generation.
    """
    args = parse_args(argv)

    model = Model(
        plane=args.plane,
        num_slices=args.num_slices,
        modality=args.modality,
        enhancement=args.enhancement,
        k_folds=args.k_folds,
    )

    patient = Patient(
        id=args.patient_id,
        plane=model.plane,
        modality=args.modality,
        enhancement=model.enhancement,
    )

    run_flow(
        patient=patient,
        model=model,
        epochs=args.epochs,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
