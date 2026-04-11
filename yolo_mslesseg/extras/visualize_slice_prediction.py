"""
Script: visualize_slice_prediction.py

Description:
    Generates a figure to visualize the model prediction on a specific slice of
    a patient and compare it with the ground truth mask, overlaying both on the
    original image. If no specific slice is indicated, the script evaluates all
    available slices, computes the DSC for each one, and visualizes only the
    slice with the best performance.

CLI Usage:
    python -m yolo_mslesseg.extras.visualize_slice_prediction \
        --patient_id P14 \
        --plane sagittal \
        --modality FLAIR \
        --num_slices P50 \
        --enhancement HE \
        --epochs 50 \
        --k_folds 5

Inputs:
    - Image of the selected slice in PNG format.
    - Predicted and ground truth masks in PNG format.

Outputs:
    - PNG image with the generated visualization.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.constants import EXT_PNG, VISUALIZATIONS_DIR
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.utils import (
    compute_fold,
    int_or_percentile,
    DSC,
    create_directory,
    build_config_name,
    path_exists,
    patient_paths,
    get_patient_slices,
    prepare_pred_gt_slices,
)

logger = get_logger(__file__)

# ======================================
#           BASE FUNCTIONS
# ======================================


def normalize_img(img_array: np.ndarray) -> np.ndarray:
    """Normalises an image to the [0, 1] range, avoiding division by zero.

    Args:
        img_array: Input image array to normalise.

    Returns:
        Normalised image array with values in [0, 1].
    """
    img_array = img_array.astype(float)
    return (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)


def validate_shapes(pred_array: np.ndarray, gt_array: np.ndarray) -> None:
    """Validates that the prediction and ground truth masks have the same shape.

    Args:
        pred_array: Predicted binary mask array.
        gt_array: Ground truth binary mask array.

    Raises:
        RuntimeError: If the two arrays have different shapes.
    """
    if pred_array.shape != gt_array.shape:
        raise RuntimeError(
            f"Incompatible shapes: pred {pred_array.shape}, GT {gt_array.shape}"
        )


# ======================================
#           SLICE EXTRACTION
# ======================================


def load_and_process_slice(
    patient: Patient,
    model: Model,
    slice_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Loads and processes the image, prediction, and ground truth for a given slice.

    Args:
        patient: Patient instance providing the ID and enhancement configuration.
        model: Model instance providing the plane and modality settings.
        slice_num: Index of the slice to load.

    Returns:
        Tuple of (img_array, pred_array, gt_array, dsc) where img_array is the
        normalised image, pred_array and gt_array are the binary masks, and dsc
        is the Dice Similarity Coefficient between them.

    Raises:
        RuntimeError: If a required file is missing for the given slice.
    """
    paths = patient_paths(patient=patient, model=model, slice_idx=slice_num)

    # Validate that all files exist
    for file_type, path in paths.items():
        if not path_exists(path):
            raise RuntimeError(
                f"Missing '{file_type}' file for slice {slice_num} of patient {patient.id}:\n{path}"
            )

    img_array, pred_array, gt_array = prepare_pred_gt_slices(
        img_path=paths["img"], pred_path=paths["pred"], gt_path=paths["gt"]
    )

    img_array = normalize_img(img_array)
    validate_shapes(pred_array, gt_array)

    dsc = DSC(pred_array, gt_array)

    return img_array, pred_array, gt_array, dsc


def select_best_slice(
    patient: Patient,
    model: Model,
) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates all available slices and selects the one with the highest DSC.

    Args:
        patient: Patient instance providing the ID and enhancement configuration.
        model: Model instance providing the plane and modality settings.

    Returns:
        Tuple of (best_slice, best_dsc, img_array, gt_array, pred_array) for the
        best-performing slice.

    Raises:
        RuntimeError: If no PNG slices are found for the patient.
    """
    slices = get_patient_slices(patient, model)

    if not slices:
        raise RuntimeError(f"No PNG slices found for patient {patient.id}.")

    best_slice = None
    best_dsc = -1.0
    best_img = best_pred = best_gt = None

    for slice_num in slices:
        img_array, pred_array, gt_array, dsc = load_and_process_slice(
            patient=patient, model=model, slice_num=slice_num
        )

        if dsc > best_dsc:
            best_dsc = dsc
            best_slice = slice_num
            best_img = img_array
            best_pred = pred_array
            best_gt = gt_array

    return best_slice, best_dsc, best_img, best_gt, best_pred


# ======================================
#          FIGURE GENERATION
# ======================================


def generate_figure(
    img_array: np.ndarray,
    gt_array: np.ndarray,
    pred_array: np.ndarray,
    output_path: Path,
    slice_num: int,
    title: str | None = None,
) -> None:
    """Generates and saves a figure overlaying TP, FP, and FN masks on the base image.

    Args:
        img_array: Normalised grayscale image array for the slice.
        gt_array: Binary ground truth mask array.
        pred_array: Binary prediction mask array.
        output_path: Path where the output PNG figure will be saved.
        slice_num: Slice index displayed in the figure label.
        title: Optional title text to display at the top of the figure.
    """
    # TP / FP / FN masks
    tp = (pred_array == 1) & (gt_array == 1)  # True positives
    fp = (pred_array == 1) & (gt_array == 0)  # False positives
    fn = (pred_array == 0) & (gt_array == 1)  # False negatives

    # Base figure
    h, w = img_array.shape
    fig_w = 6.0
    fig_h = fig_w * (h / w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Base image
    ax.imshow(img_array, cmap="gray", vmin=0, vmax=1)

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

    # Title
    if title:
        ax.text(
            0.5,
            0.98,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=38,
            fontweight="bold",
            fontname="Arial",
        )

    # Slice number
    ax.text(
        0.01,
        0.02,
        f"Slice {slice_num}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
    )

    # Legend
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

    plt.savefig(output_path, dpi=300, pad_inches=0)
    plt.close(fig)


# ======================================
#              PROCESSING
# ======================================


def visualize_best_slice(
    patient: Patient,
    model: Model,
    output_dir: Path,
    clean: bool,
) -> None:
    """Selects the slice with the best DSC and saves the corresponding figure.

    Args:
        patient: Patient instance defining the patient to visualize.
        model: Model instance providing the plane and modality settings.
        output_dir: Directory where the output figure will be saved.
        clean: If True, deletes the previous figure before generating a new one.
    """
    str_enhancement = patient.enhancement if patient.enhancement is not None else "Base"
    logger.info(
        f"🔎 Searching for best slice for patient {patient.id} ({str_enhancement}, {patient.plane})."
    )

    best_slice, best_dsc, img_array, gt_array, pred_array = select_best_slice(
        patient=patient, model=model
    )

    logger.info(f"🏅 Best slice found: {best_slice}  (DSC = {best_dsc:.3f}).")

    output_path = output_dir / f"{patient.id}_{patient.modality_str}_{best_slice}.png"

    if clean and path_exists(output_path):
        logger.info(f"♻️ Cleaning previous figure.")
        output_path.unlink(missing_ok=True)

    title = f"DSC = {best_dsc:.3f}"

    generate_figure(
        img_array=img_array,
        gt_array=gt_array,
        pred_array=pred_array,
        title=title,
        slice_num=best_slice,
        output_path=output_path,
    )

    logger.info(f"✅ Figure generated successfully.")


def visualize_specific_slice(
    patient: Patient,
    model: Model,
    slice_num: int,
    output_dir: Path,
    clean: bool,
) -> None:
    """Generates and saves the figure for a specific slice of the patient.

    Args:
        patient: Patient instance defining the patient to visualize.
        model: Model instance providing the plane and modality settings.
        slice_num: Index of the slice to visualize.
        output_dir: Directory where the output figure will be saved.
        clean: If True, deletes the previous figure before generating a new one.
    """
    str_enhancement = patient.enhancement if patient.enhancement is not None else "Base"
    logger.info(
        f"🖼️ Visualising slice {slice_num} for patient {patient.id} "
        f"({str_enhancement}, {patient.plane})."
    )

    img_array, pred_array, gt_array, dsc = load_and_process_slice(
        patient=patient, model=model, slice_num=slice_num
    )

    title = f"DSC = {dsc:.3f}"

    output_path = (
        output_dir / f"{patient.id}_{patient.modality_str}_{slice_num}{EXT_PNG}"
    )

    if clean and path_exists(output_path):
        logger.info(f"♻️ Cleaning previous figure.")
        output_path.unlink(missing_ok=True)

    generate_figure(
        img_array=img_array,
        gt_array=gt_array,
        pred_array=pred_array,
        title=title,
        slice_num=slice_num,
        output_path=output_path,
    )

    logger.info(f"✅ Figure generated successfully.")


# ======================================
#               MAIN FLOW
# ======================================


def run_flow(
    patient: Patient,
    model: Model,
    epochs: int,
    slice_num: int | None,
    clean: bool,
) -> None:
    """Executes the visualization flow for a specific or best-performing slice.

    Args:
        patient: Patient instance defining the patient to visualize.
        model: Model instance providing the plane and modality settings.
        epochs: Number of training epochs of the YOLO model.
        slice_num: Specific slice index to visualize, or None to auto-select the best.
        clean: If True, deletes the previous figure before generating a new one.

    Raises:
        ValueError: If the patient belongs to the train split when k_folds == 1.
    """
    logger.header(f"\n🖼️ Generating prediction visualization")

    root = Path.cwd()  # respects demo/ if called from a demo script
    global_config = build_config_name(model, epochs)

    patient_id = patient.id
    plane = patient.plane
    enhancement = patient.enhancement if patient.enhancement else "Base"

    # k_folds > 1 → use fold
    if model.k_folds > 1:
        patient_fold = compute_fold(patient_id, model.k_folds)

        output_dir = (
            root
            / VISUALIZATIONS_DIR
            / enhancement
            / global_config
            / f"fold{patient_fold}"
            / patient_id
            / plane
        )

    # k_folds == 1 → use split group (test/train)
    else:
        group = patient.split

        if getattr(patient, "split", None) != "test":
            raise ValueError(
                f"Patient {patient_id} belongs to 'train'. "
                "With k_folds == 1, visualizations are only allowed for 'test' patients."
            )

        output_dir = (
            root
            / VISUALIZATIONS_DIR
            / enhancement
            / global_config
            / group
            / patient_id
            / plane
        )

    create_directory(output_dir)

    # Delegate based on mode
    if slice_num is None:
        visualize_best_slice(
            patient=patient, model=model, output_dir=output_dir, clean=clean
        )
    else:
        visualize_specific_slice(
            patient=patient,
            model=model,
            slice_num=slice_num,
            output_dir=output_dir,
            clean=clean,
        )


# ======================================
#          CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the script arguments from the command line or a provided argument list.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Generate a figure to visualize the prediction and ground truth on a patient slice."
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="ID of the patient to visualize.",
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
        "--slice",
        type=int,
        metavar="<slice>",
        help="Specific slice number to visualize.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the previously generated figure.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parses arguments, builds the Model and Patient instances, and executes the visualization.

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
        slice_num=args.slice,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
