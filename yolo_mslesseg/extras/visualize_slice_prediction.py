"""
Script: visualize_slice_prediction.py

Description:
    Generates a figure to visualise the model prediction on a specific slice of
    a patient and compare it with the ground truth mask, overlaying both on the
    original image. If no specific slice is indicated, the script evaluates all
    available slices, computes the DSC for each one, and visualises only the
    slice with the best performance.

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

    --slice (int, optional)
        Exact slice number to visualise.
        If not specified, the script iterates over all patient slices
        and automatically selects the one with maximum DSC.

    --clean (flag, optional)
        If a previous figure exists, deletes it before generating a new one.

CLI Usage:
    python -m yolo_mslesseg.extras.visualize_slice_prediction \\
        --patient_id P14 \\
        --plane sagital \\
        --modality FLAIR \\
        --num_slices P50 \\
        --enhancement HE \\
        --epochs 50 \\
        --k_folds 5

Inputs:
    - Image of the selected slice in PNG format.
    - Predicted and ground truth masks in PNG format.

Outputs:
    - PNG image with the generated visualisation.
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
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_PNG
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
    """
    Normalises the image to the [0, 1] range,
    avoiding division by zero.
    """
    img_array = img_array.astype(float)
    return (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)


def validate_shapes(pred_array: np.ndarray, gt_array: np.ndarray) -> None:
    """
    Validates that the prediction and ground truth masks
    have the same shape.
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
    """
    Loads and processes the data for a given slice:
        - normalised image,
        - rotated prediction mask,
        - ground truth mask,
        - DSC between prediction and ground truth.

    Returns a tuple (img_array, pred_array, gt_array, dsc).
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
    """
    Evaluates all available slices of the patient and selects
    the one with the highest DSC between prediction and ground truth.

    Returns:
        (best_slice, best_dsc, img_array, gt_array, pred_array)
        corresponding to the best-performing slice.
    """
    slices = get_patient_slices(patient, model)

    if not slices:
        raise RuntimeError(
            f"No PNG slices found for patient {patient.id}."
        )

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
    """
    Generates a figure overlaying:
        - True positives (green)
        - False positives (orange)
        - False negatives (blue)
    on the original image.
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
    """
    Automatically selects the slice with the best DSC for the patient and
    generates the corresponding figure in the output directory.
    If `clean` is True, deletes the previous figure before generating a new one.
    """
    str_enhancement = patient.enhancement if patient.enhancement is not None else "Base"
    logger.info(
        f"🔎 Searching for best slice for patient {patient.id} ({str_enhancement}, {patient.plane})."
    )

    best_slice, best_dsc, img_array, gt_array, pred_array = select_best_slice(
        patient=patient, model=model
    )

    logger.info(f"🏅 Best slice found: {best_slice}  (DSC = {best_dsc:.3f}).")

    output_path = (
        output_dir / f"{patient.id}_{patient.modality_str}_{best_slice}.png"
    )

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
    """
    Generates the figure for a specific slice of the patient.
    Computes the corresponding DSC and saves the visualisation in the
    output directory. If `clean` is True, deletes the previous figure
    before generating a new one.
    """
    img_array, pred_array, gt_array, dsc = load_and_process_slice(
        patient=patient, model=model, slice_num=slice_num
    )

    title = f"DSC = {dsc:.3f}"

    output_path = (
        output_dir / f"{patient.id}_{patient.modality_str}_{slice_num}{EXT_PNG}"
    )

    if clean and path_exists(output_path):
        logger.warning(f"♻️ Cleaning previous figure.")
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
    """
    Executes the visualisation flow for the specified slice or, if no slice
    is indicated, automatically selects the best slice for the patient.
    """
    logger.header(f"\n🖼️ Generating prediction visualisation")

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
            / "visualizaciones"
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
                "With k_folds == 1, visualisations are only allowed for 'test' patients."
            )

        output_dir = (
            root
            / "visualizaciones"
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
    """
    Parses the script arguments.
    If no argument list is provided, reads from the command line.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Generate a figure to visualise the prediction and ground truth on a patient slice."
    )
    parser.add_argument(
        "--patient_id",
        type=str,
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
        "--slice",
        type=int,
        metavar="<slice>",
        help="Specific slice number to visualise.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the previously generated figure.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point: parses arguments, builds the Model and Patient instances,
    and executes the visualisation.
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
