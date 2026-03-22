"""
Script: eval.py

Description:
    Evaluates the performance of a YOLO model, either for an individual patient
    or for all patients in a fold. Can be executed both as a standalone script
    (from the CLI) and internally from within the pipeline (`run_pipeline.py`).

    Computes segmentation metrics (DSC, AUC, precision, recall) from the
    reconstructed volumes and generates JSON files with the results.

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
        Anatomical plane of the model ('axial', 'coronal', 'sagital', 'consenso').

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
        Compute metrics for all patients in the indicated fold,
        used as the test set.

    --patient_id (str, mutually exclusive with --fold_test)
        Compute metrics only for the specified patient.

    --clean (flag, optional)
        Clean the directory with binary 2D predictions before generating new ones.

CLI Usage:
    python -m yolo_mslesseg.scripts.eval \\
        --plane axial \\
        --modality FLAIR \\
        --num_slices P50 \\
        --epochs 60 \\
        --fold_test 5 \\
        --clean

Inputs:
    - Predicted volumes (.nii.gz): generated previously by `reconstruct_volume.py` in
        pred_vols/<enhancement>/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/<fold_test>/PX/.

    - Ground truth (.nii.gz): original volumes located in GT/<patient_id>/
        used as reference for evaluation.

    - Classes:
        * ConfigEval → manages paths and global variables for evaluation.
        * Model      → defines the plane, modalities, enhancement, and num_slices.

Outputs:
    - JSON with per-patient or fold-average metrics in
        results/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/foldX/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_NIFTI, MASK_SUFFIX, EXT_JSON, ENHANCEMENTS
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    load_volume,
    path_exists,
    list_patients,
    is_valid_reconstruction,
    write_json,
    log_fold_status,
    read_json,
    AUC,
    precision,
    recall,
    DSC,
)

# Configure logger
logger = get_logger(__file__)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def build_metrics_dict(gt_vol, pred_vol):
    """
    Computes a dictionary of metrics (DSC, AUC, Precision, Recall)
    from the predicted volume and the ground truth volume.
    """
    metrics = {
        "DSC": DSC(gt_vol, pred_vol),
        "AUC": AUC(gt_vol, pred_vol),
        "Precision": precision(gt_vol, pred_vol),
        "Recall": recall(gt_vol, pred_vol),
    }

    return metrics


def compute_metrics(gt_vol_path, pred_vol_path):
    """Loads the volumes, validates them, and computes the metrics."""
    # Validate the reconstruction before loading
    if not is_valid_reconstruction(pred_vol_path, gt_vol_path):
        logger.warning(f"⚠️ Invalid reconstruction: {Path(pred_vol_path).name}")
        return {}

    pred_vol = load_volume(pred_vol_path)
    gt_vol = load_volume(gt_vol_path)

    return build_metrics_dict(gt_vol, pred_vol)


def compute_averages(metrics_dict):
    """
    Computes the mean and standard deviation for all metrics
    in a dictionary.
    """
    if not metrics_dict:
        raise ValueError("The metrics dictionary is empty.")

    averages = {
        metric: {
            "media": float(np.round(np.mean(value), 3)),
            "std": float(np.round(np.std(value), 3)),
        }
        for metric, value in metrics_dict.items()
    }

    return averages


# ======================================
#              PROCESSING
# ======================================


def process_patient_eval(config, paths_dir=None, fold_mode=False):
    """Executes the metric computation for an individual patient."""
    # If no directories are provided → patient mode → use config directories
    if paths_dir is None:
        paths_dir = {
            "pred_vol": config.patient_pred_vol,
            "gt_vol": config.patient_gt_vol,
            "results_json": config.patient_results_json,
        }

    pred_vol = paths_dir["pred_vol"]
    gt_vol = paths_dir["gt_vol"]
    results_json = paths_dir["results_json"]

    # If the metrics JSON already exists
    if path_exists(results_json):
        if fold_mode:
            return read_json(results_json)
        return None  # Direct call → do not recompute

    # If it does not exist, compute new metrics
    metrics_dict = compute_metrics(
        gt_vol_path=gt_vol, pred_vol_path=pred_vol
    )
    write_json(dic=metrics_dict, json_path=results_json)

    return metrics_dict


def build_paths(patient_id, config):
    """
    Builds a dictionary of paths (pred_vol, gt_vol, results_json)
    for an individual patient.
    """
    root_pred_vols = config.pred_vols_fold_dir / patient_id
    root_gt = config.gt_dir / patient_id
    root_res = config.results_fold_dir / patient_id

    return {
        "pred_vol": root_pred_vols / f"{patient_id}_{config.plane}{EXT_NIFTI}",
        "gt_vol": root_gt / f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}",
        "results_json": root_res / f"{patient_id}_{config.plane}_results{EXT_JSON}",
    }


def compute_fold_metrics(input_dir, config):
    """
    Computes per-patient metrics and the fold average.
    """
    output_path = config.results_fold_json

    # Skip if results already exist
    if path_exists(output_path):
        return

    patients = list_patients(input_dir)
    fold_metrics = {}

    for patient_id in patients:
        patient_paths = build_paths(patient_id, config)
        patient_metrics = process_patient_eval(
            config=config, paths_dir=patient_paths, fold_mode=True
        )
        if not patient_metrics:
            logger.warning(f"⚠️ No metrics found for patient {patient_id}.")
            continue

        # Accumulate metrics for the fold
        for metric, value in patient_metrics.items():
            fold_metrics.setdefault(metric, []).append(value)

    # Compute mean and std and save to JSON
    metrics_stats = compute_averages(fold_metrics)
    write_json(dic=metrics_stats, json_path=output_path)

    return metrics_stats


# ======================================
#               MAIN FLOW
# ======================================


def run_eval_flow(config, clean, verbose=False):
    """
    Executes the main evaluation flow,
    over a full fold or an individual patient.
    """
    if verbose:
        if config.is_individual_patient:
            str_header = f"patient {config.patient}"
        elif config.single_fold:
            str_header = f"group {config.group}"
        else:
            str_header = f"fold {config.fold_test}"

        logger.header(
            f"\n📈 Computing metrics ({config.plane}) for {str_header}."
        )

    # Clean if requested
    if clean:
        if verbose:
            logger.info(f"♻️ Cleaning previous results.")
        config.clean()

    # Verify paths
    config.verify_paths()

    # Patient execution
    if config.is_individual_patient:
        patient_metrics = process_patient_eval(config=config)
        if patient_metrics is None:
            logger.skip(f"⏩ Metrics already exist.")
        elif isinstance(patient_metrics, (dict, list)):
            logger.info(f"✅ Metrics computed successfully.")
        else:
            logger.warning(f"⚠️ Unknown status when computing metrics.")

    # Fold execution
    else:
        fold_metrics = compute_fold_metrics(
            input_dir=config.pred_vols_fold_dir, config=config
        )

        if config.k_folds == 1:
            if fold_metrics is None:
                logger.skip(f"⏩ Metrics already exist.")
            elif isinstance(fold_metrics, (dict, list)):
                logger.info(f"🆗 Metrics computed successfully.")
            else:
                logger.warning("⚠️ Unknown status when computing metrics.")
        else:
            log_fold_status(
                logger=logger,
                result=fold_metrics,
                fold=config.fold_test,
            )


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
        description="Evaluate a trained model on the test set using DSC, AUC, precision, and recall.",
    )
    parser.add_argument(
        "--plane",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital", "consenso"],
        metavar="[axial, coronal, sagital, consenso]",
        help="Anatomical plane of the model.",
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
        help="Compute metrics for the indicated fold, used as the test set.",
    )
    group.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="Compute metrics only for the specified patient.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean previously computed results.",
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
    CLI entry point: parses arguments, builds Model/Patient/ConfigEval instances,
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
    config = ConfigEval(
        model=model,
        epochs=args.epochs,
        k_folds=args.k_folds,
        patient=patient,
        fold_test=args.fold_test,
        forced_plane=args.plane,
    )
    run_eval_flow(config=config, clean=args.clean, verbose=True)


def run_eval_pipeline(
    model,
    plane=None,
    patient=None,
    fold_test=None,
    epochs=50,
    k_folds=5,
    clean=False,
):
    """
    Internal pipeline entry point: receives pre-built objects and executes
    the flow without using the CLI parser.
    """
    config = ConfigEval(
        model=model,
        epochs=epochs,
        k_folds=k_folds,
        patient=patient,
        fold_test=fold_test,
        forced_plane=plane,
    )
    run_eval_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
