"""
Script: average_folds.py

Description:
    Computes the mean and standard deviation of the metrics obtained across
    each cross-validation fold, generating a JSON file with the global
    experiment results. Can be executed both as a standalone script (from the
    CLI) and internally from within the pipeline (`run_pipeline.py`).

    This script must be executed after evaluating all folds in order to
    summarise the global performance of a model.

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.
       - A Model instance is created.

    2. Internal (from `run_pipeline.py`):
       - A pre-built Model instance is received.
       - The argument parser is not used.

CLI Arguments:
    --plane (str, required)
        Anatomical plane of the model ('axial', 'coronal', 'sagital').

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

    --clean (flag, optional)
        Clean previous global results before computing new ones.

CLI Usage:
    python -m yolo_mslesseg.scripts.average_folds \\
        --plane coronal \\
        --num_slices 40 \\
        --epochs 80 \\
        --k_folds 5

Inputs:
    - JSON files with per-fold metrics: generated previously by `eval.py` in
        results/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/foldX/foldX_<plane>_results.json.

Outputs:
    - JSON with global experiment metrics (mean and standard deviation) in
        results/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/<plane>_global_results.json
"""

import argparse
import sys

import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.constants import EXT_JSON, ENHANCEMENTS, RESULTS_SUFFIX, RESULTS_GLOBAL_PREFIX
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    path_exists,
    write_json,
    read_json,
    get_logger,
)

# Configure logger
logger = get_logger(__file__)

# ======================================
#              BASE FUNCTIONS
# ======================================


def aggregate_fold_metrics(total_dict, file):
    """Aggregates the metrics read from a fold into the accumulator dictionary."""
    metrics = read_json(file)
    for k, v in metrics.items():
        # Case 1: fold format {"media": x, "std": y}
        if isinstance(v, dict) and "media" in v:
            total_dict.setdefault(k, []).append(v["media"])
        # Case 2: patient format {"metric": value}
        elif isinstance(v, (int, float)):
            total_dict.setdefault(k, []).append(float(v))
        else:
            logger.warning(
                f"⚠️ Unexpected format for metric '{k}' in {file}: {v}"
            )


def read_fold_metrics(config):
    """
    Reads the metrics files for each fold and accumulates their values in a dictionary.
    Returns a dictionary {metric: [values_per_fold]}.
    """
    fold_metrics = {}

    for fold_dir in config.results_base_dir.iterdir():
        if fold_dir.is_dir():
            if not fold_dir.name.startswith("fold"):
                continue

            results_json = (
                fold_dir / f"{fold_dir.name}_{config.plane}{RESULTS_SUFFIX}{EXT_JSON}"
            )

            if not path_exists(results_json):
                logger.warning(f"⚠️ {results_json} not found, skipping this fold.")
                continue

            aggregate_fold_metrics(total_dict=fold_metrics, file=results_json)

    if not fold_metrics:
        logger.warning("⚠️ No valid metrics found across folds.")

    return fold_metrics


def compute_experiment_summary(fold_metrics):
    """Computes the mean and standard deviation for each metric across folds."""
    results = {}
    for metric, values in fold_metrics.items():
        results[metric] = {
            "media": float(np.round(np.mean(values), 3)),
            "std": float(np.round(np.std(values, ddof=1), 3)),
        }
    return results


def export_experiment_results(results, output_path):
    """Saves the global experiment summary in JSON format."""
    if not results:
        logger.warning("⚠️ No results to export.")
        return
    write_json(dic=results, json_path=output_path)


# ======================================
#              PROCESSING
# ======================================


def process_results(config):
    """Computes the experiment average metrics from the per-fold results."""
    # Output file path for the experiment results
    output_path = config.results_base_dir / f"{RESULTS_GLOBAL_PREFIX}{config.plane}{RESULTS_SUFFIX}{EXT_JSON}"

    # Skip if results already exist
    if path_exists(output_path):
        return

    # Read per-fold results and compute the average
    fold_metrics = read_fold_metrics(config)
    experiment_results = compute_experiment_summary(fold_metrics)

    export_experiment_results(experiment_results, output_path)
    return experiment_results


# ======================================
#               MAIN FLOW
# ======================================


def run_average_folds_flow(config, clean, verbose=False):
    """
    Executes the main fold-average metric computation flow.
    """
    if verbose:
        logger.header(f"🧮 Computing experiment fold averages.")

    if clean:
        if verbose:
            logger.info(f"♻️ Cleaning previous fold averages.")
        config.clean()

    config.verify_paths()

    experiment_metrics = process_results(config=config)

    if experiment_metrics is None:  # No results processed
        logger.skip(f"⏩ Fold averages already exist.")
    elif len(experiment_metrics) > 0:  # All results processed
        logger.info(f"🆗 Fold averages computed successfully.")
    else:
        logger.warning("⚠️ Unknown status when computing fold averages.")


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
        description="Compute global experiment results by averaging the per-fold metrics.",
    )
    parser.add_argument(
        "--plane",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Anatomical plane of the model.",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=["T1", "T2", "FLAIR"],
        default=["T1", "T2", "FLAIR"],
        metavar="",
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
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean previously generated results.",
    )

    args = parser.parse_args(argv)

    if args.k_folds == 1:
        parser.error(
            "This script can only be run with cross-validation (k_folds > 1).\n"
            "With k_folds == 1, there are no folds to average."
        )

    return args


def main(argv=None):
    """
    CLI entry point: parses arguments, builds Model/ConfigEval instances,
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

    config = ConfigEval(model=model, epochs=args.epochs, k_folds=args.k_folds)

    run_average_folds_flow(config=config, clean=args.clean, verbose=True)


def run_average_folds_pipeline(
    model, plane=None, epochs=50, k_folds=5, clean=False
):
    """
    Internal pipeline entry point: receives pre-built objects and executes
    the flow without using the CLI parser.
    """
    config = ConfigEval(
        model=model,
        epochs=epochs,
        k_folds=k_folds,
        fold_test=None,
        forced_plane=plane,
    )

    run_average_folds_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
