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
        Anatomical plane of the model ('axial', 'coronal', 'sagittal').

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
    python -m yolo_mslesseg.scripts.average_folds \
        --plane coronal \
        --num_slices 40 \
        --epochs 80 \
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
from pathlib import Path

import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.constants import EXT_JSON, ENHANCEMENTS, RESULTS_SUFFIX, RESULTS_GLOBAL_PREFIX
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    path_exists,
    write_json,
    read_json,
)

# Configure logger
logger = get_logger(__file__)

# ======================================
#              BASE FUNCTIONS
# ======================================


def aggregate_fold_metrics(total_dict: dict[str, list[float]], file: Path) -> None:
    """Reads metrics from a fold JSON file and appends them to an accumulator dictionary.

    Args:
        total_dict: Accumulator dictionary mapping metric names to lists of values.
        file: Path to the fold results JSON file.
    """
    metrics = read_json(file)
    for k, v in metrics.items():
        # Case 1: fold format {"mean": x, "std": y}
        if isinstance(v, dict) and "mean" in v:
            total_dict.setdefault(k, []).append(v["mean"])
        # Case 2: patient format {"metric": value}
        elif isinstance(v, (int, float)):
            total_dict.setdefault(k, []).append(float(v))
        else:
            logger.warning(
                f"⚠️ Unexpected format for metric '{k}' in {file}: {v}"
            )


def read_fold_metrics(config: ConfigEval) -> dict[str, list[float]]:
    """Reads per-fold metrics JSON files and accumulates their values.

    Args:
        config: ConfigEval instance providing the results base directory and plane.

    Returns:
        Dictionary mapping metric names to lists of per-fold values.
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


def compute_experiment_summary(fold_metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Computes the mean and standard deviation for each metric across folds.

    Args:
        fold_metrics: Dictionary mapping metric names to lists of per-fold values.

    Returns:
        Dictionary mapping metric names to dicts with 'media' (mean) and 'std' keys.
    """
    results = {}
    for metric, values in fold_metrics.items():
        results[metric] = {
            "mean": float(np.round(np.mean(values), 3)),
            "std": float(np.round(np.std(values, ddof=1), 3)),
        }
    return results


def export_experiment_results(results: dict[str, dict[str, float]], output_path: Path) -> None:
    """Saves the global experiment summary to a JSON file.

    Args:
        results: Dictionary of experiment-level metric statistics to serialise.
        output_path: Path where the JSON file will be written.
    """
    if not results:
        logger.warning("⚠️ No results to export.")
        return
    write_json(dic=results, json_path=output_path)


# ======================================
#              PROCESSING
# ======================================


def process_results(config: ConfigEval) -> dict[str, dict[str, float]] | None:
    """Computes and saves the experiment-level average metrics from per-fold results.

    Skips computation if the global results JSON already exists.

    Args:
        config: ConfigEval instance providing directory and plane settings.

    Returns:
        Dictionary of experiment-level metric statistics, or None if already computed.
    """
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


def run_average_folds_flow(config: ConfigEval, clean: bool, verbose: bool = False) -> None:
    """Executes the main fold-average metric computation flow.

    Args:
        config: ConfigEval instance defining the experiment configuration.
        clean: If True, deletes existing global results before computing new ones.
        verbose: If True, logs a header message at the start of execution.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the fold-average computation script.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
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
        choices=["axial", "coronal", "sagittal"],
        metavar="[axial, coronal, sagittal]",
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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parses arguments and executes the fold-average computation flow.

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

    config = ConfigEval(model=model, epochs=args.epochs)

    run_average_folds_flow(config=config, clean=args.clean, verbose=True)


def run_average_folds_pipeline(
    model: Model, plane: str | None = None, epochs: int = 50, clean: bool = False
) -> None:
    """Internal pipeline entry point: executes the fold-average flow programmatically.

    Args:
        model: Model instance defining the experiment configuration.
        plane: Plane label overriding the model's plane, or None.
        epochs: Number of training epochs of the YOLO model.
        clean: If True, deletes existing global results before computing new ones.
    """
    config = ConfigEval(
        model=model,
        epochs=epochs,
        fold_test=None,
        forced_plane=plane,
    )

    run_average_folds_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
