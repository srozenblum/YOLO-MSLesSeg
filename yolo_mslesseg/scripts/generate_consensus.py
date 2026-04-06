"""
Script: generate_consensus.py

Description:
    Combines the volumetric predictions obtained from the three anatomical planes
    (axial, coronal, and sagittal) to generate a 3D consensus volume in NIfTI format.
    The consensus is computed through majority voting (threshold ≥ 2) and is
    automatically validated against the ground truth. Can be executed at patient or
    fold level.

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.
       - Model and (optionally) Patient instances are created.

    2. Internal (from `run_pipeline.py`):
       - Pre-built Model and (optionally) Patient instances are received,
         along with the remaining parameters.
       - The argument parser is not used.

CLI Usage:
    python -m yolo_mslesseg.scripts.generate_consensus \
        --epochs 50 \
        --num_slices 20 \
        --k_folds 5 \
        --fold_test 1

Inputs:
    - Predicted volumes (.nii.gz): generated previously by `reconstruct_volume.py`,
        stored in pred_vols/<enhancement>/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/<fold_test>/PX/

    - Ground truth (.nii.gz): original volumes located in GT/<patient_id>/
        used as reference for validation.

    - Classes:
        * ConfigConsensus → manages paths and global variables for consensus generation.
        * Model           → defines the modality, num_slices, enhancement, and experiment config.

Outputs:
    - 3D consensus volumes (.nii.gz) in
        pred_vols/<enhancement>/<modality>_<num_slices>slices_<k_folds>folds_<epochs>epochs/<fold_test>/PX/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from yolo_mslesseg.configs.ConfigConsensus import ConfigConsensus
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import EXT_NIFTI, MASK_SUFFIX, ENHANCEMENTS, ANATOMICAL_PLANES, StageResult
from yolo_mslesseg.utils.utils import (
    load_volume,
    save_volume,
    int_or_percentile,
    path_exists,
    list_patients,
    load_nifti_reference,
    evaluate_results,
    is_valid_reconstruction,
    log_fold_status,
)

# Configure logger
logger = get_logger(__file__)


# ======================================
#              BASE FUNCTIONS
# ======================================


def combine_volumes(axial_vol: np.ndarray, coronal_vol: np.ndarray, sagittal_vol: np.ndarray) -> np.ndarray:
    """Combines three plane volumes into a binary consensus volume using majority voting.

    Args:
        axial_vol: Predicted volume from the axial plane.
        coronal_vol: Predicted volume from the coronal plane.
        sagittal_vol: Predicted volume from the sagittal plane.

    Returns:
        Binary consensus volume as a uint8 NumPy array.
    """
    consensus = ((axial_vol + coronal_vol + sagittal_vol) >= 2).astype(np.uint8)
    return consensus


def generate_consensus(axial_path: Path, coronal_path: Path, sagittal_path: Path, output_path: Path) -> None:
    """Generates and saves a consensus NIfTI volume from three anatomical plane predictions.

    Args:
        axial_path: Path to the axial plane predicted NIfTI volume.
        coronal_path: Path to the coronal plane predicted NIfTI volume.
        sagittal_path: Path to the sagittal plane predicted NIfTI volume.
        output_path: Path where the consensus NIfTI volume will be saved.
    """
    axial_vol = load_volume(axial_path)
    coronal_vol = load_volume(coronal_path)
    sagittal_vol = load_volume(sagittal_path)
    affine = load_nifti_reference(axial_path)[1]

    # Combine volumes by majority voting to generate the consensus
    consensus = combine_volumes(
        axial_vol=axial_vol,
        coronal_vol=coronal_vol,
        sagittal_vol=sagittal_vol,
    )

    save_volume(volume=consensus, affine=affine, output_path=output_path)


# ======================================
#              PROCESSING
# ======================================


def process_patient_consensus(
    config: ConfigConsensus,
    paths_dir: dict[str, Path] | None = None,
) -> StageResult:
    """Executes the consensus generation process for an individual patient.

    Skips if the consensus volume already exists. Generates and validates the
    consensus from the three plane predicted volumes.

    Args:
        config: ConfigConsensus instance providing model and directory settings.
        paths_dir: Dictionary of paths per plane plus 'gt'. Defaults to config
            patient paths if None.

    Returns:
        StageResult.COMPLETED if the consensus was generated, StageResult.SKIPPED
        if skipped (already exists).

    Raises:
        RuntimeError: If the generated consensus volume fails validation.
    """
    # If no directories are provided → patient mode → use config directories
    if paths_dir is None:
        paths_dir = config.patient_pred_vols
        gt_vol = config.patient_gt_vol
        output_path = paths_dir["consensus"]
    else:
        gt_vol = paths_dir["gt"]
        patient_id = paths_dir["axial"].parent.name
        output_path = config.pred_vols_fold_dir / patient_id / f"{patient_id}_consensus{EXT_NIFTI}"

    # Skip if the consensus volume already exists
    if path_exists(output_path):
        return StageResult.SKIPPED

    generate_consensus(
        axial_path=paths_dir["axial"],
        coronal_path=paths_dir["coronal"],
        sagittal_path=paths_dir["sagittal"],
        output_path=output_path,
    )

    if not is_valid_reconstruction(output_path, gt_vol):
        raise RuntimeError("Consensus reconstruction is not valid.")

    return StageResult.COMPLETED


def build_paths(patient_id: str, config: ConfigConsensus) -> dict[str, Path]:
    """Builds the per-plane predicted volume and GT paths for a patient.

    Args:
        patient_id: Patient identifier string.
        config: ConfigConsensus instance providing directory settings.

    Returns:
        Dictionary mapping the three anatomical plane names and 'gt' to their
        respective Path objects. Does not include the consensus output path.
    """
    paths = {
        plane: config.pred_vols_fold_dir
        / patient_id
        / f"{patient_id}_{plane}{EXT_NIFTI}"
        for plane in ANATOMICAL_PLANES
    }
    paths["gt"] = config.gt_dir / patient_id / f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"

    return paths


def generate_consensus_for_patients(input_dir: Path, config: ConfigConsensus) -> StageResult:
    """Executes the consensus generation process for all patients in a directory.

    Args:
        input_dir: Directory containing patient subdirectories to process.
        config: ConfigConsensus instance providing model and directory settings.

    Returns:
        StageResult.COMPLETED if all patients were processed, StageResult.SKIPPED
        if all were skipped, or StageResult.PARTIAL if there was a mix.
    """
    patients = list_patients(input_dir)
    results = []

    for patient_id in patients:
        patient_paths_dir = build_paths(patient_id, config)
        try:
            consensus_result = process_patient_consensus(
                config=config,
                paths_dir=patient_paths_dir,
            )
            results.append(consensus_result)
        except Exception as e:
            logger.warning(
                f"⚠️ Error generating consensus for {patient_id}, skipping: {e}."
            )
            continue

    return evaluate_results(results)


# ======================================
#               MAIN FLOW
# ======================================


def run_consensus_flow(config: ConfigConsensus, clean: bool, verbose: bool = False) -> None:
    """Executes the main consensus generation flow.

    Args:
        config: ConfigConsensus instance defining the consensus configuration.
        clean: If True, deletes existing consensus volumes before regenerating.
        verbose: If True, logs a header message at the start of execution.
    """
    if verbose:
        if config.is_individual_patient:
            str_header = f"patient {config.patient}"
        elif config.single_fold:
            str_header = f"group {config.group}"
        else:
            str_header = f"fold {config.fold_test}"

        logger.header(f"\n🤝 Generating consensus for {str_header}.")

    # Clean if requested
    if clean:
        if verbose:
            logger.info("♻️ Cleaning previous consensus volumes.")
        config.clean()

    # Verify paths
    config.verify_paths()

    # Patient execution
    if config.is_individual_patient:
        consensus_generated = process_patient_consensus(config=config)
        if consensus_generated is StageResult.SKIPPED:
            logger.skip(f"⏩ Consensus volume already exists.")
        elif consensus_generated is StageResult.COMPLETED:
            logger.info(f"✅ Consensus generated successfully.")
        else:
            logger.warning(f"⚠️ Unknown status when generating consensus.")

    # Fold execution
    else:
        consensus_results = generate_consensus_for_patients(
            input_dir=config.pred_vols_fold_dir, config=config
        )
        if config.k_folds == 1:
            if consensus_results is StageResult.SKIPPED:
                logger.skip(f"⏩ Consensus volumes for {config.group} already exist.")
            elif consensus_results is StageResult.COMPLETED:
                logger.info(f"🆗 Consensus volumes for {config.group} generated successfully.")
            elif consensus_results is StageResult.PARTIAL:
                logger.info(
                    f"🔁 Consensus volumes for {config.group} partially updated."
                )
            else:
                logger.warning("⚠️ Unknown status when generating consensus volumes.")

        else:
            log_fold_status(
                logger=logger, result=consensus_results, fold=config.fold_test
            )


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the consensus generation script.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Generate a consensus mask through majority voting.",
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
        default=1,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 1.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--fold_test",
        type=int,
        metavar="<fold_test>",
        help="Generate consensus volumes for the indicated fold, used as the test set.",
    )
    group.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="Generate the consensus only for the specified patient.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean previously generated consensus volumes.",
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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parses arguments and executes the consensus generation flow.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.
    """
    args = parse_args(argv)

    model = Model(
        plane="consensus",
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
    config = ConfigConsensus(
        model=model,
        epochs=args.epochs,
        patient=patient,
        fold_test=args.fold_test,
    )
    run_consensus_flow(
        config=config, clean=args.clean, verbose=True
    )


def run_consensus_pipeline(
    model: Model, patient: Patient | None = None, fold_test: int | None = None, epochs: int = 50, clean: bool = False
) -> None:
    """Internal pipeline entry point: executes the consensus generation flow programmatically.

    Args:
        model: Model instance defining the consensus configuration.
        patient: Patient instance for individual execution, or None for fold mode.
        fold_test: Test fold index when using cross-validation, or None.
        epochs: Number of training epochs of the YOLO model.
        clean: If True, deletes existing consensus volumes before regenerating.
    """
    config = ConfigConsensus(
        model=model,
        epochs=epochs,
        patient=patient,
        fold_test=fold_test,
    )
    run_consensus_flow(
        config=config,
        clean=clean,
    )


if __name__ == "__main__":
    main()
