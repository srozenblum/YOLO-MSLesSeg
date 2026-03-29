"""
Script: run_pipeline.py

Description:
    Integrates the full workflow, either for an individual patient or for a
    complete experiment. Executes all pipeline stages sequentially and in a
    controlled manner, integrating the individual modules for setup, dataset
    extraction, training, prediction, reconstruction, consensus, evaluation,
    and fold averaging. Each stage automatically detects whether results
    already exist, avoiding unnecessary recomputation. This allows restarting
    executions without losing previous progress.

    Additionally, the training stage is optional. By default, the pipeline
    does not retrain YOLO models on each execution: this avoids unnecessary
    computational cost and favours reproducibility when trained weights already
    exist. Training can be explicitly enabled via the CLI flag.

Stages:
    (0) Setup                   → downloads the MSLesSeg input dataset
                                  and creates the directory structure.
    (1) Extract dataset         → extracts the YOLO dataset and annotations.
    (2) Train (optional)        → trains the YOLO model.
    (3) Generate predictions    → predicts 2D segmentation masks.
    (4) Reconstruct volumes     → reconstructs predicted 3D volumes.
    (5) Eval                    → computes performance metrics.
    (6) Generate consensus      → generates consensus volumes.

    # If running in full mode:
        (7) Average folds       → computes global experiment metrics.

CLI Arguments:
    --plane (str, required)
        Anatomical extraction plane ('axial', 'coronal', 'sagittal').

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
        Defaults to 5.

    --epochs (int, required)
        Number of training epochs.

    --consensus_threshold (int, optional)
        Voting threshold for consensus (2 = majority, 3 = unanimity).
        Defaults to 2.

    --full (flag, mutually exclusive with --patient_id)
        Execute the full workflow.

    --patient_id (str, mutually exclusive with --full)
        Execute the workflow only for the specified patient (e.g. 'P12').

    --train (flag, optional)
        Include the training stage. Omitted by default.

    --clean (flag, optional)
        Clean previous results before generating new ones.

CLI Usage:
    python -m yolo_mslesseg.run_pipeline \
        --plane axial \
        --modality FLAIR \
        --num_slices P50 \
        --enhancement HE \
        --epochs 50 \
        --full
"""

import argparse
import logging
import sys

from yolo_mslesseg.scripts.eval import run_eval_pipeline
from yolo_mslesseg.scripts.extract_dataset import run_dataset_pipeline
from yolo_mslesseg.scripts.generate_consensus import run_consensus_pipeline
from yolo_mslesseg.scripts.generate_predictions import run_predictions_pipeline
from yolo_mslesseg.scripts.average_folds import run_average_folds_pipeline
from yolo_mslesseg.scripts.reconstruct_volume import run_reconstruction_pipeline
from yolo_mslesseg.scripts.setup import run_setup_pipeline
from yolo_mslesseg.scripts.train import run_train_pipeline
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.Patient import Patient
from yolo_mslesseg.utils.logging_config import configure_logging, get_logger
from yolo_mslesseg.utils.constants import ENHANCEMENTS, SPLIT_TEST, SPLIT_TRAIN, PRED_VOLS_DIR
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    predicted_volumes_complete,
    verify_group_volumes,
    trained_model_exists,
    compute_fold,
)

# Configure logger
configure_logging(level=logging.INFO, log_file="pipeline.log")
logger = get_logger(__file__)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def verify_consensus_folds(model: Model, epochs: int, k_folds: int) -> tuple:
    """
    Checks which folds of the model have complete predicted volumes across
    all three planes (axial, coronal, and sagittal), which is required to
    generate the consensus.
    """
    assert k_folds > 1, "verify_consensus_folds must only be called with k_folds > 1"

    valid_folds = []
    incomplete_folds = []

    for fold in range(1, k_folds + 1):
        pred_vols_fold_dir = (
            PRED_VOLS_DIR / f"{model.base_path}_{epochs}epochs" / f"fold{fold}"
        )
        if verify_group_volumes(pred_vols_fold_dir):
            valid_folds.append(fold)
        else:
            incomplete_folds.append(fold)

    return valid_folds, incomplete_folds


# ======================================
#        STAGE EXECUTION FUNCTIONS
# ======================================


def run_setup(clean: bool) -> None:
    """Downloads the official dataset and prepares the directory structure."""
    logger.header(
        f"\n📦 Downloading MSLesSeg dataset and preparing directory structure"
    )
    run_setup_pipeline(clean=clean)


def run_dataset(model: Model, patient: Patient | None, k_folds: int, clean: bool) -> None:
    """
    Executes the dataset generation stage. Manages the extraction of
    slices and annotations for an individual patient or for all patients
    in the experiment.
    """
    logger.header(f"\n🧩 Preparing YOLO dataset")
    run_dataset_pipeline(
        model=model,
        patient=patient,
        clean=clean,
    )


def run_train(model: Model, epochs: int, k_folds: int, train_flag: bool, clean: bool) -> None:
    """
    Executes the YOLO model training stage.

    - k_folds > 1:
        Trains one model per fold (fold1, ..., foldK).

    - k_folds == 1:
        Trains a single model using train/ and test/ as fixed sets.
        fold_test is not used.

    This stage is optional and is only activated when `--train` is present.
    """
    logger.header(f"\n🧠 Training model")

    # =========================
    # 1) TRAINING SKIPPED
    # =========================
    if not train_flag:
        logger.info("⏹️ Training skipped (use --train to enable it).")
        return

    # =========================
    # 2) FULL MODE (k_folds == 1)
    # =========================
    if k_folds == 1:
        if trained_model_exists(model, epochs, fold_test=None):
            logger.skip("⏩ Trained model (k_folds = 1) already exists.")
            return

        run_train_pipeline(
            model=model,
            fold_test=None,
            epochs=epochs,
            clean=clean,
        )
        return

    # =========================
    # 3) FULL MODE (k_folds > 1)
    # =========================
    for fold_test in range(1, k_folds + 1):
        if trained_model_exists(model, epochs, fold_test):
            logger.skip(f"⏩ Trained model for fold {fold_test} already exists.")
        else:
            print(f"\n--- Fold {fold_test} ---\n")
            run_train_pipeline(
                model=model,
                fold_test=fold_test,
                epochs=epochs,
                clean=clean,
            )


def run_predictions(model: Model, epochs: int, k_folds: int, patient: Patient | None, clean: bool) -> None:
    """Executes the generation of binary 2D predictions for an individual
    patient or for all patients in the experiment.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        patient: Patient instance for individual execution, or None for full mode.
        clean: If True, deletes existing predictions before generating new ones.

    Returns:
        None
    """
    logger.header(f"\n🎯 Generating predictions")

    # =========================
    # 1) PATIENT MODE
    # =========================
    if patient is not None:
        run_predictions_pipeline(
            model=model,
            epochs=epochs,
            patient=patient,
            clean=clean,
        )
        return

    # =========================
    # 2) FULL MODE (k_folds == 1)
    # =========================
    if k_folds == 1:
        run_predictions_pipeline(
            model=model,
            epochs=epochs,
            fold_test=None,
            clean=clean,
        )
        return

    # =========================
    # 3) FULL MODE (k_folds > 1)
    # =========================
    for fold in range(1, k_folds + 1):
        run_predictions_pipeline(
            model=model,
            epochs=epochs,
            fold_test=fold,
            clean=clean,
        )


def run_reconstructions(model: Model, epochs: int, k_folds: int, patient: Patient | None, clean: bool) -> None:
    """Executes 3D volume reconstruction from the 2D predictions.
    Can be used for a specific patient or for all patients in the experiment.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        patient: Patient instance for individual execution, or None for full mode.
        clean: If True, deletes existing reconstructions before generating new ones.

    Returns:
        None
    """
    # =========================
    # 1) PATIENT MODE
    # =========================
    if patient is not None:
        logger.header(f"\n🧱 Reconstructing volume ({model.plane})")
        run_reconstruction_pipeline(
            model=model,
            epochs=epochs,
            patient=patient,
            clean=clean,
        )
        return

    logger.header(f"\n🧱 Reconstructing volumes ({model.plane})")

    # =========================
    # 2) FULL MODE (k_folds == 1)
    # =========================
    if k_folds == 1:
        run_reconstruction_pipeline(
            model=model,
            epochs=epochs,
            fold_test=None,
            clean=clean,
        )
        return

    # =========================
    # 3) FULL MODE (k_folds > 1)
    # =========================
    for fold in range(1, k_folds + 1):
        run_reconstruction_pipeline(
            model=model,
            epochs=epochs,
            fold_test=fold,
            clean=clean,
        )


def run_eval(model: Model, epochs: int, k_folds: int, patient: Patient | None, clean: bool) -> None:
    """Executes the evaluation metric computation (DSC, AUC, precision, recall)
    on the reconstructed volumes, for an individual patient or for all
    patients in the experiment.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        patient: Patient instance for individual execution, or None for full mode.
        clean: If True, deletes existing results before computing new ones.

    Returns:
        None
    """
    logger.header(f"\n📈 Computing metrics ({model.plane})")

    # =========================
    # 1) PATIENT MODE
    # =========================
    if patient is not None:
        run_eval_pipeline(
            model=model,
            epochs=epochs,
            patient=patient,
            clean=clean,
        )
        return

    # =========================
    # 2) FULL MODE (k_folds == 1)
    # =========================
    if k_folds == 1:
        run_eval_pipeline(
            model=model,
            epochs=epochs,
            fold_test=None,
            clean=clean,
        )
        return

    # =========================
    # 3) FULL MODE (k_folds > 1)
    # =========================
    for fold in range(1, k_folds + 1):
        run_eval_pipeline(
            model=model,
            epochs=epochs,
            fold_test=fold,
            clean=clean,
        )


def run_consensus(
    model: Model,
    epochs: int,
    k_folds: int,
    patient: Patient | None,
    consensus_threshold: int,
    clean: bool,
) -> bool:
    """Executes the consensus volume generation and its metric computation,
    for an individual patient or for all patients in the experiment.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        patient: Patient instance for individual execution, or None for full mode.
        consensus_threshold: Voting threshold (2 for majority, 3 for unanimity).
        clean: If True, deletes existing consensus volumes before generating new ones.

    Returns:
        True if the consensus was generated, False if skipped due to missing
        predicted volumes.
    """
    # =========================
    # 1) PATIENT MODE
    # =========================
    if patient is not None:

        # Determine the directory where predicted volumes should be located
        pred_vols_root = (
            PRED_VOLS_DIR
            / f"{model.base_path}_{epochs}epochs"
            / ("test" if k_folds == 1 else f"fold{compute_fold(patient.id, k_folds)}")
            / patient.id
        ).resolve()

        # If volumes are missing → cannot generate consensus
        if not predicted_volumes_complete(pred_vols_root):
            logger.warning(f"\n⚠️ Skipping consensus: missing predicted volumes.")
            return False

        logger.header("\n🤝 Generating consensus")
        run_consensus_pipeline(
            model=model,
            epochs=epochs,
            umbral=consensus_threshold,
            patient=patient,
            clean=clean,
        )

        logger.header("\n📈 Computing metrics (consensus)")
        run_eval_pipeline(
            model=model,
            plane="consenso",
            epochs=epochs,
            patient=patient,
            clean=clean,
        )
        return True

    # =========================
    # 2) FULL MODE (k_folds == 1)
    # =========================
    if k_folds == 1:
        pred_vols_test_dir = (
            PRED_VOLS_DIR / f"{model.base_path}_{epochs}epochs" / SPLIT_TEST
        )

        # Verify that all patients have predicted volumes
        if not verify_group_volumes(pred_vols_test_dir):
            logger.warning("\n⚠️ Skipping consensus: missing predicted volumes.")
            return False

        logger.header("\n🤝 Generating consensus")
        run_consensus_pipeline(
            model=model,
            epochs=epochs,
            umbral=consensus_threshold,
            fold_test=None,
            clean=clean,
        )

        logger.header("\n📈 Computing metrics (consensus)")
        run_eval_pipeline(
            model=model,
            plane="consenso",
            epochs=epochs,
            fold_test=None,
            clean=clean,
        )
        return True

    # =========================
    # 3) FULL MODE (k_folds > 1)
    # =========================
    valid_folds, incomplete_folds = verify_consensus_folds(
        model=model,
        epochs=epochs,
        k_folds=k_folds,
    )

    consensus_complete = len(valid_folds) == k_folds

    # If any fold is missing → do not generate consensus
    if not consensus_complete:
        logger.warning(
            f"\n⚠️ Skipping consensus: missing predicted volumes in fold(s) "
            f"{', '.join(map(str, incomplete_folds))}."
        )
        return False

    logger.header("\n🤝 Generating consensus")
    for fold in valid_folds:
        run_consensus_pipeline(
            model=model,
            epochs=epochs,
            umbral=consensus_threshold,
            fold_test=fold,
            clean=clean,
        )

    logger.header("\n📈 Computing metrics (consensus)")
    for fold in valid_folds:
        run_eval_pipeline(
            model=model,
            plane="consenso",
            epochs=epochs,
            fold_test=fold,
            clean=clean,
        )

    return True


def run_average_folds(
    model: Model,
    epochs: int,
    k_folds: int,
    consensus_generated: bool,
    clean: bool,
) -> None:
    """Computes the global experiment results by averaging the results
    of each fold. If the consensus was generated, also averages its metrics.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        consensus_generated: True if the consensus stage completed successfully.
        clean: If True, deletes existing fold averages before computing new ones.

    Returns:
        None
    """
    # =========================
    # 1) NOT APPLICABLE IF k_folds == 1
    # =========================
    if k_folds == 1:
        return

    # =========================
    # 2) FOLD AVERAGE
    # =========================
    logger.header(f"\n🧮 Averaging folds ({model.plane})")
    run_average_folds_pipeline(
        model=model,
        epochs=epochs,
        clean=clean,
    )

    # =========================
    # 3) CONSENSUS AVERAGE
    # =========================
    if consensus_generated:
        logger.header(f"\n🧮 Averaging folds (consensus)")
        run_average_folds_pipeline(
            model=model,
            plane="consenso",
            epochs=epochs,
            clean=clean,
        )


# ======================================
#             MAIN FLOW
# ======================================


def run_pipeline(
    model: Model,
    epochs: int,
    consensus_threshold: int,
    k_folds: int = 5,
    patient: Patient | None = None,
    full: bool | None = None,
    train_flag: bool = False,
    clean: bool = False,
) -> None:
    """
    Executes the full workflow. Includes setup, dataset extraction, training,
    prediction, reconstruction, evaluation, consensus, and fold averaging.

    - Patient mode → executes the workflow for a single patient across all stages.
    - Full mode    → executes the workflow for all folds in the dataset across all stages.
    """
    if patient is not None:
        logger.header(
            f"\n🚀 Starting individual pipeline for {patient.id} "
            f"(model = {model.model_string}, epochs = {epochs})"
        )
    else:
        logger.header(
            f"\n🚀 Starting full pipeline "
            f"(model = {model.model_string}, epochs = {epochs})"
        )

    if clean:
        logger.info("\n♻️ Cleaning previous run.")

    # --- STAGE 0 ---
    run_setup(clean)

    # --- STAGE 1 ---
    run_dataset(model, patient, k_folds, clean)

    # --- STAGE 2 ---
    run_train(model, epochs, k_folds, train_flag, clean)

    # --- STAGE 3 ---
    run_predictions(model, epochs, k_folds, patient, clean)

    # --- STAGE 4 ---
    run_reconstructions(model, epochs, k_folds, patient, clean)

    # --- STAGE 5 ---
    run_eval(model, epochs, k_folds, patient, clean)

    # --- STAGE 6 ---
    consensus_generated = run_consensus(
        model, epochs, k_folds, patient, consensus_threshold, clean
    )

    # --- STAGE 7 ---
    if full:
        run_average_folds(model, epochs, k_folds, consensus_generated, clean)

    logger.header("\n🏁 Pipeline completed successfully")


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
        description="Run the full YOLO-MSLesSeg workflow.",
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
        default=5,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--consensus_threshold",
        type=int,
        default=2,
        choices=[2, 3],
        metavar="<consensus_threshold>",
        help="Voting threshold for consensus generation (2 or 3). Defaults to 2.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--full",
        action="store_true",
        help="Execute the full workflow over all patients in the dataset.",
    )
    group.add_argument(
        "--patient_id",
        type=str,
        metavar="<patient_id>",
        help="Execute the workflow only for the specified patient.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Include the training stage. Omitted by default.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean all previously generated results.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point for `run_pipeline.py`.

    Parses the command-line arguments, builds the `Model` and (optionally)
    `Patient` instances, and delegates the full workflow execution to
    `run_pipeline`.
    """
    args = parse_args(argv)

    model = Model(
        plane=args.plane,
        num_slices=args.num_slices,
        modality=args.modality,
        k_folds=args.k_folds,
        enhancement=args.enhancement,
    )

    # Ensure the dataset exists before instantiating Patient
    run_setup_pipeline(clean=args.clean)

    patient = (
        Patient(
            id=args.patient_id,
            plane=model.plane,
            modality=model.modality,
            enhancement=model.enhancement,
        )
        if args.patient_id is not None
        else None
    )

    if patient is not None and args.k_folds == 1 and patient.split == SPLIT_TRAIN:
        raise ValueError(
            f"Patient {patient.id} belongs to the train split. "
            "With k_folds=1, only test-split patients can be processed individually."
        )

    try:
        run_pipeline(
            model=model,
            epochs=args.epochs,
            consensus_threshold=args.consensus_threshold,
            k_folds=args.k_folds,
            patient=patient,
            full=args.full,
            train_flag=args.train,
            clean=args.clean,
        )
    except Exception as e:
        logger.error(f"❌ Critical pipeline error: {e}", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
