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

    Pipeline stages (executed sequentially):
        (0) Setup                → downloads the MSLesSeg input dataset
                                   and creates the directory structure.
        (1) Extract dataset      → extracts the YOLO dataset and annotations.
        (2) Train (optional)     → trains the YOLO model.
        (3) Generate predictions → predicts 2D segmentation masks.
        (4) Reconstruct volumes  → reconstructs predicted 3D volumes.
        (5) Eval                 → computes performance metrics.
        (6) Generate consensus   → generates consensus volumes.
        (7) Average folds        → computes global experiment metrics (full mode only).

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.

    2. Internal:
       - Not applicable. This is the top-level orchestrator.

CLI Usage:
    python -m yolo_mslesseg.run_pipeline \
        --plane axial \
        --modality FLAIR \
        --num_slices P50 \
        --enhancement HE \
        --epochs 50 \
        --full

Inputs:
    - MSLesSeg-Dataset/ directory (downloaded by setup if absent).
    - Pre-trained YOLO weights (if --train is not specified).

Outputs:
    - YOLO datasets in datasets/.
    - Trained model weights in trains/ (if --train is specified).
    - Predicted 3D volumes in pred_vols/.
    - Evaluation metrics in results/.
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

configure_logging(level=logging.INFO, log_file="pipeline.log")
logger = get_logger(__file__)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def verify_consensus_folds(model: Model, epochs: int, k_folds: int) -> tuple[list[int], list[int]]:
    """Checks which folds have complete predicted volumes across all three planes.

    Determines which folds have all three anatomical plane volumes (axial,
    coronal, and sagittal) available, which is required to generate the consensus.

    Args:
        model: Model instance providing the base path and experiment configuration.
        epochs: Number of training epochs used to locate the predicted volume directory.
        k_folds: Total number of cross-validation folds to inspect.

    Returns:
        Tuple of (valid_folds, incomplete_folds) where each element is a sorted
        list of fold indices.
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


def run_dataset(model: Model, patient: Patient | None, k_folds: int, clean: bool) -> None:
    """Executes the dataset generation stage.

    Manages the extraction of slices and annotations for an individual patient
    or for all patients in the experiment.

    Args:
        model: Model instance defining the extraction configuration.
        patient: Patient instance for individual execution, or None for full mode.
        k_folds: Number of cross-validation folds.
        clean: If True, deletes existing dataset outputs before extracting.
    """
    logger.header(f"\n🧩 Preparing YOLO dataset")
    run_dataset_pipeline(
        model=model,
        patient=patient,
        clean=clean,
    )


def run_train(model: Model, epochs: int, k_folds: int, train_flag: bool, clean: bool) -> None:
    """Executes the YOLO model training stage.

    With k_folds > 1, trains one model per fold (fold1, ..., foldK). With
    k_folds == 1, trains a single model using train/ and test/ as fixed sets.
    This stage is optional and only runs when train_flag is True.

    Args:
        model: Model instance defining the training configuration.
        epochs: Number of training epochs.
        k_folds: Number of cross-validation folds (1 for fixed split).
        train_flag: If False, the stage is skipped entirely.
        clean: If True, deletes existing training outputs before starting.
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
    for fold in range(1, k_folds + 1):
        if trained_model_exists(model, epochs, fold):
            logger.skip(f"⏩ Trained model for fold {fold} already exists.")
        else:
            logger.header(f"\n--- Fold {fold} ---")
            run_train_pipeline(
                model=model,
                fold_test=fold,
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
    clean: bool,
) -> bool:
    """Executes the consensus volume generation and its metric computation,
    for an individual patient or for all patients in the experiment.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs of the YOLO model.
        k_folds: Number of cross-validation folds (1 for a fixed split).
        patient: Patient instance for individual execution, or None for full mode.
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
            patient=patient,
            clean=clean,
        )

        logger.header("\n📈 Computing metrics (consensus)")
        run_eval_pipeline(
            model=model,
            plane="consensus",
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
            fold_test=None,
            clean=clean,
        )

        logger.header("\n📈 Computing metrics (consensus)")
        run_eval_pipeline(
            model=model,
            plane="consensus",
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
            fold_test=fold,
            clean=clean,
        )

    logger.header("\n📈 Computing metrics (consensus)")
    for fold in valid_folds:
        run_eval_pipeline(
            model=model,
            plane="consensus",
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
            plane="consensus",
            epochs=epochs,
            clean=clean,
        )


# ======================================
#             MAIN FLOW
# ======================================


def run_pipeline(
    model: Model,
    epochs: int,
    k_folds: int = 5,
    patient: Patient | None = None,
    full: bool | None = None,
    train_flag: bool = False,
    clean: bool = False,
) -> None:
    """Executes the full pipeline: setup, dataset extraction, training, prediction,
    reconstruction, evaluation, consensus, and fold averaging.

    Patient mode executes the workflow for a single patient across all applicable stages.
    Full mode executes the workflow for all folds in the dataset.

    Args:
        model: Model instance defining the plane, modalities, and configuration.
        epochs: Number of training epochs for YOLO training.
        k_folds: Number of cross-validation folds (1 for a fixed split). Defaults to 5.
        patient: Patient instance for individual execution, or None for full mode.
        full: If True, runs in full mode over all patients; None defers to patient.
        train_flag: If True, includes the training stage in the pipeline run.
        clean: If True, deletes existing intermediate files before each stage.
    """
    if model.enhancement == "GC":
        str_enhancement = f"GC (γ={model.gamma})"
    elif model.enhancement:
        str_enhancement = model.enhancement
    else:
        str_enhancement = "None"

    if patient is not None:
        logger.header(
            f"\n🚀 Starting individual pipeline · Patient: {patient.id}\n"
            f"   Plane: {model.plane} · Modality: {model.modality_str} · Enhancement: {str_enhancement}\n"
            f"   Slices: {model.num_slices} · Folds: {model.k_folds} · Epochs: {epochs}"
        )
    else:
        logger.header(
            f"\n🚀 Starting full pipeline\n"
            f"   Plane: {model.plane} · Modality: {model.modality_str} · Enhancement: {str_enhancement}\n"
            f"   Slices: {model.num_slices} · Folds: {model.k_folds} · Epochs: {epochs}"
        )

    if clean:
        logger.info("\n♻️ Cleaning previous run.")

    # --- STAGE 0 ---
    run_dataset(model, patient, k_folds, clean)

    # --- STAGE 1 ---
    run_train(model, epochs, k_folds, train_flag, clean)

    # --- STAGE 2 ---
    run_predictions(model, epochs, k_folds, patient, clean)

    # --- STAGE 3 ---
    run_reconstructions(model, epochs, k_folds, patient, clean)

    # --- STAGE 4 ---
    run_eval(model, epochs, k_folds, patient, clean)

    # --- STAGE 5 ---
    consensus_generated = run_consensus(
        model, epochs, k_folds, patient, clean
    )

    # --- STAGE 6 ---
    if full:
        run_average_folds(model, epochs, k_folds, consensus_generated, clean)

    logger.header("\n🏁 Pipeline completed successfully")


# ======================================
#          CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the pipeline script.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
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
        default=1,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 1.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Number of training epochs.",
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
        "--gamma",
        type=float,
        default=2.0,
        metavar="<gamma>",
        help="Gamma correction factor. Only applies when --enhancement is GC. Defaults to 2.0.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean all previously generated results.",
    )

    args = parser.parse_args(argv)

    if args.gamma != 2.0 and args.enhancement != "GC":
        parser.error("--gamma is only valid when --enhancement is GC.")

    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for run_pipeline.py.

    Parses the command-line arguments, builds the Model and (optionally)
    Patient instances, and delegates the full workflow execution to run_pipeline.

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
        gamma=args.gamma,
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
