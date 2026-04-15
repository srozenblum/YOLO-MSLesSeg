"""
Script: train.py

Description:
    Executes the YOLO model training process under two schemes:

    - k_folds > 1 (cross-validation):
        Trains one model per fold. For each fold, builds temporary training
        and validation subsets, generates the corresponding YAML file,
        and runs the training.

    - k_folds == 1 (single training run):
        Trains a single model using train/ and test/ as fixed sets, without
        fold subdirectories inside trains/. Weights are stored directly in
        trains/<base_path>_<epochs>epochs/<plane>/weights/best.pt.

    Can be executed both as a standalone script (from the CLI) and internally
    from within the pipeline (`run_pipeline.py`).

Execution modes:
    1. CLI (standalone):
       - Arguments are read and parsed from the command line.
       - A Model instance is created.

    2. Internal (from `run_pipeline.py`):
       - A pre-built Model instance is received, along with the remaining parameters.
       - The argument parser is not used.

CLI Usage:
    python -m yolo_mslesseg.scripts.train \
        --plane "sagittal" \
        --modality "T2" \
        --num_slices 20 \
        --epochs 40 \
        --k_folds 5 \
        --fold_test 2

    python -m yolo_mslesseg.scripts.train \
        --plane "axial" \
        --modality "FLAIR" \
        --num_slices "P50" \
        --epochs 50 \
        --k_folds 1

Inputs:
    - Dataset: generated previously with `extract_dataset.py`.
        Contains the images and labels in YOLO format.

Outputs:
    - YOLO training results in the trains/ directory.
    - YAML configuration file for the experiment.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigTrain import ConfigTrain
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import WEIGHTS_FILE, WEIGHTS_SUBDIR, EXT_PNG, EXT_TXT, EXT_YAML, ENHANCEMENTS
from yolo_mslesseg.utils.utils import (
    int_or_percentile,
    delete_directory,
    load_model,
    is_ignorable_file,
    create_directory,
    path_exists,
)

logger = get_logger(__file__)
LOGGER.setLevel(logging.ERROR)

# ======================================
#           HELPER FUNCTIONS
# ======================================


def training_successful(root_dir: Path) -> bool:
    """Checks whether training produced the expected YOLO output files.

    Verifies the presence of weights/best.pt, weights/last.pt, and results.csv.

    Args:
        root_dir: Root directory of the YOLO training run.

    Returns:
        True if all expected output files exist, False otherwise.
    """
    best = root_dir / WEIGHTS_SUBDIR / WEIGHTS_FILE
    last = root_dir / WEIGHTS_SUBDIR / "last.pt"
    results = root_dir / "results.csv"
    return best.is_file() and last.is_file() and results.is_file()


def copy_directory_contents(input_dir: Path, output_dir: Path) -> None:
    """Copies the contents of a directory to another, skipping hidden or system files.

    Args:
        input_dir: Source directory whose contents will be copied.
        output_dir: Destination directory where contents will be placed.
    """
    if not path_exists(input_dir):
        return

    for f in input_dir.iterdir():
        if is_ignorable_file(f.name):
            continue

        dst = output_dir / f.name

        try:
            if f.is_dir():
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(f, dst)
        except Exception as e:
            logger.error(f"❌ Error copying '{f}' → '{dst}': {e}")
            raise


def prepare_yolo_flat(root_dir: Path) -> None:
    """Organises a flat directory into the images/ and labels/ structure required by YOLO.

    Moves all PNG files to images/ and all TXT files to labels/, then deduplicates
    modality-specific labels.

    Args:
        root_dir: Directory whose contents will be reorganised in-place.
    """
    images_output_dir = root_dir / "images"
    labels_output_dir = root_dir / "labels"
    create_directory(images_output_dir)
    create_directory(labels_output_dir)

    for img in root_dir.glob(f"*{EXT_PNG}"):
        shutil.move(str(img), str(images_output_dir / img.name))

    for lbl in root_dir.glob(f"*{EXT_TXT}"):
        shutil.move(str(lbl), str(labels_output_dir / lbl.name))


def copy_group_patients_to_flat(group_dir: Path, plane: str, output_dir: Path) -> None:
    """Copies images and labels for the given plane from a group directory to a flat directory.

    Expects group_dir/PX/<plane>/images/*.png and group_dir/PX/<plane>/labels/*.txt.
    All files are placed flat in output_dir without subdirectories.

    Args:
        group_dir: Root group directory (e.g. train/ or test/) containing patient subdirs.
        plane: Anatomical plane name used to locate the plane subdirectory.
        output_dir: Flat destination directory where all files will be copied.
    """
    for patient_dir in group_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        plane_subdir = patient_dir / plane
        if not plane_subdir.is_dir():
            continue

        images_dir = plane_subdir / "images"
        labels_dir = plane_subdir / "labels"

        if images_dir.is_dir():
            copy_directory_contents(images_dir, output_dir)
        if labels_dir.is_dir():
            copy_directory_contents(labels_dir, output_dir)


def get_existing_folds(dataset_dir: Path) -> list[int]:
    """Returns a sorted list of fold indices present in the dataset directory.

    Args:
        dataset_dir: Base dataset directory containing fold subdirectories.

    Returns:
        Sorted list of fold indices found (e.g. [1, 2, 3, 4, 5]).
    """
    folds = []
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.startswith("fold"):
            try:
                n = int(d.name.replace("fold", ""))
                folds.append(n)
            except ValueError:
                pass
    return sorted(folds)


def copy_fold_patients(fold_dir: Path, plane: str, output_dir: Path) -> None:
    """Copies images and labels for the given plane from a fold directory to a flat directory.

    Expects fold_dir/PX/<plane>/images/*.png and fold_dir/PX/<plane>/labels/*.txt.

    Args:
        fold_dir: Fold directory containing patient subdirectories.
        plane: Anatomical plane name used to locate the plane subdirectory.
        output_dir: Flat destination directory where all files will be copied.
    """
    for patient_dir in fold_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        for plane_dir in patient_dir.iterdir():
            if not (plane_dir.is_dir() and plane_dir.name.startswith(plane)):
                continue

            images_dir = plane_dir / "images"
            labels_dir = plane_dir / "labels"

            copy_directory_contents(images_dir, output_dir)
            copy_directory_contents(labels_dir, output_dir)


def is_valid_yolo_subset(root_dir: Path) -> bool:
    """Checks that a YOLO subset directory contains at least one image and one label.

    Args:
        root_dir: Root directory expected to contain images/ and labels/ subdirectories.

    Returns:
        True if both images/ and labels/ exist and are non-empty, False otherwise.
    """
    images = root_dir / "images"
    labels = root_dir / "labels"
    return (
        images.is_dir()
        and any(images.glob(f"*{EXT_PNG}"))
        and labels.is_dir()
        and any(labels.glob(f"*{EXT_TXT}"))
    )


# ======================================
#          SUBSET GENERATION
# ======================================


def create_train_subset_cv(config: ConfigTrain) -> None:
    """Builds the flat training subset for the current fold by combining all other folds.

    Args:
        config: ConfigTrain instance providing fold and directory settings.
    """
    current_fold = config.fold_test
    existing_folds = get_existing_folds(config.dataset_base_dir)
    train_output_dir = config.fold_train_dir

    fold_input_dirs = [
        config.dataset_base_dir / f"fold{i}"
        for i in existing_folds
        if i != current_fold
    ]

    if path_exists(train_output_dir):
        delete_directory(train_output_dir)
    create_directory(train_output_dir)

    for fold_dir in fold_input_dirs:
        if not path_exists(fold_dir):
            logger.warning(f"⚠️ Fold not found: {fold_dir}. Skipping.")
            continue

        copy_fold_patients(
            fold_dir=fold_dir,
            plane=config.plane,
            output_dir=train_output_dir,
        )

    prepare_yolo_flat(train_output_dir)


def create_test_subset_cv(config: ConfigTrain) -> None:
    """Creates the flat test subset from the current fold directory.

    Args:
        config: ConfigTrain instance providing fold and directory settings.
    """
    fold_input_dir = config.fold_dir
    test_output_dir = config.fold_test_dir

    if path_exists(test_output_dir):
        delete_directory(test_output_dir)
    create_directory(test_output_dir)

    copy_fold_patients(
        fold_dir=fold_input_dir,
        plane=config.plane,
        output_dir=test_output_dir,
    )

    prepare_yolo_flat(test_output_dir)


def create_single_fold_train_test(config: ConfigTrain) -> None:
    """Creates flat train_yolo/ and test_yolo/ directories for the k_folds == 1 scheme.

    Args:
        config: ConfigTrain instance providing the train/test directory settings.
    """
    train_output_dir = config.fold_train_dir
    test_output_dir = config.fold_test_dir

    if path_exists(train_output_dir):
        delete_directory(train_output_dir)
    if path_exists(test_output_dir):
        delete_directory(test_output_dir)

    create_directory(train_output_dir)
    create_directory(test_output_dir)

    copy_group_patients_to_flat(config.train_dir, config.plane, train_output_dir)

    if config.test_dir.is_dir():
        copy_group_patients_to_flat(config.test_dir, config.plane, test_output_dir)

    prepare_yolo_flat(train_output_dir)
    prepare_yolo_flat(test_output_dir)


def build_training_subsets(config: ConfigTrain) -> tuple[Path, Path]:
    """Prepares the flat train and validation directories required by YOLO.

    Args:
        config: ConfigTrain instance providing directory and fold settings.

    Returns:
        Tuple of (train_dir, val_dir), each containing images/ and labels/ subdirectories.
    """
    if config.single_fold:
        create_single_fold_train_test(config)
    else:
        create_train_subset_cv(config)
        create_test_subset_cv(config)
    return config.fold_train_dir, config.fold_test_dir


# ======================================
#           YOLO CONFIGURATION
# ======================================


def generate_yaml(config: ConfigTrain, train_dir: Path, val_dir: Path) -> dict:
    """Generates the YOLO configuration dictionary for the training run.

    Args:
        config: ConfigTrain instance providing dataset and model settings.
        train_dir: Directory containing the training images/ subdirectory.
        val_dir: Directory containing the validation images/ subdirectory.

    Returns:
        Dictionary with YOLO YAML configuration fields.
    """
    return {
        "path": str(config.dataset_base_dir.parent.resolve()),
        "train": str((train_dir / "images").resolve()),
        "val": str((val_dir / "images").resolve()),
        "names": ["lesion"],
        "nc": 1,
    }


def save_yaml(yolo_dict: dict, config: ConfigTrain) -> None:
    """Saves the YOLO configuration dictionary to the YAML file path in config.

    Args:
        yolo_dict: YOLO configuration dictionary to serialise.
        config: ConfigTrain instance providing the yaml_path.
    """
    with open(config.yaml_path, "w") as f:
        yaml.dump(yolo_dict, f, default_flow_style=False, sort_keys=False)


def copy_yaml(config: ConfigTrain) -> None:
    """Copies the YAML file to the training output directory for reference.

    Args:
        config: ConfigTrain instance providing yaml_path and train_output_dir.
    """
    yaml_dest = config.train_output_dir / f"{config.model.model_string}{EXT_YAML}"
    try:
        shutil.copy2(config.yaml_path, yaml_dest)
    except Exception as e:
        logger.warning(
            f"⚠️ Could not copy the YAML to the experiment directory: {e}."
        )


# ======================================
#              TRAINING
# ======================================


def train_fold(config: ConfigTrain) -> None:
    """Runs YOLO training using the configuration and YAML already prepared.

    Args:
        config: ConfigTrain instance providing training parameters and paths.
    """
    yolo_model = load_model(config.weights_path)

    common_kwargs = dict(
        data=config.yaml_path,
        epochs=config.epochs,
        batch=-1,  # YOLO auto-selects batch size based on available GPU memory.
        cache=True,
        val=True,
        project=config.train_output_dir,
        verbose=False,
    )

    # name="." writes outputs directly into config.train_output_dir rather than
    # creating a subdirectory. exist_ok=True allows resuming into the same directory.
    if config.single_fold:
        yolo_model.train(
            **common_kwargs,
            name=".",
            exist_ok=True,
        )
    else:
        yolo_model.train(
            **common_kwargs,
            name=f"fold{config.fold_test}",
        )


def delete_temp_files(train_dir: Path | None, val_dir: Path | None) -> None:
    """Deletes temporary flat directories used for YOLO training.

    Args:
        train_dir: Temporary training directory to delete, or None to skip.
        val_dir: Temporary validation directory to delete, or None to skip.
    """
    for d in (train_dir, val_dir):
        if d is None:
            continue
        if path_exists(d):
            delete_directory(d)


def train(config: ConfigTrain) -> None:
    """Orchestrates the full training process for a single fold.

    Prepares temporary subsets, validates them, generates and saves the YAML,
    runs training, copies the YAML, and cleans up temporary files.

    Args:
        config: ConfigTrain instance providing all training configuration.

    Raises:
        RuntimeError: If the train or validation YOLO subset is empty or invalid.
    """
    # 1. Prepare train/val subsets
    train_dir, val_dir = build_training_subsets(config)

    # 2. Validate subsets
    if not is_valid_yolo_subset(train_dir):
        raise RuntimeError(f"Train YOLO subset is empty or invalid: {train_dir}")

    if not is_valid_yolo_subset(val_dir):
        raise RuntimeError(
            f"Val YOLO subset is empty or invalid: {val_dir}. "
            "If there is no test set, create MSLesSeg-Dataset/test or disable val."
        )

    # 3. Generate YAML configuration
    yolo_dict = generate_yaml(config, train_dir=train_dir, val_dir=val_dir)
    save_yaml(yolo_dict, config)

    # 4. Run training
    train_fold(config)
    copy_yaml(config)

    # 5. Cleanup
    delete_temp_files(train_dir.parent, val_dir.parent)


# ======================================
#               MAIN FLOW
# ======================================


def run_train_flow(config: ConfigTrain, clean: bool, verbose: bool = False) -> None:
    """Executes the main training flow.

    Args:
        config: ConfigTrain instance defining the training configuration.
        clean: If True, deletes existing training outputs before starting.
        verbose: If True, logs a header message at the start of execution.
    """
    if verbose:
        if config.single_fold:
            logger.header("🧠️ Training model (single fold)")
        else:
            logger.header(f"🧠️ Training model (test = fold{config.fold_test})")

    if clean:
        config.clean_training()
        if verbose:
            logger.info("♻️ Cleaning previous training.")

    config.verify_paths()

    if config.model_path.is_file():
        if config.single_fold:
            logger.skip("⏩ Training already exists.")
        else:
            logger.skip(f"⏩ Training already exists for fold {config.fold_test}.")
        return

    train(config)

    if config.single_fold:
        root_dir = config.train_output_dir
    else:
        root_dir = config.train_output_dir / f"fold{config.fold_test}"

    if training_successful(root_dir):
        logger.info("✅ Training completed successfully.")
    else:
        logger.warning(
            "⚠️ Training did not return results. It may have been interrupted."
        )


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the training script.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.

    Returns:
        Namespace with the parsed CLI arguments.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Run YOLO model training.",
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
        metavar="[HE, CLAHE, GC, LT]",
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
        help="Number of folds for cross-validation. Defaults to 1. If k_folds == 1, a single model is trained.",
    )
    parser.add_argument(
        "--fold_test",
        type=int,
        default=None,
        metavar="<fold_test>",
        help="Fold used as the test set (1, ..., k_folds). Required only if k_folds > 1.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previously generated training results.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parses arguments and executes the training flow.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:] if None.
    """
    args = parse_args(argv)

    if args.k_folds == 1 and args.fold_test is not None:
        raise ValueError("--fold_test must not be specified when --k_folds == 1.")
    if args.k_folds > 1 and args.fold_test is None:
        raise ValueError("--fold_test must be specified when --k_folds > 1.")

    model = Model(
        plane=args.plane,
        num_slices=args.num_slices,
        modality=args.modality,
        k_folds=args.k_folds,
        enhancement=args.enhancement,
    )

    config = ConfigTrain(
        model=model,
        epochs=args.epochs,
        fold_test=args.fold_test,
    )

    run_train_flow(config=config, clean=args.clean, verbose=True)


def run_train_pipeline(model: Model, fold_test: int | None = None, epochs: int = 50, clean: bool = False) -> None:
    """Internal pipeline entry point: executes the training flow programmatically.

    Args:
        model: Model instance defining the training configuration.
        fold_test: Test fold index when using cross-validation, or None for k_folds == 1.
        epochs: Number of training epochs.
        clean: If True, deletes existing training outputs before starting.
    """
    config = ConfigTrain(
        model=model,
        epochs=epochs,
        fold_test=fold_test,
    )
    run_train_flow(config=config, clean=clean)


if __name__ == "__main__":
    main()
