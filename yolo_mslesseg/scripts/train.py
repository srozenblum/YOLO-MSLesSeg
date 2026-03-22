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

CLI Arguments:
   --plane (str, required)
        Anatomical plane of the model ('axial', 'coronal', 'sagital').

   --modality (list[str], optional)
        MRI modality or modalities ('T1', 'T2', 'FLAIR'). Defaults to all.

   --num_slices (int_or_percentile, required)
        Number of slices to extract (integer value or percentile, e.g. 50 or 'P75').

   --enhancement (str, optional)
        Image enhancement algorithm ('HE', 'CLAHE', 'GC', 'LT'). Defaults to None.

   --epochs (int, required)
        Number of training epochs.

   --k_folds (int, optional)
        Number of folds for cross-validation. Defaults to 5.
        If k_folds == 1, a single model is trained using train/ and test/.

   --fold_test (int, optional)
        Fold used as the test set (1, ..., k_folds).
        Required only if k_folds > 1.

   --clean (flag, optional)
        Clean previous training results before starting a new run.

CLI Usage:
    python -m yolo_mslesseg.scripts.train \\
        --plane "sagital" \\
        --modality "T2" \\
        --num_slices 20 \\
        --epochs 40 \\
        --k_folds 5 \\
        --fold_test 2

    python -m yolo_mslesseg.scripts.train \\
        --plane "axial" \\
        --modality "FLAIR" \\
        --num_slices "P50" \\
        --epochs 50 \\
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

import yaml
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigTrain import ConfigTrain
from yolo_mslesseg.utils.Model import Model
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import WEIGHTS_FILE, EXT_PNG, EXT_TXT, EXT_YAML, ENHANCEMENTS
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


def training_successful(root_dir):
    """
    Checks whether training was successful by verifying the essential
    YOLO training output files:
        - weights/best.pt   → best weights obtained
        - weights/last.pt   → last weights obtained
        - results.csv       → training metrics summary
    """
    best = root_dir / "weights" / WEIGHTS_FILE
    last = root_dir / "weights" / "last.pt"
    results = root_dir / "results.csv"
    return best.is_file() and last.is_file() and results.is_file()


def copy_directory_contents(input_dir, output_dir):
    """
    Copies the contents of input_dir to output_dir,
    skipping hidden or system files.
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


def duplicate_modality_labels(images_dir, labels_dir):
    """
    For each image PX_<modality>_<slice>.png, creates a label
    PX_<modality>_<slice>.txt by copying the content of PX_<slice>.txt,
    which is then deleted.
    """
    labels_base = set()

    for img in images_dir.glob(f"*{EXT_PNG}"):
        parts = img.stem.split("_")
        if len(parts) != 3:
            continue

        patient_id, mod, slice_num = parts
        label_base = labels_dir / f"{patient_id}_{slice_num}{EXT_TXT}"
        label_dest = labels_dir / f"{patient_id}_{mod}_{slice_num}{EXT_TXT}"

        if label_base.exists():
            if not label_dest.exists():
                shutil.copy2(label_base, label_dest)
            labels_base.add(label_base)

    for lb in labels_base:
        try:
            lb.unlink()
        except Exception as e:
            logger.warning(f"⚠️ Could not delete {lb}: {e}.")


def prepare_yolo_flat(root_dir):
    """
    Organises the contents of a flat directory into the format required by YOLO.
    Moves all images (.png) to 'images/' and all labels (.txt) to 'labels/'.
    """
    images_output_dir = root_dir / "images"
    labels_output_dir = root_dir / "labels"
    create_directory(images_output_dir)
    create_directory(labels_output_dir)

    for img in root_dir.glob(f"*{EXT_PNG}"):
        shutil.move(str(img), str(images_output_dir / img.name))

    for lbl in root_dir.glob(f"*{EXT_TXT}"):
        shutil.move(str(lbl), str(labels_output_dir / lbl.name))

    duplicate_modality_labels(images_output_dir, labels_output_dir)


def copy_group_patients_to_flat(group_dir, plane, output_dir):
    """
    Copies all images and labels for the given plane from a group directory
    (train/ or test/) to a flat output directory.

    Expected input structure:
        group_dir/PX/<plane>/images/*.png
        group_dir/PX/<plane>/labels/*.txt

    Output structure:
        output_dir/  (all .png and .txt files together)
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


def get_existing_folds(dataset_dir):
    """Returns the list of existing folds in the dataset."""
    folds = []
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.startswith("fold"):
            try:
                n = int(d.name.replace("fold", ""))
                folds.append(n)
            except ValueError:
                pass
    return sorted(folds)


def copy_fold_patients(fold_dir, plane, output_dir):
    """
    Copies images and labels for the given plane from a fold to a flat directory.

    Expected input structure:
        fold_dir/PX/<plane>/images/*.png
        fold_dir/PX/<plane>/labels/*.txt
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


def is_valid_yolo_subset(root_dir):
    """
    Checks that images and labels exist in the temporary training directories.
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


def create_train_subset_cv(config):
    """Builds the training subset for the current fold by combining all other folds."""
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


def create_test_subset_cv(config):
    """Creates the test subset from the current fold."""
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


def create_single_fold_train_test(config):
    """
    Creates flat train_yolo/ and test_yolo/ directories for k_folds == 1,
    from train/<plane>/ and test/<plane>/.
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


def build_training_subsets(config):
    """
    Prepares temporary flat directories for YOLO and returns (train_dir, val_dir).
    Both returned paths are roots containing images/ and labels/.
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


def generate_yaml(config, train_dir, val_dir):
    """Generates a YOLO configuration dictionary."""
    return {
        "path": str(config.dataset_base_dir.parent.resolve()),
        "train": str((train_dir / "images").resolve()),
        "val": str((val_dir / "images").resolve()),
        "names": ["lesion"],
        "nc": 1,
    }


def save_yaml(yolo_dict, config):
    """Saves the YOLO configuration dictionary to the YAML file."""
    with open(config.yaml_path, "w") as f:
        yaml.dump(yolo_dict, f, default_flow_style=False, sort_keys=False)


def copy_yaml(config):
    """Copies the YAML file to the training output directory."""
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


def train_fold(config):
    """Runs the YOLO training (without preparing subsets or generating YAML)."""
    model_yolo = load_model(config.weights_path)

    common_kwargs = dict(
        data=config.yaml_path,
        epochs=config.epochs,
        batch=-1,
        cache=True,
        val=True,
        project=config.train_output_dir,
        verbose=False,
    )

    if config.single_fold:
        model_yolo.train(
            **common_kwargs,
            name=".",
            exist_ok=True,
        )
    else:
        model_yolo.train(
            **common_kwargs,
            name=f"fold{config.fold_test}",
        )


def delete_temp_files(train_dir, val_dir):
    """
    Deletes the temporary directories containing images/ and labels/.
    Expected paths of the form .../train_yolo/<plane> and .../test_yolo/<plane>.
    """
    for d in (train_dir, val_dir):
        if d is None:
            continue
        if path_exists(d):
            delete_directory(d)


def train(config):
    """
    Orchestrates the training:
    1) prepares temporary subsets
    2) validates that they are non-empty
    3) generates and saves the YAML
    4) trains the model
    5) copies the YAML
    6) cleans up temporary files
    """
    train_dir, val_dir = build_training_subsets(config)

    if not is_valid_yolo_subset(train_dir):
        raise RuntimeError(f"Train YOLO subset is empty or invalid: {train_dir}")

    if not is_valid_yolo_subset(val_dir):
        raise RuntimeError(
            f"Val YOLO subset is empty or invalid: {val_dir}. "
            "If there is no test set, create MSLesSeg-Dataset/test or disable val."
        )

    yolo_dict = generate_yaml(config, train_dir=train_dir, val_dir=val_dir)
    save_yaml(yolo_dict, config)

    train_fold(config)
    copy_yaml(config)
    delete_temp_files(train_dir.parent, val_dir.parent)


# ======================================
#               MAIN FLOW
# ======================================


def run_train_flow(config, clean, verbose=False):
    """Executes the main training flow."""
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


def parse_args(argv=None):
    """
    Parses the script arguments.
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
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
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
        default=5,
        metavar="<k_folds>",
        help="Number of folds for cross-validation. Defaults to 5. If k_folds == 1, a single model is trained.",
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


def main(argv=None):
    """
    CLI entry point: parses arguments, builds Model/ConfigTrain instances,
    and executes the full flow.
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


def run_train_pipeline(model, fold_test=None, epochs=50, clean=False):
    """
    Internal pipeline entry point: receives pre-built objects and executes
    the flow without using the CLI parser.
    """
    config = ConfigTrain(
        model=model,
        epochs=epochs,
        fold_test=fold_test,
    )
    run_train_flow(config=config, clean=clean)


if __name__ == "__main__":
    main()
