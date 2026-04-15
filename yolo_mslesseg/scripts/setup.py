"""
Script: setup.py

Description:
    Automatically downloads the MSLesSeg dataset from the official repository
    (Figshare), decompresses the ZIP archive removing any intermediate folders,
    and organises the base dataset structure in the MSLesSeg-Dataset/ directory.
    Additionally, generates the GT/ directory with the ground truth masks for
    each patient, as required for running the pipeline.

Execution modes:
    1. CLI (standalone).
    2. Internal (from `run_pipeline.py`).

CLI Usage:
    python -m yolo_mslesseg.scripts.setup --clean

Inputs:
    - URL to the MSLesSeg dataset ZIP file
      (https://springernature.figshare.com/ndownloader/files/52771814).

Outputs:
    - MSLesSeg-Dataset/ directory with the clean official dataset structure
      without intermediate folders.

    - GT/ directory with the ground truth masks:
        GT/train/PX/PX_MASK.nii.gz
        GT/test/PX/PX_MASK.nii.gz
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from yolo_mslesseg.utils.constants import (
    SPLIT_TRAIN,
    SPLIT_TEST,
    EXT_NIFTI,
    MASK_SUFFIX,
    DATASET_DIR,
    GT_DIR,
)
from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.utils import create_directory, delete_directory

# Configure logger
logger = get_logger(__file__)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def dataset_exists(dataset_root: Path) -> bool:
    """Checks whether the MSLesSeg dataset already exists.

    Args:
        dataset_root: Root directory of the MSLesSeg dataset.

    Returns:
        True if the train/ or test/ subdirectory is present, False otherwise.
    """
    train_dir = dataset_root / SPLIT_TRAIN
    test_dir = dataset_root / SPLIT_TEST
    return train_dir.exists() or test_dir.exists()


def gt_exists(gt_root: Path) -> bool:
    """Checks whether the ground truth directory already exists.

    Args:
        gt_root: Root directory of the GT structure.

    Returns:
        True if both train/ and test/ subdirectories are present, False otherwise.
    """
    gt_train = gt_root / SPLIT_TRAIN
    gt_test = gt_root / SPLIT_TEST
    return gt_train.exists() and gt_test.exists()


# ======================================
#                DOWNLOAD
# ======================================


def download_stream(response: requests.Response, destination: Path) -> None:
    """Downloads an HTTP response in streaming mode and writes it to disk.

    Displays a progress bar if the server reports the total content length.
    Does not validate the downloaded file type.

    Args:
        response: A valid streaming HTTP response object.
        destination: Path where the downloaded content will be written.
    """
    response.raise_for_status()

    total_size = int(response.headers.get("content-length") or 0)
    chunk = 1024 * 1024  # 1 MB

    with open(destination, "wb") as f, tqdm(
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        desc=f"{destination.name}",
        ncols=80,
    ) as bar:
        for block in response.iter_content(chunk_size=chunk):
            if block:
                f.write(block)
                bar.update(len(block))


def resolve_figshare_download_url(file_id: str) -> str:
    """Resolves a direct download URL for a Figshare file via the API.

    The Figshare ndownloader endpoint can return HTTP 202 with an HTML body
    and zero content bytes instead of the actual file. This function queries
    the Figshare file API endpoint to obtain the real storage URL by following
    a single redirect and extracting the Location header.

    Args:
        file_id: Figshare file identifier (last path segment of the download URL).

    Returns:
        Direct URL string pointing to the actual file download endpoint.

    Raises:
        ValueError: If the API response does not include a redirect Location header.
    """
    api_url = f"https://api.figshare.com/v2/file/download/{file_id}"

    # Expecting a redirect (302) with a Location header.
    response = requests.get(api_url, allow_redirects=False, timeout=60)

    if (
        response.status_code in (301, 302, 303, 307, 308)
        and "Location" in response.headers
    ):
        return response.headers["Location"]

    # If no redirect, this is an unexpected case.
    response.raise_for_status()
    raise ValueError(
        f"Could not resolve the direct Figshare URL (status={response.status_code})."
    )


def download_file(url: str, destination: Path) -> None:
    """Downloads a file from a URL and saves it to disk.

    First attempts a direct download. If Figshare returns 202 with HTML and
    zero bytes (common for the ndownloader endpoint), resolves a direct URL
    via the API and retries.

    Args:
        url: Download link (typically the Figshare ndownloader endpoint).
        destination: Path where the downloaded file will be written.

    Raises:
        requests.HTTPError: If the download request fails.
        ValueError: If the direct Figshare URL cannot be resolved via the API.
    """
    headers = {"User-Agent": "Mozilla/5.0"}  # Figshare rejects downloads without a browser-like User-Agent (returns 403).

    # 1) First attempt: original URL
    response = requests.get(
        url,
        stream=True,
        headers=headers,
        allow_redirects=True,
        timeout=60,
    )

    content_type = (response.headers.get("content-type") or "").lower()
    content_length = int(response.headers.get("content-length") or 0)

    # 2) Problematic case: Figshare returns 202 + HTML + 0 bytes
    if (
        response.status_code == 202
        or "text/html" in content_type
        or content_length == 0
    ):
        response.close()

        # Extract file_id from the URL (last segment).
        file_id = url.rstrip("/").split("/")[-1]
        direct_url = resolve_figshare_download_url(file_id)

        # Download from the direct URL (usually a real storage endpoint).
        response2 = requests.get(
            direct_url,
            stream=True,
            headers=headers,
            allow_redirects=True,
            timeout=120,
        )
        download_stream(response2, destination)
        return

    # Normal download (not 202/HTML/0 bytes)
    download_stream(response, destination)


# ======================================
#              DECOMPRESSION
# ======================================


def extract_zip(zip_file: Path, destination: Path) -> None:
    """Extracts a ZIP archive to a destination directory, stripping the common root folder.

    Skips the info_dataset/ folder and any entries contained within it.

    Args:
        zip_file: Path to the ZIP archive to extract.
        destination: Directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        names = zip_ref.namelist()

        # Detect common root folder (e.g. "MSLesSeg_dataset/")
        root_folder = None
        first_parts = [n.split("/")[0] for n in names if "/" in n]

        if len(set(first_parts)) == 1:
            root_folder = list(set(first_parts))[0] + "/"

        for name in names:

            # Skip info_dataset folder and all its contents
            if "info_dataset/" in name:
                continue  # Skip the info_dataset/ folder in the MSLesSeg archive (documentation, not MRI data). Update this name if the archive structure changes in a future release.

            # Strip root folder if present
            new_name = name
            if root_folder and name.startswith(root_folder):
                new_name = name[len(root_folder) :]

            # Skip empty entries
            if not new_name.strip():
                continue

            final_destination = destination / new_name

            # If it is a directory → create it
            if name.endswith("/"):
                final_destination.mkdir(parents=True, exist_ok=True)
                continue

            # If it is a file → copy it
            final_destination.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(name) as src, open(final_destination, "wb") as dst:
                shutil.copyfileobj(src, dst)


# ======================================
#       GT DIRECTORY ORGANISATION
# ======================================


def get_mask_path(patient_dir: Path, split: str) -> Path:
    """Returns the path to the patient's ground truth mask for the given split.

    Train masks are located at PX/T1/PX_T1_MASK.nii.gz; test masks at PX/PX_MASK.nii.gz.

    Args:
        patient_dir: Directory of the patient within the MSLesSeg dataset.
        split: Dataset split, either 'train' or 'test'.

    Returns:
        Path to the patient's ground truth mask file.
    """
    patient_id = patient_dir.name

    if split == SPLIT_TRAIN:
        return patient_dir / "T1" / f"{patient_id}_T1{MASK_SUFFIX}{EXT_NIFTI}"
    else:  # test
        return patient_dir / f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"


def copy_mask(mask_path: Path, gt_root: Path, split: str, patient_id: str) -> None:
    """Copies a ground truth mask to the GT/ directory with a unified filename.

    The destination filename is always PX_MASK.nii.gz regardless of the source name.

    Args:
        mask_path: Source path of the ground truth mask file.
        gt_root: Root directory of the GT/ structure.
        split: Dataset split ('train' or 'test').
        patient_id: Patient identifier string (e.g. 'P1').
    """
    gt_patient_dir = gt_root / split / patient_id
    create_directory(gt_patient_dir)

    new_name = f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"
    shutil.copy2(mask_path, gt_patient_dir / new_name)


def process_split(dataset_root: Path, gt_root: Path, split: str) -> None:
    """Copies ground truth masks for all patients in a dataset split.

    Args:
        dataset_root: Root directory of the MSLesSeg dataset.
        gt_root: Root directory of the GT/ structure.
        split: Dataset split to process ('train' or 'test').
    """
    split_root = dataset_root / split
    if not split_root.exists():
        return

    for patient_dir in sorted(split_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        mask_path = get_mask_path(patient_dir, split)
        if not mask_path.exists():
            continue

        copy_mask(mask_path, gt_root, split, patient_id)


def copy_gt_volumes(dataset_root: Path, gt_root: Path) -> None:
    """Generates the GT/train/ and GT/test/ structure from the original dataset masks.

    Args:
        dataset_root: Root directory of the MSLesSeg dataset.
        gt_root: Root directory where the GT/ structure will be created.
    """
    create_directory(gt_root)
    create_directory(gt_root / SPLIT_TRAIN)
    create_directory(gt_root / SPLIT_TEST)

    process_split(dataset_root, gt_root, SPLIT_TRAIN)
    process_split(dataset_root, gt_root, SPLIT_TEST)


# ======================================
#              PROCESSING
# ======================================


def process_download_and_extraction(dataset_root: Path, url: str) -> None:
    """Downloads and extracts the MSLesSeg dataset ZIP archive.

    Args:
        dataset_root: Directory where the dataset will be extracted.
        url: Direct download URL of the ZIP archive.

    Raises:
        ValueError: If the downloaded file is not a valid ZIP archive.
    """
    create_directory(dataset_root)
    zip_path = dataset_root / "MSLesSeg_dataset.zip"

    if zip_path.exists():
        zip_path.unlink()

    download_file(url=url, destination=zip_path)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Invalid ZIP file.")

    logger.info(f"🗂️ Extracting {zip_path}...")
    extract_zip(zip_path, dataset_root)
    zip_path.unlink()

    logger.info(f"🆗 Download and extraction completed successfully.")


def process_gt_directory(dataset_root: Path, gt_root: Path) -> None:
    """Generates the GT/ directory by copying and unifying the original dataset masks.

    Args:
        dataset_root: Root directory of the MSLesSeg dataset.
        gt_root: Root directory where the GT/ structure will be created.
    """
    logger.info(f"📂 Generating ground truth directory (GT/)...")
    copy_gt_volumes(dataset_root=dataset_root, gt_root=gt_root)
    logger.info(f"🆗 GT/ directory generated successfully.")


# ======================================
#               MAIN FLOW
# ======================================


def run_setup_flow(url: str, clean: bool, verbose: bool = False) -> None:
    """Executes the main setup flow: downloads the dataset and generates the GT/ directory.

    Args:
        url: Download URL of the MSLesSeg dataset ZIP archive.
        clean: If True, deletes the existing GT/ directory before running.
        verbose: If True, logs a header message at the start of execution.
    """
    if verbose:
        logger.header(f"📦 Downloading MSLesSeg dataset")

    dataset_root = DATASET_DIR
    gt_root = GT_DIR

    # Clean only GT (do not delete the downloaded dataset)
    if clean:
        if verbose:
            logger.info(f"♻️ Cleaning previous GT/ directory.")
        delete_directory(gt_root)

    # State after cleaning
    dataset_ok = dataset_exists(dataset_root)
    gt_ok = gt_exists(gt_root)

    # 1) Both dataset and GT exist → skip
    if dataset_ok and gt_ok:
        logger.skip("⏩ Input dataset and ground truth directory already exist.")
        return

    # 2) Dataset: download or reuse
    if dataset_ok:
        logger.skip("⏩ Input dataset already exists.")
    else:
        process_download_and_extraction(dataset_root=dataset_root, url=url)

    # 3) GT: generate or reuse
    if gt_ok:
        logger.skip("⏩ GT/ directory already exists.")
    else:
        process_gt_directory(dataset_root=dataset_root, gt_root=gt_root)


# ======================================
#           CLI AND EXECUTION
# ======================================


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the setup script.

    Returns:
        Namespace with the parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Download the MSLesSeg dataset from Figshare and organise the directory structure for the pipeline.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean only the previously generated GT/ directory, but not MSLesSeg-Dataset/.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: parses arguments and executes the full setup flow."""
    args = parse_args()

    run_setup_flow(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        clean=args.clean,
        verbose=True,
    )


def run_setup_pipeline(clean: bool = False) -> None:
    """Internal pipeline entry point: executes the setup flow programmatically.

    Args:
        clean: If True, deletes the existing GT/ directory before running.
    """
    run_setup_flow(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        clean=clean,
    )


if __name__ == "__main__":
    main()
