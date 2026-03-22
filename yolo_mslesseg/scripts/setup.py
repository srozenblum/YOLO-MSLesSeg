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

CLI Arguments:
    --url (str, optional)
        Direct download link to the MSLesSeg dataset ZIP file.
        Defaults to the official Figshare file.

    --clean (flag, optional)
        Clean the previously generated GT/ directory, but not the
        downloaded dataset in MSLesSeg-Dataset/.

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

import requests
from tqdm import tqdm

from yolo_mslesseg.utils.logging_config import get_logger
from yolo_mslesseg.utils.constants import (
    SPLIT_TRAIN,
    SPLIT_TEST,
    EXT_NIFTI,
    MASK_SUFFIX,
    DATASET_DIR,
    GT_DIR,
)
from yolo_mslesseg.utils.utils import create_directory, delete_directory

# Configure logger
logger = get_logger(__file__)


# ======================================
#           HELPER FUNCTIONS
# ======================================


def dataset_exists(dataset_root):
    """
    Checks whether the MSLesSeg dataset already exists
    (i.e. whether the train/ or test/ directories are present).
    """
    train_dir = dataset_root / SPLIT_TRAIN
    test_dir = dataset_root / SPLIT_TEST
    return train_dir.exists() or test_dir.exists()


def gt_exists(gt_root):
    """
    Checks whether the ground truth directory already exists
    (i.e. whether both the train/ and test/ subdirectories are present).
    """
    gt_train = gt_root / SPLIT_TRAIN
    gt_test = gt_root / SPLIT_TEST
    return gt_train.exists() and gt_test.exists()


# ======================================
#                DOWNLOAD
# ======================================


def download_stream(response, destination):
    """
    Downloads the content of an HTTP response in streaming mode and writes
    it to disk, showing a progress bar if the server reports the total size.

    Notes:
        - Assumes `response` is a valid response returning the file.
        - Does not validate the downloaded file type (done later with is_zipfile).
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


def resolve_figshare_download_url(file_id):
    """
    Resolves a direct download URL for a Figshare file.

    Context:
        - In some cases, the `ndownloader` endpoint returns 202 (Accepted)
          with HTML and 0 bytes (does not deliver the ZIP).
        - The Figshare API allows obtaining a direct URL (Location) for
          downloading the actual file.

    Returns:
        - Direct URL (string) to make a GET request to download the file.
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


def download_file(url, destination):
    """
    Downloads a file from a URL and saves it to `destination`.

    Strategy:
        1) Attempts to download directly from the provided URL.
        2) If Figshare responds with 202 + HTML + 0 bytes (common for the
           `ndownloader` endpoint), obtains a direct URL via the API and retries.

    Args:
        - url: download link (by default, Figshare `ndownloader`).
        - destination: path of the file to write.

    Raises:
        - HTTP exceptions (raise_for_status) if the download fails.
        - ValueError if the direct URL cannot be resolved via the API.
    """
    headers = {"User-Agent": "Mozilla/5.0"}

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


def extract_zip(zip_file, destination):

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
                continue

            # Strip root folder if present
            new_name = name
            if root_folder and name.startswith(root_folder):
                new_name = name[len(root_folder):]

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


def get_mask_path(patient_dir, split):
    """
    Returns the path to the patient's mask according to the split.
    - train → PX/T1/PX_T1_MASK.nii.gz
    - test  → PX/PX_MASK.nii.gz
    """
    patient_id = patient_dir.name

    if split == SPLIT_TRAIN:
        return patient_dir / "T1" / f"{patient_id}_T1{MASK_SUFFIX}{EXT_NIFTI}"
    else:  # test
        return patient_dir / f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"


def copy_mask(mask_path, gt_root, split, patient_id):
    """
    Copies the ground truth mask to the GT/ directory,
    unifying filenames to PX_MASK.nii.gz.
    """
    gt_patient_dir = gt_root / split / patient_id
    create_directory(gt_patient_dir)

    new_name = f"{patient_id}{MASK_SUFFIX}{EXT_NIFTI}"
    shutil.copy2(mask_path, gt_patient_dir / new_name)


def process_split(dataset_root, gt_root, split):
    """
    Iterates over the patients in a split (train/test), locates their masks,
    and copies them.
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


def copy_gt_volumes(dataset_root, gt_root):
    """
    Generates the GT/train/ and GT/test/ structure by copying the original
    dataset masks with unified filenames.
    """
    create_directory(gt_root)
    create_directory(gt_root / SPLIT_TRAIN)
    create_directory(gt_root / SPLIT_TEST)

    process_split(dataset_root, gt_root, SPLIT_TRAIN)
    process_split(dataset_root, gt_root, SPLIT_TEST)


# ======================================
#              PROCESSING
# ======================================


def process_download_and_extraction(dataset_root, url):
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


def process_gt_directory(dataset_root, gt_root):
    """
    Executes the GT/ directory construction process,
    copying and unifying the original dataset masks
    into the required final structure.
    """
    logger.info(f"📂 Generating ground truth directory (GT/)...")
    try:
        copy_gt_volumes(dataset_root=dataset_root, gt_root=gt_root)
        logger.info(f"🆗 GT/ directory generated successfully.")
    except:
        raise


# ======================================
#               MAIN FLOW
# ======================================


def run_flow(url, clean, verbose=False):
    """
    Executes the main setup flow.
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


def parse_args():
    """
    Parses the script arguments from the command line (CLI).
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


def main():
    """
    CLI entry point: parses arguments and executes the full flow.
    """
    args = parse_args()

    run_flow(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        clean=args.clean,
        verbose=True,
    )


def run_setup_pipeline(clean=False):
    """
    Internal pipeline entry point: executes the flow without using the CLI parser.
    """
    run_flow(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        clean=clean,
    )


if __name__ == "__main__":
    main()
