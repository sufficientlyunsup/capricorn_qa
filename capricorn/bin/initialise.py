"""Download CAPRICORN's base model and optional training data.

Only the base model is downloaded by default. Training data is unnecessary
unless you want to refit the model; request it with --data.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from capricorn.bin.common import DATASETS, HUGGINGFACE_REPO_ID, MODEL_DIR, MODEL_REVISION

logger = logging.getLogger(__name__)

def download_model(force: bool = False) -> Path:
    """Download the pinned base model and return its local directory.

    The destination is models/qwen3-1.7b relative to the working directory.
    If it already exists and force is False, skip the download without any
    network requests or checks for completeness. Set force=True to download
    again, including when a previous download left an incomplete directory.
    """
    destination = MODEL_DIR.resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Model destination is not a directory: {destination}")
    if destination.exists() and not force:
        logger.info("Model directory already exists; skipping: %s", destination)
        return destination

    logger.info("Downloading %s @ %s -> %s", HUGGINGFACE_REPO_ID, MODEL_REVISION, destination)
    snapshot_download(
        repo_id=HUGGINGFACE_REPO_ID,
        repo_type="model",
        revision=MODEL_REVISION,
        local_dir=destination,
        force_download=force,
    )
    logger.info("Ready: %s", destination)
    return destination


def download_data(force: bool = False) -> None:
    """Download the configured NLI and QA datasets into local data.

    Each destination is skipped when it already exists unless force is True.
    Files remain in their native Hugging Face format and can be loaded later
    with the datasets library or a Parquet reader.
    """
    for name, repo_id, relative_destination, allow_patterns in DATASETS:
        destination = Path(relative_destination).resolve()
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(f"Dataset destination is not a directory: {destination}")
        if destination.exists() and not force:
            logger.info("%s directory already exists; skipping: %s", name, destination)
            continue
        logger.info("Downloading %s (%s) -> %s", name, repo_id, destination)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            local_dir=destination,
            allow_patterns=allow_patterns,
            force_download=force,
        )
        logger.info("Ready: %s", destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Training data is only needed when refitting CAPRICORN.",
    )
    parser.add_argument(
        "--data", action="store_true",
        help="Also download the configured NLI and QA datasets.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Download selected assets again, even if they already exist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    download_model(force=args.force)
    if args.data:
        download_data(force=args.force)


if __name__ == "__main__":
    main()
