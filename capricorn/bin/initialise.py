"""Download a CAPRICORN checkpoint or the assets needed for training.

The trained CAPRICORN checkpoint is downloaded by default. The Qwen backbone
and training data are only downloaded when --train is requested.
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from capricorn.bin.common import (
    CAPRICORN_REPO_ID,
    CHECKPOINT_PATH,
    DATASETS,
    HUGGINGFACE_REPO_ID,
    MODEL_DIR,
    MODEL_REVISION,
    QWEN_TOKENIZER_FILES,
)

logger = logging.getLogger(__name__)


def download_checkpoint(force: bool = False) -> Path:
    """Download the published CAPRICORN checkpoint."""
    destination = CHECKPOINT_PATH.resolve()
    if destination.exists() and not destination.is_file():
        raise NotADirectoryError(f"Checkpoint destination is not a file: {destination}")
    if destination.exists() and not force:
        logger.info("Checkpoint already exists; skipping: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s/capricorn-qa.pt -> %s", CAPRICORN_REPO_ID, destination)
    downloaded = hf_hub_download(
        repo_id=CAPRICORN_REPO_ID,
        repo_type="model",
        filename="capricorn-qa.pt",
        local_dir=destination.parent,
        force_download=force,
    )
    downloaded_path = Path(downloaded).resolve()
    if downloaded_path != destination:
        downloaded_path.replace(destination)
    logger.info("Ready: %s", destination)
    return destination


def download_tokenizer(force: bool = False) -> Path:
    """Download only the Qwen config and tokenizer needed for inference."""
    destination = MODEL_DIR.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not force and all(
        (destination / filename).is_file() for filename in QWEN_TOKENIZER_FILES
    ):
        logger.info("Qwen config/tokenizer already exists; skipping: %s", destination)
        return destination
    logger.info("Downloading Qwen config/tokenizer -> %s", destination)
    snapshot_download(
        repo_id=HUGGINGFACE_REPO_ID,
        repo_type="model",
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=QWEN_TOKENIZER_FILES,
        force_download=force,
    )
    return destination


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
        epilog="Use --train only when refitting CAPRICORN.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Download the Qwen backbone and configured NLI/QA datasets.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Download selected assets again, even if they already exist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.train:
        download_model(force=args.force)
        download_data(force=args.force)
    else:
        download_checkpoint(force=args.force)
        download_tokenizer(force=args.force)


if __name__ == "__main__":
    main()
