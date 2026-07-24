"""Fetch the raw medical abstracts corpus from the Hugging Face Hub.

The dataset is pinned to a specific repository revision and every downloaded
file is checksum-verified, so a fresh clone reproduces byte-identical raw
inputs without the corpus being versioned in Git.

Run with::

    uv run python -m src.data.download
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)

REPO_ID = "123rc/medical_text"
REPO_TYPE = "dataset"
# Pinned so the corpus cannot change under the pipeline. Bump deliberately.
REVISION = "3ad6e1681522de5e45284379f7466a2bf1284a3d"
DATASET_FILES = ("train.csv", "test.csv", "about", "README.md")

# sha256 of the pinned revision, verified on every download.
CHECKSUMS = {
    "train.csv": "ad53aebc682d6b87a5647f619a079bb446d286fdc93bf0159b812418f5758609",
    "test.csv": "1eecea73c9ecad292c55e10403bd139fab9580545d6878482997c5564d51ac05",
}

DEFAULT_RAW_DIR = Path("data/raw_abstracts")

# The hub client logs one INFO line per HTTP request, which drowns out the
# pipeline's own logs. Callers configuring logging should call this first.
NOISY_LOGGERS = ("httpx", "huggingface_hub")


def quiet_http_logs() -> None:
    """Raise the log level of the hub/HTTP client loggers to WARNING."""
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(raw_dir: Path) -> None:
    """Raise if any checksummed file is missing or does not match."""
    for filename, expected in CHECKSUMS.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"expected dataset file is missing: {path}")
        actual = sha256sum(path)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {path}: expected {expected}, got {actual}. "
                "Delete the file and re-run to download it again."
            )
        LOGGER.debug("checksum ok: %s", path)


def is_present(raw_dir: Path) -> bool:
    """Return True when every checksummed file already exists locally."""
    return all((raw_dir / filename).is_file() for filename in CHECKSUMS)


def ensure_dataset(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Download the pinned dataset revision into ``raw_dir`` if needed.

    Idempotent: an existing, checksum-valid copy is reused and no network
    call is made. Returns the directory holding the raw CSV files.
    """
    if is_present(raw_dir) and not force:
        LOGGER.info("raw dataset already present at %s, skipping download", raw_dir)
        verify_checksums(raw_dir)
        return raw_dir

    LOGGER.info(
        "downloading %s (revision %s) into %s", REPO_ID, REVISION[:12], raw_dir
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=REVISION,
        local_dir=str(raw_dir),
        allow_patterns=list(DATASET_FILES),
    )

    verify_checksums(raw_dir)
    LOGGER.info("download complete and checksums verified")
    return raw_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files exist"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    quiet_http_logs()
    args = parse_args(argv)
    ensure_dataset(args.raw_dir, args.force)


if __name__ == "__main__":
    main()
