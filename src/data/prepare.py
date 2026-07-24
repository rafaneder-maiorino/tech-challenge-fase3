"""Data preparation pipeline for the medical abstracts corpus.

Downloads the corpus from the Hugging Face Hub when it is not present (see
:mod:`src.data.download`), deduplicates abstracts,
removes train/test leakage, splits the training pool into train/validation
and writes the result to ``data/processed`` as parquet files.

The corpus is a flattened multi-label export, so the same abstract can appear
on several rows with different labels. The default ``unambiguous`` strategy
discards those conflicting abstracts; ``--label-strategy majority`` restores
the previous behaviour (majority vote with a lowest-label tie-break) and is
kept so ``src.experiments.label_strategy_comparison`` stays reproducible.

Run with::

    uv run python -m src.data.prepare
    uv run python -m src.data.prepare --label-strategy majority
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.download import ensure_dataset, quiet_http_logs
from src.labels import CONDITION_NAMES, LABEL_COLUMN, TEXT_COLUMN

LOGGER = logging.getLogger(__name__)

# Re-exported so existing callers can keep importing them from here.
__all__ = [
    "CONDITION_NAMES",
    "LABEL_COLUMN",
    "TEXT_COLUMN",
    "prepare",
    "resolve_labels",
]

DEFAULT_RAW_DIR = Path("data/raw_abstracts")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_VAL_SIZE = 0.15
RANDOM_STATE = 42

LABEL_STRATEGIES = ("unambiguous", "majority")
DEFAULT_LABEL_STRATEGY = "unambiguous"


def load_raw(path: Path) -> pd.DataFrame:
    """Load a raw CSV split and drop rows with missing text or label."""
    frame = pd.read_csv(path, usecols=[LABEL_COLUMN, TEXT_COLUMN])
    LOGGER.info("loaded %s: shape=%s", path, frame.shape)

    missing = frame[[LABEL_COLUMN, TEXT_COLUMN]].isna().sum()
    if missing.any():
        LOGGER.warning("dropping rows with nulls: %s", missing.to_dict())
        frame = frame.dropna(subset=[LABEL_COLUMN, TEXT_COLUMN])

    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype("int64")
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype("string").str.strip()
    return frame.reset_index(drop=True)


def log_distribution(frame: pd.DataFrame, name: str) -> None:
    """Log the absolute and relative class distribution of a split."""
    counts = frame[LABEL_COLUMN].value_counts().sort_index()
    total = len(frame)
    LOGGER.info("class distribution [%s] (n=%d):", name, total)
    for label, count in counts.items():
        LOGGER.info(
            "  %d %-32s %5d (%5.2f%%)",
            label,
            CONDITION_NAMES.get(int(label), "unknown"),
            count,
            100 * count / total,
        )


def deduplicate(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse repeated abstracts into a single labelled row.

    The corpus is a flattened multi-label export: the same abstract may
    appear on up to four rows carrying different ``condition_label`` values.
    Conflicts are resolved deterministically by majority vote, breaking ties
    with the smallest label id.
    """
    before = len(frame)
    unique_texts = frame[TEXT_COLUMN].nunique()

    labels_per_text = frame.groupby(TEXT_COLUMN, observed=True)[LABEL_COLUMN].nunique()
    ambiguous = int((labels_per_text > 1).sum())

    votes = (
        frame.groupby([TEXT_COLUMN, LABEL_COLUMN], observed=True)
        .size()
        .rename("votes")
        .reset_index()
        .sort_values(["votes", LABEL_COLUMN], ascending=[False, True])
    )
    deduped = (
        votes.drop_duplicates(subset=TEXT_COLUMN, keep="first")[[TEXT_COLUMN, LABEL_COLUMN]]
        .sort_values(TEXT_COLUMN)
        .reset_index(drop=True)
    )

    LOGGER.info(
        "dedup [%s]: %d rows -> %d unique abstracts (%d duplicate rows removed)",
        name,
        before,
        len(deduped),
        before - len(deduped),
    )
    LOGGER.info(
        "dedup [%s]: %d/%d abstracts (%.1f%%) had conflicting labels, "
        "resolved by majority vote with lowest-label tie-break",
        name,
        ambiguous,
        unique_texts,
        100 * ambiguous / unique_texts if unique_texts else 0.0,
    )
    return deduped


def keep_unambiguous(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep only abstracts that carry exactly one distinct label.

    Discards the ambiguous abstracts instead of forcing a single label onto
    them. Shares the output contract of :func:`deduplicate`: one row per
    abstract, sorted by text, with a reset index.
    """
    before = len(frame)
    labels_per_text = frame.groupby(TEXT_COLUMN, observed=True)[LABEL_COLUMN].nunique()
    unique_texts = len(labels_per_text)
    unambiguous_texts = labels_per_text[labels_per_text == 1].index

    kept = (
        frame[frame[TEXT_COLUMN].isin(unambiguous_texts)]
        .drop_duplicates(subset=TEXT_COLUMN, keep="first")[[TEXT_COLUMN, LABEL_COLUMN]]
        .sort_values(TEXT_COLUMN)
        .reset_index(drop=True)
    )

    discarded = unique_texts - len(kept)
    LOGGER.info(
        "unambiguous [%s]: %d rows -> %d unique abstracts -> %d kept",
        name,
        before,
        unique_texts,
        len(kept),
    )
    LOGGER.info(
        "unambiguous [%s]: %d/%d abstracts (%.1f%%) discarded for carrying "
        "conflicting labels",
        name,
        discarded,
        unique_texts,
        100 * discarded / unique_texts if unique_texts else 0.0,
    )
    return kept


def resolve_labels(
    frame: pd.DataFrame, name: str, strategy: str = DEFAULT_LABEL_STRATEGY
) -> pd.DataFrame:
    """Collapse repeated abstracts using the requested label strategy.

    ``unambiguous`` drops every abstract carrying more than one distinct
    label; ``majority`` keeps them all, resolving conflicts by majority vote
    with a lowest-label tie-break. See ``docs/dataset-card.md`` for the
    experiment behind the default.
    """
    if strategy == "unambiguous":
        return keep_unambiguous(frame, name)
    if strategy == "majority":
        return deduplicate(frame, name)
    raise ValueError(
        f"unknown label strategy {strategy!r}; expected one of {LABEL_STRATEGIES}"
    )


def remove_leakage(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Drop from ``train`` every abstract that also appears in ``test``.

    The test split is the untouched holdout, so overlapping abstracts are
    removed from the training pool rather than from the evaluation set.
    """
    overlap = set(train[TEXT_COLUMN]) & set(test[TEXT_COLUMN])
    LOGGER.info(
        "leakage check: %d abstracts appear in both splits "
        "(%.1f%% of train, %.1f%% of test)",
        len(overlap),
        100 * len(overlap) / len(train) if len(train) else 0.0,
        100 * len(overlap) / len(test) if len(test) else 0.0,
    )

    if not overlap:
        return train

    cleaned = train[~train[TEXT_COLUMN].isin(overlap)].reset_index(drop=True)
    LOGGER.info(
        "leakage removal: train %d -> %d rows (test kept intact at %d rows)",
        len(train),
        len(cleaned),
        len(test),
    )
    return cleaned


def split_train_val(
    frame: pd.DataFrame,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training pool into stratified train/validation sets."""
    train, val = train_test_split(
        frame,
        test_size=val_size,
        stratify=frame[LABEL_COLUMN],
        random_state=random_state,
        shuffle=True,
    )
    LOGGER.info(
        "stratified split %.0f/%.0f: train=%d, val=%d",
        100 * (1 - val_size),
        100 * val_size,
        len(train),
        len(val),
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def save_split(frame: pd.DataFrame, path: Path) -> None:
    """Write a split to parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    LOGGER.info("saved %s: shape=%s", path, frame.shape)


def prepare(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
    label_strategy: str = DEFAULT_LABEL_STRATEGY,
) -> dict[str, pd.DataFrame]:
    """Run the full preparation pipeline and persist the processed splits."""
    LOGGER.info("=== ensuring raw data is available ===")
    ensure_dataset(raw_dir)

    LOGGER.info("=== loading raw data from %s ===", raw_dir)
    raw_train = load_raw(raw_dir / "train.csv")
    raw_test = load_raw(raw_dir / "test.csv")
    log_distribution(raw_train, "raw train")
    log_distribution(raw_test, "raw test")

    LOGGER.info("=== deduplicating (label strategy: %s) ===", label_strategy)
    train_pool = resolve_labels(raw_train, "train", label_strategy)
    test = resolve_labels(raw_test, "test", label_strategy)

    LOGGER.info("=== leakage between train and test ===")
    train_pool = remove_leakage(train_pool, test)

    LOGGER.info("=== stratified train/validation split ===")
    train, val = split_train_val(train_pool, val_size, random_state)

    LOGGER.info("=== final splits ===")
    splits = {"train": train, "val": val, "test": test}
    for name, frame in splits.items():
        log_distribution(frame, name)
        save_split(frame, output_dir / f"{name}.parquet")

    LOGGER.info(
        "summary: raw train %d rows -> %d train + %d val; raw test %d rows -> %d test",
        len(raw_train),
        len(train),
        len(val),
        len(raw_test),
        len(test),
    )
    return splits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--label-strategy",
        choices=LABEL_STRATEGIES,
        default=DEFAULT_LABEL_STRATEGY,
        help=(
            "how to handle abstracts carrying more than one distinct label: "
            "'unambiguous' discards them (default), 'majority' resolves them "
            "by majority vote with a lowest-label tie-break"
        ),
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
    prepare(
        args.raw_dir,
        args.output_dir,
        args.val_size,
        args.random_state,
        args.label_strategy,
    )


if __name__ == "__main__":
    main()
