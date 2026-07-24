"""Compare label-resolution strategies for the ambiguous medical abstracts.

Around 20.7% of the unique training abstracts carry more than one distinct
``condition_label`` because the corpus is a flattened multi-label export.
This experiment holds every other pipeline step constant and varies only how
that ambiguity is resolved:

* ``majority``    -- majority vote with a lowest-label tie-break.
* ``unambiguous`` -- discard every abstract carrying more than one label
  (adopted as the production default on the evidence below).

Both are executed through :func:`src.data.prepare.resolve_labels`, so this
experiment exercises the production code path rather than a reimplementation.

Both strategies are applied to ``train.csv`` **and** ``test.csv`` with the
same rule, then run through the same leakage removal, the same stratified
85/15 split and the same TF-IDF + logistic regression pipeline.

Because ``unambiguous`` also shrinks the holdout, the two test sets are not
identical. The report therefore includes a third view -- ``common_test`` --
where both models are scored on exactly the same rows (the abstracts present
in both holdouts), which is the only strictly apples-to-apples comparison.

This module only reads from the production code: it imports from
``src.data.prepare`` and ``src.models.baseline`` and modifies neither.

Run with::

    uv run python -m src.experiments.label_strategy_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.data.prepare import (
    CONDITION_NAMES,
    DEFAULT_RAW_DIR,
    DEFAULT_VAL_SIZE,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEXT_COLUMN,
    load_raw,
    remove_leakage,
    resolve_labels,
    split_train_val,
)
from src.models.baseline import LOGREG_PARAMS, TFIDF_PARAMS, build_pipeline

LOGGER = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path("reports/label_strategy_comparison.json")
STRATEGIES = ("majority", "unambiguous")
LABELS = sorted(CONDITION_NAMES)


def describe_distribution(frame: pd.DataFrame) -> dict:
    """Return per-class counts and proportions for a split."""
    counts = frame[LABEL_COLUMN].value_counts()
    total = len(frame)
    return {
        str(label): {
            "condition": CONDITION_NAMES[label],
            "count": int(counts.get(label, 0)),
            "proportion": float(counts.get(label, 0) / total) if total else 0.0,
        }
        for label in LABELS
    }


def log_distribution(frame: pd.DataFrame, name: str) -> None:
    """Log the class distribution of a split."""
    total = len(frame)
    counts = frame[LABEL_COLUMN].value_counts()
    LOGGER.info("  distribution [%s] (n=%d):", name, total)
    for label in LABELS:
        count = int(counts.get(label, 0))
        LOGGER.info(
            "    %d %-32s %5d (%5.2f%%)",
            label,
            CONDITION_NAMES[label],
            count,
            100 * count / total if total else 0.0,
        )


def build_splits(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    strategy: str,
    val_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Run the shared preparation protocol under one label strategy."""
    LOGGER.info("--- preparing splits for strategy '%s' ---", strategy)
    train_pool = resolve_labels(raw_train, f"{strategy}/train", strategy)
    test = resolve_labels(raw_test, f"{strategy}/test", strategy)
    train_pool = remove_leakage(train_pool, test)
    train, val = split_train_val(train_pool, val_size, random_state)
    return {"train": train, "val": val, "test": test}


def score(pipeline: Pipeline, frame: pd.DataFrame) -> dict:
    """Score a fitted pipeline on a split."""
    y_true = frame[LABEL_COLUMN]
    y_pred = pipeline.predict(frame[TEXT_COLUMN])
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "n_samples": len(frame),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {
            str(label): float(value)
            for label, value in zip(LABELS, per_class, strict=True)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
    }


def run_strategy(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    strategy: str,
    val_size: float,
    random_state: int,
) -> tuple[dict, Pipeline, dict[str, pd.DataFrame]]:
    """Prepare data, fit the baseline and evaluate it for one strategy."""
    splits = build_splits(raw_train, raw_test, strategy, val_size, random_state)
    for name, frame in splits.items():
        log_distribution(frame, f"{strategy}/{name}")

    LOGGER.info(
        "fitting baseline for '%s' on %d abstracts", strategy, len(splits["train"])
    )
    pipeline = build_pipeline()
    pipeline.fit(splits["train"][TEXT_COLUMN], splits["train"][LABEL_COLUMN])

    result = {
        "strategy": strategy,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "distributions": {
            name: describe_distribution(frame) for name, frame in splits.items()
        },
        "validation": score(pipeline, splits["val"]),
        "test": score(pipeline, splits["test"]),
    }
    LOGGER.info(
        "[%s] validation acc=%.4f macro_f1=%.4f | test acc=%.4f macro_f1=%.4f",
        strategy,
        result["validation"]["accuracy"],
        result["validation"]["macro_f1"],
        result["test"]["accuracy"],
        result["test"]["macro_f1"],
    )
    return result, pipeline, splits


def score_common_test(
    pipelines: dict[str, Pipeline], splits: dict[str, dict[str, pd.DataFrame]]
) -> dict:
    """Score every strategy on the abstracts shared by all holdouts.

    The ``unambiguous`` strategy shrinks the test set, so the per-strategy
    ``test`` numbers are measured on different rows. This restricts every
    model to the intersection so the comparison is strictly like-for-like.
    """
    common = set.intersection(
        *(set(split["test"][TEXT_COLUMN]) for split in splits.values())
    )
    LOGGER.info("common test subset: %d abstracts", len(common))

    results = {"n_samples": len(common), "strategies": {}}
    for strategy, pipeline in pipelines.items():
        frame = splits[strategy]["test"]
        subset = frame[frame[TEXT_COLUMN].isin(common)].sort_values(TEXT_COLUMN)
        results["strategies"][strategy] = score(pipeline, subset)
    return results


def log_comparison(results: dict[str, dict], common: dict) -> None:
    """Log a side-by-side comparison table for all strategies."""
    strategies = list(results)
    label_width = 42
    width = 15

    def row(label: str, values: list[str]) -> None:
        LOGGER.info(
            "  %-*s %s", label_width, label, "".join(f"{v:>{width}}" for v in values)
        )

    LOGGER.info("=" * 76)
    LOGGER.info("LABEL STRATEGY COMPARISON")
    LOGGER.info("=" * 76)
    row("metric", strategies)
    LOGGER.info("  " + "-" * 73)

    for key, label in (
        ("n_train", "n_train"),
        ("n_val", "n_val"),
        ("n_test", "n_test"),
    ):
        row(label, [f"{results[s][key]}" for s in strategies])

    LOGGER.info("  " + "-" * 73)
    row(
        "validation accuracy",
        [f"{results[s]['validation']['accuracy']:.4f}" for s in strategies],
    )
    row(
        "validation macro-F1",
        [f"{results[s]['validation']['macro_f1']:.4f}" for s in strategies],
    )
    row("test accuracy", [f"{results[s]['test']['accuracy']:.4f}" for s in strategies])
    row("test macro-F1", [f"{results[s]['test']['macro_f1']:.4f}" for s in strategies])

    LOGGER.info("  " + "-" * 73)
    LOGGER.info("  test F1 per class:")
    for label in LABELS:
        row(
            f"    {label} - {CONDITION_NAMES[label]}",
            [
                f"{results[s]['test']['f1_per_class'][str(label)]:.4f}"
                for s in strategies
            ],
        )

    LOGGER.info("  " + "-" * 73)
    LOGGER.info("  common test subset (n=%d, identical rows):", common["n_samples"])
    row(
        "    accuracy",
        [f"{common['strategies'][s]['accuracy']:.4f}" for s in strategies],
    )
    row(
        "    macro-F1",
        [f"{common['strategies'][s]['macro_f1']:.4f}" for s in strategies],
    )
    LOGGER.info("=" * 76)

    for strategy in strategies:
        LOGGER.info(
            "test confusion matrix [%s] (rows=true, cols=pred, labels=%s):",
            strategy,
            LABELS,
        )
        matrix = results[strategy]["test"]["confusion_matrix"]["matrix"]
        for label, line in zip(LABELS, matrix, strict=True):
            LOGGER.info("  %d %s", label, line)


def compare(
    raw_dir: Path = DEFAULT_RAW_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run both strategies end to end and persist the comparison report."""
    raw_train = load_raw(raw_dir / "train.csv")
    raw_test = load_raw(raw_dir / "test.csv")

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    for strategy in STRATEGIES:
        result, pipeline, split = run_strategy(
            raw_train, raw_test, strategy, val_size, random_state
        )
        results[strategy] = result
        pipelines[strategy] = pipeline
        splits[strategy] = split

    common = score_common_test(pipelines, splits)
    log_comparison(results, common)

    report = {
        "experiment": "label_resolution_strategy",
        "protocol": {
            "raw_dir": str(raw_dir),
            "val_size": val_size,
            "random_state": random_state,
            "leakage_removal": "abstracts shared with test are dropped from train",
            "model": "tfidf+logistic_regression",
            "tfidf": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
            "logistic_regression": LOGREG_PARAMS,
        },
        "condition_names": {str(k): v for k, v in CONDITION_NAMES.items()},
        "strategies": results,
        "common_test": common,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("saved comparison report to %s", report_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    compare(args.raw_dir, args.report_path, args.val_size, args.random_state)


if __name__ == "__main__":
    main()
