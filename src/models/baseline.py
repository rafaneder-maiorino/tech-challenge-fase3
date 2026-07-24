"""TF-IDF + Logistic Regression baseline for medical abstract classification.

Trains on the processed training split, reports validation and holdout test
metrics, and persists both the fitted pipeline and the metrics report.

Run with::

    uv run python -m src.models.baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.data.prepare import CONDITION_NAMES, LABEL_COLUMN, TEXT_COLUMN

LOGGER = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODEL_PATH = Path("models/baseline.joblib")
DEFAULT_METRICS_PATH = Path("reports/baseline_metrics.json")
RANDOM_STATE = 42

TFIDF_PARAMS = {
    "min_df": 2,
    "max_features": 50_000,
    "ngram_range": (1, 2),
    "stop_words": "english",
}
LOGREG_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    """Load a processed parquet split."""
    path = processed_dir / f"{name}.parquet"
    frame = pd.read_parquet(path)
    LOGGER.info("loaded %s: shape=%s", path, frame.shape)
    return frame


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + logistic regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(**LOGREG_PARAMS)),
        ]
    )


def evaluate(pipeline: Pipeline, frame: pd.DataFrame, split_name: str) -> dict:
    """Evaluate the pipeline on a split and return a metrics dictionary."""
    y_true = frame[LABEL_COLUMN]
    y_pred = pipeline.predict(frame[TEXT_COLUMN])

    labels = sorted(CONDITION_NAMES)
    target_names = [f"{label} - {CONDITION_NAMES[label]}" for label in labels]

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    LOGGER.info(
        "[%s] accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
        split_name,
        accuracy,
        macro_f1,
        weighted_f1,
    )
    LOGGER.info(
        "[%s] classification report:\n%s",
        split_name,
        classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, digits=4
        ),
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    LOGGER.info(
        "[%s] confusion matrix (rows=true, cols=pred, labels=%s):\n%s",
        split_name,
        labels,
        matrix,
    )

    return {
        "n_samples": int(len(frame)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
    }


def train_baseline(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict:
    """Train the baseline, evaluate it and persist model plus metrics."""
    train = load_split(processed_dir, "train")
    val = load_split(processed_dir, "val")
    test = load_split(processed_dir, "test")

    LOGGER.info("fitting baseline on %d training abstracts", len(train))
    pipeline = build_pipeline()
    pipeline.fit(train[TEXT_COLUMN], train[LABEL_COLUMN])
    vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    LOGGER.info("tf-idf vocabulary size: %d", vocabulary_size)

    metrics = {
        "model": "tfidf+logistic_regression",
        "params": {
            "tfidf": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
            "logistic_regression": LOGREG_PARAMS,
        },
        "vocabulary_size": vocabulary_size,
        "condition_names": {str(k): v for k, v in CONDITION_NAMES.items()},
        "splits": {
            "validation": evaluate(pipeline, val, "validation"),
            "test": evaluate(pipeline, test, "test"),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("saved metrics to %s", metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    LOGGER.info(
        "saved pipeline to %s (%.1f MB)",
        model_path,
        model_path.stat().st_size / 1024**2,
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    train_baseline(args.processed_dir, args.model_path, args.metrics_path)


if __name__ == "__main__":
    main()
