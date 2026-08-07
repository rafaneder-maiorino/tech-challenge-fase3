"""Ingestion and training pipeline for the medical abstract classifier.

Three chained tasks that reuse the project's CLI modules as libraries:

1. ``ingest``  — download the pinned corpus if absent, verify sha256 checksums.
2. ``prepare`` — deduplicate, remove train/test leakage, write stratified splits.
3. ``train``   — fit TF-IDF + LogisticRegression, evaluate, gate on macro-F1 and
   publish the artifact.

The task callables import ``src.*`` lazily, inside the function body, so that
parsing this file stays cheap. A top-level ``import pandas`` would run on every
DAG-parse cycle in the scheduler, not just when a task actually executes.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pendulum
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

LOGGER = logging.getLogger(__name__)

LOCAL_TZ = "America/Sao_Paulo"

# Where the project tree is mounted inside the container (see the volumes in
# docker-compose.airflow.yml). Every path below is absolute on purpose: the
# src.* modules default to paths relative to the current working directory,
# which is /opt/airflow here, not the project root.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
RAW_DIR = PROJECT_ROOT / "data" / "raw_abstracts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# What the API loads at startup (src/api/main.py::DEFAULT_MODEL_PATH).
BASELINE_MODEL_PATH = MODELS_DIR / "baseline.joblib"

# Quality gate. Overridable at runtime from the UI (Admin > Variables) without
# touching this file — which is how the gate gets tested.
MIN_MACRO_F1_VARIABLE = "min_macro_f1"
# Calibrated from the observed baseline: the reference run scores 0.6707 macro-F1
# on the held-out test split. 0.62 sits roughly five points below that — wide
# enough to absorb the ordinary variation of a retrain, tight enough that a real
# regression still trips it. Raise it as the model improves; a threshold left far
# behind the actual score stops being a gate.
MIN_MACRO_F1_DEFAULT = 0.62

# How many timestamped models to keep in models/. baseline.joblib is never a
# candidate for removal: it does not match the model_*.joblib glob.
MAX_MODELS_KEPT = 5


def _restore_stock_pickler() -> None:
    """Undo dill's global patch of the standard-library pickler.

    Airflow's task runtime imports ``dill`` and calls ``dill.extend()``, which
    injects around forty of its own reducers into ``pickle._Pickler.dispatch``.
    ``joblib`` copies that table into ``NumpyPickler.dispatch`` the first time
    it is imported, so a model dumped from inside a task serialises the numpy
    scalars in the TF-IDF vocabulary through ``dill._dill._get_attr`` — and the
    resulting file then raises ``ModuleNotFoundError: No module named 'dill'``
    anywhere dill is not installed. The API is exactly such a place: it depends
    only on the serving group of pyproject.toml.

    Reverting the patch before ``joblib`` is imported keeps the artifact
    portable. Airflow itself is unaffected: XCom serialisation here is JSON
    (``enable_xcom_pickling`` defaults to false) and carries plain dicts.
    """
    import dill

    dill.extend(False)

    # Defensive: if something imported joblib before this ran, its dispatch
    # table already holds a tainted copy that extend(False) cannot reach.
    joblib_pickle = sys.modules.get("joblib.numpy_pickle")
    if joblib_pickle is not None:
        dispatch = joblib_pickle.NumpyPickler.dispatch
        for key, reducer in list(dispatch.items()):
            if getattr(reducer, "__module__", "").startswith("dill"):
                del dispatch[key]


def _prune_old_models(models_dir: Path, keep: int) -> list[str]:
    """Delete all but the ``keep`` most recent timestamped model files.

    Filenames embed a ``YYYYMMDD_HHMMSS`` stamp, so lexicographic order is
    chronological order — no ``stat()`` calls, and no surprises from mtimes
    that a copy or a volume mount might rewrite.
    """
    versioned = sorted(models_dir.glob("model_*.joblib"))
    stale = versioned[:-keep] if len(versioned) > keep else []
    for path in stale:
        path.unlink()
        LOGGER.info("pruned old model: %s", path.name)
    LOGGER.info(
        "retention: %d model(s) kept, %d removed (limit %d)",
        len(versioned) - len(stale),
        len(stale),
        keep,
    )
    return [path.name for path in stale]


def ingest(**context: Any) -> dict[str, Any]:
    """Ensure the raw corpus is present locally and checksum-valid."""
    from src.data.download import (
        CHECKSUMS,
        REPO_ID,
        REVISION,
        ensure_dataset,
        is_present,
        quiet_http_logs,
        sha256sum,
    )

    # The hub client logs one INFO line per HTTP request, which would bury the
    # pipeline's own output in the task log.
    quiet_http_logs()

    was_present = is_present(RAW_DIR)
    LOGGER.info(
        "input: repo=%s revision=%s raw_dir=%s already_present=%s",
        REPO_ID,
        REVISION[:12],
        RAW_DIR,
        was_present,
    )

    # Idempotent: reuses a valid local copy, downloads otherwise. Verifies the
    # sha256 of every checksummed file in both branches and raises on mismatch.
    ensure_dataset(RAW_DIR)

    files: dict[str, dict[str, Any]] = {}
    for name in sorted(CHECKSUMS):
        path = RAW_DIR / name
        digest = sha256sum(path)
        files[name] = {"bytes": path.stat().st_size, "sha256": digest}
        LOGGER.info(
            "output: %s bytes=%d sha256=%s (verified)",
            name,
            path.stat().st_size,
            digest,
        )

    return {
        "raw_dir": str(RAW_DIR),
        "downloaded": not was_present,
        "files": files,
    }


def prepare(**context: Any) -> dict[str, Any]:
    """Deduplicate, remove leakage and write the stratified splits."""
    from src.data.prepare import prepare as prepare_splits

    upstream = context["ti"].xcom_pull(task_ids="ingest")
    LOGGER.info(
        "input: raw_dir=%s files=%s downloaded_this_run=%s",
        upstream["raw_dir"],
        sorted(upstream["files"]),
        upstream["downloaded"],
    )

    # prepare() calls ensure_dataset() itself; because ingest already ran, that
    # inner call is a no-op checksum re-verification rather than a download.
    splits = prepare_splits(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR)

    counts: dict[str, int] = {}
    paths: dict[str, str] = {}
    for name, frame in splits.items():
        counts[name] = len(frame)
        paths[name] = str(PROCESSED_DIR / f"{name}.parquet")
        LOGGER.info(
            "output: split=%s shape=%s path=%s", name, tuple(frame.shape), paths[name]
        )
    LOGGER.info("output: total rows across splits=%d", sum(counts.values()))

    return {
        "processed_dir": str(PROCESSED_DIR),
        "counts": counts,
        "paths": paths,
    }


def train(**context: Any) -> dict[str, Any]:
    """Fit the baseline, enforce the quality gate and publish the artifact."""
    # Must run before joblib is imported (train_baseline pulls it in), or the
    # dumped model will only load where dill is installed.
    _restore_stock_pickler()

    from src.models.baseline import train_baseline

    upstream = context["ti"].xcom_pull(task_ids="prepare")
    LOGGER.info(
        "input: processed_dir=%s counts=%s",
        upstream["processed_dir"],
        upstream["counts"],
    )

    # Derived from logical_date rather than wall clock so that a retry of the
    # same DAG run overwrites its own artifact instead of leaving a new one
    # behind on every attempt.
    stamp = context["logical_date"].in_timezone(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"model_{stamp}.joblib"
    metrics_path = REPORTS_DIR / f"metrics_{stamp}.json"

    metrics = train_baseline(
        processed_dir=PROCESSED_DIR,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    validation = metrics["splits"]["validation"]
    test = metrics["splits"]["test"]
    macro_f1 = float(test["macro_f1"])
    LOGGER.info(
        "output: vocabulary_size=%d validation_macro_f1=%.4f test_macro_f1=%.4f "
        "test_accuracy=%.4f",
        metrics["vocabulary_size"],
        validation["macro_f1"],
        macro_f1,
        test["accuracy"],
    )
    LOGGER.info("output: model=%s metrics=%s", model_path, metrics_path)

    # --- quality gate -----------------------------------------------------
    # Runs BEFORE publishing. A model that fails the gate stays on disk as a
    # timestamped file for inspection, but baseline.joblib — the artifact the
    # API serves — keeps pointing at the last model that passed.
    threshold = float(
        Variable.get(MIN_MACRO_F1_VARIABLE, default_var=MIN_MACRO_F1_DEFAULT)
    )
    LOGGER.info("quality gate: test macro_f1=%.4f, minimum=%.4f", macro_f1, threshold)
    if macro_f1 < threshold:
        # AirflowFailException fails the task immediately, bypassing the retry
        # policy. The gate is deterministic — the same splits and the same
        # RANDOM_STATE produce the same macro-F1, so a retry would burn a
        # minute to reach the identical verdict. Transient failures earlier in
        # this task (reading the parquet splits, writing the artifact) still
        # retry normally, because those raise ordinary exceptions.
        raise AirflowFailException(
            f"quality gate failed: test macro_f1 {macro_f1:.4f} is below the "
            f"minimum of {threshold:.4f}. {BASELINE_MODEL_PATH.name} was left "
            f"untouched; the rejected model is at {model_path}."
        )
    LOGGER.info("quality gate: PASSED")

    # --- publish ----------------------------------------------------------
    # copy2, not symlink: a symlink in models/ would not survive a checkout on
    # Windows without developer mode, and the API just opens the path.
    shutil.copy2(model_path, BASELINE_MODEL_PATH)
    LOGGER.info(
        "published %s -> %s (%.1f MB)",
        model_path.name,
        BASELINE_MODEL_PATH,
        BASELINE_MODEL_PATH.stat().st_size / 1024**2,
    )

    removed = _prune_old_models(MODELS_DIR, MAX_MODELS_KEPT)

    return {
        "model_path": str(model_path),
        "baseline_path": str(BASELINE_MODEL_PATH),
        "metrics_path": str(metrics_path),
        "macro_f1": macro_f1,
        "accuracy": float(test["accuracy"]),
        "threshold": threshold,
        "vocabulary_size": int(metrics["vocabulary_size"]),
        "pruned_models": removed,
    }


with DAG(
    dag_id="train_pipeline",
    description="Ingestion, preparation and training of the triage baseline",
    # Manual trigger only. The pinned corpus revision does not change on its
    # own, so there is nothing for a schedule to pick up.
    #
    # In production this would become a real cadence — e.g. schedule="0 3 * * 1"
    # for Mondays at 03:00, or a Dataset-driven trigger firing when new labelled
    # abstracts land. Retraining would then be justified by data drift, which is
    # what the monitoring plan in docs/ tracks; `catchup=False` should stay
    # either way, since backfilling a training job would just retrain the same
    # model N times over the same corpus.
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz=LOCAL_TZ),
    catchup=False,
    # retries stay on for every task: ingest hits the network and prepare does
    # real I/O, so a transient failure there is worth a second attempt. The
    # quality gate is exempt without touching this policy, because
    # AirflowFailException bypasses retries on its own.
    default_args={
        "owner": "tech-challenge",
        "retries": 2,
        "retry_delay": timedelta(minutes=1),
    },
    tags=["tech-challenge", "fase-3", "training"],
    doc_md=__doc__,
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest",
        python_callable=ingest,
        doc_md="Download the pinned corpus if absent and verify its sha256.",
    )
    prepare_task = PythonOperator(
        task_id="prepare",
        python_callable=prepare,
        doc_md="Deduplicate, drop train/test leakage, write stratified splits.",
    )
    train_task = PythonOperator(
        task_id="train",
        python_callable=train,
        doc_md="Fit the baseline, enforce the macro-F1 gate, publish the model.",
    )

    ingest_task >> prepare_task >> train_task
