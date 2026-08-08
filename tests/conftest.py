"""Shared pytest fixtures for the API tests.

The tests run against a small stand-in model built on synthetic text rather
than ``models/baseline.joblib``. That keeps them fast, hermetic and runnable
in CI without a training step, and it keeps them testing the API contract
rather than model quality.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.backends import ONNX_BACKEND, SKLEARN_BACKEND
from src.labels import CONDITION_NAMES

REAL_MODEL_PATH = Path("models/baseline.joblib")

# Two documents per class so the stand-in can be fitted and can emit any of
# the five labels the API is expected to return.
SYNTHETIC_CORPUS = [
    ("malignant tumour carcinoma metastasis oncology staging", 1),
    ("neoplasm biopsy carcinoma chemotherapy tumour growth", 1),
    ("gastric ulcer colon bowel hepatic digestion endoscopy", 2),
    ("intestinal liver pancreas gastro duodenal mucosa", 2),
    ("cerebral seizure neuron epilepsy stroke neurology", 3),
    ("spinal cord brain lesion neurological deficit", 3),
    ("myocardial infarction coronary artery cardiac angioplasty", 4),
    ("hypertension aortic valve heart failure vascular", 4),
    ("inflammation general lesion syndrome pathology chronic", 5),
    ("clinical findings general condition patients follow up", 5),
]


@pytest.fixture(scope="session")
def stub_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fit a tiny classifier covering all five classes and persist it."""
    texts = [text for text, _ in SYNTHETIC_CORPUS]
    labels = [label for _, label in SYNTHETIC_CORPUS]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)

    path = tmp_path_factory.mktemp("model") / "stub.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture(scope="session")
def stub_onnx_path(stub_model_path: Path) -> Path:
    """Export the same stand-in model to ONNX, via the real exporter.

    Going through ``src.models.export_onnx`` rather than calling skl2onnx
    directly means the suite exercises the converter the pipeline actually
    ships with, including the tokenizer correction.
    """
    pytest.importorskip(
        "skl2onnx", reason="ONNX export needs the 'train' group: uv sync"
    )
    from src.models.export_onnx import convert_to_onnx

    pipeline = joblib.load(stub_model_path)
    path = stub_model_path.with_suffix(".onnx")
    path.write_bytes(convert_to_onnx(pipeline).SerializeToString())
    return path


@contextmanager
def _serving_env(backend: str, path: Path) -> Iterator[TestClient]:
    """Run a TestClient against one backend, restoring the environment after.

    Both variables are set together: leaving MODEL_PATH from a previous
    parametrisation behind would point the ONNX backend at a joblib file.
    """
    previous = {key: os.environ.get(key) for key in ("MODEL_BACKEND", "MODEL_PATH")}
    os.environ["MODEL_BACKEND"] = backend
    os.environ["MODEL_PATH"] = str(path)
    try:
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Every test taking `client` runs once per backend. The API contract is the
# same promise regardless of which engine is behind it, so it is the same
# suite that has to hold -- not a second, weaker one written for ONNX.
@pytest.fixture(params=[SKLEARN_BACKEND, ONNX_BACKEND])
def backend(request: pytest.FixtureRequest) -> str:
    """The serving backend under test."""
    return request.param


@pytest.fixture
def client(
    backend: str, stub_model_path: Path, request: pytest.FixtureRequest
) -> Iterator[TestClient]:
    """Return a TestClient serving the stand-in model on the given backend."""
    path = (
        stub_model_path
        if backend == SKLEARN_BACKEND
        else request.getfixturevalue("stub_onnx_path")
    )
    with _serving_env(backend, path) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(backend: str, tmp_path: Path) -> Iterator[TestClient]:
    """Return a TestClient pointed at a model file that does not exist.

    Parametrised too: the backends fail differently -- joblib raises
    FileNotFoundError, onnxruntime raises its own error type -- and both have
    to end in a degraded service rather than a crash on startup.
    """
    suffix = ".joblib" if backend == SKLEARN_BACKEND else ".onnx"
    with _serving_env(backend, tmp_path / f"missing{suffix}") as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_labels() -> set[int]:
    """The five labels the service is allowed to return."""
    return set(CONDITION_NAMES)
