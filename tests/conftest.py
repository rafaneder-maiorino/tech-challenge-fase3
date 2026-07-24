"""Shared pytest fixtures for the API tests.

The tests run against a small stand-in model built on synthetic text rather
than ``models/baseline.joblib``. That keeps them fast, hermetic and runnable
in CI without a training step, and it keeps them testing the API contract
rather than model quality.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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


@pytest.fixture
def client(stub_model_path: Path) -> Iterator[TestClient]:
    """Return a TestClient whose app has the stand-in model loaded."""
    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(stub_model_path)
    try:
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client
    finally:
        if previous is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = previous


@pytest.fixture
def client_without_model(tmp_path: Path) -> Iterator[TestClient]:
    """Return a TestClient pointed at a model file that does not exist."""
    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(tmp_path / "missing.joblib")
    try:
        from src.api.main import app

        with TestClient(app) as test_client:
            yield test_client
    finally:
        if previous is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = previous


@pytest.fixture(scope="session")
def valid_labels() -> set[int]:
    """The five labels the service is allowed to return."""
    return set(CONDITION_NAMES)
