"""Contract tests for the prediction API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import MAX_TEXT_CHARS
from src.labels import CONDITION_NAMES

SAMPLE_TEXTS = [
    "Malignant carcinoma of the breast treated with adjuvant chemotherapy.",
    "Endoscopic findings in patients with chronic gastric ulcer disease.",
    "Cerebral infarction and seizure activity following carotid surgery.",
    "Coronary artery bypass grafting outcomes in hypertensive patients.",
    "General inflammatory response observed during long term follow up.",
]

REAL_MODEL_PATH = Path("models/baseline.joblib")
REAL_ONNX_PATH = Path("models/model.onnx")


def test_health_returns_200(client: TestClient) -> None:
    """/health answers 200 and reports the model as loaded."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_reports_degraded_without_model(
    client_without_model: TestClient,
) -> None:
    """A missing model file degrades /health instead of killing startup."""
    response = client_without_model.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_predict_without_model_returns_503(client_without_model: TestClient) -> None:
    """/predict refuses to answer when the model never loaded."""
    response = client_without_model.post("/predict", json={"text": "any abstract"})

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_predict_returns_one_of_the_five_classes(
    client: TestClient, text: str, valid_labels: set[int]
) -> None:
    """Valid text yields a label within the five known classes."""
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in valid_labels
    assert body["label_name"] == CONDITION_NAMES[body["label"]]


def test_predict_response_has_expected_fields(client: TestClient) -> None:
    """The response carries exactly the documented fields and types."""
    response = client.post("/predict", json={"text": SAMPLE_TEXTS[0]})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "label_name", "confidence", "latency_ms"}
    assert isinstance(body["label"], int)
    assert isinstance(body["label_name"], str)
    assert isinstance(body["confidence"], float)
    assert isinstance(body["latency_ms"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0.0


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"text": ""}, "empty string"),
        ({"text": "   "}, "whitespace only"),
        ({"text": "\n\t  \n"}, "newlines and tabs only"),
        ({}, "missing field"),
        ({"text": 42}, "wrong type"),
    ],
)
def test_predict_rejects_invalid_text_with_422(
    client: TestClient, payload: dict, reason: str
) -> None:
    """Invalid input is rejected with 422 and a readable message."""
    response = client.post("/predict", json=payload)

    assert response.status_code == 422, reason
    detail = response.json()["detail"]
    assert detail, "422 responses must explain what is wrong"
    assert "text" in detail[0]["loc"]


def test_predict_rejects_text_over_the_size_limit(client: TestClient) -> None:
    """Oversized payloads are refused rather than silently truncated."""
    response = client.post("/predict", json={"text": "word " * MAX_TEXT_CHARS})

    assert response.status_code == 422
    assert "at most" in response.json()["detail"][0]["msg"]


def test_predict_accepts_text_at_the_size_limit(client: TestClient) -> None:
    """A payload exactly at the limit is still served."""
    response = client.post("/predict", json={"text": "a" * MAX_TEXT_CHARS})

    assert response.status_code == 200


def test_predict_strips_surrounding_whitespace(client: TestClient) -> None:
    """Padding does not change the prediction."""
    padded = client.post("/predict", json={"text": f"  \n{SAMPLE_TEXTS[3]}\t "})
    plain = client.post("/predict", json={"text": SAMPLE_TEXTS[3]})

    assert padded.status_code == plain.status_code == 200
    assert padded.json()["label"] == plain.json()["label"]


@pytest.mark.skipif(
    not REAL_MODEL_PATH.is_file(),
    reason="trained baseline not present; run uv run python -m src.models.baseline",
)
def test_real_baseline_model_serves_predictions(valid_labels: set[int]) -> None:
    """The production artifact loads and answers through the same contract."""
    import os

    previous = {key: os.environ.get(key) for key in ("MODEL_BACKEND", "MODEL_PATH")}
    os.environ["MODEL_BACKEND"] = "sklearn"
    os.environ["MODEL_PATH"] = str(REAL_MODEL_PATH)
    try:
        from src.api.main import app

        with TestClient(app) as real_client:
            assert real_client.get("/health").json()["model_loaded"] is True
            body = real_client.post("/predict", json={"text": SAMPLE_TEXTS[3]}).json()
            assert body["label"] in valid_labels
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(
    not REAL_ONNX_PATH.is_file(),
    reason="ONNX artifact not present; run uv run python -m src.models.export_onnx",
)
def test_real_onnx_model_agrees_with_the_sklearn_baseline(
    valid_labels: set[int],
) -> None:
    """The two production artifacts answer the same way on the same inputs.

    Not a proof of equivalence -- ``reports/onnx_equivalence.json`` measures
    that over the whole test split, where 0.79% of predictions do differ. This
    guards the shipped pair against a gross mismatch, such as an ONNX export
    left stale after the model was retrained.
    """
    import os

    def serve(backend: str, path) -> list[dict]:
        previous = {k: os.environ.get(k) for k in ("MODEL_BACKEND", "MODEL_PATH")}
        os.environ["MODEL_BACKEND"] = backend
        os.environ["MODEL_PATH"] = str(path)
        try:
            from src.api.main import app

            with TestClient(app) as client:
                assert client.get("/health").json()["model_backend"] == backend
                return [
                    client.post("/predict", json={"text": text}).json()
                    for text in SAMPLE_TEXTS
                ]
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    sklearn_results = serve("sklearn", REAL_MODEL_PATH)
    onnx_results = serve("onnx", REAL_ONNX_PATH)

    for text, from_sklearn, from_onnx in zip(
        SAMPLE_TEXTS, sklearn_results, onnx_results, strict=True
    ):
        assert from_onnx["label"] in valid_labels
        assert from_onnx["label"] == from_sklearn["label"], text
        # The envelope is wide on purpose. skl2onnx cannot express
        # stop_words='english' before n-gram construction, so ONNX misses the
        # bigrams that span a removed stopword; the measured worst case over
        # the full test split is 0.0803 (reports/onnx_equivalence.json). These
        # short crafted sentences are stopword-dense, so they sit near it.
        assert from_onnx["confidence"] == pytest.approx(
            from_sklearn["confidence"], abs=0.10
        ), text
