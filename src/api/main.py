"""FastAPI service exposing the medical abstract classifier.

The model is loaded once during application startup and reused across
requests; nothing is loaded or fitted inside a request handler.

Run with::

    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator

from src.api.backends import artifact_path, backend_name, load_backend
from src.api.metrics import (
    initialise_prediction_series,
    metrics_middleware,
    metrics_response,
    observe_prediction,
    set_model_loaded,
)
from src.labels import CONDITION_NAMES

LOGGER = logging.getLogger(__name__)

# Corpus abstracts top out at 3,999 characters. The cap is generous enough to
# accept any realistic report while bounding worst-case inference latency.
MAX_TEXT_CHARS = 20_000
MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline-tfidf-logreg")


def model_path() -> Path:
    """Resolve the artifact path for the configured backend."""
    return artifact_path(backend_name())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier once at startup and release it on shutdown."""
    name = backend_name()
    path = model_path()
    app.state.model = None
    app.state.model_path = str(path)
    app.state.model_version = MODEL_VERSION
    app.state.model_backend = name
    initialise_prediction_series()
    set_model_loaded(False)

    try:
        started = time.perf_counter()
        app.state.model = load_backend(name, path)
        elapsed_ms = (time.perf_counter() - started) * 1000
        set_model_loaded(True)
        LOGGER.info("backend %r loaded from %s in %.1f ms", name, path, elapsed_ms)
    except FileNotFoundError:
        # Startup deliberately succeeds so /health can report the problem
        # instead of the container crash-looping with no diagnosis.
        LOGGER.error(
            "model file not found at %s; /predict will return 503. "
            "Run: uv run python -m src.models.baseline"
            " (or -m src.models.export_onnx for the ONNX backend)",
            path,
        )
    except Exception:
        LOGGER.exception(
            "failed to load backend %r from %s; /predict will return 503", name, path
        )

    yield

    app.state.model = None
    set_model_loaded(False)
    LOGGER.info("model released")


app = FastAPI(
    title="Medical Abstract Triage API",
    description=(
        "Classifies a medical abstract into one of five clinical areas. "
        "Reference model for the FIAP Tech Challenge, phase 3."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Registered as the only middleware, so the timing it records is the whole
# server-side cost of a request: validation, inference and serialisation.
app.middleware("http")(metrics_middleware)


class PredictRequest(BaseModel):
    """Input payload for a single classification request."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_CHARS,
            description="Free-text medical abstract, in English.",
            json_schema_extra={"example": "Acute myocardial infarction after ..."},
        ),
    ]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only input, which ``min_length`` alone allows."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must contain at least one non-whitespace character")
        return stripped


class PredictResponse(BaseModel):
    """Classification result for a single abstract."""

    label: int = Field(description="Predicted condition_label, 1 to 5.")
    label_name: str = Field(description="Human-readable name of the class.")
    confidence: float = Field(description="Probability of the predicted class.")
    latency_ms: float = Field(description="Server-side inference time.")


class HealthResponse(BaseModel):
    """Liveness and readiness information."""

    status: str = Field(description="'ok' when the model is ready.")
    model_loaded: bool
    model_path: str
    model_version: str
    # Reported so a side-by-side latency run can prove which engine answered,
    # rather than inferring it from which container was started.
    model_backend: str = Field(description="'sklearn' or 'onnx'.")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Report whether the service is up and the model is loaded."""
    loaded = request.app.state.model is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_path=request.app.state.model_path,
        model_version=request.app.state.model_version,
        model_backend=request.app.state.model_backend,
    )


@app.get(
    "/metrics",
    tags=["ops"],
    summary="Prometheus metrics",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE_LATEST: {}}}},
    include_in_schema=False,
)
def metrics() -> Response:
    """Expose the Prometheus exposition format for scraping."""
    return metrics_response()


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classify one abstract into a clinical area."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"model is not loaded (expected at {request.app.state.model_path}); "
                "the service cannot serve predictions"
            ),
        )

    started = time.perf_counter()
    label, confidence = model.predict(payload.text)
    latency_ms = (time.perf_counter() - started) * 1000
    observe_prediction(label)

    return PredictResponse(
        label=label,
        label_name=CONDITION_NAMES.get(label, "unknown"),
        confidence=confidence,
        latency_ms=round(latency_ms, 3),
    )
