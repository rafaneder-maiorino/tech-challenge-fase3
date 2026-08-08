"""Pluggable inference backends for the serving API.

The service can answer from either the scikit-learn pipeline or the exported
ONNX graph, chosen by ``MODEL_BACKEND``. The point is not that either is
faster -- the model is already sub-millisecond -- but that the ONNX path lets
the runtime image be built without scikit-learn and scipy at all.

For that to hold, **every backend imports its engine inside its own
constructor**. A module-level ``import joblib`` here would make this file
unimportable in the ONNX image, which has no joblib, and the split would be
worthless. ``numpy`` is the one exception: it is a hard dependency of both
scikit-learn and onnxruntime, so it is present either way.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

LOGGER = logging.getLogger(__name__)

SKLEARN_BACKEND = "sklearn"
ONNX_BACKEND = "onnx"
DEFAULT_BACKEND = SKLEARN_BACKEND

# Each backend has its own natural artifact, so MODEL_PATH can stay unset and
# still resolve to the right file when MODEL_BACKEND changes.
DEFAULT_ARTIFACTS = {
    SKLEARN_BACKEND: Path("models/baseline.joblib"),
    ONNX_BACKEND: Path("models/model.onnx"),
}


@runtime_checkable
class Backend(Protocol):
    """What the API needs from an inference engine, and nothing more."""

    name: str
    artifact_path: Path

    def predict(self, text: str) -> tuple[int, float]:
        """Return the predicted label and its probability."""
        ...


class SklearnBackend:
    """Serve predictions from the pickled scikit-learn pipeline."""

    name = SKLEARN_BACKEND

    def __init__(self, artifact_path: Path) -> None:
        """Load the joblib artifact."""
        import joblib

        self.artifact_path = artifact_path
        self._pipeline = joblib.load(artifact_path)
        # Cached as plain ints so the request path never touches numpy scalars.
        self._classes = [int(label) for label in self._pipeline.classes_]

    def predict(self, text: str) -> tuple[int, float]:
        """Classify one abstract."""
        probabilities = self._pipeline.predict_proba([text])[0]
        best = int(probabilities.argmax())
        return self._classes[best], float(probabilities[best])


class OnnxBackend:
    """Serve predictions from the exported ONNX graph.

    Deliberately free of scikit-learn: the graph carries the vectoriser, the
    idf weights and the classifier coefficients, so onnxruntime is the only
    engine needed.
    """

    name = ONNX_BACKEND

    def __init__(self, artifact_path: Path) -> None:
        """Open an inference session over the graph."""
        import onnxruntime as ort

        self.artifact_path = artifact_path
        options = ort.SessionOptions()
        # One request carries one abstract, so intra-op parallelism buys
        # nothing and costs thread hand-offs. Phase 4 part 2 measures this.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(artifact_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, text: str) -> tuple[int, float]:
        """Classify one abstract."""
        # The graph takes a 2D string tensor: one row per document, one column.
        payload = np.array([[text]], dtype=object)
        labels, probabilities = self._session.run(None, {self._input_name: payload})
        row = np.asarray(probabilities)[0]
        best = int(row.argmax())
        return int(np.asarray(labels).ravel()[0]), float(row[best])


BACKENDS = {
    SKLEARN_BACKEND: SklearnBackend,
    ONNX_BACKEND: OnnxBackend,
}


def backend_name() -> str:
    """Resolve the configured backend name, defaulting to scikit-learn."""
    return os.getenv("MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()


def artifact_path(name: str) -> Path:
    """Resolve the artifact path for a backend.

    ``MODEL_PATH`` wins when set, so an operator can point either backend at a
    specific file; otherwise each backend falls back to its own default.
    """
    override = os.getenv("MODEL_PATH")
    if override:
        return Path(override)
    return DEFAULT_ARTIFACTS.get(name, DEFAULT_ARTIFACTS[DEFAULT_BACKEND])


def load_backend(name: str, path: Path) -> Backend:
    """Instantiate the named backend against an artifact.

    Raises:
        ValueError: if the backend name is not recognised.
        ImportError: if the engine for that backend is not installed. The
            message names the extra to install, because the most likely cause
            is running the ONNX image's code in a scikit-learn environment or
            the reverse.
    """
    try:
        backend_class = BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown MODEL_BACKEND {name!r}; expected one of {sorted(BACKENDS)}"
        ) from None

    try:
        backend = backend_class(path)
    except ImportError as error:
        raise ImportError(
            f"backend {name!r} needs a dependency that is not installed "
            f"({error}). Install it with: uv sync --extra {name}"
        ) from error

    LOGGER.info("backend %r loaded from %s", name, path)
    return backend
