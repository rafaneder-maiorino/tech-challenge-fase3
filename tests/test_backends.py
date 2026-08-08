"""Tests for the pluggable inference backends.

The API contract tests in ``test_api.py`` already run against both backends
through the parametrised ``client`` fixture. What is left here is the backend
selection machinery itself, and the dependency isolation that the whole point
of the ONNX path rests on.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.api.backends import (
    DEFAULT_ARTIFACTS,
    ONNX_BACKEND,
    SKLEARN_BACKEND,
    artifact_path,
    backend_name,
    load_backend,
)

SAMPLE_TEXT = "Coronary artery bypass grafting outcomes in hypertensive patients."


def test_backend_defaults_to_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset MODEL_BACKEND keeps the behaviour phases 1 to 3 shipped."""
    monkeypatch.delenv("MODEL_BACKEND", raising=False)

    assert backend_name() == SKLEARN_BACKEND


@pytest.mark.parametrize("value", ["onnx", "ONNX", "  Onnx  "])
def test_backend_name_is_normalised(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Case and stray whitespace in an env var must not select a bad backend."""
    monkeypatch.setenv("MODEL_BACKEND", value)

    assert backend_name() == ONNX_BACKEND


def test_each_backend_has_its_own_default_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching backend without setting MODEL_PATH must not load a joblib."""
    monkeypatch.delenv("MODEL_PATH", raising=False)

    assert artifact_path(SKLEARN_BACKEND) == DEFAULT_ARTIFACTS[SKLEARN_BACKEND]
    assert artifact_path(ONNX_BACKEND) == DEFAULT_ARTIFACTS[ONNX_BACKEND]
    assert artifact_path(SKLEARN_BACKEND) != artifact_path(ONNX_BACKEND)


def test_model_path_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit MODEL_PATH wins for either backend."""
    monkeypatch.setenv("MODEL_PATH", "/somewhere/custom.onnx")

    assert artifact_path(ONNX_BACKEND) == Path("/somewhere/custom.onnx")
    assert artifact_path(SKLEARN_BACKEND) == Path("/somewhere/custom.onnx")


def test_unknown_backend_is_rejected_by_name() -> None:
    """A typo must fail loudly at startup, not serve from the wrong engine."""
    with pytest.raises(ValueError, match="unknown MODEL_BACKEND"):
        load_backend("tensorflow", Path("models/whatever"))


def test_both_backends_agree_on_the_stub_model(
    stub_model_path: Path, stub_onnx_path: Path
) -> None:
    """The two engines classify the same text the same way."""
    sklearn_backend = load_backend(SKLEARN_BACKEND, stub_model_path)
    onnx_backend = load_backend(ONNX_BACKEND, stub_onnx_path)

    sklearn_label, sklearn_confidence = sklearn_backend.predict(SAMPLE_TEXT)
    onnx_label, onnx_confidence = onnx_backend.predict(SAMPLE_TEXT)

    assert sklearn_label == onnx_label
    assert onnx_confidence == pytest.approx(sklearn_confidence, abs=1e-4)


def test_backends_return_plain_python_scalars(stub_onnx_path: Path) -> None:
    """Numpy scalars would serialise, but not as the documented JSON types."""
    label, confidence = load_backend(ONNX_BACKEND, stub_onnx_path).predict(SAMPLE_TEXT)

    assert type(label) is int
    assert type(confidence) is float


def test_onnx_backend_does_not_import_scikit_learn(stub_onnx_path: Path) -> None:
    """The whole reason the ONNX runtime image can be smaller.

    Checked in a subprocess: this test session already imported scikit-learn
    to build the stand-in model, so an in-process assertion on ``sys.modules``
    would pass no matter what the backend does.
    """
    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from src.api.backends import ONNX_BACKEND, load_backend

        backend = load_backend(ONNX_BACKEND, Path({str(stub_onnx_path)!r}))
        label, confidence = backend.predict({SAMPLE_TEXT!r})
        assert isinstance(label, int), label

        forbidden = sorted(
            name for name in sys.modules
            if name.split(".")[0] in {{"sklearn", "scipy", "joblib", "pandas"}}
        )
        if forbidden:
            sys.exit("ONNX backend pulled in: " + ", ".join(forbidden))
    """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stdout + result.stderr
