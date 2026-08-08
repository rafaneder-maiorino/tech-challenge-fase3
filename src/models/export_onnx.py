"""Export the fitted sklearn pipeline to ONNX and quantize it.

Produces two artifacts from ``models/baseline.joblib``:

* ``models/model.onnx`` -- the converted graph
* ``models/model.quantized.onnx`` -- the same graph with int8 weights

and a report at ``reports/onnx_equivalence.json`` comparing all three backends
on the held-out test split. Conversion without that comparison would be a leap
of faith: a pipeline can convert cleanly and still not compute the same thing.

Run with (training dependencies required)::

    uv run python -m src.models.export_onnx
    uv run python -m src.models.export_onnx --skip-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import onnx
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.labels import LABEL_COLUMN, TEXT_COLUMN

LOGGER = logging.getLogger(__name__)

DEFAULT_JOBLIB_PATH = Path("models/baseline.joblib")
DEFAULT_ONNX_PATH = Path("models/model.onnx")
DEFAULT_QUANTIZED_PATH = Path("models/model.quantized.onnx")
DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_REPORT_PATH = Path("reports/onnx_equivalence.json")

# Opset 18 is what onnxruntime 1.28 fully supports; pinning it keeps the graph
# reproducible instead of tracking whatever the converter defaults to.
TARGET_OPSET = 18

# The quality gate inherited from the Airflow retraining DAG. A backend that
# falls below it must not be shipped, however small the artifact.
MACRO_F1_GATE = 0.62

# scikit-learn's default token_pattern is ``(?u)\b\w\w+\b`` -- at least TWO word
# characters. skl2onnx emits a Tokenizer with ``mincharnum=1``, which keeps
# single-character tokens sklearn drops. That shifts the whole token stream and
# therefore every bigram built from it, so the boundary is corrected after
# conversion. Measured effect on the test split: 30 divergent predictions
# before, 21 after.
SKLEARN_MIN_TOKEN_CHARS = 2


def load_pipeline(path: Path) -> Pipeline:
    """Load the fitted sklearn pipeline."""
    pipeline = joblib.load(path)
    LOGGER.info("loaded pipeline from %s", path)
    return pipeline


def convert_to_onnx(pipeline: Pipeline) -> onnx.ModelProto:
    """Convert the fitted pipeline to an ONNX graph.

    Two converter choices are load-bearing:

    ``black_op={"LinearClassifier"}`` forces the logistic regression to be
    emitted as ``MatMul + Add + Softmax`` instead of the single ``ai.onnx.ml``
    ``LinearClassifier`` operator. The default is a perfectly good graph, but
    it stores the coefficients as *node attributes*, and
    ``quantize_dynamic`` only ever rewrites *initializers* feeding
    MatMul/Gemm/Conv. Converted the default way, quantization is a silent
    no-op: byte-identical output, zero nodes changed. Decomposed, the 50000x5
    coefficient matrix becomes a quantizable initializer.

    ``zipmap=False`` returns probabilities as a plain float tensor rather than
    a list of dictionaries, which is both faster and simpler to consume.
    """
    with warnings.catch_warnings():
        # skl2onnx warns about TfidfVectorizer edge cases it cannot express;
        # those are measured and reported by validate_backends instead.
        warnings.simplefilter("ignore")
        model = to_onnx(
            pipeline,
            initial_types=[("input", StringTensorType([None, 1]))],
            options={id(pipeline): {"zipmap": False}},
            target_opset=TARGET_OPSET,
            black_op={"LinearClassifier"},
        )

    _align_tokenizer_with_sklearn(model)
    _pin_string_normalizer_locale(model)
    onnx.checker.check_model(model)
    return model


def _pin_string_normalizer_locale(model: onnx.ModelProto) -> None:
    """Pin StringNormalizer to the C locale so slim base images can load it.

    onnxruntime defaults this operator to ``en_US.UTF-8`` and constructs a
    ``std::locale`` from it at session initialisation. ``python:3.11-slim``
    ships no locale definitions, so the session throws before serving a single
    request -- a failure that appears only inside the image, never on a
    developer machine.

    Installing ``locales`` in the runtime would fix it and cost image size,
    which is the one thing this phase is trying to reduce. ``C`` is guaranteed
    to exist everywhere. The only behaviour it changes is case folding outside
    ASCII, and the tokenizer regex that runs immediately after is
    ``[a-zA-Z0-9_]+``, so non-ASCII characters are dropped either way.
    """
    for node in model.graph.node:
        if node.op_type != "StringNormalizer":
            continue
        attribute = next((a for a in node.attribute if a.name == "locale"), None)
        if attribute is None:
            node.attribute.append(onnx.helper.make_attribute("locale", "C"))
        else:
            attribute.s = b"C"
        LOGGER.info("pinned StringNormalizer locale to C for %s", node.name or "node")


def _align_tokenizer_with_sklearn(model: onnx.ModelProto) -> None:
    """Raise the tokenizer's minimum token length to match scikit-learn."""
    for node in model.graph.node:
        if node.op_type != "Tokenizer":
            continue
        for attribute in node.attribute:
            if attribute.name == "mincharnum":
                previous = attribute.i
                attribute.i = SKLEARN_MIN_TOKEN_CHARS
                LOGGER.info(
                    "tokenizer mincharnum %d -> %d (sklearn token_pattern is "
                    r"\b\w\w+\b)",
                    previous,
                    SKLEARN_MIN_TOKEN_CHARS,
                )
                return
    LOGGER.warning("no Tokenizer node found; graph may not use text input")


def quantize(source: Path, destination: Path) -> None:
    """Apply dynamic int8 quantization to the exported graph.

    ``DefaultTensorType`` is required because the tokenizer chain comes from
    the ``com.microsoft`` domain, which ONNX shape inference cannot type. With
    no hint the quantizer aborts on the MatMul it is trying to rewrite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(
            source,
            destination,
            weight_type=QuantType.QUInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
    LOGGER.info("quantized %s -> %s", source, destination)


def _predict_sklearn(pipeline: Pipeline, texts: np.ndarray) -> tuple:
    """Return (labels, probabilities) from the sklearn pipeline."""
    probabilities = pipeline.predict_proba(texts)
    labels = pipeline.classes_[probabilities.argmax(axis=1)]
    return labels, probabilities


def _predict_onnx(path: Path, texts: np.ndarray) -> tuple:
    """Return (labels, probabilities) from an ONNX graph."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    labels, probabilities = session.run(None, {"input": texts.reshape(-1, 1)})
    return np.asarray(labels).ravel(), np.asarray(probabilities)


def _compare(reference_labels, reference_proba, labels, proba) -> dict:
    """Quantify how far a backend drifts from the sklearn reference."""
    disagreements = reference_labels != labels
    absolute_difference = np.abs(reference_proba - proba)
    return {
        "disagreements": int(disagreements.sum()),
        "disagreement_rate": float(disagreements.mean()),
        "prob_max_abs_diff": float(absolute_difference.max()),
        "prob_mean_abs_diff": float(absolute_difference.mean()),
        "prob_p99_abs_diff": float(np.percentile(absolute_difference, 99)),
    }


def validate_backends(
    pipeline: Pipeline,
    onnx_path: Path,
    quantized_path: Path,
    test_path: Path,
) -> dict:
    """Score all three backends on the test split and compare them.

    The reference is always sklearn, because that is the artifact every metric
    reported in phases 1 to 3 was computed against.
    """
    test = pd.read_parquet(test_path)
    texts = test[TEXT_COLUMN].to_numpy()
    truth = test[LABEL_COLUMN].to_numpy()
    LOGGER.info("validating on %d held-out abstracts", len(texts))

    sklearn_labels, sklearn_proba = _predict_sklearn(pipeline, texts)
    onnx_labels, onnx_proba = _predict_onnx(onnx_path, texts)
    quantized_labels, quantized_proba = _predict_onnx(quantized_path, texts)

    def score(labels) -> dict:
        return {
            "accuracy": float(accuracy_score(truth, labels)),
            "macro_f1": float(f1_score(truth, labels, average="macro")),
            "weighted_f1": float(f1_score(truth, labels, average="weighted")),
        }

    report = {
        "n_samples": int(len(texts)),
        "macro_f1_gate": MACRO_F1_GATE,
        "backends": {
            "sklearn": {
                "artifact": str(DEFAULT_JOBLIB_PATH),
                "metrics": score(sklearn_labels),
            },
            "onnx": {
                "artifact": str(onnx_path),
                "metrics": score(onnx_labels),
                "vs_sklearn": _compare(
                    sklearn_labels, sklearn_proba, onnx_labels, onnx_proba
                ),
            },
            "onnx_quantized": {
                "artifact": str(quantized_path),
                "metrics": score(quantized_labels),
                "vs_sklearn": _compare(
                    sklearn_labels, sklearn_proba, quantized_labels, quantized_proba
                ),
                "vs_onnx": _compare(
                    onnx_labels, onnx_proba, quantized_labels, quantized_proba
                ),
            },
        },
    }

    below_gate = []
    for name, entry in report["backends"].items():
        macro_f1 = entry["metrics"]["macro_f1"]
        passed = macro_f1 >= MACRO_F1_GATE
        LOGGER.info(
            "[%s] macro_f1=%.6f accuracy=%.6f -- %s",
            name,
            macro_f1,
            entry["metrics"]["accuracy"],
            "OK" if passed else "BELOW GATE",
        )
        if not passed:
            below_gate.append((name, macro_f1))

    report["gate_passed"] = not below_gate
    if below_gate:
        # Raised rather than logged: the image build runs this step, and a
        # quantization that quietly cost accuracy is exactly the failure this
        # phase could introduce. Better a red build than a smaller artifact
        # that classifies worse.
        summary = ", ".join(f"{name}={value:.6f}" for name, value in below_gate)
        raise RuntimeError(
            f"macro-F1 below the {MACRO_F1_GATE} gate for: {summary}. "
            "The artifact must not be shipped."
        )

    return report


def artifact_sizes(paths: dict[str, Path]) -> dict:
    """Return byte sizes plus the reduction against the joblib baseline."""
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    baseline = sizes["joblib"]
    return {
        name: {
            "bytes": size,
            "mib": round(size / 1024**2, 3),
            "vs_joblib_pct": round(100 * (size - baseline) / baseline, 2),
        }
        for name, size in sizes.items()
    }


def export(
    joblib_path: Path = DEFAULT_JOBLIB_PATH,
    onnx_path: Path = DEFAULT_ONNX_PATH,
    quantized_path: Path = DEFAULT_QUANTIZED_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    validate: bool = True,
) -> dict:
    """Convert, quantize, validate and persist the report."""
    pipeline = load_pipeline(joblib_path)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = convert_to_onnx(pipeline)
    onnx_path.write_bytes(model.SerializeToString())
    LOGGER.info("wrote %s (%.2f MiB)", onnx_path, onnx_path.stat().st_size / 1024**2)

    quantize(onnx_path, quantized_path)
    LOGGER.info(
        "wrote %s (%.2f MiB)", quantized_path, quantized_path.stat().st_size / 1024**2
    )

    report = {
        "target_opset": TARGET_OPSET,
        "sizes": artifact_sizes(
            {
                "joblib": joblib_path,
                "onnx": onnx_path,
                "onnx_quantized": quantized_path,
            }
        ),
    }
    if validate:
        report["equivalence"] = validate_backends(
            pipeline, onnx_path, quantized_path, test_path
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("wrote %s", report_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joblib-path", type=Path, default=DEFAULT_JOBLIB_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--quantized-path", type=Path, default=DEFAULT_QUANTIZED_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Convert and quantize without scoring the test split.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    export(
        joblib_path=args.joblib_path,
        onnx_path=args.onnx_path,
        quantized_path=args.quantized_path,
        test_path=args.test_path,
        report_path=args.report_path,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
