"""Compare sklearn and ONNX serving latency across three isolated levels.

The optimisation of phase 4 touches exactly one thing: how long it takes to
turn a string into a probability vector. Measuring only end-to-end HTTP would
bury that inside request parsing, threadpool queueing and JSON serialisation,
and report a "gain" that says more about the web stack than about the model.
So three levels are measured separately:

  a. pure inference  -- backend.predict() called in-process, no HTTP at all.
                        This is what ONNX and quantization actually control.
  b. HTTP sequential -- one request at a time, the phase 1 protocol.
  c. HTTP concurrent -- N requests in flight, the phase 3 scenario, where
                        server-side queueing dominates.

Levels b and c also record the server-reported ``latency_ms``, which times
only the predict call inside the handler. Comparing it against client wall
time is what separates inference from queueing without guessing.

Both containers stay up for the whole run, but load is applied to one at a
time -- see ``CONTAINER_PROTOCOL``.

Run with (both containers already up)::

    uv run python scripts/compare_backends.py
    uv run python scripts/compare_backends.py --concurrency 16 --rounds 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

# Same fixed payload and the same percentile definition as the phase 1
# baseline, so every number here is directly comparable to
# reports/latency_baseline.json instead of merely similar.
from benchmark_latency import PAYLOAD_TEXT, describe_environment, summarize

LOGGER = logging.getLogger(__name__)

# Running this file directly puts scripts/ on sys.path, which is what makes the
# benchmark_latency import above work -- but it leaves the repository root off,
# so `src.api.backends` would not resolve. Level A loads the backends
# in-process, so it needs it.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SKLEARN_URL = "http://localhost:8000"
DEFAULT_ONNX_URL = "http://localhost:8001"
DEFAULT_OUTPUT = Path("reports/backend_comparison.json")
DEFAULT_EQUIVALENCE = Path("reports/onnx_equivalence.json")

DEFAULT_INFERENCE_ITERATIONS = 2000
DEFAULT_HTTP_REQUESTS = 500
DEFAULT_CONCURRENT_REQUESTS = 2000
DEFAULT_CONCURRENCY = 8
DEFAULT_WARMUP = 50
DEFAULT_ROUNDS = 3

SKLEARN = "sklearn"
ONNX = "onnx"

# Which artifact each backend is measured on. The ONNX side uses the quantized
# graph because that is the one the phase is proposing to ship; both graphs are
# baked into the image, so no rebuild is needed to point at it.
ARTIFACTS = {
    SKLEARN: Path("models/baseline.joblib"),
    ONNX: Path("models/model.quantized.onnx"),
}
CONTAINER_ARTIFACTS = {
    SKLEARN: "/app/models/baseline.joblib",
    ONNX: "/app/models/model.quantized.onnx",
}

CONTAINER_PROTOCOL = (
    "Both containers stay up for the entire run, and load is applied to one "
    "backend at a time. Keeping them up avoids the drift that a stop/start "
    "between measurements would introduce (page cache, CPU frequency, Docker "
    "VM state); loading only one at a time avoids the CPU contention that "
    "would make the concurrent level measure scheduler pressure instead of "
    "the backend. The idle container costs effectively nothing, and its "
    "healthcheck is disabled for the run so it cannot spawn an interpreter "
    "mid-measurement. Levels are interleaved sklearn-then-onnx so the two "
    "sit as close together in time as possible."
)


# --------------------------------------------------------------------------
# Level A: pure inference, in-process
# --------------------------------------------------------------------------
def measure_pure_inference(
    backend_name: str, artifact: Path, iterations: int, warmup: int
) -> list[float]:
    """Time ``predict`` directly, with no HTTP anywhere in the path."""
    from src.api.backends import load_backend

    backend = load_backend(backend_name, artifact)

    for _ in range(warmup):
        backend.predict(PAYLOAD_TEXT)

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        backend.predict(PAYLOAD_TEXT)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def measure_pure_inference_interleaved(
    iterations: int, warmup: int, rounds: int
) -> dict[str, list[float]]:
    """Alternate between backends across rounds and pool the samples.

    A single block per backend would let a CPU frequency change or a
    background process land entirely on one of them. Alternating blocks
    spreads any such drift across both.
    """
    pooled: dict[str, list[float]] = {SKLEARN: [], ONNX: []}
    per_round = max(1, iterations // rounds)

    for round_index in range(rounds):
        for name in (SKLEARN, ONNX):
            samples = measure_pure_inference(
                name, ARTIFACTS[name], per_round, warmup if round_index == 0 else 0
            )
            pooled[name].extend(samples)
        LOGGER.info(
            "pure inference round %d/%d done (%d samples each)",
            round_index + 1,
            rounds,
            per_round,
        )
    return pooled


# --------------------------------------------------------------------------
# Levels B and C: over HTTP
# --------------------------------------------------------------------------
def check_target(url: str, expected_backend: str, expected_path: str) -> dict:
    """Verify a container is serving what the report is about to claim."""
    with httpx.Client(base_url=url, timeout=10.0) as client:
        response = client.get("/health")
        response.raise_for_status()
        health = response.json()

    if not health.get("model_loaded"):
        raise RuntimeError(f"{url} has no model loaded")
    actual_backend = health.get("model_backend")
    if actual_backend != expected_backend:
        raise RuntimeError(
            f"{url} serves backend {actual_backend!r}, expected {expected_backend!r}"
        )
    actual_path = health.get("model_path")
    if actual_path != expected_path:
        raise RuntimeError(
            f"{url} serves artifact {actual_path!r}, expected {expected_path!r}. "
            "Start the ONNX container with "
            "-e MODEL_PATH=/app/models/model.quantized.onnx"
        )
    LOGGER.info("%s -> backend=%s artifact=%s", url, actual_backend, actual_path)
    return health


def measure_http_sequential(
    url: str, count: int, warmup: int
) -> tuple[list[float], list[float], float]:
    """One at a time; returns (client wall ms, server inference ms, seconds)."""
    payload = {"text": PAYLOAD_TEXT}
    wall: list[float] = []
    server: list[float] = []

    with httpx.Client(base_url=url, timeout=30.0) as client:
        for _ in range(warmup):
            client.post("/predict", json=payload)

        block_started = time.perf_counter()
        for _ in range(count):
            started = time.perf_counter()
            response = client.post("/predict", json=payload)
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            wall.append(elapsed)
            server.append(float(response.json()["latency_ms"]))
        duration = time.perf_counter() - block_started

    return wall, server, duration


async def _concurrent_worker(
    client: httpx.AsyncClient,
    queue: asyncio.Queue,
    wall: list[float],
    server: list[float],
) -> None:
    """Consume slots from the queue, timing each request."""
    payload = {"text": PAYLOAD_TEXT}
    while True:
        try:
            await queue.get()
        except asyncio.CancelledError:
            return
        try:
            started = time.perf_counter()
            response = await client.post("/predict", json=payload)
            elapsed = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            wall.append(elapsed)
            server.append(float(response.json()["latency_ms"]))
        finally:
            queue.task_done()


async def _measure_http_concurrent(
    url: str, count: int, concurrency: int, warmup: int
) -> tuple[list[float], list[float], float]:
    """Keep ``concurrency`` requests in flight until ``count`` are done."""
    wall: list[float] = []
    server: list[float] = []
    limits = httpx.Limits(max_connections=concurrency)

    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limits) as client:
        payload = {"text": PAYLOAD_TEXT}
        await asyncio.gather(
            *(client.post("/predict", json=payload) for _ in range(warmup))
        )

        # Unbounded queue filled up front: the goal is saturation, so requests
        # are never paced. The concurrency cap comes from the worker count.
        queue: asyncio.Queue = asyncio.Queue()
        for _ in range(count):
            queue.put_nowait(None)

        started = time.perf_counter()
        workers = [
            asyncio.create_task(_concurrent_worker(client, queue, wall, server))
            for _ in range(concurrency)
        ]
        await queue.join()
        duration = time.perf_counter() - started
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return wall, server, duration


def measure_http_concurrent(
    url: str, count: int, concurrency: int, warmup: int
) -> tuple[list[float], list[float], float]:
    """Synchronous wrapper around the concurrent measurement."""
    return asyncio.run(_measure_http_concurrent(url, count, concurrency, warmup))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def delta(baseline: dict, candidate: dict) -> dict:
    """Absolute and relative change from sklearn to ONNX, per statistic."""
    out = {}
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
        before, after = baseline[key], candidate[key]
        out[key.replace("_ms", "")] = {
            "sklearn_ms": round(before, 4),
            "onnx_ms": round(after, 4),
            "delta_ms": round(after - before, 4),
            "delta_pct": round(100 * (after - before) / before, 2) if before else None,
        }
    return out


def artifact_info(path: Path) -> dict:
    """Size and checksum of a model artifact."""
    if not path.is_file():
        return {"path": str(path), "present": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "present": True,
        "bytes": len(data),
        "mib": round(len(data) / 1024**2, 3),
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def image_sizes(tags: dict[str, str]) -> dict:
    """Read image sizes from the local Docker daemon, if it is reachable."""
    sizes = {}
    for name, tag in tags.items():
        try:
            raw = subprocess.run(
                ["docker", "image", "inspect", tag, "--format", "{{.Size}}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            human = subprocess.run(
                ["docker", "images", tag, "--format", "{{.Size}}"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            sizes[name] = {
                "tag": tag,
                "compressed_bytes": int(raw),
                "docker_images_size": human,
            }
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            LOGGER.warning("could not read size for image %s", tag)
            sizes[name] = {"tag": tag, "error": "unavailable"}
    return sizes


def equivalence_caveat(path: Path) -> dict:
    """Carry the phase 4 part 1 equivalence finding into this report.

    Without it the table below reads as one model on two runtimes, which is
    not what is being compared: skl2onnx cannot apply stop_words before
    n-gram construction, so the ONNX graph sees a different feature vector.
    """
    caveat = {
        "identical_models": False,
        "explanation": (
            "The two backends do not compute the same function. skl2onnx has "
            "no way to remove stop words before n-gram construction, so the "
            "ONNX graph misses every bigram that spans a removed stop word "
            "-- about 24% of vocabulary bigram occurrences, and 71% of the "
            "vocabulary is bigrams. Aggregate quality is indistinguishable "
            "(macro-F1 0.6707 vs 0.6720, both far above the 0.62 gate), but "
            "0.79% of test-set predictions differ. This is a latency "
            "comparison between two slightly different models, not the same "
            "model on two runtimes."
        ),
    }
    if not path.is_file():
        caveat["source"] = f"{path} not found"
        return caveat

    report = json.loads(path.read_text())
    backends = report.get("equivalence", {}).get("backends", {})
    caveat["source"] = str(path)
    caveat["macro_f1"] = {
        name: entry["metrics"]["macro_f1"] for name, entry in backends.items()
    }
    quantized = backends.get("onnx_quantized", {}).get("vs_sklearn", {})
    caveat["quantized_vs_sklearn"] = quantized
    return caveat


def log_level_table(title: str, note: str, results: dict) -> None:
    """Print one level as a comparison table."""
    LOGGER.info("")
    LOGGER.info("=" * 74)
    LOGGER.info("%s", title)
    LOGGER.info("%s", note)
    LOGGER.info("=" * 74)
    LOGGER.info(
        "  %-8s %11s %11s %11s %11s", "stat", "sklearn", "onnx", "delta", "delta %"
    )
    for key in ("p50", "p95", "p99", "mean"):
        row = results[key]
        pct = f"{row['delta_pct']:+.1f}%" if row["delta_pct"] is not None else "n/a"
        LOGGER.info(
            "  %-8s %9.3fms %9.3fms %9.3fms %11s",
            key.upper(),
            row["sklearn_ms"],
            row["onnx_ms"],
            row["delta_ms"],
            pct,
        )
    LOGGER.info("=" * 74)


def compare(args: argparse.Namespace) -> dict:
    """Run all three levels for both backends and build the report."""
    urls = {SKLEARN: args.sklearn_url, ONNX: args.onnx_url}

    health = {
        name: check_target(urls[name], name, CONTAINER_ARTIFACTS[name])
        for name in (SKLEARN, ONNX)
    }

    # ---- Level A ----------------------------------------------------------
    LOGGER.info(
        "level A: pure inference, %d iterations each", args.inference_iterations
    )
    pure = measure_pure_inference_interleaved(
        args.inference_iterations, args.warmup, args.rounds
    )
    pure_stats = {name: summarize(samples) for name, samples in pure.items()}

    # ---- Level B ----------------------------------------------------------
    LOGGER.info("level B: sequential HTTP, %d requests each", args.http_requests)
    sequential_wall, sequential_server, sequential_rps = {}, {}, {}
    for name in (SKLEARN, ONNX):
        wall, server, duration = measure_http_sequential(
            urls[name], args.http_requests, args.warmup
        )
        sequential_wall[name] = summarize(wall)
        sequential_server[name] = summarize(server)
        sequential_rps[name] = round(args.http_requests / duration, 1)

    # ---- Level C ----------------------------------------------------------
    LOGGER.info(
        "level C: concurrent HTTP, %d requests at concurrency %d",
        args.concurrent_requests,
        args.concurrency,
    )
    concurrent_wall, concurrent_server, concurrent_rps = {}, {}, {}
    for name in (SKLEARN, ONNX):
        wall, server, duration = measure_http_concurrent(
            urls[name], args.concurrent_requests, args.concurrency, args.warmup
        )
        concurrent_wall[name] = summarize(wall)
        concurrent_server[name] = summarize(server)
        concurrent_rps[name] = round(args.concurrent_requests / duration, 1)

    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": {
            "container_protocol": CONTAINER_PROTOCOL,
            "levels": {
                "pure_inference": (
                    "backend.predict() in-process, no HTTP; interleaved rounds"
                ),
                "http_sequential": "one request at a time over HTTP",
                "http_concurrent": (f"{args.concurrency} requests in flight over HTTP"),
            },
            "inference_iterations": args.inference_iterations,
            "inference_rounds": args.rounds,
            "http_requests": args.http_requests,
            "concurrent_requests": args.concurrent_requests,
            "concurrency": args.concurrency,
            "warmup_discarded": args.warmup,
            "payload_chars": len(PAYLOAD_TEXT),
            "payload_sha256": hashlib.sha256(PAYLOAD_TEXT.encode()).hexdigest(),
        },
        "environment": describe_environment(),
        "targets": {
            name: {
                "url": urls[name],
                "model_backend": health[name].get("model_backend"),
                "model_path": health[name].get("model_path"),
                "model_version": health[name].get("model_version"),
            }
            for name in (SKLEARN, ONNX)
        },
        "artifacts": {name: artifact_info(path) for name, path in ARTIFACTS.items()},
        "images": image_sizes(
            {SKLEARN: args.sklearn_image, ONNX: args.onnx_image},
        ),
        "equivalence_caveat": equivalence_caveat(args.equivalence_report),
        "levels": {
            "a_pure_inference": {
                "samples": pure_stats,
                "comparison": delta(pure_stats[SKLEARN], pure_stats[ONNX]),
            },
            "b_http_sequential": {
                "client_latency": sequential_wall,
                "server_inference": sequential_server,
                "throughput_rps": sequential_rps,
                "comparison": delta(sequential_wall[SKLEARN], sequential_wall[ONNX]),
                "comparison_server": delta(
                    sequential_server[SKLEARN], sequential_server[ONNX]
                ),
            },
            "c_http_concurrent": {
                "client_latency": concurrent_wall,
                "server_inference": concurrent_server,
                # Under saturation this is the metric that is not an artefact
                # of queue depth: both backends always have `concurrency`
                # requests in flight, so whoever finishes the fixed batch
                # sooner did more work per second.
                "throughput_rps": concurrent_rps,
                "comparison": delta(concurrent_wall[SKLEARN], concurrent_wall[ONNX]),
                "comparison_server": delta(
                    concurrent_server[SKLEARN], concurrent_server[ONNX]
                ),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("saved report to %s", args.output)
    return report


def log_report(report: dict) -> None:
    """Print all three levels plus the artifact and image summary."""
    levels = report["levels"]

    log_level_table(
        "LEVEL A - PURE INFERENCE (no HTTP)",
        "What ONNX and quantization actually control.",
        levels["a_pure_inference"]["comparison"],
    )
    log_level_table(
        "LEVEL B - END TO END, SEQUENTIAL (client wall time)",
        "Inference plus HTTP, validation and serialisation.",
        levels["b_http_sequential"]["comparison"],
    )
    log_level_table(
        "LEVEL B' - SEQUENTIAL, SERVER-REPORTED INFERENCE ONLY",
        "The predict() call inside the handler, excluding HTTP.",
        levels["b_http_sequential"]["comparison_server"],
    )
    log_level_table(
        "LEVEL C - END TO END, CONCURRENT (client wall time)",
        "Includes Starlette threadpool queueing.",
        levels["c_http_concurrent"]["comparison"],
    )
    log_level_table(
        "LEVEL C' - CONCURRENT, SERVER-REPORTED INFERENCE ONLY",
        "Same requests, inference time alone.",
        levels["c_http_concurrent"]["comparison_server"],
    )

    LOGGER.info("")
    LOGGER.info("THROUGHPUT (requests/second completed)")
    for level_key, label in (
        ("b_http_sequential", "sequential"),
        ("c_http_concurrent", "concurrent"),
    ):
        rps = levels[level_key]["throughput_rps"]
        change = 100 * (rps[ONNX] - rps[SKLEARN]) / rps[SKLEARN]
        LOGGER.info(
            "  %-11s sklearn=%8.1f rps   onnx=%8.1f rps   %+.1f%%",
            label,
            rps[SKLEARN],
            rps[ONNX],
            change,
        )

    LOGGER.info("")
    LOGGER.info("ARTIFACTS")
    for name, info in report["artifacts"].items():
        if info.get("present"):
            LOGGER.info("  %-8s %8.3f MiB  %s", name, info["mib"], info["path"])
    LOGGER.info("IMAGES")
    for name, info in report["images"].items():
        if "error" not in info:
            LOGGER.info(
                "  %-8s %-10s (%s)",
                name,
                info["docker_images_size"],
                info["tag"],
            )
    LOGGER.info("")
    LOGGER.warning("CAVEAT: %s", report["equivalence_caveat"]["explanation"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sklearn-url", default=DEFAULT_SKLEARN_URL)
    parser.add_argument("--onnx-url", default=DEFAULT_ONNX_URL)
    parser.add_argument("--sklearn-image", default="tc-fase3-api:sklearn")
    parser.add_argument("--onnx-image", default="tc-fase3-api:onnx")
    parser.add_argument(
        "--inference-iterations", type=int, default=DEFAULT_INFERENCE_ITERATIONS
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--http-requests", type=int, default=DEFAULT_HTTP_REQUESTS)
    parser.add_argument(
        "--concurrent-requests", type=int, default=DEFAULT_CONCURRENT_REQUESTS
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--equivalence-report", type=Path, default=DEFAULT_EQUIVALENCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    try:
        report = compare(args)
    except (RuntimeError, httpx.HTTPError) as error:
        LOGGER.error("%s", error)
        LOGGER.error(
            "Both containers must be up. See docs/optimization.md, section 11."
        )
        return 1
    log_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
