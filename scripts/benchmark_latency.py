"""Measure /predict latency against a running API container.

Produces the Phase 1 latency baseline that Phase 4 compares the optimized
model against, so the run is pinned to a fixed payload and records the
hardware and model version alongside the numbers.

Run with (container must already be up)::

    uv run python scripts/benchmark_latency.py
    uv run python scripts/benchmark_latency.py --requests 500 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8000"
DEFAULT_REQUESTS = 200
DEFAULT_WARMUP = 20
DEFAULT_OUTPUT = Path("reports/latency_baseline.json")

# Fixed payload: latency must be comparable across runs, so the input never
# varies. Real abstract from the corpus, 1,056 characters, near the corpus
# median of 1,208. Its sha256 is recorded in the report.
PAYLOAD_TEXT = (
    "Tissue changes around loose prostheses. A canine model to investigate "
    "the effects of an antiinflammatory agent. The aseptically loosened "
    "prosthesis provided a means for investigating the in vivo and in vitro "
    "activity of the cells associated with the loosening process in seven "
    "dogs. The cells were isolated and maintained in culture for sufficient "
    "periods of time so that their biologic activity could be studied as "
    "well as the effect of different agents added to the cells in vivo or in "
    "vitro. The biologic response as determined by interleukin-1 and "
    "prostaglandin E2 activity paralleled the roentgenographic appearance of "
    "loosening and the technetium images and observations made at the time "
    "of revision surgery. The correlation between clinical, roentgenographic, "
    "histologic, and biochemical loosening indicates that the canine model is "
    "suitable for investigating the mechanisms of prosthetic failure. A "
    "canine model permits the study of possible nonsurgical therapeutic "
    "interventions with the ultimate hope of stopping or slowing the "
    "loosening process."
)


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile of an unsorted list."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, float]:
    """Reduce raw latency samples to the reported statistics."""
    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "max_ms": max(samples),
    }


def describe_environment() -> dict[str, str]:
    """Capture the hardware and interpreter the run happened on."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def fetch_target_info(client: httpx.Client) -> dict:
    """Read /health so the report records what was actually measured."""
    response = client.get("/health", timeout=10.0)
    response.raise_for_status()
    return response.json()


def run_requests(
    client: httpx.Client, count: int, payload: dict, label: str
) -> tuple[list[float], list[float]]:
    """Issue ``count`` sequential requests, returning wall and server times."""
    wall_ms: list[float] = []
    server_ms: list[float] = []

    for index in range(count):
        started = time.perf_counter()
        response = client.post("/predict", json=payload, timeout=30.0)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()

        wall_ms.append(elapsed_ms)
        server_ms.append(float(response.json()["latency_ms"]))

        if (index + 1) % 50 == 0:
            LOGGER.info("%s: %d/%d requests", label, index + 1, count)

    return wall_ms, server_ms


def benchmark(
    url: str = DEFAULT_URL,
    requests_count: int = DEFAULT_REQUESTS,
    warmup: int = DEFAULT_WARMUP,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict:
    """Run the benchmark and persist the report."""
    payload = {"text": PAYLOAD_TEXT}

    with httpx.Client(base_url=url) as client:
        health = fetch_target_info(client)
        LOGGER.info("target %s | model_loaded=%s", url, health.get("model_loaded"))
        if not health.get("model_loaded"):
            raise RuntimeError(f"target at {url} has no model loaded; aborting")

        LOGGER.info("warmup: %d requests (discarded)", warmup)
        run_requests(client, warmup, payload, "warmup")

        LOGGER.info("measuring: %d sequential requests", requests_count)
        wall_ms, server_ms = run_requests(client, requests_count, payload, "measure")

    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_url": url,
        "protocol": {
            "requests": requests_count,
            "warmup_requests": warmup,
            "concurrency": 1,
            "mode": "sequential",
            "payload_chars": len(PAYLOAD_TEXT),
            "payload_sha256": hashlib.sha256(PAYLOAD_TEXT.encode()).hexdigest(),
        },
        "model": {
            "version": health.get("model_version"),
            "path": health.get("model_path"),
        },
        "environment": describe_environment(),
        # End-to-end latency seen by the client: this is the Phase 1 baseline.
        "client_latency": summarize(wall_ms),
        # Inference time reported by the server, excluding HTTP overhead.
        "server_latency": summarize(server_ms),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("saved report to %s", output_path)
    return report


def log_report(report: dict) -> None:
    """Log the headline numbers as a small table."""
    LOGGER.info("=" * 62)
    LOGGER.info("LATENCY BASELINE - %s", report["target_url"])
    LOGGER.info("=" * 62)
    LOGGER.info("  %-12s %10s %10s", "metric", "client", "server")
    for key, label in (
        ("p50_ms", "P50"),
        ("p95_ms", "P95"),
        ("p99_ms", "P99"),
        ("mean_ms", "mean"),
        ("stdev_ms", "stdev"),
        ("min_ms", "min"),
        ("max_ms", "max"),
    ):
        LOGGER.info(
            "  %-12s %9.2fms %9.2fms",
            label,
            report["client_latency"][key],
            report["server_latency"][key],
        )
    LOGGER.info("=" * 62)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="base URL of the API")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args(argv)
    report = benchmark(args.url, args.requests, args.warmup, args.output)
    log_report(report)


if __name__ == "__main__":
    main()
