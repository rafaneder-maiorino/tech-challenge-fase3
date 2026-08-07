"""Prometheus instrumentation for the serving API.

Metrics are registered manually against ``prometheus_client``'s default
registry rather than through ``prometheus-fastapi-instrumentator``. The
deciding factor was bucket control: the classifier answers in well under a
millisecond (see ``reports/latency_baseline.json``), and the instrumentator's
default histogram starts at 5 ms, which would drop every real request into the
first bucket and make P50/P95/P99 indistinguishable. Overriding its buckets is
possible, but once the model-level metrics below are needed anyway --
predictions per class, model readiness -- the library stops paying for itself
and only adds a dependency.

The registry is process-global, so this module must be imported exactly once
per worker. The API runs a single uvicorn worker per container, which keeps
the numbers whole; multi-worker deployments would need
``PROMETHEUS_MULTIPROC_DIR``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram
from prometheus_client import generate_latest as render_metrics

from src.labels import CONDITION_NAMES

# Path that serves the exposition format. Excluded from the request metrics:
# Prometheus scrapes it every few seconds, so counting it would dominate the
# request-rate panel and dilute the error ratio with traffic nobody sent.
METRICS_PATH = "/metrics"

# Label used when a request matches no route (404s from scanners and typos).
# Recording the raw path there would let an outsider create unbounded label
# cardinality with a loop of random URLs.
UNMATCHED_ENDPOINT = "unmatched"

# Buckets in seconds, tuned to two measured regimes rather than to the client
# library default (which begins at 5 ms and would collapse everything below it
# into one bucket).
#
# Regime 1, sequential: the Phase 1 baseline measured ~0.55 ms P50 and ~0.62 ms
# P99 for inference at concurrency 1 (reports/latency_baseline.json). The four
# boundaries below 1 ms resolve that.
#
# Regime 2, concurrent: under the load generator at 30 req/s with 8 requests in
# flight, end-to-end handler time measures ~3.8 ms mean and ~7.5 ms P99. The
# gap is not inference getting slower -- ``predict`` is a sync ``def``, so
# Starlette runs it in the anyio worker threadpool, and the middleware timer
# includes the queueing that concurrency creates. That is the honest number for
# a request, so the 1-10 ms decade gets seven boundaries instead of the two it
# had originally, where P50 was interpolated across a 2.5 ms-wide bucket and P99
# read 9.7 ms -- 30% high, an artifact of interpolating inside the 5-10 ms
# bucket rather than anything the service was doing.
#
# Above 10 ms the boundaries only need to catch a regression, not resolve one.
LATENCY_BUCKETS = (
    0.00025,  # 0.25 ms
    0.0005,  # 0.5 ms
    0.00075,  # 0.75 ms
    0.001,  # 1 ms
    0.0015,  # 1.5 ms
    0.002,  # 2 ms
    0.003,  # 3 ms
    0.004,  # 4 ms
    0.005,  # 5 ms
    0.0075,  # 7.5 ms
    0.01,  # 10 ms
    0.025,  # 25 ms
    0.05,  # 50 ms
    0.1,  # 100 ms
    0.5,  # 500 ms
    1.0,  # 1 s
    float("inf"),
)

REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "HTTP requests handled, by route and response status.",
    ("method", "endpoint", "status"),
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "End-to-end handler latency, from middleware entry to response.",
    ("method", "endpoint"),
    buckets=LATENCY_BUCKETS,
)

# Model-level observability, not HTTP-level: a service can answer every
# request with 200 and still have its prediction mix drift away from the
# distribution it was trained on.
PREDICTIONS_TOTAL = Counter(
    "api_predictions_total",
    "Predictions returned, by predicted class.",
    ("label", "label_name"),
)

# The distinction that /health already draws between 'ok' and 'degraded':
# the process can be up and serving 503s because the artifact never loaded.
MODEL_LOADED = Gauge(
    "api_model_loaded",
    "1 when the classifier is loaded and able to serve, 0 otherwise.",
)


def initialise_prediction_series() -> None:
    """Create a zero-valued counter for every class.

    Without this, a class that has not been predicted yet has no time series
    at all, so the distribution panel silently omits it instead of showing a
    zero. Called at startup.
    """
    for label, name in CONDITION_NAMES.items():
        PREDICTIONS_TOTAL.labels(label=str(label), label_name=name)


def observe_prediction(label: int) -> None:
    """Record one prediction under its class."""
    PREDICTIONS_TOTAL.labels(
        label=str(label), label_name=CONDITION_NAMES.get(label, "unknown")
    ).inc()


def set_model_loaded(loaded: bool) -> None:
    """Publish whether the classifier is ready to serve."""
    MODEL_LOADED.set(1 if loaded else 0)


def _endpoint_of(request: Request) -> str:
    """Return the route template that handled the request, not the raw path.

    ``/predict`` is a fixed path today, but labelling by template is what
    keeps cardinality bounded once any route takes a parameter.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or UNMATCHED_ENDPOINT


async def metrics_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Time every request and count it by route and status."""
    if request.url.path == METRICS_PATH:
        return await call_next(request)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled exception still becomes a 500 for the client, so it has
        # to appear in the error ratio rather than vanish from the counters.
        elapsed = time.perf_counter() - started
        endpoint = _endpoint_of(request)
        REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(
            elapsed
        )
        REQUESTS_TOTAL.labels(
            method=request.method, endpoint=endpoint, status="500"
        ).inc()
        raise

    elapsed = time.perf_counter() - started
    endpoint = _endpoint_of(request)
    REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(elapsed)
    REQUESTS_TOTAL.labels(
        method=request.method, endpoint=endpoint, status=str(response.status_code)
    ).inc()
    return response


def metrics_response() -> Response:
    """Render the default registry in the Prometheus exposition format."""
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)
