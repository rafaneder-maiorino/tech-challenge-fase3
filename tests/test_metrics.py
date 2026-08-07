"""Contract tests for the Prometheus instrumentation.

The default registry is process-global and every test in this session shares
it, so nothing here asserts an absolute counter value. The assertions are on
deltas around a request and on the shape of the exposition output, which is
what a scrape actually depends on.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src.api.metrics import LATENCY_BUCKETS, metrics_middleware, metrics_response
from src.labels import CONDITION_NAMES

SAMPLE_TEXT = "Coronary artery bypass grafting outcomes in hypertensive patients."


def sample_value(body: str, metric: str, **labels: str) -> float:
    """Return one sample from an exposition payload, or 0.0 if absent."""
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name == metric and all(
                sample.labels.get(key) == value for key, value in labels.items()
            ):
                return sample.value
    return 0.0


def scrape(client: TestClient) -> str:
    """Fetch /metrics the way Prometheus would."""
    response = client.get("/metrics")
    assert response.status_code == 200
    return response.text


def test_metrics_endpoint_serves_the_exposition_format(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE api_requests_total counter" in response.text
    assert "# TYPE api_request_duration_seconds histogram" in response.text
    assert "# TYPE api_predictions_total counter" in response.text
    assert "# TYPE api_model_loaded gauge" in response.text


def test_successful_predict_increments_the_request_counter(client: TestClient) -> None:
    labels = {"method": "POST", "endpoint": "/predict", "status": "200"}
    before = sample_value(scrape(client), "api_requests_total", **labels)

    client.post("/predict", json={"text": SAMPLE_TEXT})

    after = sample_value(scrape(client), "api_requests_total", **labels)
    assert after == before + 1


def test_rejected_payload_is_counted_under_its_own_status(client: TestClient) -> None:
    """A 422 must land in a separate series, or the error panel reads zero."""
    labels = {"method": "POST", "endpoint": "/predict", "status": "422"}
    before = sample_value(scrape(client), "api_requests_total", **labels)

    assert client.post("/predict", json={"text": ""}).status_code == 422

    after = sample_value(scrape(client), "api_requests_total", **labels)
    assert after == before + 1


def test_unmatched_routes_do_not_create_a_series_per_path(client: TestClient) -> None:
    """404s from scanners must not turn the raw path into a label value."""
    client.get("/does-not-exist-a")
    client.get("/does-not-exist-b")

    body = scrape(client)
    assert sample_value(body, "api_requests_total", endpoint="unmatched") >= 2
    assert "/does-not-exist-a" not in body


def test_predictions_are_counted_per_predicted_class(client: TestClient) -> None:
    predicted = client.post("/predict", json={"text": SAMPLE_TEXT}).json()["label"]
    labels = {
        "label": str(predicted),
        "label_name": CONDITION_NAMES[predicted],
    }
    after = sample_value(scrape(client), "api_predictions_total", **labels)

    client.post("/predict", json={"text": SAMPLE_TEXT})

    assert sample_value(scrape(client), "api_predictions_total", **labels) == after + 1


def test_every_class_has_a_series_before_it_is_ever_predicted(
    client: TestClient,
) -> None:
    """Zero-valued series keep the distribution panel from hiding a class."""
    body = scrape(client)

    for label, name in CONDITION_NAMES.items():
        assert (
            f'api_predictions_total{{label="{label}",label_name="{name}"}}' in body
        ), f"class {label} has no series"


def test_model_loaded_gauge_is_one_when_the_model_is_ready(
    client: TestClient,
) -> None:
    assert sample_value(scrape(client), "api_model_loaded") == 1.0


def test_model_loaded_gauge_is_zero_when_the_model_is_missing(
    client_without_model: TestClient,
) -> None:
    """The metric has to separate 'up' from 'able to serve'."""
    body = scrape(client_without_model)

    assert sample_value(body, "api_model_loaded") == 0.0
    assert client_without_model.get("/health").json()["model_loaded"] is False


def test_latency_is_observed_into_the_histogram(client: TestClient) -> None:
    labels = {"method": "POST", "endpoint": "/predict"}
    counter = "api_request_duration_seconds_count"
    before = sample_value(scrape(client), counter, **labels)

    client.post("/predict", json={"text": SAMPLE_TEXT})

    assert sample_value(scrape(client), counter, **labels) == before + 1


def test_buckets_resolve_sub_millisecond_latency(client: TestClient) -> None:
    """Both measured regimes need boundaries, or the quantiles flatten out.

    Sequential inference is ~0.55 ms P50 (reports/latency_baseline.json); under
    concurrent load the end-to-end handler sits at ~3.8 ms mean and ~7.5 ms
    P99. With the client library's 5 ms default lower bound the first regime
    would fall entirely in one bucket and P50, P95 and P99 would all report the
    same number.
    """
    sub_millisecond = [edge for edge in LATENCY_BUCKETS if edge < 0.001]
    one_to_ten_ms = [edge for edge in LATENCY_BUCKETS if 0.001 <= edge <= 0.01]

    assert len(sub_millisecond) >= 3
    assert len(one_to_ten_ms) >= 6
    assert min(LATENCY_BUCKETS) <= 0.00025

    client.post("/predict", json={"text": SAMPLE_TEXT})
    body = scrape(client)
    below_one_ms = sample_value(
        body,
        "api_request_duration_seconds_bucket",
        method="POST",
        endpoint="/predict",
        le="0.001",
    )
    assert below_one_ms > 0, "no request landed under 1 ms; buckets are miscalibrated"


def test_scrapes_do_not_count_themselves(client: TestClient) -> None:
    """Otherwise a 5 s scrape interval dominates the request-rate panel."""
    scrape(client)
    body = scrape(client)

    assert sample_value(body, "api_requests_total", endpoint="/metrics") == 0.0


def test_unhandled_exception_is_counted_as_500() -> None:
    """A crash still reaches the client as a 500, so it must reach the panel.

    Exercised against a throwaway app rather than the real one: the point is
    the middleware's failure path, and the service has no route that raises.
    """
    app = FastAPI()
    app.middleware("http")(metrics_middleware)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("deliberate failure")

    @app.get("/metrics")
    def metrics():
        return metrics_response()

    with TestClient(app, raise_server_exceptions=False) as failing_client:
        before = sample_value(
            scrape(failing_client),
            "api_requests_total",
            endpoint="/boom",
            status="500",
        )

        assert failing_client.get("/boom").status_code == 500

        body = scrape(failing_client)
        assert (
            sample_value(body, "api_requests_total", endpoint="/boom", status="500")
            == before + 1
        )
        # The latency of a failed request is data too: a crash after a long
        # wait looks nothing like a crash on entry.
        assert (
            sample_value(body, "api_request_duration_seconds_count", endpoint="/boom")
            == before + 1
        )


def test_middleware_reraises_so_the_client_still_sees_the_error() -> None:
    """Counting must not swallow the exception."""
    app = FastAPI()
    app.middleware("http")(metrics_middleware)

    @app.get("/raises")
    def raises() -> None:
        raise RuntimeError("deliberate failure")

    with TestClient(app) as strict_client, pytest.raises(RuntimeError):
        strict_client.get("/raises")
