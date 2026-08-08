"""Generate realistic traffic against the API so the dashboards have data.

Without a load generator the Grafana panels are empty during the demo, and an
empty panel proves nothing about the instrumentation. The mix is deliberate:
valid abstracts drawn from all five classes so the prediction-distribution
panel is not a single bar, plus a configurable share of malformed payloads so
the error-ratio panel has something above zero to plot.

Run with (stack already up)::

    uv run python scripts/generate_load.py
    uv run python scripts/generate_load.py --duration 300 --rps 40 --error-rate 0.1
    uv run python scripts/generate_load.py --requests 5000 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from collections import Counter

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_DURATION_S = 120.0
DEFAULT_RPS = 25.0
DEFAULT_CONCURRENCY = 8
DEFAULT_ERROR_RATE = 0.08
DEFAULT_HEALTH_RATE = 0.05

# Two abstracts per class, so every one of the five prediction counters moves
# and the distribution panel shows a real spread instead of one dominant bar.
# Wording follows the corpus register (condensed clinical abstracts).
VALID_TEXTS: tuple[str, ...] = (
    # neoplasms
    "Adjuvant chemotherapy following radical mastectomy for invasive ductal "
    "carcinoma. Tumour staging and lymph node involvement were reviewed in a "
    "cohort of patients with malignant breast neoplasm over five years.",
    "Prognostic significance of histologic grade in metastatic melanoma. "
    "Biopsy specimens from patients with cutaneous malignancy were graded and "
    "correlated with survival after surgical excision of the primary tumour.",
    # digestive system diseases
    "Endoscopic evaluation of chronic gastric ulcer disease. Mucosal biopsies "
    "from patients with recurrent epigastric pain showed duodenal inflammation "
    "and Helicobacter colonisation of the gastric antrum.",
    "Hepatic function after portal decompression in cirrhotic patients. Serum "
    "bilirubin and hepatic enzyme levels were monitored following portosystemic "
    "shunt surgery for variceal bleeding.",
    # nervous system diseases
    "Cerebral infarction and seizure activity following carotid endarterectomy. "
    "Neurological deficits were assessed postoperatively in patients undergoing "
    "surgery for high grade internal carotid artery stenosis.",
    "Electroencephalographic findings in refractory temporal lobe epilepsy. "
    "Seizure frequency and cortical lesion localisation were compared before "
    "and after anticonvulsant therapy in adult patients.",
    # cardiovascular diseases
    "Coronary artery bypass grafting outcomes in hypertensive patients. "
    "Perioperative myocardial infarction rates were compared across patients "
    "with and without preexisting left ventricular hypertrophy.",
    "Thrombolytic therapy in acute myocardial infarction. Left ventricular "
    "ejection fraction and coronary patency were measured after intravenous "
    "streptokinase administration in the emergency department.",
    # general pathological conditions
    "Inflammatory response and wound healing after elective abdominal surgery. "
    "Serum markers of systemic inflammation were followed in patients during "
    "the postoperative recovery period.",
    "Long term follow up of chronic pain syndrome in a general clinical "
    "population. Symptom severity, functional status and treatment response "
    "were recorded at six month intervals.",
)

# Every entry is rejected with 422 by the request model, each through a
# different validation path: length, whitespace-only, missing field, wrong
# type, and the MAX_TEXT_CHARS ceiling.
INVALID_PAYLOADS: tuple[dict, ...] = (
    {"text": ""},
    {"text": "   "},
    {"text": "\n\t  \n"},
    {},
    {"text": 42},
    {"text": "a" * 25_000},
)


def bucket_of(status: int) -> str:
    """Group a status code the way the error-ratio panel does."""
    return f"{status // 100}xx"


async def send_one(
    client: httpx.AsyncClient, rng: random.Random, args: argparse.Namespace
) -> tuple[str, int]:
    """Issue a single request and return (kind, status code)."""
    roll = rng.random()

    if roll < args.health_rate:
        response = await client.get("/health")
        return "health", response.status_code

    if roll < args.health_rate + args.error_rate:
        payload = rng.choice(INVALID_PAYLOADS)
        response = await client.post("/predict", json=payload)
        return "invalid", response.status_code

    response = await client.post("/predict", json={"text": rng.choice(VALID_TEXTS)})
    return "valid", response.status_code


async def worker(
    name: int,
    queue: asyncio.Queue[None],
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tally: Counter,
) -> None:
    """Drain scheduled slots from the queue until it is closed."""
    # Per-worker seed derived from the run seed: the mix stays reproducible
    # without every worker replaying the identical sequence.
    rng = random.Random((args.seed + name) if args.seed is not None else None)

    while True:
        try:
            await queue.get()
        except asyncio.CancelledError:
            return
        try:
            kind, status = await send_one(client, rng, args)
            tally[f"{kind}:{bucket_of(status)}"] += 1
            tally["total"] += 1
        except httpx.HTTPError as error:
            # A connection failure is a result too: it usually means the API
            # container is not up yet, and silently retrying would hide that.
            tally[f"transport:{type(error).__name__}"] += 1
            tally["total"] += 1
        finally:
            queue.task_done()


async def schedule(queue: asyncio.Queue[None], args: argparse.Namespace) -> int:
    """Feed the queue at the target rate; return how many slots were issued.

    Paced against a monotonic clock rather than ``sleep(1/rps)`` per request,
    so the achieved rate does not drift downward by the cost of each loop.
    """
    interval = 1.0 / args.rps
    started = time.monotonic()
    issued = 0

    while True:
        if args.requests is not None and issued >= args.requests:
            return issued
        elapsed = time.monotonic() - started
        if args.requests is None and elapsed >= args.duration:
            return issued

        await queue.put(None)
        issued += 1

        drift = started + issued * interval - time.monotonic()
        if drift > 0:
            await asyncio.sleep(drift)


async def run(args: argparse.Namespace) -> Counter:
    """Drive the configured load and return the tally by outcome."""
    tally: Counter = Counter()
    queue: asyncio.Queue[None] = asyncio.Queue(maxsize=args.concurrency * 4)

    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(
        base_url=args.url, timeout=args.timeout, limits=limits
    ) as client:
        workers = [
            asyncio.create_task(worker(index, queue, client, args, tally))
            for index in range(args.concurrency)
        ]
        started = time.monotonic()
        issued = await schedule(queue, args)
        await queue.join()
        tally["elapsed_s"] = round(time.monotonic() - started, 2)
        tally["issued"] = issued

        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    return tally


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixed traffic against the triage API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the API.")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Seconds to keep sending traffic. Ignored when --requests is set.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Send exactly this many requests instead of running for --duration.",
    )
    parser.add_argument(
        "--rps", type=float, default=DEFAULT_RPS, help="Target requests per second."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Requests allowed in flight at once.",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=DEFAULT_ERROR_RATE,
        help="Share of requests sent with a payload that fails validation (422).",
    )
    parser.add_argument(
        "--health-rate",
        type=float,
        default=DEFAULT_HEALTH_RATE,
        help="Share of requests sent to /health instead of /predict.",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the payload mix for a reproducible run.",
    )

    args = parser.parse_args(argv)

    if args.rps <= 0:
        parser.error("--rps must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if not 0.0 <= args.error_rate <= 1.0:
        parser.error("--error-rate must be between 0 and 1")
    if not 0.0 <= args.health_rate <= 1.0:
        parser.error("--health-rate must be between 0 and 1")
    if args.error_rate + args.health_rate > 1.0:
        parser.error("--error-rate plus --health-rate must not exceed 1")

    return args


def report(tally: Counter) -> None:
    """Print the outcome breakdown to stdout."""
    total = tally.pop("total", 0)
    elapsed = tally.pop("elapsed_s", 0.0)
    tally.pop("issued", 0)

    print(f"\nsent {total} requests in {elapsed}s", end="")
    if elapsed:
        print(f" ({total / elapsed:.1f} req/s achieved)")
    else:
        print()

    for key in sorted(tally):
        print(f"  {key:<28} {tally[key]}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"target: {args.url}")
    if args.requests is not None:
        print(f"plan:   {args.requests} requests at ~{args.rps} req/s")
    else:
        print(f"plan:   {args.duration}s at ~{args.rps} req/s")
    print(
        f"mix:    {1 - args.error_rate - args.health_rate:.0%} valid /predict, "
        f"{args.error_rate:.0%} invalid /predict (422), "
        f"{args.health_rate:.0%} /health"
    )

    try:
        tally = asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    report(tally)

    transport_failures = sum(
        count for key, count in tally.items() if key.startswith("transport:")
    )
    if transport_failures:
        print(
            f"\n{transport_failures} request(s) never reached the API. "
            f"Is the stack up? docker compose -f docker-compose.monitoring.yml ps",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
