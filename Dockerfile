# Three stages so the runtime image carries neither the build tooling nor the
# training libraries: 'deps' resolves the serving dependencies, 'trainer'
# produces the model artifact, 'runtime' assembles only those two outputs.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.9.9

# --------------------------------------------------------------------------
# Stage 1: serving dependencies only (no pandas, pyarrow or hub client)
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups

# --------------------------------------------------------------------------
# Stage 2: download the pinned corpus and train the baseline
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS trainer

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/build

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group train

COPY src/ ./src/
# Fetches the pinned dataset revision, verifies checksums, then fits the
# pipeline. Makes the image self-contained: no artifact has to exist on the
# host before `docker build`.
RUN python -m src.data.prepare && python -m src.models.baseline

# --------------------------------------------------------------------------
# Stage 3: runtime
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/baseline.joblib \
    PORT=8000

RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --from=trainer /build/models/baseline.joblib /app/models/baseline.joblib
COPY src/ /app/src/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Readiness, not just liveness: a process that is up but could not load the
# model reports model_loaded=false and must fail the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "\
import json, sys, urllib.request; \
r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); \
sys.exit(0 if r.status == 200 and json.load(r)['model_loaded'] else 1)"

# A shell is needed so ${PORT} expands when a platform injects it (Cloud Run,
# Azure Container Apps); 'exec' then replaces it so uvicorn is PID 1 and
# receives SIGTERM directly on shutdown.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
