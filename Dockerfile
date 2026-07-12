FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Dependency layer first: uv sync needs only the project metadata + src (the wheel target), so
# editing api/ or data/ must not bust this expensive layer (REVIEW_ISSUES.md REV-15).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Only the serve extra (fastapi/uvicorn) is needed by the scoring API. Do NOT use --all-extras:
# the platform extras (streaming/featurestore/orchestration/tracking/dashboard) are Phase 2-6
# infra and would bloat this image + risk native-dep build failures the API never uses.
RUN uv sync --extra serve --no-dev

COPY api ./api
COPY data ./data
COPY churn_prediction.py ./

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Train + register a model at build time so the image is self-contained: a bare
# `docker run` serves /score with no mounted volume or external model store. The
# model registry (models/) is .dockerignore'd from the build *context*, but this
# RUN writes a fresh run *inside* the image, so /health reports model_loaded=true
# out of the box. (A production setup would pull a versioned artifact from a model
# store; baking one in is the right call for a self-contained demo.)
RUN python -m churn.train

EXPOSE 8000

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
