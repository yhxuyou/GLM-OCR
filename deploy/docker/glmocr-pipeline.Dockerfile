FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY glmocr/ glmocr/

RUN pip install --no-cache-dir -e ".[server,layout]" \
    && pip install --no-cache-dir \
        uvicorn \
        prometheus-client \
        structlog

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

ENV PYTHONUNBUFFERED=1
ENV GLMOCR_LOG_LEVEL=INFO

EXPOSE 5002

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:5002/health || exit 1

ENTRYPOINT ["python", "-m", "glmocr.production_server"]