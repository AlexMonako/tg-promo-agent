FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

# Railway mounts a persistent volume at /data when configured.
ENV STATE_DB_PATH=/data/state.sqlite3 \
    CONFIG_PATH=/app/config.yaml \
    PORT=8080

# Default config is shipped but should be overridden via volume mount.
COPY config.example.yaml /app/config.yaml

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["python", "-m", "tg_promo_agent.main"]
