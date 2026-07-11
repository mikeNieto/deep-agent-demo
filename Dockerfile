FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY memory/ ./memory/
COPY skills/ ./skills/
COPY src/ ./src/
COPY static/ ./static/

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/data/audio

EXPOSE 8000

CMD ["uv", "run", "agent-api"]
