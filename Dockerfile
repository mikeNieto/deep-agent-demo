FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY static/ ./static/

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "live-api"]
