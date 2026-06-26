FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (sync without dev deps)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY memory/ ./memory/
COPY skills/ ./skills/
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/

# Install the project itself
RUN uv sync --frozen --no-dev

# Create data directories
RUN mkdir -p /app/data/sqlite /app/data/audio

EXPOSE 8000
EXPOSE 8501

# Default command: FastAPI server
CMD ["uv", "run", "agent-api"]
