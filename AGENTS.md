# AGENTS.md

## Stack

- **Python 3.13**, **uv** package manager (not pip)
- **FastAPI** backend with WebSockets
- **Deep Agents** framework for agent orchestration
- All AI services via **Google Gemini** (LLM, STT, TTS, Grounding)

## Entrypoints

| Command | What |
|---|---|
| `uv run agent-api` | FastAPI server (app.main:run) |
| `uv run pytest` | Run all tests |

## Project map

```
src/
  app/
    main.py       -- FastAPI + WebSocket handler + HTML serving + lifespan
    config.py     -- pydantic-settings, reads .env
    agent.py      -- GeminiModel, Google Search grounding, tools, prompts
    audio.py      -- STT (Gemini multimodal) + TTS (Gemini native)
static/
  index.html      -- Web client (vanilla JS + WebSocket + Web Audio API)
memory/           -- AGENTS.md loaded as deepagents agent memory at runtime
skills/           -- deepagents skills loaded at runtime
data/
  audio/          -- TTS output, audio uploads (gitignored)
```

## Config

All env vars in `.env` (see `.env.example`). Key ones:

- `GOOGLE_API_KEY` -- required; without it the agent graph is not created
- `DEEPAGENT_MODEL` -- LLM model (default: `gemini-2.5-flash`)
- `STT_MODEL` -- STT via Gemini multimodal (default: `gemini-2.5-flash`)
- `TTS_MODEL` -- TTS via Gemini native audio (default: `gemini-2.5-flash-preview-tts`)
- `THINKING_LEVEL` -- Gemini 3+ thinking level (MINIMAL, LOW, MEDIUM, HIGH)
- `THINKING_BUDGET` -- Gemini 2.5 thinking budget (integer tokens, 0 to disable)

## Architecture notes

- Agent graph uses `deepagents.create_deep_agent` with `GeminiModel` (ChatGoogleGenerativeAI subclass)
- `GeminiModel` overrides `bind_tools()` to always inject Google Search grounding
- WebSocket flow: audio chunks -> STT (Gemini multimodal) -> Agent -> TTS (Gemini native) -> audio back
- No TTS preprocessing (agent response is sent directly to TTS, same language as user)
- No conversation confirmation step for audio input
- Audio: PCM 16-bit 16kHz mono (WAV) for ESP32-S3 compatibility

## API routes

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/` | Web client (HTML) |
| WS | `/ws` | WebSocket endpoint for audio + text |

## Tests

- `uv run pytest` runs all tests
- Tests use `TestClient` (FastAPI)

## Conventions

- `memory/AGENTS.md` is agent runtime memory (loaded by deepagents), not OpenCode instructions
- `skills/` contains deepagents skills (loaded at agent init)
- Audio files live in `data/audio/`
- No formatters or linters configured in pyproject.toml
- `uv.lock` is checked in; use `uv sync` to install, `uv add` / `uv remove` for deps
- Imports use `app.*` (not `src.app.*`); uv makes `src/` importable automatically
