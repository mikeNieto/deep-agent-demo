# AGENTS.md

## Stack

- **Python 3.13**, **uv** package manager (not pip)
- **FastAPI** backend with WebSockets
- **Deep Agents** framework for agent orchestration
- **Google Gemini** for LLM + STT (multimodal audio)
- **Google Cloud Text-to-Speech (WaveNet)** for TTS via REST API
- **Google Search grounding** via `google-genai` SDK in the `web_search` tool
- SQLite checkpointing via `langgraph-checkpoint-sqlite` (`AsyncSqliteSaver`)

## Entrypoints

| Command | What |
|---|---|
| `uv run agent-api` | FastAPI server (`app.main:run`) |
| `uv run pytest` | Run all tests |
| `docker compose up` | Run via Docker (builds from Dockerfile, mounts `data/` as volume) |

## Project map

```
src/
  app/
    main.py       -- FastAPI + WebSocket handler + HTML serving + lifespan
    config.py     -- pydantic-settings, reads .env
    agent.py      -- Deep Agent graph, system prompt, tools, ChatGoogleGenerativeAI
    audio.py      -- STT (Gemini multimodal) + TTS (Cloud TTS via REST) + markdown cleanup
    usage.py      -- UsageTracker (persistent JSON metrics)
static/
  index.html      -- Web client (vanilla JS + WebSocket + Web Audio API)
memory/
  AGENTS.md       -- Loaded as deepagents agent runtime memory (NOT OpenCode instructions)
skills/
  assistant/
    SKILL.md      -- deepagents skill for conversational style
data/
  audio/          -- TTS output, audio uploads, interim cache (gitignored)
  sqlite/         -- Agent conversation checkpoints (gitignored)
  usage.json      -- Usage metrics (gitignored)
  files/          -- Agent virtual filesystem backend (runtime, gitignored)
tests/
  test_main.py    -- FastAPI TestClient tests
  test_agent.py   -- Tool unit tests (BTC price hits live API)
  test_audio.py   -- STT/TTS unit tests (all mocked)
```

## Config

All env vars in `.env` (see `.env.example`). Key ones:

- `GOOGLE_API_KEY` — required for Gemini (LLM + STT); without it the agent graph is not created
- `GOOGLE_CLOUD_API_KEY` — required for TTS if Gemini uses an AI Studio key. Falls back to `GOOGLE_API_KEY`
- `DEEPAGENT_MODEL` — LLM model (default: `gemini-2.5-flash`)
- `STT_MODEL` — STT model (default: `gemini-2.5-flash`)
- `TTS_VOICE_EN` / `TTS_VOICE_ES` — WaveNet voice names (default: `en-US-Wavenet-F` / `es-ES-Wavenet-C`)
- `THINKING_LEVEL` — Gemini 3+ thinking level (MINIMAL, LOW, MEDIUM, HIGH)
- `THINKING_BUDGET` — Gemini 2.5 thinking budget (integer tokens, 0 to disable)
- `API_HOST` / `API_PORT` — server bind (default: `0.0.0.0:8000`)
- `AUDIO_TEMP_DIR` — audio temp dir (default: `data/audio`)

## Architecture notes

- Agent graph uses `deepagents.create_deep_agent` with `ChatGoogleGenerativeAI`, `CompositeBackend` (FilesystemBackend in virtual_mode), and `AsyncSqliteSaver`
- Chat model created in `create_chat_model()` which auto-detects Gemini version for thinking_level vs thinking_budget
- `web_search` tool uses `google-genai` SDK directly (not via LangChain) for Google Search grounding
- WebSocket flow: audio chunks (PCM 16-bit 16kHz mono WAV) -> STT (Gemini multimodal, returns JSON with text + language) -> Agent -> TTS (Cloud Text-to-Speech REST API) -> WAV audio back
- TTS response is cleaned with `_clean_for_tts()` — strips markdown, emojis, code blocks, links before sending to TTS
- No conversation confirmation step for audio input
- Agent responses auto-detect user language; system prompt forbids markdown, emojis, greetings, farewells
- STT transcription rejects known error phrases (e.g., "unable to", "cannot fulfill", "only 00:00:00")
- Interim "Checking..." audio is played during tool calls (cached in memory + disk by language)
- Agent tools: `get_current_datetime` (Colombia TZ), `get_current_bitcoin_price` (Binance), `web_search`

## API routes

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/` | Web client (HTML) |
| GET | `/static/*` | Static files (mounted if `static/` exists) |
| WS | `/ws` | WebSocket endpoint for audio + text |

## Tests

- `uv run pytest` — runs all tests with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed for async fixtures automatically, though tests use the marker explicitly)
- `TestClient` from `fastapi.testclient` (re-exports starlette's)
- `test_bitcoin_price_tool_is_async` hits the live Binance API — no mocking
- All STT/TTS tests use `patch` / `monkeypatch` on `ChatGoogleGenerativeAI` and `httpx.AsyncClient`
- Use `pytest -k PATTERN` to filter, e.g. `uv run pytest -k tts`

## Conventions

- `memory/AGENTS.md` is agent runtime memory (loaded by deepagents), NOT OpenCode instructions
- `skills/` contains deepagents skills (loaded at agent init)
- Audio files stored in `data/audio/`
- No formatters or linters configured in pyproject.toml
- `uv.lock` is checked in; use `uv sync` to install, `uv add` / `uv remove` for deps
- Imports use `app.*` (not `src.app.*`); uv makes `src/` importable automatically
- Docker build uses `uv sync --frozen` for reproducible builds; `AGENTS.md` is excluded from Docker image via `.dockerignore`
