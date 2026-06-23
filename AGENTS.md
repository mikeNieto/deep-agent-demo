# AGENTS.md

## Stack

- **Python 3.13**, **uv** package manager (not pip)
- **FastAPI** backend, **Streamlit** frontend
- **Deep Agents** v0.6.8 (`deepagents`) framework for agent orchestration
- All AI services via **OpenRouter** (LLM, STT, TTS)

## Entrypoints

| Command | What |
|---|---|
| `uv run agent-api` | FastAPI server (app.main:run) |
| `uv run streamlit run streamlit_app/app.py` | Streamlit UI |
| `uv run pytest` | Run all tests |
| `uv run pytest tests/test_services.py` | Run a single test file |
| `uv run pytest -k "test_datetime"` | Run tests matching a pattern |

The `agent-streamlit` script in pyproject.toml is misleading — it calls `streamlit_app.app:main` directly, which does not start the Streamlit server. Always use `uv run streamlit run streamlit_app/app.py`.

## Project map

```
src/
  app/            -- FastAPI application
    main.py       -- app factory + lifespan
    config.py     -- pydantic-settings, reads .env
    api/          -- health, chat, audio routers
    agents/       -- Deep Agents graph, tools, prompts, memory
    services/     -- chat, conversation, STT, TTS, TTS-preparation
    schemas/      -- pydantic models
    storage/      -- checkpoint saver, file helpers
    utils/        -- ID generation, logging
  agent/          -- uv build shim (imports app.main:run)
streamlit_app/    -- Streamlit frontend (modifies sys.path to import from src/)
memory/           -- AGENTS.md loaded as deepagents agent memory at runtime
skills/           -- deepagents skills loaded at runtime
data/             -- audio uploads, TTS output, SQLite checkpoints (gitignored)
```

## Config

All env vars in `.env` (see `.env.example`). Key ones:

- `OPENROUTER_API_KEY` -- required; without it the agent graph is not created (chat returns 503)
- `DEEPAGENT_MODEL` -- LLM model (default: `deepseek/deepseek-v4-flash`)
- `STT_MODEL`, `TTS_MODEL`, `TTS_PREPARATION_MODEL`, `TTS_VOICE`

**Important:** `.env.example` values differ from code defaults in `config.py`. The `.env.example` values are the intended runtime config. For example, `TTS_MODEL` defaults to `hexgrad/kokoro-82m` in code but `.env.example` uses `microsoft/mai-voice-2`.

## Architecture notes

- Agent graph is built in `app/agents/factory.py` using `deepagents.create_deep_agent`
- Chat flow: `POST /api/chat/message` -> `ChatService.send_message()` -> agent `ainvoke` -> optional TTS preparation -> optional TTS synthesis
- TTS is **two-step**: preparation (OpenRouter LLM rewrites agent text to Spanish for speech) then synthesis (OpenRouter audio API)
- TTS synthesis has automatic **PCM fallback** for models that reject mp3 (e.g. Gemini via OpenRouter); runtime dep: `ffmpeg`
- Duration extraction uses `ffprobe` at runtime; gracefully degrades if missing
- STT uses OpenRouter chat completions with inline base64 audio (not a dedicated STT API)
- Conversation history is in-memory (`ConversationService`), not persisted
- Checkpoints use langgraph's `AsyncSqliteSaver` for thread state persistence
- Agent tools: `get_current_datetime` (Colombia timezone), `get_current_bitcoin_price` (hardcoded)
- Agent name: **SYLYS** (set in system prompt)

### Language contradiction

The system prompt (`app/agents/prompts.py`) says "Respond in English," but both `memory/AGENTS.md` and `skills/assistant/SKILL.md` say "Respond in Spanish." These are loaded into the agent at runtime and may override the system prompt. Test to confirm which language the agent actually uses before changing.

## API routes

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/health/models` | Model readiness status |
| POST | `/api/chat/message` | Send message, get response + optional audio |
| GET | `/api/conversations/{thread_id}` | Get conversation history |
| POST | `/api/audio/transcribe` | STT (upload audio file) |
| POST | `/api/audio/synthesize` | TTS (text -> MP3) |
| GET | `/api/audio/files/{filename}` | Serve generated audio |

## Tests

- `uv run pytest` runs all tests
- Tests use `TestClient` (FastAPI) and mocking (no external services required)
- `get_settings.cache_clear()` must be called after monkeypatching env vars, and again after the test (see `test_audio_api.py`)
- Chat service tests verify TTS errors degrade gracefully (response still returns with `audio_url=None`)
- Test functions that call `asyncio.run()` (not `async def`) — no pytest-asyncio mode needed despite the dependency
- No pytest config in pyproject.toml (no conftest.py, no markers)

## Conventions

- `memory/AGENTS.md` is agent runtime memory (loaded by deepagents), not OpenCode instructions
- `skills/` contains deepagents skills (loaded at agent init)
- Audio files (uploaded and generated) live in `data/audio/`
- No formatters or linters configured in pyproject.toml
- The `src/agent/__init__.py` is a compatibility shim, not functional code
- `uv.lock` is checked in; use `uv sync` to install, `uv add` / `uv remove` for deps
- Imports use `app.*` (not `src.app.*`); uv makes `src/` importable automatically
