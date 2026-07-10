# AGENTS.md

## Commands

```bash
uv sync                    # install deps
uv run live-api            # start server (port 8000)
uv run pytest tests/ -v    # run tests
docker compose up -d --build  # deploy
```

## Architecture

Single FastAPI process. No frontend build step — `static/index.html` is served directly. WebSocket endpoint at `/api/live/ws`. Gemini Live SDK session runs in a `TaskGroup` with concurrent sender/receiver tasks.

## Google Gemini Live SDK gotchas

These are all non-obvious from the SDK API surface:

1. `client.aio.live.connect()` is an `@asynccontextmanager` — use `async with`, **NOT** `await`.
2. `session.receive()` is an async generator — use `async for`, **NOT** `await`.
3. `session.receive()` is exhausted after each turn. Wrap in `while True` to get a fresh iterator per turn.
4. `send_client_content` and `receive().__anext__()` must be called from the **same** asyncio task. Use `asyncio.TaskGroup()` to run sender and receiver in separate tasks.
5. `send_client_content(text, turn_complete=True)` — the old `send()` is deprecated.
6. `send_realtime_input(audio={...})` — for audio chunks.

## Model constraints

- `gemini-3.1-flash-live-preview` only supports `response_modalities=["AUDIO"]`. TEXT modality is rejected.
- Transcription is NOT in `response.text`. It's in:
  - `response.server_content.output_transcription.text` (model speech)
  - `response.server_content.input_transcription.text` (user speech)
  - `response.server_content.interim_input_transcription.text` (partial user speech)
- Enable both with `input_audio_transcription={}` and `output_audio_transcription={}` in config.

## WebSocket protocol

JSON messages over `/api/live/ws`:

```
C→S: setup, audio_config, audio_in, text_in
S→C: audio_out (pcm24k base64), text (role+content), status (idle/speaking), error
```

## Dependencies (minimal)

`fastapi`, `uvicorn[standard]`, `google-genai`, `pydantic-settings`. No streamlit, no pandas, no deepagents.
