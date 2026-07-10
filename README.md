# Gemini 3.1 Flash Live — Test

Real-time voice chat using Google's Gemini 3.1 Flash Live model via WebSocket.

## Setup

```bash
uv sync
```

Add your Google API key to `.env`:

```bash
echo 'GOOGLE_API_KEY=your-key-here' >> .env
```

Or copy `.env.example` and fill in the key.

## Run

```bash
uv run live-api
```

Opens at **http://localhost:8000** — a single-page app served from the same server.

- Click **Connect** to establish the WebSocket + Gemini Live session
- Click **Start Mic** and speak — Gemini responds with audio
- Or type text and press **Send**
- Speak while Gemini is talking to interrupt (voice activity detection)

## WebSocket API

External clients connect to `ws://localhost:8000/api/live/ws`.

### Messages

| Type | Direction | Description |
|---|---|---|
| `setup` | C→S | Must be first. `{"type":"setup","system_instruction":"..."}` |
| `audio_config` | C→S | Mic sample rate. `{"type":"audio_config","sample_rate":48000}` |
| `audio_in` | C→S | Audio chunk. `{"type":"audio_in","data":"<base64 pcm>"}` |
| `text_in` | C→S | Text message. `{"type":"text_in","text":"Hello"}` |
| `audio_out` | S→C | Audio response. `{"type":"audio_out","data":"<base64 pcm24k>"}` |
| `text` | S→C | Transcript. `{"type":"text","role":"user\|model"[,"interim":true],"content":"..."}` |
| `status` | S→C | `{"type":"status","state":"idle"\|"speaking"}` |
| `error` | S→C | `{"type":"error","message":"..."}` |

### Test with wscat

```bash
wscat -c ws://localhost:8000/api/live/ws
# Send:
{"type":"setup"}
{"type":"text_in","text":"Hola"}
```

## Docker

```bash
# Build and run with compose
docker compose up -d --build

# Or standalone
docker build -t gemini-live .
docker run -d -p 8000:8000 --env-file .env gemini-live
```

Opens at **http://localhost:8000**.

## Configuration

| Env var | Default |
|---|---|
| `GOOGLE_API_KEY` | — |
| `GEMINI_LIVE_MODEL` | `gemini-3.1-flash-live-preview` |
| `API_HOST` | `0.0.0.0` |
| `API_PORT` | `8000` |
