## SYLYS — Conversational Agent

Agente conversacional con **FastAPI + WebSockets**, **Deep Agents** y **Google Gemini** (LLM, STT, TTS, Google Search Grounding).

### Requisitos

- Python 3.13
- `uv`
- `GOOGLE_API_KEY` configurada (Gemini)

### Variables de entorno

Crear `.env` basado en `.env.example`.

### Instalacion

```bash
uv sync
```

### Ejecutar

```bash
uv run agent-api
```

Abre `http://localhost:8000` en el navegador.

### Docker

```bash
docker compose up -d
```

Servicio unico en `http://localhost:8000`.

### Tests

```bash
uv run pytest
```
