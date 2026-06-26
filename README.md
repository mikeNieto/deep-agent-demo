## Conversational Agent MVP

MVP de un agente conversacional con `FastAPI`, `Deep Agents`, `Streamlit`, STT y TTS con OpenRouter.

### Requisitos

- Python 3.13
- `uv`
- `OPENROUTER_API_KEY` configurada (para chat, STT y TTS)

### Variables de entorno

Crear `.env` basado en `.env.example`.

### Instalacion

```bash
uv sync
```

### Ejecutar API

```bash
uv run agent-api
```

### Ejecutar Streamlit

```bash
uv run streamlit run streamlit_app/app.py
```

### Docker

```bash
# API + Streamlit
docker compose up -d

# Solo API
docker compose up -d api
```

Requisitos: Docker y `docker compose`. El archivo `.env` debe existir con `OPENROUTER_API_KEY` configurada.

Servicios:

| Servicio | Puerto | Acceso |
|---|---|---|
| `api` | `8000` | `http://localhost:8000` |
| `streamlit` | `8501` | `http://localhost:8501` |

### Tests

```bash
uv run pytest
```
