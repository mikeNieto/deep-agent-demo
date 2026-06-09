## Conversational Agent MVP

MVP de un agente conversacional con `FastAPI`, `Deep Agents`, `Streamlit`, STT con `Gemini` y TTS con OpenRouter.

### Requisitos

- Python 3.13
- `uv`
- `OPENROUTER_API_KEY` configurada (para chat y TTS)
- `GEMINI_API_KEY` configurada (para STT)

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

### Tests

```bash
uv run pytest
```
