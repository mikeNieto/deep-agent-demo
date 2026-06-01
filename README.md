## Conversational Agent MVP

MVP de un agente conversacional con `FastAPI`, `Deep Agents`, `Streamlit`, STT local con `faster-whisper` y TTS local con `kokoro-onnx`.

### Requisitos

- Python 3.13
- `uv`
- `ffmpeg` instalado en el sistema para exportar MP3 con `pydub`
- `OPENROUTER_API_KEY` configurada

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
