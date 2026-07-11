## SYLYS — Conversational Agent

Agente conversacional con **FastAPI + WebSockets**, **Deep Agents** y **Google Gemini** (LLM, STT, Google Search Grounding).
TTS via **Google Cloud Text-to-Speech (WaveNet)** con 1 millon de caracteres gratis por mes.

### Requisitos

- Python 3.13
- `uv`
- `GOOGLE_API_KEY` configurada (Gemini)
- `GOOGLE_CLOUD_API_KEY` (opcional, para TTS si GOOGLE_API_KEY es de AI Studio)

### Configuracion de API Keys

Gemini (LLM y STT) funciona con keys de **Google AI Studio** o **Google Cloud Console**.
Cloud Text-to-Speech (WaveNet) **solo funciona con keys de Google Cloud Console**.

**Caso 1: key de Google Cloud Console para todo (recomendado)**
```env
GOOGLE_API_KEY=tu-key-de-cloud-console
```
La misma key funciona para Gemini y Cloud TTS. Asegurate de que la key no tenga restricciones
de API, o que permita tanto `Generative Language API` como `Cloud Text-to-Speech API`.

**Caso 2: key de AI Studio + key de Cloud Console**
```env
GOOGLE_API_KEY=tu-key-de-ai-studio
GOOGLE_CLOUD_API_KEY=tu-key-de-cloud-console
```
Si tu key de Gemini es de AI Studio (no funciona con Cloud APIs), usa `GOOGLE_CLOUD_API_KEY`
para TTS. Si `GOOGLE_CLOUD_API_KEY` no esta definida, TTS usa `GOOGLE_API_KEY` como fallback.

### Habilitar Google Cloud Text-to-Speech

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un proyecto (o usa uno existente)
3. Habilita **billing** (obligatorio, aunque uses el tier gratuito)
4. Ve a **APIs & Services > Library** y habilita **Cloud Text-to-Speech API**
5. Ve a **APIs & Services > Credentials** y crea una **API Key**
6. Copia la key en tu `.env` como `GOOGLE_CLOUD_API_KEY` (o `GOOGLE_API_KEY` si usas Caso 1)

### Variables de entorno

Crear `.env` basado en `.env.example`.

| Variable | Default | Descripcion |
|---|---|---|
| `GOOGLE_API_KEY` | — | API key para Gemini (LLM + STT) |
| `GOOGLE_CLOUD_API_KEY` | — | API key para Cloud TTS. Si no se define, usa `GOOGLE_API_KEY` |
| `DEEPAGENT_MODEL` | `gemini-2.5-flash` | Modelo del agente |
| `STT_MODEL` | `gemini-2.5-flash` | Modelo de STT |
| `TTS_VOICE_EN` | `en-US-Wavenet-F` | Voz WaveNet en ingles (femenina) |
| `TTS_VOICE_ES` | `es-ES-Wavenet-C` | Voz WaveNet en espanol (femenina) |
| `THINKING_LEVEL` | — | Nivel de thinking (MINIMAL, LOW, MEDIUM, HIGH) |
| `THINKING_BUDGET` | `0` | Budget de tokens para thinking (Gemini 2.5) |

### Voces WaveNet disponibles

Las voces WaveNet tienen pricing estandar de $16/millon de caracteres.
Los primeros **1 millon de caracteres por mes son gratis**.

Voces femeninas en ingles (US):
- `en-US-Wavenet-C` — media, natural
- `en-US-Wavenet-E` — aguda, expresiva
- `en-US-Wavenet-F` — grave, suave *(default)*
- `en-US-Wavenet-G` — juvenil
- `en-US-Wavenet-H` — neutra

Voces femeninas en espanol (Espana):
- `es-US-Wavenet-A` — femenina, media latina
- `es-ES-Wavenet-C` — femenina, natural *(default)*
- `es-ES-Wavenet-D` — femenina, alternativa
- `es-ES-Wavenet-A` — femenina, clasica

Para cambiar la voz, edita `TTS_VOICE_EN` o `TTS_VOICE_ES` en tu `.env`.
Lista completa de voces: [Google Cloud TTS Voices](https://cloud.google.com/text-to-speech/docs/voices)
