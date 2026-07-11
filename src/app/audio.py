from __future__ import annotations

import base64
import json as _json
import logging
import struct
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings

logger = logging.getLogger(__name__)

_MEMORY_CACHE: dict[str, tuple[bytes, str]] = {}
_INTERIM_PHRASES = {
    "es": "Revisando...",
    "en": "Checking...",
    "fr": "Je vérifie...",
    "pt": "Verificando...",
    "zh": "正在查看...",
    "de": "Ich prüfe...",
    "it": "Controllo...",
    "ja": "確認中...",
    "ko": "확인 중...",
    "ru": "Проверяю...",
    "ar": "جارٍ التحقق...",
    "hi": "जांच हो रही है...",
    "nl": "Even controleren...",
    "pl": "Sprawdzam...",
    "tr": "Kontrol ediyorum...",
    "sv": "Kontrollerar...",
    "da": "Tjekker...",
    "fi": "Tarkistan...",
    "no": "Sjekker...",
    "cs": "Kontroluji...",
    "ro": "Verific...",
    "hu": "Ellenőrzöm...",
    "th": "กำลังตรวจสอบ...",
    "vi": "Đang kiểm tra...",
    "id": "Memeriksa...",
    "ms": "Sedang memeriksa...",
    "uk": "Перевіряю...",
    "he": "בודק...",
    "el": "Έλεγχος...",
}
_DEFAULT_LANG = "en"


def _interim_path(settings: Settings, lang: str) -> Path:
    return settings.audio_temp_dir / f"interim_{lang}.wav"


def _load_from_disk(path: Path) -> tuple[bytes, str] | None:
    if path.exists():
        return path.read_bytes(), "audio/wav"
    return None


def _save_to_disk(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    return header + pcm_data


async def transcribe_audio(
    settings: Settings,
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
) -> tuple[str, str]:
    model = ChatGoogleGenerativeAI(
        model=settings.stt_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe this audio. Return ONLY a JSON object with keys: "
                        '"text" (the transcription) and "language" (ISO 639-1 code). '
                        "No other text."
                    ),
                },
                {"type": "audio", "base64": audio_b64, "mime_type": mime_type},
            ],
        }
    ]
    response = await model.ainvoke(messages)
    raw = response.text if hasattr(response, "text") else str(response.content)
    if isinstance(raw, list):
        raw = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in raw
        )

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()
        data = _json.loads(clean)
        text = str(data.get("text", "")).strip()
        language = str(data.get("language", _DEFAULT_LANG)).strip().lower()
        return text, language
    except Exception:
        logger.warning("STT did not return valid JSON, using raw text: %s", raw[:200])
        return raw.strip(), _DEFAULT_LANG


async def synthesize_speech(
    settings: Settings,
    text: str,
) -> tuple[bytes, str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.tts_model,
        contents=f"Say this exactly: {text}",
        config=types.GenerateContentConfig(
            response_modalities=[types.Modality.AUDIO],
        ),
    )
    for part in response.parts or []:
        if part.inline_data and part.inline_data.data:
            raw = part.inline_data.data
            mime = getattr(part.inline_data, "mime_type", "") or ""
            logger.info("TTS audio mime_type=%s bytes=%s", mime, len(raw))
            if "pcm" in mime.lower() or "L16" in mime:
                return _pcm_to_wav(raw), "audio/wav"
            return raw, mime or "audio/wav"
    return b"", "audio/wav"


async def get_interim_audio(settings: Settings, language: str) -> tuple[bytes, str] | None:
    lang = language.strip().lower() or _DEFAULT_LANG

    if lang in _MEMORY_CACHE:
        return _MEMORY_CACHE[lang]

    path = _interim_path(settings, lang)
    cached = _load_from_disk(path)
    if cached is not None:
        _MEMORY_CACHE[lang] = cached
        return cached

    phrase = _INTERIM_PHRASES.get(lang)
    if phrase is None:
        phrase = await _translate_interim(settings, lang)

    audio, mime = await synthesize_speech(settings, phrase)
    if audio:
        _save_to_disk(path, audio)
        result = (audio, mime)
        _MEMORY_CACHE[lang] = result
        return result
    return None


async def _translate_interim(settings: Settings, lang: str) -> str:
    model = ChatGoogleGenerativeAI(
        model=settings.deepagent_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )
    response = await model.ainvoke(
        f'Translate the word "Checking..." to the language with ISO 639-1 code "{lang}". '
        "Return ONLY the translation, nothing else."
    )
    text = response.text if hasattr(response, "text") else str(response.content)
    if isinstance(text, list):
        text = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in text
        )
    phrase = text.strip() or _INTERIM_PHRASES[_DEFAULT_LANG]
    logger.info("Auto-generated interim phrase for lang=%s: %s", lang, phrase)
    return phrase
