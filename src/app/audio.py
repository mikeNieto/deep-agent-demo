from __future__ import annotations

import base64
import json as _json
import logging
import re
import struct
from pathlib import Path

import httpx
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


_EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\U0000FE0F" "\U0000200D"
    "]+"
)


def _clean_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)                                 # emojis
    text = re.sub(r" {2,}", " ", text)                              # collapsed spaces
    text = re.sub(r"!\[.+?\]\(.+?\)", "", text)                  # images
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)               # links
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # bold
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"\1", text)         # italic
    text = re.sub(r"__(.+?)__", r"\1", text)                      # underline
    text = re.sub(r"~~(.+?)~~", r"\1", text)                      # strikethrough
    text = re.sub(r"`{3}.*?\n(.*?)`{3}", r"\1", text, flags=re.S) # code block
    text = re.sub(r"`{1,2}(.+?)`{1,2}", r"\1", text)              # inline code
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)            # headings
    text = re.sub(r"\s+#{1,6}\s+", " ", text)                     # inline headings
    text = re.sub(r"^>+\s+", "", text, flags=re.M)                 # blockquote
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.M)              # numbered lists
    text = re.sub(r"^[-*+]\s", "", text, flags=re.M)               # bullet lists
    text = re.sub(r"[-*_]{3,}", "", text)                          # horizontal rules
    text = re.sub(r"\b(\d+)-(\d+)\b", r"\1 \2", text)              # scores: "2-1"
    text = re.sub(r"([^\s]),\s*([^\s])", r"\1 \2", text)           # commas: dates, lists
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)                  # 3+ newlines
    text = re.sub(r"^[ \t]+", "", text, flags=re.M)                # leading space per line
    return text.strip()


async def synthesize_speech(
    settings: Settings,
    text: str,
    language: str = "en",
) -> tuple[bytes, str]:
    text = _clean_for_tts(text)
    voice_name = settings.tts_voice_es if language == "es" else settings.tts_voice_en
    lang_code = "es-ES" if language == "es" else "en-US"

    body = {
        "input": {"text": text},
        "voice": {"languageCode": lang_code, "name": voice_name},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 24000,
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={settings.tts_api_key}",
            json=body,
        )
        if response.status_code == 403:
            logger.error(
                "Cloud Text-to-Speech API returned 403. "
                "Make sure the API is enabled at: "
                "https://console.cloud.google.com/apis/library/texttospeech.googleapis.com"
            )
            raise RuntimeError(
                "Cloud Text-to-Speech API is not enabled for this API key. "
                "Enable it at https://console.cloud.google.com/apis/library/texttospeech.googleapis.com"
            )
        response.raise_for_status()
        data = response.json()

    audio_b64 = data.get("audioContent", "")
    if not audio_b64:
        logger.warning("TTS returned empty audioContent")
        return b"", "audio/wav"

    raw = base64.b64decode(audio_b64)
    logger.info("TTS audio bytes=%s voice=%s", len(raw), voice_name)
    return _pcm_to_wav(raw), "audio/wav"


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

    audio, mime = await synthesize_speech(settings, phrase, lang)
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
