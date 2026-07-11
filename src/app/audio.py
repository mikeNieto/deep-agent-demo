from __future__ import annotations

import base64
import logging
import struct

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings

logger = logging.getLogger(__name__)


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
        1,  # PCM
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
) -> str:
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
                {"type": "text", "text": "Transcribe this audio. Return ONLY the transcription, nothing else."},
                {"type": "audio", "base64": audio_b64, "mime_type": mime_type},
            ],
        }
    ]
    response = await model.ainvoke(messages)
    text = response.text if hasattr(response, "text") else str(response.content)
    if isinstance(text, list):
        text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in text)
    return text.strip()


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
