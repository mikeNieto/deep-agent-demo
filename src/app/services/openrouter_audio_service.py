from __future__ import annotations

import base64
import json
import logging
import struct
from pathlib import Path

import httpx

from app.config import Settings
from app.storage.files import ensure_parent
from app.utils.ids import generate_id


logger = logging.getLogger(__name__)

OPENROUTER_CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAudioServiceError(RuntimeError):
    pass


class OpenRouterAudioService:
    """
    Service that uses OpenRouter's multimodal audio model (gpt-audio-mini)
    to receive audio input and return both text and audio responses.

    Audio output from OpenRouter REQUIRES streaming (stream: true).
    The response is delivered as SSE chunks with audio data and transcript
    in the delta.audio field.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client()

    def process_audio(
        self,
        audio_bytes: bytes,
        audio_format: str = "wav",
        prompt: str = "Analiza este audio y responde con tu propia voz.",
        voice: str = "alloy",
    ) -> tuple[str, Path | None]:
        """
        Send audio to OpenRouter's multimodal model and get text + audio response.

        Uses streaming SSE as required by OpenRouter for audio output.

        Args:
            audio_bytes: Raw audio file bytes
            audio_format: Audio format (mp3, wav, etc.)
            prompt: Text prompt to accompany the audio
            voice: Voice for the audio response

        Returns:
            Tuple of (text_response, audio_file_path_or_None)
        """
        if not self._settings.openrouter_api_key:
            raise OpenRouterAudioServiceError("OPENROUTER_API_KEY is not configured")

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "model": self._settings.openrouter_audio_model,
            "modalities": ["text", "audio"],
            "audio": {
                "voice": voice,
                "format": "pcm16",  # pcm16 is required when stream=true
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            "stream": True,
        }

        logger.info(
            "Sending audio request to OpenRouter: model=%s, format=%s, voice=%s",
            self._settings.openrouter_audio_model,
            audio_format,
            voice,
        )

        try:
            with self._client.stream(
                "POST",
                OPENROUTER_CHAT_API_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Deep Agent Demo",
                },
                json=payload,
                timeout=180.0,
            ) as response:
                if response.status_code != 200:
                    # Read error body for diagnostics
                    response.read()
                    error_body = response.text
                    logger.error(
                        "OpenRouter Audio API error: %s %s",
                        response.status_code,
                        error_body,
                    )
                    raise OpenRouterAudioServiceError(
                        f"OpenRouter Audio API failed: {response.status_code} {error_body}"
                    )

                # Parse SSE stream to collect audio chunks and transcript
                audio_data_chunks: list[str] = []
                transcript_chunks: list[str] = []

                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse SSE chunk: %s", data_str[:200])
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Collect audio data from delta.audio.data
                    audio = delta.get("audio", {})
                    if audio.get("data"):
                        audio_data_chunks.append(audio["data"])

                    # Collect transcript from delta.audio.transcript
                    if audio.get("transcript"):
                        transcript_chunks.append(audio["transcript"])

                    # Also check delta.content for text-only responses
                    content = delta.get("content")
                    if content and not audio.get("transcript"):
                        transcript_chunks.append(content)

        except httpx.HTTPError as e:
            logger.error("OpenRouter Audio API error: %s", e)
            raise OpenRouterAudioServiceError(f"OpenRouter Audio API error: {e}") from e

        # Combine transcript
        text_response = "".join(transcript_chunks).strip()
        logger.info("OpenRouter audio transcript: %s", text_response[:200])

        # Combine and decode audio chunks, save to file
        audio_path = None
        if audio_data_chunks:
            try:
                # Decode and concatenate all PCM16 chunks
                full_audio_b64 = "".join(audio_data_chunks)
                pcm_bytes = base64.b64decode(full_audio_b64)

                # Wrap PCM16 data in WAV header
                # OpenAI audio output: 24kHz, 16-bit signed, mono
                wav_bytes = self._pcm16_to_wav(pcm_bytes, sample_rate=24000)

                audio_path = ensure_parent(
                    self._settings.audio_temp_dir
                    / f"{generate_id('openrouter-audio')}.wav"
                )
                audio_path.write_bytes(wav_bytes)
                logger.info(
                    "Saved OpenRouter audio response to %s (%d bytes PCM, %d bytes WAV)",
                    audio_path,
                    len(pcm_bytes),
                    len(wav_bytes),
                )
            except Exception as e:
                logger.warning("Failed to decode/save audio response: %s", e)
        else:
            logger.warning("No audio data chunks received from OpenRouter")

        return text_response, audio_path

    @staticmethod
    def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
        """
        Convert raw PCM16 bytes to WAV format.

        Args:
            pcm_bytes: Raw PCM16 data (signed 16-bit little-endian, mono)
            sample_rate: Sample rate in Hz (default 24000 for OpenAI audio)

        Returns:
            WAV file bytes with proper header
        """
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_bytes)

        # WAV header structure
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",  # ChunkID
            36 + data_size,  # ChunkSize
            b"WAVE",  # Format
            b"fmt ",  # Subchunk1ID
            16,  # Subchunk1Size (PCM)
            1,  # AudioFormat (PCM = 1)
            num_channels,  # NumChannels
            sample_rate,  # SampleRate
            byte_rate,  # ByteRate
            block_align,  # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",  # Subchunk2ID
            data_size,  # Subchunk2Size
        )

        return header + pcm_bytes
