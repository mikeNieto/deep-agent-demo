from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import httpx

from app.config import Settings
from app.storage.files import ensure_parent
from app.utils.ids import generate_id


logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/audio/speech"

PCM_ONLY_MODEL_PREFIXES = ("google/",)


class TTSServiceError(RuntimeError):
    pass


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient()
        self._pcm_only = settings.tts_model.startswith(PCM_ONLY_MODEL_PREFIXES)

    async def synthesize_to_mp3(
        self, text: str, voice: str | None = None
    ) -> tuple[Path, float | None]:
        """
        Synthesize text to speech using OpenRouter API.

        Args:
            text: The text to synthesize
            voice: Optional voice name override (uses config default if not provided)

        Returns:
            Tuple of (audio_file_path, duration_in_seconds)
        """
        chosen_voice = voice or self._settings.tts_voice
        started_at = perf_counter()
        logger.info(
            "TTS synthesis started voice=%s text_chars=%s",
            chosen_voice,
            len(text),
        )

        response_format = "pcm" if self._pcm_only else "mp3"

        try:
            response = await self._client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Deep Agent Demo",
                },
                json={
                    "model": self._settings.tts_model,
                    "input": text,
                    "voice": chosen_voice,
                    "response_format": response_format,
                },
                timeout=self._settings.tts_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    body = resp.text
                except Exception:
                    body = "<could not read response body>"
                logger.error(
                    "OpenRouter TTS API status error: %s %s",
                    getattr(resp, "status_code", "?"),
                    body,
                )

                body_lower = body.lower() if isinstance(body, str) else ""
                if response_format == "mp3" and (
                    "gemini tts" in body_lower
                    or 'response_format="mp3"' in body_lower
                    or 'got "mp3"' in body_lower
                ):
                    logger.info(
                        "OpenRouter indicates mp3 unsupported; retrying with pcm response_format"
                    )
                    return await self._synthesize_pcm_and_convert(
                        text, chosen_voice, started_at
                    )

                raise TTSServiceError(
                    f"Failed to synthesize speech: status={resp.status_code}, body={body}"
                )
            logger.error(
                "OpenRouter TTS API status error (no response attached): %s", e
            )
            raise TTSServiceError(f"Failed to synthesize speech: {e}")
        except httpx.HTTPError as e:
            logger.error(f"OpenRouter TTS API error: {e}")
            raise TTSServiceError(f"Failed to synthesize speech: {e}")

        if response_format == "pcm":
            return await self._save_pcm_audio(response, started_at, chosen_voice)
        return await self._save_mp3_audio(response, started_at, chosen_voice)

    async def _synthesize_pcm_and_convert(
        self, text: str, voice: str, started_at: float
    ) -> tuple[Path, float | None]:
        """Fallback: retry TTS call with PCM format and convert to MP3."""
        try:
            retry_resp = await self._client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Deep Agent Demo",
                },
                json={
                    "model": self._settings.tts_model,
                    "input": text,
                    "voice": voice,
                    "response_format": "pcm",
                },
                timeout=self._settings.tts_timeout,
            )
            retry_resp.raise_for_status()
        except httpx.HTTPError as e2:
            logger.error("Retry to pcm also failed: %s", e2)
            raise TTSServiceError(f"Failed to synthesize speech: {e2}")

        return await self._save_pcm_audio(retry_resp, started_at, voice)

    async def _save_pcm_audio(
        self, response: httpx.Response, started_at: float, voice: str
    ) -> tuple[Path, float | None]:
        """Save PCM audio: WAV containers are saved directly; raw PCM is converted via ffmpeg."""
        content_type = response.headers.get("content-type", "")
        pcm_data = response.content

        if "wav" in content_type or pcm_data[:4] == b"RIFF":
            wav_path = ensure_parent(
                self._settings.audio_temp_dir / f"{generate_id('tts')}.wav"
            )
            wav_path.write_bytes(pcm_data)
            logger.info("Saved TTS audio (WAV) to %s", wav_path)
            duration = await self._extract_audio_duration(wav_path)
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.info(
                "TTS synthesis completed voice=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
                voice,
                elapsed_ms,
                duration,
                wav_path.name,
            )
            return wav_path, duration

        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            logger.error("ffmpeg not found; cannot convert PCM to MP3")
            raise TTSServiceError("ffmpeg not available to convert PCM to MP3")

        mp3_path = ensure_parent(
            self._settings.audio_temp_dir / f"{generate_id('tts')}.mp3"
        )

        with tempfile.TemporaryDirectory() as td:
            pcm_path = Path(td) / "out.pcm"
            pcm_path.write_bytes(pcm_data)
            proc = await asyncio.create_subprocess_exec(
                ffmpeg_cmd,
                "-y",
                "-f",
                "s16le",
                "-ar",
                "24000",
                "-ac",
                "1",
                "-i",
                str(pcm_path),
                str(mp3_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                stderr_text = stderr.decode() if stderr else ""
                logger.error("ffmpeg conversion failed: %s", stderr_text)
                raise TTSServiceError("Failed to convert PCM to MP3")

        logger.info("Saved TTS audio to %s", mp3_path)
        duration = await self._extract_audio_duration(mp3_path)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "TTS synthesis completed voice=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
            voice,
            elapsed_ms,
            duration,
            mp3_path.name,
        )
        return mp3_path, duration

    async def _save_mp3_audio(
        self, response: httpx.Response, started_at: float, voice: str
    ) -> tuple[Path, float | None]:
        mp3_path = ensure_parent(
            self._settings.audio_temp_dir / f"{generate_id('tts')}.mp3"
        )
        mp3_path.write_bytes(response.content)
        logger.info("Saved TTS audio to %s", mp3_path)
        duration = await self._extract_audio_duration(mp3_path)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "TTS synthesis completed voice=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
            voice,
            elapsed_ms,
            duration,
            mp3_path.name,
        )
        return mp3_path, duration

    @staticmethod
    async def _extract_audio_duration(file_path: Path) -> float | None:
        try:
            ffprobe_cmd = shutil.which("ffprobe")
            if not ffprobe_cmd:
                logger.info(
                    "ffprobe not found; cannot extract audio duration file=%s",
                    file_path,
                )
                return None

            proc = await asyncio.create_subprocess_exec(
                ffprobe_cmd,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                stderr_text = stderr.decode().strip() if stderr else str(proc.returncode)
                logger.warning("ffprobe failed for %s: %s", file_path, stderr_text)
                return None

            payload = json.loads(stdout)
            duration = payload.get("format", {}).get("duration")
            if duration in (None, ""):
                logger.info(
                    "ffprobe did not return audio duration file=%s stdout=%s",
                    file_path,
                    stdout.decode().strip() if stdout else "",
                )
                return None
            return round(float(duration), 3)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not parse audio duration for %s: %s", file_path, exc)
            return None
        except Exception as e:
            logger.warning("Could not extract audio duration: %s", e)
            return None
