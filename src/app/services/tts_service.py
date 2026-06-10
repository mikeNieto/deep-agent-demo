from __future__ import annotations

import logging
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import httpx

from app.config import Settings
from app.storage.files import ensure_parent
from app.utils.ids import generate_id


logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/audio/speech"


class TTSServiceError(RuntimeError):
    pass


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client()

    def synthesize_to_mp3(
        self, text: str, voice: str | None = None
    ) -> tuple[Path, float | None]:
        """
        Synthesize text to speech using OpenRouter API.

        Args:
            text: The text to synthesize
            voice: Optional voice name override (uses config default if not provided)

        Returns:
            Tuple of (mp3_file_path, duration_in_seconds)
        """
        chosen_voice = voice or self._settings.tts_voice
        started_at = perf_counter()
        logger.info(
            "TTS synthesis started voice=%s text_chars=%s",
            chosen_voice,
            len(text),
        )

        try:
            response = self._client.post(
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
                    "response_format": "mp3",
                },
                timeout=self._settings.tts_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Log full response content for 4xx/5xx errors to aid debugging
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

                # If provider (e.g. Gemini via OpenRouter) requires PCM, retry with PCM
                body_lower = body.lower() if isinstance(body, str) else ""
                if (
                    "gemini tts" in body_lower
                    or 'response_format="mp3"' in body_lower
                    or 'got "mp3"' in body_lower
                ):
                    logger.info(
                        "OpenRouter indicates mp3 unsupported; retrying with pcm response_format"
                    )
                    try:
                        retry_resp = self._client.post(
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
                                "response_format": "pcm",
                            },
                            timeout=self._settings.tts_timeout,
                        )
                        retry_resp.raise_for_status()
                    except httpx.HTTPError as e2:
                        logger.error("Retry to pcm also failed: %s", e2)
                        raise TTSServiceError(f"Failed to synthesize speech: {e2}")

                    # Convert PCM/WAV to MP3 using ffmpeg if available
                    content_type = retry_resp.headers.get("content-type", "")
                    pcm_data = retry_resp.content

                    ffmpeg_cmd = shutil.which("ffmpeg")
                    if not ffmpeg_cmd:
                        logger.error("ffmpeg not found; cannot convert PCM to MP3")
                        raise TTSServiceError(
                            "ffmpeg not available to convert PCM to MP3"
                        )

                    mp3_path = ensure_parent(
                        self._settings.audio_temp_dir / f"{generate_id('tts')}.mp3"
                    )

                    with tempfile.TemporaryDirectory() as td:
                        # If response is a WAV container, save as .wav and convert
                        wav_path = Path(td) / "out.wav"
                        pcm_path = Path(td) / "out.pcm"
                        try:
                            if "wav" in content_type or pcm_data[:4] == b"RIFF":
                                wav_path.write_bytes(pcm_data)
                                cmd = [
                                    ffmpeg_cmd,
                                    "-y",
                                    "-i",
                                    str(wav_path),
                                    str(mp3_path),
                                ]
                            else:
                                # Assume signed 16-bit little-endian PCM, 1 channel, 24000 Hz
                                pcm_path.write_bytes(pcm_data)
                                cmd = [
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
                                ]

                            subprocess.run(
                                cmd,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                        except subprocess.CalledProcessError as cpe:
                            logger.error("ffmpeg conversion failed: %s", cpe)
                            raise TTSServiceError("Failed to convert PCM to MP3")

                    logger.info(f"Saved TTS audio to %s", mp3_path)
                    duration = self._extract_mp3_duration(mp3_path)
                    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
                    logger.info(
                        "TTS synthesis completed voice=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
                        chosen_voice,
                        elapsed_ms,
                        duration,
                        mp3_path.name,
                    )
                    return mp3_path, duration

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

        # Save the MP3 file
        mp3_path = ensure_parent(
            self._settings.audio_temp_dir / f"{generate_id('tts')}.mp3"
        )

        mp3_data = response.content
        with open(mp3_path, "wb") as f:
            f.write(mp3_data)

        logger.info(f"Saved TTS audio to {mp3_path}")

        # Try to extract duration from the MP3 file
        duration = self._extract_mp3_duration(mp3_path)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "TTS synthesis completed voice=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
            chosen_voice,
            elapsed_ms,
            duration,
            mp3_path.name,
        )

        return mp3_path, duration

    @staticmethod
    def _extract_mp3_duration(file_path: Path) -> float | None:
        try:
            ffprobe_cmd = shutil.which("ffprobe")
            if not ffprobe_cmd:
                logger.info("ffprobe not found; cannot extract audio duration file=%s", file_path)
                return None

            result = subprocess.run(
                [
                    ffprobe_cmd,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(file_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)
            duration = payload.get("format", {}).get("duration")
            if duration in (None, ""):
                logger.info(
                    "ffprobe did not return audio duration file=%s stdout=%s",
                    file_path,
                    result.stdout.strip(),
                )
                return None

            return round(float(duration), 3)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not parse audio duration for %s: %s", file_path, exc)
            return None
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            logger.warning("ffprobe failed for %s: %s", file_path, stderr)
            return None
        except Exception as e:
            logger.warning(f"Could not extract MP3 duration: {e}")
            return None
