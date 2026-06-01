from __future__ import annotations

import logging
from pathlib import Path
from wave import open as wave_open

import numpy as np
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro

from app.services.audio_service import convert_wav_to_mp3
from app.config import Settings
from app.storage.files import ensure_parent
from app.utils.ids import generate_id


logger = logging.getLogger(__name__)

KOKORO_MODEL_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"


class TTSService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Kokoro | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Kokoro:
        if self._model is None:
            logger.info("Loading kokoro-onnx model")
            voice_file = f"voices/{self._settings.tts_voice}.bin"
            model_path = hf_hub_download(
                repo_id=KOKORO_MODEL_REPO,
                filename="onnx/model.onnx",
                local_dir=str(self._settings.models_dir / "kokoro"),
            )
            voices_path = hf_hub_download(
                repo_id=KOKORO_MODEL_REPO,
                filename=voice_file,
                local_dir=str(self._settings.models_dir / "kokoro"),
            )
            self._model = Kokoro(model_path=model_path, voices_path=voices_path)
        return self._model

    def synthesize_to_mp3(self, text: str, voice: str | None = None) -> tuple[Path, float | None]:
        model = self._get_model()
        chosen_voice = voice or self._settings.tts_voice
        audio, sample_rate = model.create(text=text, voice=chosen_voice, lang=self._settings.tts_lang)

        wav_path = ensure_parent(self._settings.audio_temp_dir / f"{generate_id('tts')}.wav")
        mp3_path = wav_path.with_suffix(".mp3")

        pcm = np.clip(audio, -1.0, 1.0)
        pcm_16 = (pcm * 32767).astype(np.int16)

        with wave_open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_16.tobytes())

        convert_wav_to_mp3(wav_path, mp3_path)
        duration = len(audio) / float(sample_rate)
        return mp3_path, duration
