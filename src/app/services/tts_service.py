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


class _KokoroWrapper(Kokoro):
    """Wrapper around Kokoro that converts raw .bin voice files to .npz format."""

    def __init__(self, model_path: str, voices_path: str):
        logger.info(f"Initializing Kokoro with model: {model_path}")
        logger.info(f"Voice file path: {voices_path}")

        actual_voices_path = voices_path

        # kokoro_onnx calls np.load() which only handles .npy/.npz files.
        # HuggingFace provides raw .bin float32 files, so convert to .npz.
        voices_path_obj = Path(voices_path)
        if voices_path_obj.suffix == ".bin":
            npz_path = voices_path_obj.with_suffix(".npz")
            voice_name = voices_path_obj.stem  # e.g. "af" from "af.bin"

            # Reuse cached .npz if it exists
            if not npz_path.exists():
                logger.info(
                    f"Converting {voices_path_obj.name} to .npz (voice='{voice_name}')"
                )
                raw_data = np.fromfile(str(voices_path_obj), dtype=np.float32)
                logger.info(
                    f"Read {raw_data.shape[0]} float32 values from {voices_path_obj.name}"
                )
                # Reshape from flat (N,) to (M, 1, 256) as expected by kokoro-onnx
                num_frames = raw_data.shape[0] // 256
                voice_embed = raw_data.reshape(num_frames, 1, 256)
                logger.info(f"Reshaped voice data to {voice_embed.shape}")
                np.savez(str(npz_path), **{voice_name: voice_embed})
                logger.info(f"Saved .npz to {npz_path}")

            actual_voices_path = str(npz_path)
            logger.info(f"Using .npz path: {actual_voices_path}")

        super().__init__(model_path=model_path, voices_path=actual_voices_path)
        logger.info(f"Kokoro initialized. Available voices: {list(self.voices.keys())}")


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
            self._model = _KokoroWrapper(model_path=model_path, voices_path=voices_path)
        return self._model

    def synthesize_to_mp3(
        self, text: str, voice: str | None = None
    ) -> tuple[Path, float | None]:
        model = self._get_model()
        chosen_voice = voice or self._settings.tts_voice

        # If the chosen voice is not available, use the first available voice
        available_voices = list(model.voices.keys())

        if not available_voices:
            raise RuntimeError(
                f"No voices available in the TTS model. Check voice file loading. "
                f"Voices dict is empty: {model.voices}"
            )

        if chosen_voice not in available_voices:
            logger.warning(
                f"Voice '{chosen_voice}' not found. Available voices: {available_voices}. "
                f"Using '{available_voices[0]}' instead."
            )
            chosen_voice = available_voices[0]

        audio, sample_rate = model.create(
            text=text, voice=chosen_voice, lang=self._settings.tts_lang
        )

        wav_path = ensure_parent(
            self._settings.audio_temp_dir / f"{generate_id('tts')}.wav"
        )
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
