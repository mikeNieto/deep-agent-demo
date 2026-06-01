from __future__ import annotations

import logging
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import Settings


logger = logging.getLogger(__name__)


class STTService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            logger.info("Loading faster-whisper model %s", self._settings.stt_model)
            self._model = WhisperModel(
                self._settings.stt_model,
                device=self._settings.stt_device,
                compute_type=self._settings.stt_compute_type,
                download_root=str(self._settings.models_dir),
            )
        return self._model

    def transcribe(self, file_path: Path) -> tuple[str, str | None, float | None]:
        model = self._get_model()
        segments, info = model.transcribe(str(file_path), vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration = getattr(info, "duration", None)
        return text, getattr(info, "language", None), duration
