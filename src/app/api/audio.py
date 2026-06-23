from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.dependencies import get_stt_service, get_tts_service
from app.schemas.audio import AudioSynthesisRequest, AudioSynthesisResponse, AudioTranscriptionResponse
from app.services.audio_service import save_upload
from app.services.stt_service import STTService, STTServiceError
from app.services.tts_service import TTSService
from app.utils.ids import generate_id


router = APIRouter(prefix="/api/audio", tags=["audio"])
logger = logging.getLogger(__name__)


@router.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    stt_service: STTService = Depends(get_stt_service),
) -> AudioTranscriptionResponse:
    started_at = perf_counter()
    logger.info(
        "STT request received filename=%s content_type=%s",
        file.filename,
        file.content_type,
    )
    settings = get_settings()
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    destination = settings.audio_temp_dir / f"{generate_id('upload')}{suffix}"
    file.file.seek(0)
    save_upload(file.file, destination)

    try:
        original_text, translated_text, detected_language, duration = stt_service.transcribe(
            destination,
            mime_type=file.content_type,
        )
    except STTServiceError as exc:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.warning(
            "STT request failed filename=%s elapsed_ms=%s error=%s",
            file.filename,
            elapsed_ms,
            exc,
        )
        status_code = 503 if "not configured" in str(exc).lower() else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    logger.info(
        "STT request completed filename=%s elapsed_ms=%s transcription_duration_seconds=%s",
        file.filename,
        elapsed_ms,
        duration,
    )
    return AudioTranscriptionResponse(
        original_text=original_text,
        translated_text=translated_text,
        detected_language=detected_language,
        text=translated_text,
        language=detected_language,
        duration_seconds=duration,
    )


@router.post("/synthesize", response_model=AudioSynthesisResponse)
async def synthesize_audio(
    payload: AudioSynthesisRequest,
    tts_service: TTSService = Depends(get_tts_service),
) -> AudioSynthesisResponse:
    mp3_path, duration = tts_service.synthesize_to_mp3(payload.text, payload.voice)
    return AudioSynthesisResponse(
        audio_url=f"/api/audio/files/{mp3_path.name}",
        duration_seconds=duration,
    )


@router.get("/files/{filename}")
async def get_audio_file(filename: str) -> FileResponse:
    settings = get_settings()
    file_path = settings.audio_temp_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
