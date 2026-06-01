from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.dependencies import get_stt_service, get_tts_service
from app.schemas.audio import AudioSynthesisRequest, AudioSynthesisResponse, AudioTranscriptionResponse
from app.services.audio_service import save_upload
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.utils.ids import generate_id


router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    stt_service: STTService = Depends(get_stt_service),
) -> AudioTranscriptionResponse:
    settings = get_settings()
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    destination = settings.audio_temp_dir / f"{generate_id('upload')}{suffix}"
    file.file.seek(0)
    save_upload(file.file, destination)

    text, language, duration = stt_service.transcribe(destination)
    return AudioTranscriptionResponse(text=text, language=language, duration_seconds=duration)


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
