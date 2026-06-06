from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.dependencies import (
    get_audio_chat_service,
    get_openrouter_audio_service,
    get_stt_service,
    get_tts_service,
)
from app.schemas.audio import (
    AudioSynthesisRequest,
    AudioSynthesisResponse,
    AudioTranscriptionResponse,
    OpenRouterAudioRequest,
    OpenRouterAudioResponse,
)
from app.schemas.chat import ChatMessageRequest
from app.services.audio_chat_service import AudioChatService
from app.services.audio_service import save_upload
from app.services.openrouter_audio_service import OpenRouterAudioService
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
    return AudioTranscriptionResponse(
        text=text, language=language, duration_seconds=duration
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

    # Detect MIME type from file extension
    suffix = Path(filename).suffix.lower()
    if suffix == ".wav":
        media_type = "audio/wav"
    elif suffix == ".mp3":
        media_type = "audio/mpeg"
    else:
        media_type = "audio/mpeg"  # default

    return FileResponse(file_path, media_type=media_type, filename=filename)


@router.post("/openrouter-audio", response_model=OpenRouterAudioResponse)
async def process_audio_openrouter(
    file: UploadFile = File(...),
    prompt: str = "Analiza este audio y responde con tu propia voz.",
    voice: str | None = None,
    user_id: str = "default-user",
    thread_id: str = "default-thread",
    audio_chat_service: AudioChatService = Depends(get_audio_chat_service),
) -> OpenRouterAudioResponse:
    """
    Process audio using the deepagents audio agent with gpt-audio-mini.
    This agent has access to tools, skills, and memory.
    Returns both text and audio responses.
    """
    settings = get_settings()

    # Determine audio format from filename
    filename = file.filename or "input.wav"
    suffix = Path(filename).suffix.lower().lstrip(".")
    audio_format = suffix if suffix in ("mp3", "wav", "ogg", "m4a", "flac") else "wav"

    # Read audio bytes
    file.file.seek(0)
    audio_bytes = file.file.read()

    logger.info(
        f"Received audio file: {filename}, format: {audio_format}, size: {len(audio_bytes)} bytes"
    )

    # Create chat message request
    chat_request = ChatMessageRequest(
        user_id=user_id,
        thread_id=thread_id,
        message=prompt,
        response_audio=True,
    )

    # Send audio message through the agent
    response = await audio_chat_service.send_audio_message(
        payload=chat_request,
        audio_bytes=audio_bytes,
        audio_format=audio_format,
    )

    return OpenRouterAudioResponse(
        text=response.agent_text,
        audio_url=response.audio_url,
        audio_mime_type=response.audio_mime_type or "audio/wav",
    )
