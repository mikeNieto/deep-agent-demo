from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi import Request

from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService


def _get_service(request: Request, name: str) -> Any:
    return getattr(request.app.state, name)


def get_chat_service(request: Request) -> ChatService:
    chat_service = _get_service(request, "chat_service")
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Chat service is unavailable. Configure OPENROUTER_API_KEY.")
    return chat_service


def get_conversation_service(request: Request) -> ConversationService:
    return _get_service(request, "conversation_service")


def get_stt_service(request: Request) -> STTService:
    return _get_service(request, "stt_service")


def get_tts_service(request: Request) -> TTSService:
    return _get_service(request, "tts_service")
