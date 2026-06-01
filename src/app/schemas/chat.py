from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str
    response_audio: bool = True


class ChatMessageResponse(BaseModel):
    thread_id: str
    user_message: str
    agent_text: str
    audio_url: str | None = None
    audio_mime_type: str | None = None
    resolved_model: str | None = None


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ConversationMessage]
