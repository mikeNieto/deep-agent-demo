from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_chat_service, get_conversation_service
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse, ConversationHistoryResponse
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService


router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat/message", response_model=ChatMessageResponse)
async def send_chat_message(
    payload: ChatMessageRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    return await chat_service.send_message(payload)


@router.get("/conversations/{thread_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    thread_id: str,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationHistoryResponse:
    return ConversationHistoryResponse(thread_id=thread_id, messages=conversation_service.list_messages(thread_id))
