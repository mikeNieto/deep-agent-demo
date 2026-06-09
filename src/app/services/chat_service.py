from __future__ import annotations

import logging

from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.conversation_service import ConversationService
from app.services.tts_preparation_service import (
    TTSPreparationService,
    TTSPreparationServiceError,
)
from app.services.tts_service import TTSService, TTSServiceError


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        agent_graph,
        conversation_service: ConversationService,
        tts_service: TTSService,
        tts_preparation_service: TTSPreparationService,
    ) -> None:
        self._agent_graph = agent_graph
        self._conversation_service = conversation_service
        self._tts_service = tts_service
        self._tts_preparation_service = tts_preparation_service

    async def send_message(self, payload: ChatMessageRequest) -> ChatMessageResponse:
        self._conversation_service.append(payload.thread_id, "user", payload.message)

        result = await self._agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )

        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        content = getattr(final_message, "content", "") if final_message else ""
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        agent_text = str(content).strip()
        self._conversation_service.append(payload.thread_id, "assistant", agent_text)

        resolved_model = None
        response_metadata = (
            getattr(final_message, "response_metadata", {}) if final_message else {}
        )
        if isinstance(response_metadata, dict):
            resolved_model = response_metadata.get(
                "model_name"
            ) or response_metadata.get("model")

        audio_url = None
        audio_mime_type = None
        if payload.response_audio and agent_text:
            try:
                tts_text = self._tts_preparation_service.prepare_text(agent_text)
                mp3_path, _duration = self._tts_service.synthesize_to_mp3(tts_text)
                audio_url = f"/api/audio/files/{mp3_path.name}"
                audio_mime_type = "audio/mpeg"
            except (TTSPreparationServiceError, TTSServiceError) as e:
                logger.warning(
                    "TTS generation failed for thread %s: %s. Returning chat response without audio.",
                    payload.thread_id,
                    e,
                )

        logger.info("Chat response generated for thread %s", payload.thread_id)
        return ChatMessageResponse(
            thread_id=payload.thread_id,
            user_message=payload.message,
            agent_text=agent_text,
            audio_url=audio_url,
            audio_mime_type=audio_mime_type,
            resolved_model=resolved_model,
        )
