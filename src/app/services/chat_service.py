from __future__ import annotations

import logging
from time import perf_counter

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
        deepagent_started_at = perf_counter()
        self._conversation_service.append(payload.thread_id, "user", payload.message)

        logger.info(
            "Deepagent invocation started thread_id=%s message_chars=%s",
            payload.thread_id,
            len(payload.message),
        )
        result = await self._agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": payload.message}]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )
        agent_elapsed_ms = round((perf_counter() - deepagent_started_at) * 1000, 2)

        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        content = getattr(final_message, "content", "") if final_message else ""
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        agent_text = str(content).strip()
        resolved_model = None
        response_metadata = (
            getattr(final_message, "response_metadata", {}) if final_message else {}
        )
        if isinstance(response_metadata, dict):
            resolved_model = response_metadata.get(
                "model_name"
            ) or response_metadata.get("model")

        logger.info(
            "Deepagent invocation completed thread_id=%s elapsed_ms=%s model=%s output_chars=%s",
            payload.thread_id,
            agent_elapsed_ms,
            resolved_model,
            len(agent_text),
        )

        audio_url = None
        audio_mime_type = None
        tts_text = None
        if payload.response_audio and agent_text:
            try:
                tts_text = self._tts_preparation_service.prepare_text(agent_text)
                logger.info(
                    "TTS preparation response thread_id=%s text=%s",
                    payload.thread_id,
                    tts_text,
                )
                logger.info(
                    "Sending text to TTS thread_id=%s text_chars=%s",
                    payload.thread_id,
                    len(tts_text),
                )
                tts_started_at = perf_counter()
                mp3_path, audio_duration = self._tts_service.synthesize_to_mp3(tts_text)
                tts_elapsed_ms = round((perf_counter() - tts_started_at) * 1000, 2)
                audio_url = f"/api/audio/files/{mp3_path.name}"
                audio_mime_type = "audio/mpeg"
                logger.info(
                    "TTS completed thread_id=%s elapsed_ms=%s audio_duration_seconds=%s file=%s",
                    payload.thread_id,
                    tts_elapsed_ms,
                    audio_duration,
                    mp3_path.name,
                )
            except (TTSPreparationServiceError, TTSServiceError) as e:
                logger.warning(
                    "TTS generation failed for thread %s: %s. Returning chat response without audio.",
                    payload.thread_id,
                    e,
                )
        elif payload.response_audio:
            logger.info(
                "Skipping TTS because agent returned empty text thread_id=%s",
                payload.thread_id,
            )
        else:
            logger.info(
                "Skipping TTS because response_audio is disabled thread_id=%s",
                payload.thread_id,
            )

        self._conversation_service.append(
            payload.thread_id,
            "assistant",
            tts_text or agent_text,
        )

        logger.info("Chat response generated for thread %s", payload.thread_id)
        return ChatMessageResponse(
            thread_id=payload.thread_id,
            user_message=payload.message,
            agent_text=agent_text,
            tts_text=tts_text,
            audio_url=audio_url,
            audio_mime_type=audio_mime_type,
            resolved_model=resolved_model,
        )
