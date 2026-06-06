"""Chat service for the OpenRouter audio agent using deepagents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from app.agents.audio_model import ChatOpenRouterAudio
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class AudioChatService:
    """
    Chat service that uses the deepagents audio agent with gpt-audio-mini.

    This service handles multimodal audio input and returns both text
    and audio responses, while maintaining conversation context and
    having access to tools, skills, and memory.
    """

    def __init__(
        self,
        agent_graph,
        conversation_service: ConversationService,
        audio_model: ChatOpenRouterAudio,
    ) -> None:
        self._agent_graph = agent_graph
        self._conversation_service = conversation_service
        self._audio_model = audio_model

    async def send_message(self, payload: ChatMessageRequest) -> ChatMessageResponse:
        """Process a message and return text + audio response."""

        # Store user message in conversation history
        self._conversation_service.append(payload.thread_id, "user", payload.message)

        # Prepare message content (text only for now, audio will be added separately)
        message_content = payload.message

        # Invoke the agent
        result = await self._agent_graph.ainvoke(
            {"messages": [HumanMessage(content=message_content)]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )

        # Extract response
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        content = getattr(final_message, "content", "") if final_message else ""

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        agent_text = str(content).strip()

        # Store assistant message in conversation history
        self._conversation_service.append(payload.thread_id, "assistant", agent_text)

        # Get audio path from the model
        audio_path_str = self._audio_model.get_last_audio_path()
        audio_url = None
        audio_mime_type = None

        if audio_path_str and payload.response_audio:
            audio_path = Path(audio_path_str)
            if audio_path.exists():
                audio_url = f"/api/audio/files/{audio_path.name}"
                audio_mime_type = "audio/wav"
                logger.info(
                    "Audio response generated for thread %s: %s",
                    payload.thread_id,
                    audio_path.name,
                )

        # Extract model info
        resolved_model = None
        response_metadata = (
            getattr(final_message, "response_metadata", {}) if final_message else {}
        )
        if isinstance(response_metadata, dict):
            resolved_model = response_metadata.get(
                "model_name"
            ) or response_metadata.get("model")

        logger.info("Audio chat response generated for thread %s", payload.thread_id)

        return ChatMessageResponse(
            thread_id=payload.thread_id,
            user_message=payload.message,
            agent_text=agent_text,
            audio_url=audio_url,
            audio_mime_type=audio_mime_type,
            resolved_model=resolved_model,
        )

    async def send_audio_message(
        self,
        payload: ChatMessageRequest,
        audio_bytes: bytes,
        audio_format: str = "wav",
    ) -> ChatMessageResponse:
        """
        Process an audio message and return text + audio response.

        This method sends both text and audio to the model for processing.
        """
        import base64

        # Store user message in conversation history
        self._conversation_service.append(
            payload.thread_id, "user", f"[Audio message] {payload.message}"
        )

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        logger.info(f"Encoded audio to base64: {len(audio_base64)} chars")

        # Set pending audio input in the model - it will be injected into the last user message
        self._audio_model.set_pending_audio_input(audio_base64, audio_format)

        logger.info(f"Set pending audio input in model, format: {audio_format}")

        # Create text-only message (audio will be injected by the model)
        message_text = (
            payload.message or "Analiza este audio y responde con tu propia voz."
        )

        logger.info(f"Invoking agent with message: {message_text}")

        # Invoke the agent with text message
        result = await self._agent_graph.ainvoke(
            {"messages": [HumanMessage(content=message_text)]},
            config={"configurable": {"thread_id": payload.thread_id}},
        )

        # Extract response
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        content = getattr(final_message, "content", "") if final_message else ""

        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)

        agent_text = str(content).strip()

        # Store assistant message in conversation history
        self._conversation_service.append(payload.thread_id, "assistant", agent_text)

        # Get audio path from the model
        audio_path_str = self._audio_model.get_last_audio_path()
        audio_url = None
        audio_mime_type = None

        if audio_path_str and payload.response_audio:
            audio_path = Path(audio_path_str)
            if audio_path.exists():
                audio_url = f"/api/audio/files/{audio_path.name}"
                audio_mime_type = "audio/wav"
                logger.info(
                    "Audio response generated for thread %s: %s",
                    payload.thread_id,
                    audio_path.name,
                )

        # Extract model info
        resolved_model = None
        response_metadata = (
            getattr(final_message, "response_metadata", {}) if final_message else {}
        )
        if isinstance(response_metadata, dict):
            resolved_model = response_metadata.get(
                "model_name"
            ) or response_metadata.get("model")

        logger.info("Audio chat response generated for thread %s", payload.thread_id)

        return ChatMessageResponse(
            thread_id=payload.thread_id,
            user_message=payload.message,
            agent_text=agent_text,
            audio_url=audio_url,
            audio_mime_type=audio_mime_type,
            resolved_model=resolved_model,
        )
