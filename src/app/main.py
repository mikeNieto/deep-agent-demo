from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.agents.factory import create_agent_graph, create_audio_agent_graph
from app.api.audio import router as audio_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import get_settings
from app.services.audio_chat_service import AudioChatService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.openrouter_audio_service import OpenRouterAudioService
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
from app.storage.checkpoints import checkpoint_saver
from app.utils.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()

    conversation_service = ConversationService()
    stt_service = STTService(settings)
    tts_service = TTSService(settings)
    openrouter_audio_service = (
        OpenRouterAudioService(settings) if settings.openrouter_api_key else None
    )

    async with checkpoint_saver(settings) as saver:
        agent_graph = None
        chat_service = None
        audio_agent_graph = None
        audio_chat_service = None

        if settings.openrouter_api_key:
            # Create text-based agent
            agent_graph = create_agent_graph(settings, saver)
            chat_service = ChatService(agent_graph, conversation_service, tts_service)

            # Create audio-based agent
            audio_agent_graph = create_audio_agent_graph(settings, saver)
            from app.agents.audio_model import ChatOpenRouterAudio

            audio_model = ChatOpenRouterAudio(
                model=settings.openrouter_audio_model,
                api_key=settings.openrouter_api_key,
                voice=settings.openrouter_audio_voice,
                temperature=0.2,
            )
            audio_model.set_audio_temp_dir(settings.audio_temp_dir)
            audio_chat_service = AudioChatService(
                audio_agent_graph, conversation_service, audio_model
            )

        app.state.settings = settings
        app.state.conversation_service = conversation_service
        app.state.stt_service = stt_service
        app.state.tts_service = tts_service
        app.state.chat_service = chat_service
        app.state.agent_graph = agent_graph
        app.state.openrouter_audio_service = openrouter_audio_service
        app.state.audio_chat_service = audio_chat_service
        app.state.audio_agent_graph = audio_agent_graph
        yield


app = FastAPI(title="Conversational Agent MVP", lifespan=lifespan)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(audio_router)


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        factory=False,
    )
