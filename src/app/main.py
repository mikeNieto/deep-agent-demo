from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.agents.factory import create_agent_graph
from app.api.audio import router as audio_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.config import get_settings
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.stt_service import STTService
from app.services.tts_preparation_service import TTSPreparationService
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
    tts_preparation_service = TTSPreparationService(settings)

    async with checkpoint_saver(settings) as saver:
        agent_graph = None
        chat_service = None
        if settings.openrouter_api_key:
            agent_graph = create_agent_graph(settings, saver)
            chat_service = ChatService(
                agent_graph,
                conversation_service,
                tts_service,
                tts_preparation_service,
            )

        app.state.settings = settings
        app.state.conversation_service = conversation_service
        app.state.stt_service = stt_service
        app.state.tts_service = tts_service
        app.state.tts_preparation_service = tts_preparation_service
        app.state.chat_service = chat_service
        app.state.agent_graph = agent_graph
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
