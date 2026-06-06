from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain_openrouter import ChatOpenRouter

from app.agents.audio_model import ChatOpenRouterAudio
from app.agents.memory import memory_files, skill_paths
from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tools import DEFAULT_TOOLS
from app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenRouter:
    if not settings.openrouter_api_key:
        msg = "OPENROUTER_API_KEY must be set to initialize the chat model"
        raise ValueError(msg)
    model_name = settings.openrouter_model.split(":", 1)[1]
    return ChatOpenRouter(
        model=model_name,
        api_key=settings.openrouter_api_key,
        temperature=0.2,
    )


def create_audio_chat_model(settings: Settings) -> ChatOpenRouterAudio:
    """Create a LangChain chat model for OpenRouter's gpt-audio-mini."""
    if not settings.openrouter_api_key:
        msg = "OPENROUTER_API_KEY must be set to initialize the audio chat model"
        raise ValueError(msg)
    return ChatOpenRouterAudio(
        model=settings.openrouter_audio_model,
        api_key=settings.openrouter_api_key,
        voice=settings.openrouter_audio_voice,
        temperature=0.2,
    )


def create_agent_graph(settings: Settings, checkpointer):
    backend = CompositeBackend(default=StateBackend(), routes={})
    model = create_chat_model(settings)
    return create_deep_agent(
        model=model,
        tools=DEFAULT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        memory=memory_files(settings.app_root),
        skills=skill_paths(settings.app_root),
        backend=backend,
        checkpointer=checkpointer,
        debug=False,
        name="mvp-conversational-agent",
    )


def create_audio_agent_graph(settings: Settings, checkpointer):
    """Create a deep agent that uses gpt-audio-mini for multimodal audio I/O."""
    backend = CompositeBackend(default=StateBackend(), routes={})
    model = create_audio_chat_model(settings)

    # Set audio temp directory for storing audio responses
    model.set_audio_temp_dir(settings.audio_temp_dir)

    return create_deep_agent(
        model=model,
        tools=DEFAULT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        memory=memory_files(settings.app_root),
        skills=skill_paths(settings.app_root),
        backend=backend,
        checkpointer=checkpointer,
        debug=False,
        name="mvp-audio-agent",
    )
