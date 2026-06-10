from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain_openrouter import ChatOpenRouter

from app.agents.memory import memory_files, skill_paths
from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tools import DEFAULT_TOOLS
from app.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenRouter:
    if not settings.openrouter_api_key:
        msg = "OPENROUTER_API_KEY must be set to initialize the chat model"
        raise ValueError(msg)
    model_name = settings.deepagent_model
    return ChatOpenRouter(
        model=model_name,
        api_key=settings.openrouter_api_key,
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
