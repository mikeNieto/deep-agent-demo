"""Agent package — build compatibility shim.

Required by uv_build: the project is named "agent" in pyproject.toml,
and the build backend expects `src/agent/__init__.py` for the src-layout.
"""

from app.agent import create_agent_graph
from app.audio import synthesize_speech, transcribe_audio
from app.config import Settings, get_settings
from app.usage import UsageTracker

__all__ = [
    "create_agent_graph",
    "synthesize_speech",
    "transcribe_audio",
    "Settings",
    "get_settings",
    "UsageTracker",
]
