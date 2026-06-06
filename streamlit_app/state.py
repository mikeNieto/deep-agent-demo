from __future__ import annotations

import sys
from pathlib import Path

# Add src to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

from app.utils.ids import generate_id


def ensure_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_id("thread")
    if "user_id" not in st.session_state:
        st.session_state.user_id = generate_id("user")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_audio_text" not in st.session_state:
        st.session_state.pending_audio_text = None
    if "last_audio_bytes" not in st.session_state:
        st.session_state.last_audio_bytes = None
    # OpenRouter audio agent state
    if "openrouter_messages" not in st.session_state:
        st.session_state.openrouter_messages = []
    if "last_openrouter_audio_bytes" not in st.session_state:
        st.session_state.last_openrouter_audio_bytes = None
