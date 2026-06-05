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
