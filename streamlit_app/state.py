from __future__ import annotations

import streamlit as st

from app.utils.ids import generate_id


def ensure_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_id("thread")
    if "user_id" not in st.session_state:
        st.session_state.user_id = generate_id("user")
    if "messages" not in st.session_state:
        st.session_state.messages = []
