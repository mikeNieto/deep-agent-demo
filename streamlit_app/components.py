from __future__ import annotations

import streamlit as st


def render_messages(messages: list[dict]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            audio_url = message.get("audio_url")
            if audio_url:
                st.audio(audio_url, format="audio/mp3")
