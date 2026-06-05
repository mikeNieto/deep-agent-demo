from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src to path so we can import app and streamlit_app modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import streamlit as st

from app.config import get_settings
from streamlit_app.api_client import ApiClient
from streamlit_app.components import render_messages
from streamlit_app.state import ensure_session_state


def main() -> None:
    settings = get_settings()
    client = ApiClient(settings.streamlit_api_base_url)

    st.set_page_config(
        page_title="Conversational Agent MVP", page_icon="AI", layout="wide"
    )
    st.title("Conversational Agent MVP")
    st.caption("Chat por texto o audio con STT y TTS locales")

    ensure_session_state()
    render_messages(st.session_state.messages)

    with st.sidebar:
        st.write(f"Thread ID: `{st.session_state.thread_id}`")
        st.write(f"User ID: `{st.session_state.user_id}`")

    prompt = st.chat_input("Escribe tu mensaje")
    audio_file = st.audio_input("Graba un mensaje de voz")

    if audio_file is not None:
        transcription = client.transcribe_audio(
            audio_file.getvalue(), filename=audio_file.name or "recording.wav"
        )
        transcribed_text = transcription.get("text", "").strip()
        if transcribed_text:
            _send_prompt(client, transcribed_text)

    if prompt:
        _send_prompt(client, prompt)


def _send_prompt(client: ApiClient, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = client.send_message(
        {
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id,
            "message": prompt,
            "response_audio": True,
        }
    )
    audio_url = response.get("audio_url")
    if audio_url:
        audio_url = f"{get_settings().streamlit_api_base_url}{audio_url}"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.get("agent_text", ""),
            "audio_url": audio_url,
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()
