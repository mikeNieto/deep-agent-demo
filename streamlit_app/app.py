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
    st.caption("Prueba las dos modalidades de agente en paralelo")

    ensure_session_state()

    with st.sidebar:
        st.write(f"Thread ID: `{st.session_state.thread_id}`")
        st.write(f"User ID: `{st.session_state.user_id}`")

    tab1, tab2 = st.tabs(
        [
            "Agente Actual (STT + LLM + TTS)",
            "Agente OpenRouter Audio (gpt-audio-mini)",
        ]
    )

    with tab1:
        _render_current_agent_tab(client)

    with tab2:
        _render_openrouter_audio_tab(client)


def _render_current_agent_tab(client: ApiClient) -> None:
    """Render the current agent that uses local STT + LLM + TTS."""
    st.subheader("Agente Conversacional (STT local + LLM + TTS)")
    st.caption(
        "Transcripción con faster-whisper, respuesta con LLM, voz con OpenRouter TTS"
    )

    render_messages(st.session_state.messages)

    prompt = st.chat_input("Escribe tu mensaje", key="chat_input_current")

    if prompt:
        _send_prompt(client, prompt)

    audio_file = st.audio_input("Graba un mensaje de voz", key="audio_input_current")
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        if audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            with st.spinner("Transcribiendo audio..."):
                transcription = client.transcribe_audio(
                    audio_bytes, filename=audio_file.name or "recording.wav"
                )
                st.session_state.pending_audio_text = transcription.get(
                    "text", ""
                ).strip()

    st.divider()

    if st.session_state.pending_audio_text:
        col1, col2 = st.columns([7, 1])
        with col1:
            st.info(f"📝 Transcrito: {st.session_state.pending_audio_text}")
        with col2:
            if st.button("Enviar", key="send_audio_current"):
                _send_prompt(client, st.session_state.pending_audio_text)
                st.session_state.pending_audio_text = None
                st.session_state.last_audio_bytes = None
                st.rerun()


def _render_openrouter_audio_tab(client: ApiClient) -> None:
    """Render the OpenRouter multimodal audio agent (gpt-audio-mini)."""
    st.subheader("Agente OpenRouter Audio (gpt-audio-mini)")
    st.caption(
        "Envía audio directamente al modelo multimodal y recibe texto + audio de respuesta"
    )

    render_messages(st.session_state.openrouter_messages)

    # Prompt configuration
    with st.expander("Configuración del prompt"):
        prompt_text = st.text_area(
            "Prompt para el modelo",
            value="Analiza este audio y responde con tu propia voz.",
            key="openrouter_prompt",
        )
        voice = st.selectbox(
            "Voz de respuesta",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0,
            key="openrouter_voice",
        )

    audio_file = st.audio_input(
        "Graba o sube un audio para el agente", key="audio_input_openrouter"
    )

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        if audio_bytes != st.session_state.last_openrouter_audio_bytes:
            st.session_state.last_openrouter_audio_bytes = audio_bytes

            # Show the recorded audio
            st.audio(audio_bytes, format="audio/wav")

            with st.spinner("Procesando audio con OpenRouter gpt-audio-mini..."):
                try:
                    response = client.openrouter_audio_process(
                        audio_bytes=audio_bytes,
                        filename=audio_file.name or "recording.wav",
                        prompt=prompt_text,
                        voice=voice,
                        user_id=st.session_state.user_id,
                        thread_id=st.session_state.thread_id,
                    )

                    # Add user message
                    st.session_state.openrouter_messages.append(
                        {
                            "role": "user",
                            "content": f"🎤 Audio enviado (prompt: {prompt_text})",
                        }
                    )

                    # Add assistant response
                    audio_url = response.get("audio_url")
                    if audio_url:
                        audio_url = (
                            f"{get_settings().streamlit_api_base_url}{audio_url}"
                        )

                    st.session_state.openrouter_messages.append(
                        {
                            "role": "assistant",
                            "content": response.get("text", ""),
                            "audio_url": audio_url,
                        }
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error procesando audio: {e}")


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
