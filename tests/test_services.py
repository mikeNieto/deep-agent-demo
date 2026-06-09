import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.schemas.chat import ChatMessageRequest
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.tts_preparation_service import TTSPreparationServiceError
from app.services.tts_service import TTSServiceError


def test_conversation_service_stores_messages() -> None:
    service = ConversationService()
    service.append("thread-1", "user", "hola")
    messages = service.list_messages("thread-1")
    assert len(messages) == 1
    assert messages[0].content == "hola"


def test_data_directories_exist() -> None:
    assert Path("data").exists()


def test_chat_service_handles_tts_errors_without_failing() -> None:
    class DummyAgentGraph:
        async def ainvoke(self, *args, **kwargs):
            return {
                "messages": [
                    SimpleNamespace(
                        content="Hello from the agent",
                        response_metadata={"model": "test-model"},
                    )
                ]
            }

    class FailingTTSService:
        def synthesize_to_mp3(self, text: str, voice: str | None = None):
            raise TTSServiceError("simulated provider timeout")

    class DummyTTSPreparationService:
        def prepare_text(self, agent_response: str) -> str:
            return f"Texto listo para TTS: {agent_response}"

    chat_service = ChatService(
        agent_graph=DummyAgentGraph(),
        conversation_service=ConversationService(),
        tts_service=FailingTTSService(),
        tts_preparation_service=DummyTTSPreparationService(),
    )

    payload = ChatMessageRequest(
        user_id="user-1",
        thread_id="thread-1",
        message="Say something",
        response_audio=True,
    )

    response = asyncio.run(chat_service.send_message(payload))

    assert response.agent_text == "Hello from the agent"
    assert response.tts_text == "Texto listo para TTS: Hello from the agent"
    assert response.audio_url is None
    assert response.audio_mime_type is None
    assert response.resolved_model == "test-model"


def test_chat_service_handles_tts_preparation_errors_without_failing() -> None:
    class DummyAgentGraph:
        async def ainvoke(self, *args, **kwargs):
            return {
                "messages": [
                    SimpleNamespace(
                        content="Hello from the agent",
                        response_metadata={"model": "test-model"},
                    )
                ]
            }

    class DummyTTSService:
        def synthesize_to_mp3(self, text: str, voice: str | None = None):
            raise AssertionError("TTS should not run when preparation fails")

    class FailingTTSPreparationService:
        def prepare_text(self, agent_response: str) -> str:
            raise TTSPreparationServiceError("simulated preparation failure")

    chat_service = ChatService(
        agent_graph=DummyAgentGraph(),
        conversation_service=ConversationService(),
        tts_service=DummyTTSService(),
        tts_preparation_service=FailingTTSPreparationService(),
    )

    payload = ChatMessageRequest(
        user_id="user-1",
        thread_id="thread-1",
        message="Say something",
        response_audio=True,
    )

    response = asyncio.run(chat_service.send_message(payload))

    assert response.agent_text == "Hello from the agent"
    assert response.tts_text is None
    assert response.audio_url is None
    assert response.audio_mime_type is None
    assert response.resolved_model == "test-model"


def test_chat_service_returns_tts_text_for_display() -> None:
    class DummyAgentGraph:
        async def ainvoke(self, *args, **kwargs):
            return {
                "messages": [
                    SimpleNamespace(
                        content="Hello from the agent",
                        response_metadata={"model": "test-model"},
                    )
                ]
            }

    class DummyTTSService:
        def synthesize_to_mp3(self, text: str, voice: str | None = None):
            assert text == "Texto listo para TTS: Hello from the agent"
            return Path("data/audio/test.mp3"), 1.0

    class DummyTTSPreparationService:
        def prepare_text(self, agent_response: str) -> str:
            return f"Texto listo para TTS: {agent_response}"

    conversation_service = ConversationService()
    chat_service = ChatService(
        agent_graph=DummyAgentGraph(),
        conversation_service=conversation_service,
        tts_service=DummyTTSService(),
        tts_preparation_service=DummyTTSPreparationService(),
    )

    payload = ChatMessageRequest(
        user_id="user-1",
        thread_id="thread-1",
        message="Say something",
        response_audio=True,
    )

    response = asyncio.run(chat_service.send_message(payload))

    assert response.agent_text == "Hello from the agent"
    assert response.tts_text == "Texto listo para TTS: Hello from the agent"
    assert response.audio_url == "/api/audio/files/test.mp3"

    messages = conversation_service.list_messages("thread-1")
    assert messages[-1].content == "Texto listo para TTS: Hello from the agent"
