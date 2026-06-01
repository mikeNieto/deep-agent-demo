from pathlib import Path

from app.services.conversation_service import ConversationService


def test_conversation_service_stores_messages() -> None:
    service = ConversationService()
    service.append("thread-1", "user", "hola")
    messages = service.list_messages("thread-1")
    assert len(messages) == 1
    assert messages[0].content == "hola"


def test_data_directories_exist() -> None:
    assert Path("data").exists()
