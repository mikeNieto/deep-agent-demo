from __future__ import annotations

from collections import defaultdict

from app.schemas.chat import ConversationMessage


class ConversationService:
    def __init__(self) -> None:
        self._messages: dict[str, list[ConversationMessage]] = defaultdict(list)

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._messages[thread_id].append(ConversationMessage(role=role, content=content))

    def list_messages(self, thread_id: str) -> list[ConversationMessage]:
        return list(self._messages.get(thread_id, []))
