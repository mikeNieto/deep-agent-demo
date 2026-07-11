from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class UsageTracker:
    _defaults: dict[str, int | float] = {
        "total_requests": 0,
        "stt_calls": 0,
        "agent_calls": 0,
        "tts_calls": 0,
        "web_search_calls": 0,
        "agent_input_tokens": 0,
        "agent_output_tokens": 0,
        "stt_audio_seconds": 0.0,
        "tts_chars": 0,
    }

    def __init__(self, path: Path) -> None:
        self._path = path
        self._counts: dict[str, int | float] = self._load()

    def _load(self) -> dict[str, int | float]:
        try:
            if self._path.exists():
                saved: dict[str, int | float] = {}  # type: ignore[type-arg]
                raw = json.loads(self._path.read_text())
                for k, v in self._defaults.items():
                    saved[k] = type(v)(raw.get(k, v))  # type: ignore[arg-type]
                return saved
        except Exception:
            pass
        return dict(self._defaults)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._counts, indent=2))
        except Exception:
            pass

    def _inc(self, key: str, delta: int | float = 1) -> None:
        with _lock:
            self._counts[key] = self._counts.get(key, 0) + delta
            self._save()

    def add_agent_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        with _lock:
            self._counts["total_requests"] = int(self._counts.get("total_requests", 0)) + 1
            self._counts["agent_calls"] = int(self._counts.get("agent_calls", 0)) + 1
            self._counts["agent_input_tokens"] = int(self._counts.get("agent_input_tokens", 0)) + input_tokens
            self._counts["agent_output_tokens"] = int(self._counts.get("agent_output_tokens", 0)) + output_tokens
            self._save()

    def add_stt(self, audio_duration_s: float = 0) -> None:
        self._counts["total_requests"] = int(self._counts.get("total_requests", 0)) + 1
        self._counts["stt_calls"] = int(self._counts.get("stt_calls", 0)) + 1
        self._counts["stt_audio_seconds"] = float(self._counts.get("stt_audio_seconds", 0)) + audio_duration_s
        self._save()

    def add_tts(self, chars: int = 0) -> None:
        self._counts["total_requests"] = int(self._counts.get("total_requests", 0)) + 1
        self._counts["tts_calls"] = int(self._counts.get("tts_calls", 0)) + 1
        self._counts["tts_chars"] = int(self._counts.get("tts_chars", 0)) + chars
        self._save()

    def add_web_search(self) -> None:
        self._counts["total_requests"] = int(self._counts.get("total_requests", 0)) + 1
        self._counts["web_search_calls"] = int(self._counts.get("web_search_calls", 0)) + 1
        self._save()

    def summary(self) -> dict[str, int | float]:
        with _lock:
            return dict(self._counts)

    def log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "Usage: reqs=%s stt=%s agent=%s tts=%s search=%s "
            "agent_in_tok=%s agent_out_tok=%s stt_audio_s=%s tts_chars=%s",
            s["total_requests"],
            s["stt_calls"],
            s["agent_calls"],
            s["tts_calls"],
            s["web_search_calls"],
            s["agent_input_tokens"],
            s["agent_output_tokens"],
            s["stt_audio_seconds"],
            s["tts_chars"],
        )


_tracker: UsageTracker | None = None


def get_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        from app.config import get_settings

        settings = get_settings()
        path = settings.audio_temp_dir.parent / "usage.json"
        _tracker = UsageTracker(path)
    return _tracker
