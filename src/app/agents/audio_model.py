"""LangChain wrapper for OpenRouter's gpt-audio-mini model."""

from __future__ import annotations

import base64
import json
import logging
import struct
from typing import Any, Iterator, List, Optional, Sequence, Union

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatOpenRouterAudio(BaseChatModel):
    """
    LangChain chat model wrapper for OpenRouter's gpt-audio-mini.

    This model supports multimodal input (text + audio) and returns
    both text and audio responses via streaming SSE.
    """

    model: str = "openai/gpt-audio-mini"
    api_key: str
    voice: str = "alloy"
    temperature: float = 0.2
    timeout: float = 180.0

    # Store bound tools for function calling
    bound_tools: Optional[List[dict]] = Field(default=None, exclude=True)
    tool_choice: Optional[str] = Field(default=None, exclude=True)

    # Store audio data from last response
    _last_audio_path: Optional[str] = PrivateAttr(default=None)
    _audio_temp_dir: Optional[Any] = PrivateAttr(default=None)

    # Store pending audio input to inject into the next request
    _pending_audio_input: Optional[dict] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def set_audio_temp_dir(self, temp_dir: Any) -> None:
        """Set the directory for storing audio responses."""
        self._audio_temp_dir = temp_dir

    def get_last_audio_path(self) -> Optional[str]:
        """Get the path to the last generated audio file."""
        return self._last_audio_path

    def set_pending_audio_input(self, audio_data: str, audio_format: str) -> None:
        """Set audio input to be injected into the next request."""
        self._pending_audio_input = {
            "data": audio_data,
            "format": audio_format,
        }

    def clear_pending_audio_input(self) -> None:
        """Clear the pending audio input after it's been used."""
        self._pending_audio_input = None

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, callable, BaseTool]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "ChatOpenRouterAudio":
        """
        Bind tools to the model for function calling.

        This creates a new instance with the tools bound to it.
        """
        # Convert tools to OpenAI function format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                formatted_tools.append(tool)
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                # LangChain tool
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.args_schema.schema()
                            if hasattr(tool, "args_schema") and tool.args_schema
                            else {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )
            elif callable(tool):
                # Simple callable - create a basic tool definition
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "__name__", "unknown_tool"),
                            "description": getattr(tool, "__doc__", "") or "",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )

        # Create a new instance with bound tools
        new_instance = self.copy()
        new_instance.bound_tools = formatted_tools
        new_instance.tool_choice = tool_choice
        return new_instance

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response with audio support."""

        # Convert LangChain messages to OpenRouter format
        openrouter_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    # Multimodal content (text + audio)
                    openrouter_messages.append({"role": "user", "content": content})
                else:
                    # Text only
                    openrouter_messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": str(content)}],
                        }
                    )
            elif isinstance(msg, AIMessage):
                # Handle tool calls in AI messages
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    openrouter_messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content if msg.content else None,
                            "tool_calls": [
                                {
                                    "id": tc.get("id", f"call_{i}"),
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": json.dumps(tc["args"])
                                        if isinstance(tc["args"], dict)
                                        else tc["args"],
                                    },
                                }
                                for i, tc in enumerate(msg.tool_calls)
                            ],
                        }
                    )
                else:
                    openrouter_messages.append(
                        {"role": "assistant", "content": str(msg.content)}
                    )
            elif isinstance(msg, ToolMessage):
                # Tool response
                openrouter_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": str(msg.content),
                    }
                )

        # Inject pending audio into the last user message if present
        if self._pending_audio_input:
            logger.info(
                f"Found pending audio input, format: {self._pending_audio_input.get('format')}, data length: {len(self._pending_audio_input.get('data', ''))} chars"
            )

            # Find the last user message and inject audio
            for i in range(len(openrouter_messages) - 1, -1, -1):
                if openrouter_messages[i]["role"] == "user":
                    last_user_msg = openrouter_messages[i]
                    content = last_user_msg["content"]

                    logger.info(
                        f"Found last user message at index {i}, content type: {type(content)}"
                    )

                    # If content is a string, convert to multimodal format
                    if isinstance(content, str):
                        logger.info(
                            f"Converting string content to multimodal: {content[:100]}"
                        )
                        last_user_msg["content"] = [
                            {"type": "text", "text": content},
                            {
                                "type": "input_audio",
                                "input_audio": self._pending_audio_input,
                            },
                        ]
                    # If content is already a list, append audio
                    elif isinstance(content, list):
                        # Check if audio is already present
                        has_audio = any(
                            item.get("type") == "input_audio" for item in content
                        )
                        if not has_audio:
                            logger.info(
                                f"Appending audio to existing list with {len(content)} items"
                            )
                            content.append(
                                {
                                    "type": "input_audio",
                                    "input_audio": self._pending_audio_input,
                                }
                            )
                        else:
                            logger.info("Audio already present in message")

                    logger.info("Injected pending audio into last user message")
                    break

            # Clear the pending audio after injection
            self._pending_audio_input = None
            logger.info("Cleared pending audio input")
        else:
            logger.info("No pending audio input found")

        payload = {
            "model": self.model,
            "modalities": ["text", "audio"],
            "audio": {
                "voice": self.voice,
                "format": "pcm16",  # Required for streaming
            },
            "messages": openrouter_messages,
            "temperature": self.temperature,
            "stream": True,
        }

        # Add tools if bound
        if self.bound_tools:
            payload["tools"] = self.bound_tools
            if self.tool_choice:
                payload["tool_choice"] = self.tool_choice

        logger.info(
            "Sending request to OpenRouter: model=%s, voice=%s, tools=%d",
            self.model,
            self.voice,
            len(self.bound_tools) if self.bound_tools else 0,
        )

        # Make streaming request
        client = httpx.Client()
        try:
            with client.stream(
                "POST",
                OPENROUTER_CHAT_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Deep Agent Demo",
                },
                json=payload,
                timeout=self.timeout,
            ) as response:
                if response.status_code != 200:
                    response.read()
                    error_body = response.text
                    logger.error(
                        "OpenRouter API error: %s %s",
                        response.status_code,
                        error_body,
                    )
                    raise RuntimeError(
                        f"OpenRouter API failed: {response.status_code} {error_body}"
                    )

                # Parse SSE stream
                audio_data_chunks: list[str] = []
                transcript_chunks: list[str] = []
                tool_calls_data: dict[int, dict] = {}  # index -> {id, name, arguments}

                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Could not parse SSE chunk: %s", data_str[:200])
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Collect audio data
                    audio = delta.get("audio", {})
                    if audio.get("data"):
                        audio_data_chunks.append(audio["data"])

                    # Collect transcript
                    if audio.get("transcript"):
                        transcript_chunks.append(audio["transcript"])

                    # Also check delta.content
                    content = delta.get("content")
                    if content and not audio.get("transcript"):
                        transcript_chunks.append(content)

                    # Collect tool calls
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }

                            # Update tool call data
                            if tc.get("id"):
                                tool_calls_data[idx]["id"] = tc["id"]

                            function = tc.get("function", {})
                            if function.get("name"):
                                tool_calls_data[idx]["name"] = function["name"]
                            if function.get("arguments"):
                                tool_calls_data[idx]["arguments"] += function[
                                    "arguments"
                                ]

        finally:
            client.close()

        # Combine transcript
        text_response = "".join(transcript_chunks).strip()
        logger.info("OpenRouter response: %s", text_response[:200])

        # Save audio if present
        self._last_audio_path = None
        if audio_data_chunks and self._audio_temp_dir:
            try:
                from app.utils.ids import generate_id
                from app.storage.files import ensure_parent

                full_audio_b64 = "".join(audio_data_chunks)
                pcm_bytes = base64.b64decode(full_audio_b64)
                wav_bytes = self._pcm16_to_wav(pcm_bytes, sample_rate=24000)

                audio_path = ensure_parent(
                    self._audio_temp_dir / f"{generate_id('openrouter-audio')}.wav"
                )
                audio_path.write_bytes(wav_bytes)
                self._last_audio_path = str(audio_path)
                logger.info(
                    "Saved audio response to %s (%d bytes)",
                    audio_path,
                    len(wav_bytes),
                )
            except Exception as e:
                logger.warning("Failed to save audio response: %s", e)

        # Parse tool calls if present
        tool_calls = []
        if tool_calls_data:
            for idx in sorted(tool_calls_data.keys()):
                tc_data = tool_calls_data[idx]
                try:
                    args = (
                        json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool call arguments: %s", tc_data["arguments"]
                    )
                    args = {}

                tool_calls.append(
                    {
                        "id": tc_data["id"] or f"call_{idx}",
                        "name": tc_data["name"],
                        "args": args,
                    }
                )

            logger.info("Tool calls: %s", tool_calls)

        # Create AIMessage with text response and tool calls
        if tool_calls:
            ai_message = AIMessage(
                content=text_response,
                tool_calls=tool_calls,
            )
        else:
            ai_message = AIMessage(content=text_response)

        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
            llm_output={"model": self.model},
        )

    @staticmethod
    def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
        """Convert raw PCM16 bytes to WAV format."""
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_bytes)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        return header + pcm_bytes

    @property
    def _llm_type(self) -> str:
        return "openrouter-audio"
