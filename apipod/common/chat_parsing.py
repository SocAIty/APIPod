"""Typed parsing of raw chat-model output.

Open chat models mark structure with inline tags: reasoning inside
``<think>...</think>`` (Qwen3 / DeepSeek-R1 convention) and tool invocations
inside ``<tool_call>{json}</tool_call>`` (Hermes format, used by Qwen and most
tool-tuned open models). :class:`ChatOutputParser` turns raw decoded text into
typed deltas shaped like ``socaity_schemas.ChatDelta`` so the blocking and the
streaming generation paths share one parse:

- plain content        -> ``str``
- reasoning            -> ``{"reasoning_content": str}``
- one finished tool call -> ``{"tool_calls": [{index, id, type, function}]}``

The parser is incremental: ``feed()`` accepts fragments split at arbitrary
points (token streams) and holds back text that could still turn into a tag.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Iterator, List, Optional, Tuple, Union

Delta = Union[str, dict]

THINK_TAGS = ("<think>", "</think>")
TOOL_CALL_TAGS = ("<tool_call>", "</tool_call>")


class ChatOutputParser:
    """Incremental raw-text to typed chat-delta parser.

    One instance parses one assistant turn. ``feed()`` returns the deltas that
    are safe to emit so far; ``flush()`` releases anything held back at the
    end of the stream.
    """

    def __init__(
        self,
        think_tags: Tuple[str, str] = THINK_TAGS,
        tool_call_tags: Tuple[str, str] = TOOL_CALL_TAGS,
    ):
        self.think_open, self.think_close = think_tags
        self.tool_open, self.tool_close = tool_call_tags
        self._buffer = ""
        self._tool_buffer = ""
        self._mode = "content"  # content | reasoning | tool
        self._tool_index = 0
        self.saw_tool_call = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, text: str) -> List[Delta]:
        self._buffer += text
        return list(self._drain(final=False))

    def flush(self) -> List[Delta]:
        """Release held-back text; call once after the last ``feed``."""
        deltas = list(self._drain(final=True))
        if self._tool_buffer:
            # Stream ended inside an unterminated tool call: surface it raw.
            deltas.extend(self._emit_text(self._tool_buffer, mode="content"))
            self._tool_buffer = ""
        return deltas

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _markers(self) -> Tuple[str, ...]:
        if self._mode == "content":
            return (self.think_open, self.tool_open)
        if self._mode == "reasoning":
            return (self.think_close,)
        return (self.tool_close,)

    def _drain(self, final: bool) -> Iterator[Delta]:
        while self._buffer:
            match = self._find_marker()
            if match is None:
                holdback = 0 if final else self._prefix_holdback()
                cut = len(self._buffer) - holdback
                emit, self._buffer = self._buffer[:cut], self._buffer[cut:]
                if emit:
                    yield from self._emit_text(emit, self._mode)
                return
            start, marker = match
            before = self._buffer[:start]
            self._buffer = self._buffer[start + len(marker):]
            if before:
                yield from self._emit_text(before, self._mode)
            yield from self._on_marker(marker)

    def _find_marker(self) -> Optional[Tuple[int, str]]:
        """Earliest complete occurrence of a watched marker in the buffer."""
        best: Optional[Tuple[int, str]] = None
        for marker in self._markers():
            index = self._buffer.find(marker)
            if index != -1 and (best is None or index < best[0]):
                best = (index, marker)
        return best

    def _prefix_holdback(self) -> int:
        """Length of the longest buffer suffix that could still become a marker."""
        for length in range(min(len(self._buffer), max(map(len, self._markers())) - 1), 0, -1):
            suffix = self._buffer[-length:]
            if any(marker.startswith(suffix) for marker in self._markers()):
                return length
        return 0

    def _emit_text(self, text: str, mode: str) -> Iterator[Delta]:
        if mode == "reasoning":
            yield {"reasoning_content": text}
        elif mode == "tool":
            self._tool_buffer += text
        else:
            yield text

    def _on_marker(self, marker: str) -> Iterator[Delta]:
        if marker == self.think_open:
            self._mode = "reasoning"
        elif marker == self.think_close:
            self._mode = "content"
        elif marker == self.tool_open:
            self._mode = "tool"
            self._tool_buffer = ""
        elif marker == self.tool_close:
            self._mode = "content"
            yield from self._finish_tool_call()

    def _finish_tool_call(self) -> Iterator[Delta]:
        raw, self._tool_buffer = self._tool_buffer, ""
        parsed = _parse_tool_payload(raw)
        if parsed is None:
            # Unparseable tool payload: hand it to the client as plain content.
            yield raw
            return
        name, arguments = parsed
        self.saw_tool_call = True
        yield {
            "tool_calls": [{
                "index": self._tool_index,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                },
            }]
        }
        self._tool_index += 1


# ---------------------------------------------------------------------------
# Tool payload formats
# ---------------------------------------------------------------------------

# Qwen3.5 / Qwen3-Coder XML-style function calls (inside the <tool_call> block):
#   <function=get_weather>
#   <parameter=location>
#   Boston
#   </parameter>
#   </function>
_FUNCTION_RE = re.compile(r"<function=([\w.\-]+)>(.*?)</function>", re.S)
_PARAMETER_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.S)


def _parse_tool_payload(raw: str) -> Optional[Tuple[str, Union[str, dict]]]:
    """Parse one tool-call payload: Hermes JSON or Qwen XML-style. None when unparseable."""
    raw = raw.strip()
    try:
        call = json.loads(raw)
        return call["name"], call.get("arguments", {})
    except (ValueError, KeyError, TypeError):
        pass

    match = _FUNCTION_RE.search(raw)
    if match is None:
        return None
    arguments = {}
    for key, value in _PARAMETER_RE.findall(match.group(2)):
        try:
            arguments[key] = json.loads(value)  # typed values (numbers, bools, objects)
        except ValueError:
            arguments[key] = value  # plain strings
    return match.group(1), arguments


# ---------------------------------------------------------------------------
# Delta combination (full message from deltas)
# ---------------------------------------------------------------------------


def combine_deltas(deltas: List[Delta]) -> dict:
    """Fold typed deltas into one assistant message dict (ChatCompletionMessage shape)."""
    content: List[str] = []
    reasoning: List[str] = []
    tool_calls: List[dict] = []
    for delta in deltas:
        if isinstance(delta, str):
            content.append(delta)
            continue
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        for call in delta.get("tool_calls") or []:
            tool_calls.append({key: value for key, value in call.items() if key != "index"})

    message: dict = {"role": "assistant", "content": "".join(content).strip() or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning).strip()
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def parse_chat_output(text: str, **parser_kwargs) -> dict:
    """Parse one full generated text into an assistant message dict."""
    parser = ChatOutputParser(**parser_kwargs)
    return combine_deltas(parser.feed(text) + parser.flush())
