"""Streaming service: all three streaming modes APIPod supports.

Endpoint types:
- ``/text``   plain text-token generator (generic generator endpoint),
- ``/video``  raw binary frames (byte generator, SSE-framed as base64),
- ``/chat``   ChatCompletionRequest with optional stream=True: a deterministic
              fake LLM covering plain replies, reasoning (``<think>`` tags),
              tool calls (Hermes ``<tool_call>`` tags) and logprobs, so clients
              can integration-test the full typed chat pipeline without a GPU.

Constants are exported so streaming tests can assert the exact expected output
without duplicating the data.
"""

import json

from apipod.common import schemas
from apipod.common.chat_parsing import ChatOutputParser

CHAT_TOKENS = ["Hello", ", ", "world", "!"]
TEXT_TOKENS = ["APIPod ", "streams ", "tokens ", "one ", "by ", "one."]
VIDEO_FRAMES = [bytes([i]) * 2048 for i in range(5)]

# Deterministic fake-LLM behavior for the /chat endpoint.
REASONING_TRIGGER = "think"
REASONING_TEXT = "The user greets; greet back."
TOOL_CALL_ARGS = {"location": "Boston"}
TOOL_ANSWER_PREFIX = "I will check."
TOOL_RESULT_ANSWER = "It is sunny in Boston."
FAKE_LOGPROB = -0.1


def _last_content(request: schemas.ChatCompletionRequest) -> str:
    content = request.messages[-1].content if request.messages else ""
    return content if isinstance(content, str) else ""


def _chosen_tool(request: schemas.ChatCompletionRequest):
    """Honor a NamedToolChoice; default to the first tool."""
    choice = request.tool_choice
    function = getattr(choice, "function", None)
    name = function.get("name") if isinstance(function, dict) else None
    for tool in request.tools:
        if tool.function.name == name:
            return tool
    return request.tools[0]


def _fake_reply_tokens(request: schemas.ChatCompletionRequest) -> list:
    """Raw 'model output' tokens for the request, tags included."""
    if request.messages and request.messages[-1].role == "tool":
        return [TOOL_RESULT_ANSWER]
    if request.tools and request.tool_choice != "none":
        call = {"name": _chosen_tool(request).function.name, "arguments": TOOL_CALL_ARGS}
        return [TOOL_ANSWER_PREFIX, "<tool_call>", json.dumps(call), "</tool_call>"]
    if REASONING_TRIGGER in _last_content(request).lower():
        return ["<think>", REASONING_TEXT, "</think>", *CHAT_TOKENS]
    return list(CHAT_TOKENS)


def _typed_deltas(tokens: list):
    """Route raw tokens through the shared parser, like the model presets do."""
    parser = ChatOutputParser()
    for token in tokens:
        yield from parser.feed(token)
    yield from parser.flush()


def _logprobs_response(request: schemas.ChatCompletionRequest) -> dict:
    def entry(token: str) -> dict:
        item = {"token": token, "logprob": FAKE_LOGPROB, "bytes": list(token.encode("utf-8"))}
        if request.top_logprobs:
            item["top_logprobs"] = [dict(item) for _ in range(min(request.top_logprobs, 2))]
        return item

    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(CHAT_TOKENS)},
            "logprobs": {"content": [entry(token) for token in CHAT_TOKENS]},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": len(CHAT_TOKENS), "total_tokens": 1 + len(CHAT_TOKENS)},
    }


def register(app):
    @app.endpoint("/text")
    def stream_text():
        yield from TEXT_TOKENS

    @app.endpoint("/video")
    def stream_video():
        yield from VIDEO_FRAMES

    @app.endpoint("/chat")
    def chat(request: schemas.ChatCompletionRequest):
        if request.stream:
            return (delta for delta in _typed_deltas(_fake_reply_tokens(request)))
        if request.logprobs:
            return _logprobs_response(request)
        return "".join(_fake_reply_tokens(request))
