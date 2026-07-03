"""Streaming service: all three streaming modes APIPod supports.

Endpoint types:
- ``/text``   plain text-token generator (generic generator endpoint),
- ``/video``  raw binary frames (byte generator, SSE-framed as base64),
- ``/chat``   ChatCompletionRequest with optional stream=True (schema endpoint:
              returns token deltas wrapped into ChatCompletionChunk SSE events).

Constants are exported so streaming tests can assert the exact expected output
without duplicating the data.
"""

from apipod.common import schemas

CHAT_TOKENS = ["Hello", ", ", "world", "!"]
TEXT_TOKENS = ["APIPod ", "streams ", "tokens ", "one ", "by ", "one."]
VIDEO_FRAMES = [bytes([i]) * 2048 for i in range(5)]


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
            return (token for token in CHAT_TOKENS)
        return "".join(CHAT_TOKENS)
