"""Unit tests for endpoint signature analysis helpers."""

from apipod.common import schemas
from apipod.engine.signatures.analysis import ast_suggests_request_stream, is_streaming_endpoint


def test_is_streaming_endpoint_for_conditional_generator_return():
    def chat(request: schemas.ChatCompletionRequest):
        if request.stream:
            return (token for token in ["a", "b"])
        return "".join(["a", "b"])

    assert ast_suggests_request_stream(chat) is True
    assert is_streaming_endpoint(chat) is True


def test_is_streaming_endpoint_false_for_plain_schema_handler():
    def chat(request: schemas.ChatCompletionRequest):
        return "done"

    assert ast_suggests_request_stream(chat) is False
    assert is_streaming_endpoint(chat) is False


def test_is_streaming_endpoint_for_yield_from():
    def stream_text():
        yield from ["a", "b"]

    assert is_streaming_endpoint(stream_text) is True
