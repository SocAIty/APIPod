"""
Streaming over the serverless-localhost emulation.

APIPod on ``compute="serverless", provider="localhost"`` mimics a real
deployment: streaming endpoints are queued, the in-process worker produces their
chunks into a :class:`StreamStore` (the in-memory ``LocalStreamStore`` by
default), and the client consumes them from ``GET /stream/{job_id}``. A client
that polls ``GET /status/{job_id}`` instead receives the full aggregated result
once the job has finished.

These tests exercise three chunk kinds:
    1. plain text tokens (any generator endpoint),
    2. binary "video" chunks (bytes, base64-framed over SSE),
    3. ChatCompletion deltas (schema endpoint with ``stream=true``).

Run directly:   python test/test_streaming.py
Or with pytest: pytest test/test_streaming.py
"""

import base64
import time

from fastapi.testclient import TestClient

from apipod import APIPod, LocalStreamStore
from apipod.common import schemas


# Deterministic "video" payload: a few non-trivial binary frames.
VIDEO_FRAMES = [bytes([i]) * 2048 for i in range(5)]
VIDEO_BYTES = b"".join(VIDEO_FRAMES)

TEXT_TOKENS = ["APIPod ", "streams ", "tokens ", "one ", "by ", "one."]
CHAT_TOKENS = ["Hello", ", ", "world", "!"]


def build_client() -> TestClient:
    """Build a serverless-localhost app with streaming endpoints."""
    app = APIPod(orchestrator="local", compute="serverless", provider="localhost")

    @app.endpoint("/text")
    def stream_text():
        for token in TEXT_TOKENS:
            time.sleep(0.02)
            yield token

    @app.endpoint("/video")
    def stream_video():
        # "Any generator endpoint": here it yields raw video frame bytes.
        for frame in VIDEO_FRAMES:
            time.sleep(0.02)
            yield frame

    @app.endpoint("/chat")
    def chat(request: schemas.ChatCompletionRequest):
        if request.stream:
            def token_gen():
                for token in CHAT_TOKENS:
                    time.sleep(0.02)
                    yield token
            return token_gen()
        return "".join(CHAT_TOKENS)

    fastapi_app = app.app
    fastapi_app.include_router(app)
    return TestClient(fastapi_app)


def _submit(client: TestClient, path: str, json_body=None) -> str:
    """POST an endpoint, return the stream URL from the JobResult."""
    resp = client.post(path, json=json_body)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "job_id" in body, body
    stream_url = body["links"]["stream"]
    assert stream_url, body
    return stream_url


def _collect_sse_data(client: TestClient, stream_url: str) -> list[str]:
    """Open the SSE stream and return the payloads of every ``data:`` line."""
    payloads: list[str] = []
    with client.stream("GET", stream_url) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            if not line or line.startswith(":"):  # keep-alive / blank
                continue
            if line.startswith("data: "):
                payloads.append(line[len("data: "):])
    return payloads


def test_default_stream_store_is_local():
    """Serverless localhost gets a LocalStreamStore by default; plain FastAPI does not."""
    serverless = APIPod(orchestrator="local", compute="serverless", provider="localhost")
    assert isinstance(serverless.stream_store, LocalStreamStore)

    plain = APIPod(orchestrator="local", compute="dedicated", provider="localhost")
    assert plain.stream_store is None


def test_stream_text_chunks():
    client = build_client()
    stream_url = _submit(client, "/text")

    with client.stream("GET", stream_url) as resp:
        assert resp.status_code == 200, resp.text
        body = "".join(resp.iter_text())

    assert body == "".join(TEXT_TOKENS)


def test_stream_video_bytes():
    client = build_client()
    stream_url = _submit(client, "/video")

    payloads = _collect_sse_data(client, stream_url)
    received = b"".join(base64.b64decode(p) for p in payloads)

    assert received == VIDEO_BYTES


def test_stream_chat_completion():
    client = build_client()
    stream_url = _submit(client, "/chat", {"messages": [{"role": "user", "content": "hi"}], "stream": True})

    payloads = _collect_sse_data(client, stream_url)
    assert payloads, "expected at least one chat chunk"
    assert payloads[-1] == "[DONE]"

    import json
    content = ""
    for raw in payloads[:-1]:
        chunk = json.loads(raw)
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            content += delta

    assert content == "".join(CHAT_TOKENS)


def test_status_returns_full_result_when_not_streaming():
    """A client that polls /status (instead of /stream) gets the full result."""
    client = build_client()
    resp = client.post("/text")
    assert resp.status_code == 200, resp.text
    status_url = resp.json()["links"]["status"]

    # Poll tightly until the job reaches a terminal state and exposes its result.
    full_result = None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s = client.get(status_url)
        body = s.json()
        if body.get("result") is not None:
            full_result = body["result"]
            break
        if body.get("status") in ("completed", "failed", "not_found"):
            full_result = body.get("result")
            break
    assert full_result == "".join(TEXT_TOKENS)


def main() -> int:
    tests = [
        test_default_stream_store_is_local,
        test_stream_text_chunks,
        test_stream_video_bytes,
        test_stream_chat_completion,
        test_status_returns_full_result_when_not_streaming,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"  FAIL  {test.__name__}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
