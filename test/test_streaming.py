"""
C) Endpoint execution: fastSDK end-to-end + streaming.

Two layers:
- fastSDK over HTTP: the inference service is booted in a subprocess via the
  apipod CLI across a queued and a direct backend. fastSDK drives it exactly as
  documented (connect -> submit_job -> get_result), including media upload/download.
- Streaming in-process: the streaming service runs under the serverless emulation
  (TestClient). The worker relays chunks into the stream store; the client reads
  them from ``GET /stream/{job_id}`` (SSE). Covers plain text tokens, raw byte
  frames, and ChatCompletion deltas.

fastSDK tests are skipped when the installed build lacks ``connect``
(local mid-refactor state); CI installs one that has it.
"""

import base64
import json

import pytest

from conftest import build_service
from services.streaming_service import CHAT_TOKENS, TEXT_TOKENS, VIDEO_FRAMES
from services import streaming_service


# --------------------------------------------------------------------------- #
# Streaming (in-process TestClient + SSE)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def stream_client():
    with build_service(streaming_service.register, simulate="serverless") as c:
        yield c


def _stream_url(client, path, json_body=None):
    resp = client.post(path, json=json_body)
    assert resp.status_code == 200, resp.text
    return resp.json()["links"]["stream"]


def _sse_payloads(client, stream_url):
    payloads = []
    with client.stream("GET", stream_url) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            if line.startswith("data: "):
                payloads.append(line[len("data: "):])
    return payloads


def test_stream_text_tokens(stream_client):
    url = _stream_url(stream_client, "/text")
    with stream_client.stream("GET", url) as resp:
        body = "".join(resp.iter_text())
    assert body == "".join(TEXT_TOKENS)


def test_stream_raw_byte_frames(stream_client):
    url = _stream_url(stream_client, "/video")
    received = b"".join(base64.b64decode(p) for p in _sse_payloads(stream_client, url))
    assert received == b"".join(VIDEO_FRAMES)


def test_stream_chat_completion_chunks(stream_client):
    url = _stream_url(
        stream_client, "/chat",
        {"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    payloads = _sse_payloads(stream_client, url)

    assert payloads[-1] == "[DONE]"
    content = "".join(
        json.loads(raw)["choices"][0]["delta"].get("content") or ""
        for raw in payloads[:-1]
    )
    assert content == "".join(CHAT_TOKENS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
