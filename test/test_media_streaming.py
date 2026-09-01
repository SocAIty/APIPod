"""
Media-toolkit streaming capabilities inside APIPod endpoints.

Uploads arrive as lazy MediaFile views over the request spool (no full read at
wrap time). Endpoints stream them outbound with ``stream_to`` (sync) and
``stream_to_async`` (async). On queued intents the parameters are materialized
before enqueueing, so the job owns its bytes after the request spool closes.
"""

import time

import pytest

from conftest import AUDIO_FILE, build_service
from apipod import MediaFile

if not hasattr(MediaFile, "stream_to"):
    pytest.skip(
        "media-toolkit has no stream_to yet; this suite needs a later media-toolkit.",
        allow_module_level=True,
    )


class _CollectSink:
    def __init__(self):
        self.data = b""

    def write(self, chunk: bytes):
        self.data += chunk

    def finalize(self):
        return len(self.data)


class _AsyncCollectSink:
    def __init__(self):
        self.data = b""

    async def write(self, chunk: bytes):
        self.data += chunk

    async def finalize(self):
        return len(self.data)


def register(app):
    @app.endpoint("/stream_sync")
    def stream_sync(file: MediaFile):
        sink = _CollectSink()
        written = file.stream_to(sink)
        return {
            "written": written,
            "lazy": getattr(file, "is_lazy", False),
            "content_type": file.content_type,
            "detected": file.detection.content_type,
        }

    @app.endpoint("/stream_async")
    async def stream_async(file: MediaFile):
        sink = _AsyncCollectSink()
        written = await file.stream_to_async(sink)
        return {
            "written": written,
            "lazy": getattr(file, "is_lazy", False),
            "content_type": file.content_type,
            "detected": file.detection.content_type,
        }


@pytest.fixture(scope="module")
def dev_client():
    with build_service(register) as c:
        yield c


@pytest.fixture(scope="module")
def serverless_client():
    with build_service(register, simulate="serverless") as c:
        yield c


def _post_wav(client, path):
    with open(AUDIO_FILE, "rb") as fh:
        return client.post(path, files={"file": (AUDIO_FILE.name, fh, "audio/wav")})


def test_stream_to_sync_direct(dev_client):
    resp = _post_wav(dev_client, "/stream-sync")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == AUDIO_FILE.stat().st_size
    if hasattr(MediaFile, "is_lazy"):
        assert body["lazy"] is True  # wrap of the request spool, no owned copy
    assert body["detected"] in ("audio/wav", "audio/wave")


def test_stream_to_async_direct(dev_client):
    resp = _post_wav(dev_client, "/stream-async")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == AUDIO_FILE.stat().st_size
    if hasattr(MediaFile, "is_lazy"):
        assert body["lazy"] is True
    assert body["detected"] in ("audio/wav", "audio/wave")


def _poll_job(client, submit_response, timeout=15.0):
    status_url = submit_response.json()["links"]["status"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(status_url).json()
        if body.get("result") is not None:
            return body["result"]
        if body.get("status") in ("failed", "timeout", "not_found"):
            pytest.fail(f"job ended unexpectedly: {body}")
        time.sleep(0.1)
    pytest.fail("job did not finish in time")


def test_stream_to_queued_materializes(serverless_client):
    """Queued jobs run after the request; params must own their bytes."""
    submit = _post_wav(serverless_client, "/stream-sync")
    assert submit.status_code == 200, submit.text
    result = _poll_job(serverless_client, submit)
    assert result["written"] == AUDIO_FILE.stat().st_size
    if hasattr(MediaFile, "is_lazy"):
        assert result["lazy"] is False  # materialized before enqueue


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
