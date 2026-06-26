"""
Shared test infrastructure for the APIPod suite.

The suite is built on a few small primitives so individual test files stay
declarative and DRY:

- ``FASTAPI_CONFIGS`` / ``QUEUE_CONFIGS`` — the canonical matrix of APIPod run
  intents (development, serverless, dedicated, runpod, ...). Parametrize over
  it to assert behaviour holds across every backend.
- ``build_service(register, **config)`` — build an app from a callback that
  registers endpoints and hand back a wired FastAPI ``TestClient`` (in-process,
  no socket). Used by the config / openapi / core / schema tests.
- ``live_service(simulate=...)`` — boot the example service in a subprocess via
  the real ``apipod`` CLI (the same ``start`` / ``simulate`` a user runs),
  yield its base URL, and shut it down afterwards. Used by the fastSDK
  end-to-end execution tests.

A registrar is just ``def register(app): ...``; keeping endpoints out of the
helpers lets one helper serve config, openapi, schema and execution tests alike.
"""

import contextlib
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apipod import APIPod

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Test asset files (used by the media upload/download tests)
# --------------------------------------------------------------------------- #
FILES_DIR = Path(__file__).parent / "files"
IMAGE_FILE = FILES_DIR / "test_image.png"
AUDIO_FILE = FILES_DIR / "test_audio.wav"
VIDEO_FILE = FILES_DIR / "test_video.mp4"

# Entrypoint booted by the fastSDK end-to-end execution tests.
INFERENCE_SERVICE = Path(__file__).parent / "services" / "core_service.py"


# --------------------------------------------------------------------------- #
# Run-intent matrix. Each entry is the kwargs passed to APIPod(...).
# --------------------------------------------------------------------------- #
# FastAPI-backed intents resolve to SocaityFastAPIRouter and expose a real HTTP
# app (the only ones that can serve /openapi.json or run under TestClient).
FASTAPI_CONFIGS = [
    pytest.param({}, id="development"),
    pytest.param({"simulate": "serverless"}, id="serverless"),
    pytest.param({"simulate": "dedicated"}, id="dedicated"),
    pytest.param({"simulate": "dedicated-azure"}, id="dedicated-azure"),
    pytest.param({"simulate": "serverless-runpod"}, id="serverless-runpod"),
    pytest.param({"simulate": "serverless-azure"}, id="serverless-azure-fallback"),
]

# Intents that boot a job queue (serverless emulation): endpoints return a
# JobResult processed by the in-process worker.
QUEUE_CONFIGS = [
    pytest.param({"simulate": "serverless"}, id="serverless"),
    pytest.param({"simulate": "serverless-runpod"}, id="serverless-runpod"),
]


def build_app(register, **config):
    """Create an APIPod app for ``config`` and register endpoints via callback."""
    app = APIPod(**config)
    register(app)
    return app


@contextlib.contextmanager
def build_service(register, **config):
    """
    Yield a ``TestClient`` for an app built from ``register`` under ``config``.

    The client is entered as a context manager so the app lifespan runs: that
    starts the in-process queue worker for serverless intents.
    """
    app = build_app(register, **config)
    fastapi_app = app.app
    fastapi_app.include_router(app)
    with TestClient(fastapi_app) as client:
        yield client


# --------------------------------------------------------------------------- #
# Live server (real subprocess via the apipod CLI) for fastSDK tests
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_healthy(url: str, proc: subprocess.Popen, timeout: float = 40.0):
    """Poll GET /health until the service answers, or fail if the process dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Service process exited early with code {proc.returncode}.")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)
    raise TimeoutError(f"Service at {url} did not become healthy within {timeout}s.")


@contextlib.contextmanager
def live_service(simulate: str = "serverless", entrypoint: Path = INFERENCE_SERVICE, port: int = 0):
    """
    Boot ``entrypoint`` in a subprocess through the real apipod CLI and yield its URL.

    ``simulate=None`` runs ``apipod start`` (development); otherwise it runs
    ``apipod simulate <target>`` exactly as a user would on the command line.
    """
    port = port or _free_port()
    cmd = [sys.executable, "-m", "apipod.cli"]
    cmd += ["start"] if simulate is None else ["simulate", simulate]
    cmd += [str(entrypoint), "--host", "127.0.0.1", "--port", str(port)]

    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
