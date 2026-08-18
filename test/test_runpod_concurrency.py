"""RunPod in-worker concurrency and VLLMChat HTTP proxy (no GPU, no vLLM binary)."""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import pytest

from apipod.engine.backend.runpod.handler_compat import as_runpod_async_handler
from apipod.engine.backend.runpod.router import SocaityRunpodRouter
from apipod.models.vllm.chat import VLLMChat


def test_start_config_reads_max_concurrency(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENCY", "8")
    cfg = SocaityRunpodRouter._runpod_start_config(lambda job: None)
    assert cfg["concurrency_modifier"](1) == 8
    assert cfg["return_aggregate_stream"] is True


def test_async_handler_is_asyncgen():
    async def dispatch(job):
        return {"id": job["id"]}

    handler = as_runpod_async_handler(dispatch)
    assert inspect.isasyncgenfunction(handler)


def test_async_handler_overlaps_awaitable_dispatch():
    started = []

    async def dispatch(job):
        started.append(time.monotonic())
        await asyncio.sleep(0.3)
        return {"ok": job["id"]}

    handler = as_runpod_async_handler(dispatch)

    async def consume(job_id: str):
        chunks = []
        async for chunk in handler({"id": job_id, "input": {}}):
            chunks.append(chunk)
        return chunks

    async def run():
        t0 = time.monotonic()
        first, second = await asyncio.gather(consume("a"), consume("b"))
        return time.monotonic() - t0, first, second

    elapsed, first, second = asyncio.run(run())
    assert first == [{"ok": "a"}]
    assert second == [{"ok": "b"}]
    assert elapsed < 0.5
    assert abs(started[0] - started[1]) < 0.2


class _FakeVLLM(BaseHTTPRequestHandler):
    inflight = 0
    max_inflight = 0
    delay_s = 0.3

    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        type(self).inflight += 1
        type(self).max_inflight = max(type(self).max_inflight, type(self).inflight)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            time.sleep(self.delay_s)
            payload = {
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            type(self).inflight -= 1


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture
def fake_vllm():
    _FakeVLLM.inflight = 0
    _FakeVLLM.max_inflight = 0
    server = _ThreadedHTTPServer(("127.0.0.1", 0), _FakeVLLM)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    thread.join(timeout=2)


def test_agenerate_overlaps_on_fake_vllm(fake_vllm):
    model = VLLMChat("acme/tiny")
    model._apipod_loaded = True
    model._base_url = fake_vllm

    async def run():
        t0 = time.monotonic()
        first, second = await asyncio.gather(
            model.agenerate([{"role": "user", "content": "capital?"}]),
            model.agenerate([{"role": "user", "content": "capital?"}]),
        )
        return time.monotonic() - t0, first, second

    elapsed, first, second = asyncio.run(run())
    assert first == "Paris"
    assert second == "Paris"
    assert elapsed < 0.5
    assert _FakeVLLM.max_inflight >= 2


def test_load_spawns_vllm_and_waits_for_health(monkeypatch, fake_vllm):
    host_port = fake_vllm.rsplit(":", 1)
    host = host_port[0].replace("http://", "")
    port = host_port[1]
    monkeypatch.setenv("APIPOD_VLLM_HOST", host)
    monkeypatch.setenv("APIPOD_VLLM_PORT", port)
    monkeypatch.setenv("APIPOD_VLLM_STARTUP_TIMEOUT", "5")

    proc = type("Proc", (), {"poll": lambda self: None, "terminate": lambda self: None, "wait": lambda self, timeout=None: 0, "kill": lambda self: None, "returncode": None})()
    captured = {}

    def fake_popen(argv, *args, **kwargs):
        captured["argv"] = argv
        return proc

    monkeypatch.setattr("apipod.models.vllm.chat.shutil.which", lambda name: "/usr/bin/vllm")
    monkeypatch.setattr("apipod.models.vllm.chat.subprocess.Popen", fake_popen)
    monkeypatch.setattr("apipod.models.vllm.chat.atexit.register", lambda fn: None)

    model = VLLMChat("Qwen/Qwen3.8-27B-FP8")
    model.load()
    assert captured["argv"][0] == "/usr/bin/vllm"
    assert "serve" in captured["argv"]
    assert "Qwen/Qwen3.8-27B-FP8" in captured["argv"]
    assert captured["argv"][captured["argv"].index("--max-num-seqs") + 1] == "256"
    assert model._base_url == fake_vllm
