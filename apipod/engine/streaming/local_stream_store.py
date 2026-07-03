"""
Adapter: LocalStreamStore

In-process implementation of the :class:`StreamStore` port. This is the default
backend APIPod uses on localhost to emulate how streaming behaves once a service
is deployed (e.g. on Socaity, where a Redis-backed store relays chunks from the
worker to the gateway).

Everything lives in process memory, guarded by a lock:
  - the in-process worker thread is the producer (``open_stream`` /
    ``write_chunk`` / ``close_stream``);
  - the FastAPI ``GET /stream/{job_id}`` route is the consumer
    (``read_chunks``).

It is intentionally simple — no external dependencies, no durability. A real
deployment swaps in a distributed store; the endpoint code stays the same.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncIterator, Dict, List, Optional

from apipod.engine.streaming.stream_store import StreamStore

logger = logging.getLogger(__name__)


class _LocalStream:
    """State for a single in-memory stream."""

    def __init__(self, ttl_seconds: int):
        self.chunks: List[str] = []
        self.closed: bool = False
        self.error: Optional[str] = None
        self.expires_at: float = time.monotonic() + ttl_seconds


class LocalStreamStore(StreamStore):
    """Thread-safe, in-memory :class:`StreamStore` for local development/testing."""

    def __init__(self, poll_interval_s: float = 0.02):
        self._streams: Dict[str, _LocalStream] = {}
        self._lock = threading.Lock()
        # How often the async consumer polls the in-memory buffer for new data.
        self._poll_interval_s = poll_interval_s

    # -- Producer API ---------------------------------------------------

    def open_stream(self, job_id: str, ttl_seconds: int = 3600) -> None:
        with self._lock:
            self._streams[job_id] = _LocalStream(ttl_seconds)
        logger.debug("Stream opened | job_id=%s ttl=%ss", job_id, ttl_seconds)

    def write_chunk(self, job_id: str, payload: str) -> None:
        with self._lock:
            stream = self._streams.get(job_id)
            if stream is None:
                # Be forgiving: auto-open if the producer forgot to.
                stream = _LocalStream(ttl_seconds=3600)
                self._streams[job_id] = stream
            stream.chunks.append(payload)

    def close_stream(self, job_id: str, *, error: Optional[str] = None) -> None:
        with self._lock:
            stream = self._streams.get(job_id)
            if stream is None:
                stream = _LocalStream(ttl_seconds=300)
                self._streams[job_id] = stream
            stream.error = error
            stream.closed = True
        logger.debug("Stream closed | job_id=%s error=%s", job_id, error)

    # -- Consumer API ---------------------------------------------------

    async def read_chunks(
        self,
        job_id: str,
        *,
        block_ms: int = 5000,
        batch_size: int = 50,
    ) -> AsyncIterator[str]:
        # The consumer may connect before the producer has opened the stream
        # (the worker picks the job up slightly after the HTTP response). Wait a
        # bounded time for the stream to appear before giving up.
        if not await self._await_stream(job_id, timeout_s=max(block_ms / 1000.0, 5.0)):
            return

        index = 0
        idle_s = 0.0
        keepalive_after_s = block_ms / 1000.0

        while True:
            with self._lock:
                stream = self._streams.get(job_id)
                if stream is None:
                    return
                new_chunks = stream.chunks[index:index + batch_size]
                index += len(new_chunks)
                closed = stream.closed
                error = stream.error
                drained = index >= len(stream.chunks)

            if new_chunks:
                idle_s = 0.0
                for chunk in new_chunks:
                    yield chunk
                continue

            if closed and drained:
                if error:
                    yield f'data: {{"error": {error!r}}}\n\n'
                self.delete_stream(job_id)
                return

            await asyncio.sleep(self._poll_interval_s)
            idle_s += self._poll_interval_s
            if idle_s >= keepalive_after_s:
                idle_s = 0.0
                # SSE comment keeps proxies / browsers from timing out.
                yield ": keepalive\n\n"

    async def _await_stream(self, job_id: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if job_id in self._streams:
                    return True
            await asyncio.sleep(self._poll_interval_s)
        return job_id in self._streams

    # -- Lifecycle ------------------------------------------------------

    def delete_stream(self, job_id: str) -> None:
        with self._lock:
            self._streams.pop(job_id, None)
        logger.debug("Stream deleted | job_id=%s", job_id)

    def stream_exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._streams
