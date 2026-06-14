"""
Port: StreamStore

Abstract interface for reading and writing streaming data chunks associated
with a job. Implementations may keep the chunks in process memory (the local
test backend), in Redis Streams, in Kafka, or in any other append-only log.

In a real deployment the producer and the consumer live in different
processes:

  - The **worker** (producer) calls :meth:`open_stream` / :meth:`write_chunk`
    as result chunks (ChatCompletion deltas, encoded ``AudioFile`` /
    ``VideoFile`` bytes, or chunks of any generator endpoint) are produced.
  - The **gateway** (consumer) calls :meth:`read_chunks` to relay those
    chunks to the client over Server-Sent Events.
  - Both sides use :meth:`close_stream` / :meth:`delete_stream` for lifecycle
    management.

APIPod on localhost emulates that split inside a single process: the in-process
worker thread produces, while the FastAPI ``GET /stream/{job_id}`` route
consumes. The default backend is :class:`LocalStreamStore`; deployments swap in
their own implementation (e.g. a Redis-backed store) without touching the
endpoint code.

Design decisions:
  - :meth:`read_chunks` is an **async generator** so the gateway can yield
    chunks straight into a ``StreamingResponse`` without buffering.
  - :meth:`write_chunk` / :meth:`open_stream` / :meth:`close_stream` are
    **synchronous** because the worker writes from a (sync) worker context.
  - :meth:`close_stream` signals completion (optionally with an error) so the
    consumer's read loop exits cleanly.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional


class StreamStore(ABC):
    """Port for append-only stream storage keyed by ``job_id``."""

    # -- Producer API (worker side) -------------------------------------

    @abstractmethod
    def open_stream(self, job_id: str, ttl_seconds: int = 3600) -> None:
        """
        Prepare a stream for writing.

        Implementations should create the underlying resource and set a
        safety-net TTL to prevent leaks if neither side cleans up.
        """
        ...

    @abstractmethod
    def write_chunk(self, job_id: str, payload: str) -> None:
        """
        Append a single chunk (typically one SSE ``data:`` line) to the stream.

        Must be safe to call from a synchronous (worker) context.
        """
        ...

    @abstractmethod
    def close_stream(self, job_id: str, *, error: Optional[str] = None) -> None:
        """
        Signal that the producer is done.

        Writes a terminal marker so the consumer knows the stream is complete.
        If *error* is provided, the marker carries the error payload.
        """
        ...

    # -- Consumer API (gateway side) ------------------------------------

    @abstractmethod
    async def read_chunks(
        self,
        job_id: str,
        *,
        block_ms: int = 5000,
        batch_size: int = 50,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields chunks until the terminal marker is read.

        Parameters:
            block_ms:   how long to wait for new data before emitting a
                        keep-alive (used for timeout / disconnect checks).
            batch_size: max chunks returned per internal read.
        """
        ...
        # Make this an async generator at the type level.
        if False:  # pragma: no cover
            yield ""

    # -- Lifecycle ------------------------------------------------------

    @abstractmethod
    def delete_stream(self, job_id: str) -> None:
        """
        Remove the stream resource entirely.

        Called once the consumer has read the final chunk, or by a periodic
        cleanup for abandoned streams.
        """
        ...

    @abstractmethod
    def stream_exists(self, job_id: str) -> bool:
        """Return ``True`` if the stream key exists."""
        ...
