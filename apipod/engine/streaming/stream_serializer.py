"""
Backend-neutral stream serialization shared by all transport adapters.

This module is the single place that knows how to:
  - detect a streaming runtime result (``is_streaming_result``);
  - bridge an async generator into a sync iterator (``as_sync_iter``);
  - encode raw chunks into JSON-safe strings (``encode_chunk`` for RunPod,
    ``store_chunk`` for the FastAPI stream store);
  - aggregate all raw chunks back into a full result for ``/status`` polling;
  - assemble a :class:`StreamProducer` for the FastAPI queued/direct path.

Both :class:`SocaityFastAPIRouter` and :class:`SocaityRunpodRouter` import
from here instead of each maintaining their own copy.
"""

import asyncio
import base64
import inspect
from typing import Any, Iterator, Optional

from media_toolkit import MediaFile

from apipod.engine.backend.schema_resolve import (
    SchemaBinding,
    SchemaStreamSerializer,
    SSE_DONE,
    SSE_STREAM_TAGS,
    STREAM_CHUNK_SPECS,
    wrap_schema_response,
)
from apipod.engine.jobs.job_result import JobResultFactory
from apipod.engine.streaming.stream_producer import StreamProducer


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_streaming_result(result: Any) -> bool:
    """Return True if *result* is a sync or async generator."""
    return inspect.isgenerator(result) or inspect.isasyncgen(result)


# ---------------------------------------------------------------------------
# Sync/async bridging
# ---------------------------------------------------------------------------

def as_sync_iter(result: Any) -> Iterator:
    """Return a synchronous iterator over a sync or async generator."""
    return _drain_async_gen(result) if inspect.isasyncgen(result) else result


def _drain_async_gen(agen) -> Iterator:
    """Consume an async generator synchronously (for worker / sync contexts)."""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Chunk encoding
# ---------------------------------------------------------------------------

def _to_base64(chunk: Any) -> Optional[str]:
    """Return a base64 string when *chunk* is bytes / bytearray / MediaFile; else None."""
    if isinstance(chunk, MediaFile):
        chunk = chunk.to_bytes()
    if isinstance(chunk, (bytes, bytearray)):
        return base64.b64encode(bytes(chunk)).decode("ascii")
    return None


def encode_chunk(chunk: Any) -> str:
    """JSON-safe encoding without SSE framing (for RunPod transport).

    Binary / MediaFile → base64 string; str → unchanged; other → str().
    """
    b64 = _to_base64(chunk)
    return b64 if b64 is not None else (chunk if isinstance(chunk, str) else str(chunk))


def store_chunk(chunk: Any) -> str:
    """SSE-framed encoding for the FastAPI stream store.

    Binary / MediaFile → ``data: {base64}\\n\\n``; str → unchanged (passthrough);
    other → ``data: {chunk}\\n\\n``.
    """
    b64 = _to_base64(chunk)
    if b64 is not None:
        return f"data: {b64}\n\n"
    return chunk if isinstance(chunk, str) else f"data: {chunk}\n\n"


# ---------------------------------------------------------------------------
# Aggregation (full /status result from streamed items)
# ---------------------------------------------------------------------------

def aggregate_plain(items: list) -> Any:
    """Join text chunks or serialize each item for the /status result."""
    if not items:
        return items
    if all(isinstance(i, str) for i in items):
        return "".join(items)
    if all(isinstance(i, (bytes, bytearray)) for i in items):
        return b"".join(bytes(i) for i in items)
    return [JobResultFactory._serialize_result(i) for i in items]


def aggregate_schema_tokens(items: list, binding: SchemaBinding) -> Any:
    """Reconstruct the schema response from streamed token items."""
    return JobResultFactory._serialize_result(
        wrap_schema_response("".join(str(i) for i in items), binding)
    )


# ---------------------------------------------------------------------------
# StreamProducer construction (FastAPI queued + direct path)
# ---------------------------------------------------------------------------

def build_stream_producer(result: Any, binding: Optional[SchemaBinding]) -> Optional[StreamProducer]:
    """
    Build a :class:`StreamProducer` from a streamable endpoint result, or
    return ``None`` when the result is not a generator.

    Schema endpoints with a registered chunk model (e.g. chat) wrap tokens
    into the standardized ``ChatCompletionChunk`` SSE stream; all other
    generators produce SSE-framed text / base64-encoded binary chunks.
    """
    if not is_streaming_result(result):
        return None

    raw_chunks = as_sync_iter(result)

    if binding is not None and binding.tag in STREAM_CHUNK_SPECS:
        serializer = SchemaStreamSerializer(binding)
        return StreamProducer(
            raw_chunks=raw_chunks,
            to_chunk=serializer.delta,
            closing=[serializer.finish(), SSE_DONE],
            media_type="text/event-stream",
            aggregate=lambda items, b=binding: aggregate_schema_tokens(items, b),
        )

    media_type = (
        "text/event-stream"
        if binding is None or binding.tag in SSE_STREAM_TAGS
        else "application/octet-stream"
    )
    return StreamProducer(
        raw_chunks=raw_chunks,
        to_chunk=store_chunk,
        media_type=media_type,
        aggregate=aggregate_plain,
    )
