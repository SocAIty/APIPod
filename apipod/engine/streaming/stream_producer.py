"""
StreamProducer: the bridge between a streaming endpoint and the stream store.

When a streaming endpoint runs under a job queue, its result cannot be sent to
the client directly (the HTTP response already returned a ``JobResult``).
Instead the worker produces the stream into a :class:`StreamStore` while the
client consumes it from ``GET /stream/{job_id}``.

The router knows *how* to serialize a given result (a ChatCompletion token
stream, encoded media bytes, or a plain generator); the queue worker only knows
the lifecycle. ``StreamProducer`` carries that router-built knowledge to the
worker without leaking schema details into the queue:

  - ``raw_chunks``   – the user generator's raw items (tokens / bytes / objects)
  - ``to_chunk``     – serialize one raw item into a store/SSE chunk (str)
  - ``closing``      – chunks appended after the stream (e.g. finish + ``[DONE]``)
  - ``media_type``   – content type for the direct (non-queued) StreamingResponse
  - ``aggregate``    – build the full ``/status`` result from the raw items, so a
                       client polling status (instead of streaming) gets the
                       complete result once the job finishes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List


@dataclass
class StreamProducer:
    raw_chunks: Iterator[Any]
    to_chunk: Callable[[Any], str]
    aggregate: Callable[[List[Any]], Any]
    closing: List[str] = field(default_factory=list)
    media_type: str = "text/event-stream"
