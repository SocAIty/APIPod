"""
FastAPI-specific streaming transport, factored out of the router.

This mixin owns the FastAPI-side mechanics for turning a streaming endpoint
result into an HTTP response (execution itself goes through the shared
``_BaseBackend.run_callable`` / ``run_callable_async`` runtime):

- ``_stream_generator`` — adapt a sync/async generator into an async generator
  for :class:`StreamingResponse` (direct non-queued path);
- ``_streaming_response_from_producer`` — build a :class:`StreamingResponse` from
  a :class:`StreamProducer` (direct non-queued path);
- ``_create_streaming_endpoint_decorator`` — register a plain generator endpoint
  directly on the FastAPI router (direct non-queued path);
- ``stream_job_sse`` — SSE consumer route ``GET /stream/{job_id}`` that replays
  a queued job's chunks from the stream store.

Serialization details (encoding, aggregation, producer construction) live in
:mod:`apipod.engine.streaming.stream_serializer`; endpoint introspection and
plan building live in :mod:`apipod.engine.endpoint_config`.

The mixin carries no state and needs no ``__init__``.
"""

import asyncio
import functools
import inspect
from typing import Callable

from fastapi import Request, status
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse

from apipod.engine.endpoint_config import EndpointExecutionPlan
from apipod.engine.streaming.stream_producer import StreamProducer


# Sentinel marking the end of a synchronous iterator drained via run_in_executor.
_STREAM_END = object()


class _FastAPIStreamingMixin:
    """FastAPI streaming transport helpers mixed into the FastAPI router."""

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def _stream_generator(self, result):
        """Adapt a sync or async generator into an async generator for StreamingResponse."""
        if inspect.isasyncgen(result):
            async for chunk in result:
                yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
        elif inspect.isgenerator(result):
            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, next, result, _STREAM_END)
                if chunk is _STREAM_END:
                    break
                yield chunk if isinstance(chunk, (str, bytes)) else str(chunk)
        else:
            raise TypeError(f"Expected generator, got {type(result)}")

    # ------------------------------------------------------------------
    # Direct streaming (non-queued)
    # ------------------------------------------------------------------

    def _streaming_response_from_producer(self, producer: StreamProducer) -> StreamingResponse:
        """Build a direct :class:`StreamingResponse` from a producer (non-queued path)."""
        async def _event_generator():
            loop = asyncio.get_event_loop()
            iterator = iter(producer.raw_chunks)
            while True:
                item = await loop.run_in_executor(None, next, iterator, _STREAM_END)
                if item is _STREAM_END:
                    break
                yield producer.to_chunk(item)
            for closing_chunk in producer.closing:
                yield closing_chunk

        return StreamingResponse(
            _event_generator(),
            media_type=producer.media_type,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def _create_streaming_endpoint_decorator(self, plan: EndpointExecutionPlan):
        """Register a plain generator endpoint served directly to the client (no queue)."""
        kwargs = dict(plan.route_kwargs)
        custom_headers = kwargs.pop("response_headers", None)

        fastapi_route_decorator = self.api_route(
            path=plan.path,
            methods=plan.active_methods,
            *plan.route_args,
            **kwargs,
        )

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def streaming_wrapper(*w_args, **w_kwargs):
                result = await self.run_callable_async(func, *w_args, **w_kwargs)
                generator = self._stream_generator(result)

                headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                if custom_headers:
                    headers.update(custom_headers)

                return StreamingResponse(generator, media_type="text/event-stream", headers=headers)

            with_upload = self._prepare_func_for_media_file_upload_with_fastapi(
                streaming_wrapper, plan.max_upload_file_size_mb
            )
            return fastapi_route_decorator(with_upload)

        return decorator

    # ------------------------------------------------------------------
    # SSE consumer route (queued jobs)
    # ------------------------------------------------------------------

    async def stream_job_sse(self, job_id: str, request: Request):
        """Server-Sent Events consumer for a queued streaming job (requires stream_store)."""
        job_id = job_id.strip().strip('"').strip("'").strip("?").strip("#")
        if self.stream_store is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Streaming not configured.")
        if self.job_queue is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Job queue not configured.")

        jq = self.job_queue
        job_data = jq.get_job_status(job_id) if hasattr(jq, "get_job_status") else None
        stream_open = self.stream_store.stream_exists(job_id)

        # Serve the stream even when the job record is already gone (completed +
        # cleaned up) as long as the stream is still draining.
        if job_data is None and not stream_open:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

        # Allow any not-yet-finished state; read_chunks waits for the producer.
        st = ((job_data.get("status") if job_data else "") or "").lower()
        if not stream_open and st not in {"queued", "processing", "streaming"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job '{job_id}' is not streaming (status: {st or 'unknown'}).",
            )

        async def _event_generator():
            try:
                async for chunk in self.stream_store.read_chunks(job_id):
                    if await request.is_disconnected():
                        break
                    yield chunk
            except Exception:
                self._logger.exception("Error during stream delivery | job_id=%s", job_id)
                yield 'data: {"error": "Internal stream error"}\n\n'

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
