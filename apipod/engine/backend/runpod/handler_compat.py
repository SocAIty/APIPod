"""Adapt APIPod's RunPod entrypoint to RunPod's streaming contract.

RunPod only takes the streaming path when the *registered handler function*
is a generator (``inspect.isgeneratorfunction`` or ``isasyncgenfunction``).
APIPod's router naturally *returns* a generator object for streaming endpoints.
Returning that object from a normal function makes RunPod ``json.dumps`` it
and fail with:

    Object of type generator is not JSON serializable

This module wraps the dispatch callable as a true **async** generator handler
so ``MAX_CONCURRENCY`` > 1 can overlap jobs: awaits yield the event loop;
sync generators are drained on a worker thread.
With ``return_aggregate_stream=True``, a one-shot non-stream result becomes
``[payload]`` — clients unwrap that via ``_try_unwrap_apipod``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterator, Callable, Iterator

from apipod.engine.streaming.stream_serializer import is_streaming_result


async def _iterate_chunks(result: Any) -> AsyncIterator[Any]:
    """Yield chunks from a sync/async generator without blocking the event loop."""
    if inspect.isasyncgen(result):
        async for chunk in result:
            yield chunk
        return
    if inspect.isgenerator(result):
        iterator: Iterator = result

        def _next():
            try:
                return True, next(iterator)
            except StopIteration:
                return False, None

        while True:
            has_item, chunk = await asyncio.to_thread(_next)
            if not has_item:
                return
            yield chunk
        return
    yield result


def as_runpod_async_handler(dispatch: Callable[[Any], Any]) -> Callable[[Any], AsyncIterator]:
    """Wrap *dispatch* so RunPod sees an async generator and can overlap jobs."""

    async def handler(job: Any) -> AsyncIterator:
        result = dispatch(job)
        if inspect.isawaitable(result):
            result = await result
        if is_streaming_result(result) or inspect.isasyncgen(result) or inspect.isgenerator(result):
            async for chunk in _iterate_chunks(result):
                yield chunk
            return
        yield result

    handler.__name__ = getattr(dispatch, "__name__", "handler")
    handler.__doc__ = getattr(dispatch, "__doc__", None)
    assert inspect.isasyncgenfunction(handler)
    return handler


def as_runpod_generator_handler(dispatch: Callable[[Any], Any]) -> Callable[[Any], AsyncIterator]:
    """Backward-compatible alias: RunPod handler is always an async generator."""
    return as_runpod_async_handler(dispatch)
