"""Adapt APIPod's RunPod entrypoint to RunPod's streaming contract.

RunPod only takes the streaming path when the *registered handler function*
is a generator (``inspect.isgeneratorfunction``). APIPod's router naturally
*returns* a generator object for streaming endpoints. Returning that object
from a normal function makes RunPod ``json.dumps`` it and fail with:

    Object of type generator is not JSON serializable

This module wraps the dispatch callable as a true generator handler:
stream results are ``yield``-ed; non-stream results are yielded once.
With ``return_aggregate_stream=True``, a one-shot non-stream result becomes
``[payload]`` — clients unwrap that via ``_try_unwrap_apipod``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterator

from apipod.engine.streaming.stream_serializer import as_sync_iter, is_streaming_result


def as_runpod_generator_handler(dispatch: Callable[[Any], Any]) -> Callable[[Any], Iterator]:
    """Wrap *dispatch* so RunPod detects a generator handler and streams yields."""

    def handler(job: Any) -> Iterator:
        result = dispatch(job)
        if is_streaming_result(result):
            yield from as_sync_iter(result)
            return
        # Non-stream: one yield so RunPod's generator path still works.
        # Aggregate output is ``[result]``; parsers unwrap JobResult-shaped singles.
        yield result

    handler.__name__ = getattr(dispatch, "__name__", "handler")
    handler.__doc__ = getattr(dispatch, "__doc__", None)
    # Bound methods report as generator functions; keep the same for free functions.
    assert inspect.isgeneratorfunction(handler)
    return handler
