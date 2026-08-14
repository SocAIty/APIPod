"""Regression: endpoint plans must key by unique ``__name__``, not shared ``__qualname__``.

Gate handlers are nested factories: every route shares one ``__qualname__`` while
setting a distinct ``__name__``. Keying plans by qualname collapsed all mounts onto
one slot so the last (often non-streaming) plan won and chat submissions omitted
``links.stream`` even when OpenAPI correctly showed ``stream``.
"""

from collections.abc import Iterator
from typing import Any

from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.queue.job_queue import JobQueue
from apipod.engine.streaming.local_stream_store import LocalStreamStore


def _nested_handlers():
    """Two handlers that share ``__qualname__`` but differ in ``__name__`` / streaming."""

    def make(name: str, *, streaming: bool):
        def dynamic_handler(**kwargs):
            return {"ok": True}

        dynamic_handler.__annotations__ = {
            "return": Iterator[Any] if streaming else Any,
        }
        dynamic_handler.__name__ = name
        return dynamic_handler

    chat = make("handle_svc_chat", streaming=True)
    predict = make("handle_svc_predict", streaming=False)
    assert chat.__qualname__ == predict.__qualname__
    assert chat.__name__ != predict.__name__
    return chat, predict


def test_endpoint_plans_keyed_by_name_not_qualname():
    queue = JobQueue()
    store = LocalStreamStore()
    router = SocaityFastAPIRouter(job_queue=queue, stream_store=store)

    chat, predict = _nested_handlers()
    router.endpoint("/services/v1/svc/chat", methods=["POST"], use_queue=True)(chat)
    router.endpoint("/services/v1/svc/predict", methods=["POST"], use_queue=True)(predict)

    assert len(router._endpoint_plans) == 2
    chat_key = SocaityFastAPIRouter._plan_key(chat)
    predict_key = SocaityFastAPIRouter._plan_key(predict)
    assert chat_key != predict_key
    assert router._endpoint_plans[chat_key].is_streaming is True
    assert router._endpoint_plans[predict_key].is_streaming is False

    # Simulate queue wrapper lookup: sync_wrapper copies ``__name__`` via wraps.
    chat_result = router.add_job(chat, {})
    predict_result = router.add_job(predict, {})

    assert chat_result.links is not None
    assert chat_result.links.stream is not None
    assert "/stream/" in chat_result.links.stream

    assert predict_result.links is not None
    assert predict_result.links.stream is None
