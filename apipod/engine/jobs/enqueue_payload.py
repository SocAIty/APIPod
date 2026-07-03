"""Marker for queue handlers that return submission metadata, not inference output.

Some :class:`~apipod.engine.queue.job_queue_interface.JobQueueInterface`
implementations call the wrapped handler when a job is *submitted* (not when a
worker runs it) to collect metadata for the queue. Those return values must not
be coerced through schema response wrapping.
"""


class EnqueuePayload:
    """Base class for handler results that represent job submission metadata."""

    __apipod_enqueue_payload__ = True


def is_enqueue_payload(result: object) -> bool:
    """Return True when *result* is job submission metadata, not model output."""
    return isinstance(result, EnqueuePayload)
