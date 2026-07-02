"""APIPod local JOB_STATUS helpers (in-process queue only)."""

import pytest

from apipod.engine.jobs.base_job import JOB_STATUS, STREAM_WAIT_STATUSES


def test_stream_wait_statuses_match_local_lifecycle() -> None:
    assert STREAM_WAIT_STATUSES == frozenset(
        {
            JOB_STATUS.QUEUED.value,
            JOB_STATUS.PROCESSING.value,
            JOB_STATUS.STREAMING.value,
        }
    )


def test_terminal_statuses() -> None:
    assert JOB_STATUS.FINISHED.is_terminal
    assert JOB_STATUS.FAILED.is_terminal
    assert JOB_STATUS.TIMEOUT.is_terminal
    assert not JOB_STATUS.QUEUED.is_terminal
    assert not JOB_STATUS.PROCESSING.is_terminal
    assert not JOB_STATUS.STREAMING.is_terminal
