"""
A) Core endpoint plumbing on the FastAPI backend.

Exercises the parameter shapes an author writes and the file/queue machinery:
- standard scalar types and a custom pydantic model,
- the broad ``/mixed-media`` endpoint (model + media + scalars) builds & documents,
- single file upload via multipart and via base64 (round-trips to base64 out),
- the job queue + JobProgress lifecycle (submit, poll, read result).

Development mode runs endpoints inline; serverless mode queues them.
"""

import base64
import time

import pytest

from conftest import IMAGE_FILE, build_service
from services import core_service


@pytest.fixture(scope="module")
def dev_client():
    with build_service(core_service.register) as c:
        yield c


@pytest.fixture(scope="module")
def serverless_client():
    with build_service(core_service.register, simulate="serverless") as c:
        yield c


def test_standard_types(dev_client):
    resp = dev_client.post("/scalars", data={"text": "hi", "count": 3, "ratio": 2.5, "flag": True})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "hi", "count": 3, "ratio": 2.5, "flag": True}


def test_custom_pydantic_model(dev_client):
    resp = dev_client.post("/model", data={"pam1": "burger", "pam2": 2})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"pam1": "burger", "pam2": 2}


def test_mixed_media_is_documented(dev_client):
    """The broad mixed endpoint (model + media + scalars) builds and is documented."""
    schema = dev_client.get("/openapi.json").json()
    assert "/mixed-media" in schema["paths"]


def test_single_file_upload_multipart(dev_client):
    with open(IMAGE_FILE, "rb") as fh:
        resp = dev_client.post("/single-file-upload", files={"file1": (IMAGE_FILE.name, fh, "image/png")})
    assert resp.status_code == 200, resp.text
    assert base64.b64decode(resp.json())  # valid base64 back out


def test_single_file_upload_base64(dev_client):
    payload = base64.b64encode(IMAGE_FILE.read_bytes()).decode()
    resp = dev_client.post("/single-file-upload", data={"file1": payload})
    assert resp.status_code == 200, resp.text
    assert base64.b64decode(resp.json())  # round-tripped through ImageFile


def test_job_progress_lifecycle(serverless_client):
    submit = serverless_client.post("/test-job-progress", data={"fries_name": "curly", "amount": 2})
    assert submit.status_code == 200, submit.text
    status_url = submit.json()["links"]["status"]

    deadline = time.time() + 15
    while time.time() < deadline:
        body = serverless_client.get(status_url).json()
        if body.get("result") is not None:
            assert body["result"] == "Your fries curly are ready"
            return
        if body.get("status") in ("failed", "timeout", "not_found"):
            pytest.fail(f"job ended unexpectedly: {body}")
        time.sleep(0.1)
    pytest.fail("job did not finish in time")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
