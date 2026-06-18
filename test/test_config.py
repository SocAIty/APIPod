"""
A) Configuration: every run intent resolves to the right backend and can serve a
loadable OpenAPI schema.

``APIPod(simulate=..., direct=...)`` is a factory. Part 1 pins the intent ->
backend mapping (FastAPI vs RunPod, job queue present or not). Part 2 boots a
minimal service under each FastAPI intent and asserts ``/openapi.json`` loads and
documents the routes (fastSDK and Swagger UI build everything from it).
"""

import pytest

from apipod import APIPod
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.backend.runpod.router import SocaityRunpodRouter

from conftest import FASTAPI_CONFIGS, build_service
from services import core_service

FASTAPI = SocaityFastAPIRouter
RUNPOD = SocaityRunpodRouter

# (APIPod kwargs, expected backend, expects_job_queue)
RESOLUTIONS = [
    pytest.param({}, FASTAPI, False, id="development"),
    pytest.param({"simulate": ""}, FASTAPI, True, id="simulate-bare"),
    pytest.param({"simulate": "serverless"}, FASTAPI, True, id="serverless"),
    pytest.param({"simulate": "dedicated"}, FASTAPI, False, id="dedicated"),
    pytest.param({"simulate": "dedicated-azure"}, FASTAPI, False, id="dedicated-azure"),
    pytest.param({"simulate": "serverless-runpod"}, FASTAPI, True, id="serverless-runpod"),
    pytest.param({"simulate": "serverless-azure"}, FASTAPI, True, id="serverless-azure-fallback"),
    pytest.param({"simulate": "serverless-runpod", "direct": True}, RUNPOD, False, id="runpod-direct"),
]


# --------------------------------------------------------------------------- #
# 1. Intent -> backend resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs, backend, expects_queue", RESOLUTIONS)
def test_intent_resolves_to_backend(kwargs, backend, expects_queue):
    app = APIPod(**kwargs)
    assert isinstance(app, backend)
    if backend is FASTAPI:
        assert (app.job_queue is not None) is expects_queue
    else:
        assert app.simulate is True  # runpod local emulator


@pytest.mark.parametrize("target", ["invalid", "serverless-invalid"])
def test_invalid_target_raises(target):
    with pytest.raises(ValueError, match="Invalid"):
        APIPod(simulate=target)


# --------------------------------------------------------------------------- #
# 2. OpenAPI schema loads for every backend
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("config", FASTAPI_CONFIGS)
def test_openapi_loads_for_fastapi_backends(config):
    with build_service(core_service.register_minimal, **config) as client:
        schema = client.get("/openapi.json").json()
        assert schema["openapi"].startswith("3.")
        assert {"/echo", "/add"} <= set(schema["paths"])
        assert "apipod" in schema["info"]  # APIPod stamps its version in


def test_openapi_for_runpod_direct():
    """The RunPod worker has no HTTP layer; it synthesizes the schema itself."""
    app = APIPod(simulate="serverless-runpod", direct=True)
    core_service.register_minimal(app)

    schema = app.get_openapi_schema()
    assert schema["openapi"].startswith("3.")
    assert {"/echo", "/add"} <= set(schema["paths"])
    assert "apipod" in schema["info"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
