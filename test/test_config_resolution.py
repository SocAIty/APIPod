from apipod import APIPod
from apipod.common import constants
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.backend.runpod.router import SocaityRunpodRouter
import os


def test_socaity_dedicated_auto():
    # socaity | dedicated | auto -> FastAPI
    app = APIPod(orchestrator="socaity", compute="dedicated", provider="auto")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is None


def test_socaity_dedicated_localhost():
    # socaity | dedicated | localhost -> FastAPI + job queue (test mode)
    app = APIPod(orchestrator="socaity", compute="dedicated", provider="localhost")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is not None


def test_socaity_dedicated_planned_celery():
    # socaity | dedicated | runpod/scaleway/azure -> Celery (planned)
    for provider in ["runpod", "scaleway", "azure"]:
        try:
            APIPod(orchestrator="socaity", compute="dedicated", provider=provider)
            assert False
        except NotImplementedError as e:
            assert "planned" in str(e)


def test_socaity_serverless_auto():
    # socaity | serverless | auto -> RunPod router
    app = APIPod(orchestrator="socaity", compute="serverless", provider="auto")
    assert isinstance(app, SocaityRunpodRouter)


def test_socaity_serverless_localhost():
    # socaity | serverless | localhost -> FastAPI + job queue (test mode)
    app = APIPod(orchestrator="socaity", compute="serverless", provider="localhost")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is not None


def test_socaity_serverless_runpod():
    # socaity | serverless | runpod -> RunPod router
    app = APIPod(orchestrator="socaity", compute="serverless", provider="runpod")
    assert isinstance(app, SocaityRunpodRouter)


def test_socaity_serverless_unsupported():
    # socaity | serverless | scaleway/azure -> Not supported
    for provider in ["scaleway", "azure"]:
        try:
            APIPod(orchestrator="socaity", compute="serverless", provider=provider)
            assert False
        except NotImplementedError as e:
            assert "not supported" in str(e) or "not implemented" in str(e)


def test_local_dedicated():
    # local | dedicated | * -> FastAPI
    app = APIPod(orchestrator="local", compute="dedicated", provider="localhost")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is None


def test_local_serverless_localhost():
    # local | serverless | localhost -> FastAPI + job queue
    app = APIPod(orchestrator="local", compute="serverless", provider="localhost")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is not None


def test_local_serverless_runpod():
    # local | serverless | runpod -> RunPod router
    app = APIPod(orchestrator="local", compute="serverless", provider="runpod")
    assert isinstance(app, SocaityRunpodRouter)


def test_local_serverless_unsupported():
    # local | serverless | scaleway/azure -> Not supported
    for provider in ["scaleway", "azure"]:
        try:
            APIPod(orchestrator="local", compute="serverless", provider=provider)
            assert False
        except NotImplementedError as e:
            assert "not supported" in str(e) or "not implemented" in str(e)


def test_invalid_enum_values():
    try:
        APIPod(orchestrator="invalid")
        assert False
    except ValueError as e:
        assert "Invalid ORCHESTRATOR" in str(e)

    try:
        APIPod(compute="invalid")
        assert False
    except ValueError as e:
        assert "Invalid COMPUTE" in str(e)


if __name__ == "__main__":
    test_socaity_dedicated_auto()
    test_socaity_dedicated_localhost()
    test_socaity_dedicated_planned_celery()
    test_socaity_serverless_auto()
    test_socaity_serverless_localhost()
    test_socaity_serverless_runpod()
    test_socaity_serverless_unsupported()
    test_local_dedicated()
    test_local_serverless_localhost()
    test_local_serverless_runpod()
    test_local_serverless_unsupported()
    test_invalid_enum_values()
