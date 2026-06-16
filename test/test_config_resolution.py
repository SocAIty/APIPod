from apipod import APIPod
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.backend.runpod.router import SocaityRunpodRouter


def test_development_default():
    # APIPod() -> plain FastAPI, no job queue.
    app = APIPod()
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is None


def test_simulate_serverless_default():
    # simulate (bare / "serverless") -> FastAPI + Local Job Queue.
    for target in ("", "serverless"):
        app = APIPod(simulate=target)
        assert isinstance(app, SocaityFastAPIRouter)
        assert app.job_queue is not None


def test_simulate_dedicated():
    # dedicated -> Standard FastAPI (no queue).
    app = APIPod(simulate="dedicated")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is None


def test_simulate_dedicated_azure():
    # dedicated-azure -> FastAPI (direct client), no queue.
    app = APIPod(simulate="dedicated-azure")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is None


def test_simulate_serverless_runpod():
    # serverless-runpod (no direct) -> Socaity emulation = FastAPI + job queue.
    app = APIPod(simulate="serverless-runpod")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is not None


def test_simulate_serverless_runpod_direct():
    # serverless-runpod + direct -> RunPod local emulation.
    app = APIPod(simulate="serverless-runpod", direct=True)
    assert isinstance(app, SocaityRunpodRouter)
    assert app.simulate is True


def test_simulate_serverless_azure_warns_and_falls_back():
    # azure has no serverless -> FastAPI + job queue (with warning).
    app = APIPod(simulate="serverless-azure")
    assert isinstance(app, SocaityFastAPIRouter)
    assert app.job_queue is not None


def test_invalid_target_values():
    for target in ("invalid", "serverless-invalid"):
        try:
            APIPod(simulate=target)
            assert False
        except ValueError as e:
            assert "Invalid" in str(e)


if __name__ == "__main__":
    test_development_default()
    test_simulate_serverless_default()
    test_simulate_dedicated()
    test_simulate_dedicated_azure()
    test_simulate_serverless_runpod()
    test_simulate_serverless_runpod_direct()
    test_simulate_serverless_azure_warns_and_falls_back()
    test_invalid_target_values()
    print("All config resolution tests passed.")
