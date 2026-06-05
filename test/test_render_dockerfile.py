"""Regression tests for the Dockerfile renderer.

The CLI bug these tests guard against: a plain `apipod --build` used to
overwrite `apipod.json` values with argparse defaults, so a service configured
with `provider: runpod` would render a Dockerfile carrying `uvicorn` as CMD
(RunPod's worker expects `python entrypoint.py`, which makes the container
fail health checks). The renderer itself must produce the right CMD and ENV
for whatever provider lands in the config dict.
"""

import tempfile
from pathlib import Path

from apipod.deploy.docker_factory import DockerFactory
from apipod.deploy.profile import PROFILE_ML_GPU, PROFILE_SERVERLESS_MINIMAL


def _make_project(tmp: Path) -> None:
    (tmp / "pyproject.toml").write_text(
        '[project]\nname="x"\nrequires-python=">=3.12"\n'
        'dependencies = ["apipod", "runpod"]\n'
    )
    (tmp / "main.py").write_text(
        'from apipod import APIPod\napp = APIPod(provider="runpod")\n'
    )


def test_runpod_provider_renders_python_cmd_and_env():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_project(root)
        factory = DockerFactory(root, root / "apipod-deploy")
        config = {
            "profile": PROFILE_ML_GPU,
            "entrypoint": "main.py",
            "orchestrator": "local",
            "compute": "serverless",
            "provider": "runpod",
        }
        rendered = factory.render_dockerfile(
            "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", config
        )
        assert 'ENV APIPOD_PROVIDER="runpod"' in rendered
        assert 'CMD ["python", "main.py"' in rendered
        assert 'CMD ["uvicorn"' not in rendered


def test_localhost_provider_renders_uvicorn_cmd_and_env():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_project(root)
        factory = DockerFactory(root, root / "apipod-deploy")
        config = {
            "profile": PROFILE_ML_GPU,
            "entrypoint": "main.py",
            "orchestrator": "local",
            "compute": "dedicated",
            "provider": "localhost",
        }
        rendered = factory.render_dockerfile("python:3.12-slim", config)
        assert 'ENV APIPOD_PROVIDER="localhost"' in rendered
        assert 'CMD ["uvicorn"' in rendered
        assert 'CMD ["python", "main.py"' not in rendered


def test_minimal_profile_propagates_runpod_env():
    """The minimal template hardcodes the python CMD so the CMD was right by
    accident even when the provider was wrong. The ENV must still reflect the
    real provider, because the runtime reads it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_project(root)
        factory = DockerFactory(root, root / "apipod-deploy")
        config = {
            "profile": PROFILE_SERVERLESS_MINIMAL,
            "entrypoint": "main.py",
            "orchestrator": "local",
            "compute": "serverless",
            "provider": "runpod",
        }
        rendered = factory.render_dockerfile(
            "ghcr.io/astral-sh/uv:python3.12-bookworm-slim", config
        )
        assert 'ENV APIPOD_PROVIDER="runpod"' in rendered
        assert 'uv", "run", "python", "main.py"' in rendered
