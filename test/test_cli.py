"""
B) CLI commands: scan, build, simulate.

Each command is driven through its real entry point (``apipod.cli``) against a
throwaway project, asserting it produces the expected artifacts:
- ``scan``  -> apipod-deploy/apipod.json
- ``build`` -> apipod-deploy/Dockerfile (Docker invocation itself is stubbed)
- ``simulate`` -> applies the run intent (env) and boots the resolved entrypoint
  via the app's ``start`` (uvicorn.run is stubbed so the test does not block).
"""

import argparse

import pytest

import apipod.cli as cli
from apipod.deploy.deployment_manager import DeploymentManager

SERVICE_FILE = """\
from apipod import APIPod

app = APIPod(title="cli-test-service")


@app.endpoint("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    app.start()
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A minimal apipod project; cwd is moved into it so the CLI scans it."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "svc"\nversion = "0.0.0"\n')
    (tmp_path / "main.py").write_text(SERVICE_FILE)
    monkeypatch.chdir(tmp_path)
    # Auto-confirm every interactive prompt.
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    return tmp_path


def test_project_dir_aliases_are_silent():
    parser, _ = cli._build_parser()
    help_text = parser.format_help()
    assert "--project-dir" in help_text
    assert "--cwd" not in help_text
    assert " -C " not in help_text
    assert parser.parse_args(["--project-dir", "svc", "scan"]).project_dir == "svc"
    assert parser.parse_args(["-C", "svc", "scan"]).project_dir == "svc"
    assert parser.parse_args(["--cwd", "svc", "scan"]).project_dir == "svc"


def test_scan_generates_config(project):
    cli.run_scan()

    config = project / "apipod-deploy" / "apipod.json"
    assert config.exists()

    import json
    data = json.loads(config.read_text())
    assert data["entrypoint"] == "main.py"
    assert data["title"] == "cli-test-service"


SERVE_FILE = """\
from apipod import Model, serve


class DummyLLM(Model):
    def load(self):
        pass

    def generate(self, messages, temperature=0.7, max_tokens=16):
        return "ok"


model = DummyLLM()

if __name__ == "__main__":
    serve(model, title="served-cli-service")
"""


def test_scan_detects_serve_entrypoint(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "svc"\nversion = "0.0.0"\n')
    (tmp_path / "qwen.py").write_text(SERVE_FILE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")

    cli.run_scan()

    import json
    data = json.loads((tmp_path / "apipod-deploy" / "apipod.json").read_text())
    assert data["entrypoint"] == "qwen.py"
    assert data["title"] == "served-cli-service"


def test_scan_selects_among_multiple_entrypoints(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "svc"\nversion = "0.0.0"\n')
    (tmp_path / "alpha.py").write_text(SERVE_FILE.replace("served-cli-service", "alpha-svc"))
    (tmp_path / "beta.py").write_text(SERVE_FILE.replace("served-cli-service", "beta-svc"))
    monkeypatch.chdir(tmp_path)
    answers = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    cli.run_scan()

    import json
    data = json.loads((tmp_path / "apipod-deploy" / "apipod.json").read_text())
    assert data["entrypoint"] == "beta.py"
    assert data["title"] == "beta-svc"


def test_build_generates_dockerfile(project, monkeypatch):
    # Never actually invoke Docker.
    monkeypatch.setattr(DeploymentManager, "build_docker_image", lambda self, title: True)

    cli.run_build(argparse.Namespace(file=None))

    dockerfile = project / "apipod-deploy" / "Dockerfile"
    assert dockerfile.exists()
    assert "FROM" in dockerfile.read_text()


def test_simulate_applies_intent_and_starts(project, monkeypatch):
    started = {}

    def fake_uvicorn_run(app, host=None, port=None, **kwargs):
        started["host"] = host
        started["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)

    args = argparse.Namespace(
        target="serverless", entrypoint="main.py", native=False, host="127.0.0.1", port=8123
    )
    cli.run_simulate(args)

    import os
    assert os.environ["APIPOD_SIMULATE"] == "serverless"  # intent applied as env
    assert started == {"host": "127.0.0.1", "port": 8123}  # entrypoint booted


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
