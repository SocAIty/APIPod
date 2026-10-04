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


def test_chat_kwargs_omit_unset_max_tokens():
    from socaity_schemas.public.inference.language import ChatCompletionRequest
    from apipod.serve import _chat_kwargs

    def generate(self, messages, temperature=0.7, max_tokens=None):
        return None

    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
    kwargs = _chat_kwargs(request, generate)
    assert "max_tokens" not in kwargs

    limited = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=8)
    assert _chat_kwargs(limited, generate)["max_tokens"] == 8


def test_chat_kwargs_forward_reasoning_effort():
    from socaity_schemas.public.inference.language import ChatCompletionRequest
    from apipod.serve import _chat_kwargs

    def thinking(self, messages, temperature=0.7, max_tokens=None, reasoning_effort=None):
        return None

    def minimal(self, messages, temperature=0.7, max_tokens=None):
        return None

    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], reasoning_effort="low")
    assert _chat_kwargs(request, thinking)["reasoning_effort"] == "low"
    assert "reasoning_effort" not in _chat_kwargs(request, minimal)


def test_vllm_defaults_are_explicit_overrides():
    from apipod.models.chat import _unregistered
    from apipod.models.transformers.base import Transformers
    from apipod.models.vllm import config as vllm_config
    from apipod.models.vllm.chat import VLLMChat

    assert vllm_config.MAX_MODEL_LEN == ""
    assert vllm_config.MAX_NUM_SEQS == ""
    assert vllm_config.ENABLE_AUTO_TOOL_CHOICE is False
    assert "max_new_tokens" not in Transformers._generation_kwargs(0.7, None)

    engine = _unregistered(VLLMChat, "org/model")
    body = engine._request_body([{"role": "user", "content": "hi"}])
    assert "max_tokens" not in body
    assert "chat_template_kwargs" not in body
    assert engine._request_body([{"role": "user", "content": "hi"}], max_tokens=16)["max_tokens"] == 16
    thinking = _unregistered(VLLMChat, "org/model", enable_thinking=True)
    assert thinking._request_body([{"role": "user", "content": "hi"}])["chat_template_kwargs"] == {
        "enable_thinking": True,
    }
    off = _unregistered(VLLMChat, "org/model", enable_thinking=False)
    assert off._request_body(
        [{"role": "user", "content": "hi"}], reasoning_effort="low"
    )["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "low"}
    assert off._request_body(
        [{"role": "user", "content": "hi"}], reasoning_effort="high"
    )["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert off._request_body(
        [{"role": "user", "content": "hi"}], reasoning_effort="none"
    )["chat_template_kwargs"] == {"enable_thinking": False}


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
