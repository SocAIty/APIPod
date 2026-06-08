from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader

from apipod.deploy.profile import (
    PROFILE_ML_GPU,
    PROFILE_SERVERLESS_MINIMAL,
    recommend_base_image,
)


class DockerFactory:
    """
    Encapsulates all Docker-related operations.
    """

    DEFAULT_IMAGES = [
        "python:3.12-slim",
        "python:3.11-slim",
        "python:3.10-slim",
        "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
    ]

    def __init__(
        self,
        project_root: Path,
        deploy_dir: Path,
        template_dir: Optional[Path] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.deploy_dir = Path(deploy_dir)
        self.template_dir = template_dir or Path(__file__).parent

        self.template_env = Environment(
            loader=FileSystemLoader(str(self.template_dir))
        )
        self.docker_template = self.template_env.get_template("docker_template.j2")
        self.minimal_template = self.template_env.get_template("docker_template_minimal.j2")
        self.images = self._load_images(self.template_dir / "docker_images.txt")

    def _load_images(self, images_path: Path) -> List[str]:
        if images_path.exists():
            try:
                with images_path.open("r", encoding="utf-8") as f:
                    images = [line.strip() for line in f if line.strip()]
                    if images:
                        # Keep python:3.12-slim at the top as the recommended default.
                        preferred = ["python:3.12-slim"]
                        merged = preferred + [img for img in images if img not in preferred]
                        return merged
            except Exception:
                pass
        return self.DEFAULT_IMAGES.copy()

    def recommend_image(self, config: Dict[str, Any]) -> str:
        profile = config.get("profile", PROFILE_ML_GPU)
        suggested = recommend_base_image(profile, config.get("python_version", "3.12"), config)
        if suggested in self.images:
            return suggested
        if profile == PROFILE_SERVERLESS_MINIMAL:
            version = str(config.get("python_version") or "3.12")
            for img in self.images:
                if f"python:{version}-slim" in img:
                    return img
        return suggested

    def render_dockerfile(self, base_image: str, config: Dict[str, Any]) -> str:
        profile = config.get("profile", PROFILE_ML_GPU)
        if profile == PROFILE_SERVERLESS_MINIMAL:
            return self._render_minimal(base_image, config)

        has_requirements = (self.project_root / "requirements.txt").exists()
        entrypoint = config.get("entrypoint", "main.py")
        entrypoint_module = (
            Path(entrypoint).with_suffix("").as_posix().replace("/", ".").replace("\\", ".")
        )

        is_nvidia_base = "nvidia/cuda" in base_image or "runpod/" in base_image
        needs_ml_system_libs = bool(
            config.get("pytorch")
            or config.get("tensorflow")
            or config.get("onnx")
        )
        should_install_cudnn = needs_ml_system_libs and not is_nvidia_base

        context = {
            "base_image": base_image,
            "has_requirements": has_requirements,
            "entrypoint_module": entrypoint_module,
            "entrypoint_script": Path(entrypoint).name,
            "install_cudnn": should_install_cudnn,
            "system_packages": config.get("system_packages", []),
            "orchestrator": config.get("orchestrator", "local"),
            "compute": config.get("compute", "dedicated"),
            "provider": config.get("provider", "localhost"),
        }
        return self.docker_template.render(**context)

    def _render_minimal(self, base_image: str, config: Dict[str, Any]) -> str:
        root = self.project_root
        has_uv_lock = (root / "uv.lock").is_file()
        has_pyproject = (root / "pyproject.toml").is_file()
        has_requirements = (root / "requirements.txt").is_file()
        install_project = has_pyproject and self._pyproject_defines_package(root / "pyproject.toml")

        entrypoint = config.get("entrypoint", "main.py")
        context = {
            "base_image": base_image,
            "has_uv_lock": has_uv_lock,
            "has_pyproject": has_pyproject,
            "has_requirements": has_requirements,
            "install_project": install_project,
            "entrypoint_script": Path(entrypoint).name,
            "orchestrator": config.get("orchestrator", "local"),
            "compute": config.get("compute", "serverless"),
            "provider": config.get("provider", "runpod"),
        }
        return self.minimal_template.render(**context)

    @staticmethod
    def _pyproject_defines_package(pyproject_path: Path) -> bool:
        try:
            import toml

            data = toml.load(pyproject_path)
            project = data.get("project", {})
            if project.get("name"):
                return True
        except Exception:
            pass
        return False

    def write_dockerfile(self, content: str, dockerfile_path: Path) -> Path:
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path = Path(dockerfile_path)
        dockerfile_path.write_text(content, encoding="utf-8")
        print(f"Dockerfile created at {dockerfile_path}")
        return dockerfile_path

    def write_project_dockerignore(self) -> Path:
        path = self.project_root / ".dockerignore"
        lines = [
            "apipod-deploy/",
            ".git/",
            ".venv/",
            "venv/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".mypy_cache/",
            "dist/",
            "build/",
            "*.egg-info/",
            "",
        ]
        if not path.exists():
            path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Created {path}")
        return path

    def build_image(self, tag: str, dockerfile_path: Path, context_dir: Path) -> bool:
        self.write_project_dockerignore()
        cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(Path(context_dir))]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("Build completed successfully.")
            return True
        except FileNotFoundError:
            print("Error: 'docker' command not found.")
        except subprocess.CalledProcessError:
            print("Docker build failed.")
        return False
