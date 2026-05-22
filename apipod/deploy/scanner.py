import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apipod.deploy.detectors import (
    DependencyDetector,
    EnvDetector,
    EntrypointDetector,
    FrameworkDetector,
)
from apipod.deploy.profile import infer_profile, reconcile_framework_flags


@dataclass
class DeploymentConfig:
    entrypoint: str = "main.py"
    title: str = "apipod-service"
    profile: str = "web-api"
    python_version: str = "3.10"
    orchestrator: str = "local"
    compute: str = "dedicated"
    provider: str = "localhost"
    pytorch: bool = False
    tensorflow: bool = False
    onnx: bool = False
    transformers: bool = False
    diffusers: bool = False
    cuda: bool = False
    system_packages: List[str] = field(default_factory=list)
    model_files: List[str] = field(default_factory=list)
    has_env_file: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Scanner:
    """
    Scans the package to assemble deployment configuration based on detectors.
    """

    def __init__(self, root_path: Path, config_path: Path):
        self.root_path = Path(root_path).resolve()
        self.config_path = Path(config_path)
        self.entrypoint_detector = EntrypointDetector(str(self.root_path))
        self.framework_detector = FrameworkDetector(str(self.root_path))
        self.dependency_detector = DependencyDetector(str(self.root_path))
        self.env_detector = EnvDetector(str(self.root_path))

    def scan(self, target_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs all detectors and returns an aggregated configuration dictionary.
        If target_file is provided, it forces the entrypoint to that file.
        """
        print("\n--- Starting Project Scan ---\n")

        entrypoint_info = self.entrypoint_detector.detect(target_file=target_file)
        entrypoint = entrypoint_info.get("file", target_file or "main.py")
        framework_info = self.framework_detector.detect(entrypoint=entrypoint)
        dependency_info = self.dependency_detector.detect()
        env_info = self.env_detector.detect()

        python_deps: Set[str] = set(framework_info.get("python_dependencies", []))
        entrypoint_imports: Set[str] = set(framework_info.get("entrypoint_imports", []))
        model_files: List[str] = framework_info.get("model_files", [])

        system_packages: List[str] = []
        if dependency_info.get("gcc"):
            system_packages.append("gcc")
        if dependency_info.get("libturbojpg"):
            system_packages.append("libturbojpg")

        raw_flags = {
            "pytorch": bool(framework_info.get("pytorch")),
            "tensorflow": bool(framework_info.get("tensorflow")),
            "onnx": bool(framework_info.get("onnx")),
            "transformers": bool(framework_info.get("transformers")),
            "diffusers": bool(framework_info.get("diffusers")),
            "cuda": bool(framework_info.get("cuda")),
        }
        flags = reconcile_framework_flags(
            python_deps=python_deps,
            entrypoint_imports=entrypoint_imports,
            model_files=model_files,
        )
        pytorch = flags["pytorch"]
        tensorflow = flags["tensorflow"]
        onnx = flags["onnx"]
        transformers = flags["transformers"]
        diffusers = flags["diffusers"]
        cuda = flags["cuda"]

        compute = entrypoint_info.get("compute")
        provider = entrypoint_info.get("provider")
        profile = infer_profile(
            pytorch=pytorch,
            tensorflow=tensorflow,
            onnx=onnx,
            transformers=transformers,
            diffusers=diffusers,
            cuda=cuda,
            compute=compute,
            provider=provider,
            python_deps=python_deps,
            model_files=model_files,
        )

        if flags != raw_flags:
            print(
                "Adjusted framework flags after verification "
                f"(entrypoint imports: {', '.join(sorted(entrypoint_imports)) or 'none'})"
            )

        deployment_config = DeploymentConfig(
            entrypoint=entrypoint_info.get("file", target_file or "main.py"),
            title=entrypoint_info.get("title", "apipod-service"),
            profile=profile,
            python_version=framework_info.get("python_version", "3.10"),
            orchestrator=entrypoint_info.get("orchestrator", "local"),
            compute=compute or "dedicated",
            provider=provider or "localhost",
            pytorch=pytorch,
            tensorflow=tensorflow,
            onnx=onnx,
            transformers=transformers,
            diffusers=diffusers,
            cuda=cuda,
            system_packages=system_packages,
            model_files=framework_info.get("model_files", []),
            has_env_file=env_info.get("has_env_file", False),
        )

        print(f"Deployment profile: {profile}")
        if python_deps:
            print(f"Python dependencies: {', '.join(sorted(python_deps))}")
        print("\n--- Scan Completed ---\n")
        return deployment_config.to_dict()

    def save_report(self, config: Dict[str, Any]) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
            self._write_starter_files(config)
        except Exception as exc:
            print(f"Error saving configuration: {exc}")

    def _write_starter_files(self, config: Dict[str, Any]) -> None:
        deploy_dir = self.config_path.parent
        readme_dst = deploy_dir / "README.md"
        starter = Path(__file__).parent / "starter_README.md"
        if starter.is_file() and not readme_dst.exists():
            shutil.copy(starter, readme_dst)
            print(f"Starter guide written to {readme_dst}")

        dockerignore = deploy_dir / ".dockerignore"
        if not dockerignore.exists():
            dockerignore.write_text(
                "\n".join(
                    [
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
                ),
                encoding="utf-8",
            )

    def load_report(self) -> Optional[Dict[str, Any]]:
        if not self.config_path.exists():
            return None

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Error loading configuration from {self.config_path}: {exc}")
            return None
