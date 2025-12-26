import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apipod.deploy.detectors import (
    DependencyDetector,
    EnvDetector,
    EntrypointDetector,
    FrameworkDetector,
)


@dataclass
class DeploymentConfig:
    entrypoint: str = "main.py"
    title: str = "apipod-service"
    python_version: str = "3.10"
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
        self.entrypoint_detector = EntrypointDetector(self.root_path)
        self.framework_detector = FrameworkDetector(self.root_path)
        self.dependency_detector = DependencyDetector(self.root_path)
        self.env_detector = EnvDetector(self.root_path)

    def scan(self, target_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs all detectors and returns an aggregated configuration dictionary.
        If target_file is provided, it forces the entrypoint to that file.
        """
        print("\n--- Starting Project Scan ---\n")
        
        # Pass the target_file to the entrypoint detector if it supports it
        # or override the detection result manually below.
        entrypoint_info = self.entrypoint_detector.detect(target_file=target_file)
        
        framework_info = self.framework_detector.detect()
        dependency_info = self.dependency_detector.detect()
        env_info = self.env_detector.detect()

        system_packages: List[str] = []
        if dependency_info.get("gcc"):
            system_packages.append("gcc")
        if dependency_info.get("libturbojpg"):
            system_packages.append("libturbojpg")

        deployment_config = DeploymentConfig(
            # Use the target_file if detection didn't already pick it up
            entrypoint=entrypoint_info.get("file", target_file or "main.py"),
            title=entrypoint_info.get("title", "apipod-service"),
            python_version=framework_info.get("python_version", "3.10"),
            pytorch=bool(framework_info.get("pytorch")),
            tensorflow=bool(framework_info.get("tensorflow")),
            onnx=bool(framework_info.get("onnx")),
            transformers=bool(framework_info.get("transformers")),
            diffusers=bool(framework_info.get("diffusers")),
            cuda=bool(framework_info.get("cuda")),
            system_packages=system_packages,
            model_files=framework_info.get("model_files", []),
            has_env_file=env_info.get("has_env_file", False),
        )

        print("\n--- Scan Completed ---\n")
        return deployment_config.to_dict()

    def save_report(self, config: Dict[str, Any]) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print(f"Configuration saved to {self.config_path}")
        except Exception as exc:
            print(f"Error saving configuration: {exc}")

    def load_report(self) -> Optional[Dict[str, Any]]:
        if not self.config_path.exists():
            return None

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Error loading configuration from {self.config_path}: {exc}")
            return None
