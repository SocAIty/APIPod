import ast
import json
import os
import re
import toml
from typing import Any, Dict, List, Optional, Set

from apipod.deploy.profile import (
    DIFFUSERS_PACKAGES,
    ONNX_PACKAGES,
    PYTORCH_PACKAGES,
    TENSORFLOW_PACKAGES,
    TRANSFORMERS_PACKAGES,
)
from .IDetector import Detector


class FrameworkDetector(Detector):
    def detect(self, entrypoint: Optional[str] = None) -> Dict[str, Any]:
        print("Scanning for frameworks and models...")
        self._detected_python_version = None
        config: Dict[str, Any] = {
            "pytorch": False,
            "tensorflow": False,
            "onnx": False,
            "transformers": False,
            "diffusers": False,
            "cuda": False,
            "python_version": "3.10",
            "model_files": [],
            "python_dependencies": [],
            "entrypoint_imports": [],
        }

        dep_names = self._gather_dependency_names()
        config["python_dependencies"] = sorted(dep_names)
        self._apply_dependency_packages(dep_names, config)

        entrypoint_imports = self._check_entrypoint_imports(entrypoint)
        config["entrypoint_imports"] = sorted(entrypoint_imports)
        self._apply_entrypoint_imports(entrypoint_imports, config)

        self._scan_model_files(config)
        return config

    @staticmethod
    def _has_any_framework(config: Dict[str, Any]) -> bool:
        return any(
            config[key]
            for key in ("pytorch", "tensorflow", "onnx", "transformers", "diffusers")
        )

    def _gather_dependency_names(self) -> Set[str]:
        names: Set[str] = set()
        pyproject_path = os.path.join(self.project_root, "pyproject.toml")
        if os.path.exists(pyproject_path):
            try:
                data = toml.load(pyproject_path)
                project = data.get("project", {})
                for dep in project.get("dependencies", []):
                    names.add(self._extract_package_name(dep))
                ver = project.get("requires-python", "")
                match = re.search(r"3\.(\d+)", ver)
                if match:
                    self._detected_python_version = f"3.{match.group(1)}"
                from apipod.deploy.profile import POETRY_NON_PACKAGE_KEYS

                poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
                for dep in poetry_deps.keys():
                    key = dep.lower()
                    if key not in POETRY_NON_PACKAGE_KEYS:
                        names.add(key)
            except Exception as exc:
                print(f"Warning: Error parsing pyproject.toml: {exc}")

        requirements_path = os.path.join(self.project_root, "requirements.txt")
        if os.path.exists(requirements_path):
            try:
                with open(requirements_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            names.add(self._extract_package_name(line))
            except Exception as exc:
                print(f"Warning: Error parsing requirements.txt: {exc}")

        return names

    _detected_python_version = None

    def _apply_dependency_packages(self, dep_names: Set[str], config: Dict[str, Any]) -> None:
        if self._detected_python_version:
            config["python_version"] = self._detected_python_version

        for name in dep_names:
            if name in PYTORCH_PACKAGES:
                config["pytorch"] = True
                if "cuda" in name or name.endswith("-gpu"):
                    config["cuda"] = True
            if name in TENSORFLOW_PACKAGES:
                config["tensorflow"] = True
            if name in ONNX_PACKAGES:
                config["onnx"] = True
            if name in TRANSFORMERS_PACKAGES:
                config["transformers"] = True
            if name in DIFFUSERS_PACKAGES:
                config["diffusers"] = True

        for name in dep_names:
            if name == "torch":
                config["pytorch"] = True
            lowered = name
            if "cu" in lowered and "torch" in lowered:
                config["cuda"] = True

    @staticmethod
    def _extract_package_name(dependency: str) -> str:
        dependency = dependency.split("#", 1)[0].strip()
        name = re.split(r"[\s=<>!~;\[]", dependency, maxsplit=1)[0].strip()
        return name.lower().replace("_", "-")

    def _check_entrypoint_imports(self, entrypoint: Optional[str]) -> Set[str]:
        """Inspect only the service entrypoint for ML imports (not the whole repo)."""
        if not entrypoint:
            return set()

        file_path = os.path.join(self.project_root, entrypoint)
        if not os.path.isfile(file_path):
            return set()

        top_level: Set[str] = set()
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=entrypoint)
        except Exception:
            return top_level

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".", 1)[0])
        return top_level

    def _apply_entrypoint_imports(self, entrypoint_imports: Set[str], config: Dict[str, Any]) -> None:
        mapping = {
            "pytorch": {"torch", "torchvision", "torchaudio"},
            "tensorflow": {"tensorflow", "keras"},
            "onnx": {"onnx", "onnxruntime"},
            "transformers": {"transformers"},
            "diffusers": {"diffusers"},
        }
        for key, modules in mapping.items():
            if entrypoint_imports & modules:
                config[key] = True

    def _scan_model_files(self, config: Dict[str, Any]) -> None:
        extensions = {".pt", ".pth", ".onnx", ".h5", ".safetensors", ".bin", ".gguf"}
        found_files: List[str] = []

        for root, _, files in os.walk(self.project_root):
            if self.should_ignore(root):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                ext = ext.lower()
                if ext in extensions:
                    found_files.append(os.path.relpath(file_path, self.project_root))
                elif ext == ".json" and self._is_model_json(file_path):
                    found_files.append(os.path.relpath(file_path, self.project_root))

        config["model_files"] = found_files

    def _is_model_json(self, file_path: str) -> bool:
        """
        Heuristic for spotting HuggingFace-style model config JSONs in the repo.

        Three layers, cheapest first:
        1. Filename whitelist: HF always names these the same way.
        2. Filename blocklist: common project JSONs that look related but aren't.
        3. Content sniff: open the file and look for keys that only show up in
           model configs (architectures, hidden_size, etc.). Cap at 1 MB so we
           don't accidentally parse a giant tokenizer vocab.
        """
        filename = os.path.basename(file_path).lower()
        if filename in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "adapter_config.json",
        }:
            return True
        if filename in {
            "package.json",
            "tsconfig.json",
            "apipod.json",
            "pyproject.json",
            "launch.json",
            "settings.json",
        }:
            return False
        try:
            if os.path.getsize(file_path) > 1024 * 1024:
                return False
            with open(file_path, "r", encoding="utf-8") as handle:
                content = json.load(handle)
            if isinstance(content, dict):
                keys = content.keys()
                # Keys typical of HF transformer / model configs.
                model_keys = {
                    "architectures",
                    "model_type",
                    "vocab_size",
                    "hidden_size",
                    "layer_norm_epsilon",
                }
                if any(key in keys for key in model_keys):
                    return True
        except Exception:
            pass
        return False
