import os
import ast
from pathlib import Path
from typing import Dict, Any, Optional
from .IDetector import Detector


class EntrypointDetector(Detector):
    def detect(self, target_file: Optional[str] | None = None) -> Dict[str, Any]:
        print("Scanning for entrypoint and service configuration...")

        result = {
            "file": None,
            "title": "apipod-service",  # Default
            "found_config": False
        }

        # 1. Prioritize user-provided target file
        if target_file:
            # Convert everything to absolute paths to compare fairly
            absolute_project_root = Path(self.project_root).resolve()
            
            # This handles cases where user provides ./test.py or an absolute path
            provided_path = Path(target_file)
            if not provided_path.is_absolute():
                # If relative, assume it's relative to where the user is CURRENTLY
                full_path = Path.cwd() / provided_path
            else:
                full_path = provided_path
                
            full_path = full_path.resolve()

            if full_path.exists():
                # We must store the path RELATIVE to the project root for Docker
                try:
                    rel_path = full_path.relative_to(absolute_project_root)
                    result["file"] = str(rel_path)
                    self._scan_file_for_title(str(full_path), result)
                    print(f"Using explicitly provided entrypoint: {rel_path}")
                    return result
                except ValueError:
                    print(f"Warning: {target_file} exists but is outside the project root.")
            else:
                print(f"Warning: Provided target file {target_file} not found at {full_path}")

        # 2. Check priority files (standard discovery)
        priority_files = ["main.py", "app.py", "api.py", "serve.py"]
        for filename in priority_files:
            path = os.path.join(self.project_root, filename)
            if os.path.exists(path):
                result["file"] = filename
                self._scan_file_for_title(path, result)
                if result["found_config"]:
                    print(f"Found entrypoint and config in: {filename}")
                    return result
                print(f"Found entrypoint file: {filename} (no config detected)")
                return result

        # 3. Deep scan (fallback)
        print("No standard entrypoint file found. Scanning file contents...")
        for root, _, files in os.walk(self.project_root):
            if self.should_ignore(root):
                continue
            for file in files:
                if file.endswith(".py") and file not in priority_files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    if self._scan_file_for_indicators(file_path, result):
                        result["file"] = rel_path
                        print(f"Found entrypoint in code pattern: {rel_path}")
                        return result

        if result["file"] is None:
            print("No entrypoint detected.")

        return result

    def _scan_file_for_title(self, file_path: str, result: Dict[str, Any]):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "APIPod" in content:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id == "APIPod":
                            for keyword in node.keywords:
                                if keyword.arg == "title":
                                    if isinstance(keyword.value, ast.Constant):  # Python 3.8+
                                        result["title"] = keyword.value.value
                                        result["found_config"] = True
                                    elif isinstance(keyword.value, ast.Str):  # Python < 3.8
                                        result["title"] = keyword.value.s
                                        result["found_config"] = True
        except Exception:
            pass

    def _scan_file_for_indicators(self, file_path: str, result: Dict[str, Any]) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for APIPod config
            if "APIPod" in content:
                self._scan_file_for_title(file_path, result)
                if result["found_config"]:
                    return True

            # Check for other indicators
            if "app.start()" in content or "uvicorn.run" in content:
                return True

            return False
        except Exception:
            return False
