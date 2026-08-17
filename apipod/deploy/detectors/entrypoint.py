import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .IDetector import Detector

_PRIORITY_FILES = ("main.py", "app.py", "api.py", "serve.py")
_FACTORY_NAMES = frozenset({"APIPod", "serve"})


def _empty_result() -> Dict[str, Any]:
    return {
        "file": None,
        "title": "apipod-service",
        "found_config": False,
        "kind": None,
        "orchestrator": "local",
        "compute": "dedicated",
        "provider": "localhost",
    }


def _factory_aliases(tree: ast.AST) -> Dict[str, str]:
    """Map local names to the apipod factory they refer to (``APIPod`` / ``serve`` / ``apipod``)."""
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module == "apipod" or node.module.startswith("apipod.")
        ):
            for alias in node.names:
                if alias.name in _FACTORY_NAMES:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "apipod":
                    aliases[alias.asname or "apipod"] = "apipod"
    return aliases


def _factory_kind(node: ast.Call, aliases: Dict[str, str]) -> Optional[str]:
    """Return ``APIPod`` or ``serve`` when this call constructs the service."""
    func = node.func
    if isinstance(func, ast.Name):
        kind = aliases.get(func.id)
        return kind if kind in _FACTORY_NAMES else None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if aliases.get(func.value.id) == "apipod" and func.attr in _FACTORY_NAMES:
            return func.attr
    return None


def _ast_constant(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _apply_keywords(node: ast.Call, result: Dict[str, Any]) -> None:
    for keyword in node.keywords:
        value = _ast_constant(keyword.value)
        if value is None:
            continue
        if keyword.arg in {"title", "orchestrator", "compute", "provider"}:
            result[keyword.arg] = value


def _eval_simple(node: ast.AST, module: Any) -> Any:
    """Resolve a name or ``obj.attr`` against an already-imported module."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Name):
        return getattr(module, node.id, None)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        obj = getattr(module, node.value.id, None)
        return getattr(obj, node.attr, None) if obj is not None else None
    return None


def resolve_entrypoint_title(file_path: str, module: Any) -> Optional[str]:
    """Read ``title=`` from ``APIPod()`` / ``serve()`` after the entrypoint was imported.

    Static scan only sees string literals. ``serve(model, title=spec.family)``
    is resolved here against the imported module.
    """
    try:
        tree = ast.parse(Path(file_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    aliases = _factory_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _factory_kind(node, aliases) is None:
            continue
        for keyword in node.keywords:
            if keyword.arg != "title":
                continue
            value = _eval_simple(keyword.value, module)
            if isinstance(value, str) and value:
                return value
    return None


class EntrypointDetector(Detector):
    def detect(self, target_file: Optional[str] = None) -> Dict[str, Any]:
        print("Scanning for entrypoint and service configuration...")

        if target_file:
            explicit = self._from_explicit_path(target_file)
            if explicit is not None:
                return explicit

        candidates = self._collect_candidates()
        if not candidates:
            fallback = self._priority_without_config()
            if fallback is not None:
                print(f"Found entrypoint file: {fallback['file']} (no config detected)")
                return fallback
            print("No entrypoint detected.")
            return _empty_result()

        if len(candidates) == 1:
            chosen = candidates[0]
            print(f"Found entrypoint: {chosen['file']} ({chosen['kind']})")
            return chosen

        return self._select_candidate(candidates)

    def _from_explicit_path(self, target_file: str) -> Optional[Dict[str, Any]]:
        root = Path(self.project_root).resolve()
        provided = Path(target_file)
        full_path = provided if provided.is_absolute() else Path.cwd() / provided
        full_path = full_path.resolve()

        if not full_path.exists():
            print(f"Warning: Provided target file {target_file} not found at {full_path}")
            return None
        try:
            rel = full_path.relative_to(root)
        except ValueError:
            print(f"Warning: {target_file} exists but is outside the project root.")
            return None

        candidate = self._analyze_file(full_path, root)
        if candidate is not None:
            print(f"Using explicitly provided entrypoint: {candidate['file']}")
            return candidate

        result = _empty_result()
        result["file"] = rel.as_posix()
        print(f"Using explicitly provided entrypoint: {result['file']}")
        return result

    def _collect_candidates(self) -> List[Dict[str, Any]]:
        root = Path(self.project_root).resolve()
        found: List[Dict[str, Any]] = []
        seen = set()

        for dirpath, _, files in os.walk(root):
            if self.should_ignore(dirpath):
                continue
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = Path(dirpath) / file
                candidate = self._analyze_file(path, root)
                if candidate is None or candidate["file"] in seen:
                    continue
                found.append(candidate)
                seen.add(candidate["file"])

        found.sort(key=lambda item: (item["file"] not in _PRIORITY_FILES, item["file"]))
        return found

    def _priority_without_config(self) -> Optional[Dict[str, Any]]:
        root = Path(self.project_root).resolve()
        for name in _PRIORITY_FILES:
            if (root / name).is_file():
                result = _empty_result()
                result["file"] = name
                return result
        return None

    def _analyze_file(self, path: Path, root: Path) -> Optional[Dict[str, Any]]:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        if not any(token in content for token in ("APIPod(", "serve(", "app.start()", "uvicorn.run")):
            return None

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        result = _empty_result()
        result["file"] = path.resolve().relative_to(root).as_posix()
        aliases = _factory_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kind = _factory_kind(node, aliases)
            if kind is None:
                continue
            result["found_config"] = True
            result["kind"] = kind
            _apply_keywords(node, result)

        if result["found_config"]:
            return result

        if "app.start()" in content or "uvicorn.run" in content:
            result["kind"] = "start"
            return result
        return None

    @staticmethod
    def _select_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(f"Found {len(candidates)} entrypoints. Select one:")
        for index, candidate in enumerate(candidates, 1):
            title = candidate.get("title")
            extra = f", title={title}" if title and title != "apipod-service" else ""
            print(f"  {index}. {candidate['file']} ({candidate['kind']}{extra})")

        while True:
            raw = input("Selection: ").strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(candidates):
                    chosen = candidates[idx]
                    print(f"Using entrypoint: {chosen['file']}")
                    return chosen
            except ValueError:
                pass
            print("Invalid selection. Please try again.")
