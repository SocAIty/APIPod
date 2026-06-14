import inspect
from typing import Any

from fastapi import Body, Form

from apipod.engine.backend.schema_resolve import SCHEMA_REGISTRY, resolve_request_model

# Request schemas that are interpreted as JSON bodies by router decorators,
# even when endpoint authors do not specify Body(...). Derived from the
# schema registry so the two can never drift apart.
SUPPORTED_REQUEST_SCHEMAS = tuple(SCHEMA_REGISTRY)


class FastAPISignaturePolicies:
    """Policy helpers for FastAPI signature default conversion."""

    @staticmethod
    def is_fastapi_dependency(parameter: inspect.Parameter) -> bool:
        """
        True if the parameter default is a FastAPI/Starlette dependency object.
        """
        default = parameter.default
        if default is inspect.Parameter.empty:
            return False

        if isinstance(default, (int, float, str, bool, list, dict, tuple, set, type(None))):
            return False

        module = getattr(type(default), "__module__", "")
        return module.startswith("fastapi") or module.startswith("starlette")

    @staticmethod
    def is_supported_request_schema(annotation: Any) -> bool:
        """
        True if annotation is (or contains, for Optional/Union annotations) a
        registered APIPod request schema or a subclass of one.
        """
        return resolve_request_model(annotation) is not None

    @classmethod
    def build_non_file_default(cls, annotation: Any, default: Any, is_optional: bool):
        """
        Select FastAPI parameter defaults for non-file fields.

        Registered request schemas are mapped to JSON body semantics. All other
        non-file parameters keep form semantics for backward compatibility.
        """
        normalized_default = None if is_optional else default
        if cls.is_supported_request_schema(annotation):
            return Body(default=normalized_default)
        return Form(default=normalized_default)
