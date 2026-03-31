from types import UnionType
from typing import Any, Union, get_args, get_origin

import inspect
from fastapi import Body, Form

from apipod.common.schemas import SUPPORTED_LLM_REQUEST_SCHEMAS


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
    def is_supported_llm_request_schema(annotation: Any) -> bool:
        """
        True if annotation is a supported APIPod LLM request schema.
        Supports direct and optional union annotations.
        """
        if annotation in SUPPORTED_LLM_REQUEST_SCHEMAS:
            return True

        origin = get_origin(annotation)
        if origin in (Union, UnionType):
            return any(
                arg in SUPPORTED_LLM_REQUEST_SCHEMAS
                for arg in get_args(annotation)
                if arg is not type(None)
            )

        return False

    @classmethod
    def build_non_file_default(cls, annotation: Any, default: Any, is_optional: bool):
        """
        Select FastAPI parameter defaults for non-file fields.

        Supported LLM request schemas are mapped to JSON body semantics. All other
        non-file parameters keep form semantics for backward compatibility.
        """
        normalized_default = None if is_optional else default
        if cls.is_supported_llm_request_schema(annotation):
            return Body(default=normalized_default)
        return Form(default=normalized_default)
