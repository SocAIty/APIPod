"""Parse a Starlette Request into endpoint kwargs without mounting a route.

Stage-1 signatures stay plain Python (``ImageFile``, ``ChatCompletionRequest``).
``prepare`` runs APIPod stage-2 (Form/File rewrite + upload wrap). ``materialize``
binds the prepared callable with FastAPI ``get_dependant`` / ``solve_dependencies``
and returns the converted kwargs (media-toolkit objects, parsed models).
"""
from __future__ import annotations

import inspect
from contextlib import AsyncExitStack
from typing import Any, Callable, Optional

from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from apipod.engine.backend.fastapi.file_handling_mixin import _fast_api_file_handling_mixin
from apipod.engine.endpoint_config import EndpointExecutionPlan


class RequestMaterializer(_fast_api_file_handling_mixin):
    """APIPod request materialization for both ``@endpoint`` and the generic gate."""

    def __init__(self, max_upload_file_size_mb: float | None = None) -> None:
        self.max_upload_file_size_mb = max_upload_file_size_mb if max_upload_file_size_mb is not None else 1024.0

    def prepare(
        self,
        func: Callable,
        max_upload_file_size_mb: float | None = None,
        plan: EndpointExecutionPlan | None = None,
    ) -> Callable:
        """Rewrite ``func`` the way a mounted APIPod route would (stage-2)."""
        prepared = self._prepare_func_for_media_file_upload_with_fastapi(
            func, max_upload_file_size_mb, plan=plan
        )
        # FastAPI follows ``__wrapped__`` and would see the pre-rewrite ImageFile
        # annotations. The mounted router already strips this on its wrappers.
        if hasattr(prepared, "__wrapped__"):
            del prepared.__wrapped__
        return prepared

    async def materialize(self, prepared: Callable, request: Request) -> dict[str, Any]:
        """Bind ``request`` to ``prepared`` and return converted keyword arguments.

        Calls the prepared wrapper so UploadFile / FileModel / URL values become
        media-toolkit objects. A collector that returns its kwargs is the usual
        ``prepared`` target.
        """
        values = await solve_request(prepared, request)
        result = prepared(**values)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return result
        return values


async def solve_request(func: Callable, request: Request) -> dict[str, Any]:
    """Run FastAPI dependency solving for ``func`` against ``request``."""
    from fastapi.dependencies.utils import get_dependant, solve_dependencies

    try:
        from fastapi.dependencies.utils import get_flat_dependant
    except ImportError:
        get_flat_dependant = None

    dependant = get_dependant(path=request.scope.get("path") or request.url.path, call=func)
    flat = get_flat_dependant(dependant) if get_flat_dependant is not None else dependant
    embed = _embed_body_fields(flat)
    body = await _read_body(request, flat, embed)
    signature = inspect.signature(solve_dependencies).parameters
    async with AsyncExitStack() as stack:
        kwargs: dict[str, Any] = {"request": request, "dependant": dependant}
        if "body" in signature:
            kwargs["body"] = body
        if "async_exit_stack" in signature:
            kwargs["async_exit_stack"] = stack
        if "embed_body_fields" in signature:
            kwargs["embed_body_fields"] = embed
        solved = await solve_dependencies(**kwargs)
    return _solved_values(solved)


def _embed_body_fields(flat) -> bool:
    body_params = getattr(flat, "body_params", None) or []
    try:
        from fastapi.dependencies.utils import _should_embed_body_fields

        return _should_embed_body_fields(body_params)
    except Exception:
        return len(body_params) != 1


async def _read_body(request: Request, flat, embed: bool):
    """Read JSON or form body the way FastAPI's mounted route handler does."""
    from fastapi import params

    try:
        from fastapi.dependencies.utils import get_body_field
    except ImportError:
        get_body_field = None

    body_field = (
        get_body_field(flat_dependant=flat, name="body", embed_body_fields=embed)
        if get_body_field is not None
        else None
    )
    if body_field is None:
        raw = await request.body()
        if not raw:
            return None
        content_type = request.headers.get("content-type") or ""
        if "form" in content_type:
            return await request.form()
        if (not content_type) or "json" in content_type:
            return await request.json()
        return raw
    if isinstance(getattr(body_field, "field_info", None), params.Form):
        return await request.form()
    raw = await request.body()
    if not raw:
        return None
    content_type = request.headers.get("content-type") or ""
    if (not content_type) or "json" in content_type:
        return await request.json()
    return raw


def _solved_values(solved: Any) -> dict[str, Any]:
    """Normalize FastAPI's solve_dependencies return (tuple vs SolvedDependency)."""
    if hasattr(solved, "errors") and hasattr(solved, "values"):
        if solved.errors:
            raise RequestValidationError(solved.errors)
        return dict(solved.values)
    values, errors, *_rest = solved
    if errors:
        raise RequestValidationError(errors)
    return dict(values)
