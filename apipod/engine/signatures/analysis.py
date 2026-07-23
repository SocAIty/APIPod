"""
Backend-neutral analysis of function signatures and return types.
"""

import ast
import inspect
import textwrap
from collections.abc import AsyncIterator, Iterator
from types import UnionType
from typing import Callable, Union, get_args, get_origin, get_type_hints


def _return_type_includes_iterator(return_type) -> bool:
    """True when *return_type* is or contains ``Iterator`` / ``AsyncIterator``."""
    if return_type is None:
        return False
    origin = get_origin(return_type)
    if origin in (Union, UnionType):
        return any(_return_type_includes_iterator(arg) for arg in get_args(return_type))
    resolved = origin or return_type
    return inspect.isclass(resolved) and issubclass(resolved, (Iterator, AsyncIterator))


def is_streaming_endpoint(func: Callable, schema_binding=None) -> bool:
    """Backend-neutral: True when *func* can produce a streaming response.

    Detection order:
    - generator / async-generator function;
    - return annotation contains ``Iterator`` / ``AsyncIterator`` (incl. inside a ``Union``);
    - schema endpoint whose request model has a ``stream`` field and the function body
      returns a generator (AST: ``yield``, generator expression, or under ``if request.stream``).

    ``schema_binding`` is resolved from *func* when not provided by the caller.
    """
    target = inspect.unwrap(func)
    if inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target):
        return True
    try:
        return_type = get_type_hints(target).get("return")
    except Exception:
        return_type = None
    if _return_type_includes_iterator(return_type):
        return True

    if schema_binding is None:
        # Local import: schema_resolve depends on backend modules that in turn
        # use this analysis module.
        from apipod.engine.backend.schema_resolve import get_schema_binding

        try:
            schema_binding = get_schema_binding(func)
        except Exception:
            schema_binding = None

    if schema_binding is not None and "stream" in schema_binding.request_model.model_fields:
        return ast_suggests_request_stream(target)
    return False


def _is_stream_attr_test(node: ast.AST) -> bool:
    """True when *node* tests a ``.stream`` attribute (e.g. ``request.stream``)."""
    if isinstance(node, ast.Attribute) and node.attr == "stream":
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_stream_attr_test(node.operand)
    return False


def _is_streaming_return_value(node: ast.AST | None) -> bool:
    """True when a ``return`` hands back a generator (expression, yield, or async yield)."""
    if node is None:
        return False
    return isinstance(node, (ast.GeneratorExp, ast.Yield, ast.YieldFrom))


def _statements_suggest_streaming(stmts: list[ast.stmt]) -> bool:
    """Walk *stmts* for generator returns, yields, or a ``.stream`` branch that streams."""
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(node, ast.Return) and _is_streaming_return_value(node.value):
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, ast.If) and _is_stream_attr_test(node.test):
            if _statements_suggest_streaming(node.body) or _statements_suggest_streaming(node.orelse):
                return True
    return False


def _function_ast_node(func: Callable) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Parse the source of *func* into its top-level function AST node."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, ValueError):
        return None

    expected_name = func.__name__
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == expected_name:
            return node
    return None


def ast_suggests_request_stream(func: Callable) -> bool:
    """True when the function body conditionally or unconditionally returns a generator.

    Detects patterns such as ``if request.stream: return (t for t in tokens)`` without
    requiring a return-type annotation. Returns ``False`` when source is unavailable
    (REPL, dynamic ``exec``) or the body has no streaming return path.
    """
    fn_node = _function_ast_node(inspect.unwrap(func))
    if fn_node is None:
        return False
    return _statements_suggest_streaming(fn_node.body)


def is_injected_progress_param(param: inspect.Parameter) -> bool:
    """True if the parameter is a framework-injected :class:`JobProgress`."""
    return param.name == "job_progress" or "JobProgress" in str(param.annotation)


def job_progress_param_names(func: Callable) -> list[str]:
    """Names of the parameters of *func* that should receive a :class:`JobProgress`.

    A parameter qualifies when it is literally named ``job_progress`` or when its
    annotation refers to a ``JobProgress`` type. This single detection is shared
    by every injection site (queue worker, RunPod handler, direct FastAPI path)
    so they can never disagree about what counts as a progress parameter.
    """
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return []
    return [p.name for p in params if is_injected_progress_param(p)]
