import inspect
from typing import Union, List
import re


def _is_socaity_ai_route_inference_callable(func: callable):
    try:
        return 'routeinferencecallable' in inspect.getmodule(func).__name__.lower().replace("_",'')
    except:
        return False


def get_func_signature(func: callable):
    """
    Returns the signature of a function or callable object.
    Only use if you know what you are doing.
    Excludes fastapi classes because they interfer with fast-task-api.
    """
    # a package of socaity.ai uses a callable object called RouteInferenceCallable.
    # For this reason it is treated differently here. Please don't change this without consulting the socaity.ai team.
    if not _is_socaity_ai_route_inference_callable(func):
        return inspect.signature(func)

    return inspect.signature(func.__call__)


def replace_func_signature(func: callable, new_sig: Union[inspect.Signature, List[inspect.Parameter]]):
    if isinstance(new_sig, list):
        new_sig = sorted(new_sig, key=lambda p: (p.kind, p.default is not inspect.Parameter.empty))
        new_sig = inspect.Signature(parameters=new_sig)

    # a package of socaity.ai uses a callable object called RouteInferenceCallable.
    # For this reason it is treated differently here. Please don't change this without consulting the socaity.ai team.
    if _is_socaity_ai_route_inference_callable(func):
        setattr(func.__call__, '__signature__', new_sig)
    else:
        setattr(func, '__signature__', new_sig)

    return func


def normalize_name(name: str, preserve_paths: bool = False) -> Union[str, None]:
    """
    Normalize a name to be openapi compatible and better searchable.
    Will remove any special characters. Transforms lowercase. Replaces spaces with hyphens.
    :param name: The service name to normalize
    :param preserve_paths: If True, preserves forward slashes (/) for path segments
    :return: Normalized service name
    """
    if name is None or not isinstance(name, str):
        return None

    def normalize_segment(text: str) -> str:
        """Helper function to normalize a single segment of text"""
        text = text.lower()
        text = ' '.join(text.split())  # Replace multiple spaces with single space
        text = text.replace(' ', '-').replace("_", '-')   # Replace spaces and _ with hyphens
        text = re.sub(r'[^a-z0-9-]', '', text)  # Keep only alphanumeric and hyphens
        text = re.sub(r'-+', '-', text)  # Replace multiple hyphens with single hyphen
        return text.strip('-')  # Remove leading/trailing hyphens

    if preserve_paths:
        # Normalize each non-empty path segment
        result = '/'.join(
            segment for segment in
            (normalize_segment(s) for s in name.split('/'))
            if segment
        )
    else:
        result = normalize_segment(name)

    return result if result else None