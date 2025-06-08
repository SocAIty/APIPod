import inspect
from typing import Union, List
import re


def replace_func_signature(func: callable, new_sig: Union[inspect.Signature, List[inspect.Parameter]]):
    if isinstance(new_sig, list):
        new_sig = sorted(new_sig, key=lambda p: (p.kind, p.default is not inspect.Parameter.empty))
        new_sig = inspect.Signature(parameters=new_sig)
 
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

    # make sure it does not start with a number
    if result and result[0].isdigit():
        result = 's-' + result

    return result if result else None
