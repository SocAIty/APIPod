import inspect
from typing import Union, List
import re


def replace_func_signature(func: callable, new_sig: Union[inspect.Signature, List[inspect.Parameter]]):
    if isinstance(new_sig, list):
        new_sig = sorted(new_sig, key=lambda p: (p.kind, p.default is not inspect.Parameter.empty))
        new_sig = inspect.Signature(parameters=new_sig)

    setattr(func, '__signature__', new_sig)
    return func


# copy of implementation in fastsdk/fastsdk/utils.py
def normalize_identifier(
    original: str,
    replacement_char: str,
    allowed_non_alphanum: str,
    trim_chars: str
) -> str:
    """
    Generalized identifier normalization utility.

    Args:
        original (str): The original string to normalize.
        replacement_char (str): Character to replace disallowed characters with (e.g., "-" or "_").
        allowed_non_alphanum (str): A string of non-alphanumeric characters that should be allowed.
        trim_chars (str): Characters to strip from the beginning and end of the result.

    Returns:
        str: A normalized, lowercase identifier with enforced formatting rules.
    """
    # Convert to lowercase
    normalized = original.lower()

    # Replace backslashes with slashes
    normalized = normalized.replace("\\", "/")

    # Replace all characters that are not a-z, 0-9, or explicitly allowed
    allowed = f"a-z0-9{re.escape(allowed_non_alphanum)}"
    normalized = re.sub(f"[^{allowed}]+", replacement_char, normalized)

    # Collapse multiple instances of allowed non-alphanum characters (like '//' or '__')
    for ch in set(allowed_non_alphanum + replacement_char):
        normalized = re.sub(f"{re.escape(ch)}+", ch, normalized)

    # Trim unwanted leading/trailing characters
    normalized = normalized.strip(trim_chars + replacement_char)

    # Ensure the identifier does not start with a digit
    if normalized and normalized[0].isdigit():
        normalized = replacement_char + normalized

    return normalized


def normalize_name(name: str, preserve_paths: bool = False) -> str:
    """
    Normalizes a string to be used as a Python module, method or class name by:
    - Replacing all special characters with underscores
    - Lowercasing the result
    - Preventing double underscores and invalid leading/trailing characters
    - Avoiding names that start with a digit

    Args:
        name (str): The input string to normalize.

    Returns:
        str: A Python module-safe, normalized string.
    """
    name = normalize_identifier(
        original=name,
        replacement_char='-',
        allowed_non_alphanum='/' if preserve_paths else '',
        trim_chars='-'
    )
    if len(name) == 0:
        return "no_name"
    return name
