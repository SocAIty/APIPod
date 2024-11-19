import inspect
from typing import Union, List


def is_custom_callable_object(func: callable):
    return hasattr(func, '__call__') and not 'fastapi' in type(func).__module__

def get_func_signature(func: callable):
    """
    Returns the signature of a function or callable object.
    Only use if you know what you are doing.
    Excludes fastapi classes because they interfer with fast-task-api.
    """
    is_callable_object = is_custom_callable_object(func)
    underlying_func = func.__call__ if is_callable_object else func
    return inspect.signature(underlying_func)

    #is_callable_object = hasattr(func, '__call__') and not 'fastapi' in type(func).__module__
    ## is_inference_callable = hasattr(func, '__qualname__') and 'RouteInferenceCallable' in func.__qualname__
    #if is_callable_object and isinstance(func.__call__, functools.partial) and hasattr(func.__call__, 'keywords'):
    #    sig_params = [
    #        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ptype) for name, ptype
    #        in func.__call__.keywords.items()
    #    ]
    #    return inspect.Signature(parameters=sig_params)
#
    ## underlying_func = func.__call__ if is_callable_object else func
    #return inspect.signature(func)


def replace_func_signature(func: callable, new_sig: Union[inspect.Signature, List[inspect.Parameter]]):
    if isinstance(new_sig, list):
        new_sig = sorted(new_sig, key=lambda p: (p.kind, p.default is not inspect.Parameter.empty))
        new_sig = inspect.Signature(parameters=new_sig)

    is_callable_object = is_custom_callable_object(func)
    if is_callable_object:
        func.__call__.__signature__ = new_sig
    else:
        func.__signature__ = new_sig

    return func
