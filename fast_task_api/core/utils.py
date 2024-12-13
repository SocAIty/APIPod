import inspect
from typing import Union, List


from pydantic import BaseModel

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
