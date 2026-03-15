from apipod import CONSTS
from apipod.settings import APIPOD_ORCHESTRATOR, APIPOD_COMPUTE, APIPOD_PROVIDER
from apipod.core.routers._socaity_router import _SocaityRouter
from apipod.core.routers._runpod_router import SocaityRunpodRouter
from apipod.core.routers._fastapi_router import SocaityFastAPIRouter
from apipod.core.job_queues.job_queue_interface import JobQueueInterface

from typing import Union


def APIPod(
        orchestrator: Union[CONSTS.ORCHESTRATOR, str, None] = None,
        compute: Union[CONSTS.COMPUTE, str, None] = None,
        provider: Union[CONSTS.PROVIDER, str, None] = None,
        *args, **kwargs
) -> Union[_SocaityRouter, SocaityRunpodRouter, SocaityFastAPIRouter]:
    """
    Initialize an APIPod router with the appropriate backend based on the deployment configuration.

    The resulting backend is determined by the combination of orchestrator, compute, and provider:

    | Orchestrator | Compute    | Provider  | Backend                    |
    |------------- |----------- |---------- |--------------------------- |
    | socaity      | dedicated  | auto      | FastAPI                    |
    | socaity      | dedicated  | localhost | FastAPI + job queue (test) |
    | socaity      | dedicated  | runpod    | Celery (planned)           |
    | socaity      | dedicated  | scaleway  | Celery (planned)           |
    | socaity      | dedicated  | azure     | Celery (planned)           |
    | socaity      | serverless | auto      | RunPod router              |
    | socaity      | serverless | localhost | FastAPI + job queue (test) |
    | socaity      | serverless | runpod    | RunPod router              |
    | socaity      | serverless | scaleway  | Not supported              |
    | socaity      | serverless | azure     | Not supported              |
    | local/None   | dedicated  | *         | FastAPI                    |
    | local/None   | serverless | localhost | FastAPI + job queue        |
    | local/None   | serverless | runpod    | RunPod router              |
    | local/None   | serverless | scaleway  | Not supported              |
    | local/None   | serverless | azure     | Not supported              |

    Args:
        orchestrator: "socaity" or "local" (default from env / local).
        compute: "dedicated" or "serverless" (default from env / dedicated).
        provider: "auto", "localhost", "runpod", "scaleway", "azure" (default from env / localhost).
    """
    orchestrator = _resolve_enum(orchestrator, CONSTS.ORCHESTRATOR, APIPOD_ORCHESTRATOR, CONSTS.ORCHESTRATOR.LOCAL)
    compute = _resolve_enum(compute, CONSTS.COMPUTE, APIPOD_COMPUTE, CONSTS.COMPUTE.DEDICATED)
    provider = _resolve_enum(provider, CONSTS.PROVIDER, APIPOD_PROVIDER, CONSTS.PROVIDER.LOCALHOST)

    backend_class, use_job_queue = _resolve_backend(orchestrator, compute, provider)

    job_queue = _create_job_queue() if use_job_queue else None

    if backend_class == SocaityFastAPIRouter:
        return backend_class(job_queue=job_queue, *args, **kwargs)
    else:
        return backend_class(*args, **kwargs)


def _resolve_enum(value, enum_cls, env_default, fallback):
    """Coerce a value into an enum member, falling back through env default and hard default."""
    if value is None:
        value = env_default
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            raise ValueError(f"Invalid {enum_cls.__name__} value: '{value}'. Choose from: {[e.value for e in enum_cls]}")
    if isinstance(value, enum_cls):
        return value
    return fallback


def _resolve_backend(
    orchestrator: CONSTS.ORCHESTRATOR,
    compute: CONSTS.COMPUTE,
    provider: CONSTS.PROVIDER,
) -> tuple:
    """
    Apply the configuration matrix and return (backend_class, use_job_queue).
    Raises for unsupported or not-yet-implemented combinations.
    """
    _raise_if_unsupported(compute, provider)

    if orchestrator == CONSTS.ORCHESTRATOR.SOCAITY:
        return _resolve_socaity(compute, provider)

    return _resolve_local(compute, provider)


def _raise_if_unsupported(compute: CONSTS.COMPUTE, provider: CONSTS.PROVIDER):
    unsupported = {
        (CONSTS.COMPUTE.SERVERLESS, CONSTS.PROVIDER.SCALEWAY),
        (CONSTS.COMPUTE.SERVERLESS, CONSTS.PROVIDER.AZURE),
    }
    if (compute, provider) in unsupported:
        raise NotImplementedError(
            f"Serverless compute on {provider.value} is not supported. "
            f"Use provider='runpod' for serverless or switch to dedicated compute."
        )


def _resolve_socaity(compute: CONSTS.COMPUTE, provider: CONSTS.PROVIDER) -> tuple:
    if compute == CONSTS.COMPUTE.DEDICATED:
        if provider in (CONSTS.PROVIDER.RUNPOD, CONSTS.PROVIDER.SCALEWAY, CONSTS.PROVIDER.AZURE):
            raise NotImplementedError(
                f"Celery backend for socaity + dedicated + {provider.value} is planned but not yet available."
            )
        if provider == CONSTS.PROVIDER.LOCALHOST:
            return SocaityFastAPIRouter, True
        # auto or any other -> FastAPI without queue
        return SocaityFastAPIRouter, False

    # serverless
    if provider == CONSTS.PROVIDER.LOCALHOST:
        return SocaityFastAPIRouter, True
    # auto or runpod -> RunPod router
    return SocaityRunpodRouter, False


def _resolve_local(compute: CONSTS.COMPUTE, provider: CONSTS.PROVIDER) -> tuple:
    if compute == CONSTS.COMPUTE.DEDICATED:
        return SocaityFastAPIRouter, False

    # serverless
    if provider == CONSTS.PROVIDER.LOCALHOST:
        return SocaityFastAPIRouter, True
    if provider == CONSTS.PROVIDER.RUNPOD:
        return SocaityRunpodRouter, False
    # auto -> RunPod router (same default as socaity serverless auto)
    if provider == CONSTS.PROVIDER.AUTO:
        return SocaityRunpodRouter, False

    raise NotImplementedError(f"Unsupported configuration: local + serverless + {provider.value}")


def _create_job_queue() -> JobQueueInterface:
    from apipod.core.job_queues.job_queue import JobQueue
    return JobQueue()
