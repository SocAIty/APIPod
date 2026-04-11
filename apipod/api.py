from apipod.common import constants
from apipod.common.settings import APIPOD_ORCHESTRATOR, APIPOD_COMPUTE, APIPOD_PROVIDER
from apipod.engine.base_backend import _BaseBackend
from apipod.engine.backend.runpod.router import SocaityRunpodRouter
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.queue.job_queue_interface import JobQueueInterface

from typing import Union


def APIPod(
        orchestrator: Union[constants.ORCHESTRATOR, str, None] = None,
        compute: Union[constants.COMPUTE, str, None] = None,
        provider: Union[constants.PROVIDER, str, None] = None,
        *args, **kwargs
) -> Union[_BaseBackend, SocaityRunpodRouter, SocaityFastAPIRouter]:
    """
    Initialize an APIPod router with the appropriate backend based on the deployment configuration.

    The resulting backend is determined by the combination of orchestrator, compute, and provider:

    | Orchestrator | Compute    | Provider  | Backend                    |
    |------------- |----------- |---------- |--------------------------- |
    | socaity      | dedicated  | auto      | FastAPI                    |
    | socaity      | dedicated  | localhost | FastAPI + job queue (test) |
    | socaity      | dedicated  | socaity   | FastAPI + redis (prod)     |
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
    orchestrator = _resolve_enum(orchestrator, constants.ORCHESTRATOR, APIPOD_ORCHESTRATOR, constants.ORCHESTRATOR.LOCAL)
    compute = _resolve_enum(compute, constants.COMPUTE, APIPOD_COMPUTE, constants.COMPUTE.DEDICATED)
    provider = _resolve_enum(provider, constants.PROVIDER, APIPOD_PROVIDER, constants.PROVIDER.LOCALHOST)

    backend_class, use_job_queue = _resolve_backend(orchestrator, compute, provider)

    custom_job_queue = kwargs.get('job_queue')
    if custom_job_queue:
        use_job_queue = True
        job_queue = custom_job_queue
    else:
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
    orchestrator: constants.ORCHESTRATOR,
    compute: constants.COMPUTE,
    provider: constants.PROVIDER,
) -> tuple:
    """
    Apply the configuration matrix and return (backend_class, use_job_queue).
    Raises for unsupported or not-yet-implemented combinations.
    """
    _raise_if_unsupported(compute, provider)

    if orchestrator == constants.ORCHESTRATOR.SOCAITY:
        return _resolve_socaity(compute, provider)

    return _resolve_local(compute, provider)


def _raise_if_unsupported(compute: constants.COMPUTE, provider: constants.PROVIDER):
    unsupported = {
        (constants.COMPUTE.SERVERLESS, constants.PROVIDER.SCALEWAY),
        (constants.COMPUTE.SERVERLESS, constants.PROVIDER.AZURE),
    }
    if (compute, provider) in unsupported:
        raise NotImplementedError(
            f"Serverless compute on {provider.value} is not supported. "
            f"Use provider='runpod' for serverless or switch to dedicated compute."
        )


def _resolve_socaity(compute: constants.COMPUTE, provider: constants.PROVIDER) -> tuple:
    if compute == constants.COMPUTE.DEDICATED:
        if provider == constants.PROVIDER.SOCAITY:
            return SocaityFastAPIRouter, True

        if provider in (constants.PROVIDER.RUNPOD, constants.PROVIDER.SCALEWAY, constants.PROVIDER.AZURE):
            raise NotImplementedError(
                f"Celery backend for socaity + dedicated + {provider.value} is planned but not yet available."
            )
        if provider == constants.PROVIDER.LOCALHOST:
            return SocaityFastAPIRouter, True
        # auto or any other -> FastAPI without queue
        return SocaityFastAPIRouter, False

    # serverless
    if provider == constants.PROVIDER.LOCALHOST:
        return SocaityFastAPIRouter, True
    # auto or runpod -> RunPod router
    return SocaityRunpodRouter, False


def _resolve_local(compute: constants.COMPUTE, provider: constants.PROVIDER) -> tuple:
    if compute == constants.COMPUTE.DEDICATED:
        return SocaityFastAPIRouter, False

    # serverless
    if provider == constants.PROVIDER.LOCALHOST:
        return SocaityFastAPIRouter, True
    if provider == constants.PROVIDER.RUNPOD:
        return SocaityRunpodRouter, False
    # auto -> RunPod router (same default as socaity serverless auto)
    if provider == constants.PROVIDER.AUTO:
        return SocaityRunpodRouter, False

    raise NotImplementedError(f"Unsupported configuration: local + serverless + {provider.value}")


def _create_job_queue(provider: constants.PROVIDER) -> JobQueueInterface:
    from apipod.engine.queue.job_queue import JobQueue

    return JobQueue()
