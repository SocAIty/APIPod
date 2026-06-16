from typing import Optional, Tuple, Union

from apipod.common.constants import COMPUTE, PROVIDER
from apipod.common.settings import (
    APIPOD_COMPUTE,
    APIPOD_DIRECT,
    APIPOD_PROVIDER,
    APIPOD_SIMULATE,
    IS_MANAGED_DEPLOYMENT,
)
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.backend.runpod.router import SocaityRunpodRouter

# (backend_class, use_job_queue, runpod_simulate)
_Resolution = Tuple[type, bool, bool]

_NO_SERVERLESS = (PROVIDER.AZURE, PROVIDER.SCALEWAY)


def APIPod(
        simulate: Union[str, None] = None,
        direct: Union[bool, None] = None,
        *args, **kwargs
) -> Union[SocaityFastAPIRouter, SocaityRunpodRouter]:
    """
    Build the right backend for an APIPod service from a single *intent*.

    Socaity is the implicit orchestrator, so you never wire infrastructure by hand.
    You only pick how the service should run *locally*:

    - **Development** (default, ``APIPod()``): plain FastAPI — the fastest loop.
    - **Simulation** (``simulate="{compute}-{provider}"``): emulate a deployment
      locally. The target collapses compute + provider, e.g. ``"serverless"``,
      ``"serverless-runpod"``, ``"dedicated-azure"``. Compute defaults to
      ``serverless``. ``direct=True`` bypasses Socaity to emulate the provider's
      own serverless worker (currently RunPod).

    In a **managed deployment** (``SOCAITY_DEPLOYMENT_CERT`` verified) ``simulate``
    and ``direct`` are ignored: Socaity injects ``APIPOD_COMPUTE`` /
    ``APIPOD_PROVIDER`` and the real backend is selected from them.

    Args:
        simulate: deployment target to emulate, ``"{compute}-{provider}"``.
            ``None`` runs plain FastAPI for development.
        direct: emulate the provider's native serverless worker instead of the
            Socaity job-queue emulation. Only affects ``serverless-runpod``.
    """
    if IS_MANAGED_DEPLOYMENT:
        backend_class, use_job_queue, runpod_simulate = _resolve_managed()
    else:
        backend_class, use_job_queue, runpod_simulate = _resolve_intent(simulate, direct)

    if backend_class is SocaityRunpodRouter:
        return SocaityRunpodRouter(simulate=runpod_simulate, *args, **kwargs)

    # FastAPI backend: a job queue (+ stream store) turns it into the serverless
    # emulation; without one it is plain FastAPI. A deployment may inject its own.
    job_queue = kwargs.pop("job_queue", None)
    if job_queue is not None:
        use_job_queue = True
    elif use_job_queue:
        from apipod.engine.queue.job_queue import JobQueue
        job_queue = JobQueue()

    if use_job_queue and "stream_store" not in kwargs:
        kwargs["stream_store"] = _create_stream_store()

    return SocaityFastAPIRouter(job_queue=job_queue, *args, **kwargs)


def _resolve_intent(simulate: Optional[str], direct: Optional[bool]) -> _Resolution:
    """Resolve a local run (development or simulation) into a backend selection."""
    target = APIPOD_SIMULATE if simulate is None else simulate

    # Development: no simulation requested -> plain FastAPI.
    if simulate is None and not APIPOD_SIMULATE:
        return SocaityFastAPIRouter, False, False

    direct = APIPOD_DIRECT if direct is None else bool(direct)
    compute, provider = _parse_target(target)

    if compute is COMPUTE.DEDICATED:
        # "Standard FastAPI"; with a named provider it emulates a direct client.
        return SocaityFastAPIRouter, False, False

    # serverless
    if provider in _NO_SERVERLESS:
        print(f"Warning: {provider.value} does not support serverless. "
              f"Defaulting to FastAPI + Local Job Queue.")
        return SocaityFastAPIRouter, True, False

    if provider is PROVIDER.RUNPOD and direct:
        # Emulate RunPod's native serverless worker locally.
        return SocaityRunpodRouter, False, True

    # Default serverless: Socaity emulation = FastAPI + Local Job Queue.
    return SocaityFastAPIRouter, True, False


def _resolve_managed() -> _Resolution:
    """Pick the real production backend from the env vars Socaity injects."""
    compute = COMPUTE(APIPOD_COMPUTE)
    provider = PROVIDER(APIPOD_PROVIDER)

    if compute is COMPUTE.SERVERLESS and provider is PROVIDER.RUNPOD:
        return SocaityRunpodRouter, False, False  # real serverless worker
    if compute is COMPUTE.SERVERLESS:
        return SocaityFastAPIRouter, True, False
    return SocaityFastAPIRouter, False, False  # dedicated (queue injected if needed)


def _parse_target(target: str) -> Tuple[COMPUTE, Optional[PROVIDER]]:
    """Parse a ``"{compute}-{provider}"`` target. Provider is optional; compute defaults to serverless."""
    if not target:
        return COMPUTE.SERVERLESS, None

    compute_str, _, provider_str = target.partition("-")
    try:
        compute = COMPUTE(compute_str)
    except ValueError:
        raise ValueError(f"Invalid compute '{compute_str}'. Choose from: {[c.value for c in COMPUTE]}")

    if not provider_str:
        return compute, None
    try:
        return compute, PROVIDER(provider_str)
    except ValueError:
        raise ValueError(f"Invalid provider '{provider_str}'. Choose from: {[p.value for p in PROVIDER]}")


def _create_stream_store():
    from apipod.engine.streaming.local_stream_store import LocalStreamStore
    return LocalStreamStore()
