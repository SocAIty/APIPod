from typing import Optional, Tuple, Union

from apipod.common import settings
from apipod.common.constants import COMPUTE, PROVIDER
from apipod.engine.backend.fastapi.router import SocaityFastAPIRouter
from apipod.engine.backend.runpod.router import SocaityRunpodRouter
from apipod.engine.streaming.local_stream_store import LocalStreamStore


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

    - **Development** (default, ``APIPod()``): plain FastAPI — the fastest loop
      (env defaults ``APIPOD_COMPUTE=dedicated``, ``APIPOD_PROVIDER=localhost``).
    - **Simulation** (``simulate="{compute}-{provider}"``): emulate a deployment
      locally. The target collapses compute + provider, e.g. ``"serverless"``,
      ``"serverless-runpod"``, ``"dedicated-azure"``. Compute defaults to
      ``serverless``. ``direct=True`` bypasses Socaity to emulate the provider's
      own serverless worker (currently RunPod).
    - **Deployed image** (``APIPOD_COMPUTE`` / ``APIPOD_PROVIDER`` set, e.g. by
      the Dockerfile or platform): select the real backend. User serverless
      RunPod deploys do **not** need a cert — ``serverless`` + ``runpod`` yields
      the real RunPod worker.

    ``SOCAITY_DEPLOYMENT_CERT`` marks an *official* staff deployment
    (``IS_MANAGED_DEPLOYMENT``). It does not gate user backend selection; when
    present it only forces the env-based path (ignores local ``simulate`` /
    ``direct``).

    Args:
        simulate: deployment target to emulate, ``"{compute}-{provider}"``.
            ``None`` uses env compute/provider (development defaults or deploy).
        direct: emulate the provider's native serverless worker instead of the
            Socaity job-queue emulation. Only affects ``serverless-runpod``.
    """
    if settings.IS_MANAGED_DEPLOYMENT:
        # Official staff deploy: always honor platform env, ignore simulate.
        backend_class, use_job_queue, runpod_simulate = _resolve_from_env()
    elif simulate is not None or settings.APIPOD_SIMULATE:
        backend_class, use_job_queue, runpod_simulate = _resolve_intent(simulate, direct)
    else:
        # User deploy image or local development defaults.
        backend_class, use_job_queue, runpod_simulate = _resolve_from_env()

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
        kwargs["stream_store"] = LocalStreamStore()

    return SocaityFastAPIRouter(job_queue=job_queue, *args, **kwargs)


def _resolve_intent(simulate: Optional[str], direct: Optional[bool]) -> _Resolution:
    """Resolve a local simulation into a backend selection."""
    target = settings.APIPOD_SIMULATE if simulate is None else simulate
    direct = settings.APIPOD_NATIVE if direct is None else bool(direct)
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


def _resolve_from_env() -> _Resolution:
    """Pick the real backend from ``APIPOD_COMPUTE`` / ``APIPOD_PROVIDER``.

    Used for deployed images (user or official) and for local development
    defaults (``dedicated`` + ``localhost`` → plain FastAPI).
    """
    compute = COMPUTE(settings.APIPOD_COMPUTE)
    provider = PROVIDER(settings.APIPOD_PROVIDER)

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
