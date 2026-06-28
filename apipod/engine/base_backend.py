import asyncio
import concurrent.futures
import functools
import inspect
from abc import abstractmethod
from typing import Any, Callable, Union
import importlib.metadata
from apipod.common import constants
from apipod.engine.compatibility.HealthCheck import HealthCheck
from apipod.common.settings import APIPOD_PORT


class _BaseBackend:
    """
    Base class for all routers.
    """
    def __init__(
            self, title: str = "APIPod", summary: str = "Create web-APIs for long-running tasks", *args, **kwargs
    ):
        if title is None:
            title = "APIPod"
        if summary is None:
            summary = "Create web-APIs for long-running tasks"

        self.title = title
        self.summary = summary
        self._health_check = HealthCheck()
        self.version = importlib.metadata.version("apipod")

    # ------------------------------------------------------------------
    # Execution runtime
    #
    # Executing a user-registered endpoint while transparently resolving
    # sync/async is the one operation every backend shares. There are exactly
    # two authoritative entry points so behaviour can never drift between the
    # worker thread, the RunPod handler and the direct FastAPI path.
    # ------------------------------------------------------------------
    def run_callable(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* from a synchronous context, resolving coroutines.

        Used by the two real sync execution sites: the local queue worker thread
        and the RunPod handler. Sync functions run directly; coroutine functions
        are driven to completion on an event loop (a fresh one is created and torn
        down when no usable loop exists, or when the current loop is already
        running the call is offloaded to a worker thread).
        """
        if not inspect.iscoroutinefunction(func):
            return func(*args, **kwargs)

        coro = func(*args, **kwargs)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and not loop.is_running():
            return loop.run_until_complete(coro)

        if loop is not None and loop.is_running():
            # Already inside a running loop (rare in sync handlers): run the
            # coroutine to completion on a separate thread to avoid re-entrancy.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, coro).result()

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
            asyncio.set_event_loop(None)

    async def run_callable_async(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* from an async context without blocking the event loop.

        Used by the direct (non-queued) FastAPI response path. Coroutine
        functions are awaited; sync functions are offloaded to the default
        thread-pool executor.
        """
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    @property
    def status(self) -> constants.SERVER_HEALTH:
        return self._health_check.status

    @status.setter
    def status(self, value: constants.SERVER_HEALTH):
        self._health_check.status = value

    def get_health(self) -> Union[dict, str]:
        stat, message = self._health_check.get_health_response()
        return message

    @abstractmethod
    def get_job(self, job_id: str):
        """
        Get the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    def cancel_job(self, job_id: str):
        """
        Cancel the job with the given job_id if it exists.
        :param job_id: The job id of a previously created job.
        :return:
        """
        raise NotImplementedError("Implement in subclass")

    @abstractmethod
    def start(self, port: int = APIPOD_PORT, *args, **kwargs):
        raise NotImplementedError("Implement in subclass")

    def include_router(self, router, prefix: str = "", **kwargs):
        """Mount a nested APIPod router under *prefix* (FastAPI ``include_router`` semantics)."""
        raise NotImplementedError("Implement in subclass")

    def endpoint(self, path: str = None, *args, **kwargs):
        """
        Add a route to the app. 
        Can be a task route (async job) or standard route depending on configuration.

        :param path:
            In case of fastapi will be resolved as url in form http://{host:port}/{prefix}/{path}
            In case of runpod will be resolved as url in form http://{host:port}?route={path}
        :param args: any other arguments to configure the app
        :param kwargs: any other keyword arguments to configure the app
        :return:
        """
        raise NotImplementedError("Implement in subclass. Use a decorator for that.")

    def get(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")

    def post(self, path: str = None, queue_size: int = 1, *args, **kwargs):
        raise NotImplementedError("Implement in subclass. Consider using add_route instead.")
