try:
    import redis
except ImportError:
    raise ImportError("Redis is required to use RedisJobQueue. Install with 'pip install redis'.")

import pickle
import logging
import inspect
from typing import Optional, Callable, TypeVar, Any
from datetime import datetime

from apipod.core.job_queues.job_queue_interface import JobQueueInterface
from apipod.core.job.base_job import BaseJob, JOB_STATUS
from apipod.core.job.job_progress import JobProgress

T = TypeVar('T', bound=BaseJob)

logger = logging.getLogger(__name__)


class RedisJobProgress(JobProgress):
    """
    A JobProgress implementation that updates status in Redis.
    Used by workers to report progress.
    """
    def __init__(self, redis_client: Any, job_id: str, job_prefix: str):
        super().__init__()
        self.redis = redis_client
        self.job_key = f"{job_prefix}:{job_id}"

    def set_status(self, progress: float = None, message: str = None):
        super().set_status(progress, message)
        updates = {}
        if progress is not None:
            updates["progress"] = progress
        if message is not None:
            updates["progress_msg"] = message

        if updates:
            try:
                self.redis.hset(self.job_key, mapping=updates)
            except Exception as e:
                logger.error(f"Failed to update job progress in Redis: {e}")


class RedisJobQueue(JobQueueInterface[T]):
    """
    Redis-backed JobQueue implementation.
    Persists jobs to Redis and uses Redis Lists for queuing.

    Structure in Redis:
    - Hash: {job_prefix}:{job_id} -> Job Data (status, result, error, progress, etc.)
    - List: {queue_prefix}:{func_name} -> [job_id, job_id, ...]
    - String: {queue_prefix}:limit:{func_name} -> int (queue limit)
    """

    def __init__(
        self,
        redis_url: str,
        delete_orphan_jobs_after_s: int = 60 * 30,
        queue_prefix: str = "apipod:queue",
        job_prefix: str = "apipod:job"
    ):
        self.redis_url = redis_url
        self.redis = redis.from_url(redis_url, decode_responses=False)  # Keep binary for pickle
        self.queue_prefix = queue_prefix
        self.job_prefix = job_prefix
        self._delete_orphan_jobs_after_seconds = delete_orphan_jobs_after_s

    def set_queue_size(self, job_function: Callable, queue_size: int = 500) -> None:
        """Set the queue size limit for a specific function."""
        key = f"{self.queue_prefix}:limit:{job_function.__name__}"
        self.redis.set(key, str(queue_size))

    def _get_queue_size(self, func_name: str) -> int:
        """Get the queue size limit for a specific function."""
        key = f"{self.queue_prefix}:limit:{func_name}"
        val = self.redis.get(key)
        return int(val) if val else 500

    def add_job(self, job_function: Callable, job_params: Optional[dict] = None) -> T:
        """
        Add a job to the Redis queue.
        """
        # 1. Create BaseJob instance locally to get ID and initial state
        job = BaseJob(job_function=job_function, job_params=job_params)
        func_name = job_function.__name__

        # 2. Check Queue Limit
        queue_key = f"{self.queue_prefix}:{func_name}"
        current_depth = self.redis.llen(queue_key)
        max_size = self._get_queue_size(func_name)

        if current_depth >= max_size:
            job.status = JOB_STATUS.FAILED
            job.error = f"Queue size limit reached for {func_name}"
            # We don't persist failed jobs due to queue limit to save space,
            # or we could persist them with a short TTL.
            return job

        # 3. Serialize Data
        try:
            pickled_params = pickle.dumps(job.job_params)
        except Exception as e:
            job.status = JOB_STATUS.FAILED
            job.error = f"Failed to serialize job params: {e}"
            return job

        job_data = {
            "id": job.id,
            "function_name": func_name,
            "params": pickled_params,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "queued_at": datetime.utcnow().isoformat(),
            "timeout_seconds": 3600,  # TODO: Make configurable
            "progress": 0.0,
            "progress_msg": ""
        }

        # 4. Save to Redis (Atomic Pipeline)
        job_key = f"{self.job_prefix}:{job.id}"
        pipe = self.redis.pipeline()
        # Store hash (mapping handles dict to hash conversion)
        pipe.hset(job_key, mapping=job_data)
        # Set Expiry (e.g., 24 hours)
        pipe.expire(job_key, 3600 * 24)
        # Push to Queue
        pipe.rpush(queue_key, job.id)

        try:
            pipe.execute()
            job.status = JOB_STATUS.QUEUED
        except Exception as e:
            logger.error(f"Redis error adding job: {e}")
            job.status = JOB_STATUS.FAILED
            job.error = "Internal System Error: Could not queue job."

        return job

    def get_job(self, job_id: str) -> Optional[T]:
        """
        Retrieve a job from Redis and reconstruct it.
        """
        job_key = f"{self.job_prefix}:{job_id}"

        try:
            data = self.redis.hgetall(job_key)
        except Exception as e:
            logger.error(f"Redis error getting job {job_id}: {e}")
            return None

        if not data:
            return None

        # Decode helper since we used decode_responses=False
        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v

        # Extract fields
        # Note: 'params', 'result' might be bytes if pickled.
        # 'status', 'function_name', dates are strings.

        func_name = decode(data.get(b"function_name", b"unknown"))
        status_str = decode(data.get(b"status", JOB_STATUS.FAILED.value))

        # Reconstruct Params
        params = {}
        if b"params" in data:
            try:
                params = pickle.loads(data[b"params"])
            except Exception:
                params = {"error": "Could not deserialize parameters"}

        # Create dummy function object
        def dummy_func():
            pass
        dummy_func.__name__ = func_name

        job = BaseJob(job_function=dummy_func, job_params=params)
        job.id = decode(data.get(b"id", job_id))

        try:
            job.status = JOB_STATUS(status_str)
        except ValueError:
            job.status = JOB_STATUS.FAILED

        # Result handling
        if b"result" in data:
            try:
                # Assuming result is also pickled for consistency,
                # or string if it was simple.
                # For this impl, let's assume pickle for full compatibility.
                job.result = pickle.loads(data[b"result"])
            except Exception:
                # Fallback to decoding as string if pickle fails
                job.result = decode(data[b"result"])

        job.error = decode(data.get(b"error"))

        # Timestamps
        def parse_date(key_bytes):
            if key_bytes in data:
                try:
                    return datetime.fromisoformat(decode(data[key_bytes]))
                except Exception:
                    return None
            return None

        job.created_at = parse_date(b"created_at") or job.created_at
        job.queued_at = parse_date(b"queued_at")
        job.execution_started_at = parse_date(b"execution_started_at")
        job.execution_finished_at = parse_date(b"execution_finished_at")

        # Progress
        progress_val = decode(data.get(b"progress", 0.0))
        progress_msg = decode(data.get(b"progress_msg", ""))
        try:
            job.job_progress.set_status(float(progress_val), progress_msg)
        except Exception:
            pass

        return job

    def cancel_job(self, job_id: str) -> None:
        """
        Mark a job as cancelled.
        """
        job_key = f"{self.job_prefix}:{job_id}"

        updates = {
            "status": JOB_STATUS.FAILED.value,
            "error": "Job cancelled by user",
            "execution_finished_at": datetime.utcnow().isoformat()
        }

        try:
            self.redis.hset(job_key, mapping=updates)
            # Publish control message
            self.redis.publish("apipod:control", f"cancel:{job_id}")
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")

    def shutdown(self) -> None:
        try:
            self.redis.close()
        except Exception:
            pass

    # --- Worker Helper Methods (Optional but requested for functionality) ---

    def _worker_dequeue(self, func_name: str, timeout: int = 5) -> Optional[str]:
        """
        Worker method: Pop a job ID from the queue.
        """
        queue_key = f"{self.queue_prefix}:{func_name}"
        # blpop returns (key, value) tuple or None
        item = self.redis.blpop(queue_key, timeout=timeout)
        if item:
            return item[1].decode('utf-8')
        return None

    def _worker_process_job(self, job_id: str, func_registry: dict) -> None:
        """
        Worker method: Execute a job.

        :param job_id: The ID of the job to process
        :param func_registry: Dict mapping function names to callable objects
        """
        job_key = f"{self.job_prefix}:{job_id}"

        # 1. Fetch Job Data
        data = self.redis.hgetall(job_key)
        if not data:
            return  # Job expired or gone

        def decode(v):
            return v.decode('utf-8') if isinstance(v, bytes) else v

        func_name = decode(data.get(b"function_name"))

        # 2. Update Status to Processing
        self.redis.hset(job_key, mapping={
            "status": JOB_STATUS.PROCESSING.value,
            "execution_started_at": datetime.utcnow().isoformat()
        })

        # 3. Locate Function
        func = func_registry.get(func_name)
        if not func:
            error_msg = f"Function {func_name} not found in registry"
            self.redis.hset(job_key, mapping={
                "status": JOB_STATUS.FAILED.value,
                "error": error_msg,
                "execution_finished_at": datetime.utcnow().isoformat()
            })
            return

        # 4. Deserialize Params
        try:
            params = pickle.loads(data[b"params"])
        except Exception as e:
            self.redis.hset(job_key, mapping={
                "status": JOB_STATUS.FAILED.value,
                "error": f"Param deserialization failed: {e}",
                "execution_finished_at": datetime.utcnow().isoformat()
            })
            return

        # 5. Inject RedisJobProgress
        # We check if the function expects job_progress
        sig = inspect.signature(func)
        if "job_progress" in sig.parameters:
            params["job_progress"] = RedisJobProgress(self.redis, job_id, self.job_prefix)

        # 6. Execute
        try:
            result = func(**params)

            # Serialize Result
            result_bytes = pickle.dumps(result)

            self.redis.hset(job_key, mapping={
                "status": JOB_STATUS.FINISHED.value,
                "result": result_bytes,
                "execution_finished_at": datetime.utcnow().isoformat(),
                "progress": 1.0
            })

        except Exception as e:
            # import traceback
            # traceback.print_exc()
            self.redis.hset(job_key, mapping={
                "status": JOB_STATUS.FAILED.value,
                "error": str(e),
                "execution_finished_at": datetime.utcnow().isoformat()
            })
