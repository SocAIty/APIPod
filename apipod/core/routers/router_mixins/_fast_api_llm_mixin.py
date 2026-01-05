from fastapi.responses import StreamingResponse

from apipod.core.routers.router_mixins._base_llm_mixin import _BaseLLMMixin
from apipod.settings import SERVER_DOMAIN

class _FastAPILLMMixin(_BaseLLMMixin):
    """
    LLM logic specific to the FastAPI Router
    """

    async def handle_llm_request(self, func, openai_req, should_use_queue, res_model, endpoint_type, **kwargs):
        should_stream = getattr(openai_req, "stream", False) and endpoint_type != "embedding"

        if should_stream:
            result = await self._execute_func(func, payload=openai_req, **kwargs)
            return StreamingResponse(self._stream_generator(result), media_type="text/event-stream")
        
        if should_use_queue:
            from apipod.core.job.job_result import JobResultFactory
            job = self.job_queue.add_job(
                job_function=func,
                job_params={"payload": openai_req.dict()}
            )
            ret_job = JobResultFactory.from_base_job(job)
            ret_job.refresh_job_url = f"{SERVER_DOMAIN}/status?job_id={ret_job.id}"
            return ret_job

        raw_res = await self._execute_func(func, payload=openai_req, **kwargs)
        return self._wrap_llm_response(raw_res, res_model, endpoint_type, openai_req)