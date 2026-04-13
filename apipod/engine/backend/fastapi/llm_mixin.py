from fastapi.responses import StreamingResponse

from apipod.engine.llm.base_llm_mixin import _BaseLLMMixin


class _FastApiLlmMixin(_BaseLLMMixin):
    """
    LLM execution logic for the FastAPI provider backend.
    """

    async def handle_llm_request(self, func, openai_req, should_use_queue, res_model, endpoint_type, **kwargs):
        should_stream = getattr(openai_req, "stream", False) and endpoint_type != "embedding"

        if should_stream:
            result = await self._execute_func(func, payload=openai_req, **kwargs)
            return StreamingResponse(self._stream_generator(result), media_type="text/event-stream")

        if should_use_queue:
            from apipod.engine.jobs.job_result import JobResultFactory

            job = self.job_queue.add_job(
                job_function=func,
                job_params={"payload": openai_req.dict()}
            )
            ret_job = JobResultFactory.from_base_job(job)
            return ret_job

        raw_res = await self._execute_func(func, payload=openai_req, **kwargs)
        return self._wrap_llm_response(raw_res, res_model, endpoint_type, openai_req)
