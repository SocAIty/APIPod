from apipod.engine.llm.base_llm_mixin import _BaseLLMMixin


class _RunPodLLMMixin(_BaseLLMMixin):
    """
    LLM logic specific to the RunPod Router
    """

    def handle_llm_request(self, func, openai_req, req_model, res_model, endpoint_type, w_args, w_kwargs):
        if isinstance(openai_req, dict):
            openai_req = req_model.model_validate(openai_req)
            w_kwargs["payload"] = openai_req
        
        should_stream = getattr(openai_req, "stream", False) and endpoint_type != "embedding"

        if should_stream:
            return self._yield_native_stream(func, w_args, w_kwargs)
        
        raw_res = self._execute_sync_or_async(func, w_args, w_kwargs)
        return self._wrap_llm_response(raw_res, res_model, endpoint_type, openai_req)
