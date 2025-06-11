from fastapi import Request
from fastapi.responses import JSONResponse
from fast_task_api.core.routers._exceptions import JobException


class FastAPIExceptionHandler:
    async def global_exception_handler(self, request: Request, exc: Exception):
        if isinstance(exc, JobException):
            # error was raised before the job_function was called. For example an upload failed or other problem.
            return JSONResponse(
                status_code=422,
                content=str(exc)
            )
        else:
            import traceback
            print(f"Unexpected error: {exc} traceback: {traceback.format_exc()}")

        # Return a user-friendly error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Check your input variables and try again later."
            }
        )
