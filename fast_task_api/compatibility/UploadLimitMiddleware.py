from typing import Dict, Any

from starlette.middleware.base import RequestResponseEndpoint, BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status

from media_toolkit.utils.dependency_requirements import requires

try:
    from streaming_form_data import StreamingFormDataParser
except ImportError:
    pass

class MaxBodySizeException(Exception):
    """Exception raised when request body exceeds maximum size."""
    def __init__(self, body_len: int):
        self.body_len = body_len


@requires(['streaming_form_data', 'media-toolkit', 'starlette'])
class UploadLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that handles file upload size limits and streaming.

    Args:
        app: ASGI application
        max_upload_size: Maximum total upload size in bytes
    """

    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(
            self,
            request: Request,
            call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
            try:
                content_length = int(request.headers.get("content-length", 0))
                if content_length > self.max_upload_size:
                    return Response(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content="Request body too large"
                    )

                modified_request = await self._handle_streaming_upload(request)
                return await call_next(modified_request)

            except MaxBodySizeException as e:
                return Response(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=f"Upload size limit exceeded: {e.body_len} bytes"
                )

        return await call_next(request)

    async def _handle_streaming_upload(self, request: Request) -> Request:
        """Handle streaming upload and modify request with processed data."""
        parser = StreamingFormDataParser(headers=request.headers)
        form_data: Dict[str, Any] = {}

        async def process_stream():
            total_size = 0
            async for chunk in request.stream():
                total_size += len(chunk)
                if total_size > self.max_upload_size:
                    raise MaxBodySizeException(total_size)

                parser.data_received(chunk)

        await process_stream()

        # Modify request scope with processed data
        request.scope["form_data"] = form_data
        return request