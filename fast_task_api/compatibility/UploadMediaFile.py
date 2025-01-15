import asyncio
import io
from typing import Optional

from fastapi import UploadFile, HTTPException, status



class UploadMediaFile(UploadFile):
    """
    Enhanced UploadFile that supports streaming and size limits.
    Integrates with MediaFile for advanced file handling capabilities.

    Args:
        filename: Name of the file
        content_type: MIME type of the file
        max_size: Maximum file size in bytes
        use_temp_file: Whether to use temporary file storage
    """

    def __init__(
            self,
            filename: str,
            content_type: str,
            max_size: Optional[int] = None,
            use_temp_file: bool = False
    ):
        super().__init__(filename=filename, content_type=content_type)
        self.max_size = max_size
        self._size = 0
        self._buffer = io.BytesIO()
        self._chunks = asyncio.Queue()
        self._closed = False

    async def write(self, data: bytes) -> None:
        """
        Write data to the file buffer while checking size limits.

        Args:
            data: Bytes to write

        Raises:
            HTTPException: If file size exceeds limit
        """
        if self._closed:
            raise ValueError("Cannot write to closed file")

        self._size += len(data)
        if self.max_size and self._size > self.max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds limit of {self.max_size} bytes"
            )

        await self._chunks.put(data)
        self._buffer.write(data)

    async def read(self, size: int = -1) -> bytes:
        """
        Read data from the file buffer.

        Args:
            size: Number of bytes to read, -1 for all

        Returns:
            Bytes read from buffer
        """
        if size == -1:
            return self._buffer.getvalue()

        return self._buffer.read(size)

    def to_media_file(self) -> 'MediaFile':
        """Convert to MediaFile instance for advanced processing."""
        from media_toolkit.core.media_file import MediaFile
        media_file = MediaFile(
            file_name=self.filename,
            content_type=self.content_type
        )
        media_file.from_bytesio(self._buffer, copy=True)
        return media_file