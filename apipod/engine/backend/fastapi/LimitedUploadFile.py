from fastapi import UploadFile, HTTPException, status


class LimitedUploadFile(UploadFile):
    """
    An UploadFile subclass that enforces a size limit on uploads.
    Replace the standard UploadFile with this class to enforce a size limit on uploads.
    """
    def __init__(self, *args, max_size_mb: float = None, **kwargs):
        """
        :param max_size_mb: Limit for file. if max_size_mb is None, no limit is enforced.
        """
        super().__init__(*args, **kwargs)
        self.max_size_mb = max_size_mb * 1024 * 1024 if max_size_mb else None
        self._size = 0

    async def write(self, data: bytes) -> None:
        """
        Overrides write method to enforce size limits.
        """
        if self.max_size_mb is not None:
            self._size += len(data)
            if self._size > self.max_size_mb:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds the limit of {self.max_size_mb} mb"
                )
        await super().write(data)

    #async def read(self, size: int = -1) -> bytes:
    #    """
    #    Overrides read to allow tracking the size during reads.
    #    """
    #    data = await super().read(size)
    #    self._size += len(data)
    #    if self.max_size_mb and self._size > self.max_size_mb:
    #        raise HTTPException(
    #            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    #            detail=f"File size exceeds the limit of {self.max_size_mb} bytes"
    #        )
    #    return dat