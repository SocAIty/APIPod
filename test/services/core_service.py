"""Core service: the parameter shapes and file-handling an author actually writes.

- ``register_minimal``  two trivial endpoints (OpenAPI smoke test, config matrix).
- ``register``          full set: scalars, custom pydantic model, predict/echo_image
                        (used by fastSDK tests), broad media+model endpoint,
                        JobProgress lifecycle, single file upload.

The module-level ``app`` makes this file a valid ``apipod start`` entrypoint,
so fastSDK end-to-end tests can boot it as a subprocess via ``conftest.live_service``.

APIPod normalizes ``_`` to ``-`` in routes, so e.g. ``/mixed_media`` is served
at ``/mixed-media``.
"""

import time
from typing import List, Optional

from pydantic import BaseModel
from fastapi import UploadFile as FastAPIUploadFile

from apipod import APIPod, JobProgress, MediaFile, ImageFile, AudioFile, VideoFile, FileModel


class MoreParams(BaseModel):
    pam1: str = "pam1"
    pam2: int = 42


def register_minimal(app):
    @app.endpoint("/echo")
    def echo(text: str):
        return text

    @app.endpoint("/add")
    def add(a: int, b: int = 1):
        return a + b


def register(app):
    # Plain scalars — verifies standard type coercion.
    @app.endpoint("/scalars")
    def scalars(text: str, count: int, ratio: float = 1.0, flag: bool = False):
        return {"text": text, "count": count, "ratio": ratio, "flag": flag}

    # Custom pydantic model.
    @app.endpoint("/model")
    def model(order: MoreParams):
        return {"pam1": order.pam1, "pam2": order.pam2}

    # Clean scalar + media I/O; also the primary targets for fastSDK tests.
    @app.endpoint("/predict")
    def predict(text: str, times: int = 1):
        return text * times

    @app.endpoint("/echo_image")
    def echo_image(image: ImageFile):
        return image

    # One endpoint, many input options: optional/required media, raw UploadFile,
    # union media types, media lists, a custom model and plain scalars.
    @app.endpoint("/mixed_media")
    def mixed_media(
        job_progress: JobProgress,
        anyfile1: Optional[MediaFile],
        anyfile2: FileModel,
        anyfile3: FastAPIUploadFile,
        img: ImageFile | str | bytes | FileModel,
        audio: AudioFile,
        video: VideoFile,
        anyfiles: List[MediaFile],
        a_base_model: Optional[MoreParams],
        anint2: int,
        anyImages: List[ImageFile] = ["default_value"],
        astring: str = "master_of_desaster",
        anint: int = 42,
    ):
        return anyfile3, str, anyfile1.to_base64(), img.to_base64(), anyfiles

    # JobProgress + queue lifecycle.
    @app.post(path="/test_job_progress", queue_size=10)
    def job_progress_demo(job_progress: JobProgress, fries_name: str, amount: int = 1):
        job_progress.set_status(0.1, f"started new fries creation {fries_name}")
        time.sleep(0.2)
        job_progress.set_status(0.5, f"working on it, lots to do {amount}")
        time.sleep(0.2)
        job_progress.set_status(0.8, "almost done")
        time.sleep(0.2)
        return f"Your fries {fries_name} are ready"

    # Single file upload — also verifies base64 round-trip.
    @app.endpoint("/single_file_upload")
    def single_file_upload(job_progress: JobProgress, file1: ImageFile):
        return file1.to_base64()


# Runnable entrypoint: booted by ``apipod start`` / ``simulate`` in
# fastSDK end-to-end tests via ``conftest.live_service``.
app = APIPod()  # run intent injected via env by the CLI
register(app)


if __name__ == "__main__":
    app.start()
