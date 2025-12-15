from typing import List, Optional

from pydantic import BaseModel

import time

from apipod import APIPod
from apipod import JobProgress
from apipod import MediaFile, ImageFile, AudioFile, VideoFile, FileModel
from fastapi import UploadFile as fastapiUploadFile

app = APIPod()


@app.post(path="/test_job_progress", queue_size=10)
def test_job_progress(job_progress: JobProgress, fries_name: str, amount: int = 1):
    job_progress.set_status(0.1, f"started new fries creation {fries_name}")
    time.sleep(1)
    job_progress.set_status(0.5, f"I am working on it. Lots of work to do {amount}")
    time.sleep(2)
    job_progress.set_status(0.8, "Still working on it. Almost done")
    time.sleep(2)
    return f"Your fries {fries_name} are ready"


class MoreParams(BaseModel):
    pam1: str = "pam1"
    pam2: int = 42


@app.endpoint("/mixed_media")
def test_mixed_media(
    job_progress: JobProgress,
    anyfile1: Optional[MediaFile],
    anyfile2: FileModel,
    anyfile3: fastapiUploadFile,
    img: ImageFile | str | bytes | FileModel,
    audio: AudioFile,
    video: VideoFile,
    anyfiles: List[MediaFile],
    a_base_model: Optional[MoreParams],
    anint2: int,
    anyImages: List[ImageFile] = ["default_value"],
    astring: str = "master_of_desaster",
    anint: int = 42
):
    content_one = anyfile1.to_base64()
    content_two = img.to_base64()
    return anyfile3, str, content_one, content_two, anyfiles


@app.endpoint("test_single_file_upload")
def test_single_file_upload(
    job_progress: JobProgress,
    file1: ImageFile
):
    return file1.to_base64()


@app.endpoint("/make_fries", method="POST")
def test(
    mymom: str,
    file1: fastapiUploadFile
):
    return "nok"


if __name__ == "__main__":
    # Runpod version
    app.start(port=8000, environment="localhost")
    # app.start(environment="serverless", port=8000)
    # app.start(environment="localhost", port=8000)

