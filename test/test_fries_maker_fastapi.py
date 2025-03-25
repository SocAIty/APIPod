from typing import List, Optional

from pydantic import BaseModel

from fast_task_api import JobResult
from fast_task_api import FastTaskAPI
from fast_task_api import JobProgress
from fast_task_api import MediaFile, ImageFile, AudioFile, VideoFile, FileModel
from fastapi import UploadFile as fastapiUploadFile

app = FastTaskAPI()

#@app.post(path="/make_fries", queue_size=10)
#def make_fries(job_progress: JobProgress, fries_name: str, amount: int = 1):
#    job_progress.set_status(0.1, f"started new fries creation {fries_name}")
#    time.sleep(1)
#    job_progress.set_status(0.5, f"I am working on it. Lots of work to do {amount}")
#    time.sleep(2)
#    job_progress.set_status(0.8, "Still working on it. Almost done")
#    time.sleep(2)
#    return f"Your fries {fries_name} are ready"



class MoreParams(BaseModel):
    pam1: str = "pam1"
    pam2: int = 42

@app.task_endpoint("/make_video_fries")
def make_video_fries(
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
    potato_one_content = anyfile1.to_base64()
    potato_two_content = img.to_base64()
    return potato_two_content


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

