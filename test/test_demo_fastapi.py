from apipod import APIPod
from apipod import JobProgress
from apipod import ImageFile
import time
import numpy as np


import fastapi


app = fastapi.FastAPI()


# define the app including your provider (fastapi, runpod..)
app = APIPod()


# add endpoints to your service
@app.endpoint("/predict")
def predict(my_param1: str, my_param2: int = 0):
    return f"my_awesome_prediction {my_param1} {my_param2}"


@app.endpoint("/img2img")
def img2img(upload_img: ImageFile):
    img_as_numpy = np.array(upload_img)  # this returns a np.array read with cv2
    # Do some hard work here...
    # img_as_numpy = img2img(img_as_numpy)
    return ImageFile().from_np_array(img_as_numpy)


@app.get(path="/prompt_helper", queue_size=100)
def prompt_helper(job_progress: JobProgress, text: str, enhancement: int = 1):
    """
    Submit a prompt and we will improve its quality to make the best out of your images.
    :return: a super enhanced prompt
    """
    job_progress.set_status(0.1, "enhancing your prompt with fancy addons like 8k, ultra high res")
    time.sleep(1)
    text += " 8k, ultra high res, perfect anatomy"
    job_progress.set_status(0.5, f"I am working on it. Lots of work to do {enhancement}")
    time.sleep(2)
    job_progress.set_status(0.8, "Still working on it. Almost done")
    time.sleep(2)
    return f"Your enhanced prompt {text} is ready"


@app.endpoint("/test_no_queue_endpoint", methods=["POST"], max_upload_file_size_mb=10)
def test_single_file_upload(file1: ImageFile):
    return file1


# start and run the server
app.start()
