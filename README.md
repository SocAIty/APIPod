<h1 align="center">APIPod</h1>
<h3 align="center">Build, Deploy, and publish AI Services with Ease</h3>

<p align="center">
  <a href="https://www.socaity.com">
    <img src="docs/example_images/socaity_router_icon.png" height="200" alt="APIPod Logo" />
  </a>
</p>

<p align="center">
  <b>APIPod</b> is the way for building and deploying AI services. <br/>
  Combining the developer experience of <b>FastAPI</b> with the power of <b>Serverless GPU</b> computing.
</p>

<p align="center">
  <a href="#why-apipod">Why APIPod</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#deployment">Deployment</a> 
</p>

---

## Why APIPod?

Building AI services is complex: file handling, long-running inference, job queues, deployment, scaling, and hosting provider choices all create friction at every step.

**APIPod** solves this by standardizing the entire stack.

### 🚀 Highlights
1.  **Write Powerful APIs Instantly**: Built on top of FastAPI, it feels familiar but comes with batteries included for AI services.
2.  **Standardized I/O**: Painless handling of Images, Audio, and Video via [MediaToolkit](https://github.com/SocAIty/media-toolkit).
3.  **Automatic packaging**: The package can configure docker and deployment for you. No Cuda hell; the package knows compatible options.
4.  **Streamlined Deployment**: Deploy as a standard container or to serverless providers (**Socaity.ai** or **RunPod**)  with zero configuration changes. Auth included. 
5.  **Native SDK**: Built-in support for **Asynchronous Job Queues**, polling & and progress tracking via [fastSDK](https://github.com/SocAIty/fastSDK)

## Installation

```bash
pip install apipod
```

## Quick Start

### 1. Create your Service

Zero-Hassle Migration: Replacing `FastAPI` with `APIPod` gives you instant access to all APIPod capablities..

```python
from apipod import APIPod, ImageFile

# 1. Initialize APIPod (Drop-in replacement for FastAPI)
app = APIPod()

# 2. Define a standard endpoint (Synchronous)
@app.endpoint("/hello")
def hello(name: str):
    return f"Hello {name}!"

# 3. Define a Job Endpoint (Asynchronous with Queue)
# Perfect for long-running AI inference
@app.post("/process_image", queue_size=10)
def process_image(image: ImageFile):
    # APIPod handles the file upload/parsing automatically
    img_array = image.to_np_array()
    
    # ... run your AI model here ...
    
    return ImageFile().from_np_array(img_array)

# 4. Run the server
if __name__ == "__main__":
    app.start()
```

### 2. Run Locally
```bash
python main.py
```
Visit `http://localhost:8000/docs` to see your auto-generated Swagger UI.

## Features in Depth

### 🔄 Asynchronous Jobs & Polling
For long-running tasks (e.g., inference of a large model), you don't want to block the HTTP request. 
Use `@app.post` to automatically wrap your function in a job queue.

*   **Client:** Receives a `job_id` immediately.
*   **Server:** Processes the task in the background.
*   **[SDK](https://github.com/SocAIty/fastSDK):** Automatically polls for status and result.

```python
@app.post("/generate", queue_size=50)
def generate(job_progress: JobProgress, prompt: str):
    job_progress.set_status(0.1, "Initializing model...")
    # ... heavy computation ...
    job_progress.set_status(1.0, "Done!")
    return "Generation Complete"
```

### 📁 Smart File Handling
Forget about parsing `multipart/form-data`, `base64`, or `bytes`. APIPod integrates with **MediaToolkit** to handle files as objects.

```python
from apipod import AudioFile

@app.post("/transcribe")
def transcribe(audio: AudioFile):
    # Auto-converts URLs, bytes, or uploads to a usable object
    audio_data = audio.to_bytes()
    return {"transcription": "..."}
```

### ☁️ Serverless Routing
When deploying to serverless platforms like **RunPod**, standard web frameworks often fail because they lack the necessary routing logic for the platform's specific entry points. **APIPod** detects the environment and handles the routing automatically—no separate "handler" function required.

# Deployment
APIPod is designed to run anywhere by leveraging docker.
<p align="left">
  <a href="#Create & configure container">Build & configure</a> •
  <a href="#Deploy to socaity">Deploy</a> •
</p>

## Create & configure container

All you need to do is run:

```bash
apipod build 
```
This command creates the dockerfile and the server configuration file for you.

It will ask you a few questions like GPU support if not configured earlier for deploying the service.
Advanced users can also configure or write the docker and the yaml files theirsels.

## Deploy to socaity
Right after build you can deploy the service via the [socaity.ai](https://www.socaity.ai) dashboard.
This is the simplest option.

## Deploy to runpod.
1. You will need to build the your docker image.
2. Push your image to your dockerhub repository.
3.  Deploy on RunPod Serverless. 
    *   *APIPod acts as the handler, managing job inputs/outputs compatible with RunPod's API.*

Make sure that the environment variables set to the following: ```FTAPI_DEPLOYMENT="serverless"``` and  ```FTAPI_BACKEND="runpod"```


## Debugging APIPod serverless
You can configure your environment variables in order that APIPod acts as if it where deployed on socaity.ai or on runpod.
```bash
# Deployment Mode
ENV FTAPI_DEPLOYMENT="serverless" # Options: "localhost" (default), "serverless"

# Backend Provider
ENV FTAPI_BACKEND="runpod"      # Options: "fastapi" (default), "runpod"
```


# Client SDK

While you can use `curl` or `requests`, our [FastSDK](https://github.com/SocAIty/fastSDK) makes interacting with APIPod services feel like calling native Python functions.

```python
# The SDK handles authentication, file uploads, and result polling
# create a full working client stub 
create_sdk("https://localhost:8000", save_path="my_service.py")

# Import the client. It will have a method for each of your service endpoints including all parameters and its default values.
from my_service import awesome_client
mySDK = awesome_client()
mySDK.my_method(...)

# Blocks until the remote job is finished
result = task.get_result() 
```

# Comparison

| Feature | APIPod | FastAPI | Celery | Replicate/Cog |
| :--- | :---: | :---: | :---: | :---: |
| **Setup Difficulty** | ⭐ Easy | ⭐ Easy | ⭐⭐⭐ Hard | ⭐⭐ Medium |
| **Async/Job Queue** | ✅ Built-in | ❌ Manual | ✅ Native | ✅ Native |
| **Serverless Ready** | ✅ Native | ❌ Manual | ❌ No | ✅ Native |
| **File Handling** | ✅ Standardized | ⚠️ Manual | ❌ Manual | ❌ Manual |
| **Router Support** | ✅ | ✅ | ❌ | ❌ |

## Roadmap
- MCP protocol support
- Improve async support.

---
<p align="center">
  Made with ❤️ by <a href="https://socaity.com">SocAIty</a>
</p>
