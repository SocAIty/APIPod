<h1 align="center">APIPod</h1>
<h3 align="center">Build, Deploy, and publish AI Services with Ease</h3>

<p align="center">
  <a href="https://www.socaity.com">
    <img src="docs/example_images/APIPod.png" height="200" alt="APIPod Logo" />
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

# 2. Use built-in media processing 
@app.endpoint("/process_image", queue_size=10)
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


### 🔄 Scaling services, Asynchronous Jobs, Polling and Job Progress 

Let's say you want to serve your service to many users or you have a long-running task.
Usually you need to set-up a load-balancer, kubernetes, brokers and a lot of other complicated stuff.
If you deploy to socaity / runpod this is taken care of for you. No Dev-Ops for you.

We allow you to emulate this behaviour for testing.

For long-running tasks (e.g., inference of a large model), you don't want to block the HTTP request. 
Often you want to be able to give a progress bar or updates about the current task to the user. This is what job progress is for.
It allows you to communicate a progress percentage and a status message to your user.



1. **Setup test environment for serverless (Job Queue)**:
   ```python
   # Initialize with serverless compute on localhost to enable the local job queue
   app = APIPod(compute="serverless", provider="localhost")
   ```

2. **Define Endpoint**:
   Use `@app.endpoint` (or `@app.post`). It automatically becomes a background task when a queue is configured.
   ```python
   @app.post("/generate", queue_size=50)
   def generate(job_progress: JobProgress, prompt: str):
       job_progress.set_status(0.1, "Initializing model...")
       # ... heavy computation ...
       job_progress.set_status(1.0, "Done!")
       return "Generation Complete"
   ```

   *   **Client:** Receives a `job_id` immediately.
   *   **Server:** Processes the task in the background.
   *   **[SDK](https://github.com/SocAIty/fastSDK):** Automatically polls for status and result.

3. **Opt-out**:
   If you want a standard synchronous endpoint even when queue is enabled:
   ```python
   @app.endpoint("/ping", use_queue=False)
   def ping():
       return "pong"
   ```


# Deployment
APIPod is designed to run anywhere by leveraging docker.
<p align="left">
  <a href="#Create & configure container">Build & configure</a> •
  <a href="#Deploy to socaity">Deploy</a>
</p>

## Create & configure container

All you need to do is run:

```bash
apipod build 
```
This command creates the dockerfile for you, and select the correct docker template and cuda/cudnn versions and comes with ffmpeg installed.

- For most users this already creates a sufficient solution. 
- However you are always free to create or customize the Dockerfile for your needs.

Requirements:
1. docker installed on your system.
2. Depending on your setup a cuda/cudnn installation


### APIPod Configuration
APIPod provides a flexible deployment configuration that allows developers to:
- Run services locally for development
- Deploy via the Socaity orchestration platform
- Deploy directly to cloud providers
- Choose between serverless or dedicated compute

The configuration is controlled through a combination of:
- orchestrator
- compute
- provider
- region
- CPU/GPU


| Orchestrator     | Compute      | Provider    | Resulting Backend                 |
| ---------------- | ------------ | ----------- | --------------------------------- |
| `socaity`        | `dedicated`  | `auto`      | FastAPI                           |
| `socaity`        | `dedicated`  | `localhost` | FastAPI + job queue *(test  mode)* |
| `socaity`        | `dedicated`  | `runpod`    | Celery backend *(planned)*        |
| `socaity`        | `dedicated`  | `scaleway`  | Celery backend *(planned)*        |
| `socaity`        | `dedicated`  | `azure`     | Celery backend *(planned)*        |
| `socaity`        | `serverless` | `auto`      | RunPod router backend             |
| `socaity`        | `serverless` | `localhost` | FastAPI + job queue *(test mode)* |
| `socaity`        | `serverless` | `runpod`    | RunPod router backend             |
| `socaity`        | `serverless` | `scaleway`  | ❌ Not supported                   |
| `socaity`        | `serverless` | `azure`     | ❌ Not supported                   |
| `local` / `None` | `dedicated`  | `localhost` | FastAPI                           |
| `local` / `None` | `dedicated`  | `runpod`    | FastAPI                           |
| `local` / `None` | `dedicated`  | `scaleway`  | FastAPI                           |
| `local` / `None` | `dedicated`  | `azure`     | FastAPI                           |
| `local` / `None` | `serverless` | `localhost` | FastAPI + job queue               |
| `local` / `None` | `serverless` | `runpod`    | RunPod router backend             |
| `local` / `None` | `serverless` | `scaleway`  | ❌ Not supported                   |
| `local` / `None` | `serverless` | `azure`     | ❌ Not supported                   |
| `local` / `None` | `localhost`  | `localhost` | FastAPI                           |




### 🔄 Queue Backend Support

APIPod supports multiple job queue backends to handle different deployment scenarios and scaling needs.

#### Available Backends

- **None** (default): Standard FastAPI behavior. No background jobs.
  
- **Local Queue** (`local`): In-memory job queue using threading.
  - Perfect for local development and single-instance deployments
  - No external dependencies required

#### Configuration

   ```python
   # Job queues are automatically enabled based on your configuration.
   # For example, serverless + localhost enables a local job queue for testing:
   app = APIPod(compute="serverless", provider="localhost")

   # Or via environment variables
   import os
   os.environ["APIPOD_COMPUTE"] = "serverless"
   os.environ["APIPOD_PROVIDER"] = "localhost"

   app = APIPod()  # Uses environment config
   ```

### Troubleshooting

You are always free to create or edit the Dockerfile for your needs.
Depending on your OS, your machine or your project setup you might occur one of those issues:
- Build scripts fails
- You can't build the docker container.

In this cases don't  
Advanced users can also configure or write the docker file for themselves

## Deploy to socaity
Right after build you can deploy the service via the [socaity.ai](https://www.socaity.ai) dashboard.
This is the simplest option.

## Deploy to runpod.
1. You will need to build the your docker image.
2. Push your image to your dockerhub repository.
3. Deploy on RunPod Serverless by using the runpod dashboard. 
    *   *APIPod acts as the handler, managing job inputs/outputs compatible with RunPod's API.*

Make sure that the environment variables are set to the following: ```APIPOD_COMPUTE="serverless"``` and ```APIPOD_PROVIDER="runpod"```


## Debugging APIPod serverless
You can configure your environment variables so that APIPod acts as if it were deployed on socaity.ai or on runpod.
```bash
# Orchestrator
ENV APIPOD_ORCHESTRATOR="local"    # Options: "local" (default), "socaity"

# Compute type
ENV APIPOD_COMPUTE="serverless"    # Options: "dedicated" (default), "serverless"

# Infrastructure provider
ENV APIPOD_PROVIDER="runpod"       # Options: "localhost" (default), "auto", "runpod", "scaleway", "azure"
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
- MCP protocol support.
- OpenAI-compatible default endpoints for LLMs
- Improve async support.

---
<p align="center">
  Made with ❤️ by <a href="https://socaity.com">SocAIty</a>
</p>
