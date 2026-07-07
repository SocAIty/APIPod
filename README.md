<h1 align="center">APIPod</h1>
<h3 align="center">Build, Deploy, and publish AI Services with Ease</h3>

<p align="center">
  <a href="https://www.socaity.ai">
    <img src="docs/example_images/APIPod.png" height="200" alt="APIPod Logo" />
  </a>
</p>

<p align="center">
  <b>APIPod</b> combines the developer experience of <b>FastAPI</b> with the power of <b>Serverless GPU</b> computing. <br/>
  Write your service like FastAPI. Run it anywhere with a single command. <br/>
  <b>Think Vercel — but for AI services.</b>
</p>

<p align="center">
  <a href="#why-apipod">Why APIPod</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#develop-test-and-simulate">Develop & Test</a> •
  <a href="#build--deploy">Build & Deploy</a>
</p>

---

## Why APIPod?

Building AI services is complex: file handling, long-running inference, job queues, deployment, scaling, and hosting provider choices all create friction at every step.

**APIPod** eliminates that friction. It abstracts away the AI infrastructure stack so you can focus on your model. You write the service; [Socaity](https://www.socaity.ai) handles the deployment and scaling across any cloud.

### Highlights

1. **Write once, run anywhere** — the same code runs in development, in serverless emulation, or on a real GPU cloud. Zero changes between environments.
2. **Drop-in FastAPI** — if you know FastAPI, you already know APIPod. Built on top of it, with batteries included for AI.
3. **Standardized I/O** — painless Images, Audio and Video via [media-toolkit](https://github.com/SocAIty/media-toolkit).
4. **OpenAI-compatible schemas** — built-in request/response schemas for chat, completions, embeddings, TTS, transcription, image/video generation. OpenAI clients work out of the box.
5. **Built-in job queue** — async jobs, polling and progress tracking, with no Celery/Redis/Kubernetes to wire up.
6. **One-command packaging** — `apipod build` generates your Dockerfile. No CUDA hell; APIPod picks compatible images.

## Installation

```bash
pip install apipod
```

## Quick Start

`APIPod` is a drop-in replacement for `FastAPI`. You get all of APIPod's capabilities with no migration cost.

```python
from apipod import APIPod, ImageFile

# Drop-in replacement for FastAPI
app = APIPod()

# A standard endpoint
@app.endpoint("/hello")
def hello(name: str):
    return f"Hello {name}!"

# Built-in media processing — uploads/URLs/base64 are parsed for you
@app.endpoint("/process-image")
def process_image(image: ImageFile):
    img_array = image.to_np_array()
    # ... run your AI model here ...
    return ImageFile().from_np_array(img_array)

if __name__ == "__main__":
    app.start()
```

Run it and open `http://localhost:8000/docs` for the auto-generated Swagger UI.

```bash
python main.py
# or
apipod start
```

## Smart File Handling

Forget about parsing `multipart/form-data`, `base64`, or `bytes`. APIPod integrates with **MediaToolkit** to handle files as objects. Whether the client sends a file upload, a URL, or a base64 string, your endpoint receives a ready-to-use object.

```python
from apipod import AudioFile

@app.post("/transcribe")
def transcribe(audio: AudioFile):
    # Auto-converts URLs, bytes, or uploads to a usable object
    audio_data = audio.to_bytes()
    return {"transcription": "..."}
```

## Model Loading Presets

Declare your weights; APIPod loads them at app start and the platform pre-stages them per provider (RunPod HF cache, image baking). Two built-in presets cover the transformers library:

```python
import apipod

llm = apipod.TransformersLLM("Qwen/Qwen2.5-7B-Instruct")      # chat LLM: generate / stream / embed_text
vlm = apipod.TransformersVLM("Qwen/Qwen3-VL-8B-Instruct")     # vision-language: image chat / stream / embed
```

Both pick the fastest attention backend on the machine (flash-attn 2 when installed on an Ampere+ GPU, PyTorch SDPA otherwise). Subclass `apipod.Model` for custom load logic.

## Serve a Model in One Call

`apipod.serve(model)` registers the standard OpenAI-compatible endpoints matching the model's methods, then starts the app. Model and service stay separate: the same instance works standalone (`model.generate(...)`) or served.

```python
import apipod

model = apipod.TransformersVLM("Qwen/Qwen3-VL-8B-Instruct")

if __name__ == "__main__":
    apipod.serve(model, title="Qwen3-VL", description="...")   # /chat (image+text) + /embeddings
```

Endpoint mapping: `generate`/`stream` -> `/chat`, `embed` or `embed_text` -> `/embeddings`, `generate_image` -> `/images`. Custom `apipod.Model` subclasses participate by implementing methods with those names. For custom routes, build an `APIPod` app yourself (or pass it via `serve(model, app=app)`).

## AI Services Streamlined (OpenAI-compatible)

APIPod provides built-in request/response schemas for common AI tasks (chat, TTS, image gen, etc.) that are fully OpenAI-compatible. This allows you to focus on the model logic while APIPod handles the boilerplate of validation, media parsing, and streaming.

```python
from apipod.common.schemas import ChatCompletionRequest

@app.endpoint("/chat")
def chat(request: ChatCompletionRequest):
    if request.stream:
        # Yield plain tokens — APIPod wraps them into ChatCompletionChunk SSE events.
        return my_llm.stream(request.messages)
    return my_llm.generate(request.messages) # auto-wrapped into ChatCompletionResponse
```

## Asynchronous Jobs & Scaling

For long-running tasks, APIPod provides a built-in job queue and progress reporting. When configured for serverless or with a queue, endpoints automatically return a `job_id` and run in the background.

```python
from apipod import JobProgress

@app.post("/generate", queue_size=50)
def generate(job_progress: JobProgress, prompt: str):
    job_progress.set_status(0.1, "Initializing model...")
    # ... heavy computation ...
    job_progress.set_status(1.0, "Done!")
    return "Generation Complete"
```

*   **Client:** Receives a `job_id` immediately.
*   **Server:** Processes the task in the background.
*   **SDK:** Automatically polls for status and result.

## Develop, Test, and Simulate

Just say *how you want to run the service right now*.

### Development (default)

Plain FastAPI. The fastest iteration loop.

```bash
apipod start
# or simply
python main.py
```

### Simulate a deployment

Before you ship, run your service exactly how it will behave in production — locally, with **no code changes**. `apipod simulate` takes an optional target string `{compute}-{provider}` (compute defaults to `serverless`).

```bash
apipod simulate                    # serverless emulation: FastAPI + local job queue
apipod simulate serverless         # same as above
apipod simulate dedicated          # plain FastAPI (dedicated compute)
apipod simulate serverless-runpod  # emulate Socaity routing requests to RunPod
apipod simulate dedicated-azure    # emulate a dedicated Azure deployment
```

If a provider has no serverless offering, APIPod warns and falls back to the job-queue emulation:

```bash
apipod simulate serverless-azure
# Warning: azure does not support serverless. Defaulting to FastAPI + Local Job Queue.
```

### Emulate a provider's native worker (`--native`)

By default Socaity is the orchestrator. `--native` skips Socaity and runs the provider's **own** serverless backend locally — e.g. RunPod's serverless worker (requires the `runpod` package):

```bash
apipod simulate serverless-runpod --native
```

### Configure from Python

The same intent can be set in code. Socaity **overrides** it with env vars once the service is actually managed by the platform, so what you test is what you ship.

```python
app = APIPod()                                            # development (plain FastAPI)
app = APIPod(simulate="serverless")                       # FastAPI + local job queue
app = APIPod(simulate="serverless-runpod", direct=True)   # RunPod native worker (local)
```

## Build & Deploy

### Build a container

```bash
apipod build
```

This scans your project, picks a compatible base image (CUDA/cuDNN, ffmpeg included) and generates a `Dockerfile`. For most users this is all you need; advanced users can edit or write their own Dockerfile.

Requirements: Docker installed, plus a CUDA/cuDNN setup if your model needs the GPU.

### Analyze and deploy

```bash
apipod analyze                # pre-deploy report: HF repo checks, catalog match, GPU recommendation
apipod deploy                 # analyze + create a deployment draft on Socaity
```

Both commands need a Socaity login (`socaity login`); everything else in APIPod works offline. `analyze` only prints a report. `deploy` runs the same analysis, resolves your declared models against the Socaity catalog, and creates a draft you finish in the [Socaity dashboard](https://www.socaity.ai) after pushing your container.

## Client SDK

Generate a typed client for your service using the [fastSDK](https://github.com/SocAIty/fastSDK). It handles authentication, file uploads, and automatic polling for background jobs.

```bash
fastsdk generate http://localhost:8009 -o myClient.py
```

```python
from myClient import myService

client = myService() 
client.text_to_speech("what a time to be alive")
```

## Comparison

| Feature | APIPod | FastAPI | Celery | Replicate/Cog |
| :--- | :---: | :---: | :---: | :---: |
| **Setup Difficulty** | Easy | Easy | Hard | Medium |
| **Async/Job Queue** | ✅ Built-in | ❌ Manual | ✅ Native | ✅ Native |
| **Serverless Ready** | ✅ Native | ❌ Manual | ❌ No | ✅ Native |
| **File Handling** | ✅ Standardized | ⚠️ Manual | ❌ Manual | ❌ Manual |
| **Router Support** | ✅ | ✅ | ❌ | ❌ |
| **Multi-cloud** | ✅ | ❌ | ❌ | ❌ |

## Roadmap

- One-command deploy execution (container push + provisioning) on top of the draft flow.
- MCP protocol support.

---

<p align="center">
  Made with ❤️ by <a href="https://www.socaity.ai?utm_source=github&utm_content=apipod-20-29-06-2026">SocAIty</a>
</p>