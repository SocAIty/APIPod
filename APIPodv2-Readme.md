# APIPod

Run a model. Ship an API. Deploy both.

APIPod is a Python server framework and deployment tool for GPU workloads, AI models, and ordinary APIs. It combines FastAPI-style development with standardized media handling, background jobs, streaming, container packaging, and managed deployment.

Use APIPod in three ways:

1. Run a model locally from its model ID.
2. Deploy a model as a managed service without writing server code or a Dockerfile.
3. Build your own API, with or without a model, and deploy it as a custom service.

Socaity hosts both managed models and custom services. Managed deployments currently run on RunPod. APIPod-built containers remain normal Docker images, so you can also publish and operate them yourself.

## Install

```bash
pip install apipod
```

## Run a model locally

Start a model from Hugging Face with one command:

```bash
apipod start Qwen/Qwen3.8-27B-FP8
```

APIPod resolves model capabilities, selects a compatible local engine, loads the weights, and exposes standard endpoints such as `/chat`, `/embeddings`, or `/images`.

The same flow is available from Python:

```python
from apipod import serve

serve("Qwen/Qwen3.8-27B-FP8")
```

APIPod prefers vLLM for supported production chat and vision-language models. It uses Transformers or another compatible runtime when model capabilities require it. You can select an engine explicitly, but normal usage does not need an engine choice.

## Deploy a managed model

Deploy the same model through Socaity:

```bash
apipod deploy Qwen/Qwen3.8-27B-FP8
```

No project, Dockerfile, or local image build is required. Socaity:

- resolves the model artifact and capabilities;
- selects a maintained runtime image and compatible engine;
- applies required model recipe settings;
- recommends suitable RunPod hardware;
- pre-stages model weights;
- provisions the service and publishes its standard API.

APIPod does not replace model or engine defaults without a reason. Engine defaults remain authoritative. A runtime plan adds only settings required for compatibility, selected capabilities, target hardware, or explicit user configuration.

Managed runtimes use a small maintained image family:

- vLLM for supported chat and vision-language models;
- Transformers for compatible lightweight or Transformers-only workloads;
- Diffusers and task-specific images for pipelines such as image generation.

Images contain APIPod, the serving engine, and the standard entrypoint. Model weights load at runtime, so Qwen and Llama do not need separate images. A Diffusers model such as FLUX uses the Diffusers family rather than the vLLM chat image.

## Write your own API

APIPod keeps the FastAPI development model while adding deployment-aware I/O and execution:

```python
from apipod import APIPod, ImageFile

app = APIPod()


@app.endpoint("/hello")
def hello(name: str) -> str:
    return f"Hello {name}"


@app.endpoint("/inspect")
def inspect(image: ImageFile) -> dict:
    height, width = image.to_np_array().shape[:2]
    return {"width": width, "height": height}


if __name__ == "__main__":
    app.start()
```

Run it locally:

```bash
apipod start
```

APIPod handles OpenAPI generation, uploads, URLs, base64 media, streaming responses, job queues, progress reporting, and deployment routing. Your endpoint receives typed objects and stays independent from the execution target.

## Extend a model with your own endpoints

Use one loaded model from both standard and custom endpoints:

```python
from apipod import APIPod, Chat, serve

app = APIPod()
model = Chat("Qwen/Qwen3.8-27B-FP8")


@app.endpoint("/summarize")
def summarize(text: str):
    return model.generate([{"role": "user", "content": f"Summarize:\n{text}"}])


if __name__ == "__main__":
    serve(model, app=app)
```

`serve()` adds standard endpoints without replacing explicit application routes. The custom endpoint and `/chat` share the same model process and loaded weights.

## Customize model behavior

Override the public model behavior when preprocessing or postprocessing belongs to every use of the model:

```python
from apipod import Chat


class ProjectChat(Chat):
    def generate(self, messages, **kwargs):
        transformed = normalize_messages(messages)
        result = super().generate(transformed, **kwargs)
        return normalize_result(result)
```

This is the correct layer for transformations around inference, such as input normalization or output cleanup. Engine-specific compatibility, chat templates, tokenizer behavior, tool parsers, and reasoning parsers belong to APIPod runtime adapters and model recipes instead of application-specific model subclasses.

## Replace a standard endpoint

Override an endpoint when customization changes the HTTP contract or request workflow:

```python
from apipod import APIPod, Chat, serve
from socaity_schemas.public.inference.language import ChatCompletionRequest

app = APIPod()
model = Chat("Qwen/Qwen3.8-27B-FP8")


@app.endpoint("/chat", override=True)
def chat(request: ChatCompletionRequest):
    messages = apply_project_policy(request.messages)
    max_tokens = request.max_completion_tokens or request.max_tokens
    return model.generate(messages, max_tokens=max_tokens)


if __name__ == "__main__":
    serve(model, app=app)
```

Explicit application endpoints take precedence over generated model endpoints. Once custom Python code is involved, deployment follows the custom service path.

## Deploy a custom service

From the project directory:

```bash
apipod deploy
```

APIPod scans the service, creates its deployment configuration and Dockerfile, and builds the image on your machine. Socaity uploads that image to its Harbor registry and provisions it on RunPod.

The local build is intentional. Your image contains your source code, dependencies, system packages, and any custom runtime behavior. The platform does not rebuild or reinterpret it.

## Deploy to your own RunPod account

APIPod can package a service without managing its hosting:

```bash
apipod scan
apipod build
```

Push the generated image to your container registry, then configure it in RunPod. You own the registry, RunPod account, scaling, credentials, and lifecycle. APIPod still provides the server contract, generated Dockerfile, media handling, and RunPod-compatible runtime.

## Runtime behavior

APIPod separates three facts:

- **Artifact:** concrete weights, revision, format, quantization, and download size.
- **Capabilities:** tasks, modalities, context window, tool calling, structured output, and standard API surface.
- **Runtime plan:** engine, maintained image, required dependencies, hardware, environment, and engine arguments for one deployment.

This separation keeps one model usable across local development, managed serving, and custom services without embedding deployment policy in model code.

Request options such as `max_tokens` stay request options. APIPod does not invent completion limits, context windows, batch sizes, or concurrency settings. When omitted, the selected engine and model configuration decide.

## Standard APIs

APIPod exposes model capabilities through stable, OpenAI-compatible contracts:

- `/chat` for text and multimodal chat;
- `/embeddings` for text or multimodal embeddings;
- `/images` for image generation;
- task-specific audio, video, and 3D endpoints;
- `/health`, job status, cancellation, and streaming endpoints where applicable.

Use OpenAI-compatible clients or generate a typed client with [fastSDK](https://github.com/SocAIty/fastSDK). fastSDK handles authentication, media transfer, streaming, and polling for background jobs.

## One model, three paths

```text
apipod start MODEL    Local model server, no deployment
apipod deploy MODEL   Managed model, platform runtime and image
apipod deploy         Custom service, locally built user image
```

Choose based on ownership:

- Model ID only means APIPod and Socaity own the standard serving stack.
- Python code means you own service behavior and APIPod packages it.
- A locally built image can be hosted by Socaity or deployed through your own RunPod account.
