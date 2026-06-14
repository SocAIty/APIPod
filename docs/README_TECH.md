# APIPod — Technical Guide

This document explains how APIPod works internally: its architecture, the request lifecycle, and the design principles behind it. It is aimed at developers maintaining or extending the package. For usage-oriented documentation see the [main README](../README.md).

## What APIPod is

APIPod is the standardized way to build AI services for the [socaity.ai](https://www.socaity.ai) catalog. It wraps FastAPI with the batteries an AI service needs — media file handling, job queues with progress reporting, serverless routing, standardized request/response schemas — while keeping the developer experience of plain FastAPI: you write a function, annotate its parameters, and decorate it with `@app.endpoint`.

It sits in an ecosystem of three packages:

| Package | Role |
| --- | --- |
| **APIPod** (this repo) | Server side: define and deploy AI service endpoints |
| [media-toolkit](https://github.com/SocAIty/media-toolkit) | Shared media types (`ImageFile`, `AudioFile`, `VideoFile`, …) used for I/O on both ends |
| [fastSDK](https://github.com/SocAIty/fastSDK) | Client side: generated SDKs that handle uploads, polling and streaming |

## Package layout

```
apipod/
├── api.py                  # APIPod() factory: resolves config → backend instance
├── common/
│   ├── settings.py         # Env-driven config (APIPOD_ORCHESTRATOR / _COMPUTE / _PROVIDER, host, port)
│   ├── constants.py        # Enums: ORCHESTRATOR, COMPUTE, PROVIDER, SERVER_HEALTH
│   └── schemas/
│       ├── schemas.py      # Standardized request/response models (OpenAI-compatible shapes)
│       └── media_files.py  # FileModel + typed variants: pydantic mirrors of media-toolkit files
├── engine/
│   ├── base_backend.py     # Shared backend base (title, version, health)
│   ├── endpoint_config.py  # EndpointExecutionPlan + configurator (how to register an endpoint)
│   ├── backend/
│   │   ├── fastapi/        # SocaityFastAPIRouter + file handling, exceptions
│   │   └── runpod/         # SocaityRunpodRouter (serverless, path-based)
│   ├── files/              # _BaseFileHandlingMixin + parse_schema_media_fields (inputs → MediaFile)
│   ├── jobs/               # BaseJob, JobProgress, JobResult (+ factory)
│   ├── queue/              # JobQueue, JobStore, _QueueMixin (enqueue instead of block)
│   └── signatures/         # Signature inspection policies (media params, Body vs Form, …)
└── deploy/                 # `apipod build`: Dockerfile generation, dependency/CUDA detection
```

## Core principles

1. **One decorator, many execution modes.** Developers only write `@app.endpoint(...)`. The router inspects the function and decides how to run it. They never choose a "mode" explicitly — the signature is the contract.
2. **Deployment is configuration, not code.** The same service file runs as a plain FastAPI app, a queued FastAPI app, or a RunPod serverless worker. `APIPod()` is a factory that picks the backend from `orchestrator` / `compute` / `provider` (constructor args or `APIPOD_*` env vars).
3. **Media files are objects, not bytes.** Endpoint authors annotate parameters with media-toolkit types (`ImageFile`, `AudioFile`, …) and receive parsed, ready-to-use objects — regardless of whether the client sent a multipart upload, a URL, or base64.
4. **Standardized, OpenAI-compatible schemas.** Common AI payloads (chat, completion, embeddings, image/video/audio/3D generation, vision) have canonical pydantic schemas whose wire format mirrors the OpenAI API. The shape is provider-agnostic: any model can serve them; clients written against OpenAI-compatible tooling work without translation.
5. **Long-running work returns a job, not a blocked connection.** With a queue configured, endpoints return a `JobResult` immediately; clients poll `/status/{job_id}` (fastSDK does this automatically) and can receive progress updates via `JobProgress`.

## Backend resolution

`APIPod()` in `api.py` is not a class — it is a factory. It resolves the `(orchestrator, compute, provider)` triple (explicit args → env vars → defaults) against a support matrix and returns one of:

- **`SocaityFastAPIRouter`** — an `APIRouter` subclass bound to a `FastAPI` app. Used for dedicated compute and for local serverless emulation (then paired with an in-memory `JobQueue` plus a background worker thread started via the app lifespan).
- **`SocaityRunpodRouter`** — a path-based dispatcher for RunPod serverless. There is no HTTP layer: RunPod delivers a JSON job whose `input.path` selects the registered function; the router converts files, injects `JobProgress`, executes, and returns a serialized `JobResult` (or a generator for streaming). It can also synthesize an OpenAPI schema by replaying the FastAPI signature conversion, so fastSDK clients can be generated against serverless deployments too.

The full configuration matrix is documented in the main README and in the `APIPod()` docstring.

## The endpoint pipeline (FastAPI backend)

When a function is decorated with `@app.endpoint(path)`, the router asks the `FastApiEndpointConfigurator` to build an immutable `EndpointExecutionPlan`. The plan records whether the signature contains a registered request schema, whether the function itself streams, and whether queueing is enabled.

The function then goes through the normal APIPod decorator pipeline:

1. **Schema binding detection** — a parameter (any name) annotated with a standardized request schema (e.g. `ChatCompletionRequest`) or a subclass becomes `plan.schema_binding`. This does not register a separate route. It only tells the downstream decorators to parse that JSON body into the schema object and wrap the final result into the registered response model.
2. **Streaming endpoint decorator** — for generator functions, output is bridged into a `StreamingResponse` with SSE-friendly headers.
3. **Task endpoint decorator** — when a job queue is configured (and `use_queue` is not `False`), the call enqueues a job and immediately returns a `JobResult`. Schema requests with `stream=true` are the one request-level exception: the task decorator executes them inline and streams the result instead of queueing.
4. **Standard endpoint decorator** — direct execution. It also handles non-queued schema endpoints: parse the request schema, execute the user function, wrap the result into the schema response model, and serialize non-schema media results through `JobResultFactory`.

In all cases the function then passes through the **file-handling preparation** (next section) before being handed to FastAPI's `api_route`.

### File handling: two layers

File support is split into a *signature* layer and a *runtime* layer.

- **Signature rewriting** (`engine/backend/fastapi/file_handling_mixin.py`): media-toolkit annotations in the user's signature are rewritten so FastAPI/OpenAPI understand them. `image: ImageFile` becomes `Union[LimitedUploadFile, ImageFileModel, str]` — the client may send a multipart upload, a `FileModel` JSON object (`{file_name, content_type, content}` where content is base64 or a URL), or a plain URL/base64 string. `MediaList[...]` maps to list variants. Upload size limits are enforced via a dynamically subclassed `LimitedUploadFile`. `JobProgress` parameters are stripped from the public signature (and a dummy is injected when no queue runs).
- **Runtime conversion** (`engine/files/base_file_mixin.py`): before the user function executes, every media-annotated argument is converted to the annotated media-toolkit type via `media_from_any` — whatever the client actually sent. The function body always receives real `MediaFile` objects. This layer is backend-agnostic and reused by the RunPod router.

On the way out, `JobResultFactory._serialize_result` converts returned `MediaFile`/`MediaList`/pydantic objects back into JSON-safe `FileModel` payloads.

### Standardized schemas

`common/schemas/schemas.py` defines request/response pairs for: chat completion, text completion, embeddings, image generation, video generation, transcription, speech (TTS), voice creation, voice conversion, 3D generation, vision, and multimodal embeddings. `model` is optional on every request, since an APIPod service typically serves exactly one model. The audio API is split per use case, mirroring OpenAI: `TranscriptionRequest` (STT), `SpeechRequest` (TTS, with `voice` as a named voice or a cloning reference file), `CreateVoiceRequest` (voice cloning → embedding) and `VoiceConversionRequest` (voice2voice).

`SCHEMA_REGISTRY` in `engine/schema_extension/schema_mixin.py` is the single source of truth: it maps each request schema to a `SchemaEndpointSpec` (response model, tag). Everything else derives from it — `engine/signatures/policies.py` builds `SUPPORTED_REQUEST_SCHEMAS` from the registry keys (schema-annotated parameters are read from the JSON body while plain parameters stay form-encoded), and both routers detect schema endpoints with `get_schema_binding`, which finds the schema-typed parameter by annotation regardless of its name (subclasses of registered schemas are also detected, so services can extend a schema with custom fields). Schema endpoints may not declare additional user parameters; put extra inputs on the schema or a schema subclass.

**Nested media files**: request schemas may declare `FileModel`-typed fields (`ImageGenerationRequest.image`, `TranscriptionRequest.audio`, …). Pydantic accepts uploads, FileModel JSON objects, URLs and plain base64 strings for those fields; before the endpoint function runs, `parse_schema_media_fields` (in `engine/files/base_file_mixin.py`) replaces them with parsed media-toolkit objects — the endpoint receives a ready-to-use `ImageFile`/`AudioFile`, exactly like method-level `def endpoint(image: ImageFile)` parameters.

**Response wrapping** (`wrap_schema_response`): if the function already returns the response model, it passes through. Schema helpers also accept convenient raw results (a plain string for chat/completion/transcription, raw vectors for embeddings). Everything else shares one generic path: a uniform envelope (`created`, `model`) merged with the result dict and validated by pydantic — a returned media-toolkit file is lifted into the `data` list automatically. Response IDs are not generated by APIPod schemas; IDs belong to the platform `JobResult`/socaity.ai layer.

### Jobs, queue and lifecycle

The local `JobQueue` (in-memory, threaded) drives the serverless emulation and dedicated-with-queue modes. A job passes through these stages:

- `validate_job_before_add` — parameter/permission validation
- `add_job` — persisted in the `JobStore`, returns immediately
- `create_job` / `process_job` — the worker picks it up and runs the user function
- `complete_job` — result stored, status set
- `remove_job` — cleaned up once collected (or orphaned)

`JobProgress` is the in-band progress channel: if the user function declares a `job_progress` parameter, the backend injects an implementation (`JobProgress` locally, `JobProgressRunpod` on RunPod) and `set_status(progress, message)` updates surface in `/status` polls. `JobResult` is the unified public envelope: `job_id`, `status` (`pending/processing/completed/failed`), `result`, `progress`, `message`, timing `metrics`, and hypermedia `links` (status/cancel/stream).

Standard routes registered automatically: `GET /status/{job_id}`, `POST /cancel/{job_id}`, `GET /health`, and — when a stream store is configured — `GET /stream/{job_id}` (SSE).

### Streaming

Three streaming paths exist today:

1. **Generator endpoints** — any endpoint function that is a (async) generator, or whose return annotation declares an iterator, is served as an SSE `StreamingResponse` (FastAPI) or as a native generator handed back to RunPod (`return_aggregate_stream`). Detection is annotation/inspect-based — no source-code heuristics.
2. **Schema endpoints with `stream=true`** — the request schema carries the `stream` flag (chat, completion, transcription, speech, video generation). Streaming bypasses the queue; what gets streamed depends on what the function returns:
   - a **generator of raw tokens**: for tags with a registered chunk model (`STREAM_CHUNK_SPECS`, currently `chat`), `SchemaStreamSerializer` wraps each token into the standardized chunk model (`ChatCompletionChunk`) as an SSE event — APIPod generates the stable chunk `id`, the `created` timestamp and the `object` discriminator, then closes with a final delta and the `[DONE]` sentinel. The endpoint only yields text. Other token-delta tags without a chunk model stream their tokens as-is (SSE), and non-SSE tags stream raw bytes;
   - an **`AudioFile`/`VideoFile`**: its encoded bytes are chunked into a `StreamingResponse` with the file's media content type — raw audio chunks, not SSE, matching OpenAI's `stream_format="audio"` behavior;
   - anything else: the regular wrapped JSON response (the endpoint cannot stream).
   On RunPod the same logic applies, but chunks pass through `_yield_native_stream`, which base64-encodes binary chunks because RunPod transports JSON.
3. **Job streaming** — `GET /stream/{job_id}` replays chunks from a pluggable stream store while a queued job is in `streaming` state (gateway integration).

### Deployment

`apipod build` (see `deploy/`) scans the project (entrypoint, dependencies, CUDA requirements) and generates a Dockerfile from compatible templates. The resulting container runs unchanged on dedicated hosts, on socaity.ai, or on RunPod serverless — only the `APIPOD_*` env vars differ.

## Request lifecycle, end to end

A client calls `POST /tts` with a JSON body. FastAPI validates it against the rewritten signature and calls the outermost wrapper. The file-handling layer converts any media inputs into media-toolkit objects. If the endpoint is queued, the queue mixin stores the job and returns `{job_id, status: "pending", links}` — the worker thread later executes the real function, feeding `JobProgress` updates into the store. The client (typically fastSDK) polls `/status/{job_id}` until `completed` and receives the result, with any returned `AudioFile` serialized as a `FileModel`. Without a queue the same conversion happens inline and the response returns directly. On RunPod, the identical user function is reached through `handler → _router(path) → file handling → execute`, proving the core principle: the function is written once, the backend decides how it runs.


