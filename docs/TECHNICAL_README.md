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
├── api.py                  # APIPod() factory: resolves intent → backend instance
├── common/
│   ├── settings.py         # Env-driven config (APIPOD_SIMULATE / _DIRECT / _COMPUTE / _PROVIDER, cert, host, port)
│   ├── constants.py        # Enums: COMPUTE, PROVIDER, SERVER_HEALTH
│   └── schemas/
│       ├── schemas.py      # Standardized request/response models (OpenAI-compatible shapes)
│       └── media_files.py  # FileModel + typed variants: pydantic mirrors of media-toolkit files
├── engine/
│   ├── base_backend.py     # Shared backend base (title, version, health)
│   ├── endpoint_config.py  # EndpointExecutionPlan + configurator (how to register an endpoint)
│   ├── backend/
│   │   ├── fastapi/        # SocaityFastAPIRouter + file handling, streaming mixin, exceptions
│   │   └── runpod/         # SocaityRunpodRouter (serverless, path-based)
│   ├── files/              # _BaseFileHandlingMixin + parse_schema_media_fields (inputs → MediaFile)
│   ├── jobs/               # BaseJob, JobProgress, JobResult (+ factory)
│   ├── queue/              # JobQueue, JobStore, _QueueMixin (enqueue instead of block)
│   ├── streaming/          # StreamStore port + LocalStreamStore + StreamProducer
│   └── signatures/         # Signature inspection policies (media params, Body vs Form, …)
└── deploy/                 # `apipod build`: Dockerfile generation, dependency/CUDA detection
```

## Core principles

1. **One decorator, many execution modes.** Developers only write `@app.endpoint(...)`. The router inspects the function and decides how to run it. They never choose a "mode" explicitly — the signature is the contract.
2. **Deployment is intent, not code.** The same service file runs as a plain FastAPI app, a queued FastAPI app, or a RunPod serverless worker. `APIPod()` is a factory that picks the backend from a single *intent* (`simulate` / `direct`) locally, or from the `APIPOD_COMPUTE` / `APIPOD_PROVIDER` env vars Socaity injects in a managed deployment (see [Orchestrator, compute, provider](#orchestrator-compute-provider-and-simulate)).
3. **Media files are objects, not bytes.** Endpoint authors annotate parameters with media-toolkit types (`ImageFile`, `AudioFile`, …) and receive parsed, ready-to-use objects — regardless of whether the client sent a multipart upload, a URL, or base64.
4. **Standardized, OpenAI-compatible schemas.** Common AI payloads (chat, completion, embeddings, image/video/audio/3D generation, vision) have canonical pydantic schemas whose wire format mirrors the OpenAI API. The shape is provider-agnostic: any model can serve them; clients written against OpenAI-compatible tooling work without translation.
5. **Long-running work returns a job, not a blocked connection.** With a queue configured, endpoints return a `JobResult` immediately; clients poll `/status/{job_id}` (fastSDK does this automatically) and can receive progress updates via `JobProgress`.

## Orchestrator, compute, provider, and simulate

Three concepts describe *where and how* a service runs. Historically developers had to set all three; they no longer do. Understanding them still helps when reading the code.

- **Orchestrator** — *who routes and queues requests*. There is exactly one: **Socaity**. It is the implicit orchestrator that fronts the service, distributes jobs, and handles scaling/auth. It is no longer a flag. `--native` is the one escape hatch: it *skips* Socaity to talk to a provider's native backend instead (e.g. RunPod's own serverless worker).
- **Compute** (`COMPUTE`) — *the shape of the machine*: `serverless` (scale-to-zero, job-queue semantics) or `dedicated` (an always-on box, plain request/response).
- **Provider** (`PROVIDER`) — *the cloud the compute lives on*: `runpod`, `azure`, `scaleway`, `localhost`, … Not every provider supports every compute (e.g. Azure has no serverless; APIPod warns and falls back to the job-queue emulation).

A developer never assembles this matrix by hand. They express a single **intent**:

- **Development** (`APIPod()` / `apipod start`) — plain FastAPI, the fastest loop.
- **Simulation** (`APIPod(simulate="{compute}-{provider}")` / `apipod simulate ...`) — emulate a deployment **locally**, no code changes. The target string collapses compute + provider (`serverless`, `serverless-runpod`, `dedicated-azure`); compute defaults to `serverless`. `direct=True` emulates the provider's native worker instead of the Socaity queue.
- **Managed deployment** — when the service runs on the platform, Socaity sets `SOCAITY_DEPLOYMENT_CERT` (SHA1 of a shared secret). When that cert verifies (`IS_MANAGED_DEPLOYMENT`), `simulate`/`direct` are ignored and the **real** backend is selected from the `APIPOD_COMPUTE` / `APIPOD_PROVIDER` env vars Socaity injects — so the serverless-RunPod path runs the *real* worker, not the emulator.

## Backend resolution

`APIPod()` in `api.py` is not a class — it is a factory. It resolves the intent (managed → `_resolve_managed`, otherwise → `_resolve_intent`) into a `(backend_class, use_job_queue, runpod_simulate)` triple and returns one of:

- **`SocaityFastAPIRouter`** — an `APIRouter` subclass bound to a `FastAPI` app. Used for development and dedicated compute (no queue), and for the serverless emulation (paired with an in-memory `JobQueue` + `LocalStreamStore` and a background worker thread).
- **`SocaityRunpodRouter`** — a path-based dispatcher for RunPod serverless. There is no HTTP layer: RunPod delivers a JSON job whose `input.path` selects the registered function; the router converts files, injects `JobProgress`, executes, and returns a serialized `JobResult` (or a generator for streaming). It can also synthesize an OpenAPI schema by replaying the FastAPI signature conversion, so fastSDK clients can be generated against serverless deployments too. Its `simulate` flag chooses between RunPod's local API emulator (`apipod simulate serverless-runpod --native`) and the real worker (managed deployment).

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

**Response wrapping** (`wrap_schema_response`): if the function already returns the response model, it passes through. Schema helpers also accept convenient raw results (a plain string for chat/completion/transcription, raw vectors for embeddings). A bare ``None`` or dict with ``None`` fields is coerced to the same empty shorthand (empty text or required-field defaults from the response model's pydantic JSON schema) before validation. Everything else shares one generic path: a uniform envelope (`created`, `model`) merged with the result dict and validated by pydantic — a returned media-toolkit file is lifted into the `data` list automatically. Response IDs are not generated by APIPod schemas; IDs belong to the platform `JobResult`/socaity.ai layer.

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
   On RunPod the same logic applies (`_as_native_stream`): the request is validated and media-parsed by `prepare_schema_call`, the response is wrapped by `wrap_schema_response`, and stream chunks are base64-encoded when binary because RunPod transports JSON.
3. **Job streaming (serverless emulation)** — when a queue is configured, a streaming endpoint does **not** stream on the request connection. The job is enqueued and the response returns a `JobResult` immediately (with a `stream` link). The worker (producer) then relays the chunks into a **stream store**, and the client (consumer) reads them from `GET /stream/{job_id}` as SSE. A client that prefers to wait can poll `GET /status/{job_id}` instead and receive the **full aggregated result** once the job finishes. This mirrors a real deployment, where producer (worker) and consumer (gateway) live in different processes.

### The stream store

The **stream store** is the pluggable backend that buffers a job's chunks between the worker and the gateway. Its key idea is to decouple *producing* a stream (sync, worker side) from *consuming* it (async SSE, gateway side), so the same endpoint code streams identically whether it runs on localhost or on a real platform.

- **`StreamStore`** (`engine/streaming/stream_store.py`) is the port (abstract base class). Producer methods (`open_stream`, `write_chunk`, `close_stream`) are synchronous because the worker writes from a sync context; the consumer method `read_chunks` is an async generator that yields straight into a `StreamingResponse`. `delete_stream` / `stream_exists` cover lifecycle.
- **`LocalStreamStore`** (`engine/streaming/local_stream_store.py`) is the default in-memory implementation APIPod uses on localhost — thread-safe, no external dependencies. It exists purely to *emulate* deployment behavior; a real platform (e.g. Socaity) injects its own implementation (such as a Redis Streams store) via the `stream_store` constructor argument, without changing any endpoint code.
- **`StreamProducer`** (`engine/streaming/stream_producer.py`) is the bridge the router hands to the worker: it carries the raw chunk iterator, how to serialize each chunk for the store (ChatCompletion deltas, base64-framed media bytes, or plain tokens), the closing chunks (`finish` + `[DONE]`), and how to aggregate the raw chunks into the full `/status` result.

`APIPod()` wires a `LocalStreamStore` automatically whenever a job queue is in use (serverless-localhost and dedicated-with-queue). In plain FastAPI mode (no queue) there is no stream store: streaming happens directly on the request connection (path 1/2 above).

### Deployment

`apipod build` (see `deploy/`) scans the project (entrypoint, dependencies, CUDA requirements) and generates a Dockerfile from compatible templates. The resulting container runs unchanged on dedicated hosts, on socaity.ai, or on RunPod serverless — only the env vars differ: Socaity injects `SOCAITY_DEPLOYMENT_CERT` plus `APIPOD_COMPUTE` / `APIPOD_PROVIDER` to select the real backend, while locally you drive the same paths with `APIPOD_SIMULATE` / `APIPOD_NATIVE` (set for you by `apipod simulate`).

## Request lifecycle, end to end

A client calls `POST /tts` with a JSON body. FastAPI validates it against the rewritten signature and calls the outermost wrapper. The file-handling layer converts any media inputs into media-toolkit objects. If the endpoint is queued, the queue mixin stores the job and returns `{job_id, status: "queued", links}` — the worker thread later executes the real function, feeding `JobProgress` updates into the store. The client (typically fastSDK) polls `/status/{job_id}` until `finished` and receives the result, with any returned `AudioFile` serialized as a `FileModel`. Without a queue the same conversion happens inline and the response returns directly. On RunPod, the identical user function is reached through `handler → _router(path) → file handling → execute`, proving the core principle: the function is written once, the backend decides how it runs.

## Testing and CI

The suite (`test/`) is built so endpoint definitions live apart from test logic, and one helper boots them under any run intent.

```
test/
├── conftest.py          # build_service (in-process TestClient), live_service (real subprocess via CLI), config matrix, file paths
├── services/            # reusable services: register(app) callbacks + a runnable entrypoint
│   ├── core_service.py    # scalars, custom model, mixed media, JobProgress, file upload
│   ├── schema_service.py  # one endpoint per standardized schema, an extended schema, raw/typed mapping CASES
│   └── exec_service.py    # runnable service for fastSDK + streaming (predict, echo_image, chat, text, video)
├── files/               # media assets for upload/download tests
├── test_config.py       # intent -> backend resolution + /openapi.json loads for every backend
├── test_core.py         # core endpoint plumbing (types, model, files, queue lifecycle)
├── test_schemas.py      # standardized schema endpoints + response-model normalization
├── test_cli.py          # scan / build / simulate produce the right artifacts
└── test_execution.py    # fastSDK end-to-end (subprocess) + SSE streaming (in-process)
```

Run `pytest`. Two primitives keep tests DRY: `build_service(register, **config)` yields a `TestClient` for an app built from a `register(app)` callback under any `APIPod(**config)` intent; `live_service(simulate=...)` boots a service as a real subprocess through `apipod start` / `simulate` and yields its URL. Backends are parametrized from `FASTAPI_CONFIGS`, so a single test asserts behaviour across development, serverless, dedicated and runpod.

The fastSDK end-to-end tests skip automatically when the installed fastsdk lacks the `connect` API (e.g. a local in-progress build); CI installs one that has it. Pytest config (`pythonpath` in `pyproject.toml`) resolves `import apipod` to the in-repo source and the `conftest`/`services` helpers by name, so no `sys.path` shims are needed and each file is also runnable directly from an IDE via its `__main__` block.

CI (`.github/workflows/publish.yml`) runs the `test` job on every push and pull request (installing `.[test]`, which includes fastSDK). The `publish` job `needs: test`, so the patch version bump and PyPI upload happen only on a push to `main` after tests pass. flake8 and mypy run as soft checks (`continue-on-error`): they surface warnings without gating the build.

