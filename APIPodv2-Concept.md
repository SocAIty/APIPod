# APIPod v2 concept

## Purpose

APIPod v2 provides one developer model for local model serving, managed model deployment, and custom API deployment.

The core distinction is ownership:

- **Local model:** APIPod resolves and runs a model on the developer's machine.
- **Managed model:** Socaity owns the standard runtime image and loads the selected model into it.
- **Custom service:** The developer owns Python behavior. APIPod builds the developer's image locally, then Socaity or the developer deploys it.

Managed hosting initially targets RunPod only. Provider abstractions remain in the architecture, but this project does not add another managed provider.

## Design principles

### Intent before infrastructure

Common operations express developer intent:

```text
apipod start MODEL    Run a model locally
apipod deploy MODEL   Deploy a standard managed model
apipod deploy         Deploy the current custom service
```

Python uses the same model:

```python
serve("Qwen/Qwen3.8-27B-FP8")
serve(model, app=app)
```

Developers choose a runtime only when the automatic choice is unsuitable.

### Defaults have one owner

APIPod must not copy engine defaults into wrappers. Missing values remain missing.

Resolution precedence is:

1. explicit user configuration;
2. compatibility requirements from a verified runtime recipe;
3. target constraints resolved by the platform;
4. native model and engine defaults.

Optional performance tuning does not become a framework default. In particular, APIPod does not invent `max_tokens`, `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`, or GPU memory utilization.

### Model is not engine

`Chat("org/model")` represents a capability. vLLM, Transformers, and Diffusers are runtime implementations.

Application code depends on public model methods and standard request schemas. Engine-specific loading, chat templates, parsers, and workarounds stay inside runtime adapters and recipes.

### Standard path stays standard

A managed model has a standard API surface and no user code. Adding Python preprocessing, postprocessing, or endpoints changes the deployment into a custom service. Model metadata and runtime recommendations can still be reused, but the user's image must be built.

## User flows

### Run a model locally

`apipod start MODEL` resolves the model, selects a locally available compatible engine, starts it, and registers endpoints from declared capabilities.

Local resolution prefers vLLM when it is installed and the model is supported. Otherwise APIPod selects a compatible adapter, commonly Transformers for chat or Diffusers for image generation. An explicit engine choice remains available.

No deployment draft, Socaity login, Harbor upload, or Docker build is involved.

### Customize model inference locally

A developer may wrap or override public model methods for behavior that applies to every caller:

- input and output normalization;
- project-specific preprocessing;
- result filtering or enrichment.

Engine compatibility behavior does not belong in project subclasses. Qwen chat-template construction, thinking controls, device placement, and vLLM parser selection belong to APIPod adapters or model recipes.

Any custom Python model behavior uses the custom service deployment path.

### Add or replace endpoints

`serve(model, app=app)` adds capability-derived endpoints to an existing application. Explicit application endpoints take precedence over generated standard endpoints. Generated registration skips a path already owned by the application. `@app.endpoint(path, override=True)` explicitly replaces a previously registered route. Duplicate paths without an explicit ownership rule fail during startup.

Additional endpoints share the same model instance:

```python
app = APIPod()
model = Chat("Qwen/Qwen3.8-27B-FP8")


@app.endpoint("/summarize")
def summarize(text: str):
    return model.generate([{"role": "user", "content": text}])


serve(model, app=app)
```

Changing `/chat` itself is appropriate when the HTTP contract or request workflow changes. Changing tokenization or model input construction belongs in the model/runtime layer.

### Deploy a managed model

`apipod deploy MODEL` sends model intent to Socaity. The backend resolves an immutable runtime plan from:

- concrete model artifact;
- model capabilities;
- maintained runtime recipe;
- available RunPod hardware and pricing;
- user configuration and credentials.

The platform selects a maintained image, injects the model reference and required settings, pre-stages weights, provisions RunPod, and exposes standard endpoints. No local image is built.

```text
resolve -> analyze -> confirm -> persist plan -> provision
```

The selected platform image is already approved and pinned by digest. Managed deployment bypasses user push credentials, Harbor staging, custom image validation, and promotion.

### Deploy a custom service through Socaity

`apipod deploy` scans the project and creates `apipod.json`. APIPod generates the Dockerfile and builds the image on the user's machine. The CLI pushes it to Socaity's Harbor registry, and the backend provisions it on RunPod.

Local image creation remains mandatory for this path. The image is the executable artifact containing user code, dependencies, and system packages.

A custom service may reference a known catalog model. The analyzer can reuse its artifact facts, capability metadata, and hardware recommendation, but it does not replace the user image with a managed image.

```text
scan -> analyze -> confirm -> local build -> staging push -> validate -> promote -> provision
```

Analysis runs before the expensive local build.

### Deploy a custom service directly to RunPod

`apipod build` produces the same RunPod-compatible service image without creating a Socaity deployment. The developer pushes it to a registry and configures RunPod directly.

APIPod owns packaging and runtime compatibility. The developer owns registry credentials, RunPod configuration, scaling, and operations.

## Target architecture

### C4 level 1: system context

```text
Developer
  |
  | APIPod Python API and CLI
  v
APIPod ----------------------------------------------------+
  |                                                       |
  | custom image                                          | model intent
  v                                                       v
Developer registry or Socaity Harbor             Socaity Platform
                                                          |
                                                          | resolved deployment
                                                          v
                                                     RunPod

External metadata:
  Hugging Face API and repository files
  vLLM Recipes
  public benchmark sources
```

APIPod provides local execution, service contracts, scanning, and custom image construction. Socaity owns managed metadata, runtime plan resolution, registry access, deployment state, billing, and RunPod lifecycle.

### C4 level 2: containers

#### APIPod package and CLI

Responsibilities:

- expose `APIPod`, `Model`, capability facades, and `serve()`;
- resolve local model execution;
- register standard and custom endpoints;
- provide media, streaming, queue, and RunPod worker behavior;
- scan and build custom services locally;
- submit model intent or a custom deployment artifact to Socaity.

APIPod does not own the global model catalog, RunPod pricing, managed image policy, or deployment history.

#### Socaity API

Responsibilities:

- authenticate users and resolve Hugging Face credentials;
- analyze deployment intent;
- resolve model, artifact, recipe, hardware, and pricing;
- create service, endpoint, deployment, and draft records;
- issue scoped Harbor upload credentials for custom images;
- provision and operate RunPod resources;
- expose the effective served capabilities to gateways and clients.

#### Metadata ingestion

Responsibilities:

- discover model identities and benchmark results;
- enrich model artifacts and capabilities from Hugging Face;
- synchronize structured vLLM recipes;
- preserve source, revision, freshness, and verification state;
- publish searchable managed-deployment readiness.

#### Managed runtime images

A small image family replaces per-model Dockerfiles:

- **vLLM runtime:** supported chat and vision-language services;
- **Transformers runtime:** lightweight, smoke-test, or Transformers-only services;
- **Diffusers runtime:** image and other diffusion pipelines;
- additional task images only when dependency or serving contracts require them.

Each image contains APIPod, its engine, and a standard entrypoint. Model weights and per-model configuration are supplied at runtime.

Qwen and Llama can share the vLLM image. FLUX normally uses the Diffusers image because image generation and vLLM chat are different runtime capabilities.

### C4 level 3: APIPod components

#### Model facade

Public capability types such as `Chat` present stable `generate`, `stream`, and related methods. They delegate to an engine adapter selected by the resolver.

#### Model resolver

The local resolver reads artifact metadata, capabilities, available packages, and explicit settings. It returns an adapter choice without modifying engine defaults.

#### Engine adapters

Adapters translate public model methods into vLLM HTTP, Transformers calls, Diffusers pipelines, or future engines. They own engine-specific request translation and lifecycle.

#### Endpoint registrar

The registrar maps capabilities to standard schemas:

- chat or multimodal chat to `/chat`;
- embeddings to `/embeddings`;
- image generation to `/images`;
- other tasks to their registered standard contracts.

It applies explicit route precedence and never silently replaces a user endpoint.

#### Custom service builder

The scanner discovers application entrypoints, models, includes, dependencies, and compute intent. The builder creates a local Dockerfile and image. This component is not used by managed model deployment.

### C4 level 3: platform components

#### Catalog service

Stores model identity, artifact facts, intrinsic capabilities, licenses, provenance, scores, and service relationships.

#### Runtime recipe registry

Stores serving compatibility from vLLM Recipes and Socaity-maintained adapters. A recipe is reusable source data, not a deployment decision.

#### Runtime plan resolver

Combines artifact, capabilities, recipe, explicit overrides, target constraints, available hardware, and pricing. It returns one validated plan or a clear unsupported result.

#### Deployment analyzer

Reports reachability, gating, managed readiness, selected engine and image, required credentials, hardware, disk, price, warnings, and explicit overrides before draft creation.

Deployment requests use an explicit discriminated intent:

```text
ManagedModelIntent  = model reference + optional explicit overrides
CustomServiceIntent = apipod.json + locally built image handoff
```

The backend must not infer managed versus custom deployment from missing fields.

#### Hosting manager and RunPod adapter

Persist deployment state and translate a resolved plan into RunPod resources. RunPod-specific API objects stay behind this adapter.

## Domain and persistence model

The three model concerns must remain independent.

### Artifact

An artifact is one concrete loadable checkpoint:

- source and repository ID;
- immutable revision or commit;
- weight format and quantization;
- parameter and active-parameter counts;
- weight size and gating;
- tokenizer and processor references;
- base model and variant relationship.

FP8, BF16, AWQ, and other checkpoints are distinct artifacts even when they implement the same logical model.

### Capabilities

Capabilities describe what the model can do independently from one deployment:

- input and output modalities;
- tasks and standard endpoint contracts;
- native context window;
- maximum model output when defined;
- tool calling, reasoning, and structured output;
- optional context-extension support.

Unknown is not false. Current non-null Boolean defaults for `tool_calling` and `structured_output` conflate missing evidence with unsupported behavior. The target schema records unknown explicitly or stores evidence-backed capability records.

The native context window and served context window are different facts. Qwen3.8 has a native 262,144-token context and an optional extended mode near one million tokens. A deployment may expose less or more depending on its runtime plan and hardware.

### Runtime recipe

A recipe describes how an artifact can run with one engine. It contains:

- engine and minimum engine version;
- maintained image and dependencies;
- required arguments and environment;
- optional feature arguments, such as tool and reasoning parsers;
- supported hardware and minimum VRAM;
- variant mapping;
- verified hardware and verification time;
- context-extension instructions;
- source URL, source revision, and synchronization time.

Required compatibility arguments are separate from optional tuning. The resolver applies the first when needed. The second requires explicit policy or user intent.

### Runtime plan

A runtime plan is an immutable deployment snapshot:

- selected artifact and revision;
- selected recipe and source revision;
- runtime image by digest;
- effective engine version;
- RunPod hardware and worker configuration;
- model and cache mounts;
- required environment and arguments;
- explicit user overrides;
- standard endpoints and effective served capabilities;
- effective served context window.

Plans are persisted with deployments so later recipe changes cannot alter a running deployment implicitly.

Today no first-class runtime plan exists. Deploy intent is split across `deployment_drafts.apipod_config`, `deployment_drafts.analysis`, `service_models`, deployment container fields, and the live OpenAPI contract. Provisioning reads the draft opportunistically, so retry and reprovision behavior depends on pipeline state. V2 makes the deployment plan durable and uses the draft only for workflow state.

### Service contract

The service contract is the observed API exposed by a running deployment. It is materialized from OpenAPI after startup and stored on the deployment.

The contract is an output of deployment, not a runtime plan input. Runtime plan hashes, image digests, and specification hashes remain separate so changes to execution intent, executable image, and public API can be detected independently.

### Database evolution

Target entities:

```text
ai_models
  1 --- n model_artifacts
  1 --- 1 model_capabilities

model_artifacts
  1 --- n model_runtime_recipes

service_deployments
  1 --- 1 deployment_runtime_plans
```

Migration can remain incremental:

1. Keep existing artifact and capability columns on `ai_models` during phase 1.
2. Add `model_runtime_recipes` and `deployment_runtime_plans` first.
3. Introduce `model_artifacts` when variant mapping is consumed by deployment.
4. Move or project flattened fields through compatibility RPCs while callers migrate.

Suggested `model_runtime_recipes` fields:

```text
id, model_id, artifact_id, engine, target_provider
source, source_ref, source_revision, synced_at
min_engine_version, runtime_image, runtime_image_digest
required_args, required_env, dependencies
feature_args, supported_hardware, verified_hardware
min_vram_gb, native_context_window, context_extensions
status, validation_errors
```

Suggested `deployment_runtime_plans` fields:

```text
deployment_id, recipe_id, artifact_revision
plan_json, explicit_overrides, resolved_at
```

`plan_json` is a versioned immutable snapshot. Search and deployment logic should use typed schema models rather than unstructured dictionaries.

## Metadata sources and synchronization

### Existing coverage

The current cron pipeline primarily discovers catalog entries and benchmark scores:

- Artificial Analysis, LM Arena, and Open LLM Leaderboard supply scores, tasks, naming hints, and organizations.
- Hugging Face card enrichment supplies family, license, modalities, parameter count, file size, format, and gating.
- Deployment analysis checks repository reachability and falls back to VRAM estimation from model names.

`context_window` exists in current schemas, but no current source populates it. Hugging Face enrichment reads the card summary, not `config.json`. It therefore misses context, architecture, quantization details, and several serving capabilities.

### Hugging Face ingestion

For each managed candidate, fetch and version:

- model card API payload;
- `config.json`;
- `tokenizer_config.json`;
- `generation_config.json`;
- repository siblings and safetensors metadata.

These sources provide artifact structure and capability evidence, including nested VLM text configuration. Context extraction must inspect the root plus `text_config` and `llm_config`, including valid RoPE scaling metadata.

HF data does not prove vLLM compatibility, required parser names, suitable images, or verified hardware.

### vLLM Recipes ingestion

Synchronize structured YAML from:

```text
vllm-project/recipes/models/{org}/{repo}.yaml
```

Recipes provide model and variant IDs, minimum vLLM versions, images, dependencies, context length, required arguments, feature parsers, hardware support, and VRAM measurements.

The sync job must:

1. fetch a pinned Git revision;
2. validate YAML against the upstream schema;
3. resolve recipe model and variant IDs to artifacts;
4. normalize required settings separately from optional features;
5. upsert recipes with source revision and freshness;
6. mark removed or invalid recipes unavailable without deleting deployment history.

There is no general vLLM public API that returns a complete deployment plan for every model. The recipes repository is the structured compatibility source. Hugging Face remains the artifact source. Socaity verification provides final operational confidence.

### Source precedence

For conflicting facts:

1. explicit Socaity verification;
2. manually curated correction with provenance;
3. pinned structured runtime recipe;
4. Hugging Face repository configuration;
5. model card metadata;
6. name-based inference.

Name-based estimates remain warnings or last-resort analysis. They must not mark a model managed-ready.

### Search

Typesense model documents should add projections for:

- managed-ready status;
- supported engines;
- available artifact precisions;
- native context window;
- input and output modalities;
- minimum verified hardware;
- recipe freshness.

Search stores projections only. PostgreSQL remains authoritative.

## Runtime plan resolution

### Managed resolution algorithm

1. Resolve exact model or artifact ID. Do not fuzzy-select a checkpoint for deployment.
2. Verify repository access and required user credentials.
3. Load capabilities and compatible active recipes.
4. Select engine from task support and recipe evidence.
5. Select the maintained image family and compatible version.
6. Resolve required feature parsers from the endpoints being exposed.
7. Match minimum VRAM and topology against active RunPod pricing rules.
8. Apply explicit user overrides and validate them.
9. Leave all other settings absent so engine defaults apply.
10. Persist and return the immutable plan.

If no verified compatible recipe exists, the backend returns `unsupported_managed`. It may recommend local or custom deployment. It must not silently select an unverified engine.

### Image policy

Runtime images are versioned platform artifacts and referenced by digest in resolved plans. Tags may be used for discovery, but deployments pin digests.

Image updates create new plans. Existing deployments do not change until redeployed.

### Configuration classes

Runtime configuration separates:

- **required:** model cannot serve correctly without it;
- **feature:** required only when exposing a capability, such as tool calling;
- **target-derived:** topology, mounts, ports, and credentials;
- **tuning:** throughput or latency choices;
- **user override:** explicit departure from normal resolution.

Only the first three are automatic.

## Context-window and defaults correction

### Current failure sources

The current stack creates values at several layers:

- `apipod/serve.py` falls back to `max_tokens=512`;
- `Chat`, `VLLMChat`, `TransformersLLM`, and `TransformersVLM` declare `max_tokens=512` and forward it;
- `TransformersLLM.embed_text` invents `8192` when `max_position_embeddings` is absent on a wrapper config;
- `APIPOD_VLLM_MAX_NUM_SEQS` defaults to `256` and is always passed;
- `VLLMChat.load()` derives and always passes `--max-model-len`;
- the Qwen deployment Dockerfile passes `--max-num-batched-tokens 8192`.

`max_num_batched_tokens=8192` is a batching budget, not a model context window. It is still an unwanted framework tuning default and makes diagnosis harder.

### Target behavior

- Request `max_tokens` fields are optional.
- APIPod omits `max_tokens` when the client omits it.
- Transformers omits `max_new_tokens` and uses model generation configuration.
- vLLM requests omit `max_tokens` and use vLLM behavior.
- APIPod does not pass `--max-model-len` unless the user or resolved plan explicitly requests it.
- APIPod does not pass `--max-num-seqs` or `--max-num-batched-tokens` by default.
- Context detection remains available for metadata, diagnostics, validation, and documentation.
- Nested VLM configuration is parsed, but parsing a value does not make it a command-line override.

For Qwen3.8, native context metadata resolves from `text_config.max_position_embeddings` to 262,144. Extending toward one million tokens is an explicit runtime mode because it requires a larger `max_model_len`, nested HF overrides, and sufficient KV-cache memory. It must not be the managed default.

### Agent summarization

Agent runtimes must use the effective served context window from the deployment or model service, not a static catalog claim. Summarization should trigger below that limit with reserved capacity for:

- requested output;
- system prompt;
- tool schemas;
- pending tool results;
- tokenizer estimation error.

APIPod exposes effective served limits. SPAINE or another agent runtime owns summarization policy. Summarization is a safety layer, not a substitute for correct serving configuration.

## Transformation plan

### Phase 1: correct defaults and remove model-specific engine leakage

#### APIPod

- Change all model generation signatures to `max_tokens: Optional[int] = None`.
- Forward generation and request arguments only when supplied.
- Remove the `512` fallback in `_chat_kwargs`.
- Remove the invented `8192` embedding fallback. Resolve a trustworthy tokenizer/model limit or leave it unset.
- Make every `APIPOD_VLLM_*` value an explicit override. Remove the default `MAX_NUM_SEQS=256`.
- Keep `max_model_len_from_config()` for metadata and diagnostics.
- Pass `--max-model-len` only when `APIPOD_VLLM_MAX_MODEL_LEN` is explicitly set.
- Remove `--max-num-batched-tokens 8192` from the Qwen Dockerfile.
- Log effective vLLM configuration without claiming a detected capability is the served override.
- Remove or implement the currently parsed but unused `apipod deploy TARGET` value. RunPod remains the only valid managed target.

#### qwen-models

- Remove `QwenVLM(TransformersVLM)` and `QwenLLM(TransformersLLM)`.
- Move generic CUDA placement into the Transformers adapter.
- Move Qwen chat-template, thinking, tool parser, and reasoning parser behavior into APIPod adapter configuration or recipes.
- Reduce Qwen, Gemma, and GLM service definitions to model lists plus catalog text until the files are replaced by managed catalog entries.
- Keep `ZImage` custom until APIPod has a generic Diffusers image-generation preset. Its pipeline loading and sampling behavior are executable logic, not model metadata.
- Scan remaining model repositories for engine subclasses that encode reusable compatibility rather than application behavior.

#### Verification

- Start Qwen3.8 with no APIPod context (locally no vLLM).
- Deploy the Qwen service privately through the current custom deployment path. Use the already deployed test-backend not prod for deployment.
- Call it through socaity sdk with more than 8,192 input tokens and no framework-supplied completion limit. Reuse/refactor existing test in socaity sdk for it.
- Verify explicit `max_tokens` still works.
- Verify the native context limit and a deliberate overflow produce accurate errors.

### Phase 2: add runtime recipe data and backend planning

#### Schemas and SQL

- Add typed `RuntimeRecipe` and `RuntimePlan` models to `socaity-schemas`.
- Add SQL migrations for `model_runtime_recipes` and `deployment_runtime_plans`.
- Add artifact identity or mapping required for recipe variants.
- Add source revision, freshness, status, and validation fields.
- Update catalog RPCs and repository mapping.

#### Ingestion

- Extend Hugging Face enrichment with repository configuration files.
- Populate native context, architecture, quantization, processor, and evidence-backed capabilities.
- Add a vLLM Recipes synchronization job.
- Normalize recipe variants and link them to exact HF artifacts.
- Expand Typesense projections and filters.

#### Backend

- Introduce a runtime recipe repository and runtime plan resolver.
- Extend `DeploymentAnalyzer` to report managed readiness and the proposed plan.
- Replace filename-based VRAM estimates with recipe measurements when available.
- Persist the resolved plan when a deployment draft is created.
- Extend hosting to use image, env, args, artifact revision, and hardware from the plan.
- Split managed provisioning from the custom staging, validation, and promotion pipeline.
- Replace the current single `HF_MODEL` pre-stage value with the artifact list from the runtime plan.
- Keep RunPod as the only target adapter.

### Phase 3: deliver end-to-end managed and local model intent

#### CLI and Python API

- Support `serve("org/model")`.
- Support `apipod start org/model`.
- Support `apipod deploy org/model` without scanning or building locally.
- Preserve `apipod deploy` for custom projects with mandatory local image build.
- Present the resolved engine, image, hardware, context, and explicit overrides before confirmation.
- Resolve the current `start` positional argument as an existing entrypoint path first and as a model reference otherwise.

#### Managed runtime images

- Publish versioned vLLM, Transformers, and Diffusers image families.
- Add a standard entrypoint that reads a validated runtime plan.
- Pre-stage Hugging Face weights through RunPod configuration.
- Pin production deployments to image and artifact revisions.

#### Endpoint composition

- Add deterministic generated-route precedence.
- Add explicit endpoint replacement.
- Ensure one model instance is shared across generated and custom endpoints.
- Generate OpenAPI and fastSDK clients from the effective endpoint set.

#### Operational verification

- Run smoke deployments on reference RunPod hardware.
- Capture engine version, image digest, served context, startup result, and verification time.
- Block managed deployment when recipes are stale, incompatible, or unresolved.
- Keep custom deployment available as the escape hatch.

## Implementation map

### APIPod

- `apipod/cli.py`: distinguish model intent from project entrypoints and bypass scan/build for managed models.
- `apipod/serve.py`: accept model references, resolve capabilities, and implement explicit route precedence.
- `apipod/models/chat.py`: keep the public capability facade free from model-family behavior and static generation defaults.
- `apipod/models/vllm/chat.py`: omit non-explicit CLI flags and request fields.
- `apipod/models/vllm/config.py`: make every engine value optional.
- `apipod/models/transformers/`: move reusable loading and input behavior into generic adapters.
- `apipod/deploy/scanner.py`, `deployment_manager.py`, and Docker generation: remain the custom service path only.

### Model definitions

- `qwen-models/qwen.py`: replace Qwen engine subclasses with catalog data after generic adapters and recipes cover their behavior.
- `qwen-models/gemma4.py` and `glm.py`: migrate model lists to catalog ingestion.
- `qwen-models/zimage.py`: migrate only after a generic Diffusers image-generation preset preserves its loading and sampling behavior.

### Schemas and backend

- `socaity-schemas/platform/ai_model.py`: represent artifact identity, capability evidence, and native context without runtime policy.
- `socaity-schemas/platform/deployment.py`: add discriminated deployment intent and versioned runtime plan schemas.
- `socaity_backend/sql/supabase/tables/004_catalog.sql`: add recipe and plan persistence, then normalize artifacts incrementally.
- deployment RPCs: persist exact recipe, artifact, image, and plan snapshots.
- `socaity_cron_jobs/data_scraping/hf_enrich.py` and `refresh_hf_metadata.py`: ingest repository configuration, not only card summaries.
- new vLLM recipe sync job: validate pinned upstream YAML and upsert normalized recipes.
- `catalog/repository.py` and Typesense collection projections: expose managed readiness and runtime filters.
- `socaity_backend/core/hosting/deployment_analyzer.py`: analyze explicit managed and custom intents through one plan resolver.
- `socaity_backend/endpoints/deployment.py`: accept managed model intent without requiring `apipod.json` or image upload.
- `socaity_backend/core/hosting/hosting_manager.py`: provision only from persisted plans.
- RunPod hosting adapter: translate the plan into provider resources without owning model policy.

### CLI and web clients

The APIPod CLI currently delegates platform operations to `socaity-cli`. That client must add managed model intent, plan preview, confirmation, and status polling while preserving the custom image push flow.

Any web deployment surface must consume the same analyze and draft contracts. Runtime planning stays in the backend, not in frontend form logic.

## Operational invariants

- Custom image promotion remains locked to the verified Harbor digest.
- Managed images are platform-maintained and pinned by digest.
- Hugging Face credentials stay deployment-scoped and never become model catalog data.
- Analyze remains side-effect-free. Draft creation is the first persistence boundary.
- Runtime plans survive draft cleanup and remain reproducible after recipe updates.
- OpenAPI finalization stays idempotent through its specification hash.
- Endpoint descriptors remain service-level while contracts remain deployment-level.
- RunPod cold and hot worker semantics and billing continue to derive from the selected pricing rule and compute mode.

## Acceptance criteria

APIPod v2 is complete when:

- a supported model starts locally from one identifier;
- the same identifier deploys as a managed RunPod service without local source or Docker;
- a custom service always builds its image locally before Socaity upload;
- the generated endpoint set follows model capabilities and respects explicit user routes;
- model code contains no engine-specific family subclasses for reusable compatibility;
- omitted generation and vLLM settings remain omitted;
- Qwen3.8 serves beyond 8,192 tokens without APIPod imposing that number;
- native and effective served context are visible as separate facts;
- managed plans are reproducible from pinned artifact, recipe, and image revisions;
- unsupported managed models fail clearly and retain local or custom deployment paths.

## Non-goals

- Adding a managed cloud provider other than RunPod.
- Building custom service images in the Socaity backend.
- Creating one container image per model.
- Treating all model families as vLLM chat workloads.
- Enabling one-million-token context by default.
- Turning optional vLLM tuning advice into APIPod defaults.
