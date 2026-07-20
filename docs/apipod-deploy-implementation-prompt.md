# APIPod Deploy Pipeline: Implementation Prompt

Use this document as the **single agent prompt** for implementing the secure container registry and full `apipod deploy` pipeline. Read `APIPod/docs/apipod-deploy.md` first. Treat that file as the architecture spec. This file adds execution instructions, codebase context, and the end-to-end test loop.

---

## Role

You are a senior platform engineer implementing Socaity's **APIPod deploy pipeline**: Harbor-backed image push, digest-locked promotion, and provisioning through the existing async deployment system.

Your job is not to redesign the architecture. It is to implement what `apipod-deploy.md` specifies, in order, and prove it with an **end-to-end test suite** that deploys real services and validates them through **fastSDK / socaity-sdk**.

---

## Part 1: Instructions

### 1.1 Think before coding
Do thinking and reasoning like a smart caveman.
- Cut all filler, keep technical substance.
- Drop articles (a, an, the), filler (just, really, basically, actually).
- Drop pleasantries (sure, certainly, happy to).
- No hedging. Fragments fine. Short synonyms.
- Technical terms stay exact. Code blocks unchanged.
- Pattern: [thing] [action] [reason]. [next step]

Before each phase:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.
- Use first-principles reasoning for security decisions (credential scope, digest binding, quarantine).

### 1.2 Simplicity first

- Minimum code that solves the problem. Nothing speculative.
- No features beyond what `apipod-deploy.md` defines for the current phase.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 1.3 Surgical changes

- Touch only what the current phase requires.
- Match existing style in each repo (`socaity_backend`, `socaity-cli`, `APIPod`).
- Do not "improve" adjacent code, comments, or formatting.
- Remove imports/functions only if **your** changes made them unused.
- Do not delete pre-existing dead code unless asked.

### 1.4 Code quality and tooling

Apply across all touched repos:

| Rule | Detail |
|------|--------|
| Docstrings | Google format. Explain non-obvious **why**, not obvious **what**. |
| Style | Ruff: ignore E203, W503, W501, W293, W291; line length 120. |
| Python | Prefer builtins, generators, comprehensions where they simplify. |
| Imports | No local imports. |
| Compatibility | No deprecated or backward-compat shims. |
| Python execution | Always use the **project venv** (`APIPod/venv`, `socaity_backend/venv`, etc.). |
| Installs | Use `make` / project tooling. No raw `pip install` in instructions. |
| Dev tools | `make`, `ruff`, `mypy`, `pytest`. |
| Tests | **Ask before adding a new test file.** When approved, place tests in the repo that owns the code under test. |
| Docs | Update `README.md` for user-facing CLI changes. Update `TECHNICAL_README.md` for architectural changes. Keep updates brief. |

### 1.5 Security non-negotiables

These override convenience:

1. **Ephemeral credentials only.** Harbor robot per deployment, TTL ≤ 15 min, push+pull on exactly one staging repo.
2. **Digest-locked promotion.** Deploy plane never references mutable tags. CLI reports digest; backend locks it before promotion.
3. **No persistent docker creds on disk.** CLI must not write `~/.docker/config.json`. Use subprocess with stdin or Docker SDK with ephemeral auth.
4. **Quarantine.** Staging images are not pullable by the deploy plane until promotion succeeds.
5. **No enumeration.** Robot accounts must not list catalog or other tenants' repos.
6. **Secrets never logged.** Registry passwords must not appear in logs, prints, or API responses after issuance.

### 1.6 Public-facing text (CLI prints, logs, errors)

- Developer-first, technically correct, direct tone.
- "We" for the platform, "you" for the reader.
- No em-dashes (U+2014). Use period, comma, parentheses, or colon.
- No marketing adverbs: seamlessly, effortlessly, robust, powerful, cutting-edge, etc.
- Bullets only for true enumerations (statuses, flags, env vars).
- Committed language: "Push failed: credentials expired." not "Push might have failed."

### 1.7 Git workflow

- Work on branch `feature/registry-pipeline` via git worktree.
- Do not commit unless explicitly asked.
- When commit use conventional commits.
- Do not push unless explicitly asked.

### 1.8 Phase gate rule

**Do not start phase N+1 until phase N passes its gate tests** (see Part 4). If a gate fails, fix and re-run before proceeding.

---

## Part 2: Context

### 2.1 Problem (one sentence)

Authenticated users must push exactly one Docker image to exactly one staging repository for a short window; the platform validates, promotes by digest to prod, provisions, and reaches `live` without dashboard interaction.

### 2.2 Architecture summary

```
apipod deploy
  → analyze + draft (existing)
  → push_credentials (new)
  → docker push to Harbor staging/{user_id}/{deployment_id}
  → POST /pushed {digest, size_bytes}
  → promotion worker: Trivy + skopeo copy → prod/{user_id}/{service_id}@{digest}
  → existing provision pipeline (health job → OpenAPI job → live)
  → CLI polls GET /deployment/{id}
```

**Harbor layout:** one instance, two projects (`staging`, `prod`). Production uses digest references only.

### 2.3 Locked decisions (do not re-open without explicit approval)

| Decision | Value |
|----------|-------|
| Staging registry | Harbor project `staging` |
| Prod registry (v1) | Harbor project `prod` |
| User registry auth | Ephemeral Harbor robot per deployment |
| Promotion trigger | CLI `POST /pushed` (Harbor webhook optional later) |
| Image reference at deploy | Digest only |
| Staging quota | Exempt from user registry quota |
| Max image size | 20 GB compressed, hard reject |
| CVE policy | Block Critical |
| Architecture | `linux/amd64` only for v1 |
| Draft endpoint | Keep `POST /v1/deployment/draft` separate |
| Failed deploy resume | New `deployment_id` per attempt, same `service_id` |
| Local Harbor | `infra/harbor-local/` docker-compose |

### 2.4 Codebase map

| Area | Path | Your work |
|------|------|-----------|
| Architecture spec | `APIPod/docs/apipod-deploy.md` | Read-only reference |
| CLI deploy flow | `socaity-cli/socaity_cli/deployment.py` | Add push + poll |
| Backend HTTP client | `socaity-cli/socaity_cli/backend_client.py` | Add registry API methods |
| APIPod CLI entry | `APIPod/apipod/cli.py` | Wire `apipod deploy` to full flow |
| Docker build | `APIPod/apipod/deploy/deployment_manager.py` | Reuse for local image tag |
| Async deploy jobs | `socaity_backend/core/hosting/runpod/async_deployment_progress.py` | Extend with promotion pre-phase |
| Deployment tracker | `socaity_backend/core/hosting/runpod/deployment_tracker.py` | Add `image_promotion_job_id` |
| File quota pattern | `socaity_backend/common/storage/file_upload_handler.py` | Mirror for registry quota |
| Existing deploy tests | `socaity_backend/tests/test_deployment_flow.py` | Extend, do not replace |
| Local Harbor | `infra/harbor-local/` | Create in phase 0 |
| Harbor client | `socaity_backend/socaity_backend/core/registry/` | Create |
| Registry push helper | `socaity-cli/socaity_cli/registry_push.py` | Create |
| Test services | `APIPod/test/debug_test_services.py` | E2E deploy targets |
| fastSDK integration tests | `fastSDK/test/test_apipod_debug_test_services.py` | E2E validation pattern |

### 2.5 Existing API surface (extend, do not replace)

Already implemented in `SocaityBackendClient`:

- `POST v1/deployment/analyze`
- `POST v1/deployment/draft`
- `POST v1/deployment/hf_token`

To implement:

| Method | Path | Body / response |
|--------|------|-----------------|
| `POST` | `/v1/deployment/{id}/push_credentials` | Returns registry, repository, username, password, expires_in, tag |
| `POST` | `/v1/deployment/{id}/pushed` | `{ "digest": "sha256:...", "size_bytes": int }` |
| `GET` | `/v1/deployment/{id}` | Status machine + phase + digest + error |
| `DELETE` | `/v1/deployment/{id}` | Cancel, revoke robot, delete staging repo |

### 2.6 Deployment status machine

```
draft → awaiting_image → pushing → validating → promoting → provisioning → live
                              ↘ failed_push | failed_validation | failed_provision | cancelled
```

Wire `image_promotion_job_id` into `DeploymentTracker` before the existing health/OpenAPI job chain.

### 2.7 Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `SOCAITY_BACKEND_URL` | CLI | Backend API base |
| `SOCAITY_API_KEY` | CLI | Bearer auth |
| `SOCAITY_REGISTRY_URL` | CLI | Override registry host (local: `localhost:5080`) |
| `HARBOR_URL` | Backend | Harbor API base |
| `HARBOR_ADMIN_USER` | Backend | Admin for robot creation |
| `HARBOR_ADMIN_PASSWORD` | Backend | Admin secret |
| `HARBOR_STAGING_PROJECT` | Backend | Default `staging` |
| `HARBOR_PROD_PROJECT` | Backend | Default `prod` |

---

## Part 3: Step-by-step implementation

Execute phases in order. Each phase ends with a **gate** (Part 4).

### Phase 0: Local Harbor infrastructure

**Goal:** Runnable Harbor on localhost for dev and E2E.

**Tasks:**

1. Create git worktree on `feature/registry-pipeline`.
2. Add `infra/harbor-local/docker-compose.yml` (Harbor standalone or official compose).
3. Add `infra/harbor-local/README.md` with start/stop, default admin creds, TLS notes for local HTTP.
4. Add `infra/harbor-local/scripts/seed-projects.sh`:
   - Create projects `staging` and `prod`.
   - Document manual robot creation for smoke tests.
5. Verify manually: `docker login localhost:5080` → push `hello-world` to a test repo in `staging`.

**Deliverables:** `infra/harbor-local/*` committed on feature branch.

**Gate:** Phase 0 gate (Part 4.1).

---

### Phase 1: Backend Harbor client + credential issuance

**Goal:** Backend can create scoped robots and expose `push_credentials`.

**Tasks:**

1. Create `socaity_backend/socaity_backend/core/registry/harbor_client.py`:
   - `create_push_robot(project, repository, ttl_seconds) -> RobotCredentials`
   - `revoke_robot(robot_id)`
   - `delete_repository(project, repository)`
   - `get_manifest_digest(project, repository, reference) -> str`
2. Create Pydantic models for credentials response and deployment status.
3. DB migration:
   - `deployments.status` (enum matching state machine)
   - `deployments.image_digest`, `image_size_bytes`, `registry_storage_bytes`
   - `deployments.credentials_expires_at`, `harbor_robot_name`
   - User limits: `registry_quota_bytes`, `registry_used_bytes`
4. Implement `POST /v1/deployment/{id}/push_credentials`:
   - Auth: user owns deployment
   - Precondition: status `awaiting_image` or `draft` (transition to `awaiting_image`)
   - Quota check: `registry_used_bytes + estimated_size ≤ registry_quota_bytes`
   - Create robot, return creds, set `credentials_expires_at`
5. Implement `GET /v1/deployment/{id}` with current status and phase.
6. Implement `DELETE /v1/deployment/{id}`: revoke robot, delete staging repo, set `cancelled`.
7. Unit tests with mocked Harbor HTTP API.

**Gate:** Phase 1 gate (Part 4.2).

---

### Phase 2: Push confirmation + promotion worker

**Goal:** After CLI reports digest, image is scanned, promoted, staging cleaned.

**Tasks:**

1. Implement `POST /v1/deployment/{id}/pushed`:
   - Validate digest format `sha256:[64 hex]`
   - Fetch manifest from Harbor; reject if digest mismatch
   - Lock `deployment.image_digest`; transition to `validating`
   - Enqueue promotion job; return 202
2. Create `socaity_backend/socaity_backend/core/registry/promotion_worker.py`:
   - Pull manifest by digest from staging
   - Run Trivy scan (Harbor API or CLI wrapper); fail on Critical CVE
   - Verify `linux/amd64`, size ≤ 20 GB
   - `skopeo copy` staging → prod by digest
   - Delete staging repository
   - Update `registry_used_bytes` (replace previous image for same service)
   - Transition to `provisioning`; trigger existing RunPod deploy with `prod/{user_id}/{service_id}@sha256:...`
3. Extend `DeploymentTracker` with `image_promotion_job_id`.
4. Wire promotion completion into `poll_async_deployment` / `maybe_advance_deployment_on_service_read`.
5. Audit log entry: user_id, deployment_id, digest, timestamp, scan summary.

**Gate:** Phase 2 gate (Part 4.3).

---

### Phase 3: CLI push + poll

**Goal:** `apipod deploy` runs analyze → draft → push → poll until terminal state.

**Tasks:**

1. Create `socaity-cli/socaity_cli/registry_push.py`:
   - `push_image(registry, repository, tag, username, password, local_tag) -> tuple[digest, size_bytes]`
   - Ephemeral auth only; `docker logout` in `finally`
   - On 401 mid-push: raise `CredentialsExpired` for caller to refresh
2. Extend `SocaityBackendClient`:
   - `get_push_credentials(deployment_id)`
   - `confirm_image_pushed(deployment_id, digest, size_bytes)`
   - `get_deployment_status(deployment_id)`
   - `cancel_deployment(deployment_id)`
3. Extend `socaity_cli/deployment.py`:
   - `push_image_for_deployment(deployment_id, local_tag) -> Optional[str]`
   - `poll_deployment_until_terminal(deployment_id, timeout_s=3600) -> Optional[Dict]`
   - `run_full_deploy(config, target, local_tag, skip_build=False) -> Optional[Dict]`
4. Update `APIPod/apipod/cli.py`:
   - `apipod deploy [target]` chains full flow
   - Flags: `--resume DEPLOYMENT_ID`, `--skip-build`, `--push-only`
   - Local tag default: `apipod-{title}` from `apipod.json`
5. Update post-draft CLI message when full pipeline enabled.
6. Update `APIPod/README.md` deploy section.

**Gate:** Phase 3 gate (Part 4.4).

---

### Phase 4: Security hardening + quota API

**Goal:** Production-ready guards before E2E.

**Tasks:**

1. Rate-limit `push_credentials` (10/hour/user).
2. Reject credential refresh if deployment not in `awaiting_image` or `pushing`.
3. Expose registry usage in existing user/storage API (mirror `UserStorageUsage`).
4. Verify robot cannot `GET /v2/_catalog` (manual or automated security check).
5. Staging GC: abandoned deployments (`awaiting_image` + expired creds + no push) cleaned after 1 hour.

**Gate:** Phase 4 gate (Part 4.5).

---

### Phase 5: End-to-end test harness

**Goal:** Automated proof that deploy + socaity-sdk works for the full service set.

See Part 4.6 for the full E2E spec. Implement only after phases 0–4 gates pass.

**Tasks:**

1. Create E2E orchestrator script (location: `APIPod/test/e2e_deploy/`).
2. Create minimal `apipod.json` + Dockerfile per test service variant.
3. Wire fastSDK validation against deployed service URLs.
4. Document how to run locally (Harbor + backend + deploy plane) vs CI.

**Gate:** Part 4.6 (final success criterion).

---

## Part 4: Test loop

Use this loop after **every phase**. Do not advance until the current gate is green.

```
┌─────────────────────────────────────────────────────────┐
│  1. Implement phase tasks                               │
│  2. Run phase gate tests                                │
│  3. If fail → diagnose → fix → return to 2              │
│  4. If pass → run regression (prior phase gates)        │
│  5. If regression fail → fix → return to 2              │
│  6. Phase complete → next phase                         │
└─────────────────────────────────────────────────────────┘
```

After Phase 5, the **full E2E suite** is the only gate that matters for feature completion.

---

### 4.1 Gate: Phase 0 (Harbor local)

```bash
cd infra/harbor-local && docker compose up -d
./scripts/seed-projects.sh
docker pull hello-world
docker tag hello-world localhost:5080/staging/e2e-smoke/hello:latest
# login with admin or seeded robot
docker push localhost:5080/staging/e2e-smoke/hello:latest
```

**Pass:** Image visible in Harbor UI under project `staging`. `docker compose down` cleans up.

---

### 4.2 Gate: Phase 1 (credentials API)

With Harbor local + backend running against it:

```bash
# socaity login (or SOCAITY_API_KEY)
apipod deploy serverless   # creates draft
# capture deployment_id from output

curl -X POST "$SOCAITY_BACKEND_URL/v1/deployment/{id}/push_credentials" \
  -H "Authorization: Bearer $SOCAITY_API_KEY"

curl "$SOCAITY_BACKEND_URL/v1/deployment/{id}" \
  -H "Authorization: Bearer $SOCAITY_API_KEY"
```

**Pass criteria:**

- [ ] Response includes registry, repository, username, password, expires_in
- [ ] `GET` returns `awaiting_image`
- [ ] Second user's credentials cannot push to first user's repository
- [ ] Quota exceeded returns 402 or 413 with clear message
- [ ] `pytest socaity_backend/tests/` passes for new registry unit tests

---

### 4.3 Gate: Phase 2 (promotion)

Using credentials from 4.2:

```bash
apipod build   # produces local image apipod-{title}
# programmatic push with issued creds (or apipod deploy --push-only once phase 3 partial)

curl -X POST "$SOCAITY_BACKEND_URL/v1/deployment/{id}/pushed" \
  -H "Authorization: Bearer $SOCAITY_API_KEY" \
  -d '{"digest":"sha256:...","size_bytes":...}'

# poll until status leaves validating/promoting
curl "$SOCAITY_BACKEND_URL/v1/deployment/{id}" ...
```

**Pass criteria:**

- [ ] Status progresses: `validating` → `promoting` → `provisioning` (or `live` if deploy plane available)
- [ ] Image exists in Harbor `prod` project at expected digest
- [ ] Staging repo deleted after promotion
- [ ] Wrong digest rejected with 409
- [ ] `pytest socaity_backend/tests/test_deployment_flow.py` passes with promotion mocks

---

### 4.4 Gate: Phase 3 (CLI full flow)

```bash
cd APIPod/test/e2e_fixtures/core_service   # minimal fixture project
apipod scan
apipod build
apipod deploy serverless-runpod
```

**Pass criteria:**

- [ ] Single command reaches `live` (or `provisioning` if deploy plane is stubbed) without dashboard
- [ ] No entry added to `~/.docker/config.json` (verify before/after)
- [ ] CLI prints digest and deployment_id
- [ ] `apipod deploy --resume {id}` works after simulated interrupt
- [ ] Credentials expiry mid-push: CLI requests fresh creds and resumes

---

### 4.5 Gate: Phase 4 (security + quota)

**Pass criteria:**

- [ ] 11th `push_credentials` call in one hour returns 429
- [ ] Issued robot cannot `curl https://registry/v2/_catalog`
- [ ] Issued robot cannot push to a different repository path
- [ ] `registry_used_bytes` updates after promotion
- [ ] Cancelled deployment: staging repo gone, robot revoked

---

### 4.6 Gate: Final E2E (feature success criterion)

**This is the definition of done.** The implementation is successful **only if every service in the E2E matrix passes fastSDK validation after `apipod deploy`.**

#### E2E service matrix

Deploy each row as a **separate** `apipod deploy` run. Each must reach `live` and pass its fastSDK test subset.

| ID | Service fixture | Source | Deploy target | fastSDK validation |
|----|-----------------|--------|---------------|-------------------|
| E2E-01 | Core minimal | `APIPod/test/services/core_service.py` (`register_minimal`) | `serverless-runpod` | `test_iterate_all_endpoints` on `/core/*` paths |
| E2E-02 | Schema (all) | `APIPod/test/services/schema_service.py` (`register_all`) | `serverless-runpod` | Schema endpoint smoke: `/schemas/chat-raw`, `/schemas/embedding-raw` |
| E2E-03 | Schema (extended) | `schema_service.register_extended` | `serverless-runpod` | `test_connect_chat_extended` |
| E2E-04 | Streaming | `APIPod/test/services/streaming_service.py` | `serverless-runpod` | `test_streaming_text`, `test_streaming_chat_schema` |
| E2E-05 | All groups | `debug_test_services.launch_all` layout | `serverless-runpod` | Full `fastSDK/test/test_apipod_debug_test_services.py` suite |

#### E2E orchestration flow (per service)

```
1. Start stack: harbor-local + socaity_backend + deploy plane (RunPod or local simulate)
2. socaity login / set SOCAITY_API_KEY
3. cd e2e_fixtures/{service_id}/
4. apipod scan && apipod build && apipod deploy serverless-runpod
5. Poll until GET /v1/deployment/{id} → status == "live"
6. Resolve service URL from deployment response (or catalog)
7. export APIPOD_DEBUG_TEST_SERVICE_URL=<live_url>
8. cd fastSDK && pytest test/test_apipod_debug_test_services.py -k "<subset>" -v
9. Record: deployment_id, digest, duration, pass/fail
10. Tear down deployment (DELETE /v1/deployment/{id} or service delete)
```

#### E2E pass criteria (all must be true)

- [ ] **E2E-01 through E2E-05** all reach `live` within 60 minutes each (adjust for GPU cold start)
- [ ] fastSDK tests pass for each service against its **deployed** URL, not localhost
- [ ] Each deployed service uses the **digest** recorded in the deployment row (verify in Harbor prod)
- [ ] No cross-tenant leakage: second test user cannot pull first user's staging image
- [ ] Total registry storage for test user matches sum of promoted images (± GC tolerance)

#### E2E failure protocol

When any E2E row fails:

1. Capture: deployment_id, status, phase, error, Harbor staging/prod state, backend logs.
2. Classify: `cli` | `backend` | `harbor` | `promotion` | `provision` | `fastsdk`.
3. Fix root cause in the owning phase. Do not patch symptoms in the E2E script.
4. Re-run **all prior phase gates**, then re-run **full E2E matrix** (not just the failed row).
5. Repeat until E2E-01..E2E-05 are green in one consecutive run.

---

## Part 5: Agent execution checklist

Copy this checklist into your session and tick items as you go.

### Setup
- [ ] Read `APIPod/docs/apipod-deploy.md` end to end
- [ ] Read `fastSDK/AGENTS.md` 
- [ ] Create worktree `feature/registry-pipeline`
- [ ] Confirm Docker, make, and project venvs available

### Phase 0
- [ ] `infra/harbor-local/` created
- [ ] Gate 4.1 passes

### Phase 1
- [ ] `harbor_client.py` + migration + endpoints
- [ ] Gate 4.2 passes

### Phase 2
- [ ] `promotion_worker.py` + `/pushed` + deploy wiring
- [ ] Gate 4.3 passes

### Phase 3
- [ ] `registry_push.py` + CLI full deploy
- [ ] Gate 4.4 passes

### Phase 4
- [ ] Rate limits, quota API, security checks
- [ ] Gate 4.5 passes

### Phase 5
- [ ] E2E fixtures + orchestrator
- [ ] **E2E-01..E2E-05 all pass (Part 4.6)**

### Done
- [ ] All phase gates green in one run
- [ ] Full E2E matrix green in one consecutive run
- [ ] README / TECHNICAL_README updated
- [ ] No secrets in git diff

---

**Success statement:** The APIPod deploy pipeline is complete when a developer can run `apipod deploy serverless-runpod` on each E2E fixture service, the platform promotes the image by digest through Harbor, the service reaches `live`, and `fastSDK/test/test_apipod_debug_test_services.py` passes against every deployed URL in the E2E matrix without manual dashboard steps.
