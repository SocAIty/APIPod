# E2E deploy matrix (gate 4.6)

Automated proof that `apipod deploy` works end to end: each matrix row is a
fixture service that is built, pushed through the registry pipeline, promoted,
provisioned, and validated with fastSDK against its deployed URL.

Spec: `APIPod/docs/apipod-deploy.md` and part 4.6 of
`APIPod/docs/apipod-deploy-implementation-prompt.md`.

## Matrix

| ID | Fixture | fastSDK validation |
|----|---------|--------------------|
| e2e-01 | core_service `register_minimal` | `test_iterate_all_endpoints` |
| e2e-02 | schema_service `register_all` | chat-raw / embedding-raw smoke |
| e2e-03 | schema_service `register_extended` | `test_connect_chat_extended` |
| e2e-04 | streaming_service | streaming text + chat schema |
| e2e-05 | all groups (`launch_all` layout) | full debug-services suite |

## Full run (real deploy plane)

Requires: local Harbor (`APIPodInferenceBE/infra/harbor-local`), a running
`socaity_backend` with `HARBOR_URL` + RunPod configured, Docker, and a login.

```bash
socaity login                          # or export SOCAITY_API_KEY=sk_...
export SOCAITY_BACKEND_URL=http://127.0.0.1:8000
python run_matrix.py
```

## Local pipeline check (no deploy plane)

With `REGISTRY_PROVISION_AFTER_PROMOTION=false` on the backend, deployments
stop at `provisioning` after the image is promoted. This still exercises
build, push, digest lock, scan, and promotion:

```bash
python run_matrix.py --expect provisioning --skip-fastsdk
```

## Useful variants

```bash
python run_matrix.py --rows e2e-01              # single row
python fixtures.py                              # only materialize fixture projects into build/
```

`build/` is generated and git-ignored. Each fixture is a standalone project
(`service.py`, `services/`, `files/`, `apipod-deploy/`), so you can also `cd`
into one and run `apipod deploy serverless-runpod` by hand.

For a small real-service smoke test (Harbor push/promote without an ~11 GB
model image), use the monorepo fixture at `simple_test_service/`:

```bash
apipod -C simple_test_service deploy serverless-runpod --yes
```

## Failure protocol

1. Capture deployment_id, status, phase, error (summary table prints them).
2. Classify: cli | backend | harbor | promotion | provision | fastsdk.
3. Fix the root cause in the owning phase; never patch the E2E script around it.
4. Re-run prior phase gates (`socaity_backend/test/gate_phase*.py`), then the full matrix.
