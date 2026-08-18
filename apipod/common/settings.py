from os import environ
from apipod.common.constants import COMPUTE, PROVIDER

# Deployment target. Deployed images (and the platform) set these so the right
# backend is selected — e.g. serverless + runpod for user RunPod deploys.
# Defaults are local development (plain FastAPI).
APIPOD_COMPUTE = environ.get("APIPOD_COMPUTE", COMPUTE.DEDICATED.value)
APIPOD_PROVIDER = environ.get("APIPOD_PROVIDER", PROVIDER.LOCALHOST.value)
APIPOD_REGION = environ.get("APIPOD_REGION", "")

# Local simulation intent. Empty = use APIPOD_COMPUTE / APIPOD_PROVIDER.
# Target string is "{compute}-{provider}", e.g. "serverless-runpod".
# Ignored when SOCAITY_DEPLOYMENT_CERT marks an official staff deployment.
APIPOD_SIMULATE = environ.get("APIPOD_SIMULATE", "")
APIPOD_NATIVE = environ.get("APIPOD_NATIVE", "").strip().lower() in ("1", "true", "yes")

APIPOD_HOST = environ.get("APIPOD_HOST", "0.0.0.0")
APIPOD_PORT = int(environ.get("APIPOD_PORT", 8000))

# RunPod worker in-flight jobs. Default 1 keeps transformers services serial.
# Official worker-vllm reads the same name into concurrency_modifier.
MAX_CONCURRENCY = int(environ.get("MAX_CONCURRENCY", "1"))

# Local vLLM HTTP server spawned by VLLMChat (not the APIPod bind port).
# Use APIPOD_VLLM_* names. Bare VLLM_* vars are reserved by the vLLM process
# (unknown ones warn; VLLM_PORT can clash with the engine).
VLLM_HOST = environ.get("APIPOD_VLLM_HOST", environ.get("VLLM_HOST", "127.0.0.1"))
VLLM_PORT = int(environ.get("APIPOD_VLLM_PORT", environ.get("VLLM_PORT", "18000")))
VLLM_MAX_MODEL_LEN = environ.get("APIPOD_VLLM_MAX_MODEL_LEN", environ.get("VLLM_MAX_MODEL_LEN", ""))
# worker-vllm default. vLLM's own default (1024) exceeds Qwen3.8 Mamba cache.
VLLM_MAX_NUM_SEQS = environ.get("APIPOD_VLLM_MAX_NUM_SEQS", environ.get("VLLM_MAX_NUM_SEQS", "256"))
VLLM_REASONING_PARSER = environ.get(
    "APIPOD_VLLM_REASONING_PARSER", environ.get("VLLM_REASONING_PARSER", "")
)
VLLM_EXTRA_ARGS = environ.get("APIPOD_VLLM_EXTRA_ARGS", environ.get("VLLM_EXTRA_ARGS", ""))
VLLM_STARTUP_TIMEOUT = int(
    environ.get("APIPOD_VLLM_STARTUP_TIMEOUT", environ.get("VLLM_STARTUP_TIMEOUT", "1200"))
)

SERVER_DOMAIN = environ.get("SERVER_DOMAIN", "")

DEFAULT_DATE_TIME_FORMAT = environ.get("FTAPI_DATETIME_FORMAT", '%Y-%m-%dT%H:%M:%S.%f%z')

# Official staff-deployment marker (SHA1 of a shared secret). Does NOT gate
# user backend selection — APIPOD_COMPUTE / APIPOD_PROVIDER alone choose the
# router for normal deploys. When verified, simulate/direct are ignored so
# official images always honor the platform env.
SOCAITY_DEPLOYMENT_CERT = environ.get("SOCAITY_DEPLOYMENT_CERT", "")
_EXPECTED_CERT_HASH = "7b35ca9da2f0c280d48f66c780a0a0d5d3f8ad8a"
IS_MANAGED_DEPLOYMENT = SOCAITY_DEPLOYMENT_CERT == _EXPECTED_CERT_HASH
