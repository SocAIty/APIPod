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

SERVER_DOMAIN = environ.get("SERVER_DOMAIN", "")

DEFAULT_DATE_TIME_FORMAT = environ.get("FTAPI_DATETIME_FORMAT", '%Y-%m-%dT%H:%M:%S.%f%z')

# Official staff-deployment marker (SHA1 of a shared secret). Does NOT gate
# user backend selection — APIPOD_COMPUTE / APIPOD_PROVIDER alone choose the
# router for normal deploys. When verified, simulate/direct are ignored so
# official images always honor the platform env.
SOCAITY_DEPLOYMENT_CERT = environ.get("SOCAITY_DEPLOYMENT_CERT", "")
_EXPECTED_CERT_HASH = "7b35ca9da2f0c280d48f66c780a0a0d5d3f8ad8a"
IS_MANAGED_DEPLOYMENT = SOCAITY_DEPLOYMENT_CERT == _EXPECTED_CERT_HASH