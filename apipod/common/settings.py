from os import environ
from apipod.common.constants import COMPUTE, PROVIDER

# Deployment target. Socaity overwrites these env vars in a managed deployment so
# the right backend is selected in production (e.g. serverless + runpod).
APIPOD_COMPUTE = environ.get("APIPOD_COMPUTE", COMPUTE.DEDICATED.value)
APIPOD_PROVIDER = environ.get("APIPOD_PROVIDER", PROVIDER.LOCALHOST.value)
APIPOD_REGION = environ.get("APIPOD_REGION", "")

# Local simulation intent (ignored in managed deployments). Empty = development.
# Target string is "{compute}-{provider}", e.g. "serverless-runpod".
APIPOD_SIMULATE = environ.get("APIPOD_SIMULATE", "")
APIPOD_NATIVE = environ.get("APIPOD_NATIVE", "").strip().lower() in ("1", "true", "yes")

APIPOD_HOST = environ.get("APIPOD_HOST", "0.0.0.0")
APIPOD_PORT = int(environ.get("APIPOD_PORT", 8000))

SERVER_DOMAIN = environ.get("SERVER_DOMAIN", "")

DEFAULT_DATE_TIME_FORMAT = environ.get("FTAPI_DATETIME_FORMAT", '%Y-%m-%dT%H:%M:%S.%f%z')

# Socaity deployment certificate (SHA1 of a shared secret) for detecting a managed
# deployment. When verified, simulate/direct are ignored and the backend is chosen
# from APIPOD_COMPUTE / APIPOD_PROVIDER.
SOCAITY_DEPLOYMENT_CERT = environ.get("SOCAITY_DEPLOYMENT_CERT", "")
_EXPECTED_CERT_HASH = "7b35ca9da2f0c280d48f66c780a0a0d5d3f8ad8a"
IS_MANAGED_DEPLOYMENT = SOCAITY_DEPLOYMENT_CERT == _EXPECTED_CERT_HASH
