"""Launch test services locally for fastSDK development. Not a pytest module.

Run or debug this file from VS Code. Uncomment one ``launch_*`` call in ``main``.

Use the **apipod** project venv (``apipod/venv``), not fastsdk's.

Each call blocks on uvicorn until you stop the debugger.

``launch_all`` mounts each service group as a prefixed sub-router so endpoints
with the same path (e.g. ``/chat`` on schemas vs streaming) do not overwrite
each other:

- ``/core/...``       core_service
- ``/schemas/...``    schema_service (all, extended, mapping)
- ``/streaming/...``  streaming_service
"""

import os
import sys
from pathlib import Path

# Allow ``from services import ...`` when run/debugged as a script.
sys.path.insert(0, str(Path(__file__).parent))

from apipod import APIPod
from services import core_service, schema_service, streaming_service

HOST = "127.0.0.1"
# Override via env before launch (see apipod ``APIPOD_SIMULATE`` / ``APIPOD_NATIVE``):
#   standard FastAPI:  (unset)
#   serverless:        APIPOD_SIMULATE=serverless
#   serverless-runpod: APIPOD_SIMULATE=serverless-runpod
#   runpod native:     APIPOD_SIMULATE=serverless-runpod  APIPOD_NATIVE=true
SIMULATE = os.environ.get("APIPOD_SIMULATE") or None
DIRECT = os.environ.get("APIPOD_NATIVE", "").strip().lower() in ("1", "true", "yes") or None


def launch_core(port: int = 8000, host: str = HOST, simulate: str = SIMULATE, direct: bool = DIRECT) -> None:
    app = APIPod(simulate=simulate, direct=direct)
    core_service.register(app)
    app.start(host=host, port=port)


def launch_schemas(port: int = 8000, host: str = HOST, simulate: str = SIMULATE, direct: bool = DIRECT) -> None:
    app = APIPod(simulate=simulate, direct=direct)
    schema_service.register_all(app)
    schema_service.register_extended(app)
    schema_service.register_mapping(app)
    app.start(host=host, port=port)


def launch_streaming(port: int = 8000, host: str = HOST, simulate: str = SIMULATE, direct: bool = DIRECT) -> None:
    app = APIPod(simulate=simulate, direct=direct)
    streaming_service.register(app)
    app.start(host=host, port=port)


def launch_all(port: int = 8000, host: str = HOST, simulate: str = SIMULATE, direct: bool = DIRECT) -> None:
    app = APIPod(simulate=simulate, direct=direct, title="APIPod test services (all)")

    core = APIPod(simulate=simulate, direct=direct)
    core_service.register(core)
    app.include_router(core, prefix="/core")

    schemas = APIPod(simulate=simulate, direct=direct)
    schema_service.register_all(schemas)
    schema_service.register_extended(schemas)
    schema_service.register_mapping(schemas)
    app.include_router(schemas, prefix="/schemas")

    streaming = APIPod(simulate=simulate, direct=direct)
    streaming_service.register(streaming)
    app.include_router(streaming, prefix="/streaming")

    app.start(host=host, port=port)


if __name__ == "__main__":
    # launch_core()
    # launch_schemas()
    # launch_streaming()
    launch_all()
