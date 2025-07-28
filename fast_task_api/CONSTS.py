from enum import Enum


class FTAPI_BACKENDS(Enum):
    RUNPOD = "runpod"
    FASTAPI = "fastapi"


class FTAPI_DEPLOYMENTS(Enum):
    LOCALHOST = "localhost"
    HOSTED = "hosted"
    SERVERLESS = "serverless"


class SERVER_HEALTH(Enum):
    INITIALIZING = "initializing"
    BOOTING = "booting"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
