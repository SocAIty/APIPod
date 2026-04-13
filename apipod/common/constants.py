from enum import Enum


class ORCHESTRATOR(Enum):
    SOCAITY = "socaity"
    LOCAL = "local"


class COMPUTE(Enum):
    DEDICATED = "dedicated"
    SERVERLESS = "serverless"


class PROVIDER(Enum):
    AUTO = "auto"
    LOCALHOST = "localhost"
    SOCAITY = "socaity"
    RUNPOD = "runpod"
    SCALEWAY = "scaleway"
    AZURE = "azure"


class SERVER_HEALTH(Enum):
    INITIALIZING = "initializing"
    BOOTING = "booting"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
