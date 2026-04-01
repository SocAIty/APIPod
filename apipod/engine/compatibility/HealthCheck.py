from apipod.common.constants import SERVER_HEALTH

import os
from typing import Dict, Union, Tuple


class HealthCheck:
    def __init__(self):
        self.status = SERVER_HEALTH.INITIALIZING

    @staticmethod
    def is_running_in_azure() -> bool:
        # Check for Azure-specific environment variables
        return "WEBSITE_INSTANCE_ID" in os.environ or "KUBERNETES_SERVICE_HOST" in os.environ

    def get_status_code(self, health_status: SERVER_HEALTH) -> int:
        """Map internal health states to appropriate HTTP status codes."""
        # So far reflect azure mapping https://learn.microsoft.com/en-us/azure/container-apps/health-probes?tabs=arm-template
        # Azure interprets 200 to 400 as success, everything else as failure
        status_mapping = {
            SERVER_HEALTH.RUNNING: 200,
            SERVER_HEALTH.INITIALIZING: 200,
            SERVER_HEALTH.BOOTING: 200,
            SERVER_HEALTH.BUSY: 200,
            SERVER_HEALTH.ERROR: 503  # service unavailable
        }
        return status_mapping.get(health_status, 200)

    def get_health_response(self) -> Tuple[int, Union[Dict, str]]:
        """
        Generate health check response based on deployment environment.
        Returns a tuple of HTTP status code and response body.
        """
        ret_val = self.status.value
        if self.is_running_in_azure():
            # Azure-specific format
            is_healthy = self.status in [SERVER_HEALTH.RUNNING, SERVER_HEALTH.BUSY]
            ret_val = {
                "status": "Healthy" if is_healthy else "Unhealthy",
                "details": {
                    "state": self.status.value,
                    "checks": [
                        {
                            "name": "api_health",
                            "status": "Healthy" if is_healthy else "Unhealthy",
                            "description": f"API is in {self.status.value} state"
                        }
                    ]
                }
            }

        return self.get_status_code(self.status), ret_val
