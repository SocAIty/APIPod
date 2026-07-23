"""E2E deploy matrix orchestrator (gate 4.6).

For each matrix row: build the fixture project, run ``apipod deploy
serverless-runpod --yes`` in it, poll until the expected terminal state,
optionally run the fastSDK validation subset against the live URL, then tear
the deployment down. Prints a summary table and exits non-zero on any failure.

Usage:

    # full E2E (real deploy plane, expects live + fastSDK):
    python run_matrix.py

    # local pipeline check (no deploy plane; stops at provisioning, no fastSDK):
    python run_matrix.py --expect provisioning --skip-fastsdk

    # single row:
    python run_matrix.py --rows e2e-01

Requirements: logged-in CLI (socaity login) or SOCAITY_API_KEY env var,
SOCAITY_BACKEND_URL pointing at the backend, Docker, and the workspace venv.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fixtures import MATRIX, MatrixRow, build_fixture

E2E_DIR = Path(__file__).resolve().parent
FASTSDK_DIR = E2E_DIR.parent.parent.parent / "fastSDK"

_DEPLOYMENT_ID_RE = re.compile(r"deployment_id=([0-9a-f-]{36})")


@dataclass
class RowResult:
    row_id: str
    deployment_id: Optional[str] = None
    status: Optional[str] = None
    digest: Optional[str] = None
    fastsdk: str = "skipped"
    duration_s: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.fastsdk in ("passed", "skipped")


def backend_request(method: str, path: str) -> tuple[int, dict]:
    import httpx

    backend = os.environ.get("SOCAITY_BACKEND_URL", "https://webapi.socaity.ai").rstrip("/")
    api_key = os.environ.get("SOCAITY_API_KEY", "")
    response = httpx.request(method, f"{backend}/{path}",
                             headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {}


def run_deploy(fixture: Path, timeout_s: int) -> tuple[Optional[str], str]:
    """Run apipod deploy in the fixture dir; returns (deployment_id, full output)."""
    process = subprocess.run(
        [sys.executable, "-m", "apipod.cli", "deploy", "serverless-runpod", "--yes"],
        cwd=fixture, capture_output=True, text=True, timeout=timeout_s,
    )
    output = process.stdout + process.stderr
    print(output)
    match = _DEPLOYMENT_ID_RE.search(output)
    return (match.group(1) if match else None), output


def poll_status(deployment_id: str, expect: str, timeout_s: int) -> Optional[dict]:
    deadline = time.time() + timeout_s
    status = None
    while time.time() < deadline:
        code, status = backend_request("GET", f"v1/deployment/{deployment_id}")
        if code == 200 and status.get("status") in (expect, "live", "failed_push",
                                                    "failed_validation", "failed_provision", "cancelled"):
            return status
        time.sleep(10)
    return status


def resolve_service_url(status: dict) -> Optional[str]:
    """Live service URL from the catalog (service -> deployments -> url)."""
    service_id = status.get("service_id")
    if not service_id:
        return None
    code, service = backend_request("GET", f"v1/catalog/services/{service_id}?expand=deployments")
    if code != 200:
        return None
    for deployment in service.get("deployments") or []:
        url = deployment.get("url") or deployment.get("service_url")
        if url:
            return url
    return None


def run_fastsdk(subset: str, service_url: str) -> bool:
    env = {**os.environ, "APIPOD_DEBUG_TEST_SERVICE_URL": service_url}
    cmd = [sys.executable, "-m", "pytest", "test/test_apipod_debug_test_services.py", "-v"]
    if subset:
        cmd += ["-k", subset]
    process = subprocess.run(cmd, cwd=FASTSDK_DIR, env=env)
    return process.returncode == 0


def run_row(row: MatrixRow, expect: str, skip_fastsdk: bool, timeout_s: int) -> RowResult:
    result = RowResult(row_id=row.row_id)
    started = time.time()
    print(f"\n=== {row.row_id}: {row.title} ===")
    try:
        fixture = build_fixture(row)
        deployment_id, output = run_deploy(fixture, timeout_s)
        result.deployment_id = deployment_id
        if deployment_id is None:
            result.error = "deploy produced no deployment_id"
            return result

        status = poll_status(deployment_id, expect, timeout_s)
        result.status = (status or {}).get("status")
        result.digest = (status or {}).get("image_digest")
        if result.status not in (expect, "live"):
            result.error = f"expected {expect}, got {result.status}: {(status or {}).get('error')}"
            return result

        if not skip_fastsdk and result.status == "live":
            url = resolve_service_url(status)
            if not url:
                result.error = "could not resolve live service URL"
                return result
            result.fastsdk = "passed" if run_fastsdk(row.fastsdk_subset, url) else "failed"
            if result.fastsdk == "failed":
                result.error = "fastSDK validation failed"
    finally:
        result.duration_s = time.time() - started
        if result.deployment_id:
            code, status = backend_request("GET", f"v1/deployment/{result.deployment_id}")
            backend_request("DELETE", f"v1/deployment/{result.deployment_id}")
            if code == 200:
                cleanup_prod_repository(status.get("service_id"))
    return result


def cleanup_prod_repository(service_id: Optional[str]) -> None:
    """Delete the promoted prod repo of a test service (local Harbor only)."""
    import httpx

    harbor_url = os.environ.get("HARBOR_URL", "").rstrip("/")
    password = os.environ.get("HARBOR_ADMIN_PASSWORD", "")
    if not service_id or not harbor_url or not password:
        return
    auth = (os.environ.get("HARBOR_ADMIN_USER", "admin"), password)
    try:
        repos = httpx.get(f"{harbor_url}/api/v2.0/projects/prod/repositories?page_size=100",
                          auth=auth, timeout=30).json() or []
        for repo in repos:
            name = repo.get("name", "")
            if name.endswith(service_id):
                from urllib.parse import quote
                encoded = quote(quote(name.split("/", 1)[1], safe=""), safe="")
                httpx.delete(f"{harbor_url}/api/v2.0/projects/prod/repositories/{encoded}",
                             auth=auth, timeout=30)
                print(f"  cleaned {name}")
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        print(f"  prod cleanup skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", nargs="*", default=None, help="Row ids to run (default: all).")
    parser.add_argument("--expect", default="live", choices=["live", "provisioning"],
                        help="Terminal state that counts as success (provisioning = no deploy plane).")
    parser.add_argument("--skip-fastsdk", action="store_true", help="Skip fastSDK validation.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-row timeout in seconds.")
    args = parser.parse_args()

    rows = [r for r in MATRIX if args.rows is None or r.row_id in args.rows]
    if not rows:
        print(f"No matching rows. Available: {[r.row_id for r in MATRIX]}")
        return 2

    results = [run_row(row, args.expect, args.skip_fastsdk, args.timeout) for row in rows]

    print("\n=== E2E deploy matrix summary ===")
    print(f"{'row':<8} {'status':<18} {'fastsdk':<9} {'duration':<10} error")
    for r in results:
        print(f"{r.row_id:<8} {r.status or '-':<18} {r.fastsdk:<9} {r.duration_s:>7.0f}s   {r.error or ''}")

    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} rows passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
