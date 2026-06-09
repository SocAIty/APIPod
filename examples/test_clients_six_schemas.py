"""
Exercises the test service in three ways, as Matthias asked on PR #15:
    1. fastSDK client
    2. HTTP request with base64
    3. HTTP request with URL

Run the service first in another shell:
    python examples/test_service_six_schemas.py

Then run this:
    python examples/test_clients_six_schemas.py
"""

import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
# A small public PNG used for the "URL" test. Picked because it doesn't need
# auth and returns a real PNG header (so media_toolkit can sniff it).
PUBLIC_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/120px-PNG_transparency_demonstration_1.png"


def _make_local_png_bytes() -> bytes:
    """Build a small but real-sized PNG (32x32 solid red) in-process.

    media_toolkit's base64 sniffer needs a payload long enough to be confident
    it is base64 and not a short identifier-looking string. The original 1x1
    PNG (67 bytes / 92 b64 chars) sat below the heuristic; 32x32 (~100 bytes /
    ~140 b64 chars) is comfortably above it and matches realistic JSON callers.
    """
    import struct
    import zlib

    width = height = 32
    color = (255, 0, 0)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(color * width)
    idat = zlib.compress(raw)

    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
        + struct.pack(">I", len(idat)) + b"IDAT" + idat + struct.pack(">I", zlib.crc32(b"IDAT" + idat))
        + struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
    )


def _print_result(label: str, status: int, body) -> None:
    print(f"\n=== {label} ===")
    print(f"HTTP {status}")
    print(json.dumps(body, indent=2) if isinstance(body, (dict, list)) else body)


# -------------------------------------------------------------------
# 1. HTTP with base64
# -------------------------------------------------------------------
def test_http_base64() -> bool:
    png = _make_local_png_bytes()
    b64 = base64.b64encode(png).decode("ascii")
    payload = {
        "model": "test-model",
        "image": b64,
    }
    r = requests.post(f"{BASE_URL}/vision", json=payload, timeout=15)
    try:
        body = r.json()
    except Exception:
        body = r.text
    _print_result("HTTP base64", r.status_code, body)
    return r.status_code == 200


# -------------------------------------------------------------------
# 2. HTTP with URL
# -------------------------------------------------------------------
def test_http_url() -> bool:
    payload = {
        "model": "test-model",
        "image": PUBLIC_IMAGE_URL,
    }
    r = requests.post(f"{BASE_URL}/vision", json=payload, timeout=30)
    try:
        body = r.json()
    except Exception:
        body = r.text
    _print_result("HTTP URL", r.status_code, body)
    return r.status_code == 200


# -------------------------------------------------------------------
# 3. fastSDK client
# -------------------------------------------------------------------
def test_fastsdk() -> bool:
    """
    Hits /vision through fastSDK.

    Note: fastSDK's public entry points (`FastClient`, `create_sdk`,
    `APISeex`) all expect either a service registered in the Registry
    or a generated client class. There is no ad-hoc "given this URL,
    just POST this payload" path. For the purposes of PR #15 this means
    a single localhost service like the one in this example is not the
    intended fastSDK use case; the right shape would be to register
    the service first and then call it through the SDK.

    For now we just dispatch a low-level APISeex with a minimal
    ServiceDefinition and report whatever happens.
    """
    try:
        from fastsdk import APISeex, ImageFile
        from apipod_registry.definitions.service_definitions import (
            ServiceDefinition, EndpointDefinition, EndpointParameter,
            ServiceAddress,
        )
    except Exception as exc:
        _print_result("fastSDK", -1, f"import failed: {exc.__class__.__name__}: {exc}")
        return False

    try:
        service = ServiceDefinition(
            id="test-svc",
            display_name="test svc",
            description="local test service",
            service_address=ServiceAddress(url=BASE_URL),
            endpoints=[],
        )
        endpoint = EndpointDefinition(
            id="vision",
            path="/vision",
            display_name="vision",
            description="vision echo",
            parameters=[
                EndpointParameter(
                    name="model", type="string", required=True,
                    location="body", default="test-model",
                ),
                EndpointParameter(
                    name="image", type="image", required=True,
                    location="body",
                ),
            ],
        )
        png = _make_local_png_bytes()
        img = ImageFile().from_bytes(png)
        seex = APISeex(
            service_def=service,
            endpoint_def=endpoint,
            data={"model": "test-model", "image": img},
        )
        result = seex.wait_for_result(timeout_s=15)
        if seex.error:
            _print_result("fastSDK", -1, f"server returned error: {seex.error}")
            return False
        if result is None:
            _print_result("fastSDK", -1, "no result returned (request likely not dispatched; needs Registry setup)")
            return False
        _print_result("fastSDK", 200, str(result))
        return True
    except Exception as exc:
        _print_result("fastSDK", -1, f"call failed: {exc.__class__.__name__}: {exc}")
        return False


def main() -> int:
    results = {
        "HTTP base64": test_http_base64(),
        "HTTP URL": test_http_url(),
        "fastSDK": test_fastsdk(),
    }
    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
