"""
Minimal APIPod service used to verify media-file integration end to end for
PR #15 (the 6 new Pydantic schemas).

It exposes two endpoints:
- POST /vision      uses VisionRequest, which has a REQUIRED image field
- POST /audio       uses AudioRequest, which has an OPTIONAL audio field

Both endpoints just inspect the incoming media file and echo back a few
properties (size, content type, etc.) so we can confirm the file_handling
mixin actually deserialized URL/base64 input into a real ImageFile/AudioFile.

Run with:
    python examples/test_service_six_schemas.py
The server starts on http://0.0.0.0:8000 with OpenAPI docs at /docs.
"""

from apipod import APIPod
from apipod.common.schemas import (
    VisionRequest,
    AudioRequest,
)


app = APIPod()


def _media_summary(media) -> dict:
    """Pull a couple of safe fields off whatever media_toolkit returned."""
    if media is None:
        return {"present": False}
    return {
        "present": True,
        "type": type(media).__name__,
        "size_bytes": getattr(media, "file_size", None) or getattr(media, "size", None),
        "content_type": getattr(media, "content_type", None),
    }


@app.endpoint("/vision")
def vision(payload: VisionRequest):
    """Echoes a label built from payload.image. The point is that the image
    arrived as an ImageFile, deserialized from upload / base64 / URL."""
    summary = _media_summary(payload.image)
    return {
        "data": [
            {
                "labels": [{"label": f"echo:{summary}", "score": 1.0}],
                "text": None,
            }
        ]
    }


@app.endpoint("/audio")
def audio(payload: AudioRequest):
    """Echoes the audio properties (or text if no audio was sent)."""
    if payload.audio is not None:
        text = f"echo:{_media_summary(payload.audio)}"
    else:
        text = f"no-audio:text={payload.text!r}"
    return {
        "data": [
            {"text": text, "language": payload.language, "duration_s": None}
        ]
    }


if __name__ == "__main__":
    app.start(port=8000, host="0.0.0.0")
