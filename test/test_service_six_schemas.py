"""
Minimal APIPod service used to verify media-file integration of the
standardized schemas end to end.

It exposes two endpoints:
- POST /vision        uses VisionRequest, which has a REQUIRED image field
- POST /transcribe    uses TranscriptionRequest, which has a REQUIRED audio field

Both endpoints just inspect the incoming media file and echo back a few
properties (size, content type, etc.) so we can confirm the file handling
actually deserialized upload/URL/base64 input into a real ImageFile/AudioFile.

Run with:
    python examples/test_service_six_schemas.py
The server starts on http://0.0.0.0:8000 with OpenAPI docs at /docs.
"""

from apipod import APIPod
from apipod.common.schemas import TranscriptionRequest, VisionRequest


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
def vision(request: VisionRequest):
    """Echoes a label built from request.image. The point is that the image
    arrived as an ImageFile, deserialized from upload / base64 / URL."""
    summary = _media_summary(request.image)
    return {
        "data": [
            {
                "labels": [{"label": f"echo:{summary}", "score": 1.0}],
                "text": None,
            }
        ]
    }


@app.endpoint("/transcribe")
def transcribe(request: TranscriptionRequest):
    """Echoes the audio properties as the 'transcript'."""
    return {
        "text": f"echo:{_media_summary(request.audio)}",
        "language": request.language,
    }


if __name__ == "__main__":
    app.start(port=8000, host="0.0.0.0")
