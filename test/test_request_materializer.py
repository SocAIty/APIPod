"""Phase 0: RequestMaterializer binds a live Request without mounting the endpoint."""

from inspect import Parameter, Signature
from typing import List, Optional

from fastapi import Body, FastAPI, Request
from fastapi.testclient import TestClient
from media_toolkit import ImageFile
from socaity_schemas.public.inference.language import ChatCompletionRequest
from socaity_schemas.public.inference.media import FileModel

from apipod.engine.backend.fastapi.request_materializer import RequestMaterializer

_materializer = RequestMaterializer()


def _collector(signature: Signature):
    async def collect(**kwargs):
        return kwargs

    collect.__signature__ = signature
    collect.__name__ = "collect"
    return collect


def _materialize(func, method, **request_kwargs):
    app = FastAPI()
    prepared = _materializer.prepare(func)
    captured: dict = {}

    @app.post("/run")
    async def catch_all(request: Request):
        captured["values"] = await _materializer.materialize(prepared, request)
        return {"ok": True}

    response = TestClient(app).post("/run", **request_kwargs)
    assert response.status_code == 200, response.text
    return captured["values"]


def test_chat_completion_json():
    func = _collector(
        Signature(
            parameters=[
                Parameter(
                    "request",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=Body(...),
                    annotation=ChatCompletionRequest,
                )
            ]
        )
    )
    values = _materialize(
        func,
        "POST",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
    )
    request = values["request"]
    assert isinstance(request, ChatCompletionRequest)
    assert request.messages[0].content == "hello"
    assert request.stream is False


def test_stream_flag_survives_chat_materialize():
    func = _collector(
        Signature(
            parameters=[
                Parameter(
                    "request",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=Body(...),
                    annotation=ChatCompletionRequest,
                )
            ]
        )
    )
    values = _materialize(
        func,
        "POST",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert values["request"].stream is True


def test_filemodel_url_passthrough():
    func = _collector(
        Signature(
            parameters=[
                Parameter(
                    "file",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=Body(...),
                    annotation=FileModel,
                )
            ]
        )
    )
    values = _materialize(
        func,
        "POST",
        json={
            "file_name": "photo.png",
            "content_type": "image/png",
            "content": "https://example.com/photo.png",
        },
    )
    file = values["file"]
    assert isinstance(file, FileModel)
    assert str(file.content) == "https://example.com/photo.png"


def test_empty_file_array():
    func = _collector(
        Signature(
            parameters=[
                Parameter(
                    "images",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=Optional[List[ImageFile]],
                )
            ]
        )
    )
    values = _materialize(func, "POST", json={"images": []})
    assert values["images"] in (None, [])


def test_imagefile_multipart():
    func = _collector(
        Signature(
            parameters=[
                Parameter("image", Parameter.POSITIONAL_OR_KEYWORD, annotation=ImageFile)
            ]
        )
    )
    values = _materialize(
        func,
        "POST",
        files={"image": ("x.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert isinstance(values["image"], ImageFile)
