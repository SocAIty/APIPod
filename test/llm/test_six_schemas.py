"""
Tests for the six modality schemas added in #179, plus the FRICTION #15 fix
in _BaseLLMMixin._wrap_llm_response.

These tests do not require torch / transformers — they exercise the schema
validation path and the mixin wrap path directly.
"""

import pytest
from pydantic import ValidationError

from apipod.common import schemas
from apipod.engine.llm.base_llm_mixin import _BaseLLMMixin


# ----------------------------------------------------------------------
# Registry / supported tuple
# ----------------------------------------------------------------------

def test_supported_tuple_has_nine_entries():
    """SUPPORTED_LLM_REQUEST_SCHEMAS should now hold 3 original + 6 new = 9."""
    assert len(schemas.SUPPORTED_LLM_REQUEST_SCHEMAS) == 9


def test_supported_tuple_contains_new_request_classes():
    for cls in (
        schemas.ImageGenerationRequest,
        schemas.VideoGenerationRequest,
        schemas.AudioRequest,
        schemas.Generation3DRequest,
        schemas.VisionRequest,
        schemas.MultimodalEmbeddingRequest,
    ):
        assert cls in schemas.SUPPORTED_LLM_REQUEST_SCHEMAS


def test_mixin_configs_has_nine_entries():
    m = _BaseLLMMixin()
    assert len(m._llm_configs) == 9


def test_mixin_configs_maps_new_requests_to_responses():
    m = _BaseLLMMixin()
    expected = {
        schemas.ImageGenerationRequest:       (schemas.ImageGenerationResponse,       "image_generation"),
        schemas.VideoGenerationRequest:       (schemas.VideoGenerationResponse,       "video_generation"),
        schemas.AudioRequest:                 (schemas.AudioResponse,                 "audio"),
        schemas.Generation3DRequest:          (schemas.Generation3DResponse,          "generation_3d"),
        schemas.VisionRequest:                (schemas.VisionResponse,                "vision"),
        schemas.MultimodalEmbeddingRequest:   (schemas.MultimodalEmbeddingResponse,   "embedding_multimodal"),
    }
    for req_cls, expected_cfg in expected.items():
        assert m._llm_configs[req_cls] == expected_cfg


# ----------------------------------------------------------------------
# Schema field defaults and required-field validation
# ----------------------------------------------------------------------

def test_image_generation_request_minimal_valid():
    r = schemas.ImageGenerationRequest(model="flux", prompt="a cat")
    assert r.model == "flux"
    assert r.prompt == "a cat"
    assert r.num_images == 1          # documented default
    assert r.image is None
    assert r.mask is None


def test_image_generation_request_missing_prompt_raises():
    with pytest.raises(ValidationError):
        schemas.ImageGenerationRequest(model="flux")


def test_video_generation_request_defaults():
    r = schemas.VideoGenerationRequest(model="hunyuan", prompt="x")
    assert r.duration_s == 5.0
    assert r.fps == 24


def test_video_generation_request_missing_prompt_raises():
    with pytest.raises(ValidationError):
        schemas.VideoGenerationRequest(model="hunyuan")


def test_audio_request_all_inputs_optional_except_model():
    r = schemas.AudioRequest(model="whisper")
    assert r.text is None
    assert r.audio is None
    assert r.voice is None


def test_generation_3d_request_output_format_default():
    r = schemas.Generation3DRequest(model="triposr")
    assert r.output_format == "glb"


def test_vision_request_requires_image():
    with pytest.raises(ValidationError):
        schemas.VisionRequest(model="clip")


def test_multimodal_embedding_request_all_inputs_optional():
    r = schemas.MultimodalEmbeddingRequest(model="clip")
    assert r.input is None
    assert r.image is None
    assert r.audio is None


# ----------------------------------------------------------------------
# Response shape validation (Literal["..."] guards)
# ----------------------------------------------------------------------

def test_image_generation_response_object_literal():
    r = schemas.ImageGenerationResponse(
        id="imggen-1-abc",
        object="image_generation",
        created=1,
        model="flux",
        data=[schemas.ImageGenerationData(url="http://x/i.png")],
    )
    assert r.object == "image_generation"


def test_image_generation_response_wrong_object_raises():
    with pytest.raises(ValidationError):
        schemas.ImageGenerationResponse(
            id="imggen-1-abc",
            object="wrong",
            created=1,
            model="flux",
            data=[],
        )


def test_video_generation_response_object_literal():
    r = schemas.VideoGenerationResponse(
        id="vidgen-1-abc",
        object="video_generation",
        created=1,
        model="hunyuan",
        data=[schemas.VideoGenerationData(url="http://x/v.mp4")],
    )
    assert r.object == "video_generation"


def test_video_generation_response_wrong_object_raises():
    with pytest.raises(ValidationError):
        schemas.VideoGenerationResponse(
            id="vidgen-1-abc",
            object="wrong",
            created=1,
            model="hunyuan",
            data=[],
        )


def test_audio_response_object_literal():
    r = schemas.AudioResponse(
        id="aud-1-abc",
        object="audio",
        created=1,
        model="whisper",
        data=[schemas.AudioData(text="hola")],
    )
    assert r.object == "audio"


def test_audio_response_wrong_object_raises():
    with pytest.raises(ValidationError):
        schemas.AudioResponse(
            id="aud-1-abc",
            object="wrong",
            created=1,
            model="whisper",
            data=[],
        )


def test_generation_3d_response_object_literal():
    r = schemas.Generation3DResponse(
        id="gen3d-1-abc",
        object="generation_3d",
        created=1,
        model="triposr",
        data=[schemas.Generation3DData(url="http://x/c.glb")],
    )
    assert r.object == "generation_3d"


def test_generation_3d_response_wrong_object_raises():
    with pytest.raises(ValidationError):
        schemas.Generation3DResponse(
            id="gen3d-1-abc",
            object="wrong",
            created=1,
            model="triposr",
            data=[],
        )


def test_vision_response_object_literal():
    r = schemas.VisionResponse(
        id="vis-1-abc",
        object="vision",
        created=1,
        model="clip",
        data=[schemas.VisionData(labels=[])],
    )
    assert r.object == "vision"


def test_vision_response_wrong_object_raises():
    with pytest.raises(ValidationError):
        schemas.VisionResponse(
            id="vis-1-abc",
            object="wrong",
            created=1,
            model="clip",
            data=[],
        )


def test_multimodal_embedding_response_object_is_list():
    """Like EmbeddingResponse: outer object is 'list', each data item is 'embedding'."""
    r = schemas.MultimodalEmbeddingResponse(
        object="list",
        data=[
            schemas.MultimodalEmbeddingData(
                embedding=[0.1, 0.2],
                index=0,
                modality="text",
            )
        ],
        model="clip",
    )
    assert r.object == "list"
    assert r.data[0].object == "embedding"   # default on the data item


# ----------------------------------------------------------------------
# Mixin wrap path — covers FRICTION #15 fix + each new endpoint_type
# ----------------------------------------------------------------------

@pytest.fixture
def mixin():
    return _BaseLLMMixin()


def test_friction_15_embedding_wrap_outer_object_is_list(mixin):
    """
    FRICTION #15 regression. Pre-fix this raised ValidationError because the
    mixin passed object='embedding' to EmbeddingResponse, which declares
    object: Literal['list']. After the fix the wrap path succeeds and the
    outer object is 'list'.
    """
    req = schemas.EmbeddingRequest(model="bge", input="hello")
    wrapped = mixin._wrap_llm_response(
        result={
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
        response_model=schemas.EmbeddingResponse,
        endpoint_type="embedding",
        openai_req=req,
    )
    assert wrapped.object == "list"
    assert wrapped.data[0].object == "embedding"


def test_image_generation_wrap(mixin):
    req = schemas.ImageGenerationRequest(model="flux", prompt="cat")
    wrapped = mixin._wrap_llm_response(
        result={"data": [{"url": "http://x/i.png", "seed": 42}]},
        response_model=schemas.ImageGenerationResponse,
        endpoint_type="image_generation",
        openai_req=req,
    )
    assert wrapped.object == "image_generation"
    assert wrapped.data[0].url == "http://x/i.png"
    assert wrapped.id.startswith("imggen-")


def test_video_generation_wrap(mixin):
    req = schemas.VideoGenerationRequest(model="hunyuan", prompt="x")
    wrapped = mixin._wrap_llm_response(
        result={"data": [{"url": "http://x/v.mp4", "duration_s": 5.0}]},
        response_model=schemas.VideoGenerationResponse,
        endpoint_type="video_generation",
        openai_req=req,
    )
    assert wrapped.object == "video_generation"
    assert wrapped.id.startswith("vidgen-")


def test_audio_wrap(mixin):
    req = schemas.AudioRequest(model="whisper", text="hi")
    wrapped = mixin._wrap_llm_response(
        result={"data": [{"text": "hola", "language": "es"}]},
        response_model=schemas.AudioResponse,
        endpoint_type="audio",
        openai_req=req,
    )
    assert wrapped.object == "audio"
    assert wrapped.data[0].text == "hola"


def test_generation_3d_wrap(mixin):
    req = schemas.Generation3DRequest(model="triposr", prompt="chair")
    wrapped = mixin._wrap_llm_response(
        result={"data": [{"url": "http://x/c.glb", "output_format": "glb"}]},
        response_model=schemas.Generation3DResponse,
        endpoint_type="generation_3d",
        openai_req=req,
    )
    assert wrapped.object == "generation_3d"
    assert wrapped.id.startswith("gen3d-")


# 1x1 transparent PNG used to construct ImageFile without hitting the network.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def test_vision_wrap(mixin):
    """
    Vision branch in _wrap_llm_response had no test before; the elif path was
    never exercised. Covers the branch end-to-end.
    """
    req = schemas.VisionRequest(model="clip", image=_TINY_PNG_B64)
    wrapped = mixin._wrap_llm_response(
        result={
            "data": [
                {
                    "labels": [{"label": "cat", "score": 0.95}],
                    "text": None,
                }
            ]
        },
        response_model=schemas.VisionResponse,
        endpoint_type="vision",
        openai_req=req,
    )
    assert wrapped.object == "vision"
    assert wrapped.id.startswith("vis-")
    assert wrapped.data[0].labels[0].label == "cat"
    assert wrapped.data[0].labels[0].score == 0.95


def test_multimodal_embedding_wrap(mixin):
    req = schemas.MultimodalEmbeddingRequest(model="clip", input="hi")
    wrapped = mixin._wrap_llm_response(
        result={
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2],
                    "index": 0,
                    "modality": "text",
                }
            ]
        },
        response_model=schemas.MultimodalEmbeddingResponse,
        endpoint_type="embedding_multimodal",
        openai_req=req,
    )
    assert wrapped.object == "list"
    assert wrapped.data[0].embedding == [0.1, 0.2]


def test_unknown_endpoint_type_still_raises(mixin):
    req = schemas.ImageGenerationRequest(model="x", prompt="y")
    with pytest.raises(ValueError, match="Unknown endpoint type"):
        mixin._wrap_llm_response(
            result={"data": []},
            response_model=schemas.ImageGenerationResponse,
            endpoint_type="not_a_real_type",
            openai_req=req,
        )
