"""
A) Standardized (OpenAI-compatible) AI service schemas.

1. Auto-assignment: APIPod detects a registered request schema in the function
   signature and automatically assigns the correct response model — the author
   never writes ``response_model=...``. Verified by checking the OpenAPI 200
   response ``$ref`` for each endpoint against the SCHEMA_REGISTRY.
2. Schema extension: a service can subclass a request schema with extra fields;
   those fields are parsed and reachable in the endpoint.
3. Response normalization: a schema endpoint may return the response model
   directly *or* a convenient raw value (string / list / dict / media); both
   normalize into the correct response model.
"""

import pytest

from apipod.common import schemas
from apipod.engine.backend.schema_resolve import SCHEMA_REGISTRY

from conftest import build_service
from services import schema_service
from services.schema_service import CASES
from services import streaming_service


def _response_ref(spec: dict, path: str) -> str:
    """Return the $ref string from the 200 application/json response of a POST endpoint."""
    try:
        return (
            spec["paths"][path]["post"]["responses"]["200"]
            ["content"]["application/json"]["schema"]["$ref"]
        )
    except KeyError:
        return ""


def _request_schema_ref(spec: dict, path: str) -> str:
    """Return the $ref of the JSON request body schema for a POST endpoint."""
    try:
        return spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    except KeyError:
        return ""


def _schema_has_property(spec: dict, schema_ref: str, prop: str) -> bool:
    if not schema_ref.startswith("#/components/schemas/"):
        return False
    name = schema_ref.rsplit("/", 1)[-1]
    properties = spec.get("components", {}).get("schemas", {}).get(name, {}).get("properties", {})
    return prop in properties


# --------------------------------------------------------------------------- #
# 1. Auto-assignment: response model comes from the registry, not the author
# --------------------------------------------------------------------------- #
def test_response_model_auto_assigned_from_registry():
    """The 200 response schema is set to the correct response model for every
    registered schema, even though ``register_all`` never passes ``response_model``
    explicitly — APIPod reads it from SCHEMA_REGISTRY via the schema binding."""
    with build_service(schema_service.register_all) as client:
        spec = client.get("/openapi.json").json()

        for request_model, registry_spec in SCHEMA_REGISTRY.items():
            path = schema_service.tag_path(request_model)
            expected = f"#/components/schemas/{registry_spec.response_model.__name__}"
            actual = _response_ref(spec, path)
            assert actual == expected, (
                f"{request_model.__name__}: expected 200 $ref {expected!r}, got {actual!r}"
            )


def test_non_streaming_schema_accepts_stream_false_in_body():
    """OpenAI clients often send ``stream: false``; validation must not reject it."""
    with build_service(schema_service.register_all) as client:
        resp = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert resp.status_code == 200, resp.text


def test_stream_request_field_on_streaming_schema_endpoint():
    """A schema endpoint detected as streaming keeps ``stream`` in its OpenAPI body."""
    with build_service(streaming_service.register, simulate="serverless") as client:
        spec = client.get("/openapi.json").json()
        chat_ref = _request_schema_ref(spec, "/chat")
        assert _schema_has_property(spec, chat_ref, "stream")


def test_stream_request_field_hidden_on_non_streaming_schema_endpoints():
    """Non-streaming schema endpoints omit ``stream`` from OpenAPI even when the model defines it."""
    with build_service(schema_service.register_all) as client:
        spec = client.get("/openapi.json").json()
        for request_model in SCHEMA_REGISTRY:
            if "stream" not in request_model.model_fields:
                continue
            path = schema_service.tag_path(request_model)
            ref = _request_schema_ref(spec, path)
            assert ref, f"missing OpenAPI request schema for {path}"
            assert not _schema_has_property(spec, ref, "stream"), (
                f"{path} should not expose stream (handler is not streaming)"
            )


# --------------------------------------------------------------------------- #
# 2. Schema extension
# --------------------------------------------------------------------------- #
def test_extended_schema_parses_extra_field():
    """A subclassed request schema carries its extra fields through to the endpoint."""
    with build_service(schema_service.register_extended) as client:
        resp = client.post(
            "/chat-extended",
            json={"messages": [{"role": "user", "content": "ahoy"}], "persona": "captain"},
        )
        assert resp.status_code == 200, resp.text
        body = schemas.ChatCompletionResponse.model_validate(resp.json())
        assert body.choices[0].message.content == "[captain] ahoy"


# --------------------------------------------------------------------------- #
# 3. Response normalization: raw value vs. typed instance
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mapping_client():
    with build_service(schema_service.register_mapping) as c:
        yield c


_IDS = [c.request_model.__name__ for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_raw_value_wraps_into_response_model(mapping_client, case):
    """A raw return value (str / list / dict) is normalized into the response model."""
    tag = SCHEMA_REGISTRY[case.request_model].tag.replace("_", "-")
    resp = mapping_client.post(f"/{tag}-raw", json=case.payload)
    assert resp.status_code == 200, resp.text
    case.response_model.model_validate(resp.json())


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_typed_instance_passes_through(mapping_client, case):
    """A response model instance is serialized unchanged."""
    tag = SCHEMA_REGISTRY[case.request_model].tag.replace("_", "-")
    resp = mapping_client.post(f"/{tag}-typed", json=case.payload)
    assert resp.status_code == 200, resp.text
    case.response_model.model_validate(resp.json())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
