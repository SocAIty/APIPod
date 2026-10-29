"""Body() opts a custom model into JSON. A bare model stays form fields."""

from fastapi import Body
from pydantic import BaseModel

from apipod.engine.backend.runpod.router import SocaityRunpodRouter
from apipod.engine.signatures.json_body import JSON_BODY_NAME_EXTENSION
from conftest import build_service


class MoreParams(BaseModel):
    pam1: str
    pam2: int = 42


def _register(app):
    @app.endpoint("/order")
    def order(order: MoreParams):
        return {"pam1": order.pam1, "pam2": order.pam2}

    @app.endpoint("/json-order")
    def json_order(order: MoreParams = Body()):
        return {"pam1": order.pam1, "pam2": order.pam2}


def _operation(schema, path):
    return schema["paths"][path]["post"]


def test_body_publishes_json_and_keeps_the_parameter_name():
    with build_service(_register) as client:
        schema = client.get("/openapi.json").json()

    form_op = _operation(schema, "/order")
    json_op = _operation(schema, "/json-order")

    assert "application/x-www-form-urlencoded" in form_op["requestBody"]["content"]
    assert JSON_BODY_NAME_EXTENSION not in form_op

    assert "application/json" in json_op["requestBody"]["content"]
    assert json_op[JSON_BODY_NAME_EXTENSION] == "order"


def test_body_accepts_json_and_bare_model_accepts_form():
    with build_service(_register) as client:
        form = client.post("/order", data={"pam1": "burger", "pam2": 2})
        form_as_json = client.post("/order", json={"pam1": "burger", "pam2": 2})
        as_json = client.post("/json-order", json={"pam1": "burger", "pam2": 2})
        json_as_form = client.post("/json-order", data={"pam1": "burger", "pam2": 2})

    assert form.status_code == 200, form.text
    assert form.json() == {"pam1": "burger", "pam2": 2}
    assert form_as_json.status_code == 422

    assert as_json.status_code == 200, as_json.text
    assert as_json.json() == {"pam1": "burger", "pam2": 2}
    assert json_as_form.status_code == 422


def test_runpod_openapi_stamps_the_same_json_body_name():
    app = SocaityRunpodRouter(title="json-body")
    _register(app)
    schema = app.get_openapi_schema()

    form_op = _operation(schema, "/order")
    json_op = _operation(schema, "/json-order")
    assert "application/x-www-form-urlencoded" in form_op["requestBody"]["content"]
    assert JSON_BODY_NAME_EXTENSION not in form_op
    assert "application/json" in json_op["requestBody"]["content"]
    assert json_op[JSON_BODY_NAME_EXTENSION] == "order"

