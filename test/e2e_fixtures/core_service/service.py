"""Minimal CPU-only APIPod service for deploy pipeline E2E tests.

Small on purpose: the E2E suite tests the deploy pipeline (build, push,
promote, provision), not model inference.
"""
from apipod import APIPod

app = APIPod()  # run intent injected via env, title comes from apipod.json


@app.endpoint("/echo")
def echo(text: str) -> str:
    return text


@app.endpoint("/add")
def add(a: int, b: int) -> int:
    return a + b
