"""Reusable APIPod service definitions for the test suite.

Three services, each exposing a ``register(app)`` callback:

- ``core_service``      parameter shapes, file I/O, JobProgress; also a runnable
                        entrypoint for the fastSDK subprocess tests.
- ``schema_service``    one endpoint per standardized (OpenAI-compatible) schema;
                        schema extension and raw/typed response mapping cases.
- ``streaming_service`` three streaming modes: plain text tokens, binary frames,
                        ChatCompletion SSE deltas.
"""
