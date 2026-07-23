"""One-call serving: standard endpoints for a declared model, then start.

``apipod.serve(model)`` inspects the model's inference surface and registers
the matching standardized (OpenAI-compatible) endpoints, then starts the app.
Model and service stay separate: the same model instance works standalone
(``model.generate(...)``) or served, and custom endpoints can be added by
passing an existing ``APIPod`` app.

Capability detection is method-based (checked on the class, never the
instance, so the lazy-load ``__getattr__`` of :class:`apipod.Model` is not
triggered):

- ``generate(messages, images=...)``  -> POST /chat  (chat with image inputs)
- ``generate(messages)``              -> POST /chat  (text chat)
- ``embed(text=..., image=...)``      -> POST /embeddings (multimodal)
- ``embed_text(text)``                -> POST /embeddings (text)
- ``generate_image(prompt, ...)``     -> POST /images

Custom :class:`apipod.Model` subclasses participate by implementing methods
with these names and signatures.
"""
import inspect
from typing import List, Optional

from pydantic import Field

from apipod.api import APIPod
from apipod.common.schemas import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ImageGenerationRequest,
    MultimodalEmbeddingRequest,
)
from socaity_schemas import ImageFileModel
from apipod.common.settings import APIPOD_HOST, APIPOD_PORT
from apipod.models.model import Model


class VLMChatRequest(ChatCompletionRequest):
    """Chat completion extended with image inputs for vision-language chat."""

    images: Optional[List[ImageFileModel]] = Field(
        default=None,
        description="Images to analyse alongside the messages. Accepts uploads, URLs or base64. "
                    "Attached to the latest user message.",
    )


def _class_method(model: Model, name: str):
    """Resolve a method on the model's class (bypasses Model.__getattr__ lazy load)."""
    method = getattr(type(model), name, None)
    return method if callable(method) else None


def _method_params(method) -> set:
    return set(inspect.signature(method).parameters)


def serve(
    model: Model,
    title: str = None,
    summary: str = None,
    description: str = None,
    port: int = APIPOD_PORT,
    host: str = APIPOD_HOST,
    app: APIPod = None,
    **apipod_kwargs,
) -> None:
    """Register the model's standard endpoints and start the service.

    Args:
        model: A declared :class:`apipod.Model` (preset or custom subclass).
        title: Service title (OpenAPI ``info.title``).
        summary: One-line service summary (OpenAPI ``info.summary``).
        description: Long-form description; becomes the MaaS catalog text.
        port: Port to serve on.
        host: Host to bind.
        app: Existing APIPod app to register into (for extra custom endpoints).
        **apipod_kwargs: Forwarded to ``APIPod()`` when no app is given.
    """
    if app is None:
        app = APIPod(title=title, summary=summary, description=description, **apipod_kwargs)

    registered = register_model_endpoints(app, model)
    if not registered:
        raise ValueError(
            f"{type(model).__name__} exposes no servable methods. Implement one of "
            "generate / embed / embed_text / generate_image (see apipod.serve docs)."
        )
    app.start(port=port, host=host)


def register_model_endpoints(app: APIPod, model: Model) -> List[str]:
    """Register the standard endpoints matching the model's methods. Returns the paths."""
    registered = []

    generate = _class_method(model, "generate")
    if generate is not None:
        supports_images = "images" in _method_params(generate)
        app.endpoint(path="/chat")(_vlm_chat(model) if supports_images else _llm_chat(model))
        registered.append("/chat")

    if _class_method(model, "embed") is not None:
        app.endpoint(path="/embeddings")(_multimodal_embeddings(model))
        registered.append("/embeddings")
    elif _class_method(model, "embed_text") is not None:
        app.endpoint(path="/embeddings")(_text_embeddings(model))
        registered.append("/embeddings")

    if _class_method(model, "generate_image") is not None:
        app.endpoint(path="/images")(_image_generation(model))
        registered.append("/images")

    return registered


# ----------------------------------------------------------------------------
# Endpoint factories. Free functions returning closures so each service only
# carries the handlers its model supports.
# ----------------------------------------------------------------------------

def _llm_chat(model: Model):
    def chat(request: ChatCompletionRequest):
        """Chat completions: multi-turn conversations, instruction following and text generation.

        Send OpenAI-style messages; set stream=true for token-by-token
        server-sent events.
        """
        max_tokens = request.max_tokens or 512
        if request.stream:
            return model.stream(request.messages, request.temperature, max_tokens)
        return model.generate(request.messages, request.temperature, max_tokens)

    return chat


def _vlm_chat(model: Model):
    def chat(request: VLMChatRequest):
        """Chat about images: describe scenes, answer visual questions, extract text.

        Send OpenAI-style messages plus optional images; the model sees them
        with your latest user message and keeps them in context across turns.
        Examples: VQA ("How many people are in this photo?"), OCR ("Extract
        all text from this receipt as markdown"), image description, document
        parsing and chart reading. Set stream=true for server-sent events.
        """
        max_tokens = request.max_tokens or 512
        if request.stream:
            return model.stream(request.messages, request.images, request.temperature, max_tokens)
        return model.generate(request.messages, request.images, request.temperature, max_tokens)

    return chat


def _text_embeddings(model: Model):
    def embeddings(request: EmbeddingRequest):
        """Text embeddings for semantic search, similarity, clustering and RAG."""
        return model.embed_text(request.input)

    return embeddings


def _multimodal_embeddings(model: Model):
    def embeddings(request: MultimodalEmbeddingRequest):
        """Multimodal embeddings: one vector space for text and images.

        Each text input and the image produce one L2-normalized vector,
        tagged with its modality. Search images with text queries (and vice
        versa): semantic image search, deduplication and multimodal RAG.
        """
        texts = [request.input] if isinstance(request.input, str) else list(request.input or [])
        data = [
            {"embedding": model.embed(text=text), "index": index, "modality": "text"}
            for index, text in enumerate(texts)
        ]
        if request.image is not None:
            data.append({"embedding": model.embed(image=request.image), "index": len(data), "modality": "image"})
        return {"data": data}

    return embeddings


def _image_generation(model: Model):
    def images(request: ImageGenerationRequest):
        """Generate images from text prompts.

        Photorealistic scenes, illustrations and bilingual (EN/CN) text
        rendering; control output with size, seed and steps.
        """
        return model.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            size=request.size,
            num_images=request.num_images,
            seed=request.seed,
            steps=request.steps,
        )

    return images
