"""Vision-language preset. Loads a VLM or falls back to a causal LM."""
from __future__ import annotations

from typing import List, Optional

from apipod.models.transformers.base import Transformers
from media_toolkit.utils.dependency_requirements import requires


# Auto classes tried in coverage order: ImageTextToText covers Qwen-VL and most
# open VLMs; MultimodalLM covers encoder-free unified models (e.g. Gemma 4).
_VLM_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")


def to_pil_image(image):
    """Convert a media-toolkit ImageFile, bytes, path/URL string or PIL image to RGB PIL."""
    from PIL import Image
    import io
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if hasattr(image, "to_bytes"):  # media-toolkit MediaFile / ImageFile
        return Image.open(io.BytesIO(image.to_bytes())).convert("RGB")
    return Image.open(image).convert("RGB")


class VLM(Transformers):
    """Transformers chat preset for text and vision-language checkpoints.

    ``load()`` tries VLM auto classes, then ``AutoModelForCausalLM``.
    ``generate``/``stream`` accept optional images. ``embed`` is multimodal
    when a processor is present; ``embed_text`` is always available.
    """

    default_embed_instruction = "Represent the user's input."

    @requires("transformers")
    def load(self) -> None:
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        path = str(self.weights.path)
        last_error: Optional[Exception] = None
        for class_name in _VLM_AUTO_CLASSES:
            auto_cls = getattr(transformers, class_name, None)
            if auto_cls is None:
                continue
            try:
                self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                self.net = auto_cls.from_pretrained(path, **self._from_pretrained_kwargs())
                return
            except ValueError as error:
                last_error = error
                self.processor = None

        self.processor = None
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.net = AutoModelForCausalLM.from_pretrained(path, **self._from_pretrained_kwargs())
        except Exception as error:
            raise ValueError(
                f"No transformers auto class can load {self.weights.ref!r}. "
                f"Tried {', '.join(_VLM_AUTO_CLASSES)} and AutoModelForCausalLM. "
                f"Last error: {last_error or error}"
            ) from error

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "ping"}], max_tokens=1)

    def _chat_inputs(self, messages, images=None, tools=None):
        if getattr(self, "processor", None) is None:
            return super()._chat_inputs(messages, images, tools)
        inputs = self._apply_chat_template(
            self.processor,
            self._conversation(messages, images),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return self._inputs_to_device(inputs)

    @staticmethod
    def _content_parts(content) -> List[dict]:
        """OpenAI content (str or typed parts) -> transformers chat-template parts."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        parts = []
        for part in content or []:
            if part.get("type") == "image_url":
                parts.append({"type": "image", "image": part["image_url"]["url"]})
            else:
                parts.append(part)
        return parts

    def _conversation(self, messages, images) -> List[dict]:
        """OpenAI-style messages + separate image list -> transformers chat format.

        Images are attached before the text of the last user message, the
        placement VLMs are trained on.
        """
        conversation = [
            {"role": message["role"], "content": self._content_parts(message.get("content"))}
            for message in self._normalize_messages(messages)
        ]
        rgb_images = [to_pil_image(image) for image in images or []]
        if rgb_images:
            last_user = next((m for m in reversed(conversation) if m["role"] == "user"), None)
            if last_user is None:
                last_user = {"role": "user", "content": []}
                conversation.append(last_user)
            last_user["content"][:0] = [{"type": "image", "image": img} for img in rgb_images]
        return conversation

    def embed(self, text: Optional[str] = None, image=None, instruction: Optional[str] = None) -> List[float]:
        """One L2-normalized embedding vector for a text and/or image input."""
        if getattr(self, "processor", None) is None:
            if image is not None:
                raise ValueError(f"{self.weights.ref} does not accept image embeddings.")
            if not text:
                raise ValueError("embed() needs a text input.")
            return self.embed_text(text)

        import torch

        content = []
        if image is not None:
            content.append({"type": "image", "image": to_pil_image(image)})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            raise ValueError("embed() needs a text and/or an image input.")

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_embed_instruction}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = self._inputs_to_device(inputs)

        with torch.no_grad():
            hidden = self.net(**inputs, output_hidden_states=True).hidden_states[-1]
        vector = torch.nn.functional.normalize(hidden[0, -1], p=2, dim=-1)
        return vector.float().cpu().tolist()
