"""Vision-language preset: image+text chat and multimodal embeddings."""
from __future__ import annotations

import io
from typing import Iterator, List, Optional

from socaity_cli import requires

from apipod.models.transformers.base import Transformers

# Auto classes tried in coverage order: ImageTextToText covers Qwen-VL and most
# open VLMs; MultimodalLM covers encoder-free unified models (e.g. Gemma 4).
_VLM_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForMultimodalLM")


def to_pil_image(image):
    """Convert a media-toolkit ImageFile, bytes, path/URL string or PIL image to RGB PIL."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if hasattr(image, "to_bytes"):  # media-toolkit MediaFile / ImageFile
        return Image.open(io.BytesIO(image.to_bytes())).convert("RGB")
    return Image.open(image).convert("RGB")


class TransformersVLM(Transformers):
    """Built-in vision-language preset (Qwen-VL, Gemma, ... via transformers auto classes).

    ``generate``/``stream`` run image+text chat through the model's chat
    template; ``embed`` pools the last hidden state into one vector per
    (text, image) input (last-token pooling + L2 norm, the common VLM
    embedding recipe).
    """

    default_embed_instruction = "Represent the user's input."

    @requires("transformers", cli=False)
    def load(self) -> None:
        from transformers import AutoProcessor

        path = str(self.weights.path)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.net = self._load_net(path)

    def _load_net(self, path: str):
        import transformers

        last_error: Optional[Exception] = None
        for class_name in _VLM_AUTO_CLASSES:
            auto_cls = getattr(transformers, class_name, None)
            if auto_cls is None:
                continue
            try:
                return auto_cls.from_pretrained(path, **self._from_pretrained_kwargs())
            except ValueError as error:  # config not in this auto class mapping
                last_error = error
        raise ValueError(
            f"No transformers VLM auto class can load {self.weights.ref!r}. "
            f"Tried {', '.join(_VLM_AUTO_CLASSES)}. Last error: {last_error}"
        )

    def warmup(self) -> None:
        self.generate([{"role": "user", "content": "ping"}], max_tokens=1)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def generate(self, messages, images=None, temperature: float = 0.7, max_tokens: int = 512) -> str:
        inputs = self._chat_inputs(messages, images)
        output = self.net.generate(**inputs, **self._generation_kwargs(temperature, max_tokens))
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages, images=None, temperature: float = 0.7, max_tokens: int = 512) -> Iterator[str]:
        inputs = self._chat_inputs(messages, images)
        yield from self._stream_tokens(
            self.processor.tokenizer,
            dict(**inputs, **self._generation_kwargs(temperature, max_tokens)),
        )

    def _chat_inputs(self, messages, images):
        inputs = self.processor.apply_chat_template(
            self._conversation(messages, images),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        return inputs.to(self.net.device)

    def _conversation(self, messages, images) -> List[dict]:
        """OpenAI-style messages + separate image list -> transformers chat format.

        Images are attached before the text of the last user message, the
        placement VLMs are trained on.
        """
        conversation = []
        for message in self._normalize_messages(messages):
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            conversation.append({"role": message["role"], "content": content})

        pil_images = [to_pil_image(image) for image in images or []]
        if pil_images:
            last_user = next((m for m in reversed(conversation) if m["role"] == "user"), None)
            if last_user is None:
                last_user = {"role": "user", "content": []}
                conversation.append(last_user)
            last_user["content"][:0] = [{"type": "image", "image": img} for img in pil_images]
        return conversation

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: Optional[str] = None, image=None, instruction: Optional[str] = None) -> List[float]:
        """One L2-normalized embedding vector for a text and/or image input."""
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
        inputs = inputs.to(self.net.device)

        with torch.no_grad():
            hidden = self.net(**inputs, output_hidden_states=True).hidden_states[-1]
        # Single unpadded sequence: the last position is the last real token.
        vector = torch.nn.functional.normalize(hidden[0, -1], p=2, dim=-1)
        return vector.float().cpu().tolist()
