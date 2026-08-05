from __future__ import annotations

import json
from typing import Any

from ..device import resolve_device, torch_device_name
from ..io import sha256_text
from ..schemas import AnnotationRecord, LLMAnnotationPayload
from .base import SYSTEM_PROMPT, AnnotationProvider, ProviderError, normalize_payload, taxonomy_prompt


class HuggingFaceProvider(AnnotationProvider):
    """Fallback local directo para Transformers; carga el modelo de forma diferida."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B",
        *,
        device: str = "auto",
        max_new_tokens: int = 700,
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.hardware = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def probe(self) -> dict[str, Any]:
        try:
            import transformers
        except ImportError:
            transformers = None
        return {
            "provider": "huggingface_local",
            "model": self.model,
            "transformers_installed": transformers is not None,
            "hardware": self.hardware.model_dump(),
            "model_loaded": self._pipeline is not None,
        }

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ProviderError("Instale el extra de entrenamiento para usar Hugging Face") from exc
        device = torch_device_name(self.hardware)
        kwargs: dict[str, Any] = {"model": self.model}
        if device != "cpu":
            kwargs["device"] = device
        self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n{taxonomy_prompt(self.taxonomy)}\n\n"
            f"Esquema: {json.dumps(LLMAnnotationPayload.model_json_schema(), ensure_ascii=False)}\n"
            f"CHUNK_ID: {chunk['chunk_id']}\nTEXTO: {chunk['text']}\nJSON:"
        )
        generator = self._load()
        output = generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"]
        start = output.find("{")
        end = output.rfind("}")
        if start < 0 or end < start:
            raise ProviderError("Hugging Face no devolvió un objeto JSON")
        payload = json.loads(output[start : end + 1])
        return normalize_payload(
            payload,
            text=str(chunk["text"]),
            source="huggingface_local",
            annotator_type="llm_local",
            model=self.model,
            prompt_sha256=sha256_text(prompt),
            taxonomy=self.taxonomy,
        )
