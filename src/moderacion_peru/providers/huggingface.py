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
        max_new_tokens: int = 512,
        retries: int = 1,
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.hardware = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.retries = retries
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
        kwargs: dict[str, Any] = {"model": self.model, "dtype": "auto"}
        if self.hardware.backend in {"cuda", "rocm"}:
            kwargs["device"] = 0
            kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}
        elif device == "xpu":
            kwargs["device"] = "xpu:0"
        self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    def _prompt(self, chunk: dict[str, Any], correction: str | None = None) -> str:
        user = (
            f"{taxonomy_prompt(self.taxonomy)}\n\n"
            f"Esquema: {json.dumps(LLMAnnotationPayload.model_json_schema(), ensure_ascii=False)}\n"
            f"CHUNK_ID: {chunk['chunk_id']}\nTEXTO: {chunk['text']}\nJSON:"
        )
        if correction:
            user += f"\nLa salida anterior fue inválida: {correction}. Corrígela y devuelve solo JSON."
        generator = self._load()
        tokenizer = generator.tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        generator = self._load()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            prompt = self._prompt(chunk, str(last_error) if last_error else None)
            output = generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )[0]["generated_text"]
            try:
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
            except (ValueError, ProviderError) as exc:
                last_error = exc
        raise ProviderError(
            f"Hugging Face falló después de {self.retries + 1} intentos: {last_error}"
        )
