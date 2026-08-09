from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..device import resolve_device, torch_device_name
from ..io import sha256_file, sha256_text
from ..paths import operational_prompt_path as default_operational_prompt_path
from ..schemas import AnnotationRecord
from .base import SYSTEM_PROMPT, AnnotationProvider, ProviderError, normalize_payload, taxonomy_prompt


class HuggingFaceProvider(AnnotationProvider):
    """Fallback local directo para Transformers; carga el modelo de forma diferida."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B",
        *,
        revision: str = "1cfa9a7208912126459214e8b04321603b3df60c",
        device: str = "auto",
        max_new_tokens: int = 256,
        retries: int = 1,
        records_per_request: int = 5,
        inference_batch_size: int = 4,
        operational_prompt_path: str | Path | None = None,
        label_source: str = "huggingface_local",
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.revision = revision
        self.hardware = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.retries = retries
        if records_per_request < 1 or inference_batch_size < 1:
            raise ValueError("records_per_request e inference_batch_size deben ser positivos")
        self.records_per_request = int(records_per_request)
        self.inference_batch_size = int(inference_batch_size)
        self.label_source = str(label_source)
        self.operational_prompt_path = (
            Path(operational_prompt_path)
            if operational_prompt_path
            else default_operational_prompt_path()
        )
        if not self.operational_prompt_path.is_file():
            raise FileNotFoundError(
                f"No existe el prompt operacional compacto: {self.operational_prompt_path}"
            )
        self.operational_prompt = self.operational_prompt_path.read_text(
            encoding="utf-8-sig"
        ).strip()
        self.operational_prompt_sha256 = sha256_file(self.operational_prompt_path)
        self._pipeline = None

    def probe(self) -> dict[str, Any]:
        try:
            import transformers
        except ImportError:
            transformers = None
        return {
            "provider": "huggingface_local",
            "model": self.model,
            "revision": self.revision,
            "transformers_installed": transformers is not None,
            "hardware": self.hardware.model_dump(),
            "model_loaded": self._pipeline is not None,
            "records_per_request": self.records_per_request,
            "inference_batch_size": self.inference_batch_size,
            "operational_prompt_path": str(self.operational_prompt_path),
            "operational_prompt_sha256": self.operational_prompt_sha256,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "output_contract": {
                "root_key": "annotations",
                "preserves_order": True,
                "preserves_chunk_id": True,
                "validated_with": "LLMAnnotationPayload",
            },
            "label_source": self.label_source,
        }

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ProviderError("Instale el extra de entrenamiento para usar Hugging Face") from exc
        device = torch_device_name(self.hardware)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "revision": self.revision,
            "dtype": "auto",
        }
        if self.hardware.backend in {"cuda", "rocm"}:
            kwargs["device"] = 0
            kwargs["model_kwargs"] = {"attn_implementation": "sdpa"}
        elif device == "xpu":
            kwargs["device"] = "xpu:0"
        self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    def _chat_prompt(self, user: str) -> str:
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

    def _authority(self) -> str:
        return (
            f"CONTRATO {self.taxonomy.contract_id} v{self.taxonomy.version}\n\n"
            f"{taxonomy_prompt(self.taxonomy)}\n\n"
            f"GUÍA OPERATIVA COMPACTA:\n{self.operational_prompt}"
        )

    @staticmethod
    def _compact_schema() -> str:
        return (
            "Cada anotación contiene exactamente chunk_id, coarse_labels, fine_labels, "
            "flags, needs_review, notes, score_confianza y justificacion. "
            "Devuelve solo JSON, sin markdown."
        )

    def _prompt(self, chunk: dict[str, Any], correction: str | None = None) -> str:
        context = ""
        if chunk.get("contexto_anterior"):
            context += f"\nCONTEXTO_ANTERIOR: {chunk['contexto_anterior']}"
        if chunk.get("contexto_posterior"):
            context += f"\nCONTEXTO_POSTERIOR: {chunk['contexto_posterior']}"
        user = (
            f"{self._authority()}\n\n{self._compact_schema()}\n"
            f"CHUNK_ID: {chunk['chunk_id']}\nTEXTO: {chunk['text']}{context}\nJSON:"
        )
        if correction:
            user += f"\nLa salida anterior fue inválida: {correction}. Corrígela y devuelve solo JSON."
        return self._chat_prompt(user)

    def _batch_prompt(self, chunks: list[dict[str, Any]]) -> str:
        records = []
        for chunk in chunks:
            item = {"chunk_id": chunk["chunk_id"], "text": chunk["text"]}
            for key in ("contexto_anterior", "contexto_posterior"):
                if chunk.get(key):
                    item[key] = chunk[key]
            records.append(item)
        user = (
            f"{self._authority()}\n\n{self._compact_schema()}\n"
            "Clasifica independientemente los registros. Conserva el orden y cada chunk_id. "
            "Devuelve exactamente {\"annotations\":[...]} con una anotación por registro.\n"
            f"REGISTROS: {json.dumps(records, ensure_ascii=False)}\nJSON:"
        )
        return self._chat_prompt(user)

    @staticmethod
    def _extract_json(output: str) -> dict[str, Any]:
        start = output.find("{")
        end = output.rfind("}")
        if start < 0 or end < start:
            raise ProviderError("Hugging Face no devolvió un objeto JSON")
        payload = json.loads(output[start : end + 1])
        if not isinstance(payload, dict):
            raise ProviderError("Hugging Face no devolvió un objeto JSON")
        return payload

    def _generate(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
        generator = self._load()
        generated = generator(
            prompts,
            batch_size=min(self.inference_batch_size, len(prompts)),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if len(prompts) == 1 and generated and isinstance(generated[0], dict):
            generated = [generated]
        if len(generated) != len(prompts):
            raise ProviderError(
                f"Hugging Face devolvió {len(generated)} respuestas para {len(prompts)} prompts"
            )
        return [str(group[0]["generated_text"]) for group in generated]

    def _normalize(
        self, payload: dict[str, Any], chunk: dict[str, Any], prompt: str
    ) -> AnnotationRecord:
        return normalize_payload(
            payload,
            text=str(chunk["text"]),
            source=self.label_source,
            annotator_type="llm_local",
            model=self.model,
            video_id=str(chunk["video_id"]) if chunk.get("video_id") else None,
            chunk_metadata=chunk,
            source_record_sha256=str(
                chunk.get("text_sha256") or chunk.get("transcript_sha256") or ""
            )
            or None,
            prompt_sha256=sha256_text(prompt),
            taxonomy=self.taxonomy,
        )

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            prompt = self._prompt(chunk, str(last_error) if last_error else None)
            try:
                output = self._generate([prompt], max_new_tokens=self.max_new_tokens)[0]
                return self._normalize(self._extract_json(output), chunk, prompt)
            except (ValueError, ProviderError) as exc:
                last_error = exc
        raise ProviderError(
            f"Hugging Face falló después de {self.retries + 1} intentos: {last_error}"
        )

    def annotate_batch(
        self, chunks: list[dict[str, Any]]
    ) -> list[AnnotationRecord | Exception]:
        if not chunks:
            return []
        groups = [
            chunks[start : start + self.records_per_request]
            for start in range(0, len(chunks), self.records_per_request)
        ]
        prompts = [self._batch_prompt(group) for group in groups]
        token_limit = 64 + self.max_new_tokens * max(len(group) for group in groups)
        try:
            outputs = self._generate(prompts, max_new_tokens=token_limit)
        except (ValueError, ProviderError, RuntimeError) as exc:
            return [exc for _ in chunks]

        by_id: dict[str, AnnotationRecord | Exception] = {}
        for group, prompt, output in zip(groups, prompts, outputs):
            expected = [str(chunk["chunk_id"]) for chunk in group]
            try:
                wrapper = self._extract_json(output)
                annotations = wrapper.get("annotations")
                if not isinstance(annotations, list) or len(annotations) != len(group):
                    raise ProviderError("El wrapper annotations no conserva el tamaño del lote")
                received = [str(row.get("chunk_id")) for row in annotations if isinstance(row, dict)]
                if received != expected:
                    raise ProviderError("El wrapper annotations no conserva orden y chunk_id")
                for chunk, payload in zip(group, annotations):
                    try:
                        by_id[str(chunk["chunk_id"])] = self._normalize(payload, chunk, prompt)
                    except (ValueError, ProviderError) as exc:
                        by_id[str(chunk["chunk_id"])] = exc
            except (ValueError, ProviderError) as exc:
                for chunk in group:
                    by_id[str(chunk["chunk_id"])] = exc

        # Un fallo de formato en un lote no invalida las demás filas: se reintenta
        # solo esa entrada con el esquema individual.
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            if isinstance(by_id.get(chunk_id), Exception):
                try:
                    by_id[chunk_id] = self.annotate(chunk)
                except (ProviderError, ValueError, RuntimeError) as exc:
                    by_id[chunk_id] = exc
        return [by_id[str(chunk["chunk_id"])] for chunk in chunks]

    def unload(self) -> None:
        """Libera el modelo para alternar primera pasada y revisor en una sola GPU."""

        self._pipeline = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return
