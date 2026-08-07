from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from ..io import sha256_file, sha256_text
from ..paths import find_project_root
from ..schemas import AnnotationRecord, LLMAnnotationPayload
from .base import SYSTEM_PROMPT, AnnotationProvider, ProviderError, normalize_payload, taxonomy_prompt


_PRICE_USD_PER_MILLION = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


class DeepSeekProvider(AnnotationProvider):
    """DeepSeek V4 con lotes compactos, concurrencia, costo y reanudación externa."""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 240.0,
        retries: int = 1,
        max_workers: int = 32,
        records_per_request: int = 5,
        max_cost_usd: float | None = None,
        label_source: str = "deepseek_remote",
        annotator_type: str = "llm_remote",
        operational_prompt_path: str | Path | None = None,
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.timeout = float(timeout)
        self.retries = int(retries)
        if max_workers < 1 or records_per_request < 1:
            raise ValueError("max_workers y records_per_request deben ser positivos")
        if max_cost_usd is not None and max_cost_usd <= 0:
            raise ValueError("max_cost_usd debe ser positivo o None")
        self.max_workers = int(max_workers)
        self.records_per_request = int(records_per_request)
        self.max_cost_usd = float(max_cost_usd) if max_cost_usd is not None else None
        self.label_source = label_source
        self.annotator_type = annotator_type
        self.operational_prompt_path = (
            Path(operational_prompt_path)
            if operational_prompt_path
            else find_project_root() / "config" / "prompt_operacional_ollama_v2.md"
        )
        if not self.operational_prompt_path.is_file():
            raise FileNotFoundError(
                f"No existe el prompt operacional compacto: {self.operational_prompt_path}"
            )
        self.operational_prompt = self.operational_prompt_path.read_text(
            encoding="utf-8-sig"
        ).strip()
        self.operational_prompt_sha256 = sha256_file(self.operational_prompt_path)
        self._instruction_prompt = (
            f"CONTRATO {self.taxonomy.contract_id} v{self.taxonomy.version}\n"
            f"{taxonomy_prompt(self.taxonomy)}\n"
            f"GUÍA OPERATIVA COMPACTA:\n{self.operational_prompt}\n"
            "Devuelve un objeto JSON con la clave annotations. Su valor debe ser una lista "
            "con exactamente una anotación por registro, en el mismo orden y conservando chunk_id.\n"
            f"ESQUEMA DE CADA ANOTACIÓN:\n"
            f"{json.dumps(LLMAnnotationPayload.model_json_schema(), ensure_ascii=False)}"
        )
        self.prompt_sha256 = sha256_text(SYSTEM_PROMPT + "\n" + self._instruction_prompt)
        self._usage_lock = threading.Lock()
        self._usage = {
            "requests": 0,
            "input_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("Falta DEEPSEEK_API_KEY")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def probe(self) -> dict[str, Any]:
        return {
            "provider": "deepseek_http",
            "base_url": self.base_url,
            "model": self.model,
            "credential_configured": bool(self.api_key),
            "network_called": False,
            "max_workers": self.max_workers,
            "records_per_request": self.records_per_request,
            "max_cost_usd": self.max_cost_usd,
            "label_source": self.label_source,
            "operational_prompt_path": str(self.operational_prompt_path),
            "operational_prompt_sha256": self.operational_prompt_sha256,
            "prompt_sha256": self.prompt_sha256,
            "official_price_usd_per_million": _PRICE_USD_PER_MILLION.get(self.model),
        }

    def validate_connection(self) -> dict[str, Any]:
        """Comprueba credencial y catálogo sin enviar ningún texto del corpus."""

        response = requests.get(
            f"{self.base_url}/models", headers=self.headers, timeout=min(self.timeout, 60)
        )
        response.raise_for_status()
        model_ids = sorted(
            str(row.get("id"))
            for row in response.json().get("data", [])
            if row.get("id")
        )
        return {
            "status": "credential_and_models_verified_no_corpus_sent",
            "configured_model": self.model,
            "model_available": self.model in model_ids,
            "available_models": model_ids,
        }

    def usage_summary(self) -> dict[str, Any]:
        with self._usage_lock:
            usage = dict(self._usage)
        usage["estimated_cost_usd"] = round(float(usage["estimated_cost_usd"]), 6)
        priced_input = int(usage["cache_hit_tokens"]) + int(usage["cache_miss_tokens"])
        usage["cache_hit_rate"] = (
            round(int(usage["cache_hit_tokens"]) / priced_input, 6)
            if priced_input
            else None
        )
        usage["groq_gpt_oss_20b_batch_equivalent_usd"] = round(
            (
                int(usage["input_tokens"]) * 0.0375
                + int(usage["output_tokens"]) * 0.15
            )
            / 1_000_000,
            6,
        )
        usage["max_cost_usd"] = self.max_cost_usd
        usage["budget_exhausted"] = bool(
            self.max_cost_usd is not None
            and float(usage["estimated_cost_usd"]) >= self.max_cost_usd
        )
        return usage

    def _record_usage(self, response_usage: dict[str, Any]) -> None:
        prompt = int(response_usage.get("prompt_tokens") or 0)
        completion = int(response_usage.get("completion_tokens") or 0)
        hit = int(response_usage.get("prompt_cache_hit_tokens") or 0)
        miss = int(response_usage.get("prompt_cache_miss_tokens") or 0)
        if not hit and not miss:
            miss = prompt
        elif hit + miss < prompt:
            miss += prompt - hit - miss
        prices = _PRICE_USD_PER_MILLION.get(self.model)
        cost = 0.0
        if prices:
            cost = (
                hit * prices["cache_hit"]
                + miss * prices["cache_miss"]
                + completion * prices["output"]
            ) / 1_000_000
        with self._usage_lock:
            self._usage["requests"] += 1
            self._usage["input_tokens"] += prompt
            self._usage["cache_hit_tokens"] += hit
            self._usage["cache_miss_tokens"] += miss
            self._usage["output_tokens"] += completion
            self._usage["estimated_cost_usd"] += cost

    def _ensure_budget(self) -> None:
        if self.max_cost_usd is None:
            return
        current = float(self.usage_summary()["estimated_cost_usd"])
        if current >= self.max_cost_usd:
            raise ProviderError(
                f"Presupuesto DeepSeek agotado: US${current:.4f} >= US${self.max_cost_usd:.4f}"
            )

    def _sanitize_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Repara únicamente irregularidades inocuas observadas en la campaña histórica."""

        payload = dict(raw)
        payload["notes"] = str(payload.get("notes") or "")[:160]
        payload["justificacion"] = str(payload.get("justificacion") or "")[:1200]
        flags = list(payload.get("flags") or [])
        known_flags = set(self.taxonomy.flags)
        for field in ("coarse_labels", "fine_labels"):
            cleaned = []
            for value in payload.get(field) or []:
                if value in known_flags:
                    flags.append(value)
                else:
                    cleaned.append(value)
            payload[field] = cleaned
        payload["flags"] = list(dict.fromkeys(flags))
        return payload

    def _request_group(
        self, chunks: list[dict[str, Any]]
    ) -> list[AnnotationRecord | Exception]:
        self._ensure_budget()
        records = [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"]),
                **(
                    {"contexto_anterior": str(chunk["contexto_anterior"])}
                    if chunk.get("contexto_anterior")
                    else {}
                ),
                **(
                    {"contexto_posterior": str(chunk["contexto_posterior"])}
                    if chunk.get("contexto_posterior")
                    else {}
                ),
            }
            for chunk in chunks
        ]
        prompt = self._instruction_prompt + "\nREGISTROS:\n" + json.dumps(
            records, ensure_ascii=False, separators=(",", ":")
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=body,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_json = response.json()
                self._record_usage(response_json.get("usage") or {})
                payload = json.loads(response_json["choices"][0]["message"]["content"])
                raw_annotations = payload.get("annotations") if isinstance(payload, dict) else None
                if raw_annotations is None and len(chunks) == 1 and isinstance(payload, dict):
                    raw_annotations = [payload]
                if not isinstance(raw_annotations, list) or len(raw_annotations) != len(chunks):
                    raise ProviderError(
                        f"DeepSeek devolvió {len(raw_annotations) if isinstance(raw_annotations, list) else 0} "
                        f"anotaciones para {len(chunks)} registros"
                    )
                results: list[AnnotationRecord | Exception] = []
                for raw, chunk in zip(raw_annotations, chunks):
                    try:
                        if not isinstance(raw, dict):
                            raise ProviderError("La anotación del lote no es un objeto JSON")
                        raw = self._sanitize_payload(raw)
                        if str(raw.get("chunk_id") or "") != str(chunk["chunk_id"]):
                            raise ProviderError("DeepSeek cambió el chunk_id o el orden del lote")
                        results.append(
                            normalize_payload(
                                raw,
                                text=str(chunk["text"]),
                                source=self.label_source,
                                annotator_type=self.annotator_type,
                                model=self.model,
                                video_id=str(chunk["video_id"]) if chunk.get("video_id") else None,
                                chunk_metadata=chunk,
                                source_record_sha256=str(
                                    chunk.get("text_sha256") or chunk.get("transcript_sha256") or ""
                                )
                                or None,
                                prompt_sha256=self.prompt_sha256,
                                taxonomy=self.taxonomy,
                            )
                        )
                    except (KeyError, TypeError, ValueError, ProviderError) as exc:
                        results.append(exc)
                return results
            except (requests.RequestException, KeyError, TypeError, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise ProviderError(f"DeepSeek falló después de {self.retries + 1} intentos: {last_error}")

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        result = self._request_group([chunk])[0]
        if isinstance(result, Exception):
            raise ProviderError(str(result))
        return result

    def annotate_batch(
        self, chunks: list[dict[str, Any]]
    ) -> list[AnnotationRecord | Exception]:
        """Procesa grupos de cinco en paralelo y reintenta individualmente los grupos inválidos."""

        if not chunks:
            return []
        groups = [
            chunks[start : start + self.records_per_request]
            for start in range(0, len(chunks), self.records_per_request)
        ]

        def run(group: list[dict[str, Any]]) -> list[AnnotationRecord | Exception]:
            try:
                first_pass = list(self._request_group(group))
            except (ProviderError, ValueError, RuntimeError):
                first_pass = [ProviderError("falló la solicitud agrupada")] * len(group)
            recovered: list[AnnotationRecord | Exception] = []
            for chunk, result in zip(group, first_pass):
                if not isinstance(result, Exception):
                    recovered.append(result)
                    continue
                try:
                    recovered.append(self.annotate(chunk))
                except (ProviderError, ValueError, RuntimeError) as exc:
                    recovered.append(exc)
            return recovered

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(groups))) as executor:
            nested = list(executor.map(run, groups))
        return [result for group_results in nested for result in group_results]
