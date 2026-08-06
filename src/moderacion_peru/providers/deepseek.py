from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from ..io import sha256_text
from ..schemas import AnnotationRecord, LLMAnnotationPayload
from .base import SYSTEM_PROMPT, AnnotationProvider, ProviderError, normalize_payload, taxonomy_prompt


class DeepSeekProvider(AnnotationProvider):
    """Adaptador remoto opcional. Nunca realiza llamadas durante un preflight."""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 240.0,
        retries: int = 1,
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.timeout = timeout
        self.retries = retries

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
        }

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        prompt = (
            f"CONTRATO {self.taxonomy.contract_id} v{self.taxonomy.version}\n"
            f"{taxonomy_prompt(self.taxonomy)}\n"
            f"JSON SCHEMA:\n{json.dumps(LLMAnnotationPayload.model_json_schema(), ensure_ascii=False)}\n"
            f"CHUNK_ID: {chunk['chunk_id']}\nTEXTO:\n{chunk['text']}"
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
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
                payload = json.loads(response.json()["choices"][0]["message"]["content"])
                return normalize_payload(
                    payload,
                    text=str(chunk["text"]),
                    source="deepseek_remote",
                    annotator_type="llm_remote",
                    model=self.model,
                    video_id=str(chunk["video_id"]) if chunk.get("video_id") else None,
                    chunk_metadata=chunk,
                    source_record_sha256=str(chunk.get("text_sha256") or chunk.get("transcript_sha256") or "") or None,
                    prompt_sha256=sha256_text(SYSTEM_PROMPT + "\n" + prompt),
                    taxonomy=self.taxonomy,
                )
            except (requests.RequestException, KeyError, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise ProviderError(f"DeepSeek falló después de {self.retries + 1} intentos: {last_error}")
