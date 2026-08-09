from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from ..io import sha256_file, sha256_text
from ..paths import find_project_root
from ..paths import operational_prompt_path as default_operational_prompt_path
from ..schemas import AnnotationRecord, LLMAnnotationPayload
from .base import (
    SYSTEM_PROMPT,
    AnnotationProvider,
    ProviderError,
    normalize_payload,
    taxonomy_prompt,
)


class OllamaProvider(AnnotationProvider):
    """Proveedor oficial local mediante la API HTTP de Ollama."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        *,
        base_url: str | None = None,
        timeout: float = 240.0,
        retries: int = 1,
        think: bool = False,
        seed: int = 20260805,
        operational_prompt_path: str | Path | None = None,
        taxonomy=None,
    ) -> None:
        super().__init__(model, taxonomy)
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.seed = int(seed)
        # La clasificación estructurada no necesita una cadena de razonamiento
        # extensa. Desactivarla reduce latencia y evita agotar el timeout en CPU.
        self.think = think
        self.operational_prompt_path = (
            Path(operational_prompt_path)
            if operational_prompt_path
            else default_operational_prompt_path()
        )
        if not self.operational_prompt_path.is_file():
            raise FileNotFoundError(
                f"No existe la guía operativa de Ollama: {self.operational_prompt_path}"
            )
        self.operational_prompt = self.operational_prompt_path.read_text(
            encoding="utf-8-sig"
        ).strip()
        self.operational_prompt_sha256 = sha256_file(self.operational_prompt_path)

    def probe(self) -> dict[str, Any]:
        try:
            version = requests.get(f"{self.base_url}/api/version", timeout=5).json()
            tags_response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            tags_response.raise_for_status()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(
                f"Ollama no responde en {self.base_url}: {exc}"
            ) from exc
        tags = tags_response.json()
        model_rows = tags.get("models", [])
        models = [item.get("name") or item.get("model") for item in model_rows]
        selected = next(
            (
                item
                for item in model_rows
                if (item.get("name") or item.get("model")) == self.model
            ),
            None,
        )
        try:
            prompt_path_for_report = (
                self.operational_prompt_path.resolve()
                .relative_to(find_project_root().resolve())
                .as_posix()
            )
        except ValueError:
            prompt_path_for_report = str(self.operational_prompt_path)
        return {
            "provider": "ollama_http",
            "base_url": self.base_url,
            "version": version.get("version"),
            "model": self.model,
            "model_available": self.model in models,
            "model_digest": selected.get("digest") if selected else None,
            "model_details": selected.get("details") if selected else None,
            "operational_prompt_path": prompt_path_for_report,
            "operational_prompt_sha256": self.operational_prompt_sha256,
            "seed": self.seed,
            "models": models,
        }

    def _prompt(self, chunk: dict[str, Any]) -> str:
        return (
            f"CONTRATO {self.taxonomy.contract_id} v{self.taxonomy.version}\n\n"
            f"{taxonomy_prompt(self.taxonomy)}\n\n"
            f"FLAGS PERMITIDOS: {', '.join(self.taxonomy.flags)}\n\n"
            f"GUÍA OPERATIVA VIGENTE:\n{self.operational_prompt}\n\n"
            f"CHUNK_ID: {chunk['chunk_id']}\nTEXTO:\n{chunk['text']}"
        )

    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        if not chunk.get("chunk_id") or not str(chunk.get("text", "")).strip():
            raise ValueError("El chunk requiere chunk_id y texto")
        prompt = self._prompt(chunk)
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        body = {
            "model": self.model,
            "messages": list(base_messages),
            "stream": False,
            "think": self.think,
            "format": LLMAnnotationPayload.model_json_schema(),
            "options": {
                "temperature": 0,
                "seed": self.seed,
                # El JSON esperado es corto. El límite evita que una respuesta
                # inválida siga generándose después del timeout del cliente.
                "num_predict": 512,
            },
            "keep_alive": "10m",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            content = ""
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat", json=body, timeout=self.timeout
                )
                response.raise_for_status()
                envelope = response.json()
                content = envelope.get("message", {}).get("content", "")
                payload = json.loads(content)
                prompt_material = "\n".join(
                    f"{message['role']}: {message['content']}"
                    for message in body["messages"]
                )
                return normalize_payload(
                    payload,
                    text=str(chunk["text"]),
                    source="ollama_local",
                    annotator_type="llm_local",
                    model=self.model,
                    video_id=str(chunk["video_id"]) if chunk.get("video_id") else None,
                    chunk_metadata=chunk,
                    source_record_sha256=str(
                        chunk.get("text_sha256") or chunk.get("transcript_sha256") or ""
                    )
                    or None,
                    prompt_sha256=sha256_text(prompt_material),
                    taxonomy=self.taxonomy,
                )
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt < self.retries:
                    # Cuando Ollama sí respondió pero el JSON o el contrato son
                    # inválidos, el segundo intento recibe la salida previa y una
                    # instrucción de corrección. Repetir exactamente el mismo
                    # prompt con temperatura 0 reproduciría el mismo error.
                    if content:
                        body["messages"] = [
                            *base_messages,
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "La salida anterior no cumple el esquema o la "
                                    f"taxonomía: {exc}. Devuelve únicamente un objeto "
                                    "JSON válido. Usa solo categorías permitidas y "
                                    "haz que coarse_labels sea exactamente la unión "
                                    "derivada de fine_labels."
                                ),
                            },
                        ]
                    else:
                        body["messages"] = list(base_messages)
                    time.sleep(min(2**attempt, 4))
        raise ProviderError(
            f"Ollama falló después de {self.retries + 1} intentos: {last_error}"
        )
