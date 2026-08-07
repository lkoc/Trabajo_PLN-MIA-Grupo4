from __future__ import annotations

import json

import pytest

from moderacion_peru.io import sha256_file
from moderacion_peru.providers.base import ProviderError, normalize_payload
from moderacion_peru.providers.huggingface import HuggingFaceProvider
from moderacion_peru.providers.ollama import OllamaProvider
from moderacion_peru.pilot import multilabel_metrics
from moderacion_peru.training import resolve_prediction


def payload(**overrides):
    data = {
        "chunk_id": "c1",
        "coarse_labels": ["SEGURO"],
        "fine_labels": ["seguro"],
        "flags": [],
        "needs_review": False,
        "notes": "",
        "score_confianza": 0.9,
        "justificacion": "No aparece un daño cubierto.",
    }
    data.update(overrides)
    return data


def test_provider_requires_explicit_safe_fine_label():
    with pytest.raises(ProviderError, match="estado fino seguro"):
        normalize_payload(
            payload(fine_labels=[]),
            text="neutral",
            source="test",
            annotator_type="llm_local",
            model="fixture",
        )


def test_provider_checks_fine_to_coarse_consistency():
    with pytest.raises(ProviderError, match="Inconsistencia"):
        normalize_payload(
            payload(coarse_labels=["ACOSO_AMENAZA"]),
            text="neutral",
            source="test",
            annotator_type="llm_local",
            model="fixture",
        )


def test_prediction_conflict_requires_review():
    labels = [
        "SEGURO",
        "RACISMO_DISCRIMINACION",
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
        "CONTENIDO_SEXUAL",
    ]
    scores = dict.fromkeys(labels, 0.1)
    thresholds = dict.fromkeys(labels, 0.5)
    scores["SEGURO"] = 0.9
    scores["ACOSO_AMENAZA"] = 0.8
    result = resolve_prediction(scores, thresholds, uncertainty_margin=0)
    assert result.requires_review
    assert "conflicto_seguro_dano" in result.review_reasons


def test_no_output_is_not_turned_into_safe():
    labels = [
        "SEGURO",
        "RACISMO_DISCRIMINACION",
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
        "CONTENIDO_SEXUAL",
    ]
    result = resolve_prediction(
        dict.fromkeys(labels, 0.1), dict.fromkeys(labels, 0.5), uncertainty_margin=0
    )
    assert result.labels == ()
    assert "sin_categoria_sobre_umbral" in result.review_reasons


def test_pilot_reports_macro_f1_for_harms():
    metrics = multilabel_metrics(
        [["ACOSO_AMENAZA"], ["RACISMO_DISCRIMINACION"]],
        [["ACOSO_AMENAZA"], []],
        ["RACISMO_DISCRIMINACION", "ACOSO_AMENAZA"],
    )
    assert metrics["per_label"]["ACOSO_AMENAZA"]["f1"] == 1.0
    assert metrics["per_label"]["RACISMO_DISCRIMINACION"]["f1"] == 0.0
    assert metrics["f1_macro"] == 0.5


def test_ollama_disables_thinking_and_limits_structured_output(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": json.dumps(payload())}}

    def fake_post(url, *, json, timeout):
        captured.update({"url": url, "body": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr("moderacion_peru.providers.ollama.requests.post", fake_post)
    provider = OllamaProvider(model="fixture", retries=0)
    result = provider.annotate({"chunk_id": "c1", "text": "contenido neutral"})

    assert result.coarse_labels == ["SEGURO"]
    assert captured["body"]["think"] is False
    assert captured["body"]["options"]["num_predict"] == 512
    prompt = "\n".join(message["content"] for message in captured["body"]["messages"])
    assert "GUÍA OPERATIVA VIGENTE" in prompt
    assert "ridiculo_encubridor" in prompt
    assert result.prompt_sha256


def test_ollama_retry_asks_to_correct_invalid_structured_output(monkeypatch):
    captured = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": self.content}}

    responses = iter([Response("sin JSON"), Response(json.dumps(payload()))])

    def fake_post(url, *, json, timeout):
        captured.append(
            {
                "messages": [dict(message) for message in json["messages"]],
                "seed": json["options"]["seed"],
            }
        )
        return next(responses)

    monkeypatch.setattr("moderacion_peru.providers.ollama.requests.post", fake_post)
    result = OllamaProvider(model="fixture", retries=1, seed=17).annotate(
        {"chunk_id": "c1", "text": "contenido neutral"}
    )

    assert result.coarse_labels == ["SEGURO"]
    assert len(captured) == 2
    assert captured[0]["seed"] == captured[1]["seed"] == 17
    assert len(captured[0]["messages"]) == 2
    assert len(captured[1]["messages"]) == 4
    assert "salida anterior no cumple" in captured[1]["messages"][-1]["content"]


def test_provider_preserves_video_group_and_temporal_metadata(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": json.dumps(payload())}}

    monkeypatch.setattr(
        "moderacion_peru.providers.ollama.requests.post",
        lambda *args, **kwargs: Response(),
    )
    result = OllamaProvider(model="fixture", retries=0).annotate(
        {
            "chunk_id": "c1",
            "video_id": "video_with_underscore",
            "text": "contenido neutral",
            "start_seconds": 3.0,
            "end_seconds": 9.0,
            "video_title": "Título",
        }
    )
    assert result.video_id == "video_with_underscore"
    assert result.start_seconds == 3.0
    assert result.end_seconds == 9.0
    assert result.video_title == "Título"


def test_ollama_probe_records_exact_model_digest(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    responses = {
        "/api/version": {"version": "fixture"},
        "/api/tags": {
            "models": [
                {
                    "name": "qwen3.5:4b",
                    "digest": "2a654d98e6fb",
                    "details": {"quantization_level": "Q4_K_M"},
                }
            ]
        },
    }

    def fake_get(url, *, timeout):
        return Response(
            responses[next(path for path in responses if url.endswith(path))]
        )

    monkeypatch.setattr("moderacion_peru.providers.ollama.requests.get", fake_get)
    result = OllamaProvider().probe()
    assert result["model_digest"] == "2a654d98e6fb"
    assert result["model_details"]["quantization_level"] == "Q4_K_M"
    assert len(result["operational_prompt_sha256"]) == 64
    assert result["operational_prompt_sha256"] == sha256_file(
        OllamaProvider().operational_prompt_path
    )
    assert result["operational_prompt_path"].endswith(
        "config\\prompt_operacional_ollama_v2.md"
    ) or result["operational_prompt_path"].endswith(
        "config/prompt_operacional_ollama_v2.md"
    )
    assert result["seed"] == 20260805


def test_huggingface_uses_chat_template_without_thinking_and_retries():
    class Tokenizer:
        calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            return "\n".join(message["content"] for message in messages)

    class Generator:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.calls = 0

        def __call__(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return [{"generated_text": "sin JSON"}]
            return [{"generated_text": json.dumps(payload())}]

    generator = Generator()
    provider = HuggingFaceProvider(model="fixture", retries=1)
    provider._pipeline = generator
    result = provider.annotate({"chunk_id": "c1", "text": "contenido neutral"})

    assert result.coarse_labels == ["SEGURO"]
    assert generator.calls == 2
    assert generator.tokenizer.calls[0]["enable_thinking"] is False
    assert provider.probe()["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
