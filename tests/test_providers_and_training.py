from __future__ import annotations

import json

import pytest

from moderacion_peru.labeling import (
    annotate_batched_incremental,
    annotate_incremental,
    recover_historical_annotations,
)
from moderacion_peru.labeling_calibration import (
    build_directed_review_queue,
    calibrate_primary_against_reviewer,
    select_calibration_panel,
)
from moderacion_peru.io import read_jsonl
from moderacion_peru.io import sha256_file, sha256_text
from moderacion_peru.providers.base import ProviderError, normalize_payload
from moderacion_peru.providers.deepseek import DeepSeekProvider
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


def test_incremental_labeling_reports_progress_and_none_removes_limit(tmp_path):
    class FixtureProvider:
        def annotate(self, chunk):
            return normalize_payload(
                payload(chunk_id=chunk["chunk_id"]),
                text=chunk["text"],
                source="fixture_local",
                annotator_type="llm_local",
                model="fixture",
            )

    records = [
        {"chunk_id": f"c{index}", "text": f"texto {index}"}
        for index in range(3)
    ]
    output = tmp_path / "annotations.jsonl"
    progress = []
    first = annotate_incremental(
        records,
        FixtureProvider(),
        output,
        limit=2,
        progress_callback=progress.append,
    )

    assert first == {"already_completed": 0, "selected": 2, "labeled": 2, "errors": 0}
    assert [event["status"] for event in progress] == [
        "started",
        "labeled",
        "labeled",
        "finished",
    ]
    assert sum(event["advance"] for event in progress) == 2

    second = annotate_incremental(records, FixtureProvider(), output, limit=None)
    assert second == {"already_completed": 2, "selected": 1, "labeled": 1, "errors": 0}
    assert len(list(read_jsonl(output))) == 3


@pytest.mark.parametrize("invalid_limit", [0, -1, True])
def test_incremental_labeling_rejects_invalid_limits(tmp_path, invalid_limit):
    with pytest.raises(ValueError, match="None o un entero positivo"):
        annotate_incremental([], object(), tmp_path / "annotations.jsonl", limit=invalid_limit)


def test_deepseek_batches_and_retries_only_invalid_row(monkeypatch):
    calls = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": json.dumps(self.content)}}],
                "usage": {
                    "prompt_tokens": 100,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 20,
                    "completion_tokens": 10,
                },
            }

    def fake_post(url, *, headers, json, timeout):
        calls.append(json)
        if len(calls) == 1:
            return Response(
                {
                    "annotations": [
                        payload(
                            chunk_id="c1",
                            notes=None,
                            fine_labels=["seguro", "humor_encubridor"],
                        ),
                        payload(chunk_id="c2", fine_labels=[]),
                    ]
                }
            )
        return Response({"annotations": [payload(chunk_id="c2")]})

    monkeypatch.setattr("moderacion_peru.providers.deepseek.requests.post", fake_post)
    provider = DeepSeekProvider(
        api_key="fixture", max_workers=1, records_per_request=2, retries=0
    )
    results = provider.annotate_batch(
        [
            {"chunk_id": "c1", "text": "uno", "video_id": "v1"},
            {"chunk_id": "c2", "text": "dos", "video_id": "v2"},
        ]
    )

    assert len(calls) == 2
    assert all(call["thinking"] == {"type": "disabled"} for call in calls)
    assert [result.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].notes == ""
    assert results[0].flags == ["humor_encubridor"]
    assert provider.usage_summary()["requests"] == 2
    assert provider.usage_summary()["estimated_cost_usd"] > 0
    assert provider.probe()["thinking"] == {"type": "disabled"}


def test_deepseek_preflight_sends_no_corpus(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "deepseek-v4-flash"}]}

    observed = {}

    def fake_get(url, *, headers, timeout):
        observed.update({"url": url, "headers": headers, "timeout": timeout})
        return Response()

    monkeypatch.setattr("moderacion_peru.providers.deepseek.requests.get", fake_get)
    provider = DeepSeekProvider(api_key="fixture")
    result = provider.validate_connection()

    assert observed["url"].endswith("/models")
    assert result["model_available"] is True
    assert result["status"] == "credential_and_models_verified_no_corpus_sent"


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


def test_huggingface_provider_batches_wrapped_annotations():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(message["content"] for message in messages)

    class Generator:
        tokenizer = Tokenizer()

        def __call__(self, prompts, **kwargs):
            outputs = []
            for prompt_text in prompts:
                ids = [part.split('"')[0] for part in prompt_text.split('"chunk_id": "')[1:]]
                annotations = [payload(chunk_id=chunk_id) for chunk_id in ids]
                outputs.append([{"generated_text": json.dumps({"annotations": annotations})}])
            return outputs

    provider = HuggingFaceProvider(
        model="fixture", records_per_request=2, inference_batch_size=2, retries=0
    )
    provider._pipeline = Generator()
    rows = provider.annotate_batch(
        [
            {"chunk_id": "c1", "video_id": "v1", "text": "uno"},
            {"chunk_id": "c2", "video_id": "v2", "text": "dos"},
            {"chunk_id": "c3", "video_id": "v3", "text": "tres"},
        ]
    )

    assert [row.chunk_id for row in rows] == ["c1", "c2", "c3"]
    assert provider.probe()["records_per_request"] == 2
    assert len(provider.probe()["operational_prompt_sha256"]) == 64


def test_batched_incremental_is_resumable_and_signed(tmp_path):
    class BatchFixtureProvider:
        def annotate(self, chunk):
            return normalize_payload(
                payload(chunk_id=chunk["chunk_id"]),
                text=chunk["text"],
                source="fixture",
                annotator_type="llm_local",
                model="fixture",
            )

        def annotate_batch(self, chunks):
            return [self.annotate(chunk) for chunk in chunks]

    records = [
        {"chunk_id": f"c{index}", "video_id": f"v{index}", "text": f"texto {index}"}
        for index in range(5)
    ]
    output = tmp_path / "batched.jsonl"
    progress = []
    first = annotate_batched_incremental(
        records,
        BatchFixtureProvider(),
        output,
        limit=4,
        processing_batch_size=2,
        progress_callback=progress.append,
        run_metadata={"model": "fixture", "prompt": "v1"},
    )
    second = annotate_batched_incremental(
        records,
        BatchFixtureProvider(),
        output,
        processing_batch_size=2,
        run_metadata={"model": "fixture", "prompt": "v1"},
    )

    assert first["labeled"] == 4
    assert first["batches"] == 2
    assert second["already_completed"] == 4
    assert second["labeled"] == 1
    assert len(list(read_jsonl(output))) == 5
    assert sum(event.get("advance", 0) for event in progress) == 4
    with output.open("a", encoding="utf-8") as handle:
        handle.write('{"chunk_id":"corrupto","text":"sin contrato"}\n')
    third = annotate_batched_incremental(
        records,
        BatchFixtureProvider(),
        output,
        processing_batch_size=2,
        run_metadata={"model": "fixture", "prompt": "v1"},
        quarantine_invalid_progress=True,
    )
    assert third["quarantined_progress"] is not None
    assert len(list(read_jsonl(output))) == 5
    assert list(tmp_path.glob("batched.jsonl.quarantine-*.jsonl"))
    with pytest.raises(ValueError, match="otro modelo, prompt o configuración"):
        annotate_batched_incremental(
            records,
            BatchFixtureProvider(),
            output,
            run_metadata={"model": "different", "prompt": "v1"},
        )


def test_historical_recovery_rekeys_only_exact_unique_text_and_labels_pending(tmp_path):
    historical_chunks = tmp_path / "legacy_chunks.jsonl"
    historical_annotations = tmp_path / "legacy_flash.jsonl"
    historical_chunks.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "old-1",
                        "video_id": "v1",
                        "text": "Texto  exacto",
                        "start_seconds": 0,
                        "end_seconds": 30,
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "old-2",
                        "video_id": "v2",
                        "text": "Segmentación histórica distinta",
                        "start_seconds": 0,
                        "end_seconds": 35,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    historical_annotations.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "old-1",
                        "labels": ["seguro"],
                        "flags": [],
                        "needs_review": False,
                        "score_confianza": 0.95,
                        "notes": "",
                        "justificacion": "Histórica",
                        "annotator_model": "deepseek-v4-flash",
                        "annotated_at": "2026-07-25T21:25:13-05:00",
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "old-2",
                        "labels": ["amenaza_directa"],
                        "flags": [],
                        "needs_review": False,
                        "score_confianza": 0.9,
                        "notes": "",
                        "justificacion": "Histórica",
                        "annotator_model": "deepseek-v4-flash",
                        "annotated_at": "2026-07-25T21:25:14-05:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    current = [
        {
            "chunk_id": "new-1",
            "video_id": "v1",
            "text": "texto exacto",
            "text_sha256": sha256_text("texto exacto"),
            "start_seconds": 0,
            "end_seconds": 30,
        },
        {
            "chunk_id": "new-2",
            "video_id": "v2",
            "text": "Segmentación nueva",
            "text_sha256": sha256_text("Segmentación nueva"),
            "start_seconds": 0,
            "end_seconds": 30,
        },
    ]
    output = tmp_path / "primary.jsonl"
    run_metadata = {"provider": {"model": "deepseek-v4-flash", "history": "fixture"}}
    first = recover_historical_annotations(
        current,
        historical_chunks,
        [historical_annotations],
        output,
        expected_model="deepseek-v4-flash",
        historical_prompt_sha256="a" * 64,
        run_metadata=run_metadata,
    )
    second = recover_historical_annotations(
        current,
        historical_chunks,
        [historical_annotations],
        output,
        expected_model="deepseek-v4-flash",
        historical_prompt_sha256="a" * 64,
        run_metadata=run_metadata,
    )

    recovered = list(read_jsonl(output))
    assert first["recovered_new"] == 1
    assert first["historical_not_reusable"] == 1
    assert first["pending_current_after_recovery"] == 1
    assert second["recovered_new"] == 0
    assert second["already_present_matches"] == 1
    assert recovered[0]["chunk_id"] == "new-1"
    assert recovered[0]["coarse_labels"] == ["SEGURO"]
    assert recovered[0]["consolidated_sources"] == ["legacy_flash.jsonl:old-1"]

    class PendingProvider:
        def annotate_batch(self, chunks):
            return [
                normalize_payload(
                    payload(chunk_id=chunk["chunk_id"]),
                    text=chunk["text"],
                    source="current",
                    annotator_type="llm_remote",
                    model="deepseek-v4-flash",
                    source_record_sha256=chunk["text_sha256"],
                )
                for chunk in chunks
            ]

    resumed = annotate_batched_incremental(
        current,
        PendingProvider(),
        output,
        run_metadata=run_metadata,
    )
    assert resumed["already_completed"] == 1
    assert resumed["labeled"] == 1
    assert {row["chunk_id"] for row in read_jsonl(output)} == {"new-1", "new-2"}


def test_incremental_stream_checkpoint_survives_keyboard_interrupt(tmp_path):
    records = [
        {
            "chunk_id": f"c{index}",
            "video_id": f"v{index}",
            "text": f"texto {index}",
            "text_sha256": sha256_text(f"texto {index}"),
        }
        for index in range(3)
    ]

    def annotation(chunk):
        return normalize_payload(
            payload(chunk_id=chunk["chunk_id"]),
            text=chunk["text"],
            source="fixture",
            annotator_type="llm_remote",
            model="fixture",
            source_record_sha256=chunk["text_sha256"],
        )

    class InterruptedProvider:
        def iter_annotate_batch(self, chunks):
            yield [(0, annotation(chunks[0]))]
            raise KeyboardInterrupt

    class ResumeProvider:
        def annotate_batch(self, chunks):
            return [annotation(chunk) for chunk in chunks]

    output = tmp_path / "interrupted.jsonl"
    checkpoints = []
    with pytest.raises(KeyboardInterrupt):
        annotate_batched_incremental(
            records,
            InterruptedProvider(),
            output,
            processing_batch_size=3,
            checkpoint_callback=checkpoints.append,
            run_metadata={"model": "fixture"},
        )
    assert [row["chunk_id"] for row in read_jsonl(output)] == ["c0"]
    assert checkpoints[-1]["status"] == "interrupted_checkpoint"
    assert checkpoints[-1]["labeled"] == 1

    resumed = annotate_batched_incremental(
        records,
        ResumeProvider(),
        output,
        processing_batch_size=3,
        run_metadata={"model": "fixture"},
    )
    assert resumed["already_completed"] == 1
    assert resumed["labeled"] == 2
    assert {row["chunk_id"] for row in read_jsonl(output)} == {"c0", "c1", "c2"}


def test_run_manifest_safely_upgrades_non_thinking_and_historical_provenance(tmp_path):
    record = {
        "chunk_id": "c1",
        "video_id": "v1",
        "text": "texto",
        "text_sha256": sha256_text("texto"),
    }

    class Provider:
        def annotate_batch(self, chunks):
            return [
                normalize_payload(
                    payload(chunk_id=chunk["chunk_id"]),
                    text=chunk["text"],
                    source="fixture",
                    annotator_type="llm_remote",
                    model="deepseek-v4-flash",
                    source_record_sha256=chunk["text_sha256"],
                )
                for chunk in chunks
            ]

    output = tmp_path / "campaign.jsonl"
    old_metadata = {
        "provider": {"model": "deepseek-v4-flash", "prompt_sha256": "a" * 64},
        "taxonomy_version": "2.1.0",
    }
    annotate_batched_incremental([record], Provider(), output, run_metadata=old_metadata)
    old_manifest = json.loads(
        output.with_suffix(".jsonl.run.json").read_text(encoding="utf-8")
    )
    new_metadata = {
        "provider": {
            "model": "deepseek-v4-flash",
            "prompt_sha256": "a" * 64,
            "thinking": {"type": "disabled"},
            "historical_recovery": {"mapping": "fixture"},
        },
        "taxonomy_version": "2.1.0",
    }
    result = annotate_batched_incremental(
        [record], Provider(), output, run_metadata=new_metadata
    )
    upgraded = json.loads(output.with_suffix(".jsonl.run.json").read_text(encoding="utf-8"))

    assert result["already_completed"] == 1
    assert result["labeled"] == 0
    assert upgraded["compatible_predecessor_run_signature"] == old_manifest["run_signature"]
    assert upgraded["provider"]["thinking"] == {"type": "disabled"}
    assert upgraded["provider"]["historical_recovery"] == {"mapping": "fixture"}


def test_calibration_and_historical_routing_policy():
    chunks = [
        {
            "chunk_id": f"c{index}",
            "video_id": f"v{index}",
            "channel_title": f"canal-{index % 2}",
            "start_seconds": 0,
            "text": f"texto {index}",
        }
        for index in range(100)
    ]
    panel = select_calibration_panel(chunks, panel_size=20, seed=7)
    assert len(panel) == 20
    assert len({row["video_id"] for row in panel}) == 20

    primary = []
    reviewer = []
    for index, chunk in enumerate(chunks):
        primary.append(
            {
                **chunk,
                "coarse_labels": ["SEGURO"],
                "needs_review": index == 0,
                "score_confianza": 0.95,
            }
        )
        reviewer.append({**chunk, "coarse_labels": ["SEGURO"]})
    calibration = calibrate_primary_against_reviewer(
        primary,
        reviewer,
        minimum_auto_count=10,
        bootstrap_replicates=20,
    )
    assert calibration["threshold_status"] == "calibrated"
    assert calibration["selected_threshold"] == 0.70

    primary[1]["coarse_labels"] = ["ACOSO_AMENAZA"]
    primary[2]["score_confianza"] = 0.60
    queue, manifest = build_directed_review_queue(
        chunks,
        primary,
        confidence_threshold=0.90,
        safe_control_rate=0,
    )
    reasons = {row["chunk_id"]: row["routing_reason"] for row in queue}
    assert reasons == {
        "c0": "needs_review",
        "c1": "damage",
        "c2": "low_confidence",
    }
    assert manifest["routing_reasons"] == {
        "needs_review": 1,
        "damage": 1,
        "low_confidence": 1,
    }


def test_directed_review_routing_is_case_insensitive():
    chunks = [
        {"chunk_id": "safe", "video_id": "v1", "text": "neutral"},
        {"chunk_id": "damage", "video_id": "v2", "text": "ataque"},
    ]
    primary = [
        {
            **chunks[0],
            "coarse_labels": ["seguro"],
            "needs_review": False,
            "score_confianza": 0.99,
        },
        {
            **chunks[1],
            "coarse_labels": ["acoso_amenaza"],
            "needs_review": False,
            "score_confianza": 0.99,
        },
    ]

    queue, manifest = build_directed_review_queue(
        chunks,
        primary,
        confidence_threshold=0.90,
        safe_control_rate=0,
    )

    assert [(row["chunk_id"], row["routing_reason"]) for row in queue] == [
        ("damage", "damage")
    ]
    assert manifest["routing_reasons"] == {"damage": 1}
