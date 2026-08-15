import json
from pathlib import Path

import pytest

from moderacion_peru.io import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from moderacion_peru.registry import (
    compare_and_publish_registry,
    publish_frozen_ensemble_registry,
)
from moderacion_peru.schemas import ReviewEvent
from moderacion_peru.servers import (
    _consensus_result,
    _ensemble_soft_mean_result,
    _is_labeling_excluded,
    _is_labeling_priority,
    _is_labeling_urgent,
    _labeling_bulk_events,
    _labeling_campaign_page,
    _labeling_dashboard,
    _labeling_progress,
    _labeling_scope_rows,
    _production_feedback,
    _production_registry_paths,
    _requires_labeling_action,
)
from moderacion_peru.taxonomy import load_taxonomy


def test_labeling_campaign_is_paged_and_uses_latest_review_state():
    rows = [
        {"chunk_id": "c1", "cohort": "a", "text": "uno"},
        {"chunk_id": "c2", "cohort": "a", "text": "dos"},
        {"chunk_id": "c3", "cohort": "b", "text": "tres"},
    ]
    reviews = {
        "c1": {"chunk_id": "c1", "action": "accept"},
        "c2": {"chunk_id": "c2", "action": "defer"},
    }

    page = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=1,
        cohort="a",
        only_pending=True,
    )

    assert page["total"] == 1
    assert page["indices"] == [1]
    assert [row["chunk_id"] for row in page["rows"]] == ["c2"]
    assert page["reviews"] == {"c2": reviews["c2"]}
    assert _labeling_progress(rows, reviews) == {
        "total": 3,
        "reviewed": 2,
        "resolved": 2,
        "deferred": 1,
        "pending": 0,
        "excluded_total": 0,
        "progress_pct": pytest.approx(200 / 3),
    }


def test_labeling_urgent_and_pro_priority_queues_are_distinct():
    rows = [
        {
            "chunk_id": "urgent-flash",
            "annotator_model": "deepseek-v4-flash",
            "coarse_labels": ["SEGURO"],
            "consolidation_warning": "conflicting_top_priority_decisions",
        },
        {
            "chunk_id": "pro-unresolved",
            "annotator_model": "deepseek-v4-pro",
            "coarse_labels": [],
            "needs_review": True,
            "decision_status": "needs_review",
        },
        {
            "chunk_id": "pro-damage",
            "annotator_model": "deepseek-v4-pro",
            "coarse_labels": ["ACOSO_AMENAZA"],
            "needs_review": False,
            "decision_status": "resolved",
        },
        {
            "chunk_id": "pro-safe",
            "annotator_model": "deepseek-v4-pro",
            "coarse_labels": ["SEGURO"],
            "needs_review": False,
            "decision_status": "resolved",
        },
    ]

    assert [_is_labeling_urgent(row) for row in rows] == [True, False, False, False]
    assert [_is_labeling_priority(row) for row in rows] == [False, True, True, False]
    urgent = _labeling_campaign_page(rows, {}, offset=0, limit=10, urgent_only=True)
    priority = _labeling_campaign_page(rows, {}, offset=0, limit=10, priority_only=True)

    assert [row["chunk_id"] for row in urgent["rows"]] == ["urgent-flash"]
    assert [row["chunk_id"] for row in priority["rows"]] == [
        "pro-unresolved",
        "pro-damage",
    ]
    assert _labeling_progress(rows, {}, urgent_only=True)["total"] == 1
    assert _labeling_progress(rows, {}, priority_only=True)["total"] == 2

    higher_reviews = {
        "pro-unresolved": {
            "chunk_id": "pro-unresolved",
            "action": "accept",
            "final_labels": ["SEGURO"],
        },
        "pro-safe": {
            "chunk_id": "pro-safe",
            "action": "modify",
            "final_labels": ["ATAQUE_POR_GENERO_IDENTIDAD"],
        },
    }
    effective_priority = _labeling_campaign_page(
        rows,
        higher_reviews,
        offset=0,
        limit=10,
        priority_only=True,
    )
    assert [row["chunk_id"] for row in effective_priority["rows"]] == [
        "pro-damage",
        "pro-safe",
    ]
    assert _labeling_progress(rows, higher_reviews, priority_only=True)["total"] == 2


def test_labeling_action_queue_contains_only_pending_or_deferred_cases():
    rows = [
        {"chunk_id": "automatic", "decision_status": "resolved"},
        {"chunk_id": "pending", "decision_status": "needs_review"},
        {"chunk_id": "deferred", "decision_status": "resolved"},
        {"chunk_id": "human", "decision_status": "needs_review"},
    ]
    reviews = {
        "deferred": {"chunk_id": "deferred", "action": "defer"},
        "human": {
            "chunk_id": "human",
            "action": "modify",
            "final_labels": ["SEGURO"],
        },
    }

    assert [_requires_labeling_action(row, reviews) for row in rows] == [
        False,
        True,
        True,
        False,
    ]
    action_page = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=10,
        only_pending=True,
    )
    assert [row["chunk_id"] for row in action_page["rows"]] == [
        "pending",
        "deferred",
    ]
    assert _labeling_progress(rows, reviews, action_only=True)["total"] == 2


def test_labeling_excluded_queue_uses_latest_effective_decision():
    rows = [
        {"chunk_id": "human-reject", "decision_status": "resolved"},
        {"chunk_id": "base-excluded", "decision_status": "excluded"},
        {"chunk_id": "restored", "decision_status": "excluded"},
        {"chunk_id": "pending", "decision_status": "needs_review"},
    ]
    reviews = {
        "human-reject": {"chunk_id": "human-reject", "action": "reject"},
        "restored": {"chunk_id": "restored", "action": "modify"},
    }

    assert [_is_labeling_excluded(row, reviews) for row in rows] == [
        True,
        True,
        False,
        False,
    ]
    page = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=10,
        only_pending=True,
        excluded_only=True,
    )
    assert [row["chunk_id"] for row in page["rows"]] == [
        "human-reject",
        "base-excluded",
    ]
    assert page["total"] == 2
    assert _labeling_progress(rows, reviews, excluded_only=True) == {
        "total": 2,
        "reviewed": 1,
        "resolved": 2,
        "deferred": 0,
        "pending": 0,
        "excluded_total": 2,
        "progress_pct": 100.0,
    }


def test_labeling_filters_combine_categories_flags_and_review_status():
    rows = [
        {
            "chunk_id": "pending-safe",
            "decision_status": "needs_review",
            "needs_review": True,
            "coarse_labels": ["SEGURO"],
            "flags": [],
        },
        {
            "chunk_id": "human-sexual",
            "decision_status": "needs_review",
            "coarse_labels": ["SEGURO"],
            "flags": [],
        },
        {
            "chunk_id": "excluded-acoso",
            "decision_status": "resolved",
            "coarse_labels": ["ACOSO_AMENAZA"],
            "flags": ["contexto_necesario"],
        },
        {
            "chunk_id": "multi-damage",
            "decision_status": "needs_review",
            "coarse_labels": [],
            "flags": [],
        },
        {
            "chunk_id": "unlabeled",
            "decision_status": "needs_review",
            "needs_review": True,
            "coarse_labels": [],
            "flags": [],
        },
    ]
    reviews = {
        "human-sexual": {
            "chunk_id": "human-sexual",
            "action": "modify",
            "final_labels": ["CONTENIDO_SEXUAL"],
            "flags": ["contexto_necesario"],
        },
        "excluded-acoso": {
            "chunk_id": "excluded-acoso",
            "action": "reject",
            "final_labels": [],
            "flags": [],
        },
        "multi-damage": {
            "chunk_id": "multi-damage",
            "action": "modify",
            "final_labels": ["ACOSO_AMENAZA", "CONTENIDO_SEXUAL"],
            "flags": ["humor_encubridor", "contexto_necesario"],
        },
    }

    sexual_resolved = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=10,
        filter_labels={"CONTENIDO_SEXUAL"},
        filter_statuses={"resolved"},
    )
    assert [row["chunk_id"] for row in sexual_resolved["rows"]] == [
        "human-sexual",
        "multi-damage",
    ]

    excluded_with_reference = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=10,
        filter_labels={"ACOSO_AMENAZA"},
        filter_statuses={"excluded"},
        filter_flags={"contexto_necesario"},
    )
    assert [row["chunk_id"] for row in excluded_with_reference["rows"]] == [
        "excluded-acoso"
    ]

    all_categories_and_flags = _labeling_progress(
        rows,
        reviews,
        filter_labels={"ACOSO_AMENAZA", "CONTENIDO_SEXUAL"},
        filter_flags={"humor_encubridor", "contexto_necesario"},
        match_all=True,
    )
    assert all_categories_and_flags["total"] == 1

    unlabeled = _labeling_campaign_page(
        rows,
        reviews,
        offset=0,
        limit=10,
        filter_labeling={"unlabeled"},
    )
    assert [row["chunk_id"] for row in unlabeled["rows"]] == ["unlabeled"]

    unlabeled_pending = _labeling_progress(
        rows,
        reviews,
        filter_labeling={"unlabeled"},
        filter_statuses={"pending"},
    )
    assert unlabeled_pending["total"] == 1


def test_labeling_dashboard_uses_effective_decisions_and_audit_metrics():
    taxonomy = load_taxonomy()
    rows = [
        {
            "chunk_id": "safe",
            "decision_status": "resolved",
            "coarse_labels": ["SEGURO"],
            "annotator_model": "deepseek-v4-flash",
            "channel_title": "Canal A",
            "video_id": "v1",
        },
        {
            "chunk_id": "harm",
            "decision_status": "resolved",
            "coarse_labels": ["ACOSO_AMENAZA"],
            "annotator_model": "deepseek-v4-pro",
            "channel_title": "Canal A",
            "video_id": "v1",
        },
        {
            "chunk_id": "unlabeled",
            "decision_status": "needs_review",
            "needs_review": True,
            "coarse_labels": [],
            "annotator_model": "deepseek-v4-pro",
            "channel_title": "Canal B",
            "video_id": "v2",
        },
        {
            "chunk_id": "excluded",
            "decision_status": "resolved",
            "coarse_labels": ["SEGURO"],
            "annotator_model": "deepseek-v4-flash",
            "channel_title": "Canal B",
            "video_id": "v2",
        },
        {
            "chunk_id": "modified",
            "decision_status": "resolved",
            "coarse_labels": ["SEGURO"],
            "annotator_model": "deepseek-v4-pro",
            "channel_title": "Canal B",
            "video_id": "v2",
        },
        {
            "chunk_id": "deferred",
            "decision_status": "resolved",
            "coarse_labels": ["SEGURO"],
            "annotator_model": "deepseek-v4-flash",
            "channel_title": "Canal C",
            "video_id": "v3",
        },
    ]
    reviews = {
        "excluded": {
            "chunk_id": "excluded",
            "action": "reject",
            "reviewer": "CODEX",
            "created_at": "2026-08-09T10:10:00+00:00",
        },
        "modified": {
            "chunk_id": "modified",
            "action": "modify",
            "final_labels": ["RACISMO_DISCRIMINACION", "CONTENIDO_SEXUAL"],
            "flags": ["contexto_necesario"],
            "reviewer": "CODEX",
            "created_at": "2026-08-09T11:10:00+00:00",
        },
        "deferred": {
            "chunk_id": "deferred",
            "action": "defer",
            "reviewer": "LKG",
            "created_at": "2026-08-09T11:15:00+00:00",
        },
    }
    audit = {
        "generated_by": "test",
        "sample": {"size": 100},
        "reference": {"warning": "Referencia interna."},
        "systems": {
            "cascada_flash_pro_consolidada": {
                "answered": 90,
                "coverage_over_sample": 0.9,
                "abstention_over_sample": 0.1,
                "point": {
                    "exact_agreement": 0.95,
                    "binary_f1": 0.9,
                    "binary_mcc": 0.88,
                    "multilabel_micro_f1": 0.87,
                    "hamming_loss": 0.02,
                },
                "exact_agreement_wilson_95": [0.9, 0.98],
                "confidence": {
                    "mean": 0.92,
                    "brier_for_exact_correctness": 0.04,
                    "ece_10_equal_width": 0.03,
                },
            }
        },
        "paired_flash_vs_pro_on_common_answered": {
            "n": 50,
            "pro_minus_flash_exact_agreement": 0.2,
        },
        "inference": {"bootstrap_replicates": 2000},
    }

    result = _labeling_dashboard(rows, reviews, taxonomy, audit_metrics=audit)
    corpus = result["live"]["corpus"]

    assert corpus == {
        "total": 6,
        "eligible": 5,
        "excluded": 1,
        "excluded_pct": pytest.approx(100 / 6),
        "safe": 2,
        "harm": 2,
        "unlabeled": 1,
        "labeled": 4,
        "coverage_pct": 80.0,
        "harm_prevalence_pct": 40.0,
        "channels": 3,
        "videos": 3,
        "avg_chunks_per_channel": pytest.approx(5 / 3),
        "median_chunks_per_channel": 2.0,
        "avg_chunks_per_video": pytest.approx(5 / 3),
        "median_chunks_per_video": 2.0,
        "multilabel_assignments": 3,
        "invalid_safe_harm": 0,
        "intermediate_pro_needs_review": 1,
        "intermediate_review_overridden": 0,
        "final_pending": 1,
    }
    labels = {row["id"]: row["count"] for row in result["live"]["labels"]}
    assert labels == {
        "RACISMO_DISCRIMINACION": 1,
        "ATAQUE_POR_GENERO_IDENTIDAD": 0,
        "ACOSO_AMENAZA": 1,
        "CONTENIDO_SEXUAL": 1,
    }
    assert result["live"]["queues"]["priority"]["total"] == 3
    assert result["live"]["actions"][0] == {
        "id": "reject",
        "label": "Excluido",
        "count": 1,
    }
    assert result["audit"]["available"] is True
    assert result["audit"]["systems"][0]["exact_agreement"] == 0.95
    assert any("Flash/Pro" in insight["title"] for insight in result["insights"])


def test_labeling_bulk_scope_uses_video_and_channel_title_fallback():
    taxonomy = load_taxonomy()
    rows = [
        {
            "chunk_id": "v1-1",
            "video_id": "v1",
            "video_title": "Video uno",
            "channel_id": None,
            "channel_title": "Canal Perú",
            "coarse_labels": [taxonomy.safe_label],
        },
        {
            "chunk_id": "v1-2",
            "video_id": "v1",
            "video_title": "Video uno",
            "channel_id": None,
            "channel_title": "  CANAL   PERÚ ",
            "coarse_labels": [taxonomy.damage_labels[0]],
        },
        {
            "chunk_id": "v2-1",
            "video_id": "v2",
            "video_title": "Video dos",
            "channel_id": None,
            "channel_title": "Canal Perú",
            "coarse_labels": [],
        },
        {
            "chunk_id": "other-1",
            "video_id": "other",
            "channel_title": "Otro canal",
            "coarse_labels": [taxonomy.safe_label],
        },
    ]
    reviews = {
        "v1-1": {"chunk_id": "v1-1", "action": "accept"},
        "v1-2": {"chunk_id": "v1-2", "action": "defer"},
    }

    video_summary, video_rows = _labeling_scope_rows(
        rows, reviews, anchor_chunk_id="v1-2", scope="video"
    )
    channel_summary, channel_rows = _labeling_scope_rows(
        rows, reviews, anchor_chunk_id="v1-2", scope="channel"
    )

    assert [row["chunk_id"] for row in video_rows] == ["v1-1", "v1-2"]
    assert video_summary == {
        "scope": "video",
        "scope_key": "video:v1",
        "display_name": "Video uno",
        "total": 2,
        "pending": 1,
        "resolved": 1,
        "deferred": 1,
        "acceptable_total": 2,
        "acceptable_pending": 1,
        "without_proposal_total": 0,
        "without_proposal_pending": 0,
    }
    assert [row["chunk_id"] for row in channel_rows] == ["v1-1", "v1-2", "v2-1"]
    assert channel_summary["scope_key"] == "channel-title:canal perú"
    assert channel_summary["pending"] == 2
    assert channel_summary["without_proposal_pending"] == 1


def test_labeling_bulk_events_are_idempotent_and_preserve_each_proposal():
    taxonomy = load_taxonomy()
    rows = [
        {
            "chunk_id": "c1",
            "video_id": "v1",
            "video_title": "Video",
            "channel_title": "Canal",
            "coarse_labels": [taxonomy.safe_label],
            "flags": [],
            "annotator_model": "modelo-a",
        },
        {
            "chunk_id": "c2",
            "video_id": "v1",
            "video_title": "Video",
            "channel_title": "Canal",
            "coarse_labels": [taxonomy.damage_labels[0]],
            "flags": [taxonomy.flags[0]],
            "annotator_model": "modelo-b",
        },
        {
            "chunk_id": "c3",
            "video_id": "v1",
            "video_title": "Video",
            "channel_title": "Canal",
            "coarse_labels": [],
            "flags": [],
        },
    ]
    reviews = {"c1": {"chunk_id": "c1", "action": "accept"}}
    arguments = {
        "anchor_chunk_id": "c2",
        "scope": "video",
        "action": "accept",
        "include_resolved": False,
        "reviewer": "reviewer-test",
        "notes": "confianza revisada",
        "batch_id": "batch-test-1",
    }

    summary, events = _labeling_bulk_events(rows, reviews, **arguments)
    repeated_summary, repeated_events = _labeling_bulk_events(
        rows, reviews, **arguments
    )

    assert summary["selected"] == 2
    assert summary["events_ready"] == 1
    assert summary["skipped_without_proposal"] == 1
    assert [event.chunk_id for event in events] == ["c2"]
    assert events[0].final_labels == [taxonomy.damage_labels[0]]
    assert events[0].flags == [taxonomy.flags[0]]
    assert events[0].decision_scope == "video"
    assert events[0].decision_scope_key == "video:v1"
    assert events[0].batch_target_count == 2
    assert [event.event_id for event in repeated_events] == [
        event.event_id for event in events
    ]
    assert repeated_summary == summary

    modify_summary, modified = _labeling_bulk_events(
        rows,
        reviews,
        **{
            **arguments,
            "scope": "channel",
            "action": "modify",
            "batch_id": "batch-test-classify",
            "final_labels": [
                taxonomy.damage_labels[0],
                taxonomy.damage_labels[1],
            ],
            "flags": [taxonomy.flags[0]],
        },
    )
    assert modify_summary["events_ready"] == 2
    assert modify_summary["applied_labels"] == list(taxonomy.damage_labels[:2])
    assert modify_summary["applied_flags"] == [taxonomy.flags[0]]
    assert {event.chunk_id for event in modified} == {"c2", "c3"}
    assert all(event.action == "modify" for event in modified)
    assert all(event.decision_scope == "channel" for event in modified)
    assert all(
        event.final_labels == list(taxonomy.damage_labels[:2])
        and event.flags == [taxonomy.flags[0]]
        for event in modified
    )

    with pytest.raises(ValueError, match="mutuamente excluyente"):
        _labeling_bulk_events(
            rows,
            reviews,
            **{
                **arguments,
                "action": "modify",
                "batch_id": "batch-test-invalid",
                "final_labels": [taxonomy.safe_label, taxonomy.damage_labels[0]],
            },
        )

    reject_summary, rejected = _labeling_bulk_events(
        rows,
        reviews,
        **{
            **arguments,
            "scope": "channel",
            "action": "reject",
            "include_resolved": True,
            "batch_id": "batch-test-2",
        },
    )
    assert reject_summary["selected"] == 3
    assert len(rejected) == 3
    assert all(
        event.action == "reject" and not event.final_labels for event in rejected
    )


def _candidate(
    root: Path, dataset: Path, family: str, identifier: str, score: float
) -> None:
    taxonomy = load_taxonomy()
    directory = root / identifier
    directory.mkdir(parents=True)
    write_json_atomic(directory / "checkpoint_manifest.json", {"files": []})
    write_json_atomic(directory / "metrics.json", {"fixture": True})
    write_json_atomic(
        directory / "inference.json",
        {"type": "sklearn_joblib", "model": "unused.joblib"},
    )
    write_json_atomic(
        directory / "candidate.json",
        {
            "status": "complete",
            "candidate_id": identifier,
            "model_family": family,
            "run_signature": f"run-{identifier}",
            "dataset_sha256": sha256_file(dataset),
            "target_labels": list(taxonomy.target_labels),
            "checkpoint_manifest": "checkpoint_manifest.json",
            "metrics_path": "metrics.json",
            "inference": {"type": "sklearn_joblib", "bundle": "inference.json"},
            "thresholds": {label: 0.5 for label in taxonomy.target_labels},
            "validation_metrics": {
                "false_safe_rate_on_damage": 0.1,
                "f1_macro_damage": score,
                "average_precision_macro_damage": score,
                "review_load_rate": 0.2,
            },
            "test_metrics": {"f1_macro_damage": score - 0.1},
        },
    )


def test_registry_publishes_best_member_of_each_historical_frontend_slot(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"fixture":true}\n', encoding="utf-8")
    candidates = tmp_path / "candidates"
    _candidate(candidates, dataset, "classical:sgd", "classic", 0.7)
    _candidate(candidates, dataset, "flat_minilm", "transformer", 0.8)
    _candidate(candidates, dataset, "qwen_lora", "qwen", 0.9)
    registry = tmp_path / "registro.json"

    result = compare_and_publish_registry(dataset, [candidates], registry)
    repeated = compare_and_publish_registry(dataset, [candidates], registry)
    payload = json.loads(registry.read_text(encoding="utf-8"))

    assert result["consensus_available"] is True
    assert repeated["status"] == "noop"
    assert set(payload["comparison_registries"]) == {"classical", "transformer", "qwen"}
    _, resolved = _production_registry_paths(registry, Path.cwd())
    assert set(resolved) == {"classical", "transformer", "qwen"}
    for slot, reference in payload["comparison_registries"].items():
        member = Path(reference["path"])
        assert member.is_file()
        assert sha256_file(member) == reference["sha256"]
        assert (
            json.loads(member.read_text(encoding="utf-8"))["model_id"]
            == result["selected_by_slot"][slot]
        )


def test_registry_materializes_the_frozen_soft_mean(tmp_path):
    taxonomy = load_taxonomy()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"fixture":true}\n', encoding="utf-8")
    candidates = tmp_path / "candidates"
    _candidate(candidates, dataset, "classical:sgd", "classic", 0.7)
    _candidate(candidates, dataset, "flat_minilm", "transformer", 0.8)
    _candidate(candidates, dataset, "qwen_lora", "qwen", 0.9)
    members = ["classic", "transformer", "qwen"]
    calibrators = [
        {"type": "sigmoid_platt", "coefficient": 1.0, "intercept": 0.0}
        for _ in taxonomy.target_labels
    ]
    freeze = tmp_path / "freeze.json"
    write_json_atomic(
        freeze,
        {
            "selected_id": "ensemble_soft_mean",
            "members": members,
            "dataset_sha256": sha256_file(dataset),
            "comparison_signature": "fixture-signature",
            "thresholds": {label: 0.5 for label in taxonomy.target_labels},
            "score_calibrators": calibrators,
            "member_thresholds": {
                member: {label: 0.5 for label in taxonomy.target_labels}
                for member in members
            },
            "member_score_calibrators": {
                member: calibrators for member in members
            },
            "any_damage_threshold": 0.2,
            "needs_review_policy": {"selected_delta": 0.03},
            "winner_status": "statistical_tie_or_inconclusive",
        },
    )
    registry = tmp_path / "registro.json"

    result = publish_frozen_ensemble_registry(freeze, [candidates], registry)
    repeated = publish_frozen_ensemble_registry(freeze, [candidates], registry)
    payload = json.loads(registry.read_text(encoding="utf-8"))

    assert result["selected"] == "ensemble_soft_mean"
    assert repeated["status"] == "noop"
    assert payload["ensemble_kind"] == "soft_mean"
    assert payload["status"] == "shadow_only"
    assert payload["selected_members"] == members
    assert set(payload["comparison_registries"]) == {
        "classical",
        "transformer",
        "qwen",
    }


def _model_event(
    event_id: str, slot: str, labels: list[str], text: str = "texto uno"
) -> dict:
    taxonomy = load_taxonomy()
    return {
        "event_id": event_id,
        "chunk_id": "chunk-1" if text == "texto uno" else "chunk-2",
        "text": text,
        "video_id": "video-1",
        "start_seconds": 0.0 if text == "texto uno" else 30.0,
        "end_seconds": 30.0 if text == "texto uno" else 60.0,
        "model_id": f"model-{slot}",
        "model_family": slot,
        "model_slot": slot,
        "model_label": slot,
        "labels": labels,
        "scores": {label: 0.5 for label in taxonomy.target_labels},
        "requires_review": False,
    }


def test_consensus_is_two_of_three_and_routes_disagreement_to_review():
    taxonomy = load_taxonomy()
    events = [
        _model_event("e1", "classical", [taxonomy.safe_label]),
        _model_event("e2", "transformer", [taxonomy.safe_label]),
        _model_event("e3", "qwen", [taxonomy.damage_labels[0]]),
    ]
    result = _consensus_result(events)
    assert result["labels"] == [taxonomy.safe_label]
    assert result["votes"][taxonomy.safe_label] == 2
    assert result["requires_review"] is True
    assert "desacuerdo_entre_modelos" in result["review_reasons"]


def test_frozen_soft_mean_averages_raw_scores_then_calibrates():
    taxonomy = load_taxonomy()
    events = [
        _model_event("e1", "classical", [taxonomy.safe_label]),
        _model_event("e2", "transformer", [taxonomy.safe_label]),
        _model_event("e3", "qwen", [taxonomy.safe_label]),
    ]
    for event in events:
        event["raw_scores"] = {
            label: 0.8 if label == taxonomy.safe_label else 0.1
            for label in taxonomy.target_labels
        }
    registry = {
        "model_id": "ensemble_soft_mean",
        "ensemble_kind": "soft_mean",
        "thresholds": {label: 0.5 for label in taxonomy.target_labels},
        "score_calibrators": [
            {"type": "sigmoid_platt", "coefficient": 10.0, "intercept": -5.0}
            for _ in taxonomy.target_labels
        ],
        "any_damage_threshold": 0.1,
        "needs_review_policy": {"selected_delta": 0.03},
    }

    result = _ensemble_soft_mean_result(events, registry)

    assert result["model_slot"] == "ensemble"
    assert result["labels"] == [taxonomy.safe_label]
    assert result["requires_review"] is False
    assert result["scores"][taxonomy.safe_label] > 0.95
    assert max(result["scores"][label] for label in taxonomy.damage_labels) < 0.02


def test_production_feedback_is_append_only_linked_deduplicated_and_conflict_safe(
    tmp_path,
):
    taxonomy = load_taxonomy()
    inference_path = tmp_path / "inferences.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    ready_path = tmp_path / "ready.jsonl"
    inferences = [
        _model_event("e1", "classical", [taxonomy.safe_label]),
        _model_event("e2", "transformer", [taxonomy.safe_label]),
        _model_event("e3", "qwen", [taxonomy.damage_labels[0]]),
        _model_event("e4", "consensus", [taxonomy.damage_labels[1]], text="texto dos"),
    ]
    reviews = [
        {
            "event_id": "r1",
            "source_event_id": "e1",
            "chunk_id": "chunk-1",
            "model_id": "model-classical",
            "reviewer": "h1",
            "action": "accept",
            "final_labels": [taxonomy.safe_label],
            "flags": [],
        },
        {
            "event_id": "r2",
            "source_event_id": "e2",
            "chunk_id": "chunk-1",
            "model_id": "model-transformer",
            "reviewer": "h2",
            "action": "accept",
            "final_labels": [taxonomy.safe_label],
            "flags": [],
        },
        {
            "event_id": "r3",
            "source_event_id": "e3",
            "chunk_id": "chunk-1",
            "model_id": "model-qwen",
            "reviewer": "h3",
            "action": "modify",
            "final_labels": [taxonomy.damage_labels[0]],
            "flags": [],
        },
        {
            "event_id": "r4",
            "source_event_id": "e4",
            "chunk_id": "chunk-2",
            "model_id": "model-consensus",
            "reviewer": "h1",
            "action": "accept",
            "final_labels": [taxonomy.damage_labels[1]],
            "flags": [],
        },
    ]
    write_jsonl_atomic(inference_path, inferences)
    write_jsonl_atomic(review_path, reviews)

    statistics = _production_feedback(inference_path, review_path, ready_path)
    ready = list(read_jsonl(ready_path))

    assert statistics["total_events"] == 4
    assert statistics["total_human_reviews"] == 4
    assert statistics["retraining_readiness"]["conflicting_chunks_excluded"] == 1
    assert len(ready) == 1
    assert ready[0]["coarse_labels"] == [taxonomy.damage_labels[1]]
    assert ready[0]["exclude_from_existing_validation_test"] is True


def test_review_flags_require_a_final_damage_category():
    taxonomy = load_taxonomy()
    base = {
        "event_id": "review-1",
        "chunk_id": "chunk-1",
        "reviewer": "reviewer-fixture",
        "proposed_labels": [taxonomy.safe_label],
    }
    with pytest.raises(ValueError, match="flags requieren"):
        ReviewEvent(
            **base,
            action="modify",
            final_labels=[taxonomy.safe_label],
            flags=[taxonomy.flags[0]],
        )
    accepted = ReviewEvent(
        **base,
        action="modify",
        final_labels=[taxonomy.damage_labels[0]],
        flags=[taxonomy.flags[0]],
    )
    assert accepted.flags == [taxonomy.flags[0]]
