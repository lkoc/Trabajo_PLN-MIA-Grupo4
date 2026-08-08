import json
from pathlib import Path

import pytest

from moderacion_peru.io import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from moderacion_peru.registry import compare_and_publish_registry
from moderacion_peru.schemas import ReviewEvent
from moderacion_peru.servers import (
    _consensus_result,
    _production_feedback,
    _production_registry_paths,
)
from moderacion_peru.taxonomy import load_taxonomy


def _candidate(root: Path, dataset: Path, family: str, identifier: str, score: float) -> None:
    taxonomy = load_taxonomy()
    directory = root / identifier
    directory.mkdir(parents=True)
    write_json_atomic(directory / "checkpoint_manifest.json", {"files": []})
    write_json_atomic(directory / "metrics.json", {"fixture": True})
    write_json_atomic(directory / "inference.json", {"type": "sklearn_joblib", "model": "unused.joblib"})
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
        assert json.loads(member.read_text(encoding="utf-8"))["model_id"] == result["selected_by_slot"][slot]


def _model_event(event_id: str, slot: str, labels: list[str], text: str = "texto uno") -> dict:
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


def test_production_feedback_is_append_only_linked_deduplicated_and_conflict_safe(tmp_path):
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
        {"event_id": "r1", "source_event_id": "e1", "chunk_id": "chunk-1", "model_id": "model-classical", "reviewer": "h1", "action": "accept", "final_labels": [taxonomy.safe_label], "flags": []},
        {"event_id": "r2", "source_event_id": "e2", "chunk_id": "chunk-1", "model_id": "model-transformer", "reviewer": "h2", "action": "accept", "final_labels": [taxonomy.safe_label], "flags": []},
        {"event_id": "r3", "source_event_id": "e3", "chunk_id": "chunk-1", "model_id": "model-qwen", "reviewer": "h3", "action": "modify", "final_labels": [taxonomy.damage_labels[0]], "flags": []},
        {"event_id": "r4", "source_event_id": "e4", "chunk_id": "chunk-2", "model_id": "model-consensus", "reviewer": "h1", "action": "accept", "final_labels": [taxonomy.damage_labels[1]], "flags": []},
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
