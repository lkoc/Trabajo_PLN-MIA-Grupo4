from __future__ import annotations

import json
from datetime import datetime, timezone

from moderacion_peru.consolidation import consolidate_annotations, reconcile_human_reviews
from moderacion_peru.datasets import materialize_versioned_training_snapshot
from moderacion_peru.experiments import train_classical_experiments
from moderacion_peru import experiments
from moderacion_peru.io import read_jsonl, write_jsonl_atomic
from moderacion_peru.registry import ProductionPredictor, compare_and_publish_registry
from moderacion_peru.schemas import AnnotationRecord, ModelReadyRecord, ReviewEvent
from moderacion_peru.taxonomy import load_taxonomy


def test_human_events_close_annotation_to_versioned_snapshot_and_noop(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write_jsonl_atomic(
        chunks,
        [
            {
                "chunk_id": "video_with_underscore_deadbeef",
                "video_id": "video_with_underscore",
                "text": "ataque",
                "start_seconds": 10.0,
                "end_seconds": 20.0,
            }
        ],
    )
    consolidated = tmp_path / "consolidated.jsonl"
    base = AnnotationRecord(
        chunk_id="video_with_underscore_deadbeef",
        text="ataque",
        coarse_labels=["ACOSO_AMENAZA"],
        flags=["contexto_necesario"],
        label_source="ollama_local",
        annotator_type="llm_local",
        annotator_model="fixture",
    )
    write_jsonl_atomic(consolidated, [base.model_dump(mode="json")])
    reviews = tmp_path / "reviews.jsonl"
    event = ReviewEvent(
        event_id="event-1",
        chunk_id=base.chunk_id,
        action="modify",
        proposed_labels=["ACOSO_AMENAZA"],
        final_labels=["ATAQUE_POR_GENERO_IDENTIDAD"],
        flags=["humor_encubridor"],
        reviewer="reviewer-fixture",
        created_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )
    write_jsonl_atomic(reviews, [event.model_dump(mode="json")])
    reviewed = tmp_path / "reviewed.jsonl"

    reconciliation_progress = []
    first = reconcile_human_reviews(
        consolidated,
        [reviews],
        reviewed,
        chunks_source=chunks,
        progress_callback=reconciliation_progress.append,
    )
    second = reconcile_human_reviews(consolidated, [reviews], reviewed, chunks_source=chunks)
    assert first["status"] == "updated"
    assert second["status"] == "noop"
    row = next(read_jsonl(reviewed))
    assert row["video_id"] == "video_with_underscore"
    assert row["coarse_labels"] == ["ATAQUE_POR_GENERO_IDENTIDAD"]
    assert row["flags"] == ["humor_encubridor"]
    assert row["label_source"] == "human_modified"
    assert {event["phase"] for event in reconciliation_progress} >= {
        "loading_chunks",
        "loading_review_events",
        "reconciling",
        "reconciliation",
    }
    assert reconciliation_progress[-1]["status"] == "finished"

    canonical = tmp_path / "model_ready" / "dataset_5_salidas.jsonl"
    snapshot_progress = []
    snapshot_first = materialize_versioned_training_snapshot(
        reviewed, canonical, progress_callback=snapshot_progress.append
    )
    snapshot_second = materialize_versioned_training_snapshot(reviewed, canonical)
    assert snapshot_first["status"] == "updated"
    assert snapshot_second["status"] == "noop"
    ready = ModelReadyRecord.model_validate(next(read_jsonl(canonical)))
    assert ready.video_id == "video_with_underscore"
    assert ready.flags_reference_only == ["humor_encubridor"]
    assert ready.split in {"train", "validation", "test"}
    assert {event["phase"] for event in snapshot_progress} >= {
        "preparing_snapshot",
        "deduplicating_snapshot",
        "validating_video_splits",
        "snapshot",
    }
    assert snapshot_progress[-1]["status"] == "finished"


def test_annotation_consolidation_reports_each_long_phase(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write_jsonl_atomic(
        chunks,
        [{"chunk_id": "c1", "video_id": "v1", "text": "texto", "start_seconds": 0.0}],
    )
    transcripts = tmp_path / "transcripts.jsonl"
    write_jsonl_atomic(transcripts, [{"video_id": "v1", "title": "Video"}])
    local = tmp_path / "local.jsonl"
    write_jsonl_atomic(
        local,
        [
            AnnotationRecord(
                chunk_id="c1",
                video_id="v1",
                text="texto",
                coarse_labels=["SEGURO"],
                fine_labels=["seguro"],
                label_source="ollama_local",
                annotator_type="llm_local",
            ).model_dump(mode="json")
        ],
    )
    destination = tmp_path / "consolidated.jsonl"
    progress = []
    result = consolidate_annotations(
        [local],
        destination,
        chunks_source=chunks,
        transcripts_source=transcripts,
        progress_callback=progress.append,
    )

    assert result == {"status": "updated", "chunks": 1, "conflicts": 0}
    assert {event["phase"] for event in progress} >= {
        "loading_annotations",
        "loading_chunks",
        "loading_transcripts",
        "consolidating",
        "consolidation",
    }
    assert progress[-1]["status"] == "finished"


def test_annotation_consolidation_batches_notebook_progress(tmp_path):
    source = tmp_path / "annotations.jsonl"
    write_jsonl_atomic(
        source,
        [
            {
                "chunk_id": f"c{index}",
                "video_id": "v1",
                "text": f"texto {index}",
                "coarse_labels": ["SEGURO"],
                "label_source": "ollama_local",
            }
            for index in range(2501)
        ],
    )
    progress = []

    consolidate_annotations(
        [source],
        tmp_path / "consolidated.jsonl",
        progress_callback=progress.append,
    )

    for phase in ("loading_annotations", "consolidating"):
        advances = [
            event["advance"]
            for event in progress
            if event["phase"] == phase and event["status"] == "progress"
        ]
        assert advances == [1000, 1000, 501]


def test_rejected_human_event_is_valid_but_never_trains(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    write_jsonl_atomic(chunks, [{"chunk_id": "c1", "video_id": "v1", "text": "x"}])
    consolidated = tmp_path / "base.jsonl"
    write_jsonl_atomic(
        consolidated,
        [
            AnnotationRecord(
                chunk_id="c1",
                video_id="v1",
                text="x",
                coarse_labels=["SEGURO"],
                label_source="ollama_local",
                annotator_type="llm_local",
            ).model_dump(mode="json")
        ],
    )
    reviews = tmp_path / "reviews.jsonl"
    write_jsonl_atomic(
        reviews,
        [
            ReviewEvent(
                event_id="reject-1",
                chunk_id="c1",
                action="reject",
                reviewer="reviewer-fixture",
            ).model_dump(mode="json")
        ],
    )
    reviewed = tmp_path / "reviewed.jsonl"
    reconcile_human_reviews(consolidated, [reviews], reviewed, chunks_source=chunks)
    row = AnnotationRecord.model_validate(next(read_jsonl(reviewed)))
    assert row.decision_status == "excluded"
    assert not row.training_eligible
    assert row.coarse_labels == []


def _fixture_dataset(path):
    taxonomy = load_taxonomy()
    rows = []
    for split, prefix in (("train", "train"), ("validation", "valid"), ("test", "test")):
        for index in range(15):
            label = taxonomy.target_labels[index % len(taxonomy.target_labels)]
            rows.append(
                ModelReadyRecord(
                    chunk_id=f"{prefix}-chunk-{index}",
                    video_id=f"{prefix}-video-{index}",
                    text=f"{label.lower()} ejemplo peruano número {index} {label.lower()}",
                    coarse_labels=[label],
                    label_source="human_modified",
                    split=split,
                ).model_dump(mode="json")
            )
    write_jsonl_atomic(path, rows)


def test_classical_fit_calibration_test_registry_and_inference_are_end_to_end(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _fixture_dataset(dataset)
    model_root = tmp_path / "models"
    trained = train_classical_experiments(dataset, model_root)
    repeated = train_classical_experiments(dataset, model_root)
    assert trained["status"] == "trained"
    assert repeated["status"] == "noop"
    assert len(trained["candidates"]) == 5
    for candidate in trained["candidates"]:
        assert set(candidate["thresholds"]) == set(load_taxonomy().target_labels)
        assert "false_safe_rate_on_damage" in candidate["validation_metrics"]
        assert "test_metrics" in candidate

    registry = tmp_path / "registro.json"
    published = compare_and_publish_registry(dataset, [model_root], registry)
    repeated_publish = compare_and_publish_registry(dataset, [model_root], registry)
    assert published["status"] == "created"
    assert repeated_publish["status"] == "noop"
    entry = json.loads(registry.read_text(encoding="utf-8"))
    assert entry["status"] == "validated"
    assert entry["dataset_sha256"]

    scores = ProductionPredictor(registry).scores("contenido seguro ejemplo peruano")
    assert set(scores) == set(load_taxonomy().target_labels)
    assert all(0 <= score <= 1 for score in scores.values())


def test_classical_smoke_subset_reuses_complete_training_path(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _fixture_dataset(dataset)
    result = train_classical_experiments(
        dataset,
        tmp_path / "smoke_models",
        model_names=("complement_nb", "sgd_incremental"),
        max_features=500,
    )
    assert result["status"] == "trained"
    assert {candidate["experiment"] for candidate in result["candidates"]} == {
        "complement_nb",
        "sgd_incremental",
    }


def test_each_neural_notebook_path_reaches_candidate_with_mocked_backbone(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    _fixture_dataset(dataset)

    class DummyModel:
        pass

    monkeypatch.setattr(experiments, "_build_hf_model", lambda *args, **kwargs: (object(), DummyModel()))
    monkeypatch.setattr(experiments, "_fit_hf", lambda *args, **kwargs: None)

    def fake_predict(model, tokenizer, rows, max_length, hardware, output_count):
        coarse = experiments.encode_targets(rows).astype(float)
        if output_count == 1:
            return coarse[:, 1:].max(axis=1, keepdims=True) * 0.8 + 0.1
        if output_count == 4:
            return coarse[:, 1:] * 0.8 + 0.1
        output = __import__("numpy").zeros((len(rows), output_count), dtype=float) + 0.1
        output[:, :5] = coarse * 0.8 + 0.1
        return output

    def fake_save(model, tokenizer, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"fixture")

    monkeypatch.setattr(experiments, "_predict_hf", fake_predict)
    monkeypatch.setattr(experiments, "_save_hf", fake_save)

    for experiment in ("flat_minilm", "flat_e5", "cascade", "multitask", "qwen_lora", "qwen_structured"):
        output = tmp_path / experiment
        first = experiments.train_neural_experiment(
            dataset,
            output,
            experiment=experiment,
            device="cpu",
        )
        second = experiments.train_neural_experiment(
            dataset,
            output,
            experiment=experiment,
            device="cpu",
        )
        assert first["status"] == "trained"
        assert second["status"] == "noop"
        assert set(first["candidate"]["thresholds"]) == set(load_taxonomy().target_labels)
        assert first["candidate"]["checkpoint_manifest"] == "checkpoint_manifest.json"
