from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from moderacion_peru import experiments
from moderacion_peru.consolidation import (
    consolidate_annotations,
    reconcile_human_reviews,
)
from moderacion_peru.datasets import (
    deterministic_safe_downsample,
    materialize_versioned_training_snapshot,
)
from moderacion_peru.experiments import train_classical_experiments
from moderacion_peru.io import read_jsonl, write_json_atomic, write_jsonl_atomic
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
                "channel_id": "channel-fixture",
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
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
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
    second = reconcile_human_reviews(
        consolidated, [reviews], reviewed, chunks_source=chunks
    )
    assert first["status"] == "updated"
    assert second["status"] == "noop"
    row = next(read_jsonl(reviewed))
    assert row["video_id"] == "video_with_underscore"
    assert row["channel_id"] == "channel-fixture"
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
    assert ready.channel_id == "channel-fixture"
    assert ready.flags_reference_only == ["humor_encubridor"]
    assert ready.split in {"train", "validation", "test"}
    assert ready.channel_split in {"train", "validation", "test"}
    assert len(ready.coarse_observed_mask) == 5
    assert len(ready.fine_observed_mask) == 14
    assert len(ready.flags_observed_mask) == 3
    assert ready.flags_observed_mask == [1, 1, 1]
    assert {event["phase"] for event in snapshot_progress} >= {
        "preparing_snapshot",
        "deduplicating_snapshot",
        "validating_video_splits",
        "snapshot",
    }
    assert snapshot_progress[-1]["status"] == "finished"


def test_safe_downsample_is_deterministic_and_never_removes_damage():
    rows = [
        {
            "chunk_id": f"safe-{index}",
            "channel_id": f"channel-{index % 3}",
            "coarse_labels": ["SEGURO"],
        }
        for index in range(30)
    ] + [
        {
            "chunk_id": f"damage-{index}",
            "channel_id": f"channel-{index % 3}",
            "coarse_labels": ["ACOSO_AMENAZA"],
        }
        for index in range(5)
    ]
    first, summary = deterministic_safe_downsample(
        rows, safe_to_damage_ratio=4.0, seed=7
    )
    second, repeated = deterministic_safe_downsample(
        rows, safe_to_damage_ratio=4.0, seed=7
    )
    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    assert summary == repeated
    assert summary["safe_rows_after"] == 20
    assert summary["damage_rows"] == 5
    assert {f"damage-{index}" for index in range(5)} <= {
        row["chunk_id"] for row in first
    }


def test_training_splits_reduce_train_validation_but_keep_full_test(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            {
                "chunk_id": f"{split}-safe-{index}",
                "channel_id": f"channel-{index % 2}",
                "coarse_labels": ["SEGURO"],
                "split": split,
            }
            for index in range(12)
        )
        rows.extend(
            {
                "chunk_id": f"{split}-damage-{index}",
                "channel_id": f"channel-{index % 2}",
                "coarse_labels": ["ACOSO_AMENAZA"],
                "split": split,
            }
            for index in range(2)
        )
    write_jsonl_atomic(dataset, rows)

    train, validation, test, sampling = experiments._dataset_splits(dataset)

    assert len(train) == 10
    assert len(validation) == 10
    assert len(test) == 14
    assert sampling["policy"] == "fixed_4_to_1_train_validation_full_sealed_test"
    assert sampling["test_sealed"]["policy"] == "full_natural_prevalence"
    assert sampling["test_reporting"]["single_inference_pass"] is True


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


def test_historical_recovery_keeps_flash_pro_precedence(tmp_path):
    flash = tmp_path / "flash.jsonl"
    write_jsonl_atomic(
        flash,
        [
            {
                "chunk_id": "c1",
                "video_id": "v1",
                "text": "texto",
                "coarse_labels": [],
                "fine_labels": [],
                "needs_review": True,
                "training_eligible": False,
                "decision_status": "needs_review",
                "label_source": "deepseek_remote_historical_recovered",
                "annotator_model": "deepseek-v4-flash",
                "created_at": "2026-08-08T18:00:00Z",
            }
        ],
    )
    pro = tmp_path / "pro.jsonl"
    write_jsonl_atomic(
        pro,
        [
            {
                "chunk_id": "c1",
                "video_id": "v1",
                "text": "texto",
                "coarse_labels": ["SEGURO"],
                "fine_labels": ["seguro"],
                "needs_review": False,
                "training_eligible": True,
                "decision_status": "resolved",
                "label_source": "llm_remote_review_historical_recovered",
                "annotator_model": "deepseek-v4-pro",
                "created_at": "2026-08-08T20:00:00Z",
            }
        ],
    )
    destination = tmp_path / "consolidated.jsonl"

    result = consolidate_annotations([flash, pro], destination)
    row = next(read_jsonl(destination))

    assert result == {"status": "updated", "chunks": 1, "conflicts": 0}
    assert row["coarse_labels"] == ["SEGURO"]
    assert row["annotator_model"] == "deepseek-v4-pro"
    assert row.get("consolidation_warning") is None


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
    for split, prefix in (
        ("train", "train"),
        ("validation", "valid"),
        ("test", "test"),
    ):
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
        assert "fit_quality" in candidate

    svm = next(
        candidate
        for candidate in trained["candidates"]
        if candidate["model_family"].endswith(":linear_svm")
    )
    assert svm["fit_quality"]["converged"] is True
    legacy_svm = json.loads(Path(svm["candidate_path"]).read_text(encoding="utf-8"))
    legacy_svm.pop("fit_quality")
    write_json_atomic(svm["candidate_path"], legacy_svm)

    registry = tmp_path / "registro.json"
    published = compare_and_publish_registry(dataset, [model_root], registry)
    repeated_publish = compare_and_publish_registry(dataset, [model_root], registry)
    assert published["status"] == "created"
    assert repeated_publish["status"] == "noop"
    comparison = json.loads(
        (tmp_path / "comparacion_modelos_5_salidas.json").read_text(encoding="utf-8")
    )
    rejected_svm = next(
        row
        for row in comparison["rejected"]
        if row["candidate_id"] == svm["candidate_id"]
    )
    assert "svm_convergence_not_verified" in rejected_svm["reasons"]
    entry = json.loads(registry.read_text(encoding="utf-8"))
    assert entry["status"] == "validated"
    assert entry["dataset_sha256"]

    scores = ProductionPredictor(registry).scores("contenido seguro ejemplo peruano")
    assert set(scores) == set(load_taxonomy().target_labels)
    assert all(0 <= score <= 1 for score in scores.values())


def test_classical_smoke_subset_reuses_features_and_parallel_heads(
    tmp_path, monkeypatch
):
    from sklearn.pipeline import FeatureUnion

    fit_transform_calls = []
    original_fit_transform = FeatureUnion.fit_transform

    def tracked_fit_transform(self, *args, **kwargs):
        fit_transform_calls.append(1)
        return original_fit_transform(self, *args, **kwargs)

    monkeypatch.setattr(FeatureUnion, "fit_transform", tracked_fit_transform)
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
    assert len(fit_transform_calls) == 1
    assert all(
        candidate["runtime_optimization"]["parallel_workers"] == 4
        for candidate in result["candidates"]
    )
    assert all(
        candidate["runtime_optimization"]["feature_extraction"]["policy"]
        == "fit_once_per_variant_reuse_across_models"
        for candidate in result["candidates"]
    )
    assert all(
        set(candidate["stage_timings_seconds"])
        == {
            "shared_feature_extraction_variant",
            "model_fit",
            "validation_inference_and_metrics",
            "model_total_before_candidate_write",
        }
        for candidate in result["candidates"]
    )


def test_each_neural_notebook_path_reaches_candidate_with_mocked_backbone(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    _fixture_dataset(dataset)

    class DummyModel:
        pass

    monkeypatch.setattr(
        experiments, "_build_hf_model", lambda *args, **kwargs: (object(), DummyModel())
    )
    fit_primary_counts = []

    def fake_fit(*args, **kwargs):
        fit_primary_counts.append(kwargs.get("primary_output_count", 5))

    monkeypatch.setattr(experiments, "_fit_hf", fake_fit)

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

    for experiment in (
        "flat_minilm",
        "flat_e5",
        "cascade",
        "multitask",
        "qwen_lora",
        "qwen_structured",
    ):
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
        assert set(first["candidate"]["thresholds"]) == set(
            load_taxonomy().target_labels
        )
        assert first["candidate"]["checkpoint_manifest"] == "checkpoint_manifest.json"

    assert fit_primary_counts == [5, 5, 1, 4, 5, 5, 5]
