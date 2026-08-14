from __future__ import annotations

import json

import numpy as np

from moderacion_peru import ensemble_evaluation
from moderacion_peru.io import sha256_file, write_json_atomic, write_jsonl_atomic
from moderacion_peru.taxonomy import load_taxonomy


def test_validation_candidate_audit_explains_eligible_and_rejected_rows(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    taxonomy = load_taxonomy()
    candidate_root = tmp_path / "candidates"
    eligible_dir = candidate_root / "eligible"
    eligible_dir.mkdir(parents=True)
    write_json_atomic(
        eligible_dir / "candidate.json",
        {
            "candidate_id": "eligible-e5",
            "model_family": "flat_e5",
            "status": "complete",
            "dataset_sha256": sha256_file(dataset),
            "target_labels": list(taxonomy.target_labels),
            "test_metrics": None,
        },
    )
    (eligible_dir / "predictions_validation.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    rejected_dir = candidate_root / "rejected"
    rejected_dir.mkdir()
    write_json_atomic(
        rejected_dir / "candidate.json",
        {
            "candidate_id": "incomplete-old-snapshot",
            "model_family": "qwen_prompt_sft",
            "status": "interrupted",
            "dataset_sha256": "0" * 64,
            "target_labels": list(taxonomy.target_labels),
            "test_metrics": None,
        },
    )

    audit = ensemble_evaluation.audit_validation_candidate_eligibility(
        dataset, [candidate_root]
    )

    assert audit["discovered_count"] == 2
    assert audit["eligible_count"] == 1
    assert audit["eligible"][0]["candidate_id"] == "eligible-e5"
    assert audit["rejected_count"] == 1
    assert set(audit["rejected"][0]["reasons"]) == {
        "incomplete",
        "different_snapshot",
        "validation_predictions_missing",
    }


def test_grouped_bootstrap_is_deterministic_across_worker_counts():
    truth = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
        * 4,
        dtype=np.int8,
    )
    scores = truth * 0.8 + 0.1
    videos = [f"video-{index // 2}" for index in range(len(truth))]

    serial = ensemble_evaluation._grouped_bootstrap_macro_ap(
        truth, scores, videos, replicates=24, seed=17, parallel_workers=1
    )
    parallel = ensemble_evaluation._grouped_bootstrap_macro_ap(
        truth, scores, videos, replicates=24, seed=17, parallel_workers=4
    )

    assert serial["samples"] == parallel["samples"]
    assert serial["point"] == parallel["point"]
    assert serial["parallel_workers"] == 1
    assert parallel["parallel_workers"] == 4


def test_crossfit_binary_policy_optimizes_balanced_accuracy_by_video():
    truth = np.asarray(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
        * 5,
        dtype=np.int8,
    )
    scores = truth * 0.8 + 0.1
    videos = [f"video-{index // 2}" for index in range(len(truth))]

    policy = ensemble_evaluation._crossfit_binary_policy(
        truth, scores, videos, folds=5
    )

    metrics = policy["oof_metrics"]
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["false_positive_rate"] == 0.0
    assert set(metrics["risk_lambda"]) == {"0.50", "0.67", "0.80"}


def test_selection_key_does_not_double_count_component_metrics():
    metrics = {
        "binary_any_damage_oof": {
            "balanced_accuracy": 0.81,
            "risk_lambda": {"0.67": 0.22},
            "false_negative_rate": 0.30,
            "false_positive_rate": 0.08,
        },
        "average_precision_macro_damage": 0.44,
        "f1_macro_damage": 0.99,
        "review_load_rate": 0.0,
    }

    assert ensemble_evaluation._selection_key(metrics) == (0.81, -0.22, 0.44)


def test_prompt_sft_test_inference_reuses_budgeted_generation_profile(
    tmp_path, monkeypatch
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    provenance = tmp_path / "prompt_provenance.json"
    write_json_atomic(provenance, {"capsule": "prompt"})
    captured = {}

    class _Model:
        def to(self, device):
            return self

    class _Peft:
        @staticmethod
        def from_pretrained(path):
            return _Model()

    class _Tokenizer:
        @staticmethod
        def from_pretrained(path):
            return object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "peft",
        type("PeftModule", (), {"AutoPeftModelForCausalLM": _Peft}),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("TransformersModule", (), {"AutoTokenizer": _Tokenizer}),
    )
    monkeypatch.setattr(
        ensemble_evaluation,
        "resolve_device",
        lambda device: type("Hardware", (), {"backend": "cpu"})(),
    )
    monkeypatch.setattr(ensemble_evaluation, "torch_device_name", lambda hardware: "cpu")

    from moderacion_peru import prompt_sft

    def fake_generate(model, tokenizer, rows, prompt, **kwargs):
        captured.update(kwargs)
        return np.zeros((len(rows), 22)), {}

    monkeypatch.setattr(prompt_sft, "_generate_json_scores", fake_generate)
    candidate_path = tmp_path / "candidate.json"
    candidate = {
        "candidate_path": str(candidate_path),
        "inference": {
            "type": "hf_prompt_sft_json",
            "model": "adapter",
            "prompt_capsule": "prompt_provenance.json",
            "max_input_length": 2368,
            "max_new_tokens": 192,
            "batch_size": 8,
        },
    }

    scores = ensemble_evaluation._score_candidate(
        candidate, [{"text": "texto"}], device="cpu"
    )

    assert scores.shape == (1, 5)
    assert captured["max_input_length"] == 2368
    assert captured["max_new_tokens"] == 192
    assert captured["batch_size"] == 8


def test_frozen_test_reports_natural_and_four_to_one_from_one_inference(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for index in range(12):
        rows.append(
            {
                "chunk_id": f"safe-{index}",
                "video_id": f"safe-video-{index}",
                "channel_id": f"channel-{index % 2}",
                "text": "contenido seguro",
                "coarse_labels": ["SEGURO"],
                "split": "test",
            }
        )
    for index in range(2):
        rows.append(
            {
                "chunk_id": f"damage-{index}",
                "video_id": f"damage-video-{index}",
                "channel_id": f"channel-{index % 2}",
                "text": "amenaza",
                "coarse_labels": ["ACOSO_AMENAZA"],
                "split": "test",
            }
        )
    write_jsonl_atomic(dataset, rows)

    candidate_path = tmp_path / "candidate.json"
    write_json_atomic(
        candidate_path,
        {
            "candidate_id": "fixture",
            "candidate_path": str(candidate_path),
            "inference": {"type": "fixture"},
            "training_sampling": {
                "split_field": "split",
                "safe_to_damage_ratio": 4.0,
                "sampling_seed": 7,
            },
        },
    )
    freeze_path = tmp_path / "freeze.json"
    write_json_atomic(
        freeze_path,
        {
            "comparison_signature": "fixture-signature",
            "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "member_candidate_paths": {"fixture": str(candidate_path)},
            "selected_id": "fixture",
            "selected_kind": "individual",
            "test_status": "sealed_ready_for_single_open",
            "score_calibrators": [
                {"type": "sigmoid_platt", "coefficient": 10.0, "intercept": -5.0}
                for _ in range(5)
            ],
            "any_damage_threshold": 0.5,
            "needs_review_policy": {"selected_delta": 0.0},
            "thresholds": {
                "SEGURO": 0.5,
                "RACISMO_DISCRIMINACION": 0.5,
                "ATAQUE_POR_GENERO_IDENTIDAD": 0.5,
                "ACOSO_AMENAZA": 0.5,
                "CONTENIDO_SEXUAL": 0.5,
            },
        },
    )
    calls = []

    def fake_scores(
        candidate,
        scored_rows,
        *,
        device,
        progress_callback=None,
        progress_phase="inferencia",
    ):
        calls.append([row["chunk_id"] for row in scored_rows])
        scores = np.full((len(scored_rows), 5), 0.01, dtype=float)
        for index, row in enumerate(scored_rows):
            scores[index, 0 if row["coarse_labels"] == ["SEGURO"] else 3] = 0.99
        return scores

    monkeypatch.setattr(ensemble_evaluation, "_score_candidate", fake_scores)
    report_path = tmp_path / "test_report.json"
    result = ensemble_evaluation.evaluate_frozen_test(
        freeze_path,
        report_path,
        confirm_single_test_open=True,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["status"] == "test_evaluated_once"
    assert len(calls) == 1
    assert len(calls[0]) == 14
    assert payload["inference_passes"] == 1
    assert payload["test_rows_natural"] == 14
    assert payload["test_rows_4_to_1"] == 10
    assert payload["metrics"] == payload["primary_metrics_natural_prevalence"]
    assert "secondary_metrics_4_to_1" in payload
    assert set(payload["stage_timings_seconds"]) == {
        "test_inference_all_selected_members",
        "natural_and_4_to_1_metrics",
        "test_total_before_report_write",
    }
