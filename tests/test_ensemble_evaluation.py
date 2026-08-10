from __future__ import annotations

import json

import numpy as np

from moderacion_peru import ensemble_evaluation
from moderacion_peru.io import sha256_file, write_json_atomic, write_jsonl_atomic


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

    def fake_scores(candidate, scored_rows, *, device):
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
