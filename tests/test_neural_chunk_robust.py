from __future__ import annotations

import pytest

from moderacion_peru.neural_chunk_robust import (
    _select_minilm_noninferiority_anchors,
    build_hierarchical_neural_synthesis,
    run_neural_chunk_robust_test,
)


def _classical(seconds: float = 30.0):
    return {"recommendation": {"recommended_seconds": seconds}}


def _neural(status: str, *, complete: bool = True, best: float = 30.0):
    return {
        "reporting_status": "complete" if complete else "partial",
        "interpretation": {
            "status": status,
            "best_point_estimate_seconds": best,
        },
    }


def test_hierarchical_synthesis_never_averages_or_changes_classical_decision():
    result = build_hierarchical_neural_synthesis(
        _classical(),
        _neural("conflict_with_classical_reference", best=20.0),
        _neural("concordant_with_classical_reference"),
    )

    assert result["final_recommended_seconds"] == 30.0
    assert result["decision_changed_by_neural_tests"] is False
    assert result["metric_aggregation_across_families"] == "none"
    assert result["independent_human_validation_required_to_override"] is True
    assert result["hierarchy_status"].startswith("conflict_hold_classical")


def test_hierarchical_synthesis_marks_partial_evidence():
    result = build_hierarchical_neural_synthesis(
        _classical(),
        _neural("concordant_with_classical_reference"),
        _neural("missing", complete=False),
    )

    assert result["hierarchy_status"] == "partial_neural_evidence_hold_classical"
    assert result["final_recommended_seconds"] == 30.0


def test_neural_profile_requires_all_five_predeclared_lengths(tmp_path):
    with pytest.raises(ValueError, match="15,20,25,30,35"):
        run_neural_chunk_robust_test(
            tmp_path / "transcripts.jsonl",
            tmp_path / "chunks.jsonl",
            tmp_path / "dataset.jsonl",
            tmp_path / "classical",
            tmp_path / "neural",
            candidate_seconds=(20, 30),
        )


def test_noninferiority_panel_uses_one_anchor_per_train_video_and_covers_damage():
    labels = (
        "RACISMO_DISCRIMINACION",
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
        "CONTENIDO_SEXUAL",
    )
    rows = []
    for label_index, label in enumerate(labels):
        for video_index in range(2):
            video_id = f"damage-{label_index}-{video_index}"
            rows.append(
                {
                    "chunk_id": f"{video_id}-anchor",
                    "video_id": video_id,
                    "coarse_labels": [label],
                    "split": "train",
                }
            )
    rows.extend(
        {
            "chunk_id": f"safe-{index}-anchor",
            "video_id": f"safe-{index}",
            "coarse_labels": ["SEGURO"],
            "split": "train",
        }
        for index in range(4)
    )

    selected = _select_minilm_noninferiority_anchors(
        rows,
        panel_size=10,
        minimum_damage_videos_per_label=1,
        selection_seed=7,
    )

    assert len(selected) == 10
    assert len({row["video_id"] for row in selected}) == 10
    for label in labels:
        assert any(label in row["coarse_labels"] for row in selected)
