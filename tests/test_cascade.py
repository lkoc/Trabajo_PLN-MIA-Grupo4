from __future__ import annotations

import numpy as np

from moderacion_peru.cascade import (
    calibrate_safety_first_gate,
    combine_safety_first_cascade_scores,
)


def test_safety_first_gate_maximizes_threshold_under_recall_and_npv_constraints():
    truth = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.7])

    calibration = calibrate_safety_first_gate(
        truth,
        scores,
        min_damage_recall=1.0,
        min_safe_npv=1.0,
    )

    assert calibration["threshold"] == 0.7
    assert calibration["damage_recall"] == 1.0
    assert calibration["safe_npv"] == 1.0
    assert calibration["blocked_damage_rows"] == 0
    assert calibration["constraint_status"] == "satisfied_with_safe_decisions"


def test_safety_first_gate_falls_back_to_route_all_and_branch_can_recover_safe():
    fallback = calibrate_safety_first_gate(
        np.asarray([0, 1], dtype=np.int8),
        np.asarray([0.9, 0.1]),
        min_damage_recall=1.0,
        min_safe_npv=1.0,
    )
    gate = np.asarray([0.05, 0.8, 0.9])
    branch = np.asarray(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.95, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.9, 0.1, 0.1, 0.1],
        ]
    )
    combined = combine_safety_first_cascade_scores(
        gate,
        branch,
        gate_threshold=0.7,
    )

    assert fallback["constraint_status"] == "fallback_route_all"
    assert fallback["threshold"] == 0.0
    assert combined[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert combined[1].tolist() == branch[1].tolist()
    assert combined[1, 0] == 0.95
    assert combined[2, 1] == 0.9
