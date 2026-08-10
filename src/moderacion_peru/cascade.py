from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_GATE_MIN_DAMAGE_RECALL = 0.99
DEFAULT_GATE_MIN_SAFE_NPV = 0.99


def calibrate_safety_first_gate(
    damage_truth: np.ndarray,
    gate_scores: np.ndarray,
    *,
    min_damage_recall: float = DEFAULT_GATE_MIN_DAMAGE_RECALL,
    min_safe_npv: float = DEFAULT_GATE_MIN_SAFE_NPV,
) -> dict[str, Any]:
    """Selecciona en validation la mayor frontera que satisface dos compuertas.

    ``min_damage_recall`` limita la fracción de daños bloqueados y
    ``min_safe_npv`` limita la fracción dañina entre los casos declarados seguros.
    Si ninguna frontera no trivial es factible, 0.0 envía todo a la segunda rama.
    """

    truth = np.asarray(damage_truth, dtype=np.int8).reshape(-1)
    scores = np.asarray(gate_scores, dtype=float).reshape(-1)
    if truth.shape != scores.shape or not len(truth):
        raise ValueError("damage_truth y gate_scores deben tener igual longitud no vacía")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("damage_truth debe ser binario")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("gate_scores debe contener probabilidades finitas entre 0 y 1")
    if not 0 < min_damage_recall <= 1:
        raise ValueError("min_damage_recall debe pertenecer a (0, 1]")
    if not 0 < min_safe_npv <= 1:
        raise ValueError("min_safe_npv debe pertenecer a (0, 1]")
    damage_total = int(truth.sum())
    safe_total = int(len(truth) - damage_total)
    if damage_total == 0 or safe_total == 0:
        raise ValueError("La calibración necesita ejemplos seguros y dañinos en validation")

    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    feasible: list[dict[str, Any]] = []
    for threshold in thresholds:
        routed = scores >= threshold
        predicted_safe = ~routed
        blocked_damage = int(((truth == 1) & predicted_safe).sum())
        true_safe = int(((truth == 0) & predicted_safe).sum())
        safe_decisions = int(predicted_safe.sum())
        damage_recall = 1.0 - blocked_damage / damage_total
        safe_npv = (
            true_safe / safe_decisions
            if safe_decisions
            else 1.0
        )
        row = {
            "threshold": float(threshold),
            "damage_recall": float(damage_recall),
            "safe_npv": float(safe_npv),
            "blocked_damage_rows": blocked_damage,
            "safe_decision_rows": safe_decisions,
            "routed_rows": int(routed.sum()),
        }
        if (
            safe_decisions
            and damage_recall >= min_damage_recall
            and safe_npv >= min_safe_npv
        ):
            feasible.append(row)

    # La salida conservadora es explícitamente threshold=0: no emite SEGURO.
    selected = (
        max(feasible, key=lambda row: row["threshold"])
        if feasible
        else {
            "threshold": 0.0,
            "damage_recall": 1.0,
            "safe_npv": 1.0,
            "blocked_damage_rows": 0,
            "safe_decision_rows": 0,
            "routed_rows": len(truth),
        }
    )
    routed_fraction = selected["routed_rows"] / len(truth)
    return {
        **selected,
        "minimum_damage_recall": float(min_damage_recall),
        "minimum_safe_npv": float(min_safe_npv),
        "blocked_damage_fraction": float(
            selected["blocked_damage_rows"] / damage_total
        ),
        "false_omission_rate_among_gate_safe": float(1.0 - selected["safe_npv"]),
        "routed_to_branch_fraction": float(routed_fraction),
        "constraint_status": (
            "satisfied_with_safe_decisions"
            if selected["safe_decision_rows"]
            else "fallback_route_all"
        ),
        "selection_partition": "validation",
        "interpretation_scope": "empirical_validation_not_population_guarantee",
    }


def combine_safety_first_cascade_scores(
    gate_scores: np.ndarray,
    branch_scores: np.ndarray,
    *,
    gate_threshold: float,
) -> np.ndarray:
    """Aplica enrutamiento duro; la rama puede recuperar seguros falsamente derivados."""

    gate = np.asarray(gate_scores, dtype=float).reshape(-1)
    branch = np.asarray(branch_scores, dtype=float)
    if branch.ndim != 2 or branch.shape != (len(gate), 5):
        raise ValueError("La segunda rama debe aportar cinco scores por fila")
    if not 0 <= gate_threshold <= 1:
        raise ValueError("gate_threshold debe pertenecer a [0, 1]")
    routed = gate >= gate_threshold
    combined = np.zeros_like(branch, dtype=float)
    combined[:, 0] = 1.0
    combined[routed] = branch[routed]
    return combined
