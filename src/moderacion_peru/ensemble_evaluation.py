from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cascade import combine_safety_first_cascade_scores
from .datasets import deterministic_safe_downsample
from .device import resolve_device, torch_device_name
from .io import (
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .registry import discover_candidates
from .taxonomy import load_taxonomy
from .training import calibrate_thresholds, classification_metrics, encode_targets

DEFAULT_BOOTSTRAP_WORKERS = 4
DEFAULT_SELECTION_FOLDS = 5
DEFAULT_REVIEW_DELTA_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15)
SELECTION_CRITERION_VERSION = "balanced-any-damage-oof-v1"
BOOTSTRAP_ENGINE = "paired-balanced-accuracy-grouped-video-threaded-v3"
ProgressCallback = Callable[[dict[str, Any]], None]


def _notify_progress(callback: ProgressCallback | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def _candidate_slot(candidate: Mapping[str, Any]) -> str:
    family = str(candidate.get("model_family", "")).casefold()
    if family.startswith("classical:"):
        return "classical"
    if family.startswith("qwen") or "prompt_sft" in family:
        return "qwen"
    return "transformer"


def audit_validation_candidate_eligibility(
    dataset_path: str | Path,
    candidate_roots: Iterable[str | Path],
) -> dict[str, Any]:
    """Explica qué candidatos pueden entrar en la comparación de validation.

    La auditoría comparte exactamente las mismas compuertas que
    :func:`compare_and_freeze_validation`. No abre ``test`` ni carga pesos.
    """

    dataset = Path(dataset_path).resolve()
    dataset_sha = sha256_file(dataset)
    taxonomy = load_taxonomy()
    discovered = discover_candidates(candidate_roots)
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in discovered:
        reasons: list[str] = []
        if candidate.get("status") != "complete":
            reasons.append("incomplete")
        if candidate.get("eligible_for_03_07") is False:
            reasons.append("explicitly_not_eligible_for_03_07")
        if (
            candidate.get("training_regime") == "budgeted_comparable"
            and candidate.get("training_budget", {}).get("validation_scope")
            != "full_common_validation"
        ):
            reasons.append("budgeted_candidate_without_full_common_validation")
        if candidate.get("dataset_sha256") != dataset_sha:
            reasons.append("different_snapshot")
        if tuple(candidate.get("target_labels", ())) != taxonomy.target_labels:
            reasons.append("wrong_contract")
        if candidate.get("test_metrics") not in (None, {}):
            reasons.append("test_was_opened_before_freeze")
        family = str(candidate.get("model_family", "")).casefold()
        if (
            family.endswith(":linear_svm")
            and candidate.get("fit_quality", {}).get("converged") is not True
        ):
            reasons.append("svm_convergence_not_verified")
        predictions = (
            Path(str(candidate["candidate_path"])).parent
            / "predictions_validation.jsonl"
        )
        if not predictions.is_file():
            reasons.append("validation_predictions_missing")
        if reasons:
            rejected.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "model_family": candidate.get("model_family"),
                    "candidate_path": candidate.get("candidate_path"),
                    "reasons": reasons,
                }
            )
        else:
            eligible.append(candidate)
    return {
        "dataset_sha256": dataset_sha,
        "discovered_count": len(discovered),
        "eligible_count": len(eligible),
        "eligible": eligible,
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Orden lexicográfico predeclarado; no es una suma de métricas."""

    binary = metrics.get("binary_any_damage_oof", metrics.get("any_damage", {}))
    return (
        float(binary.get("balanced_accuracy", 0.0)),
        -float(binary.get("risk_lambda", {}).get("0.67", 1.0)),
        float(
            metrics.get(
                "average_precision_macro_damage_oof",
                metrics.get("average_precision_macro_damage", 0.0),
            )
        ),
    )


def _load_validation_predictions(
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    taxonomy = load_taxonomy()
    path = (
        Path(str(candidate["candidate_path"])).parent / "predictions_validation.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(read_jsonl(path))
    scores = np.asarray(
        [
            [float(row["scores"][label]) for label in taxonomy.target_labels]
            for row in rows
        ],
        dtype=float,
    )
    return rows, scores


def _align_predictions(
    loaded: Sequence[tuple[Mapping[str, Any], list[dict[str, Any]], np.ndarray]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if not loaded:
        raise ValueError("No existen predicciones para alinear")
    reference_rows = loaded[0][1]
    reference_ids = [str(row["chunk_id"]) for row in reference_rows]
    aligned = []
    for candidate, rows, scores in loaded:
        index = {str(row["chunk_id"]): position for position, row in enumerate(rows)}
        if set(index) != set(reference_ids):
            raise ValueError(
                f"Validation no coincide para {candidate.get('candidate_id')}"
            )
        aligned.append(
            np.asarray([scores[index[chunk_id]] for chunk_id in reference_ids])
        )
    return reference_rows, aligned


def _pareto_front(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Frontera no dominada BA binaria--macro-AUPRC de daños."""

    vectors = {
        row["candidate_id"]: np.asarray(
            [
                row["validation_metrics"]["binary_any_damage_oof"][
                    "balanced_accuracy"
                ],
                row["validation_metrics"].get(
                    "average_precision_macro_damage_oof",
                    row["validation_metrics"]["average_precision_macro_damage"],
                ),
            ],
            dtype=float,
        )
        for row in rows
    }
    frontier = []
    for identifier, vector in vectors.items():
        dominated = any(
            other != identifier
            and np.all(other_vector >= vector)
            and np.any(other_vector > vector)
            for other, other_vector in vectors.items()
        )
        if not dominated:
            frontier.append(identifier)
    return sorted(frontier)


def _fit_sigmoid_calibrator(truth: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Ajusta Platt univariado o una constante si el fold es degenerado."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(truth, dtype=np.int8).reshape(-1)
    if np.unique(labels).size < 2:
        return {
            "type": "constant",
            "value": float(np.clip(labels.mean() if len(labels) else 0.5, 1e-6, 1 - 1e-6)),
        }
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    model.fit(values[:, None], labels)
    return {
        "type": "sigmoid_platt",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def _apply_calibrator(scores: np.ndarray, calibrator: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if calibrator["type"] == "constant":
        return np.full(values.shape, float(calibrator["value"]), dtype=float)
    logits = float(calibrator["coefficient"]) * values + float(
        calibrator["intercept"]
    )
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def _fit_score_calibrators(
    truth: np.ndarray, scores: np.ndarray
) -> list[dict[str, Any]]:
    return [
        _fit_sigmoid_calibrator(truth[:, index], scores[:, index])
        for index in range(scores.shape[1])
    ]


def _apply_score_calibrators(
    scores: np.ndarray, calibrators: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    if scores.shape[1] != len(calibrators):
        raise ValueError("La calibración no coincide con las salidas")
    return np.column_stack(
        [
            _apply_calibrator(scores[:, index], calibrator)
            for index, calibrator in enumerate(calibrators)
        ]
    )


def _balanced_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """Maximiza BA; los empates favorecen menor FNR y luego mayor umbral."""

    labels = np.asarray(truth, dtype=np.int8)
    values = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        raise ValueError("La compuerta ANY_DAMAGE requiere ambas clases")
    candidates = np.unique(
        np.concatenate(([0.0], np.linspace(0.01, 0.99, 99), values, [1.0]))
    )
    scored = []
    for threshold in candidates:
        predicted = values >= threshold
        recall = float(predicted[labels == 1].mean())
        specificity = float((~predicted[labels == 0]).mean())
        scored.append(((recall + specificity) / 2, recall, float(threshold)))
    return max(scored)[2]


def _binary_metrics(
    truth: np.ndarray, predicted: np.ndarray, scores: np.ndarray | None = None
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef

    labels = np.asarray(truth, dtype=np.int8)
    decisions = np.asarray(predicted, dtype=np.int8)
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(decisions[positives] == 1))
    fn = int(np.sum(decisions[positives] == 0))
    fp = int(np.sum(decisions[negatives] == 1))
    tn = int(np.sum(decisions[negatives] == 0))
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    fnr = 1 - recall
    fpr = 1 - specificity
    payload: dict[str, Any] = {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "prevalence_any_damage": float(labels.mean()),
        "recall": float(recall),
        "specificity": float(specificity),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "balanced_error_rate": float((fnr + fpr) / 2),
        "balanced_accuracy": float((recall + specificity) / 2),
        "f1": float(f1_score(labels, decisions, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(labels, decisions)),
        "risk_lambda": {
            f"{weight:.2f}": float(weight * fnr + (1 - weight) * fpr)
            for weight in (0.50, 0.67, 0.80)
        },
    }
    if scores is not None:
        payload["average_precision"] = float(average_precision_score(labels, scores))
    return payload


def _macro_damage_average_precision(
    truth: np.ndarray, scores: np.ndarray, indices: np.ndarray | None = None
) -> float:
    """Calcula macro-AUPRC solo sobre daños; una etiqueta ausente aporta cero."""

    from sklearn.metrics import average_precision_score

    taxonomy = load_taxonomy()
    selected = np.arange(len(truth), dtype=int) if indices is None else indices
    selected_truth = truth[selected]
    selected_scores = scores[selected]
    values = []
    for label in taxonomy.damage_labels:
        label_index = taxonomy.target_labels.index(label)
        label_truth = selected_truth[:, label_index]
        values.append(
            float(average_precision_score(label_truth, selected_scores[:, label_index]))
            if label_truth.any()
            else 0.0
        )
    return float(np.mean(values))


def _review_mask(
    calibrated_scores: np.ndarray,
    category_thresholds: np.ndarray,
    any_damage_scores: np.ndarray,
    any_damage_thresholds: np.ndarray,
    *,
    delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    taxonomy = load_taxonomy()
    safe_index = taxonomy.target_labels.index(taxonomy.safe_label)
    damage_indices = np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels]
    )
    active = calibrated_scores >= category_thresholds
    active_damage = active[:, damage_indices].any(axis=1)
    gate_damage = any_damage_scores >= any_damage_thresholds
    conflict = active[:, safe_index] & active_damage
    empty = ~active.any(axis=1)
    incoherent = gate_damage != active_damage
    near_gate = np.abs(any_damage_scores - any_damage_thresholds) <= delta
    near_category = np.any(
        np.abs(calibrated_scores - category_thresholds) <= delta, axis=1
    )
    reasons = {
        "safe_damage_conflict": conflict,
        "empty_output": empty,
        "gate_category_incoherence": incoherent,
        "near_any_damage_threshold": near_gate,
        "near_category_threshold": near_category,
    }
    review = np.logical_or.reduce(list(reasons.values()))
    return review, {key: float(value.mean()) for key, value in reasons.items()}


def _review_curve(
    truth_any_damage: np.ndarray,
    calibrated_scores: np.ndarray,
    category_thresholds: np.ndarray,
    any_damage_scores: np.ndarray,
    any_damage_thresholds: np.ndarray,
    *,
    delta_grid: Sequence[float] = DEFAULT_REVIEW_DELTA_GRID,
) -> list[dict[str, Any]]:
    rows = []
    decisions = any_damage_scores >= any_damage_thresholds
    for delta in delta_grid:
        review, reasons = _review_mask(
            calibrated_scores,
            category_thresholds,
            any_damage_scores,
            any_damage_thresholds,
            delta=float(delta),
        )
        automatic = ~review
        selected_truth = truth_any_damage[automatic]
        selected_decisions = decisions[automatic]
        has_both_classes = np.unique(selected_truth).size == 2
        selective = (
            _binary_metrics(selected_truth, selected_decisions)
            if automatic.any() and has_both_classes
            else None
        )
        rows.append(
            {
                "delta": float(delta),
                "coverage": float(automatic.mean()),
                "review_load_rate": float(review.mean()),
                "automatic_rows": int(automatic.sum()),
                "selective_binary_metrics": selective,
                "review_reason_rates": reasons,
            }
        )
    return rows


def _select_review_policy(
    curve: Sequence[Mapping[str, Any]], max_review_rate: float | None
) -> dict[str, Any]:
    if max_review_rate is None:
        return {
            "status": "pending_human_capacity",
            "max_review_rate": None,
            "selected_delta": None,
        }
    if not 0 <= max_review_rate < 1:
        raise ValueError("max_review_rate debe estar en [0, 1)")
    feasible = [
        row
        for row in curve
        if float(row["review_load_rate"]) <= max_review_rate
        and row["selective_binary_metrics"] is not None
    ]
    if not feasible:
        return {
            "status": "infeasible_human_capacity",
            "max_review_rate": float(max_review_rate),
            "selected_delta": None,
            "minimum_observed_review_rate": float(
                min(float(row["review_load_rate"]) for row in curve)
            ),
        }
    selected = min(
        feasible,
        key=lambda row: (
            row["selective_binary_metrics"]["balanced_error_rate"],
            -float(row["coverage"]),
            float(row["delta"]),
        ),
    )
    return {
        "status": "selected_on_validation_under_capacity",
        "max_review_rate": float(max_review_rate),
        "selected_delta": float(selected["delta"]),
        "validation_operating_point": dict(selected),
    }


def _crossfit_binary_policy(
    truth: np.ndarray,
    scores: np.ndarray,
    video_ids: Sequence[str],
    *,
    folds: int = DEFAULT_SELECTION_FOLDS,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    taxonomy = load_taxonomy()
    damage_indices = np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels]
    )
    truth_any = truth[:, damage_indices].max(axis=1)
    groups = np.asarray([str(value) for value in video_ids])
    split_count = min(int(folds), len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("Cross-fitting requiere videos de al menos dos grupos")
    oof_scores = np.zeros_like(scores, dtype=float)
    oof_category_thresholds = np.zeros_like(scores, dtype=float)
    oof_gate_thresholds = np.zeros(len(scores), dtype=float)
    fold_rows = []
    for fold, (train_indices, heldout_indices) in enumerate(
        GroupKFold(n_splits=split_count).split(scores, truth_any, groups=groups), start=1
    ):
        calibrators = _fit_score_calibrators(
            truth[train_indices], scores[train_indices]
        )
        calibrated_train = _apply_score_calibrators(scores[train_indices], calibrators)
        calibrated_heldout = _apply_score_calibrators(
            scores[heldout_indices], calibrators
        )
        category_thresholds = calibrate_thresholds(
            truth[train_indices], calibrated_train
        )
        ordered_thresholds = np.asarray(
            [category_thresholds[label] for label in taxonomy.target_labels]
        )
        train_any_scores = calibrated_train[:, damage_indices].max(axis=1)
        gate_threshold = _balanced_threshold(
            truth_any[train_indices], train_any_scores
        )
        oof_scores[heldout_indices] = calibrated_heldout
        oof_category_thresholds[heldout_indices] = ordered_thresholds
        oof_gate_thresholds[heldout_indices] = gate_threshold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "heldout_rows": len(heldout_indices),
                "train_videos": len(set(groups[train_indices])),
                "heldout_videos": len(set(groups[heldout_indices])),
                "any_damage_threshold": float(gate_threshold),
            }
        )
    oof_any_scores = oof_scores[:, damage_indices].max(axis=1)
    oof_decisions = oof_any_scores >= oof_gate_thresholds
    full_calibrators = _fit_score_calibrators(truth, scores)
    full_scores = _apply_score_calibrators(scores, full_calibrators)
    full_category_thresholds = calibrate_thresholds(truth, full_scores)
    full_any_scores = full_scores[:, damage_indices].max(axis=1)
    full_gate_threshold = _balanced_threshold(truth_any, full_any_scores)
    return {
        "folds": split_count,
        "fold_rows": fold_rows,
        "truth_any_damage": truth_any,
        "oof_calibrated_scores": oof_scores,
        "oof_category_thresholds": oof_category_thresholds,
        "oof_any_damage_scores": oof_any_scores,
        "oof_any_damage_thresholds": oof_gate_thresholds,
        "oof_decisions": oof_decisions.astype(np.int8),
        "oof_metrics": _binary_metrics(truth_any, oof_decisions, oof_any_scores),
        "deployment_calibrators": full_calibrators,
        "deployment_category_thresholds": full_category_thresholds,
        "deployment_any_damage_threshold": float(full_gate_threshold),
    }


def _grouped_bootstrap_macro_ap(
    truth: np.ndarray,
    scores: np.ndarray,
    video_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from joblib import Parallel, delayed
    if replicates < 1:
        raise ValueError("replicates debe ser al menos 1")
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        groups[str(video_id)].append(index)
    group_indices = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    def macro_damage_ap(indices: np.ndarray) -> float:
        return _macro_damage_average_precision(truth, scores, indices)

    master_rng = np.random.default_rng(seed)
    replicate_seeds = master_rng.integers(
        0, np.iinfo(np.uint64).max, size=replicates, dtype=np.uint64
    )
    workers = min(parallel_workers, replicates, os.cpu_count() or 1)
    seed_chunks = np.array_split(replicate_seeds, workers)

    def evaluate_chunk(seed_chunk: np.ndarray) -> list[float]:
        values = []
        for replicate_seed in seed_chunk:
            rng = np.random.default_rng(int(replicate_seed))
            sampled = rng.integers(0, len(group_indices), size=len(group_indices))
            indices = np.concatenate([group_indices[index] for index in sampled])
            values.append(macro_damage_ap(indices))
        _notify_progress(
            progress_callback,
            status="progress",
            phase="bootstrap AP agrupado por video",
            advance=len(seed_chunk),
        )
        return values

    chunks = Parallel(n_jobs=workers, prefer="threads")(
        delayed(evaluate_chunk)(seed_chunk) for seed_chunk in seed_chunks
    )
    estimates = [value for chunk in chunks for value in chunk]
    point = macro_damage_ap(np.arange(len(truth), dtype=int))
    return {
        "metric": "average_precision_macro_damage",
        "point": point,
        "confidence_level": 0.95,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
        "grouping": "video_id",
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "parallel_workers": workers,
        "samples": estimates,
    }


def _paired_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    challenger: np.ndarray,
    video_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from joblib import Parallel, delayed

    if replicates < 1:
        raise ValueError("replicates debe ser al menos 1")
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        groups[str(video_id)].append(index)
    group_indices = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    labels = np.asarray(truth, dtype=np.int8).reshape(-1)
    reference_decisions = np.asarray(reference, dtype=np.int8).reshape(-1)
    challenger_decisions = np.asarray(challenger, dtype=np.int8).reshape(-1)
    if not (
        len(labels) == len(reference_decisions) == len(challenger_decisions)
    ):
        raise ValueError("Las decisiones binarias pareadas no coinciden")

    def balanced_accuracy(decisions: np.ndarray, indices: np.ndarray) -> float:
        selected_truth = labels[indices]
        selected_decisions = decisions[indices]
        positive = selected_truth == 1
        negative = ~positive
        if not positive.any() or not negative.any():
            return float("nan")
        recall = float(selected_decisions[positive].mean())
        specificity = float((1 - selected_decisions[negative]).mean())
        return (recall + specificity) / 2

    master_rng = np.random.default_rng(seed)
    replicate_seeds = master_rng.integers(
        0, np.iinfo(np.uint64).max, size=replicates, dtype=np.uint64
    )
    workers = min(parallel_workers, replicates, os.cpu_count() or 1)
    seed_chunks = np.array_split(replicate_seeds, workers)

    def evaluate_chunk(seed_chunk: np.ndarray) -> list[float]:
        values = []
        for replicate_seed in seed_chunk:
            rng = np.random.default_rng(int(replicate_seed))
            sampled = rng.integers(0, len(group_indices), size=len(group_indices))
            indices = np.concatenate([group_indices[index] for index in sampled])
            difference = balanced_accuracy(challenger_decisions, indices) - balanced_accuracy(
                reference_decisions, indices
            )
            if not np.isnan(difference):
                values.append(difference)
        _notify_progress(
            progress_callback,
            status="progress",
            phase="bootstrap pareado",
            advance=len(seed_chunk),
        )
        return values

    chunks = Parallel(n_jobs=workers, prefer="threads")(
        delayed(evaluate_chunk)(seed_chunk) for seed_chunk in seed_chunks
    )
    differences = [value for chunk in chunks for value in chunk]
    if not differences:
        raise ValueError("El bootstrap no produjo réplicas con ambas clases")
    values = np.asarray(differences)
    lower = (int(np.sum(values <= 0)) + 1) / (len(values) + 1)
    upper = (int(np.sum(values >= 0)) + 1) / (len(values) + 1)
    p_value = min(1.0, 2 * min(lower, upper))
    return {
        "metric": "balanced_accuracy_binary_any_damage_oof",
        "difference_challenger_minus_reference": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_raw": p_value,
        "replicates": replicates,
        "grouping": "video_id",
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "parallel_workers": workers,
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value_raw"])
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row["p_value_raw"]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def compare_and_freeze_validation(
    dataset_path: str | Path,
    candidate_roots: Iterable[str | Path],
    comparison_path: str | Path,
    freeze_path: str | Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20260805,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    selection_folds: int = DEFAULT_SELECTION_FOLDS,
    max_review_rate: float | None = None,
    macro_auprc_noninferiority_margin: float | None = None,
    review_delta_grid: Sequence[float] = DEFAULT_REVIEW_DELTA_GRID,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compara individuos/ensembles en validation y congela sin abrir test."""

    comparison_started = time.perf_counter()
    dataset = Path(dataset_path).resolve()
    if selection_folds < 2:
        raise ValueError("selection_folds debe ser al menos 2")
    if macro_auprc_noninferiority_margin is not None and not (
        0 <= macro_auprc_noninferiority_margin <= 1
    ):
        raise ValueError("macro_auprc_noninferiority_margin debe estar en [0, 1]")
    if max_review_rate is not None and not 0 <= max_review_rate < 1:
        raise ValueError("max_review_rate debe estar en [0, 1)")
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    parallel_workers = min(parallel_workers, os.cpu_count() or 1)
    eligibility = audit_validation_candidate_eligibility(dataset, candidate_roots)
    dataset_sha = str(eligibility["dataset_sha256"])
    taxonomy = load_taxonomy()
    eligible = eligibility["eligible"]
    rejected = eligibility["rejected"]
    if not eligible:
        reason_summary = "; ".join(
            f"{row.get('candidate_id') or '<sin-id>'}: {','.join(row['reasons'])}"
            for row in rejected[:12]
        )
        suffix = f" Rechazos: {reason_summary}." if reason_summary else ""
        raise ValueError(
            "No hay candidatos elegibles con validation y test sellado "
            f"(descubiertos={eligibility['discovered_count']}).{suffix}"
        )
    signature = canonical_json_sha256(
        {
            "dataset_sha256": dataset_sha,
            "candidate_ids": sorted(row["candidate_id"] for row in eligible),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "parallel_workers": parallel_workers,
            "selection_criterion_version": SELECTION_CRITERION_VERSION,
            "selection_folds": selection_folds,
            "max_review_rate": max_review_rate,
            "macro_auprc_noninferiority_margin": macro_auprc_noninferiority_margin,
            "review_delta_grid": [float(value) for value in review_delta_grid],
            "seed": seed,
        }
    )
    freeze = Path(freeze_path)
    comparison = Path(comparison_path)
    if freeze.is_file() and comparison.is_file() and not force:
        previous = json.loads(freeze.read_text(encoding="utf-8"))
        if previous.get("comparison_signature") == signature:
            return {
                "status": "noop",
                "freeze": str(freeze),
                "comparison": str(comparison),
            }

    reference_loaded = (eligible[0], *_load_validation_predictions(eligible[0]))
    validation_rows, _reference_scores = _align_predictions([reference_loaded])
    truth_rows = [{"coarse_labels": row["true_labels"]} for row in validation_rows]
    truth = encode_targets(truth_rows)
    video_ids = [str(row["video_id"]) for row in validation_rows]

    evaluated: list[dict[str, Any]] = []
    eligible_by_id = {str(row["candidate_id"]): row for row in eligible}
    score_by_id: dict[str, np.ndarray] = {}
    deployment_score_by_id: dict[str, np.ndarray] = {}
    thresholds_by_id: dict[str, dict[str, float]] = {}
    policy_by_id: dict[str, dict[str, Any]] = {}

    def evaluate_scores(
        identifier: str,
        candidate_scores: np.ndarray,
        *,
        kind: str,
        family: str,
        member_ids: list[str],
        weights: list[float] | None = None,
    ) -> None:
        policy = _crossfit_binary_policy(
            truth,
            candidate_scores,
            video_ids,
            folds=selection_folds,
        )
        deployment_scores = _apply_score_calibrators(
            candidate_scores, policy["deployment_calibrators"]
        )
        deployment_thresholds = policy["deployment_category_thresholds"]
        metrics = classification_metrics(
            truth, deployment_scores, deployment_thresholds
        )
        metrics["binary_any_damage_oof"] = policy["oof_metrics"]
        metrics["average_precision_macro_damage_oof"] = (
            _macro_damage_average_precision(truth, policy["oof_calibrated_scores"])
        )
        curve = _review_curve(
            policy["truth_any_damage"],
            policy["oof_calibrated_scores"],
            policy["oof_category_thresholds"],
            policy["oof_any_damage_scores"],
            policy["oof_any_damage_thresholds"],
            delta_grid=review_delta_grid,
        )
        review_policy = _select_review_policy(curve, max_review_rate)
        metrics["needs_review"] = {
            "not_a_model_output": True,
            "ranking_uses_full_coverage": True,
            "risk_coverage_curve": curve,
            "operating_policy": review_policy,
        }
        score_by_id[identifier] = candidate_scores
        deployment_score_by_id[identifier] = deployment_scores
        thresholds_by_id[identifier] = deployment_thresholds
        policy_by_id[identifier] = policy
        evaluated.append(
            {
                "candidate_id": identifier,
                "kind": kind,
                "model_family": family,
                "members": member_ids,
                "weights": weights,
                "training_regime": (
                    eligible_by_id[identifier].get("training_regime", "standard")
                    if kind == "individual"
                    else "ensemble_of_reported_members"
                ),
                "comparison_disclaimer": (
                    eligible_by_id[identifier].get("comparison_disclaimer")
                    if kind == "individual"
                    else [
                        eligible_by_id[member].get("comparison_disclaimer")
                        for member in member_ids
                        if member in eligible_by_id
                        and eligible_by_id[member].get("comparison_disclaimer")
                    ]
                ),
                "training_budget": (
                    eligible_by_id[identifier].get("training_budget")
                    if kind == "individual"
                    else {
                        member: eligible_by_id[member].get("training_budget")
                        for member in member_ids
                        if member in eligible_by_id
                    }
                ),
                "validation_metrics": metrics,
                "crossfit": {
                    "method": "group_k_fold_by_video",
                    "folds": policy["folds"],
                    "fold_rows": policy["fold_rows"],
                },
            }
        )

    for candidate in eligible:
        rows, scores = _load_validation_predictions(candidate)
        _, aligned = _align_predictions([reference_loaded, (candidate, rows, scores)])
        candidate_scores = aligned[1]
        identifier = str(candidate["candidate_id"])
        evaluate_scores(
            identifier,
            candidate_scores,
            kind="individual",
            family=str(candidate["model_family"]),
            member_ids=[identifier],
        )

    individual_by_id = {
        row["candidate_id"]: row for row in evaluated if row["kind"] == "individual"
    }
    best_by_slot = {
        slot: max(
            (
                row
                for row in eligible
                if _candidate_slot(row) == slot
            ),
            key=lambda row: _selection_key(
                individual_by_id[str(row["candidate_id"])]["validation_metrics"]
            ),
        )
        for slot in ("classical", "transformer", "qwen")
        if any(_candidate_slot(row) == slot for row in eligible)
    }
    members = list(best_by_slot.values())
    member_ids = [str(row["candidate_id"]) for row in members]
    score_matrices = [score_by_id[identifier] for identifier in member_ids]

    if len(score_matrices) >= 2:
        weights = np.asarray(
            [
                max(
                    1e-9,
                    float(
                        individual_by_id[str(row["candidate_id"])][
                            "validation_metrics"
                        ].get("average_precision_macro_damage_oof", 0.0)
                    ),
                )
                for row in members
            ]
        )
        weights /= weights.sum()
        member_thresholds = np.asarray(
            [
                [
                    float(thresholds_by_id[identifier][label])
                    for label in taxonomy.target_labels
                ]
                for identifier in member_ids
            ]
        )
        hard = np.asarray(
            [
                deployment_score_by_id[identifier] >= member_thresholds[index]
                for index, identifier in enumerate(member_ids)
            ],
            dtype=float,
        ).mean(axis=0)
        ensemble_scores = {
            "ensemble_soft_mean": np.mean(score_matrices, axis=0),
            "ensemble_soft_validation_weighted": np.average(
                score_matrices, axis=0, weights=weights
            ),
            "ensemble_hard_majority": hard,
            "ensemble_union": np.max(score_matrices, axis=0),
            "ensemble_intersection": np.min(score_matrices, axis=0),
        }
        for identifier, scores in ensemble_scores.items():
            evaluate_scores(
                identifier,
                scores,
                kind="ensemble",
                family="ensemble",
                member_ids=member_ids,
                weights=weights.tolist() if "weighted" in identifier else None,
            )

    bootstrap_started = time.perf_counter()
    bootstrap_total = bootstrap_replicates * (2 * len(evaluated) - 1)
    _notify_progress(
        progress_callback,
        status="started",
        phase="bootstrap agrupado por video",
        total=bootstrap_total,
        advance=0,
    )
    macro_ap_samples_by_id: dict[str, np.ndarray] = {}
    for row in evaluated:
        bootstrap = _grouped_bootstrap_macro_ap(
            truth,
            policy_by_id[row["candidate_id"]]["oof_calibrated_scores"],
            video_ids,
            replicates=bootstrap_replicates,
            seed=seed,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback,
        )
        macro_ap_samples_by_id[row["candidate_id"]] = np.asarray(
            bootstrap.pop("samples"), dtype=float
        )
        row["validation_metrics"]["average_precision_macro_damage_oof"] = bootstrap[
            "point"
        ]
        row["bootstrap_grouped_by_video"] = bootstrap
    frontier = _pareto_front(evaluated)
    macro_ap_reference = max(
        (row for row in evaluated if row["kind"] == "individual"),
        key=lambda row: float(
            row["validation_metrics"]["average_precision_macro_damage_oof"]
        ),
    )
    best_individual_macro_ap = float(
        macro_ap_reference["validation_metrics"][
            "average_precision_macro_damage_oof"
        ]
    )
    reference_samples = macro_ap_samples_by_id[macro_ap_reference["candidate_id"]]
    for row in evaluated:
        observed = float(
            row["validation_metrics"]["average_precision_macro_damage_oof"]
        )
        differences = macro_ap_samples_by_id[row["candidate_id"]] - reference_samples
        difference_ci_low = float(np.quantile(differences, 0.025))
        difference_ci_high = float(np.quantile(differences, 0.975))
        row["macro_auprc_safeguard"] = {
            "method": "paired_video_cluster_bootstrap_noninferiority",
            "reference_candidate": macro_ap_reference["candidate_id"],
            "reference_best_individual": best_individual_macro_ap,
            "difference_candidate_minus_reference": observed
            - best_individual_macro_ap,
            "difference_ci_low": difference_ci_low,
            "difference_ci_high": difference_ci_high,
            "margin": macro_auprc_noninferiority_margin,
            "status": (
                "pareto_only_margin_not_predeclared"
                if macro_auprc_noninferiority_margin is None
                else (
                    "pass"
                    if difference_ci_low >= -macro_auprc_noninferiority_margin
                    else "fail"
                )
            ),
        }
    selectable = [
        row
        for row in evaluated
        if row["macro_auprc_safeguard"]["status"] != "fail"
    ]
    selected = max(
        selectable, key=lambda row: _selection_key(row["validation_metrics"])
    )
    reference_individual = max(
        (row for row in evaluated if row["kind"] == "individual"),
        key=lambda row: _selection_key(row["validation_metrics"]),
    )
    tests = []
    for row in evaluated:
        if row["candidate_id"] == selected["candidate_id"]:
            continue
        test = _paired_bootstrap(
            policy_by_id[selected["candidate_id"]]["truth_any_damage"],
            policy_by_id[selected["candidate_id"]]["oof_decisions"],
            policy_by_id[row["candidate_id"]]["oof_decisions"],
            video_ids,
            replicates=bootstrap_replicates,
            seed=seed + len(tests) + 1,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback,
        )
        test.update(
            {
                "reference": selected["candidate_id"],
                "challenger": row["candidate_id"],
            }
        )
        tests.append(test)
    _holm_adjust(tests)
    eligible_challengers = [
        row for row in tests if row["challenger"] in {item["candidate_id"] for item in selectable}
    ]
    closest_challenger = max(
        eligible_challengers,
        key=lambda test: next(
            _selection_key(row["validation_metrics"])
            for row in selectable
            if row["candidate_id"] == test["challenger"]
        ),
        default=None,
    )
    leader_confirmed = bool(
        closest_challenger is not None
        and closest_challenger["ci_high"] < 0
        and closest_challenger["p_value_holm"] <= 0.05
    )
    _notify_progress(
        progress_callback,
        status="finished",
        phase="bootstrap completo",
        total=bootstrap_total,
        completed=bootstrap_total,
    )
    bootstrap_elapsed = time.perf_counter() - bootstrap_started
    diversity = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            left_binary = score_matrices[left] >= member_thresholds[left]
            right_binary = score_matrices[right] >= member_thresholds[right]
            diversity.append(
                {
                    "left": members[left]["candidate_id"],
                    "right": members[right]["candidate_id"],
                    "labelwise_disagreement_rate": float(
                        (left_binary != right_binary).mean()
                    ),
                }
            )

    comparison_payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "dataset_sha256": dataset_sha,
        "selection_split": "validation",
        "test_status": "sealed_not_evaluated",
        "runtime_optimization": {
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "parallel_workers": parallel_workers,
            "shared_memory_threads": True,
        },
        "stage_timings_seconds": {
            "grouped_and_paired_bootstrap": bootstrap_elapsed,
            "comparison_total_before_write": time.perf_counter() - comparison_started,
        },
        "selection_policy": {
            "criterion_version": SELECTION_CRITERION_VERSION,
            "aggregation": "lexicographic_not_weighted_sum",
            "primary": "max binary_any_damage_oof.balanced_accuracy at full coverage",
            "safeguard": "macro AUPRC damage noninferiority when margin predeclared; otherwise Pareto report",
            "tie_breakers": ["min R_0.67", "max macro AUPRC damage"],
            "sensitivity_lambdas": [0.50, 0.67, 0.80],
            "needs_review": "post-inference policy selected after model under declared human capacity",
            "max_review_rate": max_review_rate,
            "macro_auprc_noninferiority_margin": macro_auprc_noninferiority_margin,
        },
        "pareto_front": frontier,
        "selected_for_freeze": selected["candidate_id"],
        "winner_status": (
            "confirmed_on_validation" if leader_confirmed else "statistical_tie_or_inconclusive"
        ),
        "closest_eligible_challenger_test": closest_challenger,
        "best_individual": reference_individual["candidate_id"],
        "best_by_family_slot": {
            key: row["candidate_id"] for key, row in best_by_slot.items()
        },
        "ranking": sorted(
            evaluated,
            key=lambda row: _selection_key(row["validation_metrics"]),
            reverse=True,
        ),
        "diversity": diversity,
        "paired_bootstrap_tests_holm": tests,
        "stacking_oof": {
            "status": "not_available",
            "reason": "requires out-of-fold predictions from every selected member; validation is not used to fit a meta-learner",
        },
        "rejected": rejected,
    }
    review_policy = selected["validation_metrics"]["needs_review"][
        "operating_policy"
    ]
    policy_ready = (
        macro_auprc_noninferiority_margin is not None
        and review_policy["status"] == "selected_on_validation_under_capacity"
    )
    frozen_test_status = (
        "sealed_ready_for_single_open"
        if policy_ready
        else "sealed_pending_predeclared_operating_policy"
    )
    freeze_payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha,
        "selected_id": selected["candidate_id"],
        "selected_kind": selected["kind"],
        "members": selected["members"],
        "selected_training_regime": selected.get("training_regime"),
        "selected_comparison_disclaimer": selected.get("comparison_disclaimer"),
        "selected_training_budget": selected.get("training_budget"),
        "member_training_disclaimers": {
            identifier: eligible_by_id[identifier].get("comparison_disclaimer")
            for identifier in selected["members"]
            if identifier in eligible_by_id
            and eligible_by_id[identifier].get("comparison_disclaimer")
        },
        "member_candidate_paths": {
            row["candidate_id"]: row["candidate_path"]
            for row in eligible
            if row["candidate_id"] in selected["members"]
        },
        "ensemble_weights": selected.get("weights"),
        "thresholds": thresholds_by_id[selected["candidate_id"]],
        "score_calibrators": policy_by_id[selected["candidate_id"]][
            "deployment_calibrators"
        ],
        "member_score_calibrators": {
            identifier: policy_by_id[identifier]["deployment_calibrators"]
            for identifier in selected["members"]
        },
        "member_thresholds": {
            identifier: thresholds_by_id[identifier]
            for identifier in selected["members"]
        },
        "any_damage_threshold": policy_by_id[selected["candidate_id"]][
            "deployment_any_damage_threshold"
        ],
        "needs_review_policy": review_policy,
        "macro_auprc_noninferiority_margin": macro_auprc_noninferiority_margin,
        "selection_criterion_version": SELECTION_CRITERION_VERSION,
        "winner_status": (
            "confirmed_on_validation" if leader_confirmed else "statistical_tie_or_inconclusive"
        ),
        "test_status": frozen_test_status,
        "publication_approved": False,
    }
    write_json_atomic(comparison, comparison_payload)
    write_json_atomic(freeze, freeze_payload)
    return {
        "status": "frozen",
        "selected": selected["candidate_id"],
        "pareto_front": frontier,
        "comparison": str(comparison),
        "freeze": str(freeze),
        "winner_status": freeze_payload["winner_status"],
        "test_status": frozen_test_status,
    }


def _asset(candidate: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return (
        path
        if path.is_absolute()
        else Path(str(candidate["candidate_path"])).parent / path
    )


def _score_candidate(
    candidate: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    device: str,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str = "inferencia",
) -> np.ndarray:
    inference = candidate["inference"]
    kind = inference["type"]
    texts = [str(row["text"]) for row in rows]
    if kind == "sklearn_joblib":
        import joblib

        bundle = json.loads(
            _asset(candidate, inference["bundle"]).read_text(encoding="utf-8")
        )
        model = joblib.load(
            _asset(candidate, inference["bundle"]).parent / bundle["model"]
        )
        _notify_progress(
            progress_callback,
            status="started",
            phase=progress_phase,
            total=1,
            advance=0,
        )
        values = np.asarray(model.predict_proba(texts), dtype=float)
        _notify_progress(
            progress_callback,
            status="finished",
            phase=progress_phase,
            total=1,
            completed=1,
        )
        return values[:, :5]
    if kind == "hf_prompt_sft_json":
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from .prompt_sft import _generate_json_scores

        model_path = _asset(candidate, inference["model"])
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        hardware = resolve_device(device)
        torch_device = torch_device_name(hardware)
        model.to(torch_device)
        provenance = json.loads(
            _asset(candidate, inference["prompt_capsule"]).read_text(encoding="utf-8")
        )

        def relay_generation_progress(event: dict[str, Any]) -> None:
            forwarded = dict(event)
            detail = str(forwarded.get("phase") or "generación")
            forwarded["phase"] = f"{progress_phase} · {detail}"
            _notify_progress(progress_callback, **forwarded)

        values, _quality = _generate_json_scores(
            model,
            tokenizer,
            rows,
            provenance["capsule"],
            device=torch_device,
            max_input_length=int(inference.get("max_input_length", 3840)),
            batch_size=int(inference.get("batch_size", 1)),
            max_new_tokens=int(inference.get("max_new_tokens", 256)),
            progress_callback=relay_generation_progress,
        )
        return values[:, :5]
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Instale moderacion-peru[entrenamiento] para abrir test"
        ) from exc
    hardware = resolve_device(device)
    torch_device = torch_device_name(hardware)

    def sequence_scores(path: Path, count: int, phase: str) -> np.ndarray:
        tokenizer = AutoTokenizer.from_pretrained(path)
        if kind == "hf_peft_sequence_classifier":
            from peft import AutoPeftModelForSequenceClassification

            model = AutoPeftModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(torch_device).eval()
        batches = []
        batch_size = 16
        batch_total = max(1, math.ceil(len(texts) / batch_size))
        _notify_progress(
            progress_callback,
            status="started",
            phase=phase,
            total=batch_total,
            advance=0,
        )
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                encoded = {
                    key: value.to(torch_device) for key, value in encoded.items()
                }
                batches.append(
                    torch.sigmoid(model(**encoded).logits).float().cpu().numpy()
                )
                _notify_progress(
                    progress_callback,
                    status="progress",
                    phase=phase,
                    advance=1,
                )
        _notify_progress(
            progress_callback,
            status="finished",
            phase=phase,
            total=batch_total,
            completed=batch_total,
        )
        return np.concatenate(batches, axis=0)[:, :count]

    if kind in {"hf_sequence_classifier", "hf_peft_sequence_classifier"}:
        return sequence_scores(_asset(candidate, inference["model"]), 5, progress_phase)
    if kind in {"hf_cascade", "hf_cascade_v2"}:
        gate = sequence_scores(
            _asset(candidate, inference["gate_model"]),
            1,
            f"{progress_phase} · compuerta",
        )
        safety_first = kind == "hf_cascade_v2"
        original_kind = inference["type"]
        inference["type"] = "hf_sequence_classifier"
        try:
            specialist = sequence_scores(
                _asset(
                    candidate,
                    inference["branch_model" if safety_first else "damage_model"],
                ),
                5 if safety_first else 4,
                f"{progress_phase} · {'rama segura/daño' if safety_first else 'daño'}",
            )
        finally:
            inference["type"] = original_kind
        if safety_first:
            return combine_safety_first_cascade_scores(
                gate[:, 0],
                specialist,
                gate_threshold=float(inference["gate_threshold"]),
            )
        return np.concatenate([1 - gate, gate * specialist], axis=1)
    raise ValueError(f"Inferencia no soportada en apertura de test: {kind}")


def evaluate_frozen_test(
    freeze_path: str | Path,
    destination_path: str | Path,
    *,
    confirm_single_test_open: bool = False,
    device: str = "auto",
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Abre test una sola vez para una decisión previamente congelada."""

    if not confirm_single_test_open:
        return {
            "status": "sealed",
            "message": "Defina confirm_single_test_open=True solo después de revisar la congelación",
        }
    evaluation_started = time.perf_counter()
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("test_status") != "sealed_ready_for_single_open":
        raise RuntimeError(
            "Test sigue sellado: declare antes el margen macro-AUPRC y una "
            "capacidad NEEDS_REVIEW factible, vuelva a comparar y congele"
        )
    destination = Path(destination_path)
    if destination.is_file() and not force:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("comparison_signature") == freeze.get("comparison_signature"):
            return {"status": "noop", "test_report": str(destination)}
        raise ValueError("Ya existe una apertura de test para otra congelación")
    dataset = Path(freeze["dataset"])
    if sha256_file(dataset) != freeze["dataset_sha256"]:
        raise ValueError("El snapshot cambió después de congelar la selección")
    candidates = {
        identifier: json.loads(Path(path).read_text(encoding="utf-8"))
        for identifier, path in freeze["member_candidate_paths"].items()
    }
    for identifier, path in freeze["member_candidate_paths"].items():
        candidates[identifier]["candidate_path"] = path
    first = next(iter(candidates.values()))
    split_field = first.get("training_sampling", {}).get("split_field", "split")
    test_rows = [row for row in read_jsonl(dataset) if row.get(split_field) == "test"]
    sampling_policy = first.get("training_sampling", {})
    test_4_to_1, test_sampling_4_to_1 = deterministic_safe_downsample(
        test_rows,
        safe_to_damage_ratio=float(sampling_policy.get("safe_to_damage_ratio", 4.0)),
        seed=int(sampling_policy.get("sampling_seed", 20260805)),
    )
    if not test_rows:
        raise ValueError("No hay filas en el test congelado")
    inference_started = time.perf_counter()
    member_scores = {}
    for identifier, candidate in candidates.items():
        member_scores[identifier] = _score_candidate(
            candidate,
            test_rows,
            device=device,
            progress_callback=progress_callback,
            progress_phase=f"test · {identifier}",
        )
    inference_elapsed = time.perf_counter() - inference_started
    taxonomy = load_taxonomy()
    matrices = list(member_scores.values())
    selected_id = freeze["selected_id"]
    if freeze["selected_kind"] == "individual":
        scores = member_scores[selected_id]
    elif selected_id == "ensemble_soft_mean":
        scores = np.mean(matrices, axis=0)
    elif selected_id == "ensemble_soft_validation_weighted":
        scores = np.average(matrices, axis=0, weights=freeze["ensemble_weights"])
    elif selected_id == "ensemble_union":
        scores = np.max(matrices, axis=0)
    elif selected_id == "ensemble_intersection":
        scores = np.min(matrices, axis=0)
    elif selected_id == "ensemble_hard_majority":
        binary = []
        for identifier, matrix in member_scores.items():
            calibrated_matrix = _apply_score_calibrators(
                matrix, freeze["member_score_calibrators"][identifier]
            )
            threshold = np.asarray(
                [
                    freeze["member_thresholds"][identifier][label]
                    for label in taxonomy.target_labels
                ]
            )
            binary.append(calibrated_matrix >= threshold)
        scores = np.mean(binary, axis=0)
    else:
        raise ValueError(f"Regla congelada desconocida: {selected_id}")
    scores = _apply_score_calibrators(scores, freeze["score_calibrators"])
    metrics_started = time.perf_counter()
    truth = encode_targets(test_rows)
    metrics_natural = classification_metrics(truth, scores, freeze["thresholds"])
    damage_indices = np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels]
    )
    truth_any_natural = truth[:, damage_indices].max(axis=1)
    any_damage_scores_natural = scores[:, damage_indices].max(axis=1)
    any_damage_decisions_natural = (
        any_damage_scores_natural >= float(freeze["any_damage_threshold"])
    )
    metrics_natural["binary_any_damage_frozen_gate"] = _binary_metrics(
        truth_any_natural,
        any_damage_decisions_natural,
        any_damage_scores_natural,
    )
    controlled_ids = {row["chunk_id"] for row in test_4_to_1}
    controlled_indices = np.asarray(
        [
            index
            for index, row in enumerate(test_rows)
            if row["chunk_id"] in controlled_ids
        ],
        dtype=int,
    )
    metrics_4_to_1 = classification_metrics(
        truth[controlled_indices],
        scores[controlled_indices],
        freeze["thresholds"],
    )
    metrics_4_to_1["binary_any_damage_frozen_gate"] = _binary_metrics(
        truth_any_natural[controlled_indices],
        any_damage_decisions_natural[controlled_indices],
        any_damage_scores_natural[controlled_indices],
    )
    selected_delta = freeze.get("needs_review_policy", {}).get("selected_delta")
    review_natural = None
    if selected_delta is not None:
        ordered_thresholds = np.asarray(
            [freeze["thresholds"][label] for label in taxonomy.target_labels]
        )
        review_mask, review_reasons = _review_mask(
            scores,
            np.broadcast_to(ordered_thresholds, scores.shape),
            any_damage_scores_natural,
            np.full(len(scores), float(freeze["any_damage_threshold"])),
            delta=float(selected_delta),
        )
        automatic = ~review_mask
        review_natural = {
            "delta": float(selected_delta),
            "coverage": float(automatic.mean()),
            "review_load_rate": float(review_mask.mean()),
            "review_reason_rates": review_reasons,
            "selective_binary_metrics": (
                _binary_metrics(
                    truth_any_natural[automatic],
                    any_damage_decisions_natural[automatic],
                )
                if automatic.any()
                and np.unique(truth_any_natural[automatic]).size == 2
                else None
            ),
        }
    metrics_natural["needs_review_frozen_policy"] = review_natural
    metrics_elapsed = time.perf_counter() - metrics_started
    write_jsonl_atomic(
        destination.with_name(destination.stem + "_predictions.jsonl"),
        [
            {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "scores": {
                    label: float(scores[index, label_index])
                    for label_index, label in enumerate(taxonomy.target_labels)
                },
                "true_labels": row["coarse_labels"],
                "predicted_any_damage": bool(any_damage_decisions_natural[index]),
                "needs_review": (
                    bool(review_mask[index]) if selected_delta is not None else None
                ),
            }
            for index, row in enumerate(test_rows)
        ],
    )
    payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": freeze["comparison_signature"],
        "dataset_sha256": freeze["dataset_sha256"],
        "selected_id": selected_id,
        "test_open_count": 1,
        "inference_passes": 1,
        "test_rows_natural": len(test_rows),
        "test_rows_4_to_1": len(test_4_to_1),
        "test_sampling_4_to_1": test_sampling_4_to_1,
        "primary_metrics_natural_prevalence": metrics_natural,
        "secondary_metrics_4_to_1": metrics_4_to_1,
        "metrics": metrics_natural,
        "interpretation": (
            "The full natural-prevalence test is primary. The deterministic 4:1 "
            "view reuses the same predictions and is secondary; it does not open "
            "or score the test a second time."
        ),
        "stage_timings_seconds": {
            "test_inference_all_selected_members": inference_elapsed,
            "natural_and_4_to_1_metrics": metrics_elapsed,
            "test_total_before_report_write": time.perf_counter() - evaluation_started,
        },
    }
    write_json_atomic(destination, payload)
    return {
        "status": "test_evaluated_once",
        "test_report": str(destination),
        "primary_metrics_natural_prevalence": metrics_natural,
        "secondary_metrics_4_to_1": metrics_4_to_1,
    }
