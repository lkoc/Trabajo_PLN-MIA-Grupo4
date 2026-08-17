from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import deterministic_safe_downsample
from .ensemble_evaluation import (
    _apply_score_calibrators,
    _binary_metrics,
    _fit_score_calibrators,
    _review_mask,
    _score_candidate,
)
from .io import canonical_json_sha256, read_jsonl, write_json_atomic, write_jsonl_atomic
from .taxonomy import load_taxonomy
from .training import calibrate_thresholds, classification_metrics, encode_targets

OPTIMIZATION_SCHEMA_VERSION = "1.0.0"
OPTIMIZATION_PROTOCOL = "nested-grouped-convex-blending-v1"
DEFAULT_MEMBER_IDS = (
    "classical-logistic_regression_c0p5-54f7971c6000",
    "cascade_v2-af78eba77883",
    "qwen_lora-4aa5ce04df05",
)
DEFAULT_HEURISTIC_WEIGHTS = np.asarray(
    [0.3316725087604802, 0.31099487972639145, 0.3573326115131283],
    dtype=float,
)


@dataclass(frozen=True)
class PredictionPanel:
    chunk_ids: np.ndarray
    video_ids: np.ndarray
    truth: np.ndarray
    raw_scores: np.ndarray
    member_ids: tuple[str, ...]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _prediction_rows_from_stream(lines: Iterable[bytes | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in lines:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if text.strip():
            rows.append(json.loads(text))
    return rows


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def load_prediction_rows_from_tar(
    archive_path: str | Path, member_path: str
) -> list[dict[str, Any]]:
    archive = Path(archive_path)
    with tarfile.open(archive, "r:*") as handle:
        member = handle.getmember(member_path)
        stream = handle.extractfile(member)
        if stream is None:
            raise FileNotFoundError(
                f"No se pudo leer {member_path} dentro de {archive}"
            )
        return _prediction_rows_from_stream(stream)


def _score_vector(row: Mapping[str, Any], labels: Sequence[str]) -> list[float]:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise TypeError(f"{row.get('chunk_id')}: scores ausentes")
    return [float(scores[label]) for label in labels]


def align_prediction_rows(
    sources: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> PredictionPanel:
    if len(sources) < 2:
        raise ValueError("La optimización requiere al menos dos miembros")
    taxonomy = load_taxonomy()
    reference_id, reference_rows = sources[0]
    if not reference_rows:
        raise ValueError(f"{reference_id}: no contiene predicciones")
    reference_by_id = {str(row["chunk_id"]): row for row in reference_rows}
    if len(reference_by_id) != len(reference_rows):
        raise ValueError(f"{reference_id}: chunk_id duplicado")
    ordered_ids = sorted(reference_by_id)
    aligned: list[np.ndarray] = []
    for member_id, rows in sources:
        by_id = {str(row["chunk_id"]): row for row in rows}
        if set(by_id) != set(ordered_ids):
            missing = len(set(ordered_ids) - set(by_id))
            extra = len(set(by_id) - set(ordered_ids))
            raise ValueError(
                f"{member_id}: panel no alineado (faltan={missing}, sobran={extra})"
            )
        for chunk_id in ordered_ids:
            if list(by_id[chunk_id].get("true_labels", [])) != list(
                reference_by_id[chunk_id].get("true_labels", [])
            ):
                raise ValueError(f"{member_id}: verdad divergente en {chunk_id}")
        aligned.append(
            np.asarray(
                [
                    _score_vector(by_id[chunk_id], taxonomy.target_labels)
                    for chunk_id in ordered_ids
                ],
                dtype=float,
            )
        )
    truth_rows = [
        {"coarse_labels": reference_by_id[chunk_id]["true_labels"]}
        for chunk_id in ordered_ids
    ]
    return PredictionPanel(
        chunk_ids=np.asarray(ordered_ids, dtype=str),
        video_ids=np.asarray(
            [str(reference_by_id[chunk_id]["video_id"]) for chunk_id in ordered_ids]
        ),
        truth=encode_targets(truth_rows),
        raw_scores=np.stack(aligned, axis=1),
        member_ids=tuple(member_id for member_id, _ in sources),
    )


def reconstruct_qwen_validation_predictions(
    *,
    candidate_path: str | Path,
    dataset_path: str | Path,
    reference_rows: Sequence[Mapping[str, Any]],
    destination: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Reconstruye validation sin abrir test y conserva un JSONL alineable."""

    candidate_file = Path(candidate_path).resolve()
    dataset_file = Path(dataset_path).resolve()
    output = Path(destination)
    reference_ids = {str(row["chunk_id"]) for row in reference_rows}
    selected: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(dataset_file):
        chunk_id = str(row.get("chunk_id") or "")
        if chunk_id in reference_ids:
            selected[chunk_id] = row
    if set(selected) != reference_ids:
        raise ValueError(
            f"El dataset no cubre el panel: {len(reference_ids - set(selected))} faltantes"
        )
    ordered_rows = [selected[str(row["chunk_id"])] for row in reference_rows]
    if any(row.get("split") != "validation" for row in ordered_rows):
        raise ValueError("La reconstrucción solo admite filas de validation")
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    candidate["candidate_path"] = str(candidate_file)
    started = time.perf_counter()
    scores = _score_candidate(candidate, ordered_rows, device=device)
    elapsed = time.perf_counter() - started
    taxonomy = load_taxonomy()
    materialized = []
    for reference, row, vector in zip(
        reference_rows, ordered_rows, scores, strict=True
    ):
        materialized.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "video_id": str(row["video_id"]),
                "split": "validation",
                "scores": {
                    label: float(vector[index])
                    for index, label in enumerate(taxonomy.target_labels)
                },
                "true_labels": list(reference["true_labels"]),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, materialized)
    return {
        "status": "reconstructed_from_frozen_adapter_without_test",
        "rows": len(materialized),
        "elapsed_seconds": elapsed,
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": _sha256(candidate_file),
        "dataset_sha256": _sha256(dataset_file),
        "destination": str(output),
        "destination_sha256": _sha256(output),
        "device_requested": device,
    }


def _simplex_grid(step: float) -> np.ndarray:
    denominator = round(1.0 / step)
    if denominator < 1 or not math.isclose(denominator * step, 1.0, abs_tol=1e-12):
        raise ValueError("step debe dividir exactamente la unidad")
    return np.asarray(
        [
            (left / denominator, middle / denominator, right / denominator)
            for left in range(denominator + 1)
            for middle in range(denominator + 1 - left)
            for right in [denominator - left - middle]
        ],
        dtype=float,
    )


def _fast_balanced_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(truth, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=float).reshape(-1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("El umbral balanceado requiere ambas clases")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    prefix_positive = np.concatenate(([0], np.cumsum(sorted_labels)))
    prefix_negative = np.arange(len(labels) + 1) - prefix_positive
    candidates = np.unique(
        np.concatenate(([0.0], np.linspace(0.01, 0.99, 99), values, [1.0]))
    )
    insertion = np.searchsorted(sorted_values, candidates, side="left")
    false_negative = prefix_positive[insertion]
    true_negative = prefix_negative[insertion]
    recall = (positives - false_negative) / positives
    specificity = true_negative / negatives
    balanced_accuracy = (recall + specificity) / 2
    # Equivale al max((BA, recall, threshold)) del evaluador original.
    index = np.lexsort((candidates, recall, balanced_accuracy))[-1]
    return float(candidates[index])


def _damage_indices() -> np.ndarray:
    taxonomy = load_taxonomy()
    return np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels],
        dtype=int,
    )


def _macro_damage_ap(truth: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    damage = _damage_indices()
    return float(
        np.mean(
            [
                average_precision_score(truth[:, index], scores[:, index])
                for index in damage
            ]
        )
    )


def _macro_damage_f1(truth: np.ndarray, decisions: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    damage = _damage_indices()
    return float(
        np.mean(
            [
                f1_score(truth[:, index], decisions[:, index], zero_division=0)
                for index in damage
            ]
        )
    )


def _selection_summary(truth: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    damage = _damage_indices()
    truth_any = truth[:, damage].max(axis=1)
    any_scores = scores[:, damage].max(axis=1)
    threshold = _fast_balanced_threshold(truth_any, any_scores)
    binary = _binary_metrics(truth_any, any_scores >= threshold, any_scores)
    macro_ap = _macro_damage_ap(truth, scores)
    return {
        "any_damage_threshold": threshold,
        "binary": binary,
        "macro_auprc_damage": macro_ap,
        "selection_key": [
            binary["balanced_accuracy"],
            -binary["risk_lambda"]["0.67"],
            macro_ap,
        ],
    }


def _binary_selection_summary(truth: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    damage = _damage_indices()
    truth_any = truth[:, damage].max(axis=1)
    any_scores = scores[:, damage].max(axis=1)
    threshold = _fast_balanced_threshold(truth_any, any_scores)
    binary = _binary_metrics(truth_any, any_scores >= threshold, any_scores)
    return {"any_damage_threshold": threshold, "binary": binary}


def _fit_member_layer(
    raw_scores: np.ndarray,
    truth: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[dict[str, Any]]]]:
    taxonomy = load_taxonomy()
    probabilities = np.zeros(
        (len(heldout_indices), raw_scores.shape[1], raw_scores.shape[2]), dtype=float
    )
    hard = np.zeros_like(probabilities)
    calibrators_by_member: list[list[dict[str, Any]]] = []
    for member in range(raw_scores.shape[1]):
        calibrators = _fit_score_calibrators(
            truth[train_indices], raw_scores[train_indices, member]
        )
        calibrated_train = _apply_score_calibrators(
            raw_scores[train_indices, member], calibrators
        )
        calibrated_heldout = _apply_score_calibrators(
            raw_scores[heldout_indices, member], calibrators
        )
        thresholds = calibrate_thresholds(truth[train_indices], calibrated_train)
        ordered_thresholds = np.asarray(
            [thresholds[label] for label in taxonomy.target_labels], dtype=float
        )
        probabilities[:, member] = calibrated_heldout
        hard[:, member] = calibrated_heldout >= ordered_thresholds
        calibrators_by_member.append(calibrators)
    return probabilities, hard, calibrators_by_member


def _member_meta_oof(
    raw_scores: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    from sklearn.model_selection import GroupKFold

    split_count = min(folds, len(np.unique(groups)))
    if split_count < 2:
        raise ValueError("Se requieren al menos dos grupos")
    probabilities = np.zeros_like(raw_scores, dtype=float)
    hard = np.zeros_like(raw_scores, dtype=float)
    fold_rows: list[dict[str, int]] = []
    for fold, (train, heldout) in enumerate(
        GroupKFold(n_splits=split_count).split(raw_scores, groups=groups), start=1
    ):
        p_fold, h_fold, _ = _fit_member_layer(raw_scores, truth, train, heldout)
        probabilities[heldout] = p_fold
        hard[heldout] = h_fold
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "heldout_rows": len(heldout),
                "train_videos": len(np.unique(groups[train])),
                "heldout_videos": len(np.unique(groups[heldout])),
            }
        )
    return probabilities, hard, fold_rows


def _combine(
    formula: str,
    probabilities: np.ndarray,
    hard: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if formula == "soft_mean":
        return probabilities.mean(axis=1)
    if formula in {"soft_weighted", "soft_optimized"}:
        if weights is None:
            raise ValueError(f"{formula} requiere pesos")
        return np.einsum("nmk,m->nk", probabilities, weights)
    if formula == "union":
        return probabilities.max(axis=1)
    if formula == "intersection":
        return probabilities.min(axis=1)
    if formula in {"hard_majority", "hard_optimized"}:
        selected = (
            np.full(hard.shape[1], 1 / hard.shape[1]) if weights is None else weights
        )
        return np.einsum("nmk,m->nk", hard, selected)
    raise ValueError(f"Fórmula desconocida: {formula}")


def _optimize_weights(
    probabilities: np.ndarray,
    hard: np.ndarray,
    truth: np.ndarray,
    *,
    formula: str,
    grid_step: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    grid = _simplex_grid(grid_step)
    for weights in grid:
        scores = _combine(formula, probabilities, hard, weights)
        summary = _binary_selection_summary(truth, scores)
        binary = summary["binary"]
        primary_key = (
            binary["balanced_accuracy"],
            -binary["risk_lambda"]["0.67"],
        )
        if best is not None and primary_key < best["_primary_key"]:
            continue
        macro_ap = _macro_damage_ap(truth, scores)
        distance_from_equal = float(np.sum((weights - 1 / len(weights)) ** 2))
        key = (*primary_key, macro_ap, -distance_from_equal)
        if best is None or key > best["_key"]:
            best = {
                "weights": weights.tolist(),
                **summary,
                "macro_auprc_damage": macro_ap,
                "selection_key": [*primary_key, macro_ap],
                "_primary_key": primary_key,
                "_key": key,
            }
    assert best is not None
    best.pop("_key")
    best.pop("_primary_key")
    best["grid_step"] = grid_step
    best["grid_candidates"] = len(grid)
    return best


def _paired_video_bootstrap_ba(
    truth_any: np.ndarray,
    reference_decisions: np.ndarray,
    challenger_decisions: np.ndarray,
    video_ids: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 20260817,
) -> dict[str, Any]:
    groups = np.unique(video_ids)

    def counts(decisions: np.ndarray) -> np.ndarray:
        rows = []
        for group in groups:
            mask = video_ids == group
            truth_group = truth_any[mask].astype(bool)
            predicted_group = decisions[mask].astype(bool)
            rows.append(
                [
                    np.sum(truth_group & predicted_group),
                    np.sum(truth_group & ~predicted_group),
                    np.sum(~truth_group & predicted_group),
                    np.sum(~truth_group & ~predicted_group),
                ]
            )
        return np.asarray(rows, dtype=float)

    reference_counts = counts(reference_decisions)
    challenger_counts = counts(challenger_decisions)

    def balanced_accuracy(total: np.ndarray) -> float:
        tp, fn, fp, tn = total
        return float(0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp)))

    point = balanced_accuracy(reference_counts.sum(axis=0)) - balanced_accuracy(
        challenger_counts.sum(axis=0)
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        differences[replicate] = balanced_accuracy(
            reference_counts[sampled].sum(axis=0)
        ) - balanced_accuracy(challenger_counts[sampled].sum(axis=0))
    nonpositive = (np.sum(differences <= 0) + 1) / (replicates + 1)
    nonnegative = (np.sum(differences >= 0) + 1) / (replicates + 1)
    return {
        "metric": "balanced_accuracy_any_damage_nested_oof",
        "difference_reference_minus_challenger": point,
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "p_two_sided_uncorrected": float(min(1.0, 2 * min(nonpositive, nonnegative))),
        "replicates": replicates,
        "grouping": "video_id",
        "seed": seed,
    }


def _formula_specs(
    soft_weights: Sequence[float], hard_weights: Sequence[float]
) -> list[tuple[str, str, np.ndarray | None]]:
    return [
        ("ensemble_soft_mean", "soft_mean", None),
        (
            "ensemble_soft_validation_weighted",
            "soft_weighted",
            DEFAULT_HEURISTIC_WEIGHTS,
        ),
        ("ensemble_union", "union", None),
        ("ensemble_hard_majority", "hard_majority", None),
        ("ensemble_intersection", "intersection", None),
        ("ensemble_soft_optimized", "soft_optimized", np.asarray(soft_weights)),
        ("ensemble_hard_optimized", "hard_optimized", np.asarray(hard_weights)),
    ]


def nested_compare_ensembles(
    panel: PredictionPanel,
    *,
    outer_folds: int = 5,
    inner_folds: int = 5,
    grid_step: float = 0.025,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    n_rows, n_members, _ = panel.raw_scores.shape
    if n_members != 3:
        raise ValueError("El protocolo vigente requiere exactamente tres miembros")
    taxonomy = load_taxonomy()
    oof_scores: dict[str, np.ndarray] = {}
    oof_decisions: dict[str, np.ndarray] = {}
    oof_any_decisions: dict[str, np.ndarray] = {}
    fold_parameters: list[dict[str, Any]] = []
    outer_split_count = min(outer_folds, len(np.unique(panel.video_ids)))
    for outer_fold, (outer_train, outer_heldout) in enumerate(
        GroupKFold(n_splits=outer_split_count).split(
            panel.raw_scores, groups=panel.video_ids
        ),
        start=1,
    ):
        inner_probabilities, inner_hard, inner_rows = _member_meta_oof(
            panel.raw_scores[outer_train],
            panel.truth[outer_train],
            panel.video_ids[outer_train],
            folds=inner_folds,
        )
        soft_fit = _optimize_weights(
            inner_probabilities,
            inner_hard,
            panel.truth[outer_train],
            formula="soft_optimized",
            grid_step=grid_step,
        )
        hard_fit = _optimize_weights(
            inner_probabilities,
            inner_hard,
            panel.truth[outer_train],
            formula="hard_optimized",
            grid_step=grid_step,
        )
        heldout_probabilities, heldout_hard, _ = _fit_member_layer(
            panel.raw_scores,
            panel.truth,
            outer_train,
            outer_heldout,
        )
        formula_parameters: dict[str, Any] = {}
        for identifier, formula, weights in _formula_specs(
            soft_fit["weights"], hard_fit["weights"]
        ):
            inner_scores = _combine(formula, inner_probabilities, inner_hard, weights)
            heldout_scores = _combine(
                formula, heldout_probabilities, heldout_hard, weights
            )
            category_thresholds = calibrate_thresholds(
                panel.truth[outer_train], inner_scores
            )
            ordered_category_thresholds = np.asarray(
                [category_thresholds[label] for label in taxonomy.target_labels]
            )
            damage = _damage_indices()
            truth_any_inner = panel.truth[outer_train][:, damage].max(axis=1)
            gate_threshold = _fast_balanced_threshold(
                truth_any_inner, inner_scores[:, damage].max(axis=1)
            )
            oof_scores.setdefault(
                identifier, np.zeros((n_rows, len(taxonomy.target_labels)))
            )[outer_heldout] = heldout_scores
            oof_decisions.setdefault(
                identifier,
                np.zeros((n_rows, len(taxonomy.target_labels)), dtype=np.int8),
            )[outer_heldout] = heldout_scores >= ordered_category_thresholds
            oof_any_decisions.setdefault(identifier, np.zeros(n_rows, dtype=np.int8))[
                outer_heldout
            ] = heldout_scores[:, damage].max(axis=1) >= gate_threshold
            formula_parameters[identifier] = {
                "weights": None if weights is None else weights.tolist(),
                "any_damage_threshold": gate_threshold,
                "category_thresholds": category_thresholds,
            }
        fold_parameters.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(outer_train),
                "heldout_rows": len(outer_heldout),
                "train_videos": len(np.unique(panel.video_ids[outer_train])),
                "heldout_videos": len(np.unique(panel.video_ids[outer_heldout])),
                "inner_folds": inner_rows,
                "soft_optimized": soft_fit,
                "hard_optimized": hard_fit,
                "formula_parameters": formula_parameters,
            }
        )
    damage = _damage_indices()
    truth_any = panel.truth[:, damage].max(axis=1)
    results = []
    for identifier, scores in oof_scores.items():
        binary = _binary_metrics(
            truth_any,
            oof_any_decisions[identifier],
            scores[:, damage].max(axis=1),
        )
        macro_ap = _macro_damage_ap(panel.truth, scores)
        results.append(
            {
                "candidate_id": identifier,
                "balanced_accuracy_any_damage_nested_oof": binary["balanced_accuracy"],
                "fnr_any_damage_nested_oof": binary["false_negative_rate"],
                "fpr_any_damage_nested_oof": binary["false_positive_rate"],
                "risk_0_67_nested_oof": binary["risk_lambda"]["0.67"],
                "auprc_any_damage_nested_oof": binary["average_precision"],
                "macro_auprc_damage_nested_oof": macro_ap,
                "macro_f1_damage_nested_oof": _macro_damage_f1(
                    panel.truth, oof_decisions[identifier]
                ),
                "selection_key": [
                    binary["balanced_accuracy"],
                    -binary["risk_lambda"]["0.67"],
                    macro_ap,
                ],
            }
        )
    results.sort(key=lambda row: tuple(row["selection_key"]), reverse=True)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    winner_id = results[0]["candidate_id"]
    paired_tests = []
    for challenger in results[1:]:
        challenger_id = challenger["candidate_id"]
        paired_tests.append(
            {
                "reference": winner_id,
                "challenger": challenger_id,
                **_paired_video_bootstrap_ba(
                    truth_any,
                    oof_any_decisions[winner_id],
                    oof_any_decisions[challenger_id],
                    panel.video_ids,
                ),
            }
        )

    full_probabilities, full_hard, full_fold_rows = _member_meta_oof(
        panel.raw_scores,
        panel.truth,
        panel.video_ids,
        folds=inner_folds,
    )
    final_soft = _optimize_weights(
        full_probabilities,
        full_hard,
        panel.truth,
        formula="soft_optimized",
        grid_step=grid_step,
    )
    final_hard = _optimize_weights(
        full_probabilities,
        full_hard,
        panel.truth,
        formula="hard_optimized",
        grid_step=grid_step,
    )
    all_indices = np.arange(n_rows)
    deployment_probabilities, deployment_hard, member_calibrators = _fit_member_layer(
        panel.raw_scores, panel.truth, all_indices, all_indices
    )
    final_parameters: dict[str, Any] = {}
    for identifier, formula, weights in _formula_specs(
        final_soft["weights"], final_hard["weights"]
    ):
        scores = _combine(formula, deployment_probabilities, deployment_hard, weights)
        final_parameters[identifier] = {
            "weights": None if weights is None else weights.tolist(),
            "category_thresholds": calibrate_thresholds(panel.truth, scores),
            "any_damage_threshold": _fast_balanced_threshold(
                truth_any, scores[:, damage].max(axis=1)
            ),
        }
    return {
        "protocol": OPTIMIZATION_PROTOCOL,
        "rows": n_rows,
        "videos": len(np.unique(panel.video_ids)),
        "outer_folds": outer_split_count,
        "inner_folds": inner_folds,
        "grid_step": grid_step,
        "ranking_nested_oof": results,
        "winner_nested_oof": winner_id,
        "paired_video_bootstrap": paired_tests,
        "outer_fold_parameters": fold_parameters,
        "final_fit_on_complete_validation": {
            "meta_fold_rows": full_fold_rows,
            "soft_optimized": final_soft,
            "hard_optimized": final_hard,
            "formula_parameters": final_parameters,
            "member_calibrators": {
                panel.member_ids[index]: calibrators
                for index, calibrators in enumerate(member_calibrators)
            },
        },
    }


def build_optimization_report(
    *,
    panel: PredictionPanel,
    nested_result: Mapping[str, Any],
    original_comparison_path: str | Path,
    source_artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    comparison_path = Path(original_comparison_path)
    original = json.loads(comparison_path.read_text(encoding="utf-8"))
    original_ensembles = []
    for rank, row in enumerate(original["ranking"], start=1):
        if row.get("kind") != "ensemble":
            continue
        metrics = row["validation_metrics"]
        binary = metrics["binary_any_damage_oof"]
        original_ensembles.append(
            {
                "rank_global": rank,
                "candidate_id": row["candidate_id"],
                "weights": row.get("weights"),
                "balanced_accuracy_any_damage_oof": binary["balanced_accuracy"],
                "fnr_any_damage_oof": binary["false_negative_rate"],
                "fpr_any_damage_oof": binary["false_positive_rate"],
                "risk_0_67_oof": binary["risk_lambda"]["0.67"],
                "macro_auprc_damage_oof": metrics["average_precision_macro_damage_oof"],
                "macro_f1_damage": metrics["f1_macro_damage"],
            }
        )
    source_manifest = {
        name: {
            "path": str(Path(path)),
            "bytes": Path(path).stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in source_artifacts.items()
    }
    return {
        "schema_version": OPTIMIZATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "validation_comparison_updated_test_pending_reanalysis",
        "dataset_sha256": original["dataset_sha256"],
        "comparison_signature": original["comparison_signature"],
        "member_ids": list(panel.member_ids),
        "source_artifacts": source_manifest,
        "original_03_07_ensembles": original_ensembles,
        "optimization": nested_result,
        "interpretation": {
            "selection_scope": "validation_only_nested_grouped_cv",
            "test_reused": "verified_member_score_checkpoints_after_formula_freeze",
            "production_replaced": True,
            "why": (
                "La optimización amplía 03_07; el ganador se congela con validation "
                "antes de aplicar la fórmula a checkpoints de la única apertura de test."
            ),
            "union_intersection": (
                "max/min no contienen coeficientes de mezcla; se recalibran en el "
                "protocolo común pero no se les inventan parámetros."
            ),
        },
    }


def write_optimization_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def load_partial_test_scores_from_tar(
    archive_path: str | Path,
    *,
    expected_member_ids: Sequence[str] = DEFAULT_MEMBER_IDS,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Carga y verifica los checkpoints creados en la apertura original de test."""

    expected = set(expected_member_ids)
    matrices: dict[str, np.ndarray] = {}
    manifests: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r:*") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        manifest_names = sorted(
            name
            for name in names
            if "test_final_abierto_una_vez_member_scores_partial/" in name
            and name.endswith(".json")
        )
        for manifest_name in manifest_names:
            stream = archive.extractfile(manifest_name)
            if stream is None:
                raise FileNotFoundError(manifest_name)
            manifest = json.loads(stream.read().decode("utf-8"))
            identifier = str(manifest["candidate_id"])
            if identifier not in expected:
                continue
            matrix_name = manifest_name.removesuffix(".json") + ".npy"
            if matrix_name not in names:
                raise FileNotFoundError(matrix_name)
            matrix_stream = archive.extractfile(matrix_name)
            if matrix_stream is None:
                raise FileNotFoundError(matrix_name)
            matrix_bytes = matrix_stream.read()
            if hashlib.sha256(matrix_bytes).hexdigest() != manifest["scores_sha256"]:
                raise ValueError(f"SHA-256 divergente para {identifier}")
            matrix = np.load(io.BytesIO(matrix_bytes), allow_pickle=False)
            expected_shape = tuple(int(value) for value in manifest["shape"])
            if matrix.shape != expected_shape or not np.isfinite(matrix).all():
                raise ValueError(f"Matriz inválida para {identifier}: {matrix.shape}")
            matrices[identifier] = np.asarray(matrix, dtype=float)
            manifests[identifier] = {
                **manifest,
                "archive_member": matrix_name,
                "archive_path": str(Path(archive_path)),
            }
    missing = expected - set(matrices)
    if missing:
        raise FileNotFoundError(
            "Faltan checkpoints de test: " + ", ".join(sorted(missing))
        )
    return matrices, manifests


def build_updated_freeze(
    *,
    original_freeze_path: str | Path,
    optimization_report: Mapping[str, Any],
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Congela el ganador de la comparación ampliada sin consultar test."""

    original = json.loads(Path(original_freeze_path).read_text(encoding="utf-8"))
    optimized = optimization_report["optimization"]
    final = optimized["final_fit_on_complete_validation"]
    parameters = final["formula_parameters"]["ensemble_soft_optimized"]
    weights = [float(value) for value in parameters["weights"]]
    parent_signature = original.get(
        "parent_comparison_signature", original["comparison_signature"]
    )
    signature = canonical_json_sha256(
        {
            "parent_comparison_signature": parent_signature,
            "protocol": optimized["protocol"],
            "selected_id": "ensemble_soft_optimized",
            "member_ids": optimization_report["member_ids"],
            "weights": weights,
            "thresholds": parameters["category_thresholds"],
            "any_damage_threshold": parameters["any_damage_threshold"],
        }
    )
    return {
        **original,
        "schema_version": "5.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "parent_comparison_signature": parent_signature,
        "dataset": str(Path(dataset_path).resolve()),
        "selected_id": "ensemble_soft_optimized",
        "members": list(optimization_report["member_ids"]),
        "ensemble_weights": weights,
        "ensemble_formula": (
            "sum_m w_m * sigmoid(a_mk * raw_score_mk + b_mk), w_m >= 0, sum_m w_m = 1"
        ),
        "ensemble_fusion_space": "member_calibrated_probabilities",
        "thresholds": parameters["category_thresholds"],
        "score_calibrators": [],
        "member_score_calibrators": final["member_calibrators"],
        "any_damage_threshold": float(parameters["any_damage_threshold"]),
        "needs_review_policy": {
            **original.get("needs_review_policy", {}),
            "selected_delta": 0.03,
            "note": (
                "Delta conservado y verificado en validation para la nueva "
                "fórmula bajo capacidad máxima 0.40; no se ajustó con test."
            ),
        },
        "selection_criterion_version": "nested-grouped-convex-blending-v1",
        "winner_status": (
            "selected_by_lexicographic_rule_pairwise_advantage_inconclusive"
        ),
        "test_status": "formula_frozen_ready_for_checkpoint_reanalysis",
        "publication_approved": True,
    }


def evaluate_optimized_test_from_archive(
    *,
    freeze: Mapping[str, Any],
    dataset_path: str | Path,
    archive_path: str | Path,
    original_test_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evalúa la fórmula congelada sobre los scores de la apertura original."""

    matrices, manifests = load_partial_test_scores_from_tar(
        archive_path, expected_member_ids=freeze["members"]
    )
    dataset = Path(dataset_path)
    test_rows = [row for row in read_jsonl(dataset) if row.get("split") == "test"]
    if not test_rows:
        raise ValueError("El snapshot no contiene filas de test")
    rows_signature = canonical_json_sha256([str(row["chunk_id"]) for row in test_rows])
    for identifier, manifest in manifests.items():
        if manifest["dataset_sha256"] != freeze["dataset_sha256"]:
            raise ValueError(f"Snapshot divergente en checkpoint {identifier}")
        if manifest["test_rows_signature"] != rows_signature:
            raise ValueError(f"Orden de test divergente en checkpoint {identifier}")
        if matrices[identifier].shape != (len(test_rows), 5):
            raise ValueError(f"Forma de test divergente en {identifier}")

    calibrated_members = [
        _apply_score_calibrators(
            matrices[identifier], freeze["member_score_calibrators"][identifier]
        )
        for identifier in freeze["members"]
    ]
    member_panel = np.stack(calibrated_members, axis=1)
    weights = np.asarray(freeze["ensemble_weights"], dtype=float)
    scores = np.einsum("nmk,m->nk", member_panel, weights)
    truth = encode_targets(test_rows)
    taxonomy = load_taxonomy()
    metrics_natural = classification_metrics(truth, scores, freeze["thresholds"])
    damage = np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels]
    )
    truth_any = truth[:, damage].max(axis=1)
    any_scores = scores[:, damage].max(axis=1)
    any_decisions = any_scores >= float(freeze["any_damage_threshold"])
    metrics_natural["binary_any_damage_frozen_gate"] = _binary_metrics(
        truth_any, any_decisions, any_scores
    )

    test_4_to_1, sampling = deterministic_safe_downsample(
        test_rows, safe_to_damage_ratio=4.0, seed=20260805
    )
    selected_ids = {str(row["chunk_id"]) for row in test_4_to_1}
    controlled = np.asarray(
        [
            index
            for index, row in enumerate(test_rows)
            if str(row["chunk_id"]) in selected_ids
        ],
        dtype=int,
    )
    metrics_4_to_1 = classification_metrics(
        truth[controlled], scores[controlled], freeze["thresholds"]
    )
    metrics_4_to_1["binary_any_damage_frozen_gate"] = _binary_metrics(
        truth_any[controlled], any_decisions[controlled], any_scores[controlled]
    )

    selected_delta = float(freeze["needs_review_policy"]["selected_delta"])
    category_thresholds = np.asarray(
        [freeze["thresholds"][label] for label in taxonomy.target_labels]
    )
    review, review_reasons = _review_mask(
        scores,
        np.broadcast_to(category_thresholds, scores.shape),
        any_scores,
        np.full(len(scores), float(freeze["any_damage_threshold"])),
        delta=selected_delta,
    )
    automatic = ~review
    metrics_natural["needs_review_frozen_policy"] = {
        "delta": selected_delta,
        "coverage": float(automatic.mean()),
        "review_load_rate": float(review.mean()),
        "review_reason_rates": review_reasons,
        "selective_binary_metrics": _binary_metrics(
            truth_any[automatic], any_decisions[automatic], any_scores[automatic]
        ),
    }
    original_selected = (
        original_test_report.get("selected_id") if original_test_report else None
    )
    return {
        "schema_version": "5.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": freeze["comparison_signature"],
        "parent_comparison_signature": freeze["parent_comparison_signature"],
        "dataset_sha256": freeze["dataset_sha256"],
        "selected_id": freeze["selected_id"],
        "test_open_count": 1,
        "test_reanalysis_count": 1,
        "new_inference_passes": 0,
        "test_rows_natural": len(test_rows),
        "test_rows_4_to_1": len(test_4_to_1),
        "member_score_sources": {
            identifier: "verified_checkpoint_from_original_single_test_open"
            for identifier in freeze["members"]
        },
        "checkpoint_manifests": manifests,
        "test_sampling_4_to_1": sampling,
        "primary_metrics_natural_prevalence": metrics_natural,
        "secondary_metrics_4_to_1": metrics_4_to_1,
        "metrics": metrics_natural,
        "supersedes_test_formula": original_selected,
        "interpretation": (
            "La fórmula optimizada se congeló solo con validation y se aplicó "
            "después a las tres matrices producidas en la apertura original de "
            "test. No hubo nueva inferencia, selección ni ajuste con test. La "
            "prevalencia natural es primaria y la vista 4:1 reutiliza los mismos "
            "scores."
        ),
    }


def write_updated_selection_and_test(
    *,
    freeze_path: str | Path,
    test_report_path: str | Path,
    freeze: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> None:
    write_json_atomic(freeze_path, freeze)
    write_json_atomic(test_report_path, test_report)
