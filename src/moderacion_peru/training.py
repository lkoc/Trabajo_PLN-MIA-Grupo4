from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .taxonomy import TaxonomyContract, load_taxonomy


@dataclass(frozen=True)
class PredictionDecision:
    labels: tuple[str, ...]
    requires_review: bool
    review_reasons: tuple[str, ...]


def encode_targets(
    rows: Iterable[Mapping[str, object]],
    taxonomy: TaxonomyContract | None = None,
) -> np.ndarray:
    contract = taxonomy or load_taxonomy()
    encoded: list[list[int]] = []
    for row in rows:
        labels = contract.normalize_categories(row.get("coarse_labels", []))  # type: ignore[arg-type]
        if not labels:
            raise ValueError("Un ejemplo entrenable debe tener una categoría explícita")
        encoded.append([int(label in labels) for label in contract.target_labels])
    return np.asarray(encoded, dtype=np.int8)


def resolve_prediction(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    uncertainty_margin: float = 0.05,
    taxonomy: TaxonomyContract | None = None,
) -> PredictionDecision:
    contract = taxonomy or load_taxonomy()
    if set(scores) != set(contract.target_labels) or set(thresholds) != set(contract.target_labels):
        raise ValueError("Scores y umbrales deben cubrir las cinco salidas")
    active = tuple(label for label in contract.target_labels if scores[label] >= thresholds[label])
    reasons: list[str] = []
    if not active:
        reasons.append("sin_categoria_sobre_umbral")
    if contract.safe_label in active and len(active) > 1:
        reasons.append("conflicto_seguro_dano")
    if any(abs(scores[label] - thresholds[label]) <= uncertainty_margin for label in contract.target_labels):
        reasons.append("cerca_del_umbral")
    return PredictionDecision(active, bool(reasons), tuple(reasons))


def calibrate_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Selecciona cada umbral en validation maximizando F1 con desempate conservador."""

    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn es necesario para calibrar umbrales") from exc
    contract = load_taxonomy()
    target_labels = tuple(labels or contract.target_labels)
    if y_true.shape != y_score.shape or y_true.shape[1] != len(target_labels):
        raise ValueError("Las matrices no coinciden con las salidas calibradas")
    thresholds: dict[str, float] = {}
    grid = np.linspace(0.05, 0.95, 91)
    for index, label in enumerate(target_labels):
        truth = y_true[:, index]
        if np.unique(truth).size < 2:
            thresholds[label] = 0.5
            continue
        scored = []
        for threshold in grid:
            predicted = (y_score[:, index] >= threshold).astype(np.int8)
            scored.append((float(f1_score(truth, predicted, zero_division=0)), float(threshold)))
        # En empate se conserva el umbral más alto para reducir falsos positivos.
        thresholds[label] = max(scored, key=lambda item: (item[0], item[1]))[1]
    return thresholds


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    errors = []
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (y_score >= lower) & (y_score < upper if upper < 1 else y_score <= upper)
        if mask.any():
            errors.append(float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_score[mask].mean())))
    return float(sum(errors))


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    try:
        from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("scikit-learn es necesario para calcular métricas") from exc
    contract = load_taxonomy()
    if y_true.shape != y_score.shape or y_true.shape[1] != len(contract.target_labels):
        raise ValueError("Las matrices no coinciden con las cinco salidas")
    ap_by_label = {}
    for index, label in enumerate(contract.target_labels):
        ap_by_label[label] = (
            float(average_precision_score(y_true[:, index], y_score[:, index]))
            if y_true[:, index].any()
            else 0.0
        )
    damage_ap = [ap_by_label[label] for label in contract.damage_labels]
    result: dict[str, object] = {
        "average_precision_by_label": ap_by_label,
        "average_precision_macro_five": float(np.mean(list(ap_by_label.values()))),
        "average_precision_macro_damage": float(np.mean(damage_ap)),
        "expected_calibration_error": _expected_calibration_error(y_true, y_score),
    }
    if thresholds is None:
        return result
    if set(thresholds) != set(contract.target_labels):
        raise ValueError("Debe existir un umbral para cada salida")
    ordered = np.asarray([thresholds[label] for label in contract.target_labels])
    predicted = (y_score >= ordered).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, average=None, zero_division=0
    )
    per_label = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(contract.target_labels)
    }
    damage_indices = [contract.target_labels.index(label) for label in contract.damage_labels]
    true_any_damage = y_true[:, damage_indices].max(axis=1)
    predicted_any_damage = predicted[:, damage_indices].max(axis=1)
    any_precision, any_recall, any_f1, _ = precision_recall_fscore_support(
        true_any_damage,
        predicted_any_damage,
        average="binary",
        zero_division=0,
    )
    safe_index = contract.target_labels.index(contract.safe_label)
    false_safe = (predicted[:, safe_index] == 1) & (true_any_damage == 1)
    conflicts = (predicted[:, safe_index] == 1) & (predicted_any_damage == 1)
    no_category = predicted.sum(axis=1) == 0
    near_threshold = np.any(np.abs(y_score - ordered) <= 0.05, axis=1)
    review = conflicts | no_category | near_threshold
    result.update(
        {
            "per_label": per_label,
            "f1_macro_five": float(np.mean(f1)),
            "f1_macro_damage": float(np.mean(f1[damage_indices])),
            "any_damage": {
                "precision": float(any_precision),
                "recall": float(any_recall),
                "f1": float(any_f1),
            },
            "false_safe_rate_on_damage": float(false_safe.sum() / max(1, true_any_damage.sum())),
            "safe_damage_conflict_rate": float(conflicts.mean()),
            "no_category_rate": float(no_category.mean()),
            "review_load_rate": float(review.mean()),
        }
    )
    return result
