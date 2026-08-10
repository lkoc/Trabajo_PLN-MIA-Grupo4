from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

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
    if set(scores) != set(contract.target_labels) or set(thresholds) != set(
        contract.target_labels
    ):
        raise ValueError("Scores y umbrales deben cubrir las cinco salidas")
    active = tuple(
        label for label in contract.target_labels if scores[label] >= thresholds[label]
    )
    reasons: list[str] = []
    if not active:
        reasons.append("sin_categoria_sobre_umbral")
    if contract.safe_label in active and len(active) > 1:
        reasons.append("conflicto_seguro_dano")
    if any(
        abs(scores[label] - thresholds[label]) <= uncertainty_margin
        for label in contract.target_labels
    ):
        reasons.append("cerca_del_umbral")
    return PredictionDecision(active, bool(reasons), tuple(reasons))


def calibrate_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[str] | None = None,
    observed_mask: np.ndarray | None = None,
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
    if observed_mask is not None and observed_mask.shape != y_true.shape:
        raise ValueError("La máscara observada no coincide con las salidas calibradas")
    thresholds: dict[str, float] = {}
    grid = np.linspace(0.05, 0.95, 91)
    for index, label in enumerate(target_labels):
        mask = (
            observed_mask[:, index].astype(bool)
            if observed_mask is not None
            else np.ones(len(y_true), dtype=bool)
        )
        truth = y_true[mask, index]
        scores = y_score[mask, index]
        if not len(truth):
            thresholds[label] = 0.5
            continue
        if np.unique(truth).size < 2:
            thresholds[label] = 0.5
            continue
        scored = []
        for threshold in grid:
            predicted = (scores >= threshold).astype(np.int8)
            scored.append(
                (float(f1_score(truth, predicted, zero_division=0)), float(threshold))
            )
        # En empate se conserva el umbral más alto para reducir falsos positivos.
        thresholds[label] = max(scored, key=lambda item: (item[0], item[1]))[1]
    return thresholds


def _expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, bins: int = 10
) -> float:
    errors = []
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (y_score >= lower) & (y_score < upper if upper < 1 else y_score <= upper)
        if mask.any():
            errors.append(
                float(mask.mean())
                * abs(float(y_true[mask].mean()) - float(y_score[mask].mean()))
            )
    return float(sum(errors))


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            hamming_loss,
            jaccard_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
            roc_auc_score,
        )
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
    calibration_by_label = {
        label: {
            "ece": _expected_calibration_error(y_true[:, index], y_score[:, index]),
            "brier": float(brier_score_loss(y_true[:, index], y_score[:, index])),
        }
        for index, label in enumerate(contract.target_labels)
    }
    result: dict[str, object] = {
        "average_precision_by_label": ap_by_label,
        "average_precision_macro_five": float(np.mean(list(ap_by_label.values()))),
        "average_precision_macro_damage": float(np.mean(damage_ap)),
        "expected_calibration_error": _expected_calibration_error(y_true, y_score),
        "calibration_by_label": calibration_by_label,
        "brier_macro": float(
            np.mean([row["brier"] for row in calibration_by_label.values()])
        ),
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
    damage_indices = [
        contract.target_labels.index(label) for label in contract.damage_labels
    ]
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
    safe_truth = y_true[:, safe_index] == 1
    false_alarm = (predicted_any_damage == 1) & safe_truth
    any_damage_score = y_score[:, damage_indices].max(axis=1)
    any_damage_ap = float(average_precision_score(true_any_damage, any_damage_score))
    any_damage_auc = (
        float(roc_auc_score(true_any_damage, any_damage_score))
        if np.unique(true_any_damage).size == 2
        else 0.0
    )
    result.update(
        {
            "per_label": per_label,
            "f1_macro_five": float(np.mean(f1)),
            "f1_macro_damage": float(np.mean(f1[damage_indices])),
            "f1_micro": float(
                f1_score(y_true, predicted, average="micro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, predicted, average="weighted", zero_division=0)
            ),
            "hamming_loss": float(hamming_loss(y_true, predicted)),
            "subset_accuracy": float(accuracy_score(y_true, predicted)),
            "jaccard_samples": float(
                jaccard_score(y_true, predicted, average="samples", zero_division=0)
            ),
            "true_label_cardinality": float(y_true.sum(axis=1).mean()),
            "predicted_label_cardinality": float(predicted.sum(axis=1).mean()),
            "label_cardinality_absolute_error": float(
                np.abs(y_true.sum(axis=1) - predicted.sum(axis=1)).mean()
            ),
            "any_damage": {
                "precision": float(any_precision),
                "recall": float(any_recall),
                "f1": float(any_f1),
                "average_precision": any_damage_ap,
                "roc_auc": any_damage_auc,
                "matthews_correlation": float(
                    matthews_corrcoef(true_any_damage, predicted_any_damage)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(true_any_damage, predicted_any_damage)
                ),
            },
            "false_safe_rate_on_damage": float(
                false_safe.sum() / max(1, true_any_damage.sum())
            ),
            "false_alarm_rate_on_safe": float(
                false_alarm.sum() / max(1, safe_truth.sum())
            ),
            "safe_damage_conflict_rate": float(conflicts.mean()),
            "no_category_rate": float(no_category.mean()),
            "review_load_rate": float(review.mean()),
        }
    )
    return result


def masked_multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    observed_mask: np.ndarray,
    labels: Sequence[str],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Métricas auxiliares calculadas solo donde existe supervisión observada."""

    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError("scikit-learn es necesario para calcular métricas") from exc
    if y_true.shape != y_score.shape or observed_mask.shape != y_true.shape:
        raise ValueError("Targets, scores y máscara deben tener la misma forma")
    if y_true.shape[1] != len(labels):
        raise ValueError("La lista de etiquetas no coincide con las matrices")
    rows: dict[str, dict[str, object]] = {}
    f1_values: list[float] = []
    ap_values: list[float] = []
    for index, label in enumerate(labels):
        mask = observed_mask[:, index].astype(bool)
        truth = y_true[mask, index]
        score = y_score[mask, index]
        positives = int(truth.sum()) if len(truth) else 0
        ap = (
            float(average_precision_score(truth, score))
            if len(truth) and positives and np.unique(truth).size > 1
            else 0.0
        )
        row: dict[str, object] = {
            "observed": int(mask.sum()),
            "positives": positives,
            "coverage": float(mask.mean()) if len(mask) else 0.0,
            "average_precision": ap,
        }
        ap_values.append(ap)
        if thresholds is not None:
            threshold = float(thresholds[label])
            predicted = (score >= threshold).astype(np.int8)
            if len(truth):
                precision, recall, f1, _ = precision_recall_fscore_support(
                    truth, predicted, average="binary", zero_division=0
                )
            else:
                precision = recall = f1 = 0.0
            row.update(
                {
                    "threshold": threshold,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
            f1_values.append(float(f1))
        rows[label] = row
    payload: dict[str, object] = {
        "per_label": rows,
        "macro_average_precision_observed": (
            float(np.mean(ap_values)) if ap_values else 0.0
        ),
    }
    if thresholds is not None:
        payload["macro_f1_observed"] = float(np.mean(f1_values)) if f1_values else 0.0
    return payload
