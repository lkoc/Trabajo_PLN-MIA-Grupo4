from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

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


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn es necesario para calcular métricas") from exc
    contract = load_taxonomy()
    if y_true.shape != y_score.shape or y_true.shape[1] != len(contract.target_labels):
        raise ValueError("Las matrices no coinciden con las cinco salidas")
    ap_by_label = {
        label: float(average_precision_score(y_true[:, index], y_score[:, index]))
        for index, label in enumerate(contract.target_labels)
    }
    damage_ap = [ap_by_label[label] for label in contract.damage_labels]
    return {
        "average_precision_by_label": ap_by_label,
        "average_precision_macro_five": float(np.mean(list(ap_by_label.values()))),
        "average_precision_macro_damage": float(np.mean(damage_ap)),
    }

