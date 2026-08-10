"""Reproduce la muestra CODEX de 16.694 chunks y sus métricas comparativas.

El script no modifica las decisiones de etiquetado. Lee la campaña, las
salidas Flash/Pro y el historial append-only; luego escribe artefactos de
reporte sin texto de transcripciones.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "datos/etiquetado/consolidado/anotaciones_v2.jsonl"
EVENTS_PATH = ROOT / "datos/etiquetado/humano/labeling_events_v2.jsonl"
FLASH_PATH = ROOT / "datos/etiquetado/cascada_deepseek_v4/primary_flash_v3_2.jsonl"
PRO_PATH = ROOT / "datos/etiquetado/cascada_deepseek_v4/review_pro_v3_2.jsonl"
OUTPUT_DIR = ROOT / "docs/artefactos"
FROZEN_SAMPLE_PATH = OUTPUT_DIR / "auditoria_16k_flash_pro_sol_eh_sample.csv"
METRICS_PATH = OUTPUT_DIR / "auditoria_16k_panel_actual_v3_2_metrics.json"
SAMPLE_PATH = OUTPUT_DIR / "auditoria_16k_panel_actual_v3_2_sample.csv"

SEED_TEXT = "CODEX-AUDIT-20260809-CLEAN"
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPLICATES = 2_000
SAMPLE_SIZE = 16_694
PRE_SAMPLE_CUTOFF = "2026-08-09T10:49"
UNLABELED_BATCH = "CODEX-UNLABELED-PROMPT-V3_1_1-20260809"
LABELS = (
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def effective_labels(
    row: dict[str, Any], event: dict[str, Any] | None
) -> tuple[str, ...]:
    if event and event.get("action") in {"accept", "modify"}:
        return tuple(
            sorted(event.get("final_labels") or event.get("proposed_labels") or [])
        )
    return tuple(sorted(row.get("coarse_labels") or []))


def label_key(labels: Iterable[str]) -> str:
    values = tuple(sorted(labels))
    return "|".join(values) if values else "PENDIENTE"


def sha_rank(chunk_id: str) -> str:
    return hashlib.sha256(f"{SEED_TEXT}|{chunk_id}".encode()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - half_width, center + half_width]


def build_sample(
    campaign: list[dict[str, Any]], events: list[dict[str, Any]]
) -> tuple[
    list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, tuple[str, ...]]
]:
    latest_pre_sample: dict[str, dict[str, Any]] = {}
    for event in events:
        if str(event.get("created_at", "")) < PRE_SAMPLE_CUTOFF:
            latest_pre_sample[event["chunk_id"]] = event

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    initial_labels: dict[str, tuple[str, ...]] = {}
    stratum_by_id: dict[str, tuple[str, str, str]] = {}
    for row in campaign:
        event = latest_pre_sample.get(row["chunk_id"])
        if event and event.get("action") == "reject":
            continue
        labels = effective_labels(row, event)
        stratum = (
            label_key(labels),
            str(row.get("annotator_model") or ""),
            "needs_review" if row.get("needs_review") else "resolved",
        )
        strata[stratum].append(row)
        initial_labels[row["chunk_id"]] = labels
        stratum_by_id[row["chunk_id"]] = stratum

    population = sum(len(rows) for rows in strata.values())
    raw_allocation = {
        stratum: SAMPLE_SIZE * len(rows) / population
        for stratum, rows in strata.items()
    }
    allocation = {
        stratum: max(1, math.floor(value)) for stratum, value in raw_allocation.items()
    }
    remaining = SAMPLE_SIZE - sum(allocation.values())
    largest_remainders = sorted(
        strata,
        key=lambda stratum: (
            -(raw_allocation[stratum] - math.floor(raw_allocation[stratum])),
            str(stratum),
        ),
    )
    for stratum in largest_remainders[:remaining]:
        allocation[stratum] += 1

    sample: list[dict[str, Any]] = []
    for stratum, rows in strata.items():
        ranked = sorted(rows, key=lambda row: sha_rank(row["chunk_id"]))
        sample.extend(ranked[: allocation[stratum]])

    if population != 157_719 or len(strata) != 35 or len(sample) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Corte inesperado: población={population}, estratos={len(strata)}, muestra={len(sample)}"
        )
    return sample, stratum_by_id, initial_labels


def load_frozen_sample(
    campaign: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, tuple[str, ...]]
]:
    """Recupera el panel longitudinal sin volver a seleccionarlo con el corpus ampliado."""

    campaign_by_id = {row["chunk_id"]: row for row in campaign}
    sample: list[dict[str, Any]] = []
    stratum_by_id: dict[str, tuple[str, str, str]] = {}
    initial_labels: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    with FROZEN_SAMPLE_PATH.open(encoding="utf-8", newline="") as source:
        for frozen in csv.DictReader(source):
            chunk_id = frozen["chunk_id"]
            row = campaign_by_id.get(chunk_id)
            if row is None:
                missing.append(chunk_id)
                continue
            stratum = ast.literal_eval(frozen["stratum"])
            if not isinstance(stratum, tuple) or len(stratum) != 3:
                raise RuntimeError(
                    f"Estrato congelado inválido para {chunk_id}: {stratum!r}"
                )
            sample.append(row)
            stratum_by_id[chunk_id] = tuple(str(value) for value in stratum)
            initial_labels[chunk_id] = tuple(
                label
                for label in frozen["initial_effective_labels"].split("|")
                if label
            )
    if missing or len(sample) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Panel congelado incompleto: recuperados={len(sample)}, ausentes={len(missing)}"
        )
    return sample, stratum_by_id, initial_labels


def damage_vector(labels: Iterable[str]) -> np.ndarray:
    values = set(labels)
    return np.asarray([int(label in values) for label in LABELS], dtype=np.int8)


def bootstrap_indices(
    strata: list[str], replicates: int, rng: np.random.Generator
) -> Iterable[np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    strata_array = np.asarray(strata)
    for value in sorted(set(strata)):
        grouped[value] = np.flatnonzero(strata_array == value)
    for _ in range(replicates):
        yield np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in grouped.values()
            ]
        )


def key_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, exact: np.ndarray
) -> dict[str, float]:
    true_binary = (y_true.sum(axis=1) > 0).astype(np.int8)
    pred_binary = (y_pred.sum(axis=1) > 0).astype(np.int8)
    tp = int(((pred_binary == 1) & (true_binary == 1)).sum())
    tn = int(((pred_binary == 0) & (true_binary == 0)).sum())
    fp = int(((pred_binary == 1) & (true_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (true_binary == 1)).sum())
    binary_precision = tp / (tp + fp) if tp + fp else 0.0
    binary_recall = tp / (tp + fn) if tp + fn else 0.0
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )
    ml_tp = int(((y_pred == 1) & (y_true == 1)).sum())
    ml_fp = int(((y_pred == 1) & (y_true == 0)).sum())
    ml_fn = int(((y_pred == 0) & (y_true == 1)).sum())
    ml_precision = ml_tp / (ml_tp + ml_fp) if ml_tp + ml_fp else 0.0
    ml_recall = ml_tp / (ml_tp + ml_fn) if ml_tp + ml_fn else 0.0
    ml_f1 = (
        2 * ml_precision * ml_recall / (ml_precision + ml_recall)
        if ml_precision + ml_recall
        else 0.0
    )
    return {
        "exact_agreement": float(exact.mean()),
        "binary_f1": binary_f1,
        "binary_mcc": float(matthews_corrcoef(true_binary, pred_binary)),
        "multilabel_micro_f1": ml_f1,
        "hamming_loss": float(np.not_equal(y_true, y_pred).mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def confidence_summary(confidence: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    bins = [(0.0, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.000001)]
    band_rows = []
    for lower, upper in bins:
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            band_rows.append(
                {
                    "lower": lower,
                    "upper": min(upper, 1.0),
                    "n": 0,
                    "mean_confidence": None,
                    "exact_agreement": None,
                }
            )
            continue
        successes = int(correct[mask].sum())
        band_rows.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "exact_agreement": float(correct[mask].mean()),
                "exact_agreement_wilson_95": wilson(successes, int(mask.sum())),
            }
        )

    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1 else (confidence <= upper)
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean() - correct[mask].mean())
            )
    return {
        "n_with_confidence": len(confidence),
        "mean": float(confidence.mean()),
        "median": float(np.median(confidence)),
        "q1": float(np.quantile(confidence, 0.25)),
        "q3": float(np.quantile(confidence, 0.75)),
        "empirical_exact_agreement": float(correct.mean()),
        "brier_for_exact_correctness": float(np.mean((confidence - correct) ** 2)),
        "ece_10_equal_width": ece,
        "bands": band_rows,
    }


def evaluate_system(
    name: str,
    model_rows: dict[str, dict[str, Any]],
    sample: list[dict[str, Any]],
    references: dict[str, tuple[str, ...]],
    stratum_by_id: dict[str, tuple[str, str, str]],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, bool]]:
    items: list[tuple[str, tuple[str, ...], tuple[str, ...], float, str]] = []
    available = 0
    for row in sample:
        chunk_id = row["chunk_id"]
        model_row = model_rows.get(chunk_id)
        if model_row is None:
            continue
        available += 1
        prediction = tuple(sorted(model_row.get("coarse_labels") or []))
        reference = references[chunk_id]
        if not prediction or not reference:
            continue
        confidence = model_row.get("score_confianza")
        items.append(
            (
                chunk_id,
                prediction,
                reference,
                float(confidence) if confidence is not None else math.nan,
                repr(stratum_by_id[chunk_id]),
            )
        )

    y_pred = np.stack([damage_vector(prediction) for _, prediction, _, _, _ in items])
    y_true = np.stack([damage_vector(reference) for _, _, reference, _, _ in items])
    exact = np.asarray(
        [prediction == reference for _, prediction, reference, _, _ in items]
    )
    true_binary = (y_true.sum(axis=1) > 0).astype(np.int8)
    pred_binary = (y_pred.sum(axis=1) > 0).astype(np.int8)

    micro = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    binary = precision_recall_fscore_support(
        true_binary, pred_binary, average="binary", zero_division=0
    )
    point = key_metrics(y_true, y_pred, exact)
    point.update(
        {
            "multilabel_micro_precision": float(micro[0]),
            "multilabel_micro_recall": float(micro[1]),
            "multilabel_micro_f1": float(micro[2]),
            "multilabel_macro_precision": float(macro[0]),
            "multilabel_macro_recall": float(macro[1]),
            "multilabel_macro_f1": float(macro[2]),
            "sample_jaccard": float(
                jaccard_score(y_true, y_pred, average="samples", zero_division=1)
            ),
            "binary_precision": float(binary[0]),
            "binary_recall_sensitivity": float(binary[1]),
            "binary_f1": float(binary[2]),
            "binary_specificity": float(
                ((pred_binary == 0) & (true_binary == 0)).sum()
                / max(1, (true_binary == 0).sum())
            ),
            "binary_balanced_accuracy": float(
                balanced_accuracy_score(true_binary, pred_binary)
            ),
            "binary_mcc": float(matthews_corrcoef(true_binary, pred_binary)),
            "binary_cohen_kappa": float(cohen_kappa_score(true_binary, pred_binary)),
        }
    )

    bootstrap_values: dict[str, list[float]] = defaultdict(list)
    item_strata = [stratum for *_, stratum in items]
    for indices in bootstrap_indices(item_strata, BOOTSTRAP_REPLICATES, rng):
        values = key_metrics(y_true[indices], y_pred[indices], exact[indices])
        for metric in (
            "exact_agreement",
            "binary_f1",
            "binary_mcc",
            "multilabel_micro_f1",
            "hamming_loss",
        ):
            bootstrap_values[metric].append(values[metric])
    bootstrap_ci = {
        metric: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for metric, values in bootstrap_values.items()
    }

    confidence_mask = np.asarray([not math.isnan(value) for *_, value, _ in items])
    confidence = np.asarray([value for *_, value, _ in items])[confidence_mask]
    confidence_correct = exact[confidence_mask].astype(float)

    per_label = {}
    for index, label in enumerate(LABELS):
        values = precision_recall_fscore_support(
            y_true[:, index], y_pred[:, index], average="binary", zero_division=0
        )
        per_label[label] = {
            "reference_support": int(y_true[:, index].sum()),
            "predicted_support": int(y_pred[:, index].sum()),
            "precision": float(values[0]),
            "recall": float(values[1]),
            "f1": float(values[2]),
            "cohen_kappa": float(cohen_kappa_score(y_true[:, index], y_pred[:, index])),
        }

    result = {
        "system": name,
        "sample_total": len(sample),
        "records_available": available,
        "answered": len(items),
        "coverage_over_sample": len(items) / len(sample),
        "abstention_over_sample": 1 - len(items) / len(sample),
        "reference_harm": int(true_binary.sum()),
        "predicted_harm": int(pred_binary.sum()),
        "point": point,
        "exact_agreement_wilson_95": wilson(int(exact.sum()), len(exact)),
        "stratified_bootstrap_percentile_95": bootstrap_ci,
        "confidence": confidence_summary(confidence, confidence_correct),
        "per_label": per_label,
    }
    correctness = {chunk_id: bool(value) for (chunk_id, *_), value in zip(items, exact)}
    return result, correctness


def provenance(
    sample_ids: set[str], events: list[dict[str, Any]]
) -> tuple[dict[str, int], dict[str, int]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[event["chunk_id"]] = event
    categories: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    for chunk_id in sample_ids:
        event = latest.get(chunk_id)
        if not event:
            categories["modelo_base_conservado"] += 1
            continue
        note = str(event.get("notes", ""))
        if event.get("batch_id") == UNLABELED_BATCH:
            categories["adjudicacion_hibrida_final"] += 1
            if "método=" in note:
                methods[note.split("método=", 1)[1].split(";", 1)[0]] += 1
        elif note.startswith("CODEX · auditoría muestral"):
            categories["cambio_muestral_alta_confianza"] += 1
        elif note.startswith(
            ("CODEX · auditoría dirigida", "CODEX | auditoría dirigida")
        ):
            categories["cambio_dirigido_alta_confianza"] += 1
        elif note.startswith("CODEX · revisión asistida"):
            categories["revision_prioritaria_pre_muestra"] += 1
        elif event.get("batch_id") == "CODEX-PRO-V3_2-FINAL-20260809":
            categories["revision_pro_v3_2_codex"] += 1
        else:
            categories["evento_manual_otro"] += 1
    return dict(sorted(categories.items())), dict(sorted(methods.items()))


def main() -> None:
    campaign = read_jsonl(CAMPAIGN_PATH)
    events = read_jsonl(EVENTS_PATH)
    flash_rows = {row["chunk_id"]: row for row in read_jsonl(FLASH_PATH)}
    pro_rows = {row["chunk_id"]: row for row in read_jsonl(PRO_PATH)}
    cascade_rows = {row["chunk_id"]: row for row in campaign}
    if FROZEN_SAMPLE_PATH.is_file():
        sample, stratum_by_id, initial_labels = load_frozen_sample(campaign)
    else:
        sample, stratum_by_id, initial_labels = build_sample(campaign, events)
    sample_ids = {row["chunk_id"] for row in sample}

    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[event["chunk_id"]] = event
    references = {
        row["chunk_id"]: effective_labels(row, latest.get(row["chunk_id"]))
        for row in sample
    }
    if any(not labels for labels in references.values()):
        raise RuntimeError("La referencia final todavía contiene casos sin etiqueta.")

    sample_corrections = {
        event["chunk_id"]
        for event in events
        if str(event.get("notes", "")).startswith(
            "CODEX · auditoría muestral estratificada 10% v1"
        )
    }
    if len(sample_corrections & sample_ids) != 62:
        raise RuntimeError(
            "La muestra reconstruida no recupera sus 62 correcciones registradas."
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    systems = {}
    correctness = {}
    for name, rows in (
        ("cascada_flash_pro_consolidada", cascade_rows),
        ("deepseek_v4_flash", flash_rows),
        ("deepseek_v4_pro", pro_rows),
    ):
        systems[name], correctness[name] = evaluate_system(
            name, rows, sample, references, stratum_by_id, rng
        )

    common_ids = sorted(
        set(correctness["deepseek_v4_flash"]) & set(correctness["deepseek_v4_pro"])
    )
    flash_only_correct = sum(
        correctness["deepseek_v4_flash"][chunk_id]
        and not correctness["deepseek_v4_pro"][chunk_id]
        for chunk_id in common_ids
    )
    pro_only_correct = sum(
        not correctness["deepseek_v4_flash"][chunk_id]
        and correctness["deepseek_v4_pro"][chunk_id]
        for chunk_id in common_ids
    )
    paired_difference = np.asarray(
        [
            int(correctness["deepseek_v4_pro"][chunk_id])
            - int(correctness["deepseek_v4_flash"][chunk_id])
            for chunk_id in common_ids
        ]
    )
    paired_strata = [repr(stratum_by_id[chunk_id]) for chunk_id in common_ids]
    paired_bootstrap = []
    for indices in bootstrap_indices(paired_strata, BOOTSTRAP_REPLICATES, rng):
        paired_bootstrap.append(float(paired_difference[indices].mean()))

    pro_routed_ids = [row["chunk_id"] for row in sample if row["chunk_id"] in pro_rows]
    intermodel_exact = np.mean(
        [
            tuple(sorted(flash_rows[chunk_id].get("coarse_labels") or []))
            == tuple(sorted(pro_rows[chunk_id].get("coarse_labels") or []))
            for chunk_id in pro_routed_ids
        ]
    )
    intermodel_binary = np.mean(
        [
            bool(set(flash_rows[chunk_id].get("coarse_labels") or []) & set(LABELS))
            == bool(set(pro_rows[chunk_id].get("coarse_labels") or []) & set(LABELS))
            for chunk_id in pro_routed_ids
        ]
    )

    initial_counter = Counter(
        (
            "PENDIENTE"
            if not initial_labels[row["chunk_id"]]
            else (
                "SEGURO"
                if initial_labels[row["chunk_id"]] == ("SEGURO",)
                else "AL_MENOS_UN_DANO"
            )
        )
        for row in sample
    )
    final_counter = Counter(
        "SEGURO" if references[row["chunk_id"]] == ("SEGURO",) else "AL_MENOS_UN_DANO"
        for row in sample
    )
    initial_assignments = Counter(
        label
        for row in sample
        for label in initial_labels[row["chunk_id"]]
        if label in LABELS
    )
    final_assignments = Counter(
        label
        for row in sample
        for label in references[row["chunk_id"]]
        if label in LABELS
    )
    final_harm = final_counter["AL_MENOS_UN_DANO"]
    provenance_counts, adjudication_methods = provenance(sample_ids, events)

    metrics = {
        "generated_by": "tools/report_audit_comparison.py",
        "sample": {
            "seed_text": SEED_TEXT,
            "sha_key": "sha256(seed_text|chunk_id)",
            "size": len(sample),
            "original_eligible_population": 157_719,
            "fraction_of_original_eligible": len(sample) / 157_719,
            "current_total_population": len(campaign),
            "current_eligible_population": sum(
                1
                for row in campaign
                if (latest.get(row["chunk_id"]) or {}).get("action") != "reject"
            ),
            "fraction_of_current_total": len(sample) / len(campaign),
            "fraction_of_current_eligible": len(sample)
            / sum(
                1
                for row in campaign
                if (latest.get(row["chunk_id"]) or {}).get("action") != "reject"
            ),
            "strata": len({stratum_by_id[row["chunk_id"]] for row in sample}),
            "initial_effective_composition": dict(sorted(initial_counter.items())),
            "final_reference_composition": dict(sorted(final_counter.items())),
            "initial_damage_label_assignments": dict(
                sorted(initial_assignments.items())
            ),
            "final_damage_label_assignments": dict(sorted(final_assignments.items())),
            "final_harm_prevalence": final_harm / len(sample),
            "final_harm_prevalence_wilson_95": wilson(final_harm, len(sample)),
            "registered_sample_corrections_recovered": len(
                sample_corrections & sample_ids
            ),
        },
        "reference": {
            "name": "CODEX–Sol-EH supervisado / adjudicación híbrida final",
            "warning": (
                "Referencia interna no independiente: los acuerdos conservaron la decisión previa y "
                "4.327 casos inicialmente pendientes se resolvieron con una jerarquía Flash/Pro/CODEX; "
                "la actualización v3.2 añadió una adjudicación CODEX dirigida de propuestas Pro."
            ),
            "numeric_sol_confidence_available": False,
            "provenance": provenance_counts,
            "unlabeled_batch_methods_in_sample": adjudication_methods,
        },
        "systems": systems,
        "paired_flash_vs_pro_on_common_answered": {
            "n": len(common_ids),
            "flash_exact_agreement": float(
                np.mean(
                    [
                        correctness["deepseek_v4_flash"][chunk_id]
                        for chunk_id in common_ids
                    ]
                )
            ),
            "pro_exact_agreement": float(
                np.mean(
                    [
                        correctness["deepseek_v4_pro"][chunk_id]
                        for chunk_id in common_ids
                    ]
                )
            ),
            "pro_minus_flash_exact_agreement": float(paired_difference.mean()),
            "stratified_bootstrap_percentile_95": [
                float(value) for value in np.quantile(paired_bootstrap, [0.025, 0.975])
            ],
            "flash_only_correct": flash_only_correct,
            "pro_only_correct": pro_only_correct,
            "mcnemar_exact_two_sided_p": float(
                binomtest(
                    min(flash_only_correct, pro_only_correct),
                    flash_only_correct + pro_only_correct,
                    0.5,
                ).pvalue
            ),
            "selection_warning": (
                "El panel común pertenece a la cola dirigida a Pro; no es una comparación aleatoria "
                "de toda la muestra."
            ),
        },
        "flash_vs_pro_on_all_pro_routed_in_sample": {
            "n": len(pro_routed_ids),
            "exact_agreement_including_joint_abstention": float(intermodel_exact),
            "binary_harm_agreement_treating_abstention_as_no_harm": float(
                intermodel_binary
            ),
            "warning": "La segunda cifra mezcla SEGURO y abstención en el polo no-daño.",
        },
        "inference": {
            "bootstrap": "percentile, resampling within the 35 frozen strata",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "proportion_interval": "Wilson score, 95%",
            "calibration_bins": "10 equal-width bins",
        },
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (CAMPAIGN_PATH, EVENTS_PATH, FLASH_PATH, PRO_PATH)
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "chunk_id",
        "stratum",
        "initial_effective_labels",
        "cascade_labels",
        "cascade_confidence",
        "flash_labels",
        "flash_confidence",
        "pro_available",
        "pro_labels",
        "pro_confidence",
        "final_reference_labels",
        "final_event_id",
        "final_reviewer",
        "final_batch_id",
    ]
    with SAMPLE_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in sorted(sample, key=lambda item: item["chunk_id"]):
            chunk_id = row["chunk_id"]
            flash = flash_rows.get(chunk_id, {})
            pro = pro_rows.get(chunk_id, {})
            final_event = latest.get(chunk_id, {})
            writer.writerow(
                {
                    "chunk_id": chunk_id,
                    "stratum": repr(stratum_by_id[chunk_id]),
                    "initial_effective_labels": "|".join(initial_labels[chunk_id]),
                    "cascade_labels": "|".join(row.get("coarse_labels") or []),
                    "cascade_confidence": row.get("score_confianza"),
                    "flash_labels": "|".join(flash.get("coarse_labels") or []),
                    "flash_confidence": flash.get("score_confianza"),
                    "pro_available": int(bool(pro)),
                    "pro_labels": "|".join(pro.get("coarse_labels") or []),
                    "pro_confidence": pro.get("score_confianza"),
                    "final_reference_labels": "|".join(references[chunk_id]),
                    "final_event_id": final_event.get("event_id", ""),
                    "final_reviewer": final_event.get("reviewer", ""),
                    "final_batch_id": final_event.get("batch_id", ""),
                }
            )

    print(
        json.dumps(
            {"metrics": str(METRICS_PATH), "sample": str(SAMPLE_PATH)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
