from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_json_atomic
from .migration import migrate_record
from .providers import OllamaProvider, ProviderError
from .taxonomy import load_taxonomy


def _rank(seed: int, chunk_id: str) -> str:
    return hashlib.sha256(f"{seed}|{chunk_id}".encode()).hexdigest()


def multilabel_metrics(
    references: list[list[str]],
    predictions: list[list[str]],
    labels: Iterable[str],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(label in ref and label in pred for ref, pred in zip(references, predictions))
        fp = sum(label not in ref and label in pred for ref, pred in zip(references, predictions))
        fn = sum(label in ref and label not in pred for ref, pred in zip(references, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": tp + fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    f1_macro = sum(float(row["f1"]) for row in per_label.values()) / len(per_label) if per_label else 0.0
    return {"per_label": per_label, "f1_macro": f1_macro}


def build_human_pilot(
    source: str | Path,
    *,
    size: int = 200,
    seed: int = 20260805,
) -> list[dict[str, Any]]:
    taxonomy = load_taxonomy()
    eligible: list[dict[str, Any]] = []
    for row in read_jsonl(source):
        source_name = str(row.get("label_source", "")).lower()
        if "human" not in source_name and "humano" not in source_name:
            continue
        if "accept" in source_name or "acept" in source_name:
            continue
        migrated = migrate_record(row, taxonomy)
        if not migrated.get("training_eligible") or not migrated.get("coarse_labels"):
            continue
        eligible.append(migrated)
    if not eligible:
        raise ValueError("No se encontraron decisiones humanas modificadas o gruesas elegibles")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = "+".join(row["coarse_labels"])
        strata[key].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: _rank(seed, row["chunk_id"]))

    sample: list[dict[str, Any]] = []
    round_index = 0
    ordered_keys = sorted(strata)
    while len(sample) < min(size, len(eligible)):
        added = False
        for key in ordered_keys:
            if round_index < len(strata[key]):
                sample.append(strata[key][round_index])
                added = True
                if len(sample) >= size:
                    break
        if not added:
            break
        round_index += 1
    return sample


def run_ollama_pilot(
    sample: Iterable[dict[str, Any]],
    models: Iterable[str],
    destination: str | Path,
) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    rows = list(sample)
    report: dict[str, Any] = {
        "schema_version": "2.1.0",
        "sample_size": len(rows),
        "taxonomy_contract": taxonomy.contract_id,
        "models": {},
        "sample_meets_minimum": len(rows) >= 200,
        "execution_complete": False,
        "selection_evaluable": False,
        "selection": None,
        "reference_counts": dict(
            Counter(label for row in rows for label in row.get("coarse_labels", []))
        ),
        "limitations": [
            "Las decisiones humanas históricas fueron asistidas y no son un gold standard ciego.",
            "La selección sustantiva solo se considera evaluable con 200 casos."
        ],
    }
    for model in models:
        provider = OllamaProvider(model=model, retries=1)
        try:
            probe = provider.probe()
            if not probe["model_available"]:
                raise ProviderError(f"El modelo {model} no está instalado en Ollama")
        except ProviderError as exc:
            report["models"][model] = {
                "preflight": {"available": False, "error": str(exc)},
                "technical_validity": 0.0,
                "passes_99_percent_gate": False,
                "exact_match": 0.0,
                "f1_macro_harms": 0.0,
                "harm_as_safe_count": 0,
                "seconds": 0.0,
                "chunks_per_second": 0.0,
                "prediction_counts": {},
                "errors": [{"chunk_id": "__preflight__", "error": str(exc)}],
                "predictions": [],
            }
            continue
        started = time.perf_counter()
        errors: list[dict[str, str]] = []
        predictions = []
        false_safe = 0
        exact = 0
        label_counts = Counter()
        for row in rows:
            try:
                prediction = provider.annotate(row)
            except (ProviderError, ValueError) as exc:
                errors.append({"chunk_id": row["chunk_id"], "error": str(exc)})
                continue
            reference = tuple(row["coarse_labels"])
            predicted = tuple(prediction.coarse_labels)
            exact += predicted == reference
            if taxonomy.safe_label in predicted and any(
                label in reference for label in taxonomy.damage_labels
            ):
                false_safe += 1
            for label in predicted:
                label_counts[label] += 1
            predictions.append(
                {
                    "chunk_id": row["chunk_id"],
                    "reference": list(reference),
                    "prediction": prediction.model_dump(mode="json"),
                }
            )
        elapsed = time.perf_counter() - started
        valid = len(predictions)
        references = [item["reference"] for item in predictions]
        predicted_labels = [item["prediction"]["coarse_labels"] for item in predictions]
        harm_metrics = multilabel_metrics(references, predicted_labels, taxonomy.damage_labels)
        report["models"][model] = {
            "preflight": probe,
            "technical_validity": valid / len(rows) if rows else 0,
            "passes_99_percent_gate": bool(rows) and valid / len(rows) >= 0.99,
            "exact_match": exact / valid if valid else 0,
            "f1_macro_harms": harm_metrics["f1_macro"],
            "per_label_harms": harm_metrics["per_label"],
            "harm_as_safe_count": false_safe,
            "seconds": elapsed,
            "chunks_per_second": valid / elapsed if elapsed else 0,
            "prediction_counts": dict(label_counts),
            "errors": errors,
            "predictions": predictions,
        }
    report["execution_complete"] = all(
        result.get("preflight", {}).get("available", True)
        for result in report["models"].values()
    )
    report["selection_evaluable"] = bool(
        report["sample_meets_minimum"] and report["execution_complete"]
    )
    if report["selection_evaluable"]:
        eligible = [
            (name, result)
            for name, result in report["models"].items()
            if result["passes_99_percent_gate"]
        ]
        if eligible:
            eligible.sort(
                key=lambda item: (
                    item[1]["harm_as_safe_count"],
                    -item[1]["f1_macro_harms"],
                    -item[1]["chunks_per_second"],
                    item[0],
                )
            )
            report["selection"] = eligible[0][0]
    write_json_atomic(destination, report)
    return report
