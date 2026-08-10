"""Materializa estadísticas reproducibles del corte final de la auditoría.

No cambia etiquetas. Compara el último snapshot entrenable con el snapshot
auditado anterior y resume campaña, decisiones efectivas, splits y eventos.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "datos/etiquetado/consolidado/anotaciones_v2.jsonl"
REVIEWED = ROOT / "datos/etiquetado/consolidado/anotaciones_revisadas_v2.jsonl"
EVENTS = ROOT / "datos/etiquetado/humano/labeling_events_v2.jsonl"
CURRENT_DATASET = ROOT / "datos/model_ready/v2/dataset_5_salidas.jsonl"
PREVIOUS_DATASET = (
    ROOT
    / "datos/model_ready/v2/snapshots/v2.1.0-05854b628c1a3b4d/dataset_5_salidas.jsonl"
)
CURRENT_MANIFEST = (
    ROOT
    / "datos/model_ready/v2/snapshots/v2.1.0-e354b3248f7418f1/snapshot_manifest.json"
)
CODEX_V32_MANIFEST = (
    ROOT / "datos/etiquetado/humano/codex_latest_pro_v3_2_reviewed.events.manifest.json"
)
CODEX_V32_EVENTS = (
    ROOT / "datos/etiquetado/humano/codex_latest_pro_v3_2_reviewed.events.jsonl"
)
PANEL_METRICS = ROOT / "docs/artefactos/auditoria_16k_panel_actual_v3_2_metrics.json"
OUTPUT = ROOT / "docs/artefactos/auditoria_estado_final_182461.json"

DAMAGE_LABELS = (
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
)


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(counter: Counter[str]) -> dict[str, float | int]:
    values = [counter[label] for label in DAMAGE_LABELS]
    mean = statistics.fmean(values)
    entropy = -sum(
        (value / sum(values)) * math.log(value / sum(values))
        for value in values
        if value
    )
    return {
        "least_label": DAMAGE_LABELS[values.index(min(values))],
        "least_count": min(values),
        "most_label": DAMAGE_LABELS[values.index(max(values))],
        "most_count": max(values),
        "max_min_ratio": max(values) / min(values),
        "population_cv": statistics.pstdev(values) / mean,
        "normalized_shannon_entropy": entropy / math.log(len(values)),
    }


def unit_summary(counter: Counter[str]) -> dict[str, float | int]:
    values = sorted(counter.values())
    return {
        "units": len(values),
        "mean_chunks": statistics.fmean(values),
        "median_chunks": statistics.median(values),
        "min_chunks": min(values),
        "max_chunks": max(values),
    }


def summarize_scope(path: Path) -> dict[str, int]:
    chunks = 0
    videos: set[str] = set()
    channels: set[str] = set()
    for row in rows(path):
        chunks += 1
        videos.add(str(row.get("video_id") or "DESCONOCIDO"))
        channels.add(
            str(row.get("channel_title") or row.get("channel_id") or "DESCONOCIDO")
        )
    return {"chunks": chunks, "videos": len(videos), "channels": len(channels)}


def summarize_effective(
    path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    channels: Counter[str] = Counter()
    videos: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    labelsets: Counter[str] = Counter()
    effective_by_id: dict[str, tuple[str, ...]] = {}
    counters = Counter()
    for row in rows(path):
        counters["rows"] += 1
        status = str(row.get("decision_status") or "")
        if status == "excluded":
            counters["excluded"] += 1
            continue
        coarse = tuple(sorted(row.get("coarse_labels") or ()))
        if not row.get("training_eligible") or row.get("needs_review") or not coarse:
            counters["pending"] += 1
            continue
        counters["eligible"] += 1
        effective_by_id[str(row["chunk_id"])] = coarse
        if coarse == ("SEGURO",):
            counters["safe"] += 1
        else:
            counters["harm"] += 1
        labels.update(coarse)
        labelsets["|".join(coarse)] += 1
        channel = str(
            row.get("channel_title") or row.get("channel_id") or "DESCONOCIDO"
        )
        channels[channel] += 1
        videos[str(row.get("video_id") or "DESCONOCIDO")] += 1
    result = {
        **dict(counters),
        "labels": dict(sorted(labels.items())),
        "labelsets": dict(sorted(labelsets.items())),
        "damage_assignments": sum(labels[label] for label in DAMAGE_LABELS),
        "multilabel_harm_chunks": sum(
            count for key, count in labelsets.items() if key != "SEGURO" and "|" in key
        ),
        "channel_distribution": unit_summary(channels),
        "video_distribution": unit_summary(videos),
        "top_channels": [
            {"channel": key, "chunks": value} for key, value in channels.most_common(10)
        ],
        "damage_balance": distribution(labels),
    }
    return result, effective_by_id


def summarize_dataset(path: Path) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels_by_id: dict[str, tuple[str, ...]] = {}
    split_videos: dict[str, set[str]] = defaultdict(set)
    split_channels: dict[str, set[str]] = defaultdict(set)
    for row in rows(path):
        split = str(row["split"])
        labels = tuple(sorted(row.get("coarse_labels") or ()))
        labels_by_id[str(row["chunk_id"])] = labels
        split_counts[split]["chunks"] += 1
        split_counts[split]["safe" if labels == ("SEGURO",) else "harm"] += 1
        split_counts[split].update(labels)
        split_videos[split].add(str(row["video_id"]))
        split_channels[split].add(str(row.get("channel_id") or "DESCONOCIDO"))
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        counts = split_counts[split]
        output[split] = {
            **dict(counts),
            "videos": len(split_videos[split]),
            "channels": len(split_channels[split]),
            "damage_balance": distribution(counts),
        }
    return output, labels_by_id


def summarize_events(path: Path) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    total = 0
    event_ids: set[str] = set()
    batches: Counter[str] = Counter()
    reviewers: Counter[str] = Counter()
    for row in rows(path):
        total += 1
        event_ids.add(str(row["event_id"]))
        batches[str(row.get("batch_id") or "SIN_LOTE")] += 1
        reviewers[str(row.get("reviewer") or "DESCONOCIDO")] += 1
        previous = latest.get(str(row["chunk_id"]))
        key = (str(row.get("created_at") or ""), str(row["event_id"]))
        previous_key = (
            (str(previous.get("created_at") or ""), str(previous["event_id"]))
            if previous
            else None
        )
        if previous_key is None or key > previous_key:
            latest[str(row["chunk_id"])] = row
    actions = Counter(str(row["action"]) for row in latest.values())
    return {
        "events": total,
        "unique_event_ids": len(event_ids),
        "latest_reviewed_chunks": len(latest),
        "latest_actions": dict(sorted(actions.items())),
        "events_by_batch": dict(batches.most_common()),
        "events_by_reviewer": dict(reviewers.most_common()),
    }


def main() -> None:
    effective, effective_by_id = summarize_effective(REVIEWED)
    splits, current_by_id = summarize_dataset(CURRENT_DATASET)
    _, previous_by_id = summarize_dataset(PREVIOUS_DATASET)
    common = set(previous_by_id) & set(current_by_id)
    added = set(current_by_id) - set(previous_by_id)
    removed = set(previous_by_id) - set(current_by_id)
    changed = {
        chunk_id
        for chunk_id in common
        if previous_by_id[chunk_id] != current_by_id[chunk_id]
    }
    added_labels = Counter(
        label for chunk_id in added for label in current_by_id[chunk_id]
    )
    added_safe = sum(current_by_id[chunk_id] == ("SEGURO",) for chunk_id in added)
    transitions = Counter(
        f"{'|'.join(previous_by_id[chunk_id])} -> {'|'.join(current_by_id[chunk_id])}"
        for chunk_id in changed
    )
    current_manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))
    codex_manifest = json.loads(CODEX_V32_MANIFEST.read_text(encoding="utf-8"))
    codex_v32_ids = {str(row["chunk_id"]) for row in rows(CODEX_V32_EVENTS)}
    panel = json.loads(PANEL_METRICS.read_text(encoding="utf-8"))
    panel_size = int(panel["sample"]["size"])
    codex_overlap_previous_snapshot = len(codex_v32_ids & set(previous_by_id))
    if codex_overlap_previous_snapshot:
        raise RuntimeError(
            "La capa CODEX v3.2 se solapa con el snapshot del panel histórico"
        )
    combined_audit_footprint = panel_size + len(codex_v32_ids)
    payload = {
        "generated_by": "tools/report_audit_current_state.py",
        "cutoff_date": "2026-08-09",
        "campaign_scope_before_exclusions": summarize_scope(CAMPAIGN),
        "effective_dataset": effective,
        "training_snapshot": {
            "snapshot_id": current_manifest["snapshot_id"],
            "dataset_sha256": current_manifest["dataset_sha256"],
            "content_signature": current_manifest["content_signature"],
            "rows": len(current_by_id),
            "splits": splits,
            "target_2000_total_train_validation_test": {
                "criterion_met": all(
                    effective["labels"][label] >= 2000 for label in DAMAGE_LABELS
                ),
                "support": {
                    label: effective["labels"][label] for label in DAMAGE_LABELS
                },
                "shortfall": {
                    label: max(0, 2000 - effective["labels"][label])
                    for label in DAMAGE_LABELS
                },
            },
            "train_only_diagnostic_shortfall_not_a_stop_condition": {
                label: max(0, 2000 - splits["train"].get(label, 0))
                for label in DAMAGE_LABELS
            },
        },
        "increment_since_previous_audit_snapshot": {
            "previous_rows": len(previous_by_id),
            "current_rows": len(current_by_id),
            "net_rows": len(current_by_id) - len(previous_by_id),
            "common_rows": len(common),
            "added_rows": len(added),
            "removed_rows": len(removed),
            "changed_labelsets_in_common_rows": len(changed),
            "changed_labelsets_fraction_of_common": len(changed) / len(common),
            "common_transitions": dict(transitions.most_common()),
            "added_safe": added_safe,
            "added_harm": len(added) - added_safe,
            "added_labels": dict(sorted(added_labels.items())),
        },
        "human_review_events": summarize_events(EVENTS),
        "codex_pro_v3_2": codex_manifest["statistics"],
        "frozen_panel": panel["sample"],
        "combined_audit_coverage": {
            "frozen_stratified_panel": panel_size,
            "directed_codex_v3_2": len(codex_v32_ids),
            "overlap_between_layers": codex_overlap_previous_snapshot,
            "unique_chunks": combined_audit_footprint,
            "fraction_of_current_eligible": combined_audit_footprint
            / len(current_by_id),
            "interpretation": (
                "El panel sostiene la inferencia longitudinal; la capa CODEX es "
                "dirigida a incertidumbre y no se interpreta como muestra aleatoria."
            ),
        },
        "panel_systems": panel["systems"],
        "paired_flash_vs_pro": panel["paired_flash_vs_pro_on_common_answered"],
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                CAMPAIGN,
                REVIEWED,
                EVENTS,
                CURRENT_DATASET,
                PREVIOUS_DATASET,
                CODEX_V32_EVENTS,
            )
        },
    }
    if len(effective_by_id) != len(current_by_id) or set(effective_by_id) != set(
        current_by_id
    ):
        raise RuntimeError(
            "La vista efectiva y el snapshot entrenable no contienen los mismos IDs"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(OUTPUT), "rows": len(current_by_id)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
