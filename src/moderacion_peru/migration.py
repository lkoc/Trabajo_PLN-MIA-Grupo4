from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .datasets import assert_no_video_leakage, stable_video_split
from .taxonomy import TaxonomyContract, load_taxonomy


def migrate_record(
    record: dict[str, Any],
    taxonomy: TaxonomyContract | None = None,
) -> dict[str, Any]:
    contract = taxonomy or load_taxonomy()
    legacy = list(record.get("coarse_labels") or record.get("labels") or [])
    migrated = dict(record)
    migrated["schema_version"] = "2.1.0"
    migrated["taxonomy_version"] = contract.version
    migrated["legacy_coarse_labels"] = legacy
    migrated["label_source_original"] = record.get("label_source")

    if not legacy:
        migrated["coarse_labels"] = []
        migrated["needs_review"] = True
        migrated["training_eligible"] = False
        migrated["decision_status"] = "needs_review"
        migrated["migration_warning"] = "empty_legacy_labels_not_assumed_safe"
        return migrated

    try:
        labels = contract.migrate_legacy_categories(legacy)
    except ValueError as exc:
        migrated["coarse_labels"] = []
        migrated["needs_review"] = True
        migrated["training_eligible"] = False
        migrated["decision_status"] = "needs_review"
        migrated["migration_warning"] = str(exc)
        return migrated

    migrated["coarse_labels"] = list(labels)
    migrated["needs_review"] = bool(record.get("needs_review", False))
    migrated["training_eligible"] = bool(record.get("training_eligible", True))
    migrated["decision_status"] = (
        "needs_review" if migrated["needs_review"] else "resolved"
    )
    return migrated


def migrate_jsonl(
    source: str | Path,
    destination: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("La migración nunca sobrescribe el archivo fuente")
    rows = [migrate_record(row) for row in read_jsonl(source_path)]
    for row in rows:
        video_id = str(row.get("video_id") or str(row["chunk_id"]).split("_", 1)[0])
        row["video_id"] = video_id
        row["split"] = row.get("split") or stable_video_split(video_id)
    assert_no_video_leakage(rows)
    write_jsonl_atomic(destination_path, rows)
    counters = Counter()
    for row in rows:
        counters["rows"] += 1
        counters["training_eligible"] += bool(row.get("training_eligible"))
        counters["needs_review"] += bool(row.get("needs_review"))
        counters[f"split:{row['split']}"] += 1
        for label in row.get("coarse_labels", []):
            counters[f"label:{label}"] += 1
    manifest = {
        "schema_version": "2.1.0",
        "operation": "legacy_to_five_trained_outputs",
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "destination": str(destination_path),
        "destination_sha256": sha256_file(destination_path),
        "counters": dict(counters),
        "safe_policy": "only_explicit_legacy_SEGURO",
        "split_policy": "sha256(video_id, seed=20260805), grouped 70/15/15",
        "damage_merge": ["ACOSO_PERSONAL", "AMENAZA_DIRECTA", "ACOSO_AMENAZA"],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
