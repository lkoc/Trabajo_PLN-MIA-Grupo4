from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_json_sha256, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .schemas import ModelReadyRecord
from .taxonomy import load_taxonomy


def stable_video_split(video_id: str, seed: int = 20260805) -> str:
    value = int(hashlib.sha256(f"{seed}|{video_id}".encode()).hexdigest()[:16], 16) / 16**16
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def materialize_training_snapshot(
    source: str | Path,
    destination: str | Path,
    *,
    previous_assignments: dict[str, str] | None = None,
    seed: int = 20260805,
) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    assignments = previous_assignments or {}
    rows = []
    counts = Counter()
    seen_chunks: set[str] = set()
    video_splits: dict[str, str] = {}
    for row in read_jsonl(source):
        if not row.get("training_eligible") or not row.get("coarse_labels"):
            counts["excluded"] += 1
            continue
        labels = taxonomy.normalize_categories(row["coarse_labels"])
        chunk_id = str(row["chunk_id"])
        if not row.get("video_id"):
            raise ValueError(
                f"Falta video_id explícito en {chunk_id}; no se puede deducir de chunk_id"
            )
        video_id = str(row["video_id"])
        if chunk_id in seen_chunks:
            counts["duplicates"] += 1
            continue
        seen_chunks.add(chunk_id)
        split = assignments.get(video_id) or row.get("split") or stable_video_split(video_id, seed)
        if video_id in video_splits and video_splits[video_id] != split:
            raise ValueError(f"Fuga de video detectada para {video_id}")
        video_splits[video_id] = split
        prepared = ModelReadyRecord(
            chunk_id=chunk_id,
            video_id=video_id,
            text=str(row["text"]),
            coarse_labels=list(labels),
            fine_labels=list(row.get("fine_labels", [])),
            flags_reference_only=list(row.get("flags_reference_only", row.get("flags", []))),
            label_source=str(row.get("label_source") or "unknown"),
            sample_weight=float(row.get("sample_weight", 1.0)),
            campaign=row.get("campaign"),
            split=split,
            needs_review=False,
            training_eligible=True,
            decision_status="resolved",
            legacy_coarse_labels=list(row.get("legacy_coarse_labels", [])),
            label_source_original=row.get("label_source_original"),
            migration_warning=row.get("migration_warning"),
        ).model_dump(mode="json")
        rows.append(prepared)
        counts[f"split:{split}"] += 1
        for label in labels:
            counts[f"label:{label}"] += 1
    rows.sort(key=lambda row: (row["split"], row["video_id"], row["chunk_id"]))
    write_jsonl_atomic(destination, rows)
    return {"rows": len(rows), "videos": len(video_splits), "counts": dict(counts)}


def previous_video_assignments(source: str | Path) -> dict[str, str]:
    path = Path(source)
    if not path.is_file():
        return {}
    assignments: dict[str, str] = {}
    for row in read_jsonl(path):
        video_id = str(row["video_id"])
        split = str(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"Snapshot previo contiene fuga para {video_id}")
    return assignments


def materialize_versioned_training_snapshot(
    source: str | Path,
    canonical_destination: str | Path,
    *,
    snapshots_dir: str | Path | None = None,
    seed: int = 20260805,
) -> dict[str, Any]:
    """Crea un snapshot inmutable y actualiza la vista canónica solo si cambió.

    Las asignaciones anteriores por ``video_id`` se heredan. El ID del snapshot
    depende de sus filas canónicas, por lo que repetir la etapa es un no-op.
    """

    source_path = Path(source)
    canonical = Path(canonical_destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    assignments = previous_video_assignments(canonical)
    prepared_rows: list[dict[str, Any]] = []
    taxonomy = load_taxonomy()
    for row in read_jsonl(source_path):
        if not row.get("training_eligible") or row.get("needs_review") or row.get("decision_status") != "resolved":
            continue
        if not row.get("coarse_labels"):
            continue
        if not row.get("video_id"):
            raise ValueError(
                f"Falta video_id explícito en {row.get('chunk_id')}; revise 02_05"
            )
        video_id = str(row["video_id"])
        split = assignments.get(video_id) or stable_video_split(video_id, seed)
        record = ModelReadyRecord(
            chunk_id=str(row["chunk_id"]),
            video_id=video_id,
            text=str(row["text"]),
            coarse_labels=list(taxonomy.normalize_categories(row["coarse_labels"])),
            fine_labels=list(row.get("fine_labels", [])),
            flags_reference_only=list(row.get("flags", row.get("flags_reference_only", []))),
            label_source=str(row.get("label_source") or "unknown"),
            sample_weight=float(row.get("sample_weight", 1.0)),
            campaign=row.get("campaign"),
            split=split,
            needs_review=False,
            training_eligible=True,
            decision_status="resolved",
            legacy_coarse_labels=list(row.get("legacy_coarse_labels", [])),
            label_source_original=row.get("label_source_original"),
            migration_warning=row.get("migration_warning"),
        )
        prepared_rows.append(record.model_dump(mode="json"))
    if not prepared_rows:
        raise ValueError("No existen decisiones resueltas y entrenables para materializar")
    unique: dict[str, dict[str, Any]] = {}
    for row in prepared_rows:
        previous = unique.get(row["chunk_id"])
        if previous is not None and canonical_json_sha256(previous) != canonical_json_sha256(row):
            raise ValueError(f"Decisiones entrenables duplicadas para {row['chunk_id']}")
        unique[row["chunk_id"]] = row
    prepared_rows = sorted(unique.values(), key=lambda row: (row["split"], row["video_id"], row["chunk_id"]))
    assert_no_video_leakage(prepared_rows)
    content_signature = canonical_json_sha256(prepared_rows)
    snapshot_id = f"v{taxonomy.version}-{content_signature[:16]}"
    snapshot_root = Path(snapshots_dir) if snapshots_dir else canonical.parent / "snapshots"
    snapshot_path = snapshot_root / snapshot_id / canonical.name
    manifest_path = snapshot_path.with_name("snapshot_manifest.json")
    if snapshot_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["content_signature"] != content_signature:
            raise ValueError(f"Colisión o snapshot alterado: {snapshot_path}")
        status = "noop"
    else:
        write_jsonl_atomic(snapshot_path, prepared_rows)
        counts = Counter(row["split"] for row in prepared_rows)
        label_counts = Counter(label for row in prepared_rows for label in row["coarse_labels"])
        manifest = {
            "schema_version": "2.1.0",
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "taxonomy_contract": taxonomy.contract_id,
            "taxonomy_version": taxonomy.version,
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "dataset": str(snapshot_path),
            "dataset_sha256": sha256_file(snapshot_path),
            "content_signature": content_signature,
            "split_policy": f"preserve previous; otherwise sha256(video_id, seed={seed}), grouped 70/15/15",
            "counts": {"rows": len(prepared_rows), "videos": len({row['video_id'] for row in prepared_rows}), **{f"split:{key}": value for key, value in counts.items()}, **{f"label:{key}": value for key, value in label_counts.items()}},
        }
        write_json_atomic(manifest_path, manifest)
        status = "created"
    snapshot_sha = sha256_file(snapshot_path)
    canonical_changed = not canonical.is_file() or sha256_file(canonical) != snapshot_sha
    if canonical_changed:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temporary = canonical.with_name(f".{canonical.name}.{snapshot_id}.partial")
        shutil.copyfile(snapshot_path, temporary)
        if sha256_file(temporary) != snapshot_sha:
            temporary.unlink(missing_ok=True)
            raise ValueError("La vista canónica no conserva el SHA-256 del snapshot")
        temporary.replace(canonical)
    return {
        "status": status if not canonical_changed else "updated",
        "snapshot_id": snapshot_id,
        "snapshot": str(snapshot_path),
        "manifest": str(manifest_path),
        "canonical": str(canonical),
        "dataset_sha256": snapshot_sha,
        "rows": len(prepared_rows),
        "videos": len({row["video_id"] for row in prepared_rows}),
    }


def load_split(source: str | Path, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(source)
        if row.get("split") == split
        and row.get("training_eligible", True)
        and not row.get("needs_review", False)
        and row.get("decision_status", "resolved") == "resolved"
    ]


def assert_no_video_leakage(rows: Iterable[dict[str, Any]]) -> None:
    assignments: dict[str, str] = {}
    for row in rows:
        video_id = str(row["video_id"])
        split = str(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"El video {video_id} aparece en {previous} y {split}")


def audit_training_snapshot(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Audita cobertura fina/flags y coherencia con las cinco salidas, sin usar GPU."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    dataset_sha = sha256_file(source_path)
    if destination_path.is_file():
        previous = json.loads(destination_path.read_text(encoding="utf-8"))
        if previous.get("dataset_sha256") == dataset_sha:
            return {"status": "noop", **previous}
    taxonomy = load_taxonomy()
    rows = list(read_jsonl(source_path))
    fine_counts = Counter()
    flag_counts = Counter()
    split_counts = Counter()
    inconsistencies = []
    for row in rows:
        split_counts[row["split"]] += 1
        fine_counts.update(row.get("fine_labels", []))
        flag_counts.update(row.get("flags_reference_only", []))
        fine = row.get("fine_labels", [])
        if fine:
            derived = set(taxonomy.derive_categories(fine))
            if not derived.issubset(set(row.get("coarse_labels", []))):
                inconsistencies.append(row["chunk_id"])
    payload = {
        "schema_version": "2.1.0",
        "operation": "audit_fine_labels_and_flags",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(source_path),
        "dataset_sha256": dataset_sha,
        "rows": len(rows),
        "splits": dict(split_counts),
        "fine_label_counts": {label: fine_counts[label] for label in taxonomy.fine_labels},
        "flag_counts": {flag: flag_counts[flag] for flag in taxonomy.flags},
        "rows_without_fine_reference": sum(not row.get("fine_labels") for row in rows),
        "fine_to_coarse_inconsistencies": len(inconsistencies),
        "inconsistent_chunk_ids_sample": inconsistencies[:100],
        "interpretation": "Las finas y flags son referencias auxiliares; no se atribuyen métricas predictivas sin predicciones gold separadas.",
    }
    write_json_atomic(destination_path, payload)
    return {"status": "updated", **payload}
