from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl_atomic
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
        video_id = str(row.get("video_id") or chunk_id.split("_", 1)[0])
        if chunk_id in seen_chunks:
            counts["duplicates"] += 1
            continue
        seen_chunks.add(chunk_id)
        split = assignments.get(video_id) or row.get("split") or stable_video_split(video_id, seed)
        if video_id in video_splits and video_splits[video_id] != split:
            raise ValueError(f"Fuga de video detectada para {video_id}")
        video_splits[video_id] = split
        prepared = dict(row)
        prepared["coarse_labels"] = list(labels)
        prepared["split"] = split
        rows.append(prepared)
        counts[f"split:{split}"] += 1
        for label in labels:
            counts[f"label:{label}"] += 1
    rows.sort(key=lambda row: (row["split"], row["video_id"], row["chunk_id"]))
    write_jsonl_atomic(destination, rows)
    return {"rows": len(rows), "videos": len(video_splits), "counts": dict(counts)}


def load_split(source: str | Path, split: str) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(source) if row.get("split") == split]


def assert_no_video_leakage(rows: Iterable[dict[str, Any]]) -> None:
    assignments: dict[str, str] = {}
    for row in rows:
        video_id = str(row["video_id"])
        split = str(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"El video {video_id} aparece en {previous} y {split}")

