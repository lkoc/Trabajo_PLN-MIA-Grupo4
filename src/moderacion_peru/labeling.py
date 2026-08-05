from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl_once, read_jsonl
from .providers.base import AnnotationProvider, ProviderError


def annotate_incremental(
    records: Iterable[dict[str, Any]],
    provider: AnnotationProvider,
    output_path: str | Path,
    *,
    error_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    output = Path(output_path)
    completed = {row["chunk_id"] for row in read_jsonl(output)} if output.exists() else set()
    pending = []
    seen = set()
    for record in records:
        chunk_id = record.get("chunk_id")
        if not chunk_id or chunk_id in completed or chunk_id in seen:
            continue
        seen.add(chunk_id)
        pending.append(record)
        if limit is not None and len(pending) >= limit:
            break

    counters = {"already_completed": len(completed), "selected": len(pending), "labeled": 0, "errors": 0}
    for record in pending:
        try:
            annotation = provider.annotate(record)
            append_jsonl_once(output, [annotation.model_dump(mode="json")], id_field="chunk_id")
            counters["labeled"] += 1
        except (ProviderError, ValueError, RuntimeError) as exc:
            counters["errors"] += 1
            if error_path:
                append_jsonl_once(
                    error_path,
                    [{"chunk_id": record.get("chunk_id"), "error": str(exc)}],
                    id_field="chunk_id",
                )
    return counters


def load_pending_chunks(source: str | Path, output: str | Path) -> list[dict[str, Any]]:
    completed = {row["chunk_id"] for row in read_jsonl(output)} if Path(output).exists() else set()
    return [row for row in read_jsonl(source) if row.get("chunk_id") not in completed]

