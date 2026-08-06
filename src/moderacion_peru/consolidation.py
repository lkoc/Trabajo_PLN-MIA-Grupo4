from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_json_sha256, input_signature, read_jsonl, write_json_atomic, write_jsonl_atomic
from .schemas import AnnotationRecord, ReviewEvent
from .taxonomy import load_taxonomy


DEFAULT_PRECEDENCE = {
    "human_modified": 50,
    "human": 45,
    "human_accepted": 40,
    "llm_remote_review": 30,
    "llm_remote": 20,
    "deepseek_remote": 20,
    "ollama_local": 10,
    "huggingface_local": 10,
    "migration": 5,
}


def consolidate_annotations(
    sources: Iterable[str | Path],
    destination: str | Path,
    *,
    precedence: dict[str, int] | None = None,
    chunks_source: str | Path | None = None,
) -> dict[str, Any]:
    priorities = precedence or DEFAULT_PRECEDENCE
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        for row in read_jsonl(source):
            candidates[row["chunk_id"]].append(row)

    chunk_lookup = {
        str(row["chunk_id"]): row
        for row in (read_jsonl(chunks_source) if chunks_source and Path(chunks_source).is_file() else [])
    }
    selected = []
    conflicts = 0
    for chunk_id, rows in candidates.items():
        rows.sort(
            key=lambda row: (
                priorities.get(str(row.get("label_source", "")), 0),
                str(row.get("created_at", "")),
            ),
            reverse=True,
        )
        winner = dict(rows[0])
        chunk = chunk_lookup.get(chunk_id, {})
        for field, alternatives in {
            "video_id": ("video_id",),
            "start_seconds": ("start_seconds",),
            "end_seconds": ("end_seconds",),
            "video_title": ("video_title", "title"),
            "channel_title": ("channel_title", "channel"),
            "source_url": ("source_url", "url"),
            "cohort": ("cohort",),
        }.items():
            if winner.get(field) is None:
                winner[field] = next((chunk.get(name) for name in alternatives if chunk.get(name) is not None), None)
        top_priority = priorities.get(str(winner.get("label_source", "")), 0)
        tied = [row for row in rows if priorities.get(str(row.get("label_source", "")), 0) == top_priority]
        decisions = {tuple(row.get("coarse_labels", [])) for row in tied}
        if len(decisions) > 1:
            conflicts += 1
            winner["coarse_labels"] = []
            winner["needs_review"] = True
            winner["training_eligible"] = False
            winner["decision_status"] = "needs_review"
            winner["consolidation_warning"] = "conflicting_top_priority_decisions"
        winner["consolidated_sources"] = [row.get("label_source") for row in rows]
        selected.append(winner)
    selected.sort(key=lambda row: row["chunk_id"])
    destination_path = Path(destination)
    if destination_path.is_file() and canonical_json_sha256(list(read_jsonl(destination_path))) == canonical_json_sha256(selected):
        status = "noop"
    else:
        write_jsonl_atomic(destination_path, selected)
        status = "updated"
    return {"status": status, "chunks": len(selected), "conflicts": conflicts}


def _latest_review_events(sources: Iterable[str | Path]) -> tuple[dict[str, ReviewEvent], int]:
    latest: dict[str, ReviewEvent] = {}
    duplicates = 0
    seen_event_ids: set[str] = set()
    for source in sources:
        for raw in read_jsonl(source):
            event = ReviewEvent.model_validate(raw)
            if event.event_id in seen_event_ids:
                duplicates += 1
                continue
            seen_event_ids.add(event.event_id)
            previous = latest.get(event.chunk_id)
            current_key = (event.created_at, event.event_id)
            previous_key = (previous.created_at, previous.event_id) if previous else None
            if previous_key is None or current_key > previous_key:
                latest[event.chunk_id] = event
    return latest, duplicates


def reconcile_human_reviews(
    consolidated_source: str | Path,
    review_sources: Iterable[str | Path],
    destination: str | Path,
    *,
    chunks_source: str | Path | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aplica el último evento humano por chunk sin modificar eventos ni propuestas.

    La salida es una vista derivada. Si las entradas y el contrato no cambian, no se
    vuelve a escribir el archivo. Los chunks permiten recuperar ``video_id`` de
    anotaciones antiguas; nunca se deduce a partir de ``chunk_id``.
    """

    source = Path(consolidated_source)
    reviews = [Path(path) for path in review_sources if Path(path).is_file()]
    destination_path = Path(destination)
    state = Path(state_path) if state_path else destination_path.with_suffix(".state.json")
    taxonomy = load_taxonomy()
    signature_inputs = [source, *reviews]
    if chunks_source:
        signature_inputs.append(Path(chunks_source))
    signature = input_signature(
        signature_inputs,
        {
            "operation": "reconcile_human_reviews_v1",
            "taxonomy": taxonomy.contract_id,
            "version": taxonomy.version,
        },
    )
    if state.is_file() and destination_path.is_file():
        import json

        previous = json.loads(state.read_text(encoding="utf-8"))
        if previous.get("input_signature") == signature:
            return {**previous.get("counters", {}), "status": "noop", "input_signature": signature}

    chunk_lookup = {
        str(row["chunk_id"]): row
        for row in (read_jsonl(chunks_source) if chunks_source and Path(chunks_source).is_file() else [])
    }
    events, duplicate_events = _latest_review_events(reviews)
    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)
    base_ids: set[str] = set()
    for raw in read_jsonl(source):
        row = dict(raw)
        chunk_id = str(row["chunk_id"])
        base_ids.add(chunk_id)
        chunk = chunk_lookup.get(chunk_id, {})
        row["video_id"] = row.get("video_id") or chunk.get("video_id")
        row["text"] = row.get("text") or chunk.get("text")
        if not row.get("video_id"):
            counters["missing_video_id"] += 1
        event = events.get(chunk_id)
        if event is not None:
            base_hash = canonical_json_sha256(row)
            row.update(
                {
                    "source_record_sha256": base_hash,
                    "review_event_id": event.event_id,
                    "review_action": event.action,
                    "reviewer_pseudonym": event.reviewer,
                    "annotator_type": "human",
                    "annotator_model": None,
                    "created_at": event.created_at,
                    "notes": event.notes,
                    "flags": list(event.flags),
                }
            )
            if event.action in {"accept", "modify"}:
                final_labels = taxonomy.normalize_categories(event.final_labels)
                row["coarse_labels"] = list(final_labels)
                # Solo se conservan finas compatibles con la decisión humana.
                row["fine_labels"] = [
                    fine
                    for fine in row.get("fine_labels", [])
                    if taxonomy.fine_label_mapping.get(fine) in final_labels
                ]
                row["needs_review"] = False
                row["training_eligible"] = True
                row["decision_status"] = "resolved"
                row["label_source"] = "human_accepted" if event.action == "accept" else "human_modified"
                counters[f"human:{event.action}"] += 1
            elif event.action == "defer":
                row["coarse_labels"] = []
                row["fine_labels"] = []
                row["needs_review"] = True
                row["training_eligible"] = False
                row["decision_status"] = "needs_review"
                row["label_source"] = "human_deferred"
                counters["human:defer"] += 1
            else:
                row["coarse_labels"] = []
                row["fine_labels"] = []
                row["needs_review"] = False
                row["training_eligible"] = False
                row["decision_status"] = "excluded"
                row["label_source"] = "human_rejected"
                counters["human:reject"] += 1
        else:
            counters["without_human_event"] += 1
        validated = AnnotationRecord.model_validate(row)
        rows.append(validated.model_dump(mode="json"))

    counters["rows"] = len(rows)
    counters["duplicate_events"] = duplicate_events
    counters["orphan_events"] = len(set(events) - base_ids)
    rows.sort(key=lambda item: item["chunk_id"])
    write_jsonl_atomic(destination_path, rows)
    payload = {
        "schema_version": "2.1.0",
        "operation": "reconcile_human_reviews",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_signature": signature,
        "sources": [str(path) for path in signature_inputs],
        "destination": str(destination_path),
        "counters": dict(counters),
    }
    write_json_atomic(state, payload)
    return {**dict(counters), "status": "updated", "input_signature": signature}
