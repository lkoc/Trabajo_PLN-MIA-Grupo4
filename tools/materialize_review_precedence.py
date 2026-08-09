"""Materializa la prevalencia Pro tras una revisión superior ya concluida.

``needs_review`` es una abstención intermedia del modelo Pro, no una categoría
final. Este comando registra un evento ``accept`` para cada propuesta Pro no
vacía que el protocolo CODEX–Sol-EH decidió conservar y que todavía no tenía un
evento superior. Así, la reconciliación puede incluirla en train/validation/test
sin borrar la propuesta ni reescribir la campaña consolidada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from moderacion_peru.io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from moderacion_peru.paths import find_project_root
from moderacion_peru.schemas import ReviewEvent

ROOT = find_project_root()
CAMPAIGN = ROOT / "datos/etiquetado/consolidado/anotaciones_v2.jsonl"
REVIEWS = ROOT / "datos/etiquetado/humano/labeling_events_v2.jsonl"
AUDIT_SAMPLE = ROOT / "docs/artefactos/auditoria_16k_flash_pro_sol_eh_sample.csv"
EVENT_SNAPSHOT = (
    ROOT / "datos/etiquetado/humano" / "codex_pro_precedence_reviewed.events.jsonl"
)
MANIFEST = EVENT_SNAPSHOT.with_suffix(".manifest.json")
BATCH_ID = "CODEX-PRO-PRECEDENCE-20260809"


def _latest_events(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in rows:
        chunk_id = str(event["chunk_id"])
        previous = latest.get(chunk_id)
        current_key = (str(event.get("created_at") or ""), str(event["event_id"]))
        previous_key = (
            (str(previous.get("created_at") or ""), str(previous["event_id"]))
            if previous
            else None
        )
        if previous_key is None or current_key > previous_key:
            latest[chunk_id] = event
    return latest


def build_precedence_events() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviews = list(read_jsonl(REVIEWS)) if REVIEWS.is_file() else []
    latest = _latest_events(reviews)
    created_at = datetime.now(timezone.utc)
    events: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()

    for row in read_jsonl(CAMPAIGN):
        chunk_id = str(row["chunk_id"])
        if chunk_id in latest:
            continue
        is_intermediate_review = (
            bool(row.get("needs_review"))
            or str(row.get("decision_status") or "") == "needs_review"
        )
        labels = list(row.get("coarse_labels") or [])
        model = str(row.get("annotator_model") or "")
        if not is_intermediate_review or not labels or "pro" not in model.casefold():
            continue

        digest = hashlib.sha256(f"{BATCH_ID}|{chunk_id}".encode()).hexdigest()[:24]
        notes = (
            "Materialización de la regla human-in-the-loop confirmada: CODEX–Sol-EH "
            "no modificó la propuesta DeepSeek Pro, por lo que prevalece Pro. "
            "needs_review se interpreta como estado intermedio y no como etiqueta final."
        )
        event = ReviewEvent(
            event_id=f"codex-pro-accept-{digest}",
            chunk_id=chunk_id,
            action="accept",
            proposed_labels=labels,
            final_labels=labels,
            flags=([] if labels == ["SEGURO"] else list(row.get("flags") or [])),
            reviewer="CODEX",
            model_id="gpt-5.6-sol",
            decision_scope="chunk",
            decision_scope_key=f"chunk:{chunk_id}",
            batch_id=BATCH_ID,
            notes=notes,
            created_at=created_at,
        ).model_dump(mode="json")
        events.append(event)
        label_counts.update(event["final_labels"])
        source_counts[str(row.get("label_source") or "sin_fuente")] += 1
        confidence_counts[str(row.get("score_confianza"))] += 1

    events.sort(key=lambda event: str(event["chunk_id"]))
    statistics = {
        "campaign_rows": sum(1 for _ in read_jsonl(CAMPAIGN)),
        "events_before": len(reviews),
        "latest_reviewed_chunks_before": len(latest),
        "precedence_events": len(events),
        "label_assignments": dict(sorted(label_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
    }
    return events, statistics


def finalize(*, apply: bool) -> dict[str, Any]:
    reviews_hash_before = sha256_file(REVIEWS) if REVIEWS.is_file() else None
    events, statistics = build_precedence_events()
    write_jsonl_atomic(EVENT_SNAPSHOT, events)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "applied" if apply else "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "reviewer": "CODEX",
        "adjudicator": "CODEX–Sol-EH",
        "reasoning_effort": "extra high",
        "rule": "a falta de cambio superior, prevalece la propuesta no vacía de Pro",
        "campaign": str(CAMPAIGN.relative_to(ROOT)).replace("\\", "/"),
        "campaign_sha256": sha256_file(CAMPAIGN),
        "reviews": str(REVIEWS.relative_to(ROOT)).replace("\\", "/"),
        "reviews_sha256_before": reviews_hash_before,
        "event_snapshot": str(EVENT_SNAPSHOT.relative_to(ROOT)).replace("\\", "/"),
        "event_snapshot_sha256": sha256_file(EVENT_SNAPSHOT),
        "audit_sample_available": AUDIT_SAMPLE.is_file(),
        "statistics": statistics,
    }
    if apply:
        added, skipped = append_jsonl_once(REVIEWS, events, id_field="event_id")
        manifest["append"] = {"added": added, "skipped": skipped}
        manifest["reviews_sha256_after"] = sha256_file(REVIEWS)
    write_json_atomic(MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(finalize(apply=arguments.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
