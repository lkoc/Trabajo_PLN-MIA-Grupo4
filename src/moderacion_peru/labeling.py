from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import (
    append_jsonl_once,
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from .incremental import normalize_text
from .providers.base import AnnotationProvider, ProviderError
from .schemas import AnnotationRecord
from .taxonomy import load_taxonomy


HISTORICAL_RECOVERY_MAPPING = "exact_unique_video_and_normalized_text_v1"


def _historical_text_key(row: dict[str, Any]) -> tuple[str, str] | None:
    video_id = str(row.get("video_id") or "")
    text = normalize_text(str(row.get("text") or ""))
    if not video_id or not text:
        return None
    return video_id, sha256_text(text.casefold())


def historical_recovery_signature(
    historical_chunks_path: str | Path,
    historical_annotation_paths: Sequence[str | Path],
    *,
    expected_model: str,
    historical_prompt_sha256: str,
) -> dict[str, Any]:
    """Firma las fuentes antiguas que pueden sembrar una corrida incremental."""

    chunks = Path(historical_chunks_path)
    annotations = [Path(path) for path in historical_annotation_paths]
    if not chunks.is_file():
        raise FileNotFoundError(f"Faltan los chunks históricos: {chunks}")
    if not annotations:
        raise ValueError("La recuperación histórica requiere al menos una fuente de anotaciones")
    missing = [str(path) for path in annotations if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan fuentes históricas: {missing}")
    if len(historical_prompt_sha256) != 64:
        raise ValueError("historical_prompt_sha256 debe ser una huella SHA-256")
    return {
        "mapping": HISTORICAL_RECOVERY_MAPPING,
        "expected_model": expected_model,
        "historical_prompt_sha256": historical_prompt_sha256,
        "historical_chunks": {
            "name": chunks.name,
            "sha256": sha256_file(chunks),
        },
        "annotations": [
            {"name": path.name, "sha256": sha256_file(path)} for path in annotations
        ],
    }


def recover_historical_annotations(
    current_records: Sequence[dict[str, Any]],
    historical_chunks_path: str | Path,
    historical_annotation_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    expected_model: str,
    historical_prompt_sha256: str,
    run_metadata: dict[str, Any],
    label_source: str = "deepseek_remote_historical_recovered",
) -> dict[str, Any]:
    """Recupera solo equivalencias textuales exactas sin sobrescribir progreso nuevo.

    Los IDs históricos eran secuenciales y no coinciden con los IDs content-addressed
    actuales. La transferencia se limita a relaciones 1:1 por video y texto
    normalizado; toda ambigüedad o segmentación distinta permanece pendiente.
    """

    output = Path(output_path)
    signature = historical_recovery_signature(
        historical_chunks_path,
        historical_annotation_paths,
        expected_model=expected_model,
        historical_prompt_sha256=historical_prompt_sha256,
    )
    manifest_path = _ensure_labeling_run_manifest(output, run_metadata)
    existing_rows, quarantine_path = _load_and_quarantine_progress(
        output,
        quarantine_invalid=True,
    )
    existing_by_id = {str(row["chunk_id"]): row for row in existing_rows}

    historical_by_id: dict[str, tuple[dict[str, Any], str]] = {}
    for raw_path in historical_annotation_paths:
        annotation_path = Path(raw_path)
        for row in read_jsonl(annotation_path):
            chunk_id = str(row.get("chunk_id") or "")
            if not chunk_id:
                raise ValueError(f"{annotation_path} contiene una fila sin chunk_id")
            if str(row.get("annotator_model") or "") != expected_model:
                raise ValueError(
                    f"{annotation_path} mezcla un modelo distinto de {expected_model}: {chunk_id}"
                )
            if chunk_id in historical_by_id:
                raise ValueError(f"chunk_id histórico duplicado entre fuentes: {chunk_id}")
            historical_by_id[chunk_id] = (row, annotation_path.name)

    historical_keys: dict[tuple[str, str], list[str]] = {}
    historical_chunks_by_id: dict[str, dict[str, Any]] = {}
    historical_chunks_seen: set[str] = set()
    for chunk in read_jsonl(historical_chunks_path):
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id not in historical_by_id:
            continue
        if chunk_id in historical_chunks_seen:
            raise ValueError(f"chunk_id duplicado en chunks históricos: {chunk_id}")
        historical_chunks_seen.add(chunk_id)
        historical_chunks_by_id[chunk_id] = chunk
        key = _historical_text_key(chunk)
        if key is not None:
            historical_keys.setdefault(key, []).append(chunk_id)

    missing_historical_chunks = set(historical_by_id) - historical_chunks_seen
    if missing_historical_chunks:
        example = sorted(missing_historical_chunks)[:5]
        raise ValueError(
            "Las anotaciones históricas no pertenecen al archivo de chunks declarado; "
            f"faltan {len(missing_historical_chunks)} IDs, por ejemplo {example}"
        )

    current_keys: dict[tuple[str, str], list[dict[str, Any]]] = {}
    current_by_id: dict[str, dict[str, Any]] = {}
    current_ids: set[str] = set()
    for record in current_records:
        chunk_id = str(record.get("chunk_id") or "")
        if not chunk_id:
            raise ValueError("El corpus actual contiene una fila sin chunk_id")
        if chunk_id in current_ids:
            raise ValueError(f"chunk_id duplicado en el corpus actual: {chunk_id}")
        current_ids.add(chunk_id)
        current_by_id[chunk_id] = record
        key = _historical_text_key(record)
        if key is not None:
            current_keys.setdefault(key, []).append(record)

    taxonomy = load_taxonomy()
    recovered: list[dict[str, Any]] = []
    matched_historical_ids: set[str] = set()
    ambiguous_keys = 0
    already_present_matches = 0

    def recover_match(
        historical_id: str,
        current: dict[str, Any],
        *,
        warning: str,
    ) -> bool:
        current_id = str(current["chunk_id"])
        matched_historical_ids.add(historical_id)
        if current_id in existing_by_id:
            return True
        historical, source_name = historical_by_id[historical_id]
        fine_labels = list(
            dict.fromkeys(
                historical.get("fine_labels") or historical.get("labels") or []
            )
        )
        coarse_labels = list(taxonomy.derive_categories(fine_labels))
        needs_review = bool(historical.get("needs_review")) or not coarse_labels
        source_record_sha256 = str(current.get("text_sha256") or "") or sha256_text(
            str(current["text"])
        )
        record = AnnotationRecord(
            chunk_id=current_id,
            video_id=str(current.get("video_id") or "") or None,
            start_seconds=current.get("start_seconds"),
            end_seconds=current.get("end_seconds"),
            video_title=current.get("video_title") or current.get("title"),
            channel_title=current.get("channel_title") or current.get("channel"),
            source_url=current.get("source_url") or current.get("url"),
            cohort=current.get("cohort"),
            text=str(current["text"]),
            coarse_labels=coarse_labels,
            fine_labels=fine_labels,
            flags=list(dict.fromkeys(historical.get("flags") or [])),
            needs_review=needs_review,
            training_eligible=not needs_review,
            decision_status="needs_review" if needs_review else "resolved",
            score_confianza=historical.get("score_confianza"),
            notes=str(historical.get("notes") or ""),
            justification=str(
                historical.get("justification")
                or historical.get("justificacion")
                or ""
            ),
            label_source=label_source,
            annotator_type="llm_remote",
            annotator_model=expected_model,
            prompt_sha256=str(
                historical.get("prompt_sha256") or historical_prompt_sha256
            ),
            source_record_sha256=source_record_sha256,
            consolidated_sources=list(
                dict.fromkeys(
                    [
                        *(historical.get("consolidated_sources") or []),
                        f"{source_name}:{historical_id}",
                    ]
                )
            ),
            consolidation_warning=warning,
            created_at=(
                historical.get("created_at")
                or historical.get("annotated_at")
                or datetime.now(timezone.utc)
            ),
        )
        recovered.append(record.model_dump(mode="json"))
        return False

    direct_id_matches = 0
    for historical_id in sorted(set(historical_by_id).intersection(current_by_id)):
        historical_key = _historical_text_key(historical_chunks_by_id[historical_id])
        current = current_by_id[historical_id]
        if historical_key is None or historical_key != _historical_text_key(current):
            continue
        direct_id_matches += 1
        already_present_matches += int(
            recover_match(
                historical_id,
                current,
                warning="historical_annotation_reused_by_identical_chunk_id_and_text",
            )
        )

    for key in sorted(set(historical_keys).intersection(current_keys)):
        old_ids = [
            historical_id
            for historical_id in historical_keys[key]
            if historical_id not in matched_historical_ids
        ]
        new_rows = [
            current
            for current in current_keys[key]
            if str(current["chunk_id"]) not in matched_historical_ids
        ]
        if not old_ids and not new_rows:
            continue
        if len(old_ids) != 1 or len(new_rows) != 1:
            ambiguous_keys += 1
            continue
        historical_id = old_ids[0]
        current = new_rows[0]
        already_present_matches += int(
            recover_match(
                historical_id,
                current,
                warning="historical_id_rekeyed_by_exact_normalized_text",
            )
        )

    if recovered:
        write_jsonl_atomic(output, [*existing_rows, *recovered])
    completed_current = (set(existing_by_id) | {str(row["chunk_id"]) for row in recovered}) & current_ids
    report = {
        "schema_version": "1.0.0",
        "operation": "recover_historical_annotations",
        "signature": signature,
        "current_rows": len(current_ids),
        "historical_rows": len(historical_by_id),
        "exact_unique_matches": len(matched_historical_ids),
        "direct_chunk_id_matches": direct_id_matches,
        "recovered_new": len(recovered),
        "already_present_matches": already_present_matches,
        "ambiguous_keys_excluded": ambiguous_keys,
        "historical_not_reusable": len(historical_by_id) - len(matched_historical_ids),
        "completed_current_after_recovery": len(completed_current),
        "pending_current_after_recovery": len(current_ids) - len(completed_current),
        "output_rows": len(existing_rows) + len(recovered),
        "run_manifest": str(manifest_path) if manifest_path else None,
        "quarantined_progress": str(quarantine_path) if quarantine_path else None,
        "safety_rule": "only exact, unique (video_id, normalized_text) matches are reused",
    }
    write_json_atomic(output.with_suffix(output.suffix + ".recovery.json"), report)
    return report


def annotate_incremental(
    records: Iterable[dict[str, Any]],
    provider: AnnotationProvider,
    output_path: str | Path,
    *,
    error_path: str | Path | None = None,
    limit: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit debe ser None o un entero positivo")
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
    if progress_callback is not None:
        progress_callback({"status": "started", "advance": 0, **counters})
    for record in pending:
        try:
            annotation = provider.annotate(record)
            append_jsonl_once(output, [annotation.model_dump(mode="json")], id_field="chunk_id")
            counters["labeled"] += 1
            status = "labeled"
        except (ProviderError, ValueError, RuntimeError) as exc:
            counters["errors"] += 1
            status = "error"
            if error_path:
                append_jsonl_once(
                    error_path,
                    [{"chunk_id": record.get("chunk_id"), "error": str(exc)}],
                    id_field="chunk_id",
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "status": status,
                    "advance": 1,
                    "chunk_id": record.get("chunk_id"),
                    **counters,
                }
            )
    if progress_callback is not None:
        progress_callback({"status": "finished", "advance": 0, **counters})
    return counters


def load_pending_chunks(source: str | Path, output: str | Path) -> list[dict[str, Any]]:
    completed = {row["chunk_id"] for row in read_jsonl(output)} if Path(output).exists() else set()
    return [row for row in read_jsonl(source) if row.get("chunk_id") not in completed]


def _ensure_labeling_run_manifest(
    output_path: Path, run_metadata: dict[str, Any] | None
) -> Path | None:
    if run_metadata is None:
        return None
    manifest_path = output_path.with_suffix(output_path.suffix + ".run.json")
    signature = canonical_json_sha256(run_metadata)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if existing.get("run_signature") != signature:
            existing_metadata = {
                key: value
                for key, value in existing.items()
                if key
                not in {
                    "schema_version",
                    "run_signature",
                    "compatible_predecessor_run_signature",
                    "manifest_upgraded_at",
                }
            }

            def compatibility_projection(metadata: dict[str, Any]) -> dict[str, Any]:
                projected = json.loads(json.dumps(metadata, ensure_ascii=False, default=str))
                provider = projected.get("provider")
                if isinstance(provider, dict):
                    thinking = provider.get("thinking")
                    if thinking in (None, {"type": "disabled"}):
                        provider.pop("thinking", None)
                    provider.pop("historical_recovery", None)
                return projected

            compatible = (
                compatibility_projection(existing_metadata)
                == compatibility_projection(run_metadata)
            )
            provider_metadata = run_metadata.get("provider")
            safe_thinking = not isinstance(provider_metadata, dict) or provider_metadata.get(
                "thinking", {"type": "disabled"}
            ) == {"type": "disabled"}
            if not compatible or not safe_thinking:
                raise ValueError(
                    "La salida existente pertenece a otro modelo, prompt o configuración; "
                    "use un archivo nuevo para no mezclar campañas."
                )
            write_json_atomic(
                manifest_path,
                {
                    "schema_version": "1.0.0",
                    "run_signature": signature,
                    "compatible_predecessor_run_signature": existing.get("run_signature"),
                    "manifest_upgraded_at": datetime.now(timezone.utc).isoformat(),
                    **run_metadata,
                },
            )
    elif output_path.exists() and output_path.stat().st_size:
        raise ValueError(
            f"{output_path} ya contiene anotaciones pero no tiene manifiesto de corrida"
        )
    else:
        write_json_atomic(
            manifest_path,
            {"schema_version": "1.0.0", "run_signature": signature, **run_metadata},
        )
    return manifest_path


def _load_and_quarantine_progress(
    output: Path,
    *,
    quarantine_invalid: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    if not output.exists():
        return [], None
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    if progress_callback is not None:
        progress_callback(
            {"status": "phase_started", "phase": "existing_progress", "total": None}
        )
    with output.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                AnnotationRecord.model_validate(row)
                chunk_id = str(row["chunk_id"])
                if chunk_id in seen:
                    raise ValueError(f"chunk_id duplicado: {chunk_id}")
                seen.add(chunk_id)
                valid.append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                invalid.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "raw_line": line.rstrip("\r\n"),
                    }
                )
            scanned += 1
            if progress_callback is not None and scanned % 1000 == 0:
                progress_callback(
                    {
                        "status": "phase_progress",
                        "phase": "existing_progress",
                        "phase_advance": 1000,
                        "scanned": scanned,
                    }
                )
    if progress_callback is not None:
        if scanned % 1000:
            progress_callback(
                {
                    "status": "phase_progress",
                    "phase": "existing_progress",
                    "phase_advance": scanned % 1000,
                    "scanned": scanned,
                }
            )
        progress_callback(
            {
                "status": "phase_finished",
                "phase": "existing_progress",
                "scanned": scanned,
            }
        )
    if not invalid:
        return valid, None
    if not quarantine_invalid:
        raise ValueError(
            f"{output} contiene {len(invalid)} filas incompatibles; active quarantine_invalid_progress"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output.with_suffix(output.suffix + f".quarantine-{timestamp}.jsonl")
    write_jsonl_atomic(quarantine, invalid)
    write_jsonl_atomic(output, valid)
    return valid, quarantine


def annotate_batched_incremental(
    records: Iterable[dict[str, Any]],
    provider: AnnotationProvider,
    output_path: str | Path,
    *,
    error_path: str | Path | None = None,
    limit: int | None = None,
    processing_batch_size: int = 20,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_every_batches: int = 50,
    run_metadata: dict[str, Any] | None = None,
    quarantine_invalid_progress: bool = True,
) -> dict[str, Any]:
    """Etiqueta en lotes GPU/API con reanudación O(n) y checkpoints periódicos.

    A diferencia del camino unitario, abre cada JSONL una sola vez y fuerza a
    disco después de cada lote. Esto evita releer una salida creciente por cada
    chunk y permite procesar corpus grandes de forma razonable.
    """

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit debe ser None o un entero positivo")
    if processing_batch_size < 1 or checkpoint_every_batches < 1:
        raise ValueError("Los tamaños de lote y checkpoint deben ser positivos")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = Path(error_path) if error_path else None
    if errors is not None:
        errors.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = _ensure_labeling_run_manifest(output, run_metadata)

    existing_rows, quarantine_path = _load_and_quarantine_progress(
        output,
        quarantine_invalid=quarantine_invalid_progress,
        progress_callback=progress_callback,
    )
    completed_rows = {str(row["chunk_id"]): row for row in existing_rows}
    completed = set(completed_rows)
    error_ids = (
        {str(row["chunk_id"]) for row in read_jsonl(errors)}
        if errors is not None and errors.exists()
        else set()
    )
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    record_total = (
        len(records) if limit is None and hasattr(records, "__len__") else None
    )
    if progress_callback is not None:
        progress_callback(
            {"status": "phase_started", "phase": "pending_scan", "total": record_total}
        )
    scanned_records = 0
    for record in records:
        scanned_records += 1
        if progress_callback is not None and scanned_records % 1000 == 0:
            progress_callback(
                {
                    "status": "phase_progress",
                    "phase": "pending_scan",
                    "phase_advance": 1000,
                    "scanned": scanned_records,
                }
            )
        chunk_id = str(record.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        if chunk_id in completed:
            current_hash = str(
                record.get("text_sha256") or record.get("transcript_sha256") or ""
            )
            saved_hash = str(completed_rows[chunk_id].get("source_record_sha256") or "")
            if current_hash and saved_hash and current_hash != saved_hash:
                raise ValueError(
                    f"El texto de {chunk_id} cambió desde la anotación guardada; use una campaña nueva"
                )
            continue
        pending.append(record)
        if limit is not None and len(pending) >= limit:
            break
    if progress_callback is not None:
        if scanned_records % 1000:
            progress_callback(
                {
                    "status": "phase_progress",
                    "phase": "pending_scan",
                    "phase_advance": scanned_records % 1000,
                    "scanned": scanned_records,
                }
            )
        progress_callback(
            {
                "status": "phase_finished",
                "phase": "pending_scan",
                "scanned": scanned_records,
            }
        )

    counters: dict[str, Any] = {
        "already_completed": len(completed),
        "selected": len(pending),
        "labeled": 0,
        "errors": 0,
        "batches": 0,
        "request_groups": 0,
    }
    if progress_callback is not None:
        progress_callback({"status": "started", "advance": 0, **counters})
    if not pending:
        if progress_callback is not None:
            progress_callback({"status": "finished", "advance": 0, **counters})
        return {
            **counters,
            "elapsed_seconds": 0.0,
            "chunks_per_minute": None,
            "run_manifest": str(manifest_path) if manifest_path else None,
            "quarantined_progress": str(quarantine_path) if quarantine_path else None,
        }

    started = time.perf_counter()
    error_mode = "a" if errors is not None else None
    try:
        with output.open("a", encoding="utf-8", newline="\n") as output_handle:
            error_handle = (
                errors.open(error_mode, encoding="utf-8", newline="\n") if errors else None
            )
            try:
                def persist_results(
                    group_records: list[dict[str, Any]],
                    group_results: list[AnnotationRecord | Exception],
                    *,
                    status: str,
                ) -> dict[str, Any]:
                    if len(group_results) != len(group_records):
                        raise RuntimeError(
                            f"El proveedor devolvió {len(group_results)} resultados "
                            f"para {len(group_records)} entradas"
                        )
                    group_labeled = group_errors = 0
                    for record, result in zip(group_records, group_results):
                        chunk_id = str(record["chunk_id"])
                        if isinstance(result, Exception):
                            counters["errors"] += 1
                            group_errors += 1
                            if error_handle is not None and chunk_id not in error_ids:
                                error_handle.write(
                                    json.dumps(
                                        {"chunk_id": chunk_id, "error": str(result)},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                error_ids.add(chunk_id)
                            continue
                        if result.chunk_id != chunk_id:
                            raise RuntimeError(
                                f"El proveedor cambió chunk_id: esperado={chunk_id}, "
                                f"recibido={result.chunk_id}"
                            )
                        output_handle.write(
                            json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n"
                        )
                        completed.add(chunk_id)
                        counters["labeled"] += 1
                        group_labeled += 1
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                    if error_handle is not None:
                        error_handle.flush()
                        os.fsync(error_handle.fileno())
                    counters["request_groups"] += 1
                    elapsed = time.perf_counter() - started
                    event = {
                        "status": status,
                        "advance": len(group_records),
                        "batch_labeled": group_labeled,
                        "batch_errors": group_errors,
                        "elapsed_seconds": elapsed,
                        "chunks_per_minute": 60
                        * (counters["labeled"] + counters["errors"])
                        / elapsed,
                        **counters,
                    }
                    usage_summary = getattr(provider, "usage_summary", None)
                    if callable(usage_summary):
                        event["provider_usage"] = usage_summary()
                    if progress_callback is not None:
                        progress_callback(event)
                    return event

                for start in range(0, len(pending), processing_batch_size):
                    batch = pending[start : start + processing_batch_size]
                    last_event: dict[str, Any] | None = None
                    incremental_iterator = getattr(provider, "iter_annotate_batch", None)
                    if callable(incremental_iterator):
                        observed_positions: set[int] = set()
                        for completed_group in incremental_iterator(batch):
                            positions = [int(index) for index, _ in completed_group]
                            if (
                                any(index < 0 or index >= len(batch) for index in positions)
                                or observed_positions.intersection(positions)
                            ):
                                raise RuntimeError("El proveedor devolvió índices de grupo inválidos")
                            observed_positions.update(positions)
                            group_records = [batch[index] for index in positions]
                            group_results = [result for _, result in completed_group]
                            last_event = persist_results(
                                group_records,
                                group_results,
                                status="request_group_finished",
                            )
                        if observed_positions != set(range(len(batch))):
                            raise RuntimeError("El proveedor no completó todos los índices del lote")
                    else:
                        results = provider.annotate_batch(batch)
                        last_event = persist_results(
                            batch,
                            results,
                            status="batch_finished",
                        )
                    counters["batches"] += 1
                    if (
                        checkpoint_callback is not None
                        and counters["batches"] % checkpoint_every_batches == 0
                    ):
                        checkpoint_callback(
                            {
                                **(last_event or {}),
                                "status": "periodic_checkpoint",
                                "advance": 0,
                                **counters,
                            }
                        )
            finally:
                if error_handle is not None:
                    error_handle.close()
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        interrupted_result = {
            **counters,
            "status": "interrupted_checkpoint",
            "advance": 0,
            "interrupted": True,
            "elapsed_seconds": round(elapsed, 3),
            "chunks_per_minute": round(
                60 * (counters["labeled"] + counters["errors"]) / elapsed,
                3,
            ),
            "run_manifest": str(manifest_path) if manifest_path else None,
            "quarantined_progress": str(quarantine_path) if quarantine_path else None,
        }
        usage_summary = getattr(provider, "usage_summary", None)
        if callable(usage_summary):
            interrupted_result["provider_usage"] = usage_summary()
        if checkpoint_callback is not None:
            checkpoint_callback(interrupted_result)
        if progress_callback is not None:
            progress_callback(interrupted_result)
        raise

    elapsed = time.perf_counter() - started
    result = {
        **counters,
        "elapsed_seconds": round(elapsed, 3),
        "chunks_per_minute": round(
            60 * (counters["labeled"] + counters["errors"]) / elapsed, 3
        ),
        "run_manifest": str(manifest_path) if manifest_path else None,
        "quarantined_progress": str(quarantine_path) if quarantine_path else None,
    }
    usage_summary = getattr(provider, "usage_summary", None)
    if callable(usage_summary):
        result["provider_usage"] = usage_summary()
    if checkpoint_callback is not None:
        checkpoint_callback({"status": "final_checkpoint", "advance": 0, **result})
    if progress_callback is not None:
        progress_callback({"status": "finished", "advance": 0, **result})
    return result

