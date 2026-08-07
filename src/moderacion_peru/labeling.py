from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import (
    append_jsonl_once,
    canonical_json_sha256,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from .providers.base import AnnotationProvider, ProviderError
from .schemas import AnnotationRecord


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
            raise ValueError(
                "La salida existente pertenece a otro modelo, prompt o configuración; "
                "use un archivo nuevo para no mezclar campañas."
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
    with output.open("a", encoding="utf-8", newline="\n") as output_handle:
        error_handle = errors.open(error_mode, encoding="utf-8", newline="\n") if errors else None
        try:
            for start in range(0, len(pending), processing_batch_size):
                batch = pending[start : start + processing_batch_size]
                results = provider.annotate_batch(batch)
                if len(results) != len(batch):
                    raise RuntimeError(
                        f"El proveedor devolvió {len(results)} resultados para {len(batch)} entradas"
                    )
                batch_labeled = batch_errors = 0
                for record, result in zip(batch, results):
                    chunk_id = str(record["chunk_id"])
                    if isinstance(result, Exception):
                        counters["errors"] += 1
                        batch_errors += 1
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
                            f"El proveedor cambió chunk_id: esperado={chunk_id}, recibido={result.chunk_id}"
                        )
                    output_handle.write(
                        json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n"
                    )
                    completed.add(chunk_id)
                    counters["labeled"] += 1
                    batch_labeled += 1
                output_handle.flush()
                os.fsync(output_handle.fileno())
                if error_handle is not None:
                    error_handle.flush()
                    os.fsync(error_handle.fileno())
                counters["batches"] += 1
                elapsed = time.perf_counter() - started
                event = {
                    "status": "batch_finished",
                    "advance": len(batch),
                    "batch_labeled": batch_labeled,
                    "batch_errors": batch_errors,
                    "elapsed_seconds": elapsed,
                    "chunks_per_minute": 60 * (counters["labeled"] + counters["errors"]) / elapsed,
                    **counters,
                }
                usage_summary = getattr(provider, "usage_summary", None)
                if callable(usage_summary):
                    event["provider_usage"] = usage_summary()
                if progress_callback is not None:
                    progress_callback(event)
                if (
                    checkpoint_callback is not None
                    and counters["batches"] % checkpoint_every_batches == 0
                ):
                    checkpoint_callback(event)
        finally:
            if error_handle is not None:
                error_handle.close()

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

