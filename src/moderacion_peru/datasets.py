from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import (
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .schemas import ModelReadyRecord, ReviewEvent
from .taxonomy import load_taxonomy

ProgressCallback = Callable[[dict[str, Any]], None]


def _notify_progress(
    callback: ProgressCallback | None,
    *,
    status: str,
    phase: str,
    advance: int = 0,
    total: int | None = None,
    **details: Any,
) -> None:
    if callback is not None:
        callback(
            {
                "status": status,
                "phase": phase,
                "advance": advance,
                "total": total,
                **details,
            }
        )


def stable_video_split(video_id: str, seed: int = 20260805) -> str:
    value = (
        int(hashlib.sha256(f"{seed}|{video_id}".encode()).hexdigest()[:16], 16) / 16**16
    )
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def stable_channel_split(channel_key: str, seed: int = 20260805) -> str:
    """Partición alternativa sin solapamiento de canal para pruebas de robustez."""

    return stable_video_split(f"channel|{channel_key}", seed)


def _ordered_binary_mask(
    value: object,
    labels: Sequence[str],
) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        mask = [int(bool(value.get(label, 0))) for label in labels]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        mask = [int(item) for item in value]
    else:
        raise TypeError(
            "Una máscara observada debe ser una lista o un mapa por etiqueta"
        )
    if len(mask) != len(labels) or any(item not in {0, 1} for item in mask):
        raise ValueError(
            f"La máscara observada debe tener {len(labels)} posiciones binarias"
        )
    return mask


def derive_supervision_masks(
    row: Mapping[str, Any],
) -> tuple[list[int], list[int], list[int]]:
    """Deriva máscaras conservadoras sin convertir una fina ausente en negativa.

    Las categorías gruesas están resueltas en todo registro entrenable. Una fina
    hija de una categoría gruesa inactiva sí es un negativo lógicamente observado.
    Para una categoría activa, sus hijas solo se consideran exhaustivamente
    observadas cuando al menos una fina de esa categoría está presente. Los flags
    son exhaustivos cuando proceden de un prompt versionado o de una decisión
    humana; en fuentes no versionadas solo se observan los positivos explícitos.
    """

    taxonomy = load_taxonomy()
    explicit_coarse = _ordered_binary_mask(
        row.get("coarse_observed_mask"), taxonomy.target_labels
    )
    explicit_fine = _ordered_binary_mask(
        row.get("fine_observed_mask"), taxonomy.fine_labels
    )
    explicit_flags = _ordered_binary_mask(
        row.get("flags_observed_mask"), taxonomy.flags
    )
    coarse_mask = explicit_coarse or [1] * len(taxonomy.target_labels)
    selected_fine = set(taxonomy.normalize_fine_labels(row.get("fine_labels", ())))
    active_coarse = set(taxonomy.normalize_categories(row.get("coarse_labels", ())))
    if explicit_fine is None:
        observed_active_parents = {
            taxonomy.fine_label_mapping[label] for label in selected_fine
        }
        fine_mask = [
            int(
                taxonomy.fine_label_mapping[label] not in active_coarse
                or taxonomy.fine_label_mapping[label] in observed_active_parents
            )
            for label in taxonomy.fine_labels
        ]
    else:
        fine_mask = explicit_fine
    selected_flags = set(row.get("flags", row.get("flags_reference_only", ())))
    if explicit_flags is None:
        exhaustive_flags = bool(
            row.get("prompt_sha256")
            or row.get("review_event_id")
            or row.get("review_action") in {"accept", "modify"}
            or str(row.get("annotator_type", "")).casefold() == "human"
        )
        flags_mask = [
            1 if exhaustive_flags else int(flag in selected_flags)
            for flag in taxonomy.flags
        ]
    else:
        flags_mask = explicit_flags
    return coarse_mask, fine_mask, flags_mask


def deterministic_safe_downsample(
    rows: Sequence[dict[str, Any]],
    *,
    safe_to_damage_ratio: float | None = 4.0,
    seed: int = 20260805,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce solo SEGURO en una partición y conserva aproximadamente sus canales.

    La selección es aleatoria reproducible mediante SHA-256, proporcional por
    canal y no altera la asignación de split ni elimina ningún chunk dañino.
    ``None`` conserva toda la partición para diagnósticos explícitos.
    """

    taxonomy = load_taxonomy()
    materialized = list(rows)
    harmful = [
        row
        for row in materialized
        if set(row.get("coarse_labels", ())).intersection(taxonomy.damage_labels)
    ]
    safe = [
        row
        for row in materialized
        if row.get("coarse_labels") == [taxonomy.safe_label]
        or tuple(row.get("coarse_labels", ())) == (taxonomy.safe_label,)
    ]
    if len(harmful) + len(safe) != len(materialized):
        raise ValueError("Train contiene una combinación gruesa no canónica")
    if safe_to_damage_ratio is None:
        selected = sorted(materialized, key=lambda row: str(row["chunk_id"]))
        return selected, {
            "policy": "all_safe",
            "ratio_requested": None,
            "damage_rows": len(harmful),
            "safe_rows_before": len(safe),
            "safe_rows_after": len(safe),
            "rows_after": len(selected),
        }
    if not math.isfinite(safe_to_damage_ratio) or safe_to_damage_ratio <= 0:
        raise ValueError("safe_to_damage_ratio debe ser positivo o None")
    target_safe = min(len(safe), math.ceil(len(harmful) * safe_to_damage_ratio))
    if target_safe == len(safe):
        selected_safe = safe
    else:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in safe:
            key = str(
                row.get("channel_id")
                or row.get("channel_title")
                or f"video:{row.get('video_id')}"
            )
            groups[key].append(row)
        raw_quotas = {
            key: target_safe * len(group) / max(1, len(safe))
            for key, group in groups.items()
        }
        quotas = {key: math.floor(value) for key, value in raw_quotas.items()}
        remaining = target_safe - sum(quotas.values())
        remainder_order = sorted(
            groups,
            key=lambda key: (
                -(raw_quotas[key] - quotas[key]),
                hashlib.sha256(f"{seed}|quota|{key}".encode()).hexdigest(),
            ),
        )
        for key in remainder_order[:remaining]:
            quotas[key] += 1
        selected_safe = []
        for key, group in groups.items():
            ranked = sorted(
                group,
                key=lambda row: hashlib.sha256(
                    f"{seed}|safe|{row['chunk_id']}".encode()
                ).hexdigest(),
            )
            selected_safe.extend(ranked[: quotas[key]])
    selected = sorted([*harmful, *selected_safe], key=lambda row: str(row["chunk_id"]))
    return selected, {
        "policy": "deterministic_channel_proportional_safe_downsample",
        "seed": seed,
        "ratio_requested": safe_to_damage_ratio,
        "damage_rows": len(harmful),
        "safe_rows_before": len(safe),
        "safe_rows_after": len(selected_safe),
        "realized_safe_to_damage_ratio": len(selected_safe) / max(1, len(harmful)),
        "rows_before": len(materialized),
        "rows_after": len(selected),
        "reduction_rows": len(materialized) - len(selected),
        "reduction_fraction": (len(materialized) - len(selected))
        / max(1, len(materialized)),
    }


def project_effective_training_rows(
    consolidated_rows: Iterable[dict[str, Any]],
    review_rows: Iterable[dict[str, Any]],
    previous_snapshot_rows: Iterable[dict[str, Any]] = (),
    *,
    seed: int = 20260805,
) -> list[dict[str, Any]]:
    """Proyecta el próximo snapshot sin escribir archivos.

    Aplica el último evento por ``chunk_id``, conserva las asignaciones históricas
    por video y excluye rechazos, diferimientos y propuestas aún no resueltas.
    Sirve para planificar adquisición antes de materializar formalmente ``02_05``.
    """

    taxonomy = load_taxonomy()
    assignments: dict[str, str] = {}
    for row in previous_snapshot_rows:
        video_id = str(row.get("video_id") or "").strip()
        split = str(row.get("split") or "").strip()
        if not video_id or split not in {"train", "validation", "test"}:
            continue
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"Snapshot previo contiene fuga para {video_id}")

    latest: dict[str, ReviewEvent] = {}
    for raw in review_rows:
        event = ReviewEvent.model_validate(raw)
        previous = latest.get(event.chunk_id)
        if previous is None or (event.created_at, event.event_id) > (
            previous.created_at,
            previous.event_id,
        ):
            latest[event.chunk_id] = event

    projected: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for raw in consolidated_rows:
        row = dict(raw)
        chunk_id = str(row.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError("La campaña contiene un chunk sin chunk_id")
        if chunk_id in seen_chunks:
            raise ValueError(f"Chunk duplicado en campaña: {chunk_id}")
        seen_chunks.add(chunk_id)
        event = latest.get(chunk_id)
        if event is not None:
            if event.action in {"reject", "defer"}:
                continue
            labels = taxonomy.normalize_categories(event.final_labels)
            label_source = (
                "human_accepted" if event.action == "accept" else "human_modified"
            )
        else:
            if (
                not row.get("training_eligible")
                or row.get("needs_review")
                or row.get("decision_status") != "resolved"
            ):
                continue
            labels = taxonomy.normalize_categories(row.get("coarse_labels") or ())
            label_source = str(row.get("label_source") or "unknown")
        if not labels:
            continue
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError(f"Falta video_id explícito en {chunk_id}")
        split = assignments.get(video_id) or stable_video_split(video_id, seed)
        projected.append(
            {
                "chunk_id": chunk_id,
                "video_id": video_id,
                "channel_title": row.get("channel_title"),
                "coarse_labels": list(labels),
                "split": split,
                "label_source": label_source,
                "training_eligible": True,
                "needs_review": False,
                "decision_status": "resolved",
            }
        )
    projected.sort(key=lambda row: (row["split"], row["video_id"], row["chunk_id"]))
    return projected


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
        split = (
            assignments.get(video_id)
            or row.get("split")
            or stable_video_split(video_id, seed)
        )
        if video_id in video_splits and video_splits[video_id] != split:
            raise ValueError(f"Fuga de video detectada para {video_id}")
        video_splits[video_id] = split
        coarse_mask, fine_mask, flags_mask = derive_supervision_masks(row)
        channel_key = str(
            row.get("channel_id") or row.get("channel_title") or f"video:{video_id}"
        )
        prepared = ModelReadyRecord(
            chunk_id=chunk_id,
            video_id=video_id,
            channel_id=row.get("channel_id"),
            channel_title=row.get("channel_title"),
            text=str(row["text"]),
            coarse_labels=list(labels),
            fine_labels=list(row.get("fine_labels", [])),
            flags_reference_only=list(
                row.get("flags_reference_only", row.get("flags", []))
            ),
            coarse_observed_mask=coarse_mask,
            fine_observed_mask=fine_mask,
            flags_observed_mask=flags_mask,
            label_source=str(row.get("label_source") or "unknown"),
            prompt_sha256=row.get("prompt_sha256"),
            sample_weight=float(row.get("sample_weight", 1.0)),
            campaign=row.get("campaign"),
            split=split,
            channel_split=stable_channel_split(channel_key, seed),
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


def previous_video_assignments(
    source: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    path = Path(source)
    if not path.is_file():
        return {}
    assignments: dict[str, str] = {}
    _notify_progress(
        progress_callback, status="phase_started", phase="loading_previous_snapshot"
    )
    for completed, row in enumerate(read_jsonl(path), 1):
        video_id = str(row["video_id"])
        split = str(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"Snapshot previo contiene fuga para {video_id}")
        _notify_progress(
            progress_callback,
            status="progress",
            phase="loading_previous_snapshot",
            advance=1,
            completed=completed,
        )
    return assignments


def materialize_versioned_training_snapshot(
    source: str | Path,
    canonical_destination: str | Path,
    *,
    snapshots_dir: str | Path | None = None,
    seed: int = 20260805,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Crea un snapshot inmutable y actualiza la vista canónica solo si cambió.

    Las asignaciones anteriores por ``video_id`` se heredan. El ID del snapshot
    depende de sus filas canónicas, por lo que repetir la etapa es un no-op.
    """

    source_path = Path(source)
    canonical = Path(canonical_destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    assignments = previous_video_assignments(
        canonical, progress_callback=progress_callback
    )
    prepared_rows: list[dict[str, Any]] = []
    taxonomy = load_taxonomy()
    _notify_progress(
        progress_callback, status="phase_started", phase="preparing_snapshot"
    )
    for completed, row in enumerate(read_jsonl(source_path), 1):
        if (
            not row.get("training_eligible")
            or row.get("needs_review")
            or row.get("decision_status") != "resolved"
        ):
            _notify_progress(
                progress_callback,
                status="progress",
                phase="preparing_snapshot",
                advance=1,
                completed=completed,
                eligible=len(prepared_rows),
            )
            continue
        if not row.get("coarse_labels"):
            _notify_progress(
                progress_callback,
                status="progress",
                phase="preparing_snapshot",
                advance=1,
                completed=completed,
                eligible=len(prepared_rows),
            )
            continue
        if not row.get("video_id"):
            raise ValueError(
                f"Falta video_id explícito en {row.get('chunk_id')}; revise 02_05"
            )
        video_id = str(row["video_id"])
        split = assignments.get(video_id) or stable_video_split(video_id, seed)
        coarse_mask, fine_mask, flags_mask = derive_supervision_masks(row)
        channel_key = str(
            row.get("channel_id") or row.get("channel_title") or f"video:{video_id}"
        )
        record = ModelReadyRecord(
            chunk_id=str(row["chunk_id"]),
            video_id=video_id,
            channel_id=row.get("channel_id"),
            channel_title=row.get("channel_title"),
            text=str(row["text"]),
            coarse_labels=list(taxonomy.normalize_categories(row["coarse_labels"])),
            fine_labels=list(row.get("fine_labels", [])),
            flags_reference_only=list(
                row.get("flags", row.get("flags_reference_only", []))
            ),
            coarse_observed_mask=coarse_mask,
            fine_observed_mask=fine_mask,
            flags_observed_mask=flags_mask,
            label_source=str(row.get("label_source") or "unknown"),
            prompt_sha256=row.get("prompt_sha256"),
            sample_weight=float(row.get("sample_weight", 1.0)),
            campaign=row.get("campaign"),
            split=split,
            channel_split=stable_channel_split(channel_key, seed),
            needs_review=False,
            training_eligible=True,
            decision_status="resolved",
            legacy_coarse_labels=list(row.get("legacy_coarse_labels", [])),
            label_source_original=row.get("label_source_original"),
            migration_warning=row.get("migration_warning"),
        )
        prepared_rows.append(record.model_dump(mode="json"))
        _notify_progress(
            progress_callback,
            status="progress",
            phase="preparing_snapshot",
            advance=1,
            completed=completed,
            eligible=len(prepared_rows),
        )
    if not prepared_rows:
        raise ValueError(
            "No existen decisiones resueltas y entrenables para materializar"
        )
    unique: dict[str, dict[str, Any]] = {}
    _notify_progress(
        progress_callback,
        status="phase_started",
        phase="deduplicating_snapshot",
        total=len(prepared_rows),
    )
    for completed, row in enumerate(prepared_rows, 1):
        previous = unique.get(row["chunk_id"])
        if previous is not None and canonical_json_sha256(
            previous
        ) != canonical_json_sha256(row):
            raise ValueError(
                f"Decisiones entrenables duplicadas para {row['chunk_id']}"
            )
        unique[row["chunk_id"]] = row
        _notify_progress(
            progress_callback,
            status="progress",
            phase="deduplicating_snapshot",
            advance=1,
            total=len(prepared_rows),
            completed=completed,
        )
    prepared_rows = sorted(
        unique.values(),
        key=lambda row: (row["split"], row["video_id"], row["chunk_id"]),
    )
    assert_no_video_leakage(prepared_rows, progress_callback=progress_callback)
    content_signature = canonical_json_sha256(prepared_rows)
    snapshot_id = f"v{taxonomy.version}-{content_signature[:16]}"
    snapshot_root = (
        Path(snapshots_dir) if snapshots_dir else canonical.parent / "snapshots"
    )
    snapshot_path = snapshot_root / snapshot_id / canonical.name
    manifest_path = snapshot_path.with_name("snapshot_manifest.json")
    if snapshot_path.is_file():
        if (
            json.loads(manifest_path.read_text(encoding="utf-8"))["content_signature"]
            != content_signature
        ):
            raise ValueError(f"Colisión o snapshot alterado: {snapshot_path}")
        status = "noop"
    else:
        write_jsonl_atomic(snapshot_path, prepared_rows)
        counts = Counter(row["split"] for row in prepared_rows)
        label_counts = Counter(
            label for row in prepared_rows for label in row["coarse_labels"]
        )
        channel_split_counts = Counter(row["channel_split"] for row in prepared_rows)
        fine_observed = sum(sum(row["fine_observed_mask"]) for row in prepared_rows)
        flag_observed = sum(sum(row["flags_observed_mask"]) for row in prepared_rows)
        manifest = {
            "schema_version": "2.1.0",
            "snapshot_id": snapshot_id,
            "created_at": datetime.now(UTC).isoformat(),
            "taxonomy_contract": taxonomy.contract_id,
            "taxonomy_version": taxonomy.version,
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "dataset": str(snapshot_path),
            "dataset_sha256": sha256_file(snapshot_path),
            "content_signature": content_signature,
            "split_policy": f"preserve previous; otherwise sha256(video_id, seed={seed}), grouped 70/15/15",
            "channel_robustness_split_policy": (
                f"sha256(channel_id|channel_title, seed={seed}), grouped 70/15/15"
            ),
            "supervision_masks": {
                "coarse": "fully observed",
                "fine": "inactive parent observed negative; active parent observed only with a fine reference",
                "flags": "exhaustive for versioned prompt or human review; otherwise explicit positives only",
                "fine_observed_positions": fine_observed,
                "flag_observed_positions": flag_observed,
            },
            "counts": {
                "rows": len(prepared_rows),
                "videos": len({row["video_id"] for row in prepared_rows}),
                **{f"split:{key}": value for key, value in counts.items()},
                **{
                    f"channel_split:{key}": value
                    for key, value in channel_split_counts.items()
                },
                **{f"label:{key}": value for key, value in label_counts.items()},
            },
        }
        write_json_atomic(manifest_path, manifest)
        status = "created"
    snapshot_sha = sha256_file(snapshot_path)
    canonical_changed = (
        not canonical.is_file() or sha256_file(canonical) != snapshot_sha
    )
    if canonical_changed:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        temporary = canonical.with_name(f".{canonical.name}.{snapshot_id}.partial")
        shutil.copyfile(snapshot_path, temporary)
        if sha256_file(temporary) != snapshot_sha:
            temporary.unlink(missing_ok=True)
            raise ValueError("La vista canónica no conserva el SHA-256 del snapshot")
        temporary.replace(canonical)
    result = {
        "status": status if not canonical_changed else "updated",
        "snapshot_id": snapshot_id,
        "snapshot": str(snapshot_path),
        "manifest": str(manifest_path),
        "canonical": str(canonical),
        "dataset_sha256": snapshot_sha,
        "rows": len(prepared_rows),
        "videos": len({row["video_id"] for row in prepared_rows}),
    }
    _notify_progress(
        progress_callback,
        status="finished",
        phase="snapshot",
        result_status=result["status"],
        rows=result["rows"],
        videos=result["videos"],
    )
    return result


def load_split(source: str | Path, split: str) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(source)
        if row.get("split") == split
        and row.get("training_eligible", True)
        and not row.get("needs_review", False)
        and row.get("decision_status", "resolved") == "resolved"
    ]


def assert_no_video_leakage(
    rows: Iterable[dict[str, Any]],
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    assignments: dict[str, str] = {}
    materialized = list(rows)
    _notify_progress(
        progress_callback,
        status="phase_started",
        phase="validating_video_splits",
        total=len(materialized),
    )
    for completed, row in enumerate(materialized, 1):
        video_id = str(row["video_id"])
        split = str(row["split"])
        previous = assignments.setdefault(video_id, split)
        if previous != split:
            raise ValueError(f"El video {video_id} aparece en {previous} y {split}")
        _notify_progress(
            progress_callback,
            status="progress",
            phase="validating_video_splits",
            advance=1,
            total=len(materialized),
            completed=completed,
        )


def audit_training_snapshot(
    source: str | Path, destination: str | Path
) -> dict[str, Any]:
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
    channel_split_counts = Counter()
    fine_observed_counts = Counter()
    flag_observed_counts = Counter()
    inconsistencies = []
    for row in rows:
        split_counts[row["split"]] += 1
        channel_split_counts[row.get("channel_split", "missing")] += 1
        fine_counts.update(row.get("fine_labels", []))
        flag_counts.update(row.get("flags_reference_only", []))
        fine_mask = row.get("fine_observed_mask", [])
        flag_mask = row.get("flags_observed_mask", [])
        for index, label in enumerate(taxonomy.fine_labels):
            if index < len(fine_mask) and fine_mask[index]:
                fine_observed_counts[label] += 1
        for index, flag in enumerate(taxonomy.flags):
            if index < len(flag_mask) and flag_mask[index]:
                flag_observed_counts[flag] += 1
        fine = row.get("fine_labels", [])
        if fine:
            derived = set(taxonomy.derive_categories(fine))
            if not derived.issubset(set(row.get("coarse_labels", []))):
                inconsistencies.append(row["chunk_id"])
    payload = {
        "schema_version": "2.1.0",
        "operation": "audit_fine_labels_and_flags",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(source_path),
        "dataset_sha256": dataset_sha,
        "rows": len(rows),
        "splits": dict(split_counts),
        "channel_robustness_splits": dict(channel_split_counts),
        "fine_label_counts": {
            label: fine_counts[label] for label in taxonomy.fine_labels
        },
        "flag_counts": {flag: flag_counts[flag] for flag in taxonomy.flags},
        "fine_observed_counts": {
            label: fine_observed_counts[label] for label in taxonomy.fine_labels
        },
        "flag_observed_counts": {
            flag: flag_observed_counts[flag] for flag in taxonomy.flags
        },
        "rows_without_fine_reference": sum(not row.get("fine_labels") for row in rows),
        "rows_with_complete_fine_mask": sum(
            len(row.get("fine_observed_mask", [])) == len(taxonomy.fine_labels)
            and all(row.get("fine_observed_mask", []))
            for row in rows
        ),
        "rows_with_complete_flag_mask": sum(
            len(row.get("flags_observed_mask", [])) == len(taxonomy.flags)
            and all(row.get("flags_observed_mask", []))
            for row in rows
        ),
        "fine_to_coarse_inconsistencies": len(inconsistencies),
        "inconsistent_chunk_ids_sample": inconsistencies[:100],
        "interpretation": "Las finas y flags son objetivos auxiliares únicamente en posiciones con máscara observada; ausente no equivale a negativo.",
    }
    write_json_atomic(destination_path, payload)
    return {"status": "updated", **payload}


def audit_auxiliary_candidate_metrics(
    candidate_roots: Sequence[str | Path],
    destination_path: str | Path,
) -> dict[str, Any]:
    """Reúne calidad de finas/flags ya calculada sobre posiciones observadas."""

    candidates = []
    for root in candidate_roots:
        path = Path(root)
        for candidate_path in path.rglob("candidate.json") if path.exists() else []:
            try:
                row = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            auxiliary = row.get("auxiliary_validation_metrics")
            if not auxiliary:
                continue
            candidates.append(
                {
                    "candidate_id": row.get("candidate_id"),
                    "dataset_sha256": row.get("dataset_sha256"),
                    "test_status": row.get("test_status"),
                    "auxiliary_validation_observed_only": auxiliary,
                    "candidate_path": str(candidate_path),
                }
            )
    payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "interpretation": "Comparar solo candidatos del mismo SHA; finas y flags se miden exclusivamente donde su máscara vale uno.",
        "candidates": candidates,
    }
    write_json_atomic(destination_path, payload)
    return {"status": "updated", "candidates": len(candidates), **payload}
