from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


TranscriptFetcher = Callable[[dict[str, Any]], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]
DEFAULT_SUBTITLE_LANGUAGES = ("es-PE", "es-419", "es")
DEFAULT_MIN_TRANSCRIPT_CHARACTERS = 200
DEFAULT_DAMAGE_LABELS = (
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
)
VIDEO_DATASET_RESET_CONFIRMATION = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"
VIDEO_DATASET_RESET_MARKER = Path("datos/raw/manifests/rebuild_from_zero.json")
LEGACY_CATEGORY_ALIASES = {
    "ACOSO_GENERO_IDENTIDAD": "ATAQUE_POR_GENERO_IDENTIDAD",
}


def _normalize_category_value(value: Any) -> Any:
    if isinstance(value, str):
        return "|".join(
            LEGACY_CATEGORY_ALIASES.get(token.strip(), token.strip())
            for token in value.split("|")
        )
    if isinstance(value, list):
        return [LEGACY_CATEGORY_ALIASES.get(str(token), token) for token in value]
    return value


def normalize_category_metadata(value: Any) -> Any:
    """Normaliza aliases solo en metadatos de adquisición; no toca el texto."""

    if isinstance(value, dict):
        return {
            key: (
                _normalize_category_value(item)
                if key in {"target_category", "target_categories", "categoria_objetivo"}
                else normalize_category_metadata(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_category_metadata(item) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    """Normaliza NaN históricos para producir JSON estricto sin alterar fuentes."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Lee candidatos JSONL o CSV conservando ``video_id`` como clave."""

    source = Path(path)
    if not source.is_file():
        return []
    if source.suffix.lower() == ".jsonl":
        rows = list(read_jsonl(source))
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"Formato de candidatos no compatible: {source.suffix}")
    return [row for row in rows if str(row.get("video_id", "")).strip()]


def merge_candidates(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Une fuentes en orden y conserva una sola fila por ``video_id``."""

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for candidate in group:
            video_id = str(candidate.get("video_id", "")).strip()
            if video_id:
                candidates_by_id.setdefault(video_id, normalize_category_metadata(candidate))
    return list(candidates_by_id.values())


def _category_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = ()
    normalized = []
    for token in raw:
        label = LEGACY_CATEGORY_ALIASES.get(str(token).strip(), str(token).strip())
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)


def build_directed_sampling_plan(
    dataset_rows: Iterable[dict[str, Any]],
    transcript_rows: Iterable[dict[str, Any]],
    *,
    damage_labels: Iterable[str] = DEFAULT_DAMAGE_LABELS,
    eligible_splits: Iterable[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Calcula déficits por video y rendimiento histórico por canal.

    El conjunto de prueba se excluye por contrato. Si no hay positivos previos
    utilizables, las cuatro categorías reciben el mismo peso de adquisición.
    Las etiquetas sirven para seleccionar fuentes; nunca se copian como verdad
    de los videos nuevos.
    """

    labels = tuple(dict.fromkeys(str(label).strip() for label in damage_labels if str(label).strip()))
    if not labels:
        raise ValueError("Se requiere al menos una categoría de daño")
    allowed_splits = {str(split) for split in eligible_splits}
    labeled_video_ids: set[str] = set()
    video_labels: dict[str, set[str]] = defaultdict(set)
    labeled_rows = 0
    for row in dataset_rows:
        if allowed_splits and str(row.get("split")) not in allowed_splits:
            continue
        video_id = str(row.get("video_id", "")).strip()
        coarse = _category_tokens(row.get("coarse_labels", ()))
        if not video_id or not coarse:
            continue
        labeled_rows += 1
        labeled_video_ids.add(video_id)
        video_labels[video_id].update(label for label in coarse if label in labels)

    support = {
        label: sum(label in video_labels[video_id] for video_id in labeled_video_ids)
        for label in labels
    }
    target_support = max(support.values(), default=0)
    deficits = {label: max(target_support - support[label], 0) for label in labels}
    total_deficit = sum(deficits.values())
    if not labeled_video_ids or not target_support or not total_deficit:
        strategy = "fallback_equal"
        weights = {label: 1.0 / len(labels) for label in labels}
        deficits = {label: 1 for label in labels}
    else:
        strategy = "deficit_weighted"
        weights = {label: deficits[label] / total_deficit for label in labels}

    channel_accumulator: dict[str, dict[str, Any]] = {}
    for transcript in transcript_rows:
        video_id = str(transcript.get("video_id", "")).strip()
        if video_id not in labeled_video_ids:
            continue
        source_candidate = transcript.get("source_candidate") or {}
        channel_id = str(
            transcript.get("channel_id") or source_candidate.get("channel_id") or ""
        ).strip()
        channel_title = str(
            transcript.get("channel_title")
            or transcript.get("channel")
            or source_candidate.get("channel_title")
            or source_candidate.get("channel")
            or ""
        ).strip()
        if not channel_id and not channel_title:
            continue
        key = channel_id or f"title:{channel_title.casefold()}"
        profile = channel_accumulator.setdefault(
            key,
            {
                "channel_id": channel_id or None,
                "channel_title": channel_title or channel_id,
                "video_ids": set(),
                "positive_video_ids": {label: set() for label in labels},
            },
        )
        profile["video_ids"].add(video_id)
        for label in video_labels[video_id]:
            profile["positive_video_ids"][label].add(video_id)

    channel_profiles = []
    for profile in channel_accumulator.values():
        video_count = len(profile["video_ids"])
        positive_counts = {
            label: len(profile["positive_video_ids"][label]) for label in labels
        }
        channel_profiles.append(
            {
                "channel_id": profile["channel_id"],
                "channel_title": profile["channel_title"],
                "labeled_videos": video_count,
                "positive_videos": positive_counts,
                "positive_rate": {
                    label: positive_counts[label] / video_count for label in labels
                },
            }
        )
    channel_profiles.sort(
        key=lambda profile: (-profile["labeled_videos"], profile["channel_title"].casefold())
    )
    return {
        "strategy": strategy,
        "damage_labels": list(labels),
        "eligible_splits": sorted(allowed_splits),
        "labeled_rows": labeled_rows,
        "labeled_videos": len(labeled_video_ids),
        "support_videos": support,
        "target_video_support": target_support,
        "deficit_videos": deficits,
        "weights": weights,
        "channel_profiles": channel_profiles,
    }


def select_directed_seed_channels(
    plan: dict[str, Any],
    curated_channels: Iterable[dict[str, Any]],
    *,
    max_channels: int = 12,
    min_historical_videos: int = 3,
) -> list[dict[str, Any]]:
    """Combina canales de alto rendimiento observado con semillas curadas."""

    if max_channels < 1 or min_historical_videos < 1:
        raise ValueError("Los límites de canales dirigidos deben ser positivos")
    labels = tuple(plan["damage_labels"])
    weights = {label: float(plan["weights"].get(label, 0.0)) for label in labels}
    ranked: list[dict[str, Any]] = []
    for profile in plan.get("channel_profiles", []):
        total = int(profile.get("labeled_videos", 0))
        channel_id = str(profile.get("channel_id") or "").strip()
        if total < min_historical_videos or not channel_id:
            continue
        label_scores = {}
        for label in labels:
            positives = int(profile.get("positive_videos", {}).get(label, 0))
            if positives and weights[label] > 0:
                smoothed_yield = (positives + 0.5) / (total + 2.0)
                label_scores[label] = weights[label] * smoothed_yield
        if not label_scores:
            continue
        target_labels = sorted(label_scores, key=lambda label: (-label_scores[label], labels.index(label)))
        reliability = total / (total + 5.0)
        score = sum(label_scores.values()) * reliability
        ranked.append(
            {
                "name": profile["channel_title"],
                "url": f"https://www.youtube.com/channel/{channel_id}",
                "channel_id": channel_id,
                "target_category": "|".join(target_labels),
                "reason": (
                    f"preclasificación histórica: {total} videos; "
                    + ", ".join(
                        f"{label}={profile['positive_videos'][label]}" for label in target_labels
                    )
                ),
                "sampling_mode": "directed",
                "priority_weight": score,
                "_targets": set(target_labels),
                "_score": score,
            }
        )

    for raw in curated_channels:
        source = normalize_category_metadata(dict(raw))
        targets = set(_category_tokens(source.get("target_category"))) & set(labels)
        active_targets = {label for label in targets if weights[label] > 0}
        if not active_targets:
            continue
        score = sum(weights[label] for label in active_targets)
        source.update(
            {
                "target_category": "|".join(label for label in labels if label in active_targets),
                "sampling_mode": "directed",
                "priority_weight": score,
                "_targets": active_targets,
                "_score": score,
            }
        )
        source.setdefault("reason", "semilla curada para ampliación dirigida")
        ranked.append(source)

    ranked.sort(key=lambda source: (-float(source["_score"]), str(source.get("name", "")).casefold()))
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def add(source: dict[str, Any]) -> bool:
        keys = {
            str(value).strip().casefold()
            for value in (source.get("channel_id"), source.get("name"), source.get("url"))
            if str(value or "").strip()
        }
        if keys & selected_keys or len(selected) >= max_channels:
            return False
        selected_keys.update(keys)
        selected.append(source)
        return True

    for label in sorted(labels, key=lambda item: (-weights[item], labels.index(item))):
        if weights[label] <= 0:
            continue
        for match in ranked:
            if label in match["_targets"] and add(match):
                break
    for source in ranked:
        add(source)
    output = []
    for source in selected:
        clean = {key: value for key, value in source.items() if not key.startswith("_")}
        output.append(clean)
    return output


def select_directed_search_queries(
    plan: dict[str, Any],
    query_catalog: Iterable[dict[str, Any]],
    *,
    max_queries: int = 12,
    max_results_per_query: int = 20,
) -> list[dict[str, Any]]:
    """Selecciona consultas temáticas según el déficit vigente."""

    if max_queries < 1 or max_results_per_query < 1:
        raise ValueError("Los límites de búsqueda dirigida deben ser positivos")
    labels = tuple(plan["damage_labels"])
    weights = {label: float(plan["weights"].get(label, 0.0)) for label in labels}
    ranked = []
    for raw in query_catalog:
        query = normalize_category_metadata(dict(raw))
        targets = set(_category_tokens(query.get("target_category"))) & set(labels)
        active = {label for label in targets if weights[label] > 0}
        if not str(query.get("query", "")).strip() or not active:
            continue
        score = sum(weights[label] for label in active)
        query.update(
            {
                "target_category": "|".join(label for label in labels if label in active),
                "sampling_mode": "directed",
                "priority_weight": score,
                "quota": min(int(query.get("quota", max_results_per_query)), max_results_per_query),
                "_targets": active,
                "_score": score,
            }
        )
        ranked.append(query)
    ranked.sort(key=lambda query: (-float(query["_score"]), str(query["query"]).casefold()))
    selected: list[dict[str, Any]] = []
    selected_queries: set[str] = set()

    def add(query: dict[str, Any]) -> None:
        key = str(query["query"]).casefold()
        if key not in selected_queries and len(selected) < max_queries:
            selected_queries.add(key)
            selected.append(query)

    for label in sorted(labels, key=lambda item: (-weights[item], labels.index(item))):
        if weights[label] <= 0:
            continue
        match = next((query for query in ranked if label in query["_targets"]), None)
        if match is not None:
            add(match)
    for query in ranked:
        add(query)
    return [{key: value for key, value in query.items() if not key.startswith("_")} for query in selected]


def expand_directed_channel_sources(
    search_candidates: Iterable[dict[str, Any]],
    plan: dict[str, Any],
    *,
    known_channel_ids: Iterable[str] = (),
    max_channels: int = 12,
    videos_per_channel: int = 20,
) -> list[dict[str, Any]]:
    """Convierte canales hallados por búsqueda temática en nuevas semillas."""

    if max_channels < 0 or videos_per_channel < 1:
        raise ValueError("Los límites de expansión de canales no son válidos")
    if max_channels == 0:
        return []
    labels = tuple(plan["damage_labels"])
    weights = {label: float(plan["weights"].get(label, 0.0)) for label in labels}
    known = {str(channel_id).strip() for channel_id in known_channel_ids if str(channel_id).strip()}
    groups: dict[str, dict[str, Any]] = {}
    for candidate in search_candidates:
        if candidate.get("discovery_type") != "search":
            continue
        channel_id = str(candidate.get("channel_id") or "").strip()
        if not channel_id or channel_id in known:
            continue
        targets = set(_category_tokens(candidate.get("target_category"))) & set(labels)
        targets = {label for label in targets if weights[label] > 0}
        if not targets:
            continue
        group = groups.setdefault(
            channel_id,
            {
                "channel_id": channel_id,
                "name": candidate.get("channel_title") or channel_id,
                "targets": set(),
                "hits": 0,
                "best_rank": 10**9,
            },
        )
        group["targets"].update(targets)
        group["hits"] += 1
        group["best_rank"] = min(group["best_rank"], int(candidate.get("discovery_rank") or 10**9))
    ranked = []
    for group in groups.values():
        score = sum(weights[label] for label in group["targets"])
        score *= 1.0 + 0.1 * min(group["hits"] - 1, 4)
        score /= 1.0 + 0.02 * max(group["best_rank"] - 1, 0)
        ranked.append((score, group))
    ranked.sort(key=lambda item: (-item[0], item[1]["best_rank"], str(item[1]["name"]).casefold()))
    output = []
    for score, group in ranked[:max_channels]:
        ordered_targets = [label for label in labels if label in group["targets"]]
        output.append(
            {
                "name": group["name"],
                "url": f"https://www.youtube.com/channel/{group['channel_id']}",
                "channel_id": group["channel_id"],
                "quota": videos_per_channel,
                "target_category": "|".join(ordered_targets),
                "reason": f"canal descubierto en {group['hits']} resultados temáticos",
                "sampling_mode": "directed",
                "priority_weight": score,
            }
        )
    return output


def select_directed_candidates(
    candidates: Iterable[dict[str, Any]],
    processed_ids: Iterable[str],
    plan: dict[str, Any],
    *,
    max_candidates: int | None = 500,
) -> list[dict[str, Any]]:
    """Crea una cohorte inédita con round-robin ponderado por déficit."""

    if max_candidates is not None and max_candidates < 0:
        raise ValueError("max_candidates no puede ser negativo")
    if max_candidates == 0:
        return []
    labels = tuple(plan["damage_labels"])
    weights = {label: float(plan["weights"].get(label, 0.0)) for label in labels}
    processed = {str(video_id).strip() for video_id in processed_ids}
    unique: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        candidate = normalize_category_metadata(dict(raw))
        video_id = str(candidate.get("video_id", "")).strip()
        targets = set(_category_tokens(candidate.get("target_category"))) & set(labels)
        if video_id and video_id not in processed and targets:
            unique.setdefault(video_id, candidate)

    queues: dict[str, deque[dict[str, Any]]] = {}
    for label in labels:
        if weights[label] <= 0:
            continue
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in unique.values():
            if label in _category_tokens(candidate.get("target_category")):
                source = str(candidate.get("discovery_source") or candidate.get("channel_id") or "")
                by_source[source].append(candidate)
        for group in by_source.values():
            group.sort(key=lambda row: (int(row.get("discovery_rank") or 10**9), str(row["video_id"])))
        source_cycle = deque(sorted(by_source))
        ordered: list[dict[str, Any]] = []
        while source_cycle:
            source = source_cycle.popleft()
            group = by_source[source]
            ordered.append(group.pop(0))
            if group:
                source_cycle.append(source)
        if ordered:
            queues[label] = deque(ordered)

    active = set(queues)
    credits = {label: 0.0 for label in active}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while active and (max_candidates is None or len(selected) < max_candidates):
        for label in active:
            credits[label] += weights[label]
        label = max(active, key=lambda item: (credits[item], -labels.index(item)))
        candidate = None
        while queues[label]:
            proposed = queues[label].popleft()
            if str(proposed["video_id"]) not in selected_ids:
                candidate = proposed
                break
        if candidate is None:
            active.remove(label)
            continue
        credits[label] -= 1.0
        selected_ids.add(str(candidate["video_id"]))
        selected.append(
            {
                **candidate,
                "directed_priority_label": label,
                "directed_selection_rank": len(selected) + 1,
            }
        )
    return selected


def reset_active_video_dataset(
    project_root: str | Path,
    confirmation: str,
) -> dict[str, Any]:
    """Archiva artefactos activos y deja el proyecto listo para reconstrucción.

    La operación es deliberadamente recuperable: mueve solo rutas activas
    conocidas a ``archivo/reinicios_dataset_videos``. No toca código, fuentes
    de configuración ni las ampliaciones históricas. Un marcador persistente
    impide que esas ampliaciones repueblen el canónico automáticamente.
    """

    if confirmation != VIDEO_DATASET_RESET_CONFIRMATION:
        raise ValueError(
            "Confirmación inválida. Use exactamente "
            f"{VIDEO_DATASET_RESET_CONFIRMATION!r}."
        )
    root = Path(project_root).resolve()
    if not (root / "src" / "moderacion_peru").is_dir() or not (root / "datos").is_dir():
        raise ValueError(f"La ruta no parece ser la raíz del proyecto: {root}")
    marker = root / VIDEO_DATASET_RESET_MARKER
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8-sig"))
        return {**previous, "status": "already_active_noop"}
    relative_targets = (
        Path("datos/raw/video_candidates.jsonl"),
        Path("datos/raw/videos_candidatos.csv"),
        Path("datos/raw/transcripts_raw.jsonl"),
        Path("datos/raw/transcripts_cache"),
        Path("datos/raw/fallos_adquisicion.jsonl"),
        Path("datos/raw/fallos_descubrimiento_ultima_ejecucion.json"),
        Path("datos/raw/directed_candidates_latest.jsonl"),
        Path("datos/raw/manifests"),
        Path("datos/processed"),
        Path("datos/etiquetado"),
        Path("datos/model_ready"),
        Path("modelos/v2"),
        Path("modelos/registro_modelos_5_salidas.json"),
        Path("resultados/modelos"),
        Path("resultados/auditorias"),
        Path("resultados/colab_bundle"),
    )
    channel_partition_dir = root / "datos" / "raw" / "transcripts_by_channel"
    channel_partition_targets = (
        [path.relative_to(root) for path in sorted(channel_partition_dir.glob("*.jsonl"))]
        + [Path("datos/raw/transcripts_by_channel/index.json")]
        if channel_partition_dir.is_dir()
        else []
    )
    relative_targets = (*relative_targets, *channel_partition_targets)
    resolved_targets = []
    for relative in relative_targets:
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError(f"Objetivo de reinicio fuera del proyecto: {target}")
        if target.exists():
            resolved_targets.append((relative, target))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = root / "archivo" / "reinicios_dataset_videos" / timestamp
    moved = []
    for relative, source in resolved_targets:
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(relative.as_posix())

    payload = {
        "status": "archived_and_armed",
        "mode": "rebuild_video_dataset_from_zero",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_snapshot_bootstrap": False,
        "archive_path": archive_root.relative_to(root).as_posix() if moved else None,
        "moved_paths": moved,
        "recovery": (
            "Restaure las rutas desde archive_path y retire este marcador solo si desea "
            "volver al estado anterior."
        ),
    }
    write_json_atomic(marker, payload)
    return payload


def discover_existing_transcript_sources(
    project_root: str | Path,
    *,
    canonical_path: str | Path | None = None,
) -> list[Path]:
    """Descubre snapshots JSONL existentes sin incluir el destino canónico."""

    root = Path(project_root).resolve()
    canonical = Path(canonical_path).resolve() if canonical_path else None
    return [
        path
        for path in sorted((root / "datos").rglob("transcripts_raw.jsonl"))
        if path.is_file() and (canonical is None or path.resolve() != canonical)
    ]


def bootstrap_canonical_from_existing(
    sources: Iterable[str | Path],
    canonical_path: str | Path,
) -> dict[str, Any]:
    """Materializa una vista canónica reutilizando snapshots; nunca los modifica."""

    destination = Path(canonical_path)
    rows: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    for raw_source in sources:
        source = Path(raw_source).resolve()
        count = 0
        for historical in read_jsonl(source):
            record = normalize_category_metadata(_json_safe(historical))
            video_id = str(record.get("video_id", "")).strip()
            if not video_id:
                continue
            record["video_id"] = video_id
            record.setdefault("acquisition_status", "reused_existing_snapshot")
            record.setdefault("source_snapshot", source.as_posix())
            record.setdefault(
                "transcript_sha256",
                sha256_text(
                    json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
                ),
            )
            rows.append(record)
            count += 1
        source_stats.append({"path": source.as_posix(), "rows": count})
    added, skipped = append_jsonl_once(destination, rows, id_field="video_id")
    return {
        "sources": source_stats,
        "candidate_rows": len(rows),
        "added": added,
        "already_canonical": skipped,
        "canonical_path": destination.resolve().as_posix(),
    }


def bootstrap_canonical_from_cache(
    cache_dir: str | Path,
    canonical_path: str | Path,
) -> dict[str, Any]:
    """Incorpora al canónico los JSON individuales válidos que aún falten."""

    source = Path(cache_dir)
    cache_files = sorted(source.glob("*.json")) if source.is_dir() else []
    stats = {"cache_files": len(cache_files), "valid_records": 0}

    def records() -> Iterable[dict[str, Any]]:
        for path in cache_files:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError(f"El caché no contiene un objeto JSON: {path}")
            record = normalize_category_metadata(_json_safe(payload))
            video_id = str(record.get("video_id") or "").strip()
            if not video_id:
                raise ValueError(f"Falta video_id en el caché {path}")
            if path.stem != video_id:
                raise ValueError(f"El caché {path} no corresponde a {video_id}")
            record["video_id"] = video_id
            record.setdefault("acquisition_status", "reused_existing_cache")
            record.setdefault("source_cache", path.resolve().as_posix())
            record.setdefault(
                "transcript_sha256",
                sha256_text(
                    json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
                ),
            )
            stats["valid_records"] += 1
            yield record

    added, skipped = append_jsonl_once(canonical_path, records(), id_field="video_id")
    return {
        **stats,
        "added": added,
        "already_canonical": skipped,
        "canonical_path": Path(canonical_path).resolve().as_posix(),
    }


CHANNEL_TRANSCRIPT_INDEX = "index.json"
DEFAULT_MAX_CHANNEL_SHARD_BYTES = 25 * 1024 * 1024
VTT_CHECKPOINT_INDEX = "index.json"
VTT_MISSING_MANIFEST = "missing_vtt.jsonl"


def _transcript_channel_fields(record: dict[str, Any]) -> tuple[str, str]:
    source = record.get("source_candidate") or {}
    channel_id = str(record.get("channel_id") or source.get("channel_id") or "").strip()
    channel_title = str(
        record.get("channel_title")
        or record.get("channel")
        or source.get("channel_title")
        or source.get("channel")
        or ""
    ).strip()
    return channel_id, channel_title


def _safe_channel_slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value).strip("-._").lower()
    return slug[:64] or "canal"


def _channel_shard_descriptor(
    record: dict[str, Any],
    *,
    title_aliases: dict[str, str] | None = None,
) -> tuple[str, str]:
    channel_id, channel_title = _transcript_channel_fields(record)
    alias = (title_aliases or {}).get(channel_title.casefold()) if channel_title else None
    if channel_id:
        key = f"id:{channel_id}"
        basename = channel_id
    elif alias:
        key = f"id:{alias}"
        basename = alias
    elif channel_title:
        key = f"title:{channel_title.casefold()}"
        basename = channel_title
    else:
        video_id = str(record.get("video_id") or "sin-identidad").strip()
        key = f"video:{video_id}"
        basename = video_id
    file_stem = f"{_safe_channel_slug(basename)}--{sha256_text(key)[:12]}"
    return key, file_stem


def _channel_part_filename(file_stem: str, part: int) -> str:
    if part < 1:
        raise ValueError("La parte de canal debe ser positiva")
    return f"{file_stem}--part-{part:04d}.jsonl"


def _channel_index_payload(
    entries: Iterable[dict[str, Any]],
    *,
    max_channel_file_bytes: int = DEFAULT_MAX_CHANNEL_SHARD_BYTES,
) -> dict[str, Any]:
    ordered = sorted(entries, key=lambda entry: str(entry["file"]))
    return {
        "schema_version": "1.0.0",
        "partition_key": "youtube_channel",
        "format": "jsonl",
        "id_field": "video_id",
        "max_channel_file_bytes": max_channel_file_bytes,
        "total_channel_files": len(ordered),
        "total_channels": len({str(entry["channel_key"]) for entry in ordered}),
        "total_videos": sum(int(entry["videos"]) for entry in ordered),
        "total_bytes": sum(int(entry["bytes"]) for entry in ordered),
        "files": ordered,
    }


def _summarize_channel_shard(
    path: Path,
    *,
    channel_key: str,
    part: int,
) -> dict[str, Any]:
    channel_ids: set[str] = set()
    channel_titles: set[str] = set()
    videos = 0
    for record in read_jsonl(path):
        if not str(record.get("video_id") or "").strip():
            raise ValueError(f"Falta video_id en {path}")
        channel_id, channel_title = _transcript_channel_fields(record)
        if channel_id:
            channel_ids.add(channel_id)
        if channel_title:
            channel_titles.add(channel_title)
        videos += 1
    return {
        "file": path.name,
        "channel_key": channel_key,
        "part": part,
        "channel_ids": sorted(channel_ids),
        "channel_titles": sorted(channel_titles, key=str.casefold),
        "videos": videos,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def materialize_transcripts_by_channel(
    canonical_path: str | Path,
    output_dir: str | Path,
    *,
    max_channel_file_bytes: int = DEFAULT_MAX_CHANNEL_SHARD_BYTES,
) -> dict[str, Any]:
    """Particiona el canónico por canal sin modificarlo ni eliminarlo.

    La materialización completa es atómica por archivo. Los títulos históricos
    sin ``channel_id`` se asocian al ID cuando el título identifica un único
    canal dentro del canónico. El índice es determinista y sirve para verificar
    y restaurar el corpus en otra máquina.
    """

    if max_channel_file_bytes < 1024:
        raise ValueError("max_channel_file_bytes debe ser al menos 1024")
    canonical = Path(canonical_path)
    target = Path(output_dir)
    if not canonical.is_file():
        target.mkdir(parents=True, exist_ok=True)
        payload = _channel_index_payload([], max_channel_file_bytes=max_channel_file_bytes)
        write_json_atomic(target / CHANNEL_TRANSCRIPT_INDEX, payload)
        return {**payload, "canonical_exists": False, "output_dir": target.as_posix()}

    title_ids: dict[str, set[str]] = defaultdict(set)
    for record in read_jsonl(canonical):
        channel_id, channel_title = _transcript_channel_fields(record)
        if channel_id and channel_title:
            title_ids[channel_title.casefold()].add(channel_id)
    title_aliases = {
        title: next(iter(channel_ids))
        for title, channel_ids in title_ids.items()
        if len(channel_ids) == 1
    }

    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    channel_keys: dict[str, str] = {}
    channel_parts: dict[str, int] = defaultdict(lambda: 1)
    channel_part_bytes: dict[tuple[str, int], int] = defaultdict(int)
    file_parts: dict[str, int] = {}
    try:
        for record in read_jsonl(canonical):
            video_id = str(record.get("video_id") or "").strip()
            if not video_id:
                raise ValueError(f"Falta video_id en {canonical}")
            channel_key, file_stem = _channel_shard_descriptor(
                record,
                title_aliases=title_aliases,
            )
            encoded = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            encoded_bytes = len(encoded.encode("utf-8"))
            part = channel_parts[channel_key]
            if (
                channel_part_bytes[(channel_key, part)]
                and channel_part_bytes[(channel_key, part)] + encoded_bytes > max_channel_file_bytes
            ):
                part += 1
                channel_parts[channel_key] = part
            filename = _channel_part_filename(file_stem, part)
            previous_key = channel_keys.setdefault(filename, channel_key)
            if previous_key != channel_key:
                raise ValueError(f"Colisión de particiones para {filename}")
            file_parts[filename] = part
            with (staging / filename).open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
            channel_part_bytes[(channel_key, part)] += encoded_bytes

        entries = [
            _summarize_channel_shard(
                path,
                channel_key=channel_keys[path.name],
                part=file_parts[path.name],
            )
            for path in sorted(staging.glob("*.jsonl"))
        ]
        payload = _channel_index_payload(
            entries,
            max_channel_file_bytes=max_channel_file_bytes,
        )
        new_filenames = {entry["file"] for entry in entries}
        for path in sorted(staging.glob("*.jsonl")):
            os.replace(path, target / path.name)
        write_json_atomic(target / CHANNEL_TRANSCRIPT_INDEX, payload)
        for stale in target.glob("*.jsonl"):
            if stale.name not in new_filenames:
                stale.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        **payload,
        "canonical_exists": True,
        "canonical_bytes": canonical.stat().st_size,
        "canonical_sha256": sha256_file(canonical),
        "output_dir": target.as_posix(),
    }


def append_transcripts_by_channel(
    output_dir: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    max_channel_file_bytes: int | None = None,
) -> dict[str, int]:
    """Añade checkpoints idempotentes al archivo pequeño de cada canal."""

    materialized = list(rows)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    index_path = target / CHANNEL_TRANSCRIPT_INDEX
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    else:
        index = _channel_index_payload([])
    shard_limit = int(
        max_channel_file_bytes
        if max_channel_file_bytes is not None
        else index.get("max_channel_file_bytes", DEFAULT_MAX_CHANNEL_SHARD_BYTES)
    )
    if shard_limit < 1024:
        raise ValueError("max_channel_file_bytes debe ser al menos 1024")
    entries_by_file = {str(entry["file"]): dict(entry) for entry in index.get("files", [])}

    id_keys: dict[str, set[str]] = defaultdict(set)
    title_keys: dict[str, set[str]] = defaultdict(set)
    entries_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for filename, entry in entries_by_file.items():
        channel_key = str(entry["channel_key"])
        entries_by_key[channel_key].append(entry)
        for channel_id in entry.get("channel_ids", []):
            id_keys[str(channel_id)].add(channel_key)
        for title in entry.get("channel_titles", []):
            title_keys[str(title).casefold()].add(channel_key)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_stems: dict[str, str] = {}
    batch_title_ids: dict[str, set[str]] = defaultdict(set)
    for record in materialized:
        channel_id, channel_title = _transcript_channel_fields(record)
        if channel_id and channel_title:
            batch_title_ids[channel_title.casefold()].add(channel_id)
    batch_title_aliases = {
        title: next(iter(channel_ids))
        for title, channel_ids in batch_title_ids.items()
        if len(channel_ids) == 1
    }
    for record in materialized:
        channel_id, channel_title = _transcript_channel_fields(record)
        known_keys = id_keys.get(channel_id, set()) if channel_id else set()
        if len(known_keys) != 1 and channel_title:
            known_keys = title_keys.get(channel_title.casefold(), set())
        if len(known_keys) == 1:
            channel_key = next(iter(known_keys))
            first_filename = str(entries_by_key[channel_key][0]["file"])
            file_stem = re.sub(r"--part-\d{4}\.jsonl$", "", first_filename)
        else:
            channel_key, file_stem = _channel_shard_descriptor(
                record,
                title_aliases=batch_title_aliases,
            )
        grouped[channel_key].append(record)
        file_stems[channel_key] = file_stem

    added = skipped = 0
    touched_files: set[str] = set()
    for channel_key, group in sorted(grouped.items()):
        channel_entries = sorted(
            entries_by_key.get(channel_key, []),
            key=lambda entry: (int(entry.get("part", 1)), str(entry["file"])),
        )
        existing_ids = {
            str(record["video_id"])
            for entry in channel_entries
            for record in read_jsonl(target / str(entry["file"]))
        }
        part = int(channel_entries[-1].get("part", 1)) if channel_entries else 1
        filename = (
            str(channel_entries[-1]["file"])
            if channel_entries
            else _channel_part_filename(file_stems[channel_key], part)
        )
        current_bytes = (target / filename).stat().st_size if (target / filename).is_file() else 0
        pending_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in group:
            video_id = str(record.get("video_id") or "").strip()
            if not video_id:
                raise ValueError("Falta video_id en una transcripción")
            if video_id in existing_ids:
                skipped += 1
                continue
            encoded_bytes = len(
                (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            )
            if current_bytes and current_bytes + encoded_bytes > shard_limit:
                part += 1
                filename = _channel_part_filename(file_stems[channel_key], part)
                current_bytes = 0
            pending_by_file[filename].append(record)
            current_bytes += encoded_bytes
            existing_ids.add(video_id)
            added += 1
        for pending_filename, pending in pending_by_file.items():
            _append_jsonl_checkpoint(target / pending_filename, pending)
            touched_files.add(pending_filename)
            pending_part = int(re.search(r"--part-(\d{4})\.jsonl$", pending_filename).group(1))
            entries_by_file[pending_filename] = _summarize_channel_shard(
                target / pending_filename,
                channel_key=channel_key,
                part=pending_part,
            )
    payload = _channel_index_payload(
        entries_by_file.values(),
        max_channel_file_bytes=shard_limit,
    )
    write_json_atomic(index_path, payload)
    return {
        "added": added,
        "already_partitioned": skipped,
        "channel_files_touched": len(touched_files),
        "total_channel_files": payload["total_channel_files"],
        "total_videos": payload["total_videos"],
    }


def restore_canonical_from_channel_transcripts(
    channel_dir: str | Path,
    canonical_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Restaura o completa el canónico desde las particiones sincronizadas."""

    source = Path(channel_dir)
    index_path = source / CHANNEL_TRANSCRIPT_INDEX
    if not index_path.is_file():
        raise FileNotFoundError(f"Falta el índice de transcripciones: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    canonical = Path(canonical_path)
    existing = processed_video_ids(canonical)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    with canonical.open("a", encoding="utf-8", newline="\n") as handle:
        for entry in index.get("files", []):
            shard = source / str(entry["file"])
            if not shard.is_file():
                raise FileNotFoundError(f"Falta la partición declarada: {shard}")
            if verify_hashes and sha256_file(shard) != entry.get("sha256"):
                raise ValueError(f"SHA-256 inválido para {shard}")
            for record in read_jsonl(shard):
                video_id = str(record.get("video_id") or "").strip()
                if not video_id:
                    raise ValueError(f"Falta video_id en {shard}")
                if video_id in existing:
                    skipped += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                existing.add(video_id)
                added += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "added": added,
        "already_canonical": skipped,
        "canonical_videos": len(existing),
        "canonical_path": canonical.as_posix(),
        "verified": verify_hashes,
    }


def vtt_video_id(path: str | Path) -> str:
    """Extrae el ID de YouTube de ``video_id.idioma.vtt``."""

    name = Path(path).name
    if Path(name).suffix.casefold() != ".vtt":
        raise ValueError(f"No es un archivo VTT: {path}")
    video_id = name.split(".", 1)[0].strip()
    if not video_id:
        raise ValueError(f"El VTT no contiene video_id: {path}")
    return video_id


def _vtt_language(path: str | Path) -> str | None:
    parts = Path(path).name.split(".")
    return parts[1] if len(parts) > 2 and parts[1] else None


def _persist_vtt_bytes(output_dir: str | Path, filename: str, payload: bytes) -> Path:
    """Guarda bytes VTT atómicamente sin sobrescribir una variante diferente."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target = target_dir / safe_name
    payload_sha = hashlib.sha256(payload).hexdigest()
    if target.is_file():
        if sha256_file(target) == payload_sha:
            return target
        target = target.with_name(f"{target.stem}.{payload_sha[:12]}.vtt")
        if target.is_file():
            if sha256_file(target) == payload_sha:
                return target
            raise ValueError(f"Colisión de VTT no resoluble: {target}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def _vtt_backfill_candidate(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source_candidate")
    candidate = dict(source) if isinstance(source, dict) else {}
    for key in (
        "video_id",
        "url",
        "title",
        "channel_id",
        "channel_title",
        "channel",
        "channel_url",
        "categoria_fuente",
        "target_category",
        "sampling_mode",
    ):
        if record.get(key) is not None:
            candidate.setdefault(key, record[key])
    candidate["video_id"] = str(record["video_id"])
    candidate["previous_subtitle_source"] = (
        record.get("subtitle_source") or record.get("fuente_subs")
    )
    return normalize_category_metadata(_json_safe(candidate))


def materialize_vtt_checkpoint(
    project_root: str | Path,
    output_dir: str | Path,
    transcript_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Consolida VTT locales por video y publica índice y cola de faltantes.

    Las fuentes originales permanecen intactas. El checkpoint plano conserva el
    nombre de cada pista y deduplica por bytes; ``channel_key`` usa exactamente
    el descriptor empleado por las particiones JSONL por canal.
    """

    root = Path(project_root).resolve()
    target = Path(output_dir).resolve()
    records: dict[str, dict[str, Any]] = {}
    for raw_record in transcript_records:
        record = normalize_category_metadata(_json_safe(raw_record))
        video_id = str(record.get("video_id") or "").strip()
        if video_id:
            records.setdefault(video_id, record)

    title_ids: dict[str, set[str]] = defaultdict(set)
    for record in records.values():
        channel_id, channel_title = _transcript_channel_fields(record)
        if channel_id and channel_title:
            title_ids[channel_title.casefold()].add(channel_id)
    title_aliases = {
        title: next(iter(channel_ids))
        for title, channel_ids in title_ids.items()
        if len(channel_ids) == 1
    }

    source_paths = [
        path.resolve()
        for path in sorted((root / "datos").rglob("*.vtt"))
        if path.is_file() and target not in path.resolve().parents
    ]
    existing_hashes = {
        path.name: sha256_file(path) for path in target.glob("*.vtt") if path.is_file()
    } if target.is_dir() else {}
    copied_sources: dict[str, list[str]] = defaultdict(list)
    copied = reused = 0
    for source in source_paths:
        source_sha = sha256_file(source)
        destination = _persist_vtt_bytes(target, source.name, source.read_bytes())
        try:
            relative_source = source.relative_to(root).as_posix()
        except ValueError:
            relative_source = source.as_posix()
        copied_sources[destination.name].append(relative_source)
        if existing_hashes.get(destination.name) == source_sha:
            reused += 1
        else:
            copied += 1
            existing_hashes[destination.name] = source_sha

    entries: list[dict[str, Any]] = []
    valid_video_ids: set[str] = set()
    invalid_files = 0
    for path in sorted(target.glob("*.vtt"), key=lambda item: item.name.casefold()):
        payload = path.read_bytes()
        valid_webvtt = payload.lstrip(b"\xef\xbb\xbf").startswith(b"WEBVTT") and b"-->" in payload
        video_id = vtt_video_id(path)
        if valid_webvtt:
            valid_video_ids.add(video_id)
        else:
            invalid_files += 1
        record = records.get(video_id, {"video_id": video_id})
        channel_key, channel_file_stem = _channel_shard_descriptor(
            record,
            title_aliases=title_aliases,
        )
        channel_id, channel_title = _transcript_channel_fields(record)
        entries.append(
            {
                "file": path.name,
                "video_id": video_id,
                "language": _vtt_language(path),
                "origin": (
                    "generated_from_transcript_api"
                    if ".transcript-api." in path.name
                    else "original_ytdlp_vtt"
                ),
                "channel_key": channel_key,
                "channel_file_stem": channel_file_stem,
                "channel_id": channel_id or None,
                "channel_title": channel_title or None,
                "transcript_available": video_id in records,
                "valid_webvtt": valid_webvtt,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_paths": sorted(set(copied_sources.get(path.name, []))),
            }
        )

    missing_ids = sorted(set(records) - valid_video_ids)
    missing = [_vtt_backfill_candidate(records[video_id]) for video_id in missing_ids]
    write_jsonl_atomic(target / VTT_MISSING_MANIFEST, missing)
    payload = {
        "schema_version": "1.0.0",
        "partition_key": "youtube_video",
        "format": "webvtt",
        "id_field": "video_id",
        "total_files": len(entries),
        "total_videos": len({entry["video_id"] for entry in entries}),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "original_vtt_files": sum(entry["origin"] == "original_ytdlp_vtt" for entry in entries),
        "transcript_api_vtt_files": sum(
            entry["origin"] == "generated_from_transcript_api" for entry in entries
        ),
        "invalid_vtt_files": invalid_files,
        "transcript_videos": len(records),
        "transcript_videos_with_vtt": len(set(records) & valid_video_ids),
        "missing_vtt_videos": len(missing),
        "missing_manifest": VTT_MISSING_MANIFEST,
        "files": entries,
    }
    write_json_atomic(target / VTT_CHECKPOINT_INDEX, payload)
    return {
        **payload,
        "source_files": len(source_paths),
        "copied": copied,
        "reused": reused,
        "output_dir": target.as_posix(),
    }


def load_vtt_backfill_candidates(vtt_dir: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(Path(vtt_dir) / VTT_MISSING_MANIFEST))


def discover_candidate_sources(project_root: str | Path) -> list[Path]:
    """Localiza inventarios de candidatos que pueden aportar metadatos a un VTT."""

    root = Path(project_root).resolve() / "datos"
    names = {
        "video_candidates.jsonl",
        "videos_candidatos.csv",
        "directed_candidates_latest.jsonl",
    }
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.name in names),
        key=str,
    )


def recover_transcripts_from_vtt(
    vtt_dir: str | Path,
    *,
    existing_video_ids: Iterable[str] = (),
    candidate_sources: Iterable[str | Path] = (),
    minimum_transcript_characters: int = DEFAULT_MIN_TRANSCRIPT_CHARACTERS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recupera transcripciones completas desde VTT locales que aún no tienen JSON.

    No consulta la red ni modifica los VTT. Cuando existen varias pistas para un
    video, selecciona determinísticamente la de mayor texto útil y conserva la
    lista completa de archivos como procedencia. Los VTT demasiado cortos se
    reportan, pero no se fuerzan al canónico.
    """

    if minimum_transcript_characters < 1:
        raise ValueError("minimum_transcript_characters debe ser positivo")
    source = Path(vtt_dir)
    candidate_paths = [Path(path) for path in candidate_sources]
    known = {str(video_id).strip() for video_id in existing_video_ids if str(video_id).strip()}
    grouped: dict[str, list[Path]] = defaultdict(list)
    if source.is_dir():
        for path in sorted(source.rglob("*.vtt")):
            grouped[vtt_video_id(path)].append(path)

    candidates = merge_candidates(
        *(load_candidates(path) for path in candidate_paths if path.is_file())
    )
    candidate_by_id = {
        str(candidate["video_id"]).strip(): candidate
        for candidate in candidates
        if str(candidate.get("video_id") or "").strip()
    }
    recovered: list[dict[str, Any]] = []
    too_short: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    already_present = 0
    for video_id, paths in sorted(grouped.items()):
        if video_id in known:
            already_present += 1
            continue
        tracks: list[tuple[int, int, str, Path, list[dict[str, Any]]]] = []
        for path in paths:
            try:
                segments = _read_vtt_segments(path)
            except Exception as exc:  # noqa: BLE001
                invalid.append(
                    {"video_id": video_id, "path": path.as_posix(), "error": str(exc)}
                )
                continue
            tracks.append(
                (
                    _transcript_characters(segments),
                    len(segments),
                    path.as_posix(),
                    path,
                    segments,
                )
            )
        if not tracks:
            continue
        characters, _, _, selected_path, segments = max(
            tracks,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if characters < minimum_transcript_characters:
            too_short.append(
                {
                    "video_id": video_id,
                    "characters": characters,
                    "selected_vtt": selected_path.as_posix(),
                }
            )
            continue
        candidate = normalize_category_metadata(dict(candidate_by_id.get(video_id, {})))
        url = str(
            candidate.get("url")
            or candidate.get("source_url")
            or f"https://www.youtube.com/watch?v={video_id}"
        )
        record = {
            "video_id": video_id,
            "url": url,
            "title": candidate.get("title") or candidate.get("video_title"),
            "channel_id": candidate.get("channel_id"),
            "channel": candidate.get("channel") or candidate.get("channel_title"),
            "language": _vtt_language(selected_path),
            "subtitle_source": "recovered-local-vtt",
            "vtt_files": [path.name for path in sorted(paths)],
            "source_vtt": selected_path.resolve().as_posix(),
            "segments": segments,
            "transcript_sha256": sha256_text(
                json.dumps(segments, ensure_ascii=False, sort_keys=True)
            ),
            "acquisition_status": "recovered_local_vtt",
        }
        if candidate:
            record["source_candidate"] = candidate
        recovered.append(_json_safe(record))
        known.add(video_id)

    return recovered, {
        "vtt_directory": source.resolve().as_posix(),
        "vtt_files": sum(len(paths) for paths in grouped.values()),
        "vtt_videos": len(grouped),
        "already_present": already_present,
        "candidate_sources": len(candidate_paths),
        "recovered": len(recovered),
        "recovered_with_candidate_metadata": sum(
            bool(record.get("source_candidate")) for record in recovered
        ),
        "too_short": len(too_short),
        "too_short_records": too_short,
        "invalid_tracks": len(invalid),
        "invalid_track_records": invalid,
        "minimum_transcript_characters": minimum_transcript_characters,
    }


def discover_derived_transcript_sources(project_root: str | Path) -> list[Path]:
    """Localiza datasets/chunks con texto, excluyendo candidatos y fallos."""

    root = Path(project_root).resolve()
    sources: set[Path] = set()
    for relative in (Path("datos/model_ready"), Path("datos/processed")):
        base = root / relative
        if base.is_dir():
            sources.update(path.resolve() for path in base.rglob("*.jsonl") if path.is_file())
    expansion_root = root / "datos" / "ampliacion"
    if expansion_root.is_dir():
        sources.update(
            path.resolve()
            for path in expansion_root.rglob("*.jsonl")
            if path.is_file() and "processed" in path.relative_to(expansion_root).parts
        )
    return sorted(sources, key=str)


def consolidate_available_transcripts(
    project_root: str | Path,
    canonical_path: str | Path,
    *,
    cache_dir: str | Path,
    channel_dir: str | Path,
    vtt_dir: str | Path | None = None,
    candidate_sources: Iterable[str | Path] | None = None,
    minimum_transcript_characters: int = DEFAULT_MIN_TRANSCRIPT_CHARACTERS,
    include_historical_snapshots: bool = True,
    verify_partition_hashes: bool = True,
) -> dict[str, Any]:
    """Consolida fuentes completas con prioridad para el checkpoint sincronizado.

    Las particiones Git son autoritativas cuando un ``video_id`` también aparece
    en una copia histórica local. Los IDs exclusivos del canónico, snapshots y
    caché se preservan. Ninguna fuente se modifica; solo se recompone la vista
    canónica de trabajo de manera atómica.
    """

    root = Path(project_root).resolve()
    canonical = Path(canonical_path)
    existing_records = list(read_jsonl(canonical)) if canonical.is_file() else []
    initial_by_id = {
        str(record["video_id"]): record
        for record in existing_records
        if str(record.get("video_id") or "").strip()
    }
    initial_ids = set(initial_by_id)

    historical_sources = (
        discover_existing_transcript_sources(root, canonical_path=canonical)
        if include_historical_snapshots
        else []
    )
    historical_records: list[dict[str, Any]] = []
    for source in historical_sources:
        for raw_record in read_jsonl(source):
            record = normalize_category_metadata(_json_safe(raw_record))
            video_id = str(record.get("video_id") or "").strip()
            if not video_id:
                continue
            record["video_id"] = video_id
            record.setdefault("acquisition_status", "reused_existing_snapshot")
            record.setdefault("source_snapshot", source.as_posix())
            record.setdefault(
                "transcript_sha256",
                sha256_text(
                    json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
                ),
            )
            historical_records.append(record)

    partitions_path = Path(channel_dir)
    partition_records: list[dict[str, Any]] = []
    partition_index = partitions_path / CHANNEL_TRANSCRIPT_INDEX
    partitions_verified = False
    if partition_index.is_file():
        index = json.loads(partition_index.read_text(encoding="utf-8-sig"))
        for entry in index.get("files", []):
            shard = partitions_path / str(entry["file"])
            if not shard.is_file():
                raise FileNotFoundError(f"Falta la partición declarada: {shard}")
            if verify_partition_hashes and sha256_file(shard) != entry.get("sha256"):
                raise ValueError(f"SHA-256 inválido para {shard}")
            partition_records.extend(read_jsonl(shard))
        partitions_verified = verify_partition_hashes

    cache_records: list[dict[str, Any]] = []
    cache_files = sorted(Path(cache_dir).glob("*.json")) if Path(cache_dir).is_dir() else []
    for path in cache_files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"El caché no contiene un objeto JSON: {path}")
        record = normalize_category_metadata(_json_safe(payload))
        video_id = str(record.get("video_id") or "").strip()
        if not video_id or path.stem != video_id:
            raise ValueError(f"El caché {path} no corresponde a {video_id or 'un video_id vacío'}")
        record["video_id"] = video_id
        record.setdefault("acquisition_status", "reused_existing_cache")
        record.setdefault("source_cache", path.resolve().as_posix())
        record.setdefault(
            "transcript_sha256",
            sha256_text(
                json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
            ),
        )
        cache_records.append(record)

    known_before_vtt = {
        str(record["video_id"])
        for records in (partition_records, existing_records, historical_records, cache_records)
        for record in records
        if record.get("video_id")
    }
    resolved_candidate_sources = (
        discover_candidate_sources(root)
        if candidate_sources is None
        else [Path(path) for path in candidate_sources]
    )
    if vtt_dir is not None:
        vtt_records, vtt_stats = recover_transcripts_from_vtt(
            vtt_dir,
            existing_video_ids=known_before_vtt,
            candidate_sources=resolved_candidate_sources,
            minimum_transcript_characters=minimum_transcript_characters,
        )
    else:
        vtt_records = []
        vtt_stats = {
            "vtt_directory": None,
            "vtt_files": 0,
            "vtt_videos": 0,
            "already_present": 0,
            "candidate_sources": len(resolved_candidate_sources),
            "recovered": 0,
            "recovered_with_candidate_metadata": 0,
            "too_short": 0,
            "too_short_records": [],
            "invalid_tracks": 0,
            "invalid_track_records": [],
            "minimum_transcript_characters": minimum_transcript_characters,
        }

    merged: dict[str, dict[str, Any]] = {}
    for source_records in (
        partition_records,
        existing_records,
        historical_records,
        cache_records,
        vtt_records,
    ):
        for raw_record in source_records:
            record = normalize_category_metadata(_json_safe(raw_record))
            video_id = str(record.get("video_id") or "").strip()
            if video_id:
                record["video_id"] = video_id
                merged.setdefault(video_id, record)
    write_jsonl_atomic(canonical, merged.values())

    historical_ids = {
        str(record["video_id"]) for record in historical_records if record.get("video_id")
    }
    partition_ids = {
        str(record["video_id"]) for record in partition_records if record.get("video_id")
    }
    cache_ids = {str(record["video_id"]) for record in cache_records if record.get("video_id")}
    refreshed = sum(
        video_id in initial_by_id and partition_record != initial_by_id[video_id]
        for video_id, partition_record in {
            str(record["video_id"]): record
            for record in partition_records
            if record.get("video_id")
        }.items()
    )
    return {
        "historical_sources": len(historical_sources),
        "historical_candidate_rows": len(historical_records),
        "historical_added": len(historical_ids - initial_ids),
        "partitions_added": len(partition_ids - initial_ids - historical_ids),
        "partitions_refreshed": refreshed,
        "partitions_verified": partitions_verified,
        "cache_files": len(cache_files),
        "cache_added": len(cache_ids - initial_ids - historical_ids - partition_ids),
        "vtt_recovery": vtt_stats,
        "vtt_added": len(vtt_records),
        "canonical_videos": len(merged),
        "canonical_path": canonical.resolve().as_posix(),
    }


def collect_project_video_inventory(
    project_root: str | Path,
    *,
    canonical_path: str | Path,
    cache_dir: str | Path,
    include_historical_sources: bool = True,
    include_derived_sources: bool = True,
) -> tuple[set[str], dict[str, Any]]:
    """Une videos con transcripción completa o texto derivado disponible.

    Los candidatos y registros de fallo se excluyen deliberadamente: conocer un
    ID por descubrimiento no demuestra que su texto haya sido adquirido.
    """

    root = Path(project_root).resolve()
    canonical_ids = processed_video_ids(canonical_path)
    cache_ids = {
        path.stem for path in Path(cache_dir).glob("*.json") if path.is_file()
    }
    historical_ids: set[str] = set()
    historical_sources = (
        discover_existing_transcript_sources(root, canonical_path=canonical_path)
        if include_historical_sources
        else []
    )
    for source in historical_sources:
        historical_ids.update(
            str(row.get("video_id") or "").strip()
            for row in read_jsonl(source)
            if str(row.get("video_id") or "").strip()
        )

    derived_ids: set[str] = set()
    derived_source_stats: list[dict[str, Any]] = []
    derived_sources = (
        discover_derived_transcript_sources(root) if include_derived_sources else []
    )
    for source in derived_sources:
        source_ids: set[str] = set()
        text_rows = 0
        for row in read_jsonl(source):
            video_id = str(row.get("video_id") or "").strip()
            text_value = str(row.get("text") or "").strip()
            if video_id and text_value:
                source_ids.add(video_id)
                text_rows += 1
        if source_ids:
            derived_ids.update(source_ids)
            derived_source_stats.append(
                {
                    "path": source.relative_to(root).as_posix(),
                    "text_rows": text_rows,
                    "unique_videos": len(source_ids),
                }
            )

    full_transcript_ids = canonical_ids | cache_ids | historical_ids
    known_ids = full_transcript_ids | derived_ids
    summary = {
        "canonical_transcripts": len(canonical_ids),
        "cache_transcripts": len(cache_ids),
        "historical_snapshot_transcripts": len(historical_ids),
        "full_transcripts_union": len(full_transcript_ids),
        "derived_text_videos": len(derived_ids),
        "derived_only_videos": len(derived_ids - full_transcript_ids),
        "known_videos_union": len(known_ids),
        "historical_source_files": len(historical_sources),
        "derived_source_files": len(derived_source_stats),
        "derived_sources": derived_source_stats,
    }
    return known_ids, summary


def _youtube_options(
    *,
    retries: int = 3,
    sleep_min_seconds: float = 1.0,
    sleep_max_seconds: float = 3.0,
    socket_timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    if retries < 0:
        raise ValueError("retries no puede ser negativo")
    if sleep_min_seconds < 0 or sleep_max_seconds < sleep_min_seconds:
        raise ValueError("El intervalo de espera de yt-dlp no es válido")
    if socket_timeout_seconds <= 0:
        raise ValueError("socket_timeout_seconds debe ser positivo")
    return {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "retries": retries,
        "extractor_retries": retries,
        "fragment_retries": retries,
        "sleep_interval": sleep_min_seconds,
        "max_sleep_interval": sleep_max_seconds,
        "sleep_interval_requests": sleep_min_seconds,
        "sleep_interval_subtitles": sleep_min_seconds,
        "socket_timeout": socket_timeout_seconds,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }


def _normalise_channel_videos_url(url: str) -> str:
    base = url.rstrip("/")
    for suffix in ("/videos", "/streams", "/shorts", "/playlists"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return f"{base}/videos"


class _QuietYtDlpLogger:
    """Conserva errores de yt-dlp sin inundar la salida interactiva."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def debug(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    @property
    def last_error(self) -> str | None:
        return self.errors[-1] if self.errors else None


def classify_acquisition_error(error: BaseException) -> str:
    """Reduce errores heterogéneos de yt-dlp a motivos auditables y estables."""

    message = str(error).casefold()
    if "http error 404" in message or "does not have a videos tab" in message:
        return "stale_channel_or_no_videos_tab"
    if "members-only" in message or "join this channel" in message or "miembros" in message:
        return "members_only"
    if "private video" in message or "video unavailable" in message or "not available" in message:
        return "unavailable_or_private"
    if "subtítulos insuficientes" in message or "subtitulos insuficientes" in message:
        return "subtitle_too_short"
    if "no tiene subt" in message or "no subtitles" in message:
        return "no_spanish_subtitles"
    if "sign in to confirm" in message or "not a bot" in message:
        return "access_challenge"
    if "429" in message or "too many requests" in message or "rate limit" in message:
        return "rate_limited"
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "premieres in" in message or "premiere in" in message or "upcoming" in message:
        return "scheduled_or_upcoming"
    return "fetch_error"


def discover_youtube_candidates(
    channel_sources: Iterable[dict[str, Any]],
    search_queries: Iterable[str | dict[str, Any]] = (),
    *,
    max_videos_per_channel: int = 75,
    max_results_per_query: int = 20,
    retries: int = 3,
    sleep_min_seconds: float = 1.0,
    sleep_max_seconds: float = 3.0,
    socket_timeout_seconds: float = 45.0,
    checkpoint_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Descubre metadatos planos con timeout y reanudación por fuente.

    Devuelve ``(candidatos, fallos_de_fuente)``. ``checkpoint_path`` guarda
    atómicamente cada canal o consulta terminada; una ejecución posterior
    reutiliza solo las fuentes cuya identidad y cuota siguen coincidiendo.
    """

    if max_videos_per_channel < 1 or max_results_per_query < 1:
        raise ValueError("Los límites de descubrimiento deben ser positivos")
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[datos] para descubrir videos") from exc

    base_options = _youtube_options(
        retries=retries,
        sleep_min_seconds=sleep_min_seconds,
        sleep_max_seconds=sleep_max_seconds,
        socket_timeout_seconds=socket_timeout_seconds,
    )
    base_options.update({"extract_flat": "in_playlist", "ignoreerrors": True})
    candidates_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpoint: dict[str, Any] = {"schema_version": 1, "sources": {}}
    if checkpoint_target is not None and checkpoint_target.is_file():
        loaded = json.loads(checkpoint_target.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict) or not isinstance(loaded.get("sources"), dict):
            raise ValueError(f"Checkpoint de descubrimiento inválido: {checkpoint_target}")
        checkpoint = loaded

    def notify(
        source: dict[str, Any],
        *,
        status: str,
        found: int,
        resumed: bool = False,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "discovery",
                    "source": source.get("name") or source.get("query"),
                    "source_type": source.get("discovery_type"),
                    "status": status,
                    "found": found,
                    "candidates_unique": len(candidates_by_id),
                    "failures": len(failures),
                    "resumed": resumed,
                }
            )

    def source_key(source_url: str, source: dict[str, Any], limit: int) -> str:
        identity = {
            "url": source_url,
            "source_type": source.get("discovery_type"),
            "limit": limit,
            "name": source.get("name"),
            "query": source.get("query"),
            "categoria_fuente": source.get("categoria_fuente"),
            "target_category": source.get("target_category"),
            "sampling_mode": source.get("sampling_mode"),
            "priority_weight": source.get("priority_weight"),
        }
        return sha256_text(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
        )

    def persist_source(
        key: str,
        source_url: str,
        source: dict[str, Any],
        limit: int,
        status: str,
        source_candidates: list[dict[str, Any]],
        failure: dict[str, Any] | None,
    ) -> None:
        if checkpoint_target is None:
            return
        checkpoint["schema_version"] = 1
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
        checkpoint.setdefault("sources", {})[key] = {
            "source": source.get("name") or source.get("query") or source_url,
            "source_type": source.get("discovery_type"),
            "url": source_url,
            "limit": limit,
            "status": status,
            "candidates": source_candidates,
            "failure": failure,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(checkpoint_target, checkpoint)

    def add_candidate(candidate: dict[str, Any]) -> None:
        video_id = str(candidate.get("video_id") or "").strip()
        if not video_id:
            return
        existing = candidates_by_id.get(video_id)
        if existing is None:
            candidates_by_id[video_id] = candidate
            return
        combined_targets = tuple(
            dict.fromkeys(
                (
                    *_category_tokens(existing.get("target_category")),
                    *_category_tokens(candidate.get("target_category")),
                )
            )
        )
        if combined_targets:
            existing["target_category"] = "|".join(combined_targets)
        if candidate.get("sampling_mode") == "directed":
            existing["sampling_mode"] = "directed"

    def fail_source(
        source_url: str,
        source: dict[str, Any],
        *,
        failure_kind: str,
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        failure = {
            "source": source.get("name") or source.get("query") or source_url,
            "url": source_url,
            "failure_kind": failure_kind,
            "error_type": error_type,
            "message": message[:2000],
        }
        failures.append(failure)
        return failure

    def collect(source_url: str, source: dict[str, Any], limit: int) -> None:
        key = source_key(source_url, source, limit)
        notify(source, status="started", found=0)
        cached = checkpoint.get("sources", {}).get(key)
        if isinstance(cached, dict) and cached.get("status") == "ok":
            source_candidates = [
                row for row in cached.get("candidates", []) if isinstance(row, dict)
            ]
            for candidate in source_candidates:
                add_candidate(candidate)
            failure = cached.get("failure")
            if isinstance(failure, dict):
                failures.append(failure)
            notify(
                source,
                status=str(cached["status"]),
                found=len(source_candidates),
                resumed=True,
            )
            return

        if not source_url:
            message = (
                "El canal no tiene URL"
                if source.get("discovery_type") == "channel"
                else "La consulta está vacía"
            )
            failure = fail_source(
                source_url,
                source,
                failure_kind="invalid_source",
                error_type="ValueError",
                message=message,
            )
            persist_source(key, source_url, source, limit, "failed", [], failure)
            notify(source, status="failed", found=0)
            return

        logger = _QuietYtDlpLogger()
        options = {**base_options, "playlist_items": f"1:{limit}", "logger": logger}
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=False)
        except Exception as exc:
            message = logger.last_error or str(exc)
            failure = fail_source(
                source_url,
                source,
                failure_kind=classify_acquisition_error(RuntimeError(message)),
                error_type=type(exc).__name__,
                message=message,
            )
            persist_source(key, source_url, source, limit, "failed", [], failure)
            notify(source, status="failed", found=0)
            return
        if not info:
            message = logger.last_error or "yt-dlp no devolvió entradas"
            failure_kind = (
                classify_acquisition_error(RuntimeError(message))
                if logger.last_error
                else "empty_discovery"
            )
            failure = fail_source(
                source_url,
                source,
                failure_kind=failure_kind,
                error_type="EmptyDiscovery",
                message=message,
            )
            persist_source(key, source_url, source, limit, "failed", [], failure)
            notify(source, status="failed", found=0)
            return
        source_candidates: list[dict[str, Any]] = []
        for rank, item in enumerate(info.get("entries", []) or [], start=1):
            if not item or not item.get("id"):
                continue
            video_id = str(item["id"]).strip()
            candidate = {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": item.get("title"),
                "channel_id": item.get("channel_id") or source.get("channel_id"),
                "channel_title": item.get("channel") or source.get("name"),
                "discovery_type": source["discovery_type"],
                "discovery_source": source.get("name") or source.get("query"),
                "discovery_rank": rank,
            }
            for metadata_key in (
                "categoria_fuente",
                "target_category",
                "reason",
                "sampling_mode",
                "priority_weight",
            ):
                if source.get(metadata_key) is not None:
                    candidate[metadata_key] = source[metadata_key]
            source_candidates.append(candidate)
            add_candidate(candidate)
        if source_candidates:
            persist_source(
                key, source_url, source, limit, "ok", source_candidates, None
            )
            notify(source, status="ok", found=len(source_candidates))
        else:
            failure = fail_source(
                source_url,
                source,
                failure_kind="empty_discovery",
                error_type="EmptyDiscovery",
                message="La fuente no devolvió videos identificables",
            )
            persist_source(key, source_url, source, limit, "failed", [], failure)
            notify(source, status="failed", found=0)

    for raw_source in channel_sources:
        source = dict(raw_source)
        source["discovery_type"] = "channel"
        url = str(source.get("url", "")).strip()
        quota = min(max_videos_per_channel, int(source.get("quota", max_videos_per_channel)))
        collect(_normalise_channel_videos_url(url) if url else "", source, quota)

    for raw_query in search_queries:
        source = {"query": raw_query} if isinstance(raw_query, str) else dict(raw_query)
        source["discovery_type"] = "search"
        query = str(source.get("query", "")).strip()
        quota = min(max_results_per_query, int(source.get("quota", max_results_per_query)))
        collect(f"ytsearch{quota}:{query}" if query else "", source, quota)

    return list(candidates_by_id.values()), failures


_VTT_TIMING = re.compile(
    r"(?P<start>(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{2,}:)?\d{2}:\d{2}[.,]\d{3})"
)


def _vtt_seconds(value: str) -> float:
    parts = [float(part) for part in value.replace(",", ".").split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    return hours * 3600 + minutes * 60 + seconds


def _vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _segments_to_vtt_bytes(segments: Iterable[dict[str, Any]]) -> bytes:
    """Serializa la respuesta cruda de transcript-api con procedencia explícita."""

    lines = ["WEBVTT", "NOTE generated from youtube-transcript-api; not an original yt-dlp file", ""]
    cue = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        cue += 1
        start = max(float(segment.get("start", 0.0)), 0.0)
        duration = max(float(segment.get("duration", 0.0)), 0.001)
        lines.extend(
            [
                str(cue),
                f"{_vtt_timestamp(start)} --> {_vtt_timestamp(start + duration)}",
                text.replace("\r", " ").replace("\n", " "),
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def _read_vtt_segments(path: Path) -> list[dict[str, Any]]:
    """Lee el VTT producido por yt-dlp con el contrato del cuaderno histórico."""

    content = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    segments: list[dict[str, Any]] = []
    seen: set[tuple[float, str]] = set()
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next(
            (index for index, line in enumerate(lines) if _VTT_TIMING.search(line)),
            None,
        )
        if timing_index is None:
            continue
        timing = _VTT_TIMING.search(lines[timing_index])
        if timing is None:
            continue
        start = _vtt_seconds(timing.group("start"))
        end = _vtt_seconds(timing.group("end"))
        text = " ".join(lines[timing_index + 1 :])
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
        text = re.sub(r"\s+", " ", text).strip()
        key = (round(start, 1), text.casefold())
        if text and key not in seen:
            seen.add(key)
            segments.append(
                {"start": start, "duration": max(end - start, 0.1), "text": text}
            )
    return segments


def _transcript_characters(segments: Iterable[dict[str, Any]]) -> int:
    text = " ".join(str(segment.get("text") or "").strip() for segment in segments)
    return len(text.strip())


def _transcript_api_fallback(
    video_id: str,
    languages: tuple[str, ...],
) -> tuple[list[dict[str, Any]], str | None, str]:
    """Último respaldo histórico; no usa API key ni descarga audio/video."""

    from requests import Session
    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

    session = Session()
    session.headers.update(_youtube_options()["http_headers"])
    transcript_list = YouTubeTranscriptApi(http_client=session).list(video_id)
    try:
        transcript = transcript_list.find_manually_created_transcript(list(languages))
    except NoTranscriptFound:
        transcript = transcript_list.find_generated_transcript(list(languages))
    segments = [
        {
            "start": float(row.get("start", 0.0)),
            "duration": max(float(row.get("duration", 0.0)), 0.0),
            "text": str(row.get("text") or "").strip(),
        }
        for row in transcript.fetch().to_raw_data()
        if str(row.get("text") or "").strip()
    ]
    source = "automatic-transcript-api" if transcript.is_generated else "manual-transcript-api"
    return segments, transcript.language_code, source


def fetch_youtube_subtitles(
    candidate: dict[str, Any],
    *,
    languages: Iterable[str] = DEFAULT_SUBTITLE_LANGUAGES,
    retries: int = 3,
    sleep_min_seconds: float = 1.0,
    sleep_max_seconds: float = 3.0,
    socket_timeout_seconds: float = 45.0,
    minimum_transcript_characters: int = DEFAULT_MIN_TRANSCRIPT_CHARACTERS,
    use_transcript_api_fallback: bool = True,
    vtt_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Descarga subtítulos y conserva todas las pistas VTT si se solicita."""

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[datos] para adquirir subtítulos nuevos") from exc
    video_id = str(candidate["video_id"])
    url = str(candidate.get("url") or f"https://www.youtube.com/watch?v={video_id}")
    language_priority = tuple(dict.fromkeys(str(value) for value in languages if str(value).strip()))
    if not language_priority:
        raise ValueError("Se requiere al menos un idioma de subtítulos")
    if minimum_transcript_characters < 1:
        raise ValueError("minimum_transcript_characters debe ser positivo")

    info: dict[str, Any] = {}
    primary_error: BaseException | None = None
    best_segments: list[dict[str, Any]] = []
    selected_language: str | None = None
    subtitle_source = "yt-dlp-vtt"
    downloaded_vtt_tracks: list[tuple[str, bytes]] = []
    with tempfile.TemporaryDirectory(prefix="moderacion_peru_subtitles_") as temp_dir:
        options = _youtube_options(
            retries=retries,
            sleep_min_seconds=sleep_min_seconds,
            sleep_max_seconds=sleep_max_seconds,
            socket_timeout_seconds=socket_timeout_seconds,
        )
        options.update(
            {
                "noplaylist": True,
                "logger": _QuietYtDlpLogger(),
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": list(language_priority),
                "subtitlesformat": "vtt",
                "outtmpl": str(Path(temp_dir) / f"{video_id}.%(ext)s"),
            }
        )
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True) or {}
        except Exception as exc:
            if classify_acquisition_error(exc) == "rate_limited":
                raise
            primary_error = exc

        scored_tracks: list[tuple[int, list[dict[str, Any]], Path]] = []
        for path in Path(temp_dir).glob("*.vtt"):
            segments = _read_vtt_segments(path)
            scored_tracks.append((_transcript_characters(segments), segments, path))
            downloaded_vtt_tracks.append((path.name, path.read_bytes()))
        if scored_tracks:
            _, best_segments, best_path = max(scored_tracks, key=lambda item: item[0])
            selected_language = next(
                (
                    language
                    for language in language_priority
                    if f".{language}." in best_path.name
                ),
                None,
            )
            manual_tracks = info.get("subtitles", {}) or {}
            automatic_tracks = info.get("automatic_captions", {}) or {}
            if selected_language and manual_tracks.get(selected_language):
                subtitle_source = "manual-yt-dlp-vtt"
            elif selected_language and automatic_tracks.get(selected_language):
                subtitle_source = "automatic-yt-dlp-vtt"

    fallback_error: BaseException | None = None
    if (
        use_transcript_api_fallback
        and _transcript_characters(best_segments) < minimum_transcript_characters
    ):
        try:
            fallback_segments, fallback_language, fallback_source = _transcript_api_fallback(
                video_id, language_priority
            )
            if _transcript_characters(fallback_segments) > _transcript_characters(best_segments):
                best_segments = fallback_segments
                selected_language = fallback_language
                subtitle_source = fallback_source
        except Exception as exc:
            if classify_acquisition_error(exc) == "rate_limited":
                raise
            fallback_error = exc

    character_count = _transcript_characters(best_segments)
    if character_count < minimum_transcript_characters:
        if primary_error is not None:
            detail = f"; fallback: {fallback_error}" if fallback_error is not None else ""
            raise RuntimeError(f"yt-dlp no pudo obtener subtítulos de {video_id}: {primary_error}{detail}")
        if character_count:
            raise RuntimeError(
                f"{video_id} devolvió subtítulos insuficientes: "
                f"{character_count} < {minimum_transcript_characters} caracteres"
            )
        raise RuntimeError(f"{video_id} no tiene subtítulos en los idiomas {language_priority}")

    persisted_vtt_files: list[str] = []
    if vtt_output_dir is not None:
        for filename, payload in downloaded_vtt_tracks:
            persisted = _persist_vtt_bytes(vtt_output_dir, filename, payload)
            persisted_vtt_files.append(persisted.name)
        if "transcript-api" in subtitle_source:
            language = selected_language or "es"
            generated_name = f"{video_id}.{language}.transcript-api.vtt"
            persisted = _persist_vtt_bytes(
                vtt_output_dir,
                generated_name,
                _segments_to_vtt_bytes(best_segments),
            )
            persisted_vtt_files.append(persisted.name)

    return {
        "video_id": video_id,
        "url": url,
        "title": info.get("title") or candidate.get("title"),
        "channel_id": info.get("channel_id") or candidate.get("channel_id"),
        "channel": info.get("channel") or candidate.get("channel_title"),
        "language": selected_language,
        "subtitle_source": subtitle_source,
        "vtt_files": sorted(set(persisted_vtt_files)),
        "segments": best_segments,
    }


def processed_video_ids(canonical_path: str | Path) -> set[str]:
    return (
        {str(row["video_id"]) for row in read_jsonl(canonical_path)}
        if Path(canonical_path).exists()
        else set()
    )


def cached_transcript(cache_dir: str | Path, video_id: str) -> dict[str, Any] | None:
    path = Path(cache_dir) / f"{video_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if str(payload.get("video_id")) != video_id:
        raise ValueError(f"El caché {path} no corresponde a {video_id}")
    payload["acquisition_status"] = "reused_cache"
    return payload


def _failure_record(candidate: dict[str, Any], error: BaseException) -> dict[str, Any]:
    video_id = str(candidate.get("video_id", "")).strip()
    failure_kind = classify_acquisition_error(error)
    return {
        "failure_id": sha256_text(f"{video_id}\0{failure_kind}"),
        "video_id": video_id,
        "url": candidate.get("url"),
        "channel_title": candidate.get("channel_title") or candidate.get("channel"),
        "failure_kind": failure_kind,
        "error_type": type(error).__name__,
        "message": str(error)[:2000],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _candidate_channel_key(candidate: dict[str, Any]) -> str | None:
    channel_id = str(candidate.get("channel_id") or "").strip()
    if channel_id:
        return f"id:{channel_id}"
    channel_title = str(
        candidate.get("channel_title") or candidate.get("channel") or ""
    ).strip()
    if channel_title:
        return f"title:{channel_title.casefold()}"
    channel_url = str(candidate.get("channel_url") or "").strip()
    return f"url:{channel_url.casefold()}" if channel_url else None


def order_candidates_for_acquisition(
    candidates: Iterable[dict[str, Any]],
    *,
    random_seed: int | str = 20260806,
) -> list[dict[str, Any]]:
    """Orden pseudoaleatorio estable, intercalando un video por canal."""

    seed = str(random_seed).strip()
    if not seed:
        raise ValueError("random_seed no puede estar vacío")
    unique: dict[str, dict[str, Any]] = {}
    for raw in candidates:
        video_id = str(raw.get("video_id") or "").strip()
        if video_id:
            unique.setdefault(video_id, dict(raw))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video_id, candidate in unique.items():
        channel_key = _candidate_channel_key(candidate) or f"video:{video_id}"
        buckets[channel_key].append(candidate)
    for group in buckets.values():
        group.sort(
            key=lambda row: sha256_text(f"{seed}\0video\0{row['video_id']}")
        )

    channels = deque(
        sorted(
            buckets,
            key=lambda channel_key: sha256_text(f"{seed}\0channel\0{channel_key}"),
        )
    )
    ordered: list[dict[str, Any]] = []
    while channels:
        channel_key = channels.popleft()
        candidate = buckets[channel_key].pop(0)
        ordered.append(
            {
                **candidate,
                "download_queue_rank": len(ordered) + 1,
                "download_queue_seed": seed,
            }
        )
        if buckets[channel_key]:
            channels.append(channel_key)
    return ordered


def _append_jsonl_checkpoint(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Anexa filas ya deduplicadas y fuerza su persistencia antes de continuar."""

    materialized = list(rows)
    if not materialized:
        return 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return len(materialized)


def ingest_incremental(
    candidates: Iterable[dict[str, Any]],
    canonical_path: str | Path,
    cache_dir: str | Path,
    *,
    fetcher: TranscriptFetcher | None = None,
    failure_path: str | Path | None = None,
    max_new_videos: int | None = None,
    network_batch_size: int | None = None,
    batch_pause_seconds: float = 0.0,
    stop_on_error: bool = False,
    exclude_rate_limited_channels: bool = True,
    progress_callback: ProgressCallback | None = None,
    channel_transcript_dir: str | Path | None = None,
) -> dict[str, int]:
    """Reutiliza corpus/caché y aísla los fallos de cada video nuevo.

    ``max_new_videos=None`` incluye toda la cola. ``network_batch_size`` agrega
    una pausa entre lotes de llamadas nuevas sin afectar la reutilización de
    caché. Con ``stop_on_error=False`` (predeterminado), un video inaccesible se
    registra y no detiene los candidatos posteriores.
    """

    if max_new_videos is not None and max_new_videos < 0:
        raise ValueError("max_new_videos no puede ser negativo")
    if network_batch_size is not None and network_batch_size < 1:
        raise ValueError("network_batch_size debe ser positivo o None")
    if batch_pause_seconds < 0:
        raise ValueError("batch_pause_seconds no puede ser negativo")
    canonical = Path(canonical_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    processed = processed_video_ids(canonical)
    output: list[dict[str, Any]] = []
    checkpoint_size = network_batch_size or 50
    failure_ids = (
        {str(row.get("failure_id")) for row in read_jsonl(failure_path)}
        if failure_path is not None and Path(failure_path).exists()
        else set()
    )
    fetch_attempts = 0
    counters = {
        "already_canonical": 0,
        "reused_cache": 0,
        "fetch_attempted": 0,
        "fetched": 0,
        "failed": 0,
        "deferred_by_limit": 0,
        "deferred_rate_limit": 0,
        "rate_limited_channels": 0,
        "batch_pauses": 0,
        "unavailable": 0,
        "added": 0,
        "skipped_duplicate": 0,
        "failure_records_added": 0,
        "failure_records_existing": 0,
        "channel_shard_added": 0,
        "channel_shard_existing": 0,
        "channel_shards_touched": 0,
    }
    rate_limited_channels: set[str] = set()

    def flush_output() -> None:
        if not output:
            return
        if channel_transcript_dir is not None:
            shard_stats = append_transcripts_by_channel(channel_transcript_dir, output)
            counters["channel_shard_added"] += shard_stats["added"]
            counters["channel_shard_existing"] += shard_stats["already_partitioned"]
            counters["channel_shards_touched"] += shard_stats["channel_files_touched"]
        counters["added"] += _append_jsonl_checkpoint(canonical, output)
        output.clear()

    def notify(video_id: str, status: str, *, advance: int = 1) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "acquisition",
                    "video_id": video_id,
                    "status": status,
                    "advance": advance,
                    "counters": dict(counters),
                }
            )

    for candidate in candidates:
        video_id = str(candidate.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("Cada candidato requiere video_id")
        if video_id in processed:
            counters["already_canonical"] += 1
            notify(video_id, "already_canonical")
            continue
        record = cached_transcript(cache, video_id)
        if record is not None:
            counters["reused_cache"] += 1
            completion_status = "reused_cache"
        elif fetcher is not None:
            channel_key = _candidate_channel_key(candidate)
            if channel_key is not None and channel_key in rate_limited_channels:
                counters["deferred_rate_limit"] += 1
                notify(video_id, "deferred_rate_limited_channel")
                continue
            if max_new_videos is not None and fetch_attempts >= max_new_videos:
                counters["deferred_by_limit"] += 1
                notify(video_id, "deferred_by_limit")
                continue
            if (
                network_batch_size is not None
                and fetch_attempts
                and fetch_attempts % network_batch_size == 0
            ):
                flush_output()
                counters["batch_pauses"] += 1
                notify(video_id, "batch_pause", advance=0)
                if batch_pause_seconds:
                    time.sleep(batch_pause_seconds)
            fetch_attempts += 1
            counters["fetch_attempted"] += 1
            try:
                record = fetcher(candidate)
            except Exception as exc:
                counters["failed"] += 1
                failure = _failure_record(candidate, exc)
                if failure_path is not None:
                    if failure["failure_id"] in failure_ids:
                        counters["failure_records_existing"] += 1
                    else:
                        _append_jsonl_checkpoint(failure_path, [failure])
                        failure_ids.add(failure["failure_id"])
                        counters["failure_records_added"] += 1
                failure_counter = f"failure_{failure['failure_kind']}"
                counters[failure_counter] = counters.get(failure_counter, 0) + 1
                if (
                    exclude_rate_limited_channels
                    and failure["failure_kind"] == "rate_limited"
                    and channel_key is not None
                ):
                    rate_limited_channels.add(channel_key)
                    counters["rate_limited_channels"] = len(rate_limited_channels)
                if stop_on_error:
                    raise
                notify(video_id, "failed")
                continue
            record["video_id"] = video_id
            record["acquisition_status"] = "fetched_new"
            write_json_atomic(cache / f"{video_id}.json", record)
            counters["fetched"] += 1
            completion_status = "fetched"
        else:
            counters["unavailable"] += 1
            notify(video_id, "network_disabled")
            continue
        record = normalize_category_metadata(record)
        record["source_candidate"] = normalize_category_metadata(candidate)
        record["transcript_sha256"] = sha256_text(
            json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
        )
        output.append(record)
        processed.add(video_id)
        if len(output) >= checkpoint_size:
            flush_output()
        notify(video_id, completion_status)
    flush_output()
    return counters


def backfill_missing_vtt(
    candidates: Iterable[dict[str, Any]],
    vtt_dir: str | Path,
    *,
    fetcher: TranscriptFetcher | None,
    failure_path: str | Path | None = None,
    max_new_videos: int | None = None,
    network_batch_size: int | None = None,
    batch_pause_seconds: float = 0.0,
    stop_on_error: bool = False,
    exclude_rate_limited_channels: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Rellena VTT faltantes aunque su transcripción JSON ya sea canónica."""

    if max_new_videos is not None and max_new_videos < 0:
        raise ValueError("max_new_videos no puede ser negativo")
    if network_batch_size is not None and network_batch_size < 1:
        raise ValueError("network_batch_size debe ser positivo o None")
    if batch_pause_seconds < 0:
        raise ValueError("batch_pause_seconds no puede ser negativo")
    target = Path(vtt_dir)
    target.mkdir(parents=True, exist_ok=True)
    available = {
        vtt_video_id(path)
        for path in target.glob("*.vtt")
        if path.is_file()
        and path.read_bytes().lstrip(b"\xef\xbb\xbf").startswith(b"WEBVTT")
        and b"-->" in path.read_bytes()
    }
    failure_ids = (
        {str(row.get("failure_id")) for row in read_jsonl(failure_path)}
        if failure_path is not None and Path(failure_path).exists()
        else set()
    )
    attempts = 0
    rate_limited_channels: set[str] = set()
    counters = {
        "already_available": 0,
        "fetch_attempted": 0,
        "fetched": 0,
        "failed": 0,
        "deferred_by_limit": 0,
        "deferred_rate_limit": 0,
        "rate_limited_channels": 0,
        "batch_pauses": 0,
        "network_disabled": 0,
        "failure_records_added": 0,
        "failure_records_existing": 0,
    }

    def notify(video_id: str, status: str, *, advance: int = 1) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "vtt_backfill",
                    "video_id": video_id,
                    "status": status,
                    "advance": advance,
                    "counters": dict(counters),
                }
            )

    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        video_id = str(candidate.get("video_id") or "").strip()
        if not video_id:
            raise ValueError("Cada candidato de backfill requiere video_id")
        if video_id in available:
            counters["already_available"] += 1
            notify(video_id, "already_available")
            continue
        channel_key = _candidate_channel_key(candidate)
        if channel_key is not None and channel_key in rate_limited_channels:
            counters["deferred_rate_limit"] += 1
            notify(video_id, "deferred_rate_limited_channel")
            continue
        if fetcher is None:
            counters["network_disabled"] += 1
            notify(video_id, "network_disabled")
            continue
        if max_new_videos is not None and attempts >= max_new_videos:
            counters["deferred_by_limit"] += 1
            notify(video_id, "deferred_by_limit")
            continue
        if (
            network_batch_size is not None
            and attempts
            and attempts % network_batch_size == 0
        ):
            counters["batch_pauses"] += 1
            notify(video_id, "batch_pause", advance=0)
            if batch_pause_seconds:
                time.sleep(batch_pause_seconds)
        attempts += 1
        counters["fetch_attempted"] += 1
        try:
            fetcher(candidate)
            persisted = [
                path
                for path in target.glob(f"{video_id}.*.vtt")
                if path.is_file()
                and path.read_bytes().lstrip(b"\xef\xbb\xbf").startswith(b"WEBVTT")
                and b"-->" in path.read_bytes()
            ]
            if not persisted:
                raise RuntimeError(f"La descarga de {video_id} no conservó ningún VTT válido")
        except Exception as exc:
            counters["failed"] += 1
            failure = _failure_record(candidate, exc)
            if failure_path is not None:
                if failure["failure_id"] in failure_ids:
                    counters["failure_records_existing"] += 1
                else:
                    _append_jsonl_checkpoint(failure_path, [failure])
                    failure_ids.add(failure["failure_id"])
                    counters["failure_records_added"] += 1
            failure_counter = f"failure_{failure['failure_kind']}"
            counters[failure_counter] = counters.get(failure_counter, 0) + 1
            if (
                exclude_rate_limited_channels
                and failure["failure_kind"] == "rate_limited"
                and channel_key is not None
            ):
                rate_limited_channels.add(channel_key)
                counters["rate_limited_channels"] = len(rate_limited_channels)
            if stop_on_error:
                raise
            notify(video_id, "failed")
            continue
        available.add(video_id)
        counters["fetched"] += 1
        notify(video_id, "fetched")
    return counters
