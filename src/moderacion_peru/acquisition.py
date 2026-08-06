from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import append_jsonl_once, read_jsonl, sha256_text, write_json_atomic


TranscriptFetcher = Callable[[dict[str, Any]], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]
DEFAULT_SUBTITLE_LANGUAGES = ("es-PE", "es-419", "es")
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
    max_candidates: int = 500,
) -> list[dict[str, Any]]:
    """Crea una cohorte inédita con round-robin ponderado por déficit."""

    if max_candidates < 0:
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
    while active and len(selected) < max_candidates:
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


def _youtube_options(
    *,
    retries: int = 3,
    sleep_min_seconds: float = 1.0,
    sleep_max_seconds: float = 3.0,
) -> dict[str, Any]:
    if retries < 0:
        raise ValueError("retries no puede ser negativo")
    if sleep_min_seconds < 0 or sleep_max_seconds < sleep_min_seconds:
        raise ValueError("El intervalo de espera de yt-dlp no es válido")
    return {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "retries": retries,
        "extractor_retries": retries,
        "sleep_interval": sleep_min_seconds,
        "max_sleep_interval": sleep_max_seconds,
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
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Descubre metadatos planos sin descargar audio, video ni subtítulos.

    Devuelve ``(candidatos, fallos_de_fuente)``. Los candidatos se deduplican
    por ``video_id`` conservando la primera fuente.
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
    )
    base_options.update({"extract_flat": "in_playlist", "ignoreerrors": True})
    candidates_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    def notify(source: dict[str, Any], *, status: str, found: int) -> None:
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
                }
            )

    def fail_source(
        source_url: str,
        source: dict[str, Any],
        *,
        failure_kind: str,
        error_type: str,
        message: str,
    ) -> None:
        failures.append(
            {
                "source": source.get("name") or source.get("query") or source_url,
                "url": source_url,
                "failure_kind": failure_kind,
                "error_type": error_type,
                "message": message[:2000],
            }
        )
        notify(source, status="failed", found=0)

    def collect(source_url: str, source: dict[str, Any], limit: int) -> None:
        logger = _QuietYtDlpLogger()
        options = {**base_options, "playlist_items": f"1:{limit}", "logger": logger}
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=False)
        except Exception as exc:
            message = logger.last_error or str(exc)
            fail_source(
                source_url,
                source,
                failure_kind=classify_acquisition_error(RuntimeError(message)),
                error_type=type(exc).__name__,
                message=message,
            )
            return
        if not info:
            message = logger.last_error or "yt-dlp no devolvió entradas"
            failure_kind = (
                classify_acquisition_error(RuntimeError(message))
                if logger.last_error
                else "empty_discovery"
            )
            fail_source(
                source_url,
                source,
                failure_kind=failure_kind,
                error_type="EmptyDiscovery",
                message=message,
            )
            return
        found = 0
        for rank, item in enumerate(info.get("entries", []) or [], start=1):
            if not item or not item.get("id"):
                continue
            found += 1
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
            for key in (
                "categoria_fuente",
                "target_category",
                "reason",
                "sampling_mode",
                "priority_weight",
            ):
                if source.get(key) is not None:
                    candidate[key] = source[key]
            existing = candidates_by_id.get(video_id)
            if existing is None:
                candidates_by_id[video_id] = candidate
            else:
                combined_targets = tuple(
                    dict.fromkeys(
                        (*_category_tokens(existing.get("target_category")), *_category_tokens(candidate.get("target_category")))
                    )
                )
                if combined_targets:
                    existing["target_category"] = "|".join(combined_targets)
                if candidate.get("sampling_mode") == "directed":
                    existing["sampling_mode"] = "directed"
        if found:
            notify(source, status="ok", found=found)
        else:
            fail_source(
                source_url,
                source,
                failure_kind="empty_discovery",
                error_type="EmptyDiscovery",
                message="La fuente no devolvió videos identificables",
            )

    for raw_source in channel_sources:
        source = dict(raw_source)
        url = str(source.get("url", "")).strip()
        if not url:
            source["discovery_type"] = "channel"
            fail_source(
                "",
                source,
                failure_kind="invalid_source",
                error_type="ValueError",
                message="El canal no tiene URL",
            )
            continue
        source["discovery_type"] = "channel"
        quota = min(max_videos_per_channel, int(source.get("quota", max_videos_per_channel)))
        collect(_normalise_channel_videos_url(url), source, quota)

    for raw_query in search_queries:
        source = {"query": raw_query} if isinstance(raw_query, str) else dict(raw_query)
        query = str(source.get("query", "")).strip()
        if not query:
            source["discovery_type"] = "search"
            fail_source(
                "",
                source,
                failure_kind="invalid_source",
                error_type="ValueError",
                message="La consulta está vacía",
            )
            continue
        source["discovery_type"] = "search"
        quota = min(max_results_per_query, int(source.get("quota", max_results_per_query)))
        collect(f"ytsearch{quota}:{query}", source, quota)

    return list(candidates_by_id.values()), failures


def fetch_youtube_subtitles(
    candidate: dict[str, Any],
    *,
    languages: Iterable[str] = DEFAULT_SUBTITLE_LANGUAGES,
    retries: int = 3,
    sleep_min_seconds: float = 1.0,
    sleep_max_seconds: float = 3.0,
) -> dict[str, Any]:
    """Descarga únicamente subtítulos; nunca descarga el video ni el audio."""

    try:
        import requests
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[datos] para adquirir subtítulos nuevos") from exc
    video_id = str(candidate["video_id"])
    url = str(candidate.get("url") or f"https://www.youtube.com/watch?v={video_id}")
    language_priority = tuple(dict.fromkeys(str(value) for value in languages if str(value).strip()))
    if not language_priority:
        raise ValueError("Se requiere al menos un idioma de subtítulos")
    options = _youtube_options(
        retries=retries,
        sleep_min_seconds=sleep_min_seconds,
        sleep_max_seconds=sleep_max_seconds,
    )
    options["noplaylist"] = True
    options["logger"] = _QuietYtDlpLogger()
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    tracks = info.get("subtitles", {}) or {}
    automatic = info.get("automatic_captions", {}) or {}
    manual_language = next((language for language in language_priority if tracks.get(language)), None)
    automatic_language = next(
        (language for language in language_priority if automatic.get(language)), None
    )
    selected_language = manual_language or automatic_language
    variants = tracks.get(selected_language) if manual_language else automatic.get(selected_language)
    if not variants:
        raise RuntimeError(f"{video_id} no tiene subtítulos en los idiomas {language_priority}")
    selected = next((item for item in variants if item.get("ext") == "json3"), variants[0])
    response = None
    for attempt in range(retries + 1):
        if sleep_max_seconds:
            time.sleep(random.uniform(sleep_min_seconds, sleep_max_seconds))
        response = requests.get(selected["url"], timeout=60)
        retryable = response.status_code == 429 or 500 <= response.status_code < 600
        if not retryable or attempt >= retries:
            response.raise_for_status()
            break
        retry_after = response.headers.get("Retry-After")
        try:
            backoff = float(retry_after) if retry_after else max(sleep_min_seconds, 1.0) * (2**attempt)
        except ValueError:
            backoff = max(sleep_min_seconds, 1.0) * (2**attempt)
        time.sleep(min(backoff, 60.0))
    if response is None:
        raise RuntimeError(f"No se obtuvo la pista de subtítulos para {video_id}")
    if selected.get("ext") != "json3":
        raise RuntimeError(
            f"Formato de subtítulo no compatible para {video_id}: {selected.get('ext')}"
        )
    payload = response.json()
    segments = []
    for event in payload.get("events", []):
        text = "".join(segment.get("utf8", "") for segment in event.get("segs", []))
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(event.get("tStartMs", 0)) / 1000,
                "duration": float(event.get("dDurationMs", 0)) / 1000,
                "text": text,
            }
        )
    if not segments:
        raise RuntimeError(f"{video_id} devolvió subtítulos vacíos")
    return {
        "video_id": video_id,
        "url": url,
        "title": info.get("title"),
        "channel_id": info.get("channel_id"),
        "channel": info.get("channel"),
        "language": selected_language,
        "subtitle_source": "manual" if manual_language else "automatic",
        "segments": segments,
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


def ingest_incremental(
    candidates: Iterable[dict[str, Any]],
    canonical_path: str | Path,
    cache_dir: str | Path,
    *,
    fetcher: TranscriptFetcher | None = None,
    failure_path: str | Path | None = None,
    max_new_videos: int | None = None,
    stop_on_error: bool = False,
    halt_on_rate_limit: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Reutiliza corpus/caché y aísla los fallos de cada video nuevo.

    ``max_new_videos`` limita las llamadas de red al ``fetcher``; no limita la
    reutilización de caché. Con ``stop_on_error=False`` (predeterminado), un
    video inaccesible se registra y no detiene los candidatos posteriores.
    """

    if max_new_videos is not None and max_new_videos < 0:
        raise ValueError("max_new_videos no puede ser negativo")
    canonical = Path(canonical_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    processed = processed_video_ids(canonical)
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fetch_attempts = 0
    counters = {
        "already_canonical": 0,
        "reused_cache": 0,
        "fetch_attempted": 0,
        "fetched": 0,
        "failed": 0,
        "deferred_by_limit": 0,
        "deferred_rate_limit": 0,
        "rate_limit_circuit_open": 0,
        "unavailable": 0,
    }
    rate_limit_open = False

    def notify(video_id: str, status: str) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "acquisition",
                    "video_id": video_id,
                    "status": status,
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
            if rate_limit_open:
                counters["deferred_rate_limit"] += 1
                notify(video_id, "deferred_rate_limit")
                continue
            if max_new_videos is not None and fetch_attempts >= max_new_videos:
                counters["deferred_by_limit"] += 1
                notify(video_id, "deferred_by_limit")
                continue
            fetch_attempts += 1
            counters["fetch_attempted"] += 1
            try:
                record = fetcher(candidate)
            except Exception as exc:
                counters["failed"] += 1
                failure = _failure_record(candidate, exc)
                failures.append(failure)
                failure_counter = f"failure_{failure['failure_kind']}"
                counters[failure_counter] = counters.get(failure_counter, 0) + 1
                if halt_on_rate_limit and failure["failure_kind"] == "rate_limited":
                    rate_limit_open = True
                    counters["rate_limit_circuit_open"] = 1
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
        notify(video_id, completion_status)
    added, skipped = append_jsonl_once(canonical, output, id_field="video_id")
    counters["added"] = added
    counters["skipped_duplicate"] = skipped
    if failure_path is not None and failures:
        failure_added, failure_skipped = append_jsonl_once(
            failure_path, failures, id_field="failure_id"
        )
        counters["failure_records_added"] = failure_added
        counters["failure_records_existing"] = failure_skipped
    else:
        counters["failure_records_added"] = 0
        counters["failure_records_existing"] = 0
    return counters
