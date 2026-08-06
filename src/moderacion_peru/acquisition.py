from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import append_jsonl_once, read_jsonl, sha256_text, write_json_atomic


TranscriptFetcher = Callable[[dict[str, Any]], dict[str, Any]]
DEFAULT_SUBTITLE_LANGUAGES = ("es-PE", "es-419", "es")
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


def classify_acquisition_error(error: BaseException) -> str:
    """Reduce errores heterogéneos de yt-dlp a motivos auditables y estables."""

    message = str(error).casefold()
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

    def collect(source_url: str, source: dict[str, Any], limit: int) -> None:
        options = {**base_options, "playlist_items": f"1:{limit}"}
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=False)
        except Exception as exc:
            failures.append(
                {
                    "source": source.get("name") or source.get("query") or source_url,
                    "url": source_url,
                    "failure_kind": classify_acquisition_error(exc),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:2000],
                }
            )
            return
        if not info:
            failures.append(
                {
                    "source": source.get("name") or source.get("query") or source_url,
                    "url": source_url,
                    "failure_kind": "empty_discovery",
                    "error_type": "EmptyDiscovery",
                    "message": "yt-dlp no devolvió entradas",
                }
            )
            return
        for rank, item in enumerate(info.get("entries", []) or [], start=1):
            if not item or not item.get("id"):
                continue
            video_id = str(item["id"]).strip()
            candidate = {
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": item.get("title"),
                "channel_id": item.get("channel_id"),
                "channel_title": item.get("channel") or source.get("name"),
                "discovery_type": source["discovery_type"],
                "discovery_source": source.get("name") or source.get("query"),
                "discovery_rank": rank,
            }
            for key in ("categoria_fuente", "target_category", "reason"):
                if source.get(key) is not None:
                    candidate[key] = source[key]
            candidates_by_id.setdefault(video_id, candidate)

    for raw_source in channel_sources:
        source = dict(raw_source)
        url = str(source.get("url", "")).strip()
        if not url:
            failures.append(
                {
                    "source": source.get("name", "canal_sin_url"),
                    "url": None,
                    "failure_kind": "invalid_source",
                    "error_type": "ValueError",
                    "message": "El canal no tiene URL",
                }
            )
            continue
        source["discovery_type"] = "channel"
        quota = min(max_videos_per_channel, int(source.get("quota", max_videos_per_channel)))
        collect(_normalise_channel_videos_url(url), source, quota)

    for raw_query in search_queries:
        source = {"query": raw_query} if isinstance(raw_query, str) else dict(raw_query)
        query = str(source.get("query", "")).strip()
        if not query:
            continue
        source["discovery_type"] = "search"
        collect(f"ytsearch{max_results_per_query}:{query}", source, max_results_per_query)

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
    response = requests.get(selected["url"], timeout=60)
    response.raise_for_status()
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
        "unavailable": 0,
    }
    for candidate in candidates:
        video_id = str(candidate.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("Cada candidato requiere video_id")
        if video_id in processed:
            counters["already_canonical"] += 1
            continue
        record = cached_transcript(cache, video_id)
        if record is not None:
            counters["reused_cache"] += 1
        elif fetcher is not None:
            if max_new_videos is not None and fetch_attempts >= max_new_videos:
                counters["deferred_by_limit"] += 1
                continue
            fetch_attempts += 1
            counters["fetch_attempted"] += 1
            try:
                record = fetcher(candidate)
            except Exception as exc:
                counters["failed"] += 1
                failures.append(_failure_record(candidate, exc))
                if stop_on_error:
                    raise
                continue
            record["video_id"] = video_id
            record["acquisition_status"] = "fetched_new"
            write_json_atomic(cache / f"{video_id}.json", record)
            counters["fetched"] += 1
        else:
            counters["unavailable"] += 1
            continue
        record = normalize_category_metadata(record)
        record["source_candidate"] = normalize_category_metadata(candidate)
        record["transcript_sha256"] = sha256_text(
            json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
        )
        output.append(record)
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
