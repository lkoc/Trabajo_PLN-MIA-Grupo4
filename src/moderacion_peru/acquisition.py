from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import append_jsonl_once, read_jsonl, sha256_text


TranscriptFetcher = Callable[[dict[str, Any]], dict[str, Any]]


def fetch_youtube_subtitles(candidate: dict[str, Any]) -> dict[str, Any]:
    """Descarga únicamente subtítulos; nunca descarga el video ni el audio."""

    try:
        import requests
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[datos] para adquirir subtítulos nuevos") from exc
    video_id = str(candidate["video_id"])
    url = str(candidate.get("url") or f"https://www.youtube.com/watch?v={video_id}")
    options = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    tracks = info.get("subtitles", {}) or {}
    automatic = info.get("automatic_captions", {}) or {}
    variants = tracks.get("es") or tracks.get("es-419") or automatic.get("es") or automatic.get("es-419")
    if not variants:
        raise RuntimeError(f"{video_id} no tiene subtítulos en español")
    selected = next((item for item in variants if item.get("ext") == "json3"), variants[0])
    response = requests.get(selected["url"], timeout=60)
    response.raise_for_status()
    if selected.get("ext") != "json3":
        raise RuntimeError(f"Formato de subtítulo no compatible para {video_id}: {selected.get('ext')}")
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
        "language": "es",
        "subtitle_source": "manual" if tracks.get("es") or tracks.get("es-419") else "automatic",
        "segments": segments,
    }


def processed_video_ids(canonical_path: str | Path) -> set[str]:
    return {str(row["video_id"]) for row in read_jsonl(canonical_path)} if Path(canonical_path).exists() else set()


def cached_transcript(cache_dir: str | Path, video_id: str) -> dict[str, Any] | None:
    path = Path(cache_dir) / f"{video_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if str(payload.get("video_id")) != video_id:
        raise ValueError(f"El caché {path} no corresponde a {video_id}")
    payload["acquisition_status"] = "reused_cache"
    return payload


def ingest_incremental(
    candidates: Iterable[dict[str, Any]],
    canonical_path: str | Path,
    cache_dir: str | Path,
    *,
    fetcher: TranscriptFetcher | None = None,
) -> dict[str, int]:
    """Reutiliza el corpus/caché y solo invoca fetcher para videos desconocidos."""

    canonical = Path(canonical_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    processed = processed_video_ids(canonical)
    output: list[dict[str, Any]] = []
    counters = {"already_canonical": 0, "reused_cache": 0, "fetched": 0, "unavailable": 0}
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
            record = fetcher(candidate)
            record["video_id"] = video_id
            record["acquisition_status"] = "fetched_new"
            (cache / f"{video_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            counters["fetched"] += 1
        else:
            counters["unavailable"] += 1
            continue
        record["source_candidate"] = candidate
        record["transcript_sha256"] = sha256_text(
            json.dumps(record.get("segments", []), ensure_ascii=False, sort_keys=True)
        )
        output.append(record)
    added, skipped = append_jsonl_once(canonical, output, id_field="video_id")
    counters["added"] = added
    counters["skipped_duplicate"] = skipped
    return counters
