from __future__ import annotations

import json
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import (
    append_jsonl_once,
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


CHUNKER_VERSION = "2.2.0"
DEFAULT_CHUNKING_CONFIGURATION = {
    "max_seconds": 30.0,
    "max_characters": 600,
    "min_characters": 90,
    "overlap_words": 12,
}


def normalize_chunking_configuration(configuration: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {**DEFAULT_CHUNKING_CONFIGURATION, **(configuration or {})}
    normalized = {
        "max_seconds": float(merged["max_seconds"]),
        "max_characters": int(merged["max_characters"]),
        "min_characters": int(merged["min_characters"]),
        "overlap_words": int(merged["overlap_words"]),
    }
    if normalized["max_seconds"] <= 0:
        raise ValueError("max_seconds debe ser positivo")
    if normalized["max_characters"] < 1 or normalized["min_characters"] < 1:
        raise ValueError("Los límites de caracteres deben ser positivos")
    if normalized["min_characters"] > normalized["max_characters"]:
        raise ValueError("min_characters no puede exceder max_characters")
    if normalized["overlap_words"] < 0:
        raise ValueError("overlap_words no puede ser negativo")
    return normalized


def chunking_signature(configuration: dict[str, Any] | None = None) -> str:
    return canonical_json_sha256(
        {
            "chunker_version": CHUNKER_VERSION,
            "configuration": normalize_chunking_configuration(configuration),
        }
    )


DEFAULT_CHUNKING_SIGNATURE = chunking_signature(DEFAULT_CHUNKING_CONFIGURATION)


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    duration: float
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    text_sha256: str
    transcript_sha256: str
    chunker_version: str = CHUNKER_VERSION
    chunking_signature: str = DEFAULT_CHUNKING_SIGNATURE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").replace("\n", " ")
    normalized = re.sub(
        r"\[(musica|música|aplausos|risas|music|applause|laughter)\]",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def remove_vtt_overlap(previous: str, following: str, *, max_words: int = 12) -> str:
    """Elimina el mayor prefijo repetido por subtítulos rodantes VTT."""

    if not previous or not following:
        return following
    previous_words = previous.casefold().split()
    following_folded = following.casefold().split()
    following_original = following.split()
    window = min(max_words, len(previous_words), len(following_folded))
    for overlap in range(window, 0, -1):
        if previous_words[-overlap:] == following_folded[:overlap]:
            return " ".join(following_original[overlap:])
    return following


def transcript_hash(segments: Iterable[TranscriptSegment]) -> str:
    canonical = "\n".join(
        f"{segment.start:.3f}\t{segment.duration:.3f}\t{normalize_text(segment.text)}"
        for segment in segments
    )
    return sha256_text(canonical)


def stable_chunk_id(
    video_id: str,
    start_seconds: float,
    end_seconds: float,
    text: str,
    *,
    chunker_version: str = CHUNKER_VERSION,
    configuration_signature: str = DEFAULT_CHUNKING_SIGNATURE,
) -> str:
    digest = sha256_text(
        f"{chunker_version}|{configuration_signature}|{video_id}|"
        f"{start_seconds:.3f}|{end_seconds:.3f}|{normalize_text(text)}"
    )[:20]
    return f"{video_id}_{digest}"


def chunk_transcript(
    video_id: str,
    segments: Iterable[TranscriptSegment],
    *,
    max_seconds: float = 30.0,
    max_characters: int = 600,
    min_characters: int = 90,
    overlap_words: int = 12,
) -> list[ChunkRecord]:
    materialized = list(segments)
    source_hash = transcript_hash(materialized)
    configuration_signature = chunking_signature(
        {
            "max_seconds": max_seconds,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "overlap_words": overlap_words,
        }
    )
    chunks: list[ChunkRecord] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        if not current:
            return
        text = normalize_text(" ".join(segment.text for segment in current))
        if len(text) < min_characters:
            current.clear()
            return
        start = current[0].start
        end = max(segment.start + segment.duration for segment in current)
        chunks.append(
            ChunkRecord(
                chunk_id=stable_chunk_id(
                    video_id,
                    start,
                    end,
                    text,
                    configuration_signature=configuration_signature,
                ),
                video_id=video_id,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                text=text,
                text_sha256=sha256_text(text.casefold()),
                transcript_sha256=source_hash,
                chunking_signature=configuration_signature,
            )
        )
        current.clear()

    for segment in materialized:
        cleaned = normalize_text(segment.text)
        if not cleaned:
            continue
        if current:
            cleaned = remove_vtt_overlap(current[-1].text, cleaned, max_words=overlap_words)
            if not cleaned:
                continue
        current.append(TranscriptSegment(segment.start, segment.duration, cleaned))
        candidate_text = normalize_text(" ".join(item.text for item in current))
        candidate_duration = (segment.start + segment.duration) - current[0].start
        if candidate_duration >= max_seconds or len(candidate_text) >= max_characters:
            flush()
    flush()
    return chunks


def deduplicate_chunks(
    rows: Iterable[dict[str, Any]],
    existing_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Conserva el primer chunk por ID y texto normalizado, incluso entre lotes."""

    existing = list(existing_rows)
    seen_ids = {str(row.get("chunk_id")) for row in existing if row.get("chunk_id")}
    seen_text = {
        str(row.get("text_sha256") or sha256_text(normalize_text(str(row.get("text", ""))).casefold()))
        for row in existing
        if row.get("text") or row.get("text_sha256")
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        chunk_id = str(row.get("chunk_id", ""))
        text_hash = str(
            row.get("text_sha256")
            or sha256_text(normalize_text(str(row.get("text", ""))).casefold())
        )
        if not chunk_id:
            raise ValueError("Registro sin chunk_id")
        if chunk_id in seen_ids or text_hash in seen_text:
            continue
        seen_ids.add(chunk_id)
        seen_text.add(text_hash)
        result.append(row)
    return result


def _numeric_summary(values: Iterable[float | int]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "standard_deviation": None,
        }

    def percentile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "mean": statistics.fmean(ordered),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "standard_deviation": statistics.pstdev(ordered),
    }


def describe_chunk_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calcula estadística descriptiva reproducible del artefacto de chunks."""

    per_video: Counter[str] = Counter()
    durations: list[float] = []
    characters: list[int] = []
    words: list[int] = []
    chunk_count = 0
    for row in rows:
        chunk_count += 1
        video_id = str(row.get("video_id") or "").strip()
        if video_id:
            per_video[video_id] += 1
        start = float(row.get("start_seconds", 0.0) or 0.0)
        end = float(row.get("end_seconds", start) or start)
        durations.append(max(end - start, 0.0))
        text = str(row.get("text") or "")
        characters.append(len(text))
        words.append(len(text.split()))
    return {
        "chunks": chunk_count,
        "videos": len(per_video),
        "chunks_per_video": _numeric_summary(per_video.values()),
        "duration_seconds": {
            **_numeric_summary(durations),
            "total": sum(durations),
            "total_hours": sum(durations) / 3600.0,
        },
        "characters_per_chunk": {
            **_numeric_summary(characters),
            "total": sum(characters),
        },
        "words_per_chunk": {
            **_numeric_summary(words),
            "total": sum(words),
        },
    }


def chunk_records_incrementally(
    transcripts: Iterable[dict[str, Any]],
    existing_rows: Iterable[dict[str, Any]] = (),
    processed_versions: Iterable[dict[str, Any]] = (),
    *,
    max_seconds: float = 30.0,
    max_characters: int = 600,
    min_characters: int = 90,
    overlap_words: int = 12,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Trocea solo versiones de video que no aparecen ya en la salida."""

    existing = list(existing_rows)
    configuration = normalize_chunking_configuration(
        {
            "max_seconds": max_seconds,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "overlap_words": overlap_words,
        }
    )
    configuration_signature = chunking_signature(configuration)

    def row_signature(row: dict[str, Any]) -> str:
        # Las filas del troceador 2.1 no registraban firma y equivalen a 30 s.
        return str(row.get("chunking_signature") or DEFAULT_CHUNKING_SIGNATURE)

    known_versions = {
        (str(row.get("video_id")), str(row.get("transcript_sha256")), row_signature(row))
        for row in existing
        if row.get("video_id") and row.get("transcript_sha256")
    }
    known_versions.update(
        (str(row.get("video_id")), str(row.get("transcript_sha256")), row_signature(row))
        for row in processed_versions
        if row.get("video_id") and row.get("transcript_sha256")
    )
    generated: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []
    stats = {
        "transcripts_seen": 0,
        "unchanged_videos": 0,
        "new_or_changed_videos": 0,
        "generated_chunks": 0,
        "new_unique_chunks": 0,
    }
    for video in transcripts:
        stats["transcripts_seen"] += 1
        segments = [
            TranscriptSegment(
                float(segment.get("start", 0)),
                float(segment.get("duration", 0)),
                str(segment.get("text", "")),
            )
            for segment in video.get("segments", [])
        ]
        source_hash = transcript_hash(segments)
        version = (str(video["video_id"]), source_hash, configuration_signature)
        if version in known_versions:
            stats["unchanged_videos"] += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "unchanged",
                        "advance": 1,
                        "video_id": version[0],
                        "generated_chunks": 0,
                        **stats,
                    }
                )
            continue
        stats["new_or_changed_videos"] += 1
        video_chunks = [
            chunk.to_dict()
            for chunk in chunk_transcript(version[0], segments, **configuration)
        ]
        for row in video_chunks:
            row.update(
                {
                    "video_title": video.get("video_title") or video.get("title"),
                    "channel_title": video.get("channel_title") or video.get("channel"),
                    "source_url": video.get("source_url") or video.get("url"),
                }
            )
        generated.extend(video_chunks)
        version_rows.append(
            {
                "version_id": sha256_text(
                    f"{version[0]}|{source_hash}|{CHUNKER_VERSION}|{configuration_signature}"
                ),
                "video_id": version[0],
                "transcript_sha256": source_hash,
                "chunker_version": CHUNKER_VERSION,
                "chunking_signature": configuration_signature,
                "configuration": configuration,
                "chunk_count": len(video_chunks),
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "materialized",
                    "advance": 1,
                    "video_id": version[0],
                    "generated_chunks_for_video": len(video_chunks),
                    **stats,
                }
            )
    stats["generated_chunks"] = len(generated)
    new_rows = deduplicate_chunks(generated, existing)
    stats["new_unique_chunks"] = len(new_rows)
    generated_counts = Counter(str(row.get("video_id")) for row in generated)
    materialized_counts = Counter(str(row.get("video_id")) for row in new_rows)
    for row in version_rows:
        video_id = str(row["video_id"])
        row["generated_chunk_count"] = generated_counts.get(video_id, 0)
        row["materialized_chunk_count"] = materialized_counts.get(video_id, 0)
    stats["videos_with_generated_chunks"] = sum(
        count > 0 for count in generated_counts.values()
    )
    stats["videos_with_new_unique_chunks"] = sum(
        count > 0 for count in materialized_counts.values()
    )
    stats["videos_without_generated_chunks"] = stats["new_or_changed_videos"] - stats[
        "videos_with_generated_chunks"
    ]
    stats["videos_without_new_unique_chunks"] = stats["new_or_changed_videos"] - stats[
        "videos_with_new_unique_chunks"
    ]
    return new_rows, version_rows, stats


def rebuild_chunk_materialization(
    project_root: str | Path,
    transcripts: Iterable[dict[str, Any]],
    *,
    source_path: str | Path,
    output_path: str | Path,
    version_index_path: str | Path,
    manifest_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    max_seconds: float = 30.0,
    max_characters: int = 600,
    min_characters: int = 90,
    overlap_words: int = 12,
) -> dict[str, Any]:
    """Reconstruye chunks atómicamente y conserva una copia recuperable previa."""

    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    versions = Path(version_index_path).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output.with_name("chunk_materialization_manifest.json")
    )
    for target in (source, output, versions, manifest):
        if target != root and root not in target.parents:
            raise ValueError(f"Ruta fuera del proyecto: {target}")
    if not source.is_file():
        raise FileNotFoundError(source)
    configuration = normalize_chunking_configuration(
        {
            "max_seconds": max_seconds,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "overlap_words": overlap_words,
        }
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = root / "archivo" / "chunk_rebuilds" / timestamp
    archived = []
    for previous in (output, versions, manifest):
        if not previous.is_file():
            continue
        relative = previous.relative_to(root)
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, destination)
        archived.append(
            {
                "source": relative.as_posix(),
                "archive": destination.relative_to(root).as_posix(),
                "bytes": previous.stat().st_size,
                "sha256": sha256_file(previous),
            }
        )
    archive_manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "backup_before_full_chunk_rebuild",
        "source_transcripts": source.relative_to(root).as_posix(),
        "source_sha256": sha256_file(source),
        "configuration": configuration,
        "files": archived,
    }
    write_json_atomic(archive_root / "backup_manifest.json", archive_manifest)

    rows, version_rows, stats = chunk_records_incrementally(
        transcripts,
        max_seconds=configuration["max_seconds"],
        max_characters=configuration["max_characters"],
        min_characters=configuration["min_characters"],
        overlap_words=configuration["overlap_words"],
        progress_callback=progress_callback,
    )
    if stats["transcripts_seen"] == 0:
        raise ValueError("La reconstrucción no puede reemplazar los chunks con una fuente vacía")
    write_jsonl_atomic(output, rows)
    write_jsonl_atomic(versions, version_rows)
    videos_without_chunks = sorted(
        str(row["video_id"])
        for row in version_rows
        if int(row.get("materialized_chunk_count", 0)) == 0
    )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "full_chunk_rebuild",
        "source": {
            "path": source.relative_to(root).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "configuration": configuration,
        "chunker_version": CHUNKER_VERSION,
        "chunking_signature": chunking_signature(configuration),
        "stats": stats,
        "coverage": {
            "transcript_videos": stats["transcripts_seen"],
            "videos_with_chunks": stats["videos_with_new_unique_chunks"],
            "videos_without_chunks": len(videos_without_chunks),
            "video_ids_without_chunks": videos_without_chunks,
            "orphan_chunk_videos": [],
        },
        "descriptive_statistics": describe_chunk_rows(rows),
        "outputs": {
            "chunks": {
                "path": output.relative_to(root).as_posix(),
                "rows": len(rows),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "versions": {
                "path": versions.relative_to(root).as_posix(),
                "rows": len(version_rows),
                "bytes": versions.stat().st_size,
                "sha256": sha256_file(versions),
            },
        },
        "backup": archive_root.relative_to(root).as_posix(),
    }
    write_json_atomic(manifest, payload)
    return payload


def materialize_chunk_records(
    project_root: str | Path,
    transcripts: Iterable[dict[str, Any]],
    *,
    source_path: str | Path,
    output_path: str | Path,
    version_index_path: str | Path,
    manifest_path: str | Path | None = None,
    rebuild: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    max_seconds: float = 30.0,
    max_characters: int = 600,
    min_characters: int = 90,
    overlap_words: int = 12,
) -> dict[str, Any]:
    """Materializa el canónico de forma incremental o mediante reconstrucción total."""

    if rebuild:
        return rebuild_chunk_materialization(
            project_root,
            transcripts,
            source_path=source_path,
            output_path=output_path,
            version_index_path=version_index_path,
            manifest_path=manifest_path,
            progress_callback=progress_callback,
            max_seconds=max_seconds,
            max_characters=max_characters,
            min_characters=min_characters,
            overlap_words=overlap_words,
        )

    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    versions = Path(version_index_path).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output.with_name("chunk_materialization_manifest.json")
    )
    for target in (source, output, versions, manifest):
        if target != root and root not in target.parents:
            raise ValueError(f"Ruta fuera del proyecto: {target}")
    if not source.is_file():
        raise FileNotFoundError(source)
    previous_manifest = (
        json.loads(manifest.read_text(encoding="utf-8-sig"))
        if manifest.is_file()
        else {}
    )
    configuration = normalize_chunking_configuration(
        {
            "max_seconds": max_seconds,
            "max_characters": max_characters,
            "min_characters": min_characters,
            "overlap_words": overlap_words,
        }
    )
    existing = list(read_jsonl(output)) if output.is_file() else []
    processed_versions = list(read_jsonl(versions)) if versions.is_file() else []
    rows, version_rows, stats = chunk_records_incrementally(
        transcripts,
        existing,
        processed_versions,
        max_seconds=configuration["max_seconds"],
        max_characters=configuration["max_characters"],
        min_characters=configuration["min_characters"],
        overlap_words=configuration["overlap_words"],
        progress_callback=progress_callback,
    )
    if stats["transcripts_seen"] == 0:
        raise ValueError("La materialización no puede operar con una fuente vacía")
    added, skipped = append_jsonl_once(output, rows, id_field="chunk_id")
    versions_added, versions_skipped = append_jsonl_once(
        versions,
        version_rows,
        id_field="version_id",
    )
    transcript_ids = {
        str(row.get("video_id") or "").strip()
        for row in read_jsonl(source)
        if str(row.get("video_id") or "").strip()
    }
    chunk_rows = 0
    chunk_video_ids: set[str] = set()
    for row in read_jsonl(output):
        chunk_rows += 1
        video_id = str(row.get("video_id") or "").strip()
        if video_id:
            chunk_video_ids.add(video_id)
    videos_without_chunks = sorted(transcript_ids - chunk_video_ids)
    stats.update(
        {
            "added": added,
            "duplicate_ids": skipped,
            "versions_registered": versions_added,
            "duplicate_version_ids": versions_skipped,
        }
    )
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "incremental_chunk_update",
        "source": {
            "path": source.relative_to(root).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "configuration": configuration,
        "chunker_version": CHUNKER_VERSION,
        "chunking_signature": chunking_signature(configuration),
        "stats": stats,
        "coverage": {
            "transcript_videos": len(transcript_ids),
            "videos_with_chunks": len(transcript_ids & chunk_video_ids),
            "videos_without_chunks": len(videos_without_chunks),
            "video_ids_without_chunks": videos_without_chunks,
            "orphan_chunk_videos": sorted(chunk_video_ids - transcript_ids),
        },
        "descriptive_statistics": describe_chunk_rows(read_jsonl(output)),
        "outputs": {
            "chunks": {
                "path": output.relative_to(root).as_posix(),
                "rows": chunk_rows,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            },
            "versions": {
                "path": versions.relative_to(root).as_posix(),
                "rows": sum(1 for _ in read_jsonl(versions)),
                "bytes": versions.stat().st_size,
                "sha256": sha256_file(versions),
            },
        },
        "backup": previous_manifest.get("backup"),
        "previous_materialization": (
            {
                "operation": previous_manifest.get("operation"),
                "created_at": previous_manifest.get("created_at"),
                "backup": previous_manifest.get("backup"),
            }
            if previous_manifest
            else None
        ),
        "last_full_rebuild": (
            {
                "created_at": previous_manifest.get("created_at"),
                "backup": previous_manifest.get("backup"),
            }
            if previous_manifest.get("operation") == "full_chunk_rebuild"
            else previous_manifest.get("last_full_rebuild")
        ),
    }
    write_json_atomic(manifest, payload)
    return payload


def pending_video_ids(candidate_ids: Iterable[str], processed_ids: Iterable[str]) -> list[str]:
    processed = set(processed_ids)
    return [video_id for video_id in dict.fromkeys(candidate_ids) if video_id not in processed]


def pending_records(
    records: Iterable[dict[str, Any]],
    processed_ids: Iterable[str],
    *,
    id_field: str = "chunk_id",
) -> list[dict[str, Any]]:
    processed = set(processed_ids)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        identifier = record.get(id_field)
        if not identifier:
            raise ValueError(f"Registro sin {id_field}")
        if identifier in processed or identifier in seen:
            continue
        seen.add(identifier)
        result.append(record)
    return result
