from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .io import canonical_json_sha256, sha256_text


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


def chunk_records_incrementally(
    transcripts: Iterable[dict[str, Any]],
    existing_rows: Iterable[dict[str, Any]] = (),
    processed_versions: Iterable[dict[str, Any]] = (),
    *,
    max_seconds: float = 30.0,
    max_characters: int = 600,
    min_characters: int = 90,
    overlap_words: int = 12,
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
        # Las filas 2.1 no registraban configuración y equivalen al contrato de 30 s.
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
    stats["generated_chunks"] = len(generated)
    new_rows = deduplicate_chunks(generated, existing)
    stats["new_unique_chunks"] = len(new_rows)
    return new_rows, version_rows, stats


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
