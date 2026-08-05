from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .io import sha256_text


CHUNKER_VERSION = "2.0.0"


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
    transcript_sha256: str
    chunker_version: str = CHUNKER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
) -> str:
    digest = sha256_text(
        f"{chunker_version}|{video_id}|{start_seconds:.3f}|{end_seconds:.3f}|{normalize_text(text)}"
    )[:20]
    return f"{video_id}_{digest}"


def chunk_transcript(
    video_id: str,
    segments: Iterable[TranscriptSegment],
    *,
    max_seconds: float = 30.0,
    max_characters: int = 600,
) -> list[ChunkRecord]:
    materialized = list(segments)
    source_hash = transcript_hash(materialized)
    chunks: list[ChunkRecord] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        if not current:
            return
        text = normalize_text(" ".join(segment.text for segment in current))
        if not text:
            current.clear()
            return
        start = current[0].start
        end = max(segment.start + segment.duration for segment in current)
        chunks.append(
            ChunkRecord(
                chunk_id=stable_chunk_id(video_id, start, end, text),
                video_id=video_id,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                text=text,
                transcript_sha256=source_hash,
            )
        )
        current.clear()

    for segment in materialized:
        candidate = [*current, segment]
        candidate_text = normalize_text(" ".join(item.text for item in candidate))
        candidate_duration = (segment.start + segment.duration) - candidate[0].start
        if current and (candidate_duration > max_seconds or len(candidate_text) > max_characters):
            flush()
        current.append(segment)
    flush()
    return chunks


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

