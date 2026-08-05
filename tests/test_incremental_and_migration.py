from __future__ import annotations

import json

from moderacion_peru.datasets import assert_no_video_leakage, stable_video_split
from moderacion_peru.acquisition import bootstrap_canonical_from_existing, load_candidates
from moderacion_peru.incremental import (
    TranscriptSegment,
    chunk_records_incrementally,
    chunk_transcript,
    deduplicate_chunks,
    pending_video_ids,
    remove_vtt_overlap,
)
from moderacion_peru.io import append_jsonl_once, read_jsonl
from moderacion_peru.migration import migrate_record


def test_video_reuse_keeps_only_new_ids():
    assert pending_video_ids(["a", "b", "a", "c"], ["a"]) == ["b", "c"]


def test_chunk_ids_are_deterministic():
    segments = [TranscriptSegment(0, 3, "Hola"), TranscriptSegment(3, 2, "mundo")]
    first = chunk_transcript("video", segments, min_characters=1)
    second = chunk_transcript("video", segments, min_characters=1)
    assert first == second
    assert first[0].transcript_sha256


def test_vtt_overlap_and_minimum_length_preserve_previous_cleaning_contract():
    assert remove_vtt_overlap("uno dos tres", "dos tres cuatro") == "cuatro"
    repeated = [
        TranscriptSegment(0, 10, "uno dos tres"),
        TranscriptSegment(10, 20, "dos tres cuatro cinco seis"),
    ]
    chunks = chunk_transcript("video", repeated, min_characters=1)
    assert chunks[0].text == "uno dos tres cuatro cinco seis"


def test_chunk_deduplication_uses_normalized_text_hash():
    rows = [
        {"chunk_id": "c1", "text": "Texto único"},
        {"chunk_id": "c2", "text": "  texto   único  "},
    ]
    assert [row["chunk_id"] for row in deduplicate_chunks(rows)] == ["c1"]


def test_incremental_chunking_skips_unchanged_video_version():
    transcripts = [{"video_id": "v1", "segments": [{"start": 0, "duration": 30, "text": "x " * 60}]}]
    first, versions, first_stats = chunk_records_incrementally(transcripts)
    second, _, second_stats = chunk_records_incrementally(transcripts, first, versions)
    assert first_stats["new_or_changed_videos"] == 1
    assert second == []
    assert second_stats["unchanged_videos"] == 1


def test_append_jsonl_is_idempotent(tmp_path):
    path = tmp_path / "rows.jsonl"
    assert append_jsonl_once(path, [{"chunk_id": "c1", "x": 1}], id_field="chunk_id") == (1, 0)
    assert append_jsonl_once(path, [{"chunk_id": "c1", "x": 2}], id_field="chunk_id") == (0, 1)
    assert list(read_jsonl(path)) == [{"chunk_id": "c1", "x": 1}]


def test_existing_transcripts_are_reused_without_touching_source(tmp_path):
    source = tmp_path / "snapshot" / "transcripts_raw.jsonl"
    source.parent.mkdir()
    source.write_text('{"video_id":"v1","segments":[{"text":"hola"}],"view_count":NaN}\n', encoding="utf-8")
    original = source.read_bytes()
    canonical = tmp_path / "canonical" / "transcripts_raw.jsonl"

    first = bootstrap_canonical_from_existing([source], canonical)
    second = bootstrap_canonical_from_existing([source], canonical)

    assert first["added"] == 1
    assert second["added"] == 0
    assert source.read_bytes() == original
    assert list(read_jsonl(canonical))[0]["view_count"] is None


def test_candidates_accept_csv_and_jsonl(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("video_id,url\nv1,https://example.invalid/v1\n", encoding="utf-8")
    assert load_candidates(csv_path)[0]["video_id"] == "v1"


def test_migration_never_infers_safe_from_empty():
    migrated = migrate_record({"chunk_id": "c1", "text": "x", "coarse_labels": []})
    assert migrated["coarse_labels"] == []
    assert migrated["needs_review"]
    assert not migrated["training_eligible"]


def test_migration_merges_legacy_damage_and_preserves_source():
    migrated = migrate_record(
        {
            "chunk_id": "c1",
            "text": "x",
            "coarse_labels": ["ACOSO_PERSONAL", "AMENAZA_DIRECTA"],
            "label_source": "humano_modified",
        }
    )
    assert migrated["coarse_labels"] == ["ACOSO_AMENAZA"]
    assert migrated["legacy_coarse_labels"] == ["ACOSO_PERSONAL", "AMENAZA_DIRECTA"]


def test_video_split_is_stable_and_no_leakage():
    assert stable_video_split("v1") == stable_video_split("v1")
    split = stable_video_split("v1")
    assert_no_video_leakage([
        {"video_id": "v1", "split": split},
        {"video_id": "v1", "split": split},
    ])
