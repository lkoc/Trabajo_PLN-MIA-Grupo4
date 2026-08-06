from __future__ import annotations

import json
import pytest

from moderacion_peru.datasets import assert_no_video_leakage, stable_video_split
from moderacion_peru.acquisition import (
    bootstrap_canonical_from_existing,
    classify_acquisition_error,
    ingest_incremental,
    load_candidates,
    normalize_category_metadata,
)
from moderacion_peru.incremental import (
    TranscriptSegment,
    chunk_records_incrementally,
    chunk_transcript,
    deduplicate_chunks,
    pending_video_ids,
    remove_vtt_overlap,
)
from moderacion_peru.io import append_jsonl_once, read_jsonl
from moderacion_peru.migration import migrate_jsonl, migrate_record
from moderacion_peru.schemas import ModelReadyRecord


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


def test_members_only_video_is_logged_and_does_not_stop_batch(tmp_path):
    canonical = tmp_path / "raw" / "transcripts_raw.jsonl"
    cache = tmp_path / "raw" / "cache"
    failures = tmp_path / "raw" / "fallos_adquisicion.jsonl"
    candidates = [
        {"video_id": "members", "url": "https://youtube.invalid/members"},
        {"video_id": "public", "url": "https://youtube.invalid/public"},
    ]

    def fetcher(candidate):
        if candidate["video_id"] == "members":
            raise RuntimeError("Join this channel to get access to members-only content")
        return {
            "video_id": candidate["video_id"],
            "segments": [{"start": 0.0, "duration": 1.0, "text": "texto público"}],
        }

    first = ingest_incremental(
        candidates,
        canonical,
        cache,
        fetcher=fetcher,
        failure_path=failures,
        stop_on_error=False,
    )
    second = ingest_incremental(
        candidates,
        canonical,
        cache,
        fetcher=fetcher,
        failure_path=failures,
        stop_on_error=False,
    )

    assert first["failed"] == 1
    assert first["fetched"] == 1
    assert [row["video_id"] for row in read_jsonl(canonical)] == ["public"]
    failure_rows = list(read_jsonl(failures))
    assert len(failure_rows) == 1
    assert failure_rows[0]["failure_kind"] == "members_only"
    assert second["failure_records_added"] == 0
    assert second["failure_records_existing"] == 1


def test_new_video_limit_counts_network_attempts_and_defers_remainder(tmp_path):
    candidates = [{"video_id": value} for value in ("v1", "v2", "v3")]
    fetched = []

    def fetcher(candidate):
        fetched.append(candidate["video_id"])
        return {
            "video_id": candidate["video_id"],
            "segments": [{"start": 0.0, "duration": 1.0, "text": "texto"}],
        }

    stats = ingest_incremental(
        candidates,
        tmp_path / "transcripts.jsonl",
        tmp_path / "cache",
        fetcher=fetcher,
        max_new_videos=2,
    )

    assert fetched == ["v1", "v2"]
    assert stats["fetch_attempted"] == 2
    assert stats["deferred_by_limit"] == 1


def test_acquisition_error_categories_cover_expected_youtube_failures():
    assert classify_acquisition_error(RuntimeError("members-only content")) == "members_only"
    assert classify_acquisition_error(RuntimeError("video unavailable")) == "unavailable_or_private"
    assert classify_acquisition_error(RuntimeError("no tiene subtítulos")) == "no_spanish_subtitles"


def test_acquisition_metadata_uses_canonical_gender_attack_name():
    normalized = normalize_category_metadata(
        {
            "target_category": "CONTENIDO_SEXUAL|ACOSO_GENERO_IDENTIDAD",
            "source_candidate": {
                "target_categories": ["ACOSO_GENERO_IDENTIDAD", "ACOSO_AMENAZA"]
            },
            "text": "ACOSO_GENERO_IDENTIDAD se conserva si forma parte del subtítulo",
        }
    )
    assert normalized["target_category"] == (
        "CONTENIDO_SEXUAL|ATAQUE_POR_GENERO_IDENTIDAD"
    )
    assert normalized["source_candidate"]["target_categories"] == [
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
    ]
    assert "ACOSO_GENERO_IDENTIDAD" in normalized["text"]


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


def test_migration_materializes_grouped_model_ready_rows(tmp_path):
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        json.dumps(
            {
                "chunk_id": "v1_0001",
                "video_id": "v1",
                "text": "ataque por género",
                "coarse_labels": ["ACOSO_GENERO_IDENTIDAD"],
                "flags_reference_only": [],
                "label_source": "humano_modified",
                "sample_weight": 1.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "v2.jsonl"
    manifest = tmp_path / "manifest.json"
    migrate_jsonl(source, destination, manifest)
    row = next(read_jsonl(destination))
    validated = ModelReadyRecord.model_validate(row)
    assert validated.coarse_labels == ["ATAQUE_POR_GENERO_IDENTIDAD"]
    assert validated.split in {"train", "validation", "test"}
    assert json.loads(manifest.read_text(encoding="utf-8"))["counters"][
        f"split:{validated.split}"
    ] == 1


def test_migration_never_guesses_video_id_from_chunk_id(tmp_path):
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        json.dumps(
            {
                "chunk_id": "youtube_id_with_underscore_0001",
                "text": "texto",
                "coarse_labels": ["SEGURO"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="video_id explícito"):
        migrate_jsonl(source, tmp_path / "out.jsonl", tmp_path / "manifest.json")


def test_video_split_is_stable_and_no_leakage():
    assert stable_video_split("v1") == stable_video_split("v1")
    split = stable_video_split("v1")
    assert_no_video_leakage([
        {"video_id": "v1", "split": split},
        {"video_id": "v1", "split": split},
    ])
