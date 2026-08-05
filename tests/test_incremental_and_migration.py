from __future__ import annotations

import json

from moderacion_peru.datasets import assert_no_video_leakage, stable_video_split
from moderacion_peru.incremental import TranscriptSegment, chunk_transcript, pending_video_ids
from moderacion_peru.io import append_jsonl_once, read_jsonl
from moderacion_peru.migration import migrate_record


def test_video_reuse_keeps_only_new_ids():
    assert pending_video_ids(["a", "b", "a", "c"], ["a"]) == ["b", "c"]


def test_chunk_ids_are_deterministic():
    segments = [TranscriptSegment(0, 3, "Hola"), TranscriptSegment(3, 2, "mundo")]
    first = chunk_transcript("video", segments)
    second = chunk_transcript("video", segments)
    assert first == second
    assert first[0].transcript_sha256


def test_append_jsonl_is_idempotent(tmp_path):
    path = tmp_path / "rows.jsonl"
    assert append_jsonl_once(path, [{"chunk_id": "c1", "x": 1}], id_field="chunk_id") == (1, 0)
    assert append_jsonl_once(path, [{"chunk_id": "c1", "x": 2}], id_field="chunk_id") == (0, 1)
    assert list(read_jsonl(path)) == [{"chunk_id": "c1", "x": 1}]


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

