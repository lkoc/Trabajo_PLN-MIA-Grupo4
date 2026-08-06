from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from moderacion_peru.datasets import assert_no_video_leakage, stable_video_split
from moderacion_peru.acquisition import (
    append_transcripts_by_channel,
    bootstrap_canonical_from_existing,
    build_directed_sampling_plan,
    classify_acquisition_error,
    discover_youtube_candidates,
    expand_directed_channel_sources,
    fetch_youtube_subtitles,
    ingest_incremental,
    load_candidates,
    materialize_transcripts_by_channel,
    normalize_category_metadata,
    order_candidates_for_acquisition,
    reset_active_video_dataset,
    restore_canonical_from_channel_transcripts,
    select_directed_candidates,
    select_directed_search_queries,
    VIDEO_DATASET_RESET_CONFIRMATION,
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


def test_directed_plan_falls_back_to_equal_damage_weights_without_prior_data():
    plan = build_directed_sampling_plan([], [])

    assert plan["strategy"] == "fallback_equal"
    assert set(plan["weights"].values()) == {0.25}
    assert set(plan["deficit_videos"].values()) == {1}
    cohort = select_directed_candidates(
        [
            {
                "video_id": f"v{index}",
                "target_category": label,
                "discovery_source": label,
            }
            for index, label in enumerate(plan["damage_labels"])
        ],
        (),
        plan,
        max_candidates=4,
    )
    assert Counter(row["directed_priority_label"] for row in cohort) == Counter(
        {label: 1 for label in plan["damage_labels"]}
    )


def test_directed_plan_uses_development_video_deficits_and_excludes_test():
    dataset = [
        {"video_id": "v1", "split": "train", "coarse_labels": ["RACISMO_DISCRIMINACION"]},
        {"video_id": "v2", "split": "train", "coarse_labels": ["ACOSO_AMENAZA"]},
        {"video_id": "v3", "split": "validation", "coarse_labels": ["ACOSO_AMENAZA"]},
        {"video_id": "v4", "split": "validation", "coarse_labels": ["CONTENIDO_SEXUAL"]},
        {
            "video_id": "test-only",
            "split": "test",
            "coarse_labels": ["ATAQUE_POR_GENERO_IDENTIDAD"],
        },
    ]
    transcripts = [
        {"video_id": "v1", "channel_id": "c1", "channel_title": "Canal uno"},
        {"video_id": "v2", "channel_id": "c1", "channel_title": "Canal uno"},
        {"video_id": "v3", "channel_id": "c1", "channel_title": "Canal uno"},
        {"video_id": "v4", "channel_id": "c2", "channel_title": "Canal dos"},
        {"video_id": "test-only", "channel_id": "c3", "channel_title": "Canal test"},
    ]

    plan = build_directed_sampling_plan(dataset, transcripts)

    assert plan["strategy"] == "deficit_weighted"
    assert plan["support_videos"] == {
        "RACISMO_DISCRIMINACION": 1,
        "ATAQUE_POR_GENERO_IDENTIDAD": 0,
        "ACOSO_AMENAZA": 2,
        "CONTENIDO_SEXUAL": 1,
    }
    assert plan["weights"] == {
        "RACISMO_DISCRIMINACION": 0.25,
        "ATAQUE_POR_GENERO_IDENTIDAD": 0.5,
        "ACOSO_AMENAZA": 0.0,
        "CONTENIDO_SEXUAL": 0.25,
    }
    assert {row["channel_id"] for row in plan["channel_profiles"]} == {"c1", "c2"}


def test_directed_queries_expansion_and_selection_follow_deficit_weights():
    plan = build_directed_sampling_plan([], [])
    queries = select_directed_search_queries(
        plan,
        [
            {"query": "racismo", "target_category": "RACISMO_DISCRIMINACION"},
            {"query": "género", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
            {"query": "amenaza", "target_category": "ACOSO_AMENAZA"},
            {"query": "sexual", "target_category": "CONTENIDO_SEXUAL"},
        ],
        max_queries=4,
    )
    assert {row["query"] for row in queries} == {"racismo", "género", "amenaza", "sexual"}

    search_candidates = [
        {
            "video_id": "s1",
            "channel_id": "new-channel",
            "channel_title": "Canal nuevo",
            "discovery_type": "search",
            "discovery_rank": 1,
            "target_category": "CONTENIDO_SEXUAL",
        }
    ]
    expanded = expand_directed_channel_sources(search_candidates, plan, max_channels=1)
    assert expanded[0]["url"] == "https://www.youtube.com/channel/new-channel"
    assert expanded[0]["target_category"] == "CONTENIDO_SEXUAL"

    cohort = select_directed_candidates(
        [
            {"video_id": "old", "target_category": "CONTENIDO_SEXUAL"},
            {"video_id": "v1", "target_category": "CONTENIDO_SEXUAL", "discovery_source": "a"},
            {"video_id": "v2", "target_category": "RACISMO_DISCRIMINACION", "discovery_source": "b"},
            {"video_id": "general"},
        ],
        {"old"},
        plan,
        max_candidates=10,
    )
    assert {row["video_id"] for row in cohort} == {"v1", "v2"}
    assert all(row.get("directed_priority_label") for row in cohort)


def test_video_dataset_reset_is_confirmed_scoped_and_recoverable(tmp_path):
    root = tmp_path / "project"
    (root / "src/moderacion_peru").mkdir(parents=True)
    (root / "datos/raw/transcripts_cache").mkdir(parents=True)
    (root / "datos/raw/transcripts_by_channel").mkdir(parents=True)
    (root / "datos/ampliacion/historico").mkdir(parents=True)
    (root / "datos/raw/transcripts_raw.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "datos/raw/transcripts_cache/v1.json").write_text("{}", encoding="utf-8")
    channel_readme = root / "datos/raw/transcripts_by_channel/README.md"
    channel_readme.write_text("documentación\n", encoding="utf-8")
    (root / "datos/raw/transcripts_by_channel/canal--part-0001.jsonl").write_text(
        '{}\n', encoding="utf-8"
    )
    (root / "datos/raw/transcripts_by_channel/index.json").write_text("{}", encoding="utf-8")
    historical = root / "datos/ampliacion/historico/transcripts_raw.jsonl"
    historical.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Confirmación inválida"):
        reset_active_video_dataset(root, "NO")
    result = reset_active_video_dataset(root, VIDEO_DATASET_RESET_CONFIRMATION)

    assert not (root / "datos/raw/transcripts_raw.jsonl").exists()
    assert historical.exists()
    assert channel_readme.exists()
    assert result["archive_path"]
    assert (root / result["archive_path"] / "datos/raw/transcripts_raw.jsonl").exists()
    assert (root / "datos/raw/manifests/rebuild_from_zero.json").exists()
    sentinel = root / "datos/raw/new_run.json"
    sentinel.write_text("{}", encoding="utf-8")
    repeated = reset_active_video_dataset(root, VIDEO_DATASET_RESET_CONFIRMATION)
    assert repeated["status"] == "already_active_noop"
    assert sentinel.exists()


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


def test_incremental_chunking_reprocesses_same_transcript_for_another_length():
    transcripts = [
        {
            "video_id": "v1",
            "segments": [
                {"start": index * 10, "duration": 10, "text": chr(97 + index) * 50}
                for index in range(4)
            ],
        }
    ]
    first, versions, _ = chunk_records_incrementally(transcripts, max_seconds=30)
    changed, changed_versions, stats = chunk_records_incrementally(
        transcripts,
        first,
        versions,
        max_seconds=20,
    )
    assert stats["new_or_changed_videos"] == 1
    assert changed
    assert changed_versions[0]["chunking_signature"] != versions[0]["chunking_signature"]


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


def test_channel_partitions_preserve_canonical_and_restore_idempotently(tmp_path):
    canonical = tmp_path / "raw" / "transcripts_raw.jsonl"
    canonical.parent.mkdir()
    rows = [
        {"video_id": "v1", "channel_title": "Canal Á", "segments": [{"text": "uno"}]},
        {
            "video_id": "v2",
            "channel_id": "channel-a",
            "channel": "Canal Á",
            "segments": [{"text": "dos"}],
        },
        {
            "video_id": "v3",
            "channel_id": "channel-b",
            "channel": "Canal B",
            "segments": [{"text": "tres"}],
        },
    ]
    canonical.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    original = canonical.read_bytes()
    partitions = tmp_path / "raw" / "transcripts_by_channel"

    materialized = materialize_transcripts_by_channel(canonical, partitions)

    assert canonical.read_bytes() == original
    assert materialized["total_videos"] == 3
    assert materialized["total_channel_files"] == 2
    index = json.loads((partitions / "index.json").read_text(encoding="utf-8"))
    assert sum(entry["videos"] for entry in index["files"]) == 3
    assert {tuple(entry["channel_ids"]) for entry in index["files"]} == {
        ("channel-a",),
        ("channel-b",),
    }

    appended = append_transcripts_by_channel(
        partitions,
        [
            {"video_id": "v4", "channel_id": "channel-a", "channel": "Canal Á"},
            {"video_id": "v5", "channel_id": "channel-c", "channel": "Canal C"},
        ],
    )
    repeated = append_transcripts_by_channel(
        partitions,
        [
            {"video_id": "v4", "channel_id": "channel-a", "channel": "Canal Á"},
            {"video_id": "v5", "channel_id": "channel-c", "channel": "Canal C"},
        ],
    )
    assert appended["added"] == 2
    assert repeated["already_partitioned"] == 2

    restored = tmp_path / "clone" / "datos" / "raw" / "transcripts_raw.jsonl"
    first = restore_canonical_from_channel_transcripts(partitions, restored)
    second = restore_canonical_from_channel_transcripts(partitions, restored)
    assert first["added"] == 5
    assert second["added"] == 0
    assert second["already_canonical"] == 5
    assert {row["video_id"] for row in read_jsonl(restored)} == {"v1", "v2", "v3", "v4", "v5"}


def test_large_channel_is_split_into_bounded_numbered_parts(tmp_path):
    canonical = tmp_path / "transcripts_raw.jsonl"
    rows = [
        {
            "video_id": f"v{index}",
            "channel_id": "large-channel",
            "channel": "Canal grande",
            "segments": [{"text": str(index) * 700}],
        }
        for index in range(4)
    ]
    canonical.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "by_channel"

    result = materialize_transcripts_by_channel(
        canonical,
        output,
        max_channel_file_bytes=1200,
    )

    parts = sorted(output.glob("*.jsonl"))
    assert result["total_channels"] == 1
    assert result["total_channel_files"] == 4
    assert [path.name.rsplit("--", 1)[-1] for path in parts] == [
        "part-0001.jsonl",
        "part-0002.jsonl",
        "part-0003.jsonl",
        "part-0004.jsonl",
    ]
    assert all(path.stat().st_size <= 1200 for path in parts)


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

    progress = []
    first = ingest_incremental(
        candidates,
        canonical,
        cache,
        fetcher=fetcher,
        failure_path=failures,
        stop_on_error=False,
        progress_callback=progress.append,
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
    assert first["failure_members_only"] == 1
    assert [event["status"] for event in progress] == ["failed", "fetched"]
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


def test_network_batches_cover_all_candidates_and_pause_between_batches(tmp_path, monkeypatch):
    candidates = [{"video_id": f"v{index}"} for index in range(5)]
    fetched = []
    pauses = []
    progress = []
    checkpoint_sizes = []
    canonical = tmp_path / "transcripts.jsonl"

    def fetcher(candidate):
        fetched.append(candidate["video_id"])
        return {
            "video_id": candidate["video_id"],
            "segments": [{"start": 0.0, "duration": 1.0, "text": "texto"}],
        }

    monkeypatch.setattr("moderacion_peru.acquisition.time.sleep", pauses.append)

    def report(event):
        progress.append(event)
        if event["status"] == "batch_pause":
            checkpoint_sizes.append(len(list(read_jsonl(canonical))))

    stats = ingest_incremental(
        candidates,
        canonical,
        tmp_path / "cache",
        fetcher=fetcher,
        max_new_videos=None,
        network_batch_size=2,
        batch_pause_seconds=30,
        progress_callback=report,
    )

    assert fetched == ["v0", "v1", "v2", "v3", "v4"]
    assert pauses == [30, 30]
    assert stats["fetched"] == 5
    assert stats["batch_pauses"] == 2
    assert stats["deferred_by_limit"] == 0
    assert checkpoint_sizes == [2, 4]
    assert [event["advance"] for event in progress if event["status"] == "batch_pause"] == [0, 0]


def test_download_order_is_seeded_complete_and_interleaved_by_channel():
    candidates = [
        {"video_id": "a1", "channel_id": "a"},
        {"video_id": "a2", "channel_id": "a"},
        {"video_id": "b1", "channel_id": "b"},
        {"video_id": "b2", "channel_id": "b"},
        {"video_id": "c1", "channel_id": "c"},
    ]

    first = order_candidates_for_acquisition(candidates, random_seed=17)
    repeated = order_candidates_for_acquisition(reversed(candidates), random_seed=17)
    different = order_candidates_for_acquisition(candidates, random_seed=18)

    assert [row["video_id"] for row in first] == [row["video_id"] for row in repeated]
    assert {row["video_id"] for row in first} == {row["video_id"] for row in candidates}
    assert [row["download_queue_rank"] for row in first] == [1, 2, 3, 4, 5]
    assert [row["video_id"] for row in first] != [row["video_id"] for row in different]
    first_round_channels = [row["channel_id"] for row in first[:3]]
    assert len(set(first_round_channels)) == 3


def test_subtitle_fetch_restores_historical_ytdlp_vtt_route(monkeypatch):
    captured = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=False):
            assert download is True
            path = Path(captured["outtmpl"].replace("%(ext)s", "es.vtt"))
            path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:03.000\n"
                "Esta es una transcripción suficientemente extensa para la prueba.\n",
                encoding="utf-8",
            )
            return {
                "title": "Video uno",
                "channel_id": "canal-1",
                "subtitles": {"es": [{"ext": "vtt"}]},
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    record = fetch_youtube_subtitles(
        {"video_id": "v1", "url": "https://youtube.invalid/v1"},
        sleep_min_seconds=0,
        sleep_max_seconds=0,
        minimum_transcript_characters=20,
        use_transcript_api_fallback=False,
    )

    assert captured["skip_download"] is True
    assert captured["writesubtitles"] is True
    assert captured["writeautomaticsub"] is True
    assert captured["subtitlesformat"] == "vtt"
    assert captured["sleep_interval_requests"] == 0
    assert record["subtitle_source"] == "manual-yt-dlp-vtt"
    assert record["segments"][0]["text"].startswith("Esta es una transcripción")


def test_rate_limit_excludes_only_affected_channel_and_continues(tmp_path):
    candidates = [
        {"video_id": "c1-fails", "channel_id": "c1"},
        {"video_id": "c2-ok-1", "channel_id": "c2"},
        {"video_id": "c1-deferred", "channel_id": "c1"},
        {"video_id": "c2-ok-2", "channel_id": "c2"},
    ]
    attempted = []

    def fetcher(candidate):
        attempted.append(candidate["video_id"])
        if candidate["channel_id"] == "c1":
            raise RuntimeError("HTTP 429 Too Many Requests")
        return {
            "video_id": candidate["video_id"],
            "segments": [{"start": 0.0, "duration": 1.0, "text": "texto"}],
        }

    stats = ingest_incremental(
        candidates,
        tmp_path / "transcripts.jsonl",
        tmp_path / "cache",
        fetcher=fetcher,
    )

    assert attempted == ["c1-fails", "c2-ok-1", "c2-ok-2"]
    assert stats["failed"] == 1
    assert stats["fetched"] == 2
    assert stats["failure_rate_limited"] == 1
    assert stats["rate_limited_channels"] == 1
    assert stats["deferred_rate_limit"] == 1


def test_rate_limit_without_channel_identity_does_not_block_other_candidates(tmp_path):
    candidates = [{"video_id": "unknown-fails"}, {"video_id": "unknown-ok"}]

    def fetcher(candidate):
        if candidate["video_id"] == "unknown-fails":
            raise RuntimeError("HTTP 429 Too Many Requests")
        return {
            "video_id": candidate["video_id"],
            "segments": [{"start": 0.0, "duration": 1.0, "text": "texto"}],
        }

    stats = ingest_incremental(
        candidates,
        tmp_path / "transcripts.jsonl",
        tmp_path / "cache",
        fetcher=fetcher,
    )

    assert stats["failed"] == 1
    assert stats["fetched"] == 1
    assert stats["rate_limited_channels"] == 0
    assert stats["deferred_rate_limit"] == 0


def test_acquisition_error_categories_cover_expected_youtube_failures():
    assert classify_acquisition_error(RuntimeError("members-only content")) == "members_only"
    assert classify_acquisition_error(RuntimeError("video unavailable")) == "unavailable_or_private"
    assert classify_acquisition_error(RuntimeError("no tiene subtítulos")) == "no_spanish_subtitles"
    assert classify_acquisition_error(RuntimeError("subtítulos insuficientes")) == (
        "subtitle_too_short"
    )
    assert classify_acquisition_error(RuntimeError("HTTP Error 404: Not Found")) == (
        "stale_channel_or_no_videos_tab"
    )
    assert classify_acquisition_error(RuntimeError("Premieres in 15 minutes")) == (
        "scheduled_or_upcoming"
    )


def test_discovery_reports_progress_and_captures_channel_errors(monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=False):
            assert download is False
            if "bad-channel" in url:
                self.options["logger"].error("HTTP Error 404: Not Found")
                return None
            return {"entries": [{"id": "v1", "title": "Video uno"}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    progress = []
    candidates, failures = discover_youtube_candidates(
        [
            {"name": "Canal correcto", "url": "https://youtube.invalid/good-channel"},
            {"name": "Canal obsoleto", "url": "https://youtube.invalid/bad-channel"},
        ],
        progress_callback=progress.append,
    )

    assert [candidate["video_id"] for candidate in candidates] == ["v1"]
    assert failures[0]["failure_kind"] == "stale_channel_or_no_videos_tab"
    assert [event["status"] for event in progress] == [
        "started",
        "ok",
        "started",
        "failed",
    ]
    assert progress[0]["source"] == "Canal correcto"
    assert progress[-1]["candidates_unique"] == 1


def test_discovery_checkpoints_each_source_and_resumes_without_network(
    monkeypatch, tmp_path
):
    calls = []

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=False):
            calls.append(url)
            assert self.options["socket_timeout"] == 17
            return {"entries": [{"id": f"v{len(calls)}", "title": "Video"}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    checkpoint = tmp_path / "discovery.json"
    sources = [
        {"name": "Canal uno", "url": "https://youtube.invalid/uno"},
        {"name": "Canal dos", "url": "https://youtube.invalid/dos"},
    ]
    first, failures = discover_youtube_candidates(
        sources,
        socket_timeout_seconds=17,
        checkpoint_path=checkpoint,
    )
    assert [row["video_id"] for row in first] == ["v1", "v2"]
    assert failures == []
    assert checkpoint.is_file()
    assert len(json.loads(checkpoint.read_text(encoding="utf-8"))["sources"]) == 2

    progress = []
    second, failures = discover_youtube_candidates(
        sources,
        socket_timeout_seconds=17,
        checkpoint_path=checkpoint,
        progress_callback=progress.append,
    )
    assert [row["video_id"] for row in second] == ["v1", "v2"]
    assert failures == []
    assert len(calls) == 2
    assert [event["resumed"] for event in progress if event["status"] != "started"] == [
        True,
        True,
    ]


def test_discovery_resumes_completed_source_after_interruption(monkeypatch, tmp_path):
    calls = Counter()
    interrupted = True

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def extract_info(self, url, download=False):
            nonlocal interrupted
            calls[url] += 1
            if url.endswith("/dos/videos") and interrupted:
                interrupted = False
                raise KeyboardInterrupt
            return {"entries": [{"id": url.split("/")[-2], "title": "Video"}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    checkpoint = tmp_path / "discovery.json"
    sources = [
        {"name": "Canal uno", "url": "https://youtube.invalid/uno"},
        {"name": "Canal dos", "url": "https://youtube.invalid/dos"},
    ]
    with pytest.raises(KeyboardInterrupt):
        discover_youtube_candidates(sources, checkpoint_path=checkpoint)

    resumed, failures = discover_youtube_candidates(sources, checkpoint_path=checkpoint)
    assert [row["video_id"] for row in resumed] == ["uno", "dos"]
    assert failures == []
    assert calls["https://youtube.invalid/uno/videos"] == 1
    assert calls["https://youtube.invalid/dos/videos"] == 2


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
