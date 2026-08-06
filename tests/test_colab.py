from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from moderacion_peru import colab
from moderacion_peru.io import sha256_file
from moderacion_peru.schemas import HardwareRecord


ROOT = Path(__file__).resolve().parents[1]


def test_colab_config_syncs_only_gpu_inputs():
    config = json.loads((ROOT / "config" / "colab_l4.json").read_text(encoding="utf-8"))
    assert set(config["inputs"]) == {"chunks_v2", "dataset_5_salidas"}
    assert set(config["notebooks"]) == {"02_01", "03_02", "03_03", "03_04", "03_05", "03_06"}
    assert "datos/raw/transcripts_raw.jsonl" in config["excluded_from_drive"]
    assert all(specification["expected_gpu"] == "NVIDIA L4" for specification in config["notebooks"].values())


def test_colab_stages_declared_input_and_restores_published_run(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    bundle = drive / "bundle"
    bundle.mkdir(parents=True)
    raw = tmp_path / "dataset.jsonl"
    raw.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    archive = bundle / "dataset_5_salidas.jsonl.gz"
    with raw.open("rb") as source, gzip.open(archive, "wb") as target:
        target.write(source.read())
    manifest = {
        "taxonomy_contract": "moderacion_peru_5_salidas_v2",
        "inputs": {
            "dataset_5_salidas": {
                "source": "datos/model_ready/v2/dataset_5_salidas.jsonl",
                "archive": archive.name,
                "source_sha256": sha256_file(raw),
                "archive_sha256": sha256_file(archive),
            }
        },
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        colab,
        "resolve_device",
        lambda _: HardwareRecord(
            backend="cuda",
            requested="cuda",
            device_name="NVIDIA L4",
            torch_version="fixture",
            runtime_version="fixture",
            total_memory_bytes=24_000_000_000,
            dtype="bfloat16",
        ),
    )
    runtime = tmp_path / "runtime"
    first = colab.prepare_colab_context(
        "03_02",
        project_root=ROOT,
        drive_root=drive,
        runtime_root=runtime,
        run_id="fixture",
    )
    assert first.input("dataset_5_salidas").read_bytes() == raw.read_bytes()
    assert not (runtime / "inputs" / "datos" / "processed" / "chunks_v2.jsonl").exists()
    (first.scratch_output_dir / "checkpoint.json").write_text('{"epoch":1}\n', encoding="utf-8")
    published = colab.publish_colab_outputs(first)
    assert sha256_file(first.drive_run_dir / "run_outputs.tar.gz") == published["archive"]["sha256"]

    second_runtime = tmp_path / "runtime_second"
    second = colab.prepare_colab_context(
        "03_02",
        project_root=ROOT,
        drive_root=drive,
        runtime_root=second_runtime,
        run_id="fixture",
    )
    assert second.resumed
    assert (second.scratch_output_dir / "checkpoint.json").is_file()


def test_local_bundle_input_is_restored_verified_and_never_silently_replaced(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    bundle = root / "resultados" / "colab_bundle"
    bundle.mkdir(parents=True)
    source = tmp_path / "source.jsonl"
    source.write_text('{"chunk_id":"c1"}\n', encoding="utf-8")
    archive = bundle / "dataset.jsonl.gz"
    with source.open("rb") as raw, gzip.open(archive, "wb") as compressed:
        compressed.write(raw.read())
    entry = {
        "source": "datos/model_ready/v2/dataset.jsonl",
        "archive": archive.name,
        "source_sha256": sha256_file(source),
        "archive_sha256": sha256_file(archive),
    }
    (root / "config" / "colab_l4.json").write_text(
        json.dumps(
            {
                "manifest": "bundle_manifest.json",
                "inputs": {"dataset": {"source": entry["source"], "archive": entry["archive"]}},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "bundle_manifest.json").write_text(
        json.dumps({"inputs": {"dataset": entry}}),
        encoding="utf-8",
    )

    first = colab.prepare_local_bundle_input("dataset", project_root=root)
    second = colab.prepare_local_bundle_input("dataset", project_root=root)
    destination = root / entry["source"]
    assert first["status"] == "restored"
    assert second["status"] == "verified_existing"
    assert destination.read_bytes() == source.read_bytes()

    destination.write_text("otro contenido\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no coincide"):
        colab.prepare_local_bundle_input("dataset", project_root=root)


def test_non_l4_runtime_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        colab,
        "resolve_device",
        lambda _: HardwareRecord(backend="cuda", requested="cuda", device_name="NVIDIA T4"),
    )
    try:
        colab.prepare_colab_context(
            "03_02",
            project_root=ROOT,
            drive_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            require_l4=True,
        )
    except RuntimeError as exc:
        assert "NVIDIA L4" in str(exc)
    else:
        raise AssertionError("Una GPU distinta no debe pasar silenciosamente como L4")
