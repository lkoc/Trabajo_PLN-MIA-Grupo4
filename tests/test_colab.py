from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from moderacion_peru import colab
from moderacion_peru.device import cuda_performance_profile, high_memory_bf16_cuda
from moderacion_peru.io import sha256_file
from moderacion_peru.schemas import HardwareRecord

ROOT = Path(__file__).resolve().parents[1]


def _generated_colab_bundle_functions() -> dict[str, object]:
    notebook = json.loads(
        (ROOT / "flujo/03_entrenamiento/03_02_transformers_planos.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_sources = [
        "".join(cell["source"])
        if isinstance(cell["source"], list)
        else str(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    setup = next(
        source
        for source in code_sources
        if "def _ensure_expected_drive_release" in source
    )
    tree = ast.parse(setup)
    function_names = {
        "_sha256",
        "_read_manifest",
        "_bundle_id_for_manifest",
        "_bundle_specs",
        "_verify_expected_bundle",
        "_bundle_is_current",
        "_write_latest_pointer",
        "_publish_expected_bundle",
        "_ensure_expected_drive_release",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    namespace = {
        "Path": Path,
        "datetime": datetime,
        "timezone": timezone,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "shutil": shutil,
        "uuid": uuid,
        "COLAB_NOTEBOOK_BUILD_BUNDLE_ID": "fixture-pending",
        "COLAB_EXPECTED_CORE_SHA256": "fixture-pending",
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<colab_setup>", "exec"),
        namespace,
    )
    return namespace


def test_generated_colab_bootstrap_auto_publishes_once_and_then_reuses_release(
    tmp_path,
):
    functions = _generated_colab_bundle_functions()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "project_core.zip").write_bytes(b"core")
    (staging / "dataset.jsonl.gz").write_bytes(b"dataset")
    manifest = {
        "schema_version": "1.0.0",
        "taxonomy_contract": "fixture",
        "taxonomy_version": "1.0.0",
        "core": {
            "name": "project_core.zip",
            "sha256": sha256_file(staging / "project_core.zip"),
        },
        "inputs": {
            "dataset": {
                "archive": "dataset.jsonl.gz",
                "archive_sha256": sha256_file(staging / "dataset.jsonl.gz"),
                "source_sha256": "fixture-source",
            }
        },
    }
    manifest["bundle_id"] = functions["_bundle_id_for_manifest"](manifest)
    (staging / "bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    functions["COLAB_NOTEBOOK_BUILD_BUNDLE_ID"] = manifest["bundle_id"]
    functions["COLAB_EXPECTED_CORE_SHA256"] = manifest["core"]["sha256"]
    functions["_verify_expected_bundle"].__defaults__ = (manifest["bundle_id"],)
    functions["COLAB_AUTO_PUBLISH_MISSING_BUNDLE"] = True
    releases = tmp_path / "bundle_releases"

    first = functions["_publish_expected_bundle"](staging, releases)
    functions["_acquire_expected_bundle"] = lambda: pytest.fail(
        "No debe volver a adquirir un release ya verificado"
    )
    second = functions["_ensure_expected_drive_release"](releases)

    assert first["status"] == "auto_published_and_verified"
    assert second["status"] == "already_present_and_verified"
    pointer = json.loads((releases / "latest.json").read_text(encoding="utf-8"))
    assert pointer["bundle_id"] == manifest["bundle_id"]
    assert (
        releases / manifest["bundle_id"] / "dataset.jsonl.gz"
    ).read_bytes() == b"dataset"


def test_colab_config_syncs_only_declared_inputs_and_keeps_api_on_cpu():
    config = json.loads((ROOT / "config" / "colab_l4.json").read_text(encoding="utf-8"))
    assert set(config["inputs"]) == {
        "chunks_v2",
        "chunks_deepseek_historicos",
        "deepseek_flash_historico",
        "deepseek_pro_historico_principal",
        "deepseek_pro_historico_umbral",
        "deepseek_pro_historico_sospechosos",
        "dataset_5_salidas",
    }
    assert set(config["notebooks"]["02_01"]["input_keys"]) == set(config["inputs"]) - {
        "dataset_5_salidas"
    }
    assert config["notebooks"]["02_02"]["input_keys"] == ["chunks_v2"]
    assert set(config["notebooks"]) == {
        "02_01",
        "02_02",
        "03_02",
        "03_03",
        "03_03b",
        "03_04",
        "03_05",
        "03_06",
        "03_06b",
        "03_07",
    }
    assert "datos/raw/transcripts_raw.jsonl" in config["excluded_from_drive"]
    assert config["notebooks"]["02_01"]["requires_cuda"] is False
    assert config["notebooks"]["02_01"]["expected_gpu"] is None
    assert config["notebooks"]["03_07"]["requires_cuda"] is False
    assert config["notebooks"]["03_07"]["expected_gpu"] is None
    qwen_a100 = {"02_02", "03_05", "03_06", "03_06b"}
    assert all(
        config["notebooks"][notebook_id]["expected_gpu"]
        == "NVIDIA A100 40GB or equivalent CUDA BF16 GPU"
        for notebook_id in qwen_a100
    )
    assert all(
        specification["requires_cuda"] is True
        and specification["expected_gpu"] == "NVIDIA L4"
        for notebook_id, specification in config["notebooks"].items()
        if notebook_id not in {"02_01", "03_07", *qwen_a100}
    )


def test_a100_40gb_activates_high_memory_bf16_profile():
    a100 = HardwareRecord(
        backend="cuda",
        requested="cuda",
        device_name="NVIDIA A100-SXM4-40GB",
        total_memory_bytes=42_405_855_232,
        dtype="bfloat16",
    )
    l4 = HardwareRecord(
        backend="cuda",
        requested="cuda",
        device_name="NVIDIA L4",
        total_memory_bytes=24_151_572_480,
        dtype="bfloat16",
    )

    assert high_memory_bf16_cuda(a100)
    assert cuda_performance_profile(a100) == "cuda_bf16_40gb_plus"
    assert not high_memory_bf16_cuda(l4)
    assert cuda_performance_profile(l4) == "cuda_standard"


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
    (first.scratch_output_dir / "checkpoint.json").write_text(
        '{"epoch":1}\n', encoding="utf-8"
    )
    published = colab.publish_colab_outputs(first)
    archive_path = first.drive_run_dir / published["archive"]["name"]
    assert (
        sha256_file(archive_path)
        == published["archive"]["sha256"]
    )

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

    # Reejecutar el bootstrap en el mismo runtime debe conservar un checkpoint
    # local posterior, no volver a extraer encima el TAR histórico de Drive.
    (second.scratch_output_dir / "checkpoint.json").write_text(
        '{"epoch":2}\n', encoding="utf-8"
    )
    same_runtime = colab.prepare_colab_context(
        "03_02",
        project_root=ROOT,
        drive_root=drive,
        runtime_root=second_runtime,
        run_id="fixture",
    )
    assert same_runtime.resumed
    assert (same_runtime.scratch_output_dir / "checkpoint.json").read_text(
        encoding="utf-8"
    ) == '{"epoch":2}\n'


def test_restore_colab_run_outputs_reuses_verified_foreign_publication(tmp_path):
    drive = tmp_path / "drive"
    scratch = tmp_path / "scratch_03_05"
    scratch.mkdir()
    (tmp_path / "runtime").mkdir()
    (scratch / "candidate.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    drive_run = drive / "runs" / "03_05" / "03_05_working_v2_1"
    context = colab.ColabContext(
        notebook_id="03_05",
        run_id="03_05_working_v2_1",
        drive_root=drive,
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=drive_run,
        hardware={},
    )
    colab.publish_colab_outputs(context)
    destination = tmp_path / "runtime" / "warm_starts" / "03_05"

    restored = colab.restore_colab_run_outputs(
        drive,
        notebook_id="03_05",
        run_id="03_05_working_v2_1",
        destination=destination,
    )
    repeated = colab.restore_colab_run_outputs(
        drive,
        notebook_id="03_05",
        run_id="03_05_working_v2_1",
        destination=destination,
    )

    assert restored["status"] == "restored_and_sha256_verified"
    assert (destination / "candidate.json").is_file()
    assert repeated["status"] == "existing_runtime_copy_kept_candidate_will_be_verified"
    with pytest.raises(ValueError, match="inseguro"):
        colab.restore_colab_run_outputs(
            drive,
            notebook_id="../03_05",
            run_id="bad",
            destination=tmp_path / "unsafe",
        )


def test_restore_colab_run_outputs_accepts_verified_multipart_archive(tmp_path):
    drive = tmp_path / "drive"
    scratch = tmp_path / "runtime" / "runs" / "03_01" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "candidate.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8"
    )
    context = colab.ColabContext(
        notebook_id="03_01",
        run_id="fixture",
        drive_root=drive,
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=drive / "runs" / "03_01" / "fixture",
        hardware={"backend": "cpu"},
    )
    published = colab.publish_colab_outputs(context)
    archive_path = context.drive_run_dir / published["archive"]["name"]
    archive_bytes = archive_path.read_bytes()
    midpoint = len(archive_bytes) // 2
    part_payloads = (archive_bytes[:midpoint], archive_bytes[midpoint:])
    parts = []
    for index, payload in enumerate(part_payloads, start=1):
        path = archive_path.with_name(f"{archive_path.name}.part{index:03d}")
        path.write_bytes(payload)
        parts.append(
            {
                "name": str(path.relative_to(context.drive_run_dir)).replace(
                    os.sep, "/"
                ),
                "bytes": len(payload),
                "sha256": sha256_file(path),
            }
        )
    archive_path.unlink()
    multipart = json.loads(
        (context.drive_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    multipart["archive"]["parts"] = parts
    for manifest_path in (
        context.drive_run_dir / "run_manifest.json",
        context.drive_run_dir
        / "publications"
        / f"run_manifest-{published['publication_slot']}.json",
    ):
        manifest_path.write_text(json.dumps(multipart), encoding="utf-8")

    assert colab._run_publication_is_verified(context.drive_run_dir, multipart)
    restored = tmp_path / "restored_multipart"
    restored.mkdir()
    assert colab._restore_run(context.drive_run_dir, restored)
    assert json.loads((restored / "candidate.json").read_text(encoding="utf-8")) == {
        "status": "complete"
    }

    first_part = context.drive_run_dir / parts[0]["name"]
    first_part.write_bytes(b"altered" + first_part.read_bytes())
    assert not colab._run_publication_is_verified(context.drive_run_dir, multipart)


def test_publish_colab_outputs_uses_verified_multipart_for_large_archive(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "runtime" / "runs" / "03_06" / "fixture"
    scratch.mkdir(parents=True)
    payload = b"candidate-payload" * 2048
    (scratch / "candidate.bin").write_bytes(payload)
    context = colab.ColabContext(
        notebook_id="03_06",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_06" / "fixture",
        hardware={"backend": "cuda", "device_name": "NVIDIA A100"},
    )
    monkeypatch.setattr(colab, "RUN_PUBLICATION_MULTIPART_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(colab, "RUN_PUBLICATION_PART_BYTES", 4096)

    published = colab.publish_colab_outputs(context)

    archive = published["archive"]
    assert archive["format"] == "tar_uncompressed_multipart"
    assert archive["verification"] == "per_part_and_concatenated_full_readback"
    assert len(archive["parts"]) > 1
    assert not (context.drive_run_dir / archive["name"]).exists()
    assert sum(part["bytes"] for part in archive["parts"]) == archive["bytes"]
    assert colab._run_publication_is_verified(context.drive_run_dir, published)

    restored = tmp_path / "restored_automatic_multipart"
    restored.mkdir()
    assert colab._restore_run(context.drive_run_dir, restored)
    assert (restored / "candidate.bin").read_bytes() == payload


def test_invalid_publication_can_rebuild_only_with_persistent_checkpoints(tmp_path):
    drive_run = tmp_path / "drive" / "runs" / "03_06" / "fixture"
    drive_run.mkdir(parents=True)
    invalid = {
        "schema_version": "1.2.0",
        "published_at": "2026-08-14T00:00:00+00:00",
        "archive": {
            "name": "publications/run_outputs-a.tar",
            "bytes": 1024,
            "sha256": "0" * 64,
        },
    }
    (drive_run / "run_manifest.json").write_text(
        json.dumps(invalid), encoding="utf-8"
    )
    without_checkpoints = tmp_path / "scratch_without_checkpoints"
    without_checkpoints.mkdir()
    with pytest.raises(ValueError, match="Ninguna publicación"):
        colab._restore_run_or_allow_checkpoint_rebuild(
            drive_run, without_checkpoints
        )

    checkpoint_root = (
        drive_run
        / "trainer_checkpoints"
        / "candidate"
        / "trainer"
    )
    checkpoint_root.mkdir(parents=True)
    checkpoint_archive = checkpoint_root / "checkpoint-10-fixture.tar"
    checkpoint_archive.write_bytes(b"valid-persistent-checkpoint")
    checkpoint_manifest = {
        "step": 10,
        "archive": {
            "name": checkpoint_archive.name,
            "bytes": checkpoint_archive.stat().st_size,
            "sha256": sha256_file(checkpoint_archive),
        },
    }
    (checkpoint_root / "checkpoint-10.json").write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    repair_scratch = tmp_path / "scratch_repair"
    repair_scratch.mkdir()

    assert not colab._restore_run_or_allow_checkpoint_rebuild(
        drive_run, repair_scratch
    )
    repair = json.loads(
        (repair_scratch / "colab_run_publication_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        repair["status"]
        == "final_publication_invalid_rebuild_from_persistent_checkpoints"
    )
    assert repair["trainer_checkpoint_root"] == str(
        drive_run / "trainer_checkpoints"
    )


def test_colab_run_manifest_waits_for_drive_readback(tmp_path, monkeypatch):
    scratch = tmp_path / "runtime" / "runs" / "03_02" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "checkpoint.json").write_text('{"epoch":2}\n', encoding="utf-8")
    context = colab.ColabContext(
        notebook_id="03_02",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_02" / "fixture",
        hardware={"backend": "cuda", "device_name": "NVIDIA L4"},
    )
    real_sha256_file = colab.sha256_file

    def corrupt_drive_readback(path):
        path = Path(path)
        if path.parent == context.drive_run_dir / "publications" and path.suffix == ".tar":
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(colab, "sha256_file", corrupt_drive_readback)

    with pytest.raises(ValueError, match="no conserva SHA-256"):
        colab.publish_colab_outputs(context)

    assert len(list((context.drive_run_dir / "publications").glob("run_outputs-*.tar"))) == 1
    assert not (context.drive_run_dir / "run_manifest.json").exists()


def test_colab_publication_uses_redundant_slots_and_excludes_trainer_state(tmp_path):
    scratch = tmp_path / "runtime" / "runs" / "03_05" / "fixture"
    (scratch / "runs" / "candidate" / "trainer" / "checkpoint-10").mkdir(
        parents=True
    )
    (scratch / "runs" / "candidate" / "candidate.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8"
    )
    (scratch / "runs" / "candidate" / "trainer" / "checkpoint-10" / "optimizer.pt").write_bytes(
        b"large transient state"
    )
    context = colab.ColabContext(
        notebook_id="03_05",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_05" / "fixture",
        hardware={"backend": "cuda", "device_name": "NVIDIA A100"},
    )

    first = colab.publish_colab_outputs(context)
    (scratch / "result.json").write_text('{"version":2}\n', encoding="utf-8")
    second = colab.publish_colab_outputs(context)

    assert first["publication_slot"] == "a"
    assert second["publication_slot"] == "b"
    assert (context.drive_run_dir / first["archive"]["name"]).is_file()
    assert (context.drive_run_dir / second["archive"]["name"]).is_file()
    active = json.loads(
        (context.drive_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert active["archive"]["sha256"] == second["archive"]["sha256"]
    restored = tmp_path / "restored"
    restored.mkdir()
    assert colab._restore_run(context.drive_run_dir, restored)
    assert (restored / "runs" / "candidate" / "candidate.json").is_file()
    assert not (restored / "runs" / "candidate" / "trainer").exists()


def test_restore_falls_back_to_previous_publication_slot(tmp_path):
    scratch = tmp_path / "runtime" / "runs" / "03_06" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "candidate.json").write_text('{"version":1}\n', encoding="utf-8")
    context = colab.ColabContext(
        notebook_id="03_06",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_06" / "fixture",
        hardware={"backend": "cuda"},
    )
    first = colab.publish_colab_outputs(context)
    (scratch / "candidate.json").write_text('{"version":2}\n', encoding="utf-8")
    second = colab.publish_colab_outputs(context)
    (context.drive_run_dir / second["archive"]["name"]).write_bytes(b"truncated")

    restored = tmp_path / "restored"
    restored.mkdir()
    assert colab._restore_run(context.drive_run_dir, restored)
    assert json.loads((restored / "candidate.json").read_text(encoding="utf-8")) == {
        "version": 1
    }
    record = json.loads(
        (restored / "colab_run_restore.json").read_text(encoding="utf-8")
    )
    assert record["archive"].endswith(first["archive"]["name"].replace("/", os.sep))


def test_publish_preserves_verified_slot_when_active_pointer_is_corrupt(tmp_path):
    scratch = tmp_path / "runtime" / "runs" / "03_06" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "candidate.json").write_text('{"version":1}\n', encoding="utf-8")
    context = colab.ColabContext(
        notebook_id="03_06",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_06" / "fixture",
        hardware={"backend": "cuda"},
    )
    first = colab.publish_colab_outputs(context)
    first_archive = context.drive_run_dir / first["archive"]["name"]
    first_sha256 = sha256_file(first_archive)
    (context.drive_run_dir / "run_manifest.json").write_text(
        "{puntero truncado", encoding="utf-8"
    )
    (scratch / "candidate.json").write_text('{"version":2}\n', encoding="utf-8")

    second = colab.publish_colab_outputs(context)

    assert second["publication_slot"] == "b"
    assert sha256_file(first_archive) == first_sha256


def test_restore_and_publish_prefer_new_verified_slot_over_stale_pointer(tmp_path):
    scratch = tmp_path / "runtime" / "runs" / "03_06b" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "candidate.json").write_text('{"version":1}\n', encoding="utf-8")
    context = colab.ColabContext(
        notebook_id="03_06b",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_06b" / "fixture",
        hardware={"backend": "cuda"},
    )
    colab.publish_colab_outputs(context)
    first_pointer = json.loads(
        (context.drive_run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    (scratch / "candidate.json").write_text('{"version":2}\n', encoding="utf-8")
    second = colab.publish_colab_outputs(context)
    second_archive = context.drive_run_dir / second["archive"]["name"]
    second_sha256 = sha256_file(second_archive)
    # Simula una caída entre la escritura del manifiesto del slot nuevo y el
    # cambio atómico del puntero activo.
    (context.drive_run_dir / "run_manifest.json").write_text(
        json.dumps(first_pointer), encoding="utf-8"
    )

    restored = tmp_path / "restored"
    restored.mkdir()
    assert colab._restore_run(context.drive_run_dir, restored)
    assert json.loads((restored / "candidate.json").read_text(encoding="utf-8")) == {
        "version": 2
    }
    (scratch / "candidate.json").write_text('{"version":3}\n', encoding="utf-8")
    third = colab.publish_colab_outputs(context)

    assert third["publication_slot"] == "a"
    assert sha256_file(second_archive) == second_sha256


def test_first_slot_publication_preserves_verified_legacy_manifest(tmp_path):
    scratch = tmp_path / "runtime" / "runs" / "03_02" / "fixture"
    scratch.mkdir(parents=True)
    (scratch / "candidate.json").write_text('{"version":2}\n', encoding="utf-8")
    context = colab.ColabContext(
        notebook_id="03_02",
        run_id="fixture",
        drive_root=tmp_path / "drive",
        runtime_root=tmp_path / "runtime",
        project_root=ROOT,
        input_paths={},
        scratch_output_dir=scratch,
        drive_run_dir=tmp_path / "drive" / "runs" / "03_02" / "fixture",
        hardware={"backend": "cuda"},
    )
    context.drive_run_dir.mkdir(parents=True)
    legacy_archive = context.drive_run_dir / "run_outputs.tar.gz"
    with __import__("tarfile").open(legacy_archive, "w:gz") as handle:
        legacy_file = tmp_path / "legacy.json"
        legacy_file.write_text('{"version":1}\n', encoding="utf-8")
        handle.add(legacy_file, arcname="candidate.json")
    legacy = {
        "schema_version": "1.1.0",
        "archive": {
            "name": legacy_archive.name,
            "sha256": sha256_file(legacy_archive),
            "bytes": legacy_archive.stat().st_size,
        },
    }
    (context.drive_run_dir / "run_manifest.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )

    published = colab.publish_colab_outputs(context)

    preserved = json.loads(
        (
            context.drive_run_dir
            / "publications"
            / "run_manifest-legacy.json"
        ).read_text(encoding="utf-8")
    )
    assert preserved["archive"]["sha256"] == legacy["archive"]["sha256"]
    assert published["publication_slot"] == "a"


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
                "inputs": {
                    "dataset": {"source": entry["source"], "archive": entry["archive"]}
                },
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
        lambda _: HardwareRecord(
            backend="cuda", requested="cuda", device_name="NVIDIA T4"
        ),
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
