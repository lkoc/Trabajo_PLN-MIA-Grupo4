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
        source for source in code_sources if "def _ensure_expected_drive_release" in source
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
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<colab_setup>", "exec"), namespace)
    return namespace


def test_generated_colab_bootstrap_auto_publishes_once_and_then_reuses_release(tmp_path):
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
    (staging / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
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
    assert (releases / manifest["bundle_id"] / "dataset.jsonl.gz").read_bytes() == b"dataset"


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
    }
    assert "datos/raw/transcripts_raw.jsonl" in config["excluded_from_drive"]
    assert config["notebooks"]["02_01"]["requires_cuda"] is False
    assert config["notebooks"]["02_01"]["expected_gpu"] is None
    assert all(
        specification["requires_cuda"] is True
        and specification["expected_gpu"] == "NVIDIA L4"
        for notebook_id, specification in config["notebooks"].items()
        if notebook_id != "02_01"
    )


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
