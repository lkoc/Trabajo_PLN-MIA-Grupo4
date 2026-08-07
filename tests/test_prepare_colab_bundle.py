from __future__ import annotations

import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_colab_bundle_under_test", ROOT / "tools" / "prepare_colab_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
bundle_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle_tools)


def _fixture_bundle(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True)
    core = directory / "project_core.zip"
    chunks = directory / "chunks.jsonl.gz"
    dataset = directory / "dataset.jsonl.gz"
    core.write_bytes(b"core-v1")
    chunks.write_bytes(b"chunks-v1")
    dataset.write_bytes(b"dataset-v1")
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at": "fecha-no-identitaria",
        "taxonomy_contract": "moderacion_peru_5_salidas_v2",
        "taxonomy_version": "2.1.0",
        "core": {
            "name": core.name,
            "sha256": bundle_tools.sha256_file(core),
            "bytes": core.stat().st_size,
        },
        "inputs": {
            "chunks_v2": {
                "archive": chunks.name,
                "archive_sha256": bundle_tools.sha256_file(chunks),
                "source_sha256": "source-chunks",
            },
            "dataset_5_salidas": {
                "archive": dataset.name,
                "archive_sha256": bundle_tools.sha256_file(dataset),
                "source_sha256": "source-dataset",
            },
        },
    }
    manifest["bundle_id"] = bundle_tools.bundle_id_for_manifest(manifest)
    (directory / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def test_bundle_id_ignores_generation_date_and_verifies_every_artifact(tmp_path):
    manifest = _fixture_bundle(tmp_path / "bundle")
    changed_date = {**manifest, "generated_at": "otra-fecha"}
    assert bundle_tools.bundle_id_for_manifest(changed_date) == manifest["bundle_id"]
    verified = bundle_tools.verify_bundle_directory(tmp_path / "bundle")
    assert verified["bundle_id"] == manifest["bundle_id"]


def test_core_text_normalization_is_independent_of_platform_line_endings(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")

    assert bundle_tools.core_file_bytes(lf) == bundle_tools.core_file_bytes(crlf)


def test_publish_drive_release_is_versioned_and_idempotent(tmp_path, monkeypatch):
    local = tmp_path / "local_bundle"
    manifest = _fixture_bundle(local)

    def fake_prepare(destination, *, progress_callback=None):
        if progress_callback is not None:
            progress_callback({"status": "started", "total": 9, "stage": "prepare"})
            progress_callback({"status": "progress", "advance": 4, "stage": "prepared"})
        return {
            "destination": str(local),
            "manifest": str(local / "bundle_manifest.json"),
            **manifest,
        }

    monkeypatch.setattr(bundle_tools, "prepare", fake_prepare)
    drive = tmp_path / "ModeracionPeru_Colab"
    first = bundle_tools.publish_drive_release(local, drive)
    second = bundle_tools.publish_drive_release(local, drive)
    expected_release = drive / "bundle_releases" / str(manifest["bundle_id"])

    assert first["status"] == "published"
    assert second["status"] == "already_present"
    assert Path(first["release_dir"]) == expected_release
    assert bundle_tools.verify_bundle_directory(expected_release)["bundle_id"] == manifest["bundle_id"]
    pointer = json.loads((drive / "bundle_releases" / "latest.json").read_text(encoding="utf-8"))
    assert pointer["bundle_id"] == manifest["bundle_id"]
    assert pointer["manifest_sha256"] == bundle_tools.sha256_file(
        expected_release / "bundle_manifest.json"
    )
