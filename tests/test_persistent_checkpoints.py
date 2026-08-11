from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from moderacion_peru.persistent_checkpoints import (
    build_persistent_checkpoint_callback,
    persist_trainer_checkpoint,
    restore_latest_trainer_checkpoint,
)


def _checkpoint(root, step: int, payload: str = "pesos", *, epoch: float | None = None):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": epoch}), encoding="utf-8"
    )
    (checkpoint / "model.safetensors").write_text(payload, encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    nested = checkpoint / "rng_state"
    nested.mkdir()
    (nested / "state.bin").write_bytes(bytes([step % 256]))
    return checkpoint


def test_checkpoint_is_persisted_atomically_and_restored(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "source" / "trainer", 120, "modelo-120", epoch=2.0
    )
    persistent = tmp_path / "drive" / "trainer_checkpoints"

    manifest = persist_trainer_checkpoint(checkpoint, persistent)

    assert manifest["step"] == 120
    assert manifest["epoch"] == 2.0
    assert (persistent / "checkpoint-120.tar").is_file()
    assert (persistent / "checkpoint-120.json").is_file()
    assert (
        json.loads((persistent / "latest.json").read_text(encoding="utf-8"))["archive"][
            "sha256"
        ]
        == manifest["archive"]["sha256"]
    )
    assert not list(persistent.glob("*.partial"))

    restored = restore_latest_trainer_checkpoint(
        persistent, tmp_path / "new_runtime" / "trainer"
    )

    assert restored is not None
    assert restored.name == "checkpoint-120"
    assert (restored / "model.safetensors").read_text(encoding="utf-8") == "modelo-120"
    assert (restored / "optimizer.pt").read_bytes() == b"optimizer"
    assert (restored / "scheduler.pt").read_bytes() == b"scheduler"
    assert (restored / "rng_state.pth").read_bytes() == b"rng"
    assert (restored / "rng_state" / "state.bin").read_bytes() == bytes([120])


def test_restore_falls_back_to_previous_verified_checkpoint(tmp_path):
    trainer = tmp_path / "source" / "trainer"
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    persist_trainer_checkpoint(_checkpoint(trainer, 10, "válido"), persistent)
    persist_trainer_checkpoint(_checkpoint(trainer, 20, "se dañará"), persistent)
    (persistent / "checkpoint-20.tar").write_bytes(b"archivo truncado")

    restored = restore_latest_trainer_checkpoint(
        persistent, tmp_path / "new_runtime" / "trainer"
    )

    assert restored is not None
    assert restored.name == "checkpoint-10"
    assert (restored / "model.safetensors").read_text(encoding="utf-8") == "válido"


def test_corrupt_latest_pointer_does_not_hide_valid_checkpoint_manifest(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "source" / "trainer", 6401, "época-uno", epoch=1.0
    )
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    persist_trainer_checkpoint(checkpoint, persistent)
    latest_path = persistent / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["archive"].pop("sha256")
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    restored = restore_latest_trainer_checkpoint(
        persistent, tmp_path / "new_runtime" / "trainer"
    )

    assert restored is not None
    assert restored.name == "checkpoint-6401"
    assert (restored / "model.safetensors").read_text(encoding="utf-8") == "época-uno"
    restore_record = json.loads(
        (restored.parent / "persistent_checkpoint_restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert restore_record["integrity_source"] == "manifest_sha256"


def test_hashless_archive_is_structurally_validated_and_manifest_is_repaired(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "source" / "trainer", 6401, "época-uno", epoch=1.0
    )
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    manifest = persist_trainer_checkpoint(checkpoint, persistent)
    (persistent / "checkpoint-6401.json").unlink()
    latest_path = persistent / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["archive"].pop("sha256")
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    restored = restore_latest_trainer_checkpoint(
        persistent, tmp_path / "new_runtime" / "trainer"
    )

    assert restored is not None
    assert (restored / "model.safetensors").read_text(encoding="utf-8") == "época-uno"
    repaired = json.loads(
        (persistent / "checkpoint-6401.json").read_text(encoding="utf-8")
    )
    assert repaired["archive"]["sha256"] == manifest["archive"]["sha256"]
    restore_record = json.loads(
        (restored.parent / "persistent_checkpoint_restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert restore_record["integrity_source"] == "rebuilt_after_structural_validation"


def test_missing_archive_never_silently_restarts_training(tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "source" / "trainer", 6401, "época-uno", epoch=1.0
    )
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    persist_trainer_checkpoint(checkpoint, persistent)
    (persistent / "checkpoint-6401.tar").unlink()

    with pytest.raises(ValueError, match="no existe checkpoint-6401.tar"):
        restore_latest_trainer_checkpoint(
            persistent, tmp_path / "new_runtime" / "trainer"
        )


def test_local_newer_checkpoint_is_not_replaced(tmp_path):
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    persist_trainer_checkpoint(
        _checkpoint(tmp_path / "source" / "trainer", 10), persistent
    )
    local = _checkpoint(tmp_path / "runtime" / "trainer", 20, "más nuevo")

    restored = restore_latest_trainer_checkpoint(persistent, local.parent)

    assert restored == local
    assert (local / "model.safetensors").read_text(encoding="utf-8") == "más nuevo"
    assert (persistent / "checkpoint-20.tar").is_file()
    assert (
        json.loads((persistent / "latest.json").read_text(encoding="utf-8"))["step"]
        == 20
    )


def test_transformers_callback_persists_each_on_save_event(tmp_path, monkeypatch):
    transformers = ModuleType("transformers")

    class TrainerCallback:
        pass

    transformers.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    output = tmp_path / "runtime" / "trainer"
    _checkpoint(output, 45)
    persistent = tmp_path / "drive" / "trainer_checkpoints"
    callback = build_persistent_checkpoint_callback(persistent)
    control = object()

    returned = callback.on_save(
        SimpleNamespace(output_dir=str(output)),
        SimpleNamespace(global_step=45),
        control,
    )

    assert returned is control
    assert (persistent / "checkpoint-45.tar").is_file()
    assert (
        json.loads((persistent / "latest.json").read_text(encoding="utf-8"))["step"]
        == 45
    )
