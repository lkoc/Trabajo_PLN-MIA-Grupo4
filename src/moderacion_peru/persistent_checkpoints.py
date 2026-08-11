from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json_atomic

CHECKPOINT_SCHEMA_VERSION = "1.1.0"
MODEL_CHECKPOINT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
MODEL_CHECKPOINT_INDEXES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix) or not path.name[len(prefix) :].isdigit():
        raise ValueError(f"Nombre de checkpoint inesperado: {path.name}")
    return int(path.name[len(prefix) :])


def _checkpoint_resume_problems(checkpoint: Path, expected_step: int) -> list[str]:
    """Valida los artefactos que Trainer necesita para una reanudación exacta."""

    problems: list[str] = []
    if not checkpoint.is_dir():
        return ["directorio ausente"]
    state_path = checkpoint / "trainer_state.json"
    try:
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        recorded_step = int(trainer_state["global_step"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        problems.append("trainer_state.json inválido")
    else:
        if recorded_step != expected_step:
            problems.append(f"trainer_state registra step {recorded_step}")

    has_model = any((checkpoint / name).is_file() for name in MODEL_CHECKPOINT_FILES)
    if not has_model:
        for index_name in MODEL_CHECKPOINT_INDEXES:
            index_path = checkpoint / index_name
            if not index_path.is_file():
                continue
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shards = {str(name) for name in index["weight_map"].values()}
                has_model = bool(shards) and all(
                    Path(name).name == name and (checkpoint / name).is_file()
                    for name in shards
                )
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                has_model = False
            if has_model:
                break
    if not has_model:
        problems.append("faltan pesos o adaptador del modelo")

    for name in ("optimizer.pt", "scheduler.pt"):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"falta {name}")
    rng_files = [
        path
        for path in checkpoint.glob("rng_state*.pth")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not rng_files:
        problems.append("falta estado RNG")
    return problems


def _copy_and_hash(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        while block := input_handle.read(8 * 1024 * 1024):
            digest.update(block)
            output_handle.write(block)
        output_handle.flush()
        try:
            os.fsync(output_handle.fileno())
        except OSError:
            # Google Drive FUSE no siempre implementa fsync; el puntero latest
            # solo se promueve después de cerrar y renombrar la copia parcial.
            pass
    return digest.hexdigest()


def _sync_and_verify_persistent_copy(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Fuerza el vaciado disponible y relee la copia final desde el montaje.

    Google Drive FUSE puede aceptar un ``rename`` de un archivo grande que aún
    solo existe en la caché del runtime. Por eso los checkpoints se escriben con
    su nombre final y no se publican manifiestos hasta cerrar, sincronizar y
    volver a leer íntegramente ese archivo.
    """

    sync = getattr(os, "sync", None)
    if sync is not None:
        try:
            sync()
        except OSError:
            # Algunos montajes FUSE no implementan una sincronización global;
            # la relectura completa que sigue continúa siendo obligatoria.
            pass
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise OSError(f"Drive no conservó la copia final {path.name}") from exc
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Drive devolvió {actual_bytes} bytes para {path.name}; "
            f"se esperaban {expected_bytes}"
        )
    durable_sha256 = sha256_file(path)
    if durable_sha256 != expected_sha256:
        raise ValueError(
            f"La relectura final desde Drive cambió el SHA-256 de {path.name}"
        )


def persist_trainer_checkpoint(
    checkpoint_dir: str | Path,
    persistent_dir: str | Path,
) -> dict[str, Any]:
    """Copia un checkpoint completo a almacenamiento persistente de forma atómica."""

    checkpoint = Path(checkpoint_dir)
    destination = Path(persistent_dir)
    step = _checkpoint_step(checkpoint)
    resume_problems = _checkpoint_resume_problems(checkpoint, step)
    if resume_problems:
        raise FileNotFoundError(
            f"Checkpoint de Trainer incompleto: {checkpoint}: "
            + "; ".join(resume_problems)
        )
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"trainer_state.json inválido en {checkpoint}") from exc
    epoch = trainer_state.get("epoch")
    epoch = float(epoch) if epoch is not None else None
    destination.mkdir(parents=True, exist_ok=True)

    archive_name = f"checkpoint-{step}.tar"
    target = destination / archive_name
    local_fd, local_name = tempfile.mkstemp(
        prefix=f".{archive_name}.", suffix=".local", dir=checkpoint.parent
    )
    os.close(local_fd)
    local_archive = Path(local_name)
    try:
        with tarfile.open(local_archive, "w") as handle:
            handle.add(checkpoint, arcname=checkpoint.name)
        local_sha256 = sha256_file(local_archive)
        local_bytes = local_archive.stat().st_size
        # Escribir directamente el nombre final evita que Drive FUSE deje en el
        # servidor un placeholder .partial de 0 bytes tras un rename grande.
        # Sin manifiesto, una copia interrumpida nunca se considera recuperable.
        copied_sha256 = _copy_and_hash(local_archive, target)
        if copied_sha256 != local_sha256:
            raise ValueError("La copia del checkpoint hacia Drive cambió su SHA-256")
        _sync_and_verify_persistent_copy(
            target,
            expected_sha256=local_sha256,
            expected_bytes=local_bytes,
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "saved_at": datetime.now(UTC).isoformat(),
            "step": step,
            "epoch": epoch,
            "checkpoint_name": checkpoint.name,
            "archive": {
                "name": target.name,
                "sha256": local_sha256,
                "bytes": local_bytes,
                "verification": "full_readback_after_close",
            },
        }
        # El manifiesto por checkpoint permite recuperar una versión anterior si
        # el último archivo se dañara. El puntero latest se promueve al final.
        write_json_atomic(destination / f"checkpoint-{step}.json", manifest)
        write_json_atomic(destination / "latest.json", manifest)
        return {**manifest, "persistent_dir": str(destination)}
    finally:
        local_archive.unlink(missing_ok=True)


def _persistent_manifests(persistent_dir: Path) -> list[dict[str, Any]]:
    paths = []
    latest = persistent_dir / "latest.json"
    if latest.is_file():
        paths.append(latest)
    paths.extend(
        sorted(
            persistent_dir.glob("checkpoint-*.json"),
            key=lambda path: _checkpoint_step(path.with_suffix("")),
            reverse=True,
        )
    )
    manifests: list[dict[str, Any]] = []
    # ``latest.json`` es solo un puntero redundante. Si quedó truncado o perdió
    # el SHA durante una desconexión de Drive, no debe ocultar el manifiesto
    # inmutable ``checkpoint-<step>.json`` del mismo checkpoint.
    seen: set[tuple[int, str, str]] = set()
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            step = int(manifest["step"])
            archive_name = str(manifest["archive"]["name"])
            archive_sha256 = str(manifest["archive"].get("sha256", "")).lower()
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        identity = (step, archive_name, archive_sha256)
        if identity not in seen:
            manifests.append(manifest)
            seen.add(identity)

    def restore_priority(item: dict[str, Any]) -> tuple[int, bool, bool]:
        archive = item["archive"]
        archive_name = str(archive["name"])
        archive_sha256 = str(archive.get("sha256", "")).lower()
        return (
            int(item["step"]),
            (persistent_dir / archive_name).is_file(),
            _is_sha256(archive_sha256),
        )

    return sorted(manifests, key=restore_priority, reverse=True)


def _latest_local_checkpoint(training_dir: Path) -> Path | None:
    candidates = []
    for path in training_dir.glob("checkpoint-*") if training_dir.is_dir() else []:
        try:
            step = _checkpoint_step(path)
        except ValueError:
            continue
        if not _checkpoint_resume_problems(path, step):
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def restore_latest_trainer_checkpoint(
    persistent_dir: str | Path | None,
    training_dir: str | Path,
) -> Path | None:
    """Elige lo más nuevo entre SSD local y Drive y restaura una época verificable."""

    training = Path(training_dir)
    local = _latest_local_checkpoint(training)
    local_step = _checkpoint_step(local) if local is not None else -1
    if persistent_dir is None:
        return local
    persistent = Path(persistent_dir)
    manifests = _persistent_manifests(persistent) if persistent.is_dir() else []
    if not manifests:
        if local is not None:
            persist_trainer_checkpoint(local, persistent)
        return local
    newest_persistent_step = max(int(manifest["step"]) for manifest in manifests)
    if local is not None and local_step > newest_persistent_step:
        # Migra inmediatamente un checkpoint creado antes de activar este
        # mecanismo, sin esperar hasta el siguiente on_save de Trainer.
        persist_trainer_checkpoint(local, persistent)
        return local

    failures = []
    training.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        step = int(manifest["step"])
        if step <= local_step:
            if failures:
                write_json_atomic(
                    training / "persistent_checkpoint_restore.json",
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "restored_at": datetime.now(UTC).isoformat(),
                        "source": str(local),
                        "step": local_step,
                        "integrity_source": "verified_local_checkpoint",
                        "skipped_newer_checkpoints": failures,
                    },
                )
                print(
                    "Checkpoint persistente más reciente no recuperable; "
                    f"se continuará automáticamente desde el step local {local_step}.",
                    flush=True,
                )
            return local
        archive_name = str(manifest["archive"]["name"])
        if Path(archive_name).name != archive_name:
            failures.append(f"step {step}: nombre de archivo inseguro")
            continue
        archive = persistent / archive_name
        expected_sha256 = str(manifest["archive"].get("sha256", "")).lower()
        if not archive.is_file():
            failures.append(f"step {step}: no existe {archive_name}")
            continue
        try:
            expected_bytes = int(manifest["archive"].get("bytes", 0))
            actual_bytes = archive.stat().st_size
        except (OSError, TypeError, ValueError):
            failures.append(f"step {step}: tamaño de archivo ilegible")
            continue
        if expected_bytes > 0 and actual_bytes != expected_bytes:
            failures.append(
                f"step {step}: tamaño inválido ({actual_bytes} de {expected_bytes} bytes)"
            )
            continue
        if expected_sha256 and not _is_sha256(expected_sha256):
            failures.append(f"step {step}: SHA-256 malformado")
            continue
        local_fd, local_name = tempfile.mkstemp(
            prefix=f".{archive_name}.", suffix=".restore", dir=training.parent
        )
        os.close(local_fd)
        local_archive = Path(local_name)
        try:
            copied_sha256 = _copy_and_hash(archive, local_archive)
            if copied_sha256 != expected_sha256:
                if expected_sha256:
                    failures.append(f"step {step}: SHA-256 inválido")
                    continue
                # Compatibilidad defensiva con un puntero de Drive que conservó
                # el TAR y sus metadatos, pero perdió solamente el SHA. El hash
                # se adopta después de validar el TAR y trainer_state.json.
                expected_sha256 = copied_sha256
                integrity_source = "rebuilt_after_structural_validation"
            else:
                integrity_source = "manifest_sha256"
            with tempfile.TemporaryDirectory(
                prefix=f".checkpoint-{step}.", dir=training.parent
            ) as staging_name:
                staging = Path(staging_name)
                with tarfile.open(local_archive, "r") as handle:
                    handle.extractall(staging, filter="data")
                extracted = staging / f"checkpoint-{step}"
                resume_problems = _checkpoint_resume_problems(extracted, step)
                if resume_problems:
                    failures.append(
                        f"step {step}: contenido no reanudable: "
                        + ", ".join(resume_problems)
                    )
                    continue
                trainer_state = json.loads(
                    (extracted / "trainer_state.json").read_text(encoding="utf-8")
                )
                target = training / extracted.name
                if target.exists():
                    backup = training / f".{target.name}.incomplete"
                    if backup.exists():
                        backup = training / f".{target.name}.incomplete-{os.getpid()}"
                    os.replace(target, backup)
                os.replace(extracted, target)
            if integrity_source == "rebuilt_after_structural_validation":
                repaired_manifest = {
                    **manifest,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "recovered_at": datetime.now(UTC).isoformat(),
                    "checkpoint_name": f"checkpoint-{step}",
                    "epoch": manifest.get("epoch", trainer_state.get("epoch")),
                    "archive": {
                        **manifest["archive"],
                        "name": archive_name,
                        "sha256": expected_sha256,
                        "bytes": archive.stat().st_size,
                    },
                }
                write_json_atomic(
                    persistent / f"checkpoint-{step}.json", repaired_manifest
                )
                latest_path = persistent / "latest.json"
                try:
                    latest_step = int(
                        json.loads(latest_path.read_text(encoding="utf-8"))["step"]
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    latest_step = -1
                if latest_step <= step:
                    write_json_atomic(latest_path, repaired_manifest)
            write_json_atomic(
                training / "persistent_checkpoint_restore.json",
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "restored_at": datetime.now(UTC).isoformat(),
                    "source": str(archive),
                    "sha256": expected_sha256,
                    "step": step,
                    "epoch": manifest.get("epoch"),
                    "integrity_source": integrity_source,
                    "skipped_newer_checkpoints": failures,
                },
            )
            if failures:
                print(
                    "Checkpoint más reciente no recuperable; "
                    f"se continuará automáticamente desde el step {step}. "
                    f"Versiones omitidas: {'; '.join(failures)}",
                    flush=True,
                )
            else:
                print(
                    "Checkpoint restaurado desde Google Drive: "
                    f"epoch={manifest.get('epoch')} · step={step} · {archive}",
                    flush=True,
                )
            return target
        except (OSError, tarfile.TarError, ValueError) as exc:
            failures.append(f"step {step}: {type(exc).__name__}: {exc}")
        finally:
            local_archive.unlink(missing_ok=True)
    if local is not None:
        return local
    # Si no hay una versión anterior verificable, continuar desde cero es la
    # única recuperación automática posible. Se registra de forma explícita
    # para que nunca parezca que la época dañada fue restaurada.
    failure_record = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "restored_at": datetime.now(UTC).isoformat(),
        "source": None,
        "step": None,
        "epoch": None,
        "integrity_source": "no_recoverable_checkpoint_restart_from_scratch",
        "skipped_newer_checkpoints": failures,
    }
    training.mkdir(parents=True, exist_ok=True)
    write_json_atomic(training / "persistent_checkpoint_restore.json", failure_record)
    print(
        "ADVERTENCIA: ningún checkpoint de Google Drive es recuperable; "
        "el entrenamiento se reiniciará automáticamente desde la época 1. "
        f"Versiones omitidas: {'; '.join(failures)}",
        flush=True,
    )
    return None


def build_persistent_checkpoint_callback(
    persistent_dir: str | Path | None,
) -> Any | None:
    """Crea un callback de Transformers sin importar esa dependencia al cargar el módulo."""

    if persistent_dir is None:
        return None
    from transformers import TrainerCallback

    destination = Path(persistent_dir)

    class PersistentCheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            manifest = persist_trainer_checkpoint(checkpoint, destination)
            print(
                "Checkpoint persistido en Google Drive: "
                f"epoch={manifest['epoch']} · step={manifest['step']} · "
                f"{manifest['persistent_dir']}",
                flush=True,
            )
            return control

    return PersistentCheckpointCallback()
