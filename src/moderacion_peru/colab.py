from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .device import resolve_device
from .io import sha256_file, write_json_atomic


@dataclass
class ColabContext:
    notebook_id: str
    run_id: str
    drive_root: Path
    runtime_root: Path
    project_root: Path
    input_paths: dict[str, Path]
    scratch_output_dir: Path
    drive_run_dir: Path
    hardware: dict[str, Any]
    resumed: bool = False

    def input(self, key: str) -> Path:
        try:
            return self.input_paths[key]
        except KeyError as exc:
            raise KeyError(f"El cuaderno {self.notebook_id} no declaró la entrada {key}") from exc

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("drive_root", "runtime_root", "project_root", "scratch_output_dir", "drive_run_dir"):
            payload[key] = str(payload[key])
        payload["input_paths"] = {key: str(value) for key, value in self.input_paths.items()}
        return payload


def is_colab_runtime() -> bool:
    return bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_BACKEND_VERSION"))


def load_colab_config(project_root: str | Path) -> dict[str, Any]:
    path = Path(project_root) / "config" / "colab_l4.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # Las publicaciones actuales usan TAR sin compresión porque los pesos
    # safetensors ya son prácticamente incompresibles. ``r:*`` conserva la
    # compatibilidad con los TAR.GZ publicados por versiones anteriores.
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(destination, filter="data")


def _safe_run_archive_path(drive_run_dir: Path, archive_name: str) -> Path:
    relative = Path(archive_name)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError(f"Nombre inseguro de publicación en Drive: {archive_name!r}")
    return drive_run_dir / relative


def _run_manifest_candidates(drive_run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Devuelve primero las copias verificables potencialmente más recientes."""

    paths = [drive_run_dir / "run_manifest.json"]
    publications = drive_run_dir / "publications"
    paths.extend(
        publications / name
        for name in ("run_manifest-a.json", "run_manifest-b.json", "run_manifest-legacy.json")
    )
    parsed: list[
        tuple[int, str, Path, dict[str, Any], tuple[str, str]]
    ] = []
    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            archive = manifest["archive"]
            identity = (str(archive["name"]), str(archive["sha256"]))
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
        parsed.append(
            (
                int(path == drive_run_dir / "run_manifest.json"),
                str(manifest.get("published_at", "")),
                path,
                manifest,
                identity,
            )
        )
    # Un slot puede haberse verificado justo antes de que el kernel muriera y
    # dejara ``run_manifest.json`` apuntando a la publicación anterior. La fecha
    # del manifiesto del slot debe prevalecer; el puntero solo desempata.
    parsed.sort(key=lambda item: (item[1], item[0]), reverse=True)
    candidates: list[tuple[Path, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, path, manifest, identity in parsed:
        if identity not in seen:
            candidates.append((path, manifest))
            seen.add(identity)
    return candidates


def _run_publication_is_verified(
    drive_run_dir: Path, manifest: dict[str, Any]
) -> bool:
    """Comprueba completamente una publicación antes de usar o reemplazar slots."""

    try:
        archive_entry = manifest["archive"]
        archive_path = _safe_run_archive_path(
            drive_run_dir, str(archive_entry["name"])
        )
        expected_bytes = int(archive_entry["bytes"])
        expected_sha256 = str(archive_entry["sha256"]).lower()
        return (
            expected_bytes > 0
            and len(expected_sha256) == 64
            and archive_path.stat().st_size == expected_bytes
            and sha256_file(archive_path) == expected_sha256
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _restore_run(drive_run_dir: Path, scratch_output_dir: Path) -> bool:
    failures: list[str] = []
    for manifest_path, manifest in _run_manifest_candidates(drive_run_dir):
        archive_entry = manifest.get("archive", {})
        archive_name = str(archive_entry.get("name") or "run_outputs.tar.gz")
        try:
            archive_path = _safe_run_archive_path(drive_run_dir, archive_name)
            expected = str(archive_entry.get("sha256", "")).lower()
            expected_bytes = int(archive_entry.get("bytes", 0))
            actual_bytes = archive_path.stat().st_size
            if expected_bytes > 0 and actual_bytes != expected_bytes:
                raise ValueError(
                    f"tamaño {actual_bytes} de {expected_bytes} bytes"
                )
            if len(expected) != 64 or sha256_file(archive_path) != expected:
                raise ValueError("SHA-256 ausente o inválido")
            with tempfile.TemporaryDirectory(
                prefix=f".{scratch_output_dir.name}.restore-",
                dir=scratch_output_dir.parent,
            ) as staging_name:
                staging = Path(staging_name)
                _safe_extract_tar(archive_path, staging)
                for restored_path in staging.iterdir():
                    destination = scratch_output_dir / restored_path.name
                    if destination.exists():
                        if restored_path.name == "colab_context.json":
                            # El contexto se regenera al final de
                            # prepare_colab_context y no contiene trabajo.
                            continue
                        raise FileExistsError(
                            f"La restauración intentaría reemplazar {destination}"
                        )
                    os.replace(restored_path, destination)
            if failures:
                write_json_atomic(
                    scratch_output_dir / "colab_run_restore.json",
                    {
                        "schema_version": "1.2.0",
                        "restored_at": datetime.now(UTC).isoformat(),
                        "manifest": str(manifest_path),
                        "archive": str(archive_path),
                        "skipped_publications": failures,
                    },
                )
            return True
        except (OSError, ValueError, tarfile.TarError) as exc:
            failures.append(f"{manifest_path.name}: {type(exc).__name__}: {exc}")
    if failures:
        raise ValueError(
            "Ninguna publicación del run en Drive es recuperable: "
            + "; ".join(failures)
        )
    return False


def _has_local_run_payload(scratch_output_dir: Path) -> bool:
    """Detecta trabajo local real sin contar el contexto regenerable."""

    return any(
        path.name != "colab_context.json" for path in scratch_output_dir.iterdir()
    )


def restore_colab_run_outputs(
    drive_root: str | Path,
    *,
    notebook_id: str,
    run_id: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Restaura en un directorio auxiliar una publicación verificable de otro run.

    Se usa para warm-start entre cuadernos (por ejemplo, 03_05→03_06) sin copiar
    pesos al repositorio ni reemplazar archivos que ya existen en el runtime.
    """

    safe_component = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
    for label, value in (("notebook_id", notebook_id), ("run_id", run_id)):
        if not safe_component.fullmatch(value):
            raise ValueError(f"{label} contiene una ruta o identificador inseguro")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    if _has_local_run_payload(target):
        return {
            "status": "existing_runtime_copy_kept_candidate_will_be_verified",
            "source": str(Path(drive_root) / "runs" / notebook_id / run_id),
            "destination": str(target),
        }
    source = Path(drive_root) / "runs" / notebook_id / run_id
    if not _restore_run(source, target):
        raise FileNotFoundError(
            f"No existe una publicación verificable para {notebook_id}/{run_id}"
        )
    return {
        "status": "restored_and_sha256_verified",
        "source": str(source),
        "destination": str(target),
    }


def _stage_gzip(
    archive: Path,
    destination: Path,
    entry: dict[str, Any],
    *,
    replace_mismatch: bool = True,
) -> str:
    if sha256_file(archive) != entry["archive_sha256"]:
        raise ValueError(f"SHA-256 inválido para {archive}")
    if destination.is_file():
        if sha256_file(destination) == entry["source_sha256"]:
            return "verified_existing"
        if not replace_mismatch:
            raise ValueError(
                f"{destination} existe, pero no coincide con el checkpoint comprimido. "
                "Actualice el bundle o retire deliberadamente la copia derivada antes de restaurar."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with gzip.open(archive, "rb") as source, os.fdopen(fd, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary_name) != entry["source_sha256"]:
            raise ValueError(f"La descompresión no reproduce {entry['source']}")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return "restored"


def prepare_local_bundle_input(
    input_key: str,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Verifica o restaura una entrada local desde el bundle sincronizado."""

    project = Path(project_root).resolve()
    config = load_colab_config(project)
    if input_key not in config["inputs"]:
        raise KeyError(f"Entrada de bundle desconocida: {input_key}")
    bundle_dir = project / "resultados" / "colab_bundle"
    manifest_path = bundle_dir / config["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto del bundle sincronizado: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    configured = config["inputs"][input_key]
    entry = manifest.get("inputs", {}).get(input_key)
    if not entry:
        raise KeyError(f"El manifiesto no declara la entrada {input_key}")
    if configured["source"] != entry.get("source") or configured["archive"] != entry.get("archive"):
        raise ValueError(f"El manifiesto y config/colab_l4.json difieren para {input_key}")
    archive = bundle_dir / configured["archive"]
    destination = project / configured["source"]
    status = _stage_gzip(
        archive,
        destination,
        entry,
        replace_mismatch=False,
    )
    actual_sha256 = sha256_file(destination)
    if actual_sha256 != entry["source_sha256"]:
        raise ValueError(f"La entrada local {input_key} no coincide después de restaurarla")
    return {
        "status": status,
        "input_key": input_key,
        "path": destination.as_posix(),
        "sha256": actual_sha256,
        "bytes": destination.stat().st_size,
        "archive": archive.as_posix(),
    }


def prepare_colab_context(
    notebook_id: str,
    *,
    project_root: str | Path,
    drive_root: str | Path,
    runtime_root: str | Path = "/content/moderacion_peru",
    run_id: str | None = None,
    require_l4: bool = True,
    resume: bool = True,
) -> ColabContext:
    """Valida la L4, copia solo las entradas declaradas y restaura un run publicado."""

    project = Path(project_root).resolve()
    drive = Path(drive_root)
    runtime = Path(runtime_root)
    config = load_colab_config(project)
    if notebook_id not in config["notebooks"]:
        raise ValueError(f"El cuaderno {notebook_id} no está habilitado para Colab")

    notebook_config = config["notebooks"][notebook_id]
    requires_cuda = bool(notebook_config.get("requires_cuda", True))
    hardware = resolve_device("cuda" if requires_cuda else "auto")
    if requires_cuda and hardware.backend != "cuda":
        raise RuntimeError("Este flujo requiere un runtime Colab con GPU CUDA")
    if requires_cuda and require_l4 and "L4" not in hardware.device_name.upper():
        raise RuntimeError(
            f"Se esperaba NVIDIA L4 y Colab asignó {hardware.device_name}. "
            "Cambie el tipo de runtime o establezca COLAB_REQUIRE_L4=False de forma explícita."
        )

    bundle_dir = drive / config["bundle_folder"]
    manifest_path = bundle_dir / config["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto del bundle: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("taxonomy_contract") != config["taxonomy_contract"]:
        raise ValueError("El bundle de Drive pertenece a otro contrato taxonómico")

    input_paths: dict[str, Path] = {}
    for key in config["notebooks"][notebook_id]["input_keys"]:
        configured = config["inputs"][key]
        entry = manifest["inputs"][key]
        if configured["source"] != entry["source"]:
            raise ValueError(f"Ruta inesperada para la entrada {key}")
        if configured["archive"] != entry["archive"]:
            raise ValueError(f"Archivo comprimido inesperado para la entrada {key}")
        destination = runtime / "inputs" / configured["source"]
        _stage_gzip(bundle_dir / entry["archive"], destination, entry)
        input_paths[key] = destination

    active_run_id = run_id or os.getenv("MODPERU_RUN_ID") or f"{notebook_id}_working_v2_1"
    scratch = runtime / "runs" / notebook_id / active_run_id
    drive_run = drive / config["runs_folder"] / notebook_id / active_run_id
    scratch.mkdir(parents=True, exist_ok=True)
    if resume and _has_local_run_payload(scratch):
        # Al reejecutar el bootstrap dentro del mismo kernel, el SSD puede
        # contener checkpoints más nuevos que el TAR publicado anteriormente.
        # No se deben sobrescribir con la copia histórica de Drive.
        resumed = True
        print(
            "Se conservará el run local del runtime actual; "
            "no se superpondrá el TAR histórico de Google Drive.",
            flush=True,
        )
    else:
        resumed = _restore_run(drive_run, scratch) if resume else False
    context = ColabContext(
        notebook_id=notebook_id,
        run_id=active_run_id,
        drive_root=drive,
        runtime_root=runtime,
        project_root=project,
        input_paths=input_paths,
        scratch_output_dir=scratch,
        drive_run_dir=drive_run,
        hardware=hardware.model_dump(),
        resumed=resumed,
    )
    write_json_atomic(scratch / "colab_context.json", context.as_dict())
    return context


def _include_in_run_publication(relative: Path) -> bool:
    """Excluye estados reanudables que ya se reflejan por separado en Drive."""

    return relative.as_posix() != "colab_context.json" and not any(
        part.startswith("checkpoint-")
        or part in {"trainer", "trainer_gate", "trainer_damage", "trainer_branch"}
        for part in relative.parts
    )


def publish_colab_outputs(context: ColabContext) -> dict[str, Any]:
    """Publica artefactos finales en dos ranuras sin invalidar la copia anterior.

    Los checkpoints completos de ``Trainer`` viven en ``trainer_checkpoints`` y
    no se vuelven a comprimir aquí. Esto evita TAR.GZ de varios GB y permite que
    la publicación final sea rápida. Cada intento escribe en la ranura inactiva;
    el puntero ``run_manifest.json`` cambia solo tras releer el archivo completo
    desde Drive.
    """

    context.drive_run_dir.mkdir(parents=True, exist_ok=True)
    publications = context.drive_run_dir / "publications"
    publications.mkdir(parents=True, exist_ok=True)
    active_slot = None
    for _, candidate in _run_manifest_candidates(context.drive_run_dir):
        candidate_slot = candidate.get("publication_slot")
        if candidate_slot in {"a", "b"} and _run_publication_is_verified(
            context.drive_run_dir, candidate
        ):
            active_slot = candidate_slot
            break
    slot = "b" if active_slot == "a" else "a"
    archive_name = f"publications/run_outputs-{slot}.tar"
    manifest_name = f"run_manifest-{slot}.json"
    local_archive = context.runtime_root / f".{context.notebook_id}.{context.run_id}.{slot}.tar"
    included_files = 0
    excluded_files = 0
    try:
        with tarfile.open(local_archive, "w") as handle:
            for path in sorted(context.scratch_output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(context.scratch_output_dir)
                if not _include_in_run_publication(relative):
                    excluded_files += 1
                    continue
                handle.add(path, arcname=relative)
                included_files += 1
        archive_sha = sha256_file(local_archive)
        archive_bytes = local_archive.stat().st_size
        target = context.drive_run_dir / archive_name
        # Se escribe el nombre final de la ranura inactiva. Si el runtime muere,
        # el manifiesto activo sigue apuntando a la ranura anterior verificada.
        shutil.copyfile(local_archive, target)
        sync = getattr(os, "sync", None)
        if sync is not None:
            try:
                sync()
            except OSError:
                pass
        if target.stat().st_size != archive_bytes:
            raise ValueError(
                f"Google Drive no conserva el tamaño completo de {archive_name}"
            )
        if sha256_file(target) != archive_sha:
            raise ValueError("La copia del run a Google Drive no conserva SHA-256")
        manifest = {
            "schema_version": "1.2.0",
            "published_at": datetime.now(UTC).isoformat(),
            "notebook_id": context.notebook_id,
            "run_id": context.run_id,
            "taxonomy_contract": "moderacion_peru_5_salidas_v2",
            "hardware": context.hardware,
            "publication_slot": slot,
            "included_files": included_files,
            "excluded_trainer_checkpoint_files": excluded_files,
            "trainer_checkpoint_storage": "trainer_checkpoints",
            "archive": {
                "name": archive_name,
                "sha256": archive_sha,
                "bytes": archive_bytes,
                "format": "tar_uncompressed",
                "verification": "full_readback_after_close",
            },
        }
        slot_manifest = publications / manifest_name
        write_json_atomic(slot_manifest, manifest)
        # En la primera migración conserva el manifiesto del TAR.GZ histórico
        # como tercera copia recuperable. No se modifica su archivo grande.
        legacy_manifest = publications / "run_manifest-legacy.json"
        active_manifest = context.drive_run_dir / "run_manifest.json"
        if active_manifest.is_file() and not legacy_manifest.exists():
            try:
                previous = json.loads(active_manifest.read_text(encoding="utf-8"))
                if previous.get("publication_slot") in {"a", "b"}:
                    raise ValueError("el puntero previo ya pertenece al esquema por slots")
                previous_archive = str(
                    previous.get("archive", {}).get("name") or "run_outputs.tar.gz"
                )
                previous_path = _safe_run_archive_path(
                    context.drive_run_dir, previous_archive
                )
                previous_sha256 = str(
                    previous.get("archive", {}).get("sha256", "")
                ).lower()
                previous_bytes = int(previous.get("archive", {}).get("bytes", 0))
                if (
                    previous_path.is_file()
                    and previous_bytes > 0
                    and previous_path.stat().st_size == previous_bytes
                    and len(previous_sha256) == 64
                    and sha256_file(previous_path) == previous_sha256
                ):
                    write_json_atomic(legacy_manifest, previous)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # La publicación nueva ya fue verificada; un legado inválido no
                # debe impedir promoverla ni presentarse como recuperable.
                pass
        # El puntero activo se promueve al final; nunca anuncia una copia parcial.
        write_json_atomic(active_manifest, manifest)
        return manifest
    finally:
        local_archive.unlink(missing_ok=True)


def colab_runtime_diagnostics() -> dict[str, Any]:
    hardware = resolve_device("auto").model_dump()
    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        nvidia_smi = None
    return {
        "is_colab": is_colab_runtime(),
        "hardware": hardware,
        "nvidia_smi": nvidia_smi,
        "cwd": str(Path.cwd()),
        "free_runtime_bytes": shutil.disk_usage("/content" if Path("/content").exists() else Path.cwd()).free,
    }
