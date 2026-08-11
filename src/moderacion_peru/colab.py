from __future__ import annotations

import gzip
import json
import os
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
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")


def _restore_run(drive_run_dir: Path, scratch_output_dir: Path) -> bool:
    manifest_path = drive_run_dir / "run_manifest.json"
    archive_path = drive_run_dir / "run_outputs.tar.gz"
    if not manifest_path.is_file() or not archive_path.is_file():
        return False
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = manifest.get("archive", {}).get("sha256")
    expected_bytes = int(manifest.get("archive", {}).get("bytes", 0))
    if expected_bytes > 0 and archive_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Checkpoint de Drive truncado: {archive_path} tiene "
            f"{archive_path.stat().st_size} de {expected_bytes} bytes"
        )
    if not expected or sha256_file(archive_path) != expected:
        raise ValueError(f"Checkpoint de Drive corrupto o incompleto: {archive_path}")
    _safe_extract_tar(archive_path, scratch_output_dir)
    return True


def _has_local_run_payload(scratch_output_dir: Path) -> bool:
    """Detecta trabajo local real sin contar el contexto regenerable."""

    return any(
        path.name != "colab_context.json" for path in scratch_output_dir.iterdir()
    )


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


def publish_colab_outputs(context: ColabContext) -> dict[str, Any]:
    """Publica el run y solo anuncia el TAR.GZ tras releerlo desde Drive."""

    context.drive_run_dir.mkdir(parents=True, exist_ok=True)
    local_archive = context.runtime_root / f".{context.notebook_id}.{context.run_id}.tar.gz"
    with tarfile.open(local_archive, "w:gz", compresslevel=6) as handle:
        for path in sorted(context.scratch_output_dir.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(context.scratch_output_dir))
    archive_sha = sha256_file(local_archive)
    archive_bytes = local_archive.stat().st_size
    target = context.drive_run_dir / "run_outputs.tar.gz"
    # Drive FUSE puede dejar un placeholder de 0 bytes si se renombra una copia
    # grande aún pendiente. Escribir el nombre final y publicar el manifiesto
    # solo tras una relectura completa evita declarar durable una copia ausente.
    shutil.copyfile(local_archive, target)
    sync = getattr(os, "sync", None)
    if sync is not None:
        try:
            sync()
        except OSError:
            pass
    if target.stat().st_size != archive_bytes:
        raise ValueError(
            "Google Drive no conserva el tamaño completo de run_outputs.tar.gz"
        )
    if sha256_file(target) != archive_sha:
        raise ValueError("La copia del run a Google Drive no conserva SHA-256")
    manifest = {
        "schema_version": "1.1.0",
        "published_at": datetime.now(UTC).isoformat(),
        "notebook_id": context.notebook_id,
        "run_id": context.run_id,
        "taxonomy_contract": "moderacion_peru_5_salidas_v2",
        "hardware": context.hardware,
        "archive": {
            "name": target.name,
            "sha256": archive_sha,
            "bytes": archive_bytes,
            "verification": "full_readback_after_close",
        },
    }
    write_json_atomic(context.drive_run_dir / "run_manifest.json", manifest)
    local_archive.unlink(missing_ok=True)
    return manifest


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
