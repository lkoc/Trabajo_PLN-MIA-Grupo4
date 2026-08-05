from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    if not expected or sha256_file(archive_path) != expected:
        raise ValueError(f"Checkpoint de Drive corrupto o incompleto: {archive_path}")
    _safe_extract_tar(archive_path, scratch_output_dir)
    return True


def _stage_gzip(archive: Path, destination: Path, entry: dict[str, Any]) -> None:
    if sha256_file(archive) != entry["archive_sha256"]:
        raise ValueError(f"SHA-256 inválido para {archive}")
    if destination.is_file() and sha256_file(destination) == entry["source_sha256"]:
        return
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

    hardware = resolve_device("cuda")
    if hardware.backend != "cuda":
        raise RuntimeError("Este flujo requiere un runtime Colab con GPU CUDA")
    if require_l4 and "L4" not in hardware.device_name.upper():
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
    """Publica el run como un único TAR.GZ y después su manifiesto."""

    context.drive_run_dir.mkdir(parents=True, exist_ok=True)
    local_archive = context.runtime_root / f".{context.notebook_id}.{context.run_id}.tar.gz"
    with tarfile.open(local_archive, "w:gz", compresslevel=6) as handle:
        for path in sorted(context.scratch_output_dir.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(context.scratch_output_dir))
    archive_sha = sha256_file(local_archive)
    target = context.drive_run_dir / "run_outputs.tar.gz"
    temporary = context.drive_run_dir / ".run_outputs.tar.gz.partial"
    shutil.copyfile(local_archive, temporary)
    if sha256_file(temporary) != archive_sha:
        raise ValueError("La copia del run a Google Drive no conserva SHA-256")
    os.replace(temporary, target)
    manifest = {
        "schema_version": "1.0.0",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "notebook_id": context.notebook_id,
        "run_id": context.run_id,
        "taxonomy_contract": "moderacion_peru_5_salidas_v2",
        "hardware": context.hardware,
        "archive": {
            "name": target.name,
            "sha256": archive_sha,
            "bytes": target.stat().st_size,
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
