"""Construye el bundle mínimo que se sube a Google Drive para Colab."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_TEXT_SUFFIXES = {".json", ".py", ".toml", ".txt"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def core_file_bytes(path: Path) -> bytes:
    """Normaliza texto a UTF-8/LF para que el core sea idéntico entre SO."""
    payload = path.read_bytes()
    if path.suffix.lower() not in CORE_TEXT_SUFFIXES:
        return payload
    text = payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


def bundle_id_for_manifest(manifest: dict[str, object]) -> str:
    """Identidad estable del contenido; excluye fechas y ubicaciones locales."""
    core = manifest["core"]
    inputs = manifest["inputs"]
    identity = {
        "schema_version": manifest["schema_version"],
        "taxonomy_contract": manifest["taxonomy_contract"],
        "taxonomy_version": manifest["taxonomy_version"],
        "core": {
            "name": core["name"],
            "sha256": core["sha256"],
        },
        "inputs": {
            key: {
                "archive": value["archive"],
                "archive_sha256": value["archive_sha256"],
                "source_sha256": value["source_sha256"],
            }
            for key, value in sorted(inputs.items())
        },
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bundle_file_specs(manifest: dict[str, object]) -> list[tuple[str, str]]:
    specs = [(manifest["core"]["name"], manifest["core"]["sha256"])]
    specs.extend(
        (entry["archive"], entry["archive_sha256"])
        for entry in manifest.get("inputs", {}).values()
    )
    for name, expected_sha256 in specs:
        if Path(name).name != name or not expected_sha256:
            raise ValueError(f"Entrada insegura o incompleta en bundle_manifest.json: {name!r}")
    return [(str(name), str(expected_sha256)) for name, expected_sha256 in specs]


def latest_pointer_for_release(release_dir: Path, manifest: dict[str, object]) -> dict[str, str]:
    return {
        "schema_version": "1.0.0",
        "bundle_id": str(manifest["bundle_id"]),
        "core_sha256": str(manifest["core"]["sha256"]),
        "manifest_sha256": sha256_file(release_dir / "bundle_manifest.json"),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_bundle_directory(directory: str | Path, expected_bundle_id: str | None = None) -> dict[str, object]:
    bundle_dir = Path(directory).expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    computed_bundle_id = bundle_id_for_manifest(manifest)
    if manifest.get("bundle_id") != computed_bundle_id:
        raise ValueError("bundle_manifest.json no contiene una identidad de contenido válida")
    if expected_bundle_id is not None and computed_bundle_id != expected_bundle_id:
        raise ValueError(
            f"Bundle inesperado: esperado={expected_bundle_id}, obtenido={computed_bundle_id}"
        )
    for name, expected_sha256 in bundle_file_specs(manifest):
        path = bundle_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Falta el artefacto declarado {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 inválido para {name}: esperado={expected_sha256}, obtenido={actual_sha256}"
            )
    return manifest


def atomic_gzip(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with source.open("rb") as raw, os.fdopen(fd, "wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9) as compressed:
                while block := raw.read(1024 * 1024):
                    compressed.write(block)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "source_sha256": sha256_file(source),
        "source_bytes": source.stat().st_size,
        "archive_sha256": sha256_file(destination),
        "archive_bytes": destination.stat().st_size,
    }


def build_core_archive(destination: Path) -> dict[str, object]:
    selected = [
        ROOT / "pyproject.toml",
        ROOT / "config" / "taxonomia_v2.json",
        ROOT / "config" / "colab_l4.json",
        ROOT / "config" / "prompt_operacional_ollama_v3_2.md",
        ROOT / "requirements" / "colab-l4.txt",
        *(ROOT / "src" / "moderacion_peru").rglob("*.py"),
    ]
    selected = sorted(path for path in selected if path.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in selected:
                info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(
                    info,
                    core_file_bytes(path),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "name": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "files": [path.relative_to(ROOT).as_posix() for path in selected],
    }


def bundle_rebuild_reasons(destination: str | Path) -> tuple[str, ...]:
    """Explica si el bundle dejó de representar el código o las entradas locales."""

    target = Path(destination).expanduser().resolve()
    config_path = ROOT / "config" / "colab_l4.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    manifest_path = target / config["manifest"]
    if not manifest_path.is_file():
        return ("falta bundle_manifest.json",)
    try:
        manifest = verify_bundle_directory(target)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return (f"bundle inválido: {exc}",)

    reasons: list[str] = []
    with tempfile.TemporaryDirectory(prefix="moderacion_peru_core_check_") as temporary:
        current_core = build_core_archive(Path(temporary) / config["core_archive"])
    if current_core["sha256"] != manifest.get("core", {}).get("sha256"):
        reasons.append("cambió el código o la configuración incluida en project_core.zip")

    declared_inputs = manifest.get("inputs", {})
    for key, specification in config["inputs"].items():
        source = ROOT / specification["source"]
        if not source.is_file():
            reasons.append(f"falta la entrada requerida {key}: {source}")
            continue
        declared = declared_inputs.get(key)
        if not isinstance(declared, dict):
            reasons.append(f"el manifiesto no declara la entrada {key}")
            continue
        if declared.get("source_sha256") != sha256_file(source):
            reasons.append(f"cambió la entrada local {key}")
        if declared.get("archive") != specification["archive"]:
            reasons.append(f"cambió el nombre de archivo configurado para {key}")
    unexpected = sorted(set(declared_inputs) - set(config["inputs"]))
    if unexpected:
        reasons.append(f"el manifiesto conserva entradas no configuradas: {unexpected}")
    return tuple(reasons)


def ensure_prepared_bundle(destination: str | Path) -> dict[str, object]:
    """Reconstruye el bundle solo cuando su código, entradas o artefactos cambiaron."""

    target = Path(destination).expanduser().resolve()
    reasons = bundle_rebuild_reasons(target)
    if reasons:
        result = prepare(target)
        return {**result, "status": "rebuilt", "rebuild_reasons": list(reasons)}
    manifest = verify_bundle_directory(target)
    return {
        "destination": str(target),
        "manifest": str(target / "bundle_manifest.json"),
        **manifest,
        "status": "current",
        "rebuild_reasons": [],
    }


def prepare(destination: str | Path, *, progress_callback=None) -> dict[str, object]:
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if progress_callback is not None:
        progress_callback({"status": "started", "total": 9, "stage": "prepare"})
    with (ROOT / "config" / "colab_l4.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    core = build_core_archive(target / config["core_archive"])
    if progress_callback is not None:
        progress_callback({"status": "progress", "advance": 1, "stage": "core"})
    inputs: dict[str, dict[str, object]] = {}
    for key, specification in config["inputs"].items():
        source = ROOT / specification["source"]
        if not source.is_file():
            raise FileNotFoundError(f"Falta la entrada requerida {key}: {source}")
        archive = target / specification["archive"]
        entry = atomic_gzip(source, archive)
        inputs[key] = {
            "source": specification["source"],
            "archive": specification["archive"],
            "required_by": specification["required_by"],
            **entry,
        }
        if progress_callback is not None:
            progress_callback({"status": "progress", "advance": 1, "stage": f"compress:{key}"})
    manifest = {
        "schema_version": config["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_contract": config["taxonomy_contract"],
        "taxonomy_version": config["taxonomy_version"],
        "core": core,
        "inputs": inputs,
        "excluded_from_drive": config["excluded_from_drive"],
    }
    manifest["bundle_id"] = bundle_id_for_manifest(manifest)
    manifest_path = target / config["manifest"]
    fd, temporary_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    if progress_callback is not None:
        progress_callback({"status": "progress", "advance": 1, "stage": "manifest"})
    return {"destination": str(target), "manifest": str(manifest_path), **manifest}


def publish_drive_release(
    destination: str | Path,
    drive_root: str | Path,
    *,
    progress_callback=None,
) -> dict[str, object]:
    """Prepara y copia atómicamente una versión inmutable al Drive montado en la PC."""
    result = prepare(destination, progress_callback=progress_callback)
    source = Path(result["destination"])
    manifest = verify_bundle_directory(source)
    bundle_id = str(manifest["bundle_id"])
    resolved_drive_root = Path(drive_root).expanduser().resolve()
    if resolved_drive_root.name != "ModeracionPeru_Colab":
        raise ValueError(
            "drive_root debe terminar exactamente en ModeracionPeru_Colab; "
            f"se recibió {resolved_drive_root}"
        )
    releases_root = resolved_drive_root / "bundle_releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    release_dir = releases_root / bundle_id
    specs = bundle_file_specs(manifest)
    if release_dir.exists():
        verify_bundle_directory(release_dir, expected_bundle_id=bundle_id)
        if progress_callback is not None:
            progress_callback(
                {"status": "progress", "advance": len(specs) + 1, "stage": "release:already_present"}
            )
        status = "already_present"
    else:
        staging = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=releases_root))
        try:
            for name, _ in specs:
                shutil.copyfile(source / name, staging / name)
                if progress_callback is not None:
                    progress_callback({"status": "progress", "advance": 1, "stage": f"copy:{name}"})
            shutil.copyfile(source / "bundle_manifest.json", staging / "bundle_manifest.json")
            if progress_callback is not None:
                progress_callback({"status": "progress", "advance": 1, "stage": "copy:manifest"})
            verify_bundle_directory(staging, expected_bundle_id=bundle_id)
            os.replace(staging, release_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        status = "published"
    latest_pointer = releases_root / "latest.json"
    pointer = latest_pointer_for_release(release_dir, manifest)
    fd, temporary_pointer = tempfile.mkstemp(prefix=".latest.", dir=releases_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(pointer, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_pointer, latest_pointer)
    finally:
        if os.path.exists(temporary_pointer):
            os.unlink(temporary_pointer)
    if progress_callback is not None:
        progress_callback({"status": "progress", "advance": 1, "stage": "publish:latest_pointer"})
    return {
        **result,
        "status": status,
        "bundle_id": bundle_id,
        "drive_root": str(resolved_drive_root),
        "release_dir": str(release_dir),
        "latest_pointer": str(latest_pointer),
        "next_step": "Espere a que Google Drive complete la sincronización antes de abrir Colab.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        default=str(ROOT / "resultados" / "colab_bundle"),
        help="Staging local del bundle reproducible",
    )
    parser.add_argument(
        "--drive-root",
        help="Opcional: publica la versión en <DriveRoot>/bundle_releases/<bundle_id>",
    )
    args = parser.parse_args()
    if args.drive_root:
        result = publish_drive_release(args.destination, args.drive_root)
    else:
        result = prepare(args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
