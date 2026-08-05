"""Construye el bundle mínimo que se sube a Google Drive para Colab."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_gzip(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with source.open("rb") as raw, os.fdopen(fd, "wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=6) as compressed:
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
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
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


def prepare(destination: str | Path) -> dict[str, object]:
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    with (ROOT / "config" / "colab_l4.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    core = build_core_archive(target / config["core_archive"])
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
    manifest = {
        "schema_version": config["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy_contract": config["taxonomy_contract"],
        "taxonomy_version": config["taxonomy_version"],
        "core": core,
        "inputs": inputs,
        "excluded_from_drive": config["excluded_from_drive"],
    }
    manifest_path = target / config["manifest"]
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return {"destination": str(target), "manifest": str(manifest_path), **manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        default=str(ROOT / "resultados" / "colab_bundle"),
        help="Carpeta bundle en Google Drive Desktop o staging local",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
