"""Restaura rutas de trabajo desde los checkpoints pequeños sincronizados por Git."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from moderacion_peru.acquisition import restore_canonical_from_channel_transcripts
from moderacion_peru.io import sha256_file


def restore_gzip(
    archive: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
    expected_archive_sha256: str,
    force: bool = False,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(f"Falta el checkpoint comprimido: {archive}")
    if sha256_file(archive) != expected_archive_sha256:
        raise ValueError(f"SHA-256 inválido para {archive}")
    if destination.is_file():
        current_sha256 = sha256_file(destination)
        if current_sha256 == expected_source_sha256:
            return {
                "status": "already_restored",
                "path": destination.as_posix(),
                "sha256": current_sha256,
            }
        if not force:
            raise FileExistsError(
                f"{destination} ya existe con otro contenido; use --force solo si desea reemplazarlo"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with gzip.open(archive, "rb") as source, os.fdopen(descriptor, "wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        restored = Path(temporary_name)
        restored_sha256 = sha256_file(restored)
        if restored_sha256 != expected_source_sha256:
            raise ValueError(f"El contenido restaurado desde {archive} no coincide con el manifiesto")
        os.replace(restored, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "status": "restored",
        "path": destination.as_posix(),
        "sha256": expected_source_sha256,
        "bytes": destination.stat().st_size,
    }


def restore(project_root: str | Path, *, force: bool = False) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    transcript_result = restore_canonical_from_channel_transcripts(
        root / "datos" / "raw" / "transcripts_by_channel",
        root / "datos" / "raw" / "transcripts_raw.jsonl",
    )

    bundle = root / "resultados" / "colab_bundle"
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto del bundle: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    restored_inputs: dict[str, dict[str, object]] = {}
    for key, entry in manifest.get("inputs", {}).items():
        restored_inputs[key] = restore_gzip(
            bundle / str(entry["archive"]),
            root / str(entry["source"]),
            expected_source_sha256=str(entry["source_sha256"]),
            expected_archive_sha256=str(entry["archive_sha256"]),
            force=force,
        )
    return {
        "project_root": root.as_posix(),
        "transcripts": transcript_result,
        "bundle_inputs": restored_inputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza solo salidas derivadas cuyo hash no coincida; el canónico se completa sin borrarse.",
    )
    args = parser.parse_args()
    print(json.dumps(restore(args.project_root, force=args.force), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
