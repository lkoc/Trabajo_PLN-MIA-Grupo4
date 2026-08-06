from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Huella estable de un objeto JSON, independiente de espacios y orden de claves."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256_text(encoded)


def input_signature(paths: Iterable[str | Path], configuration: Any | None = None) -> str:
    """Firma archivos existentes y configuración para detectar una ejecución no-op."""

    records = []
    for raw in sorted((Path(path).resolve() for path in paths), key=str):
        records.append(
            {
                "path": str(raw),
                "exists": raw.is_file(),
                "sha256": sha256_file(raw) if raw.is_file() else None,
                "bytes": raw.stat().st_size if raw.is_file() else None,
            }
        )
    return canonical_json_sha256({"inputs": records, "configuration": configuration or {}})


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {source}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Se esperaba un objeto en {source}:{line_number}")
            yield payload


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def append_jsonl_once(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    id_field: str,
) -> tuple[int, int]:
    """Añade solo IDs nuevos y devuelve (añadidos, omitidos)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = {row[id_field] for row in read_jsonl(target)} if target.exists() else set()
    added = skipped = 0
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            identifier = row.get(id_field)
            if not identifier:
                raise ValueError(f"Falta {id_field} en una fila")
            if identifier in existing:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            existing.add(identifier)
            added += 1
        handle.flush()
        os.fsync(handle.fileno())
    return added, skipped
