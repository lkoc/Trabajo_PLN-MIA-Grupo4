from __future__ import annotations

import os
from pathlib import Path


ROOT_MARKER = "pyproject.toml"


def find_project_root(start: str | Path | None = None) -> Path:
    override = os.getenv("MODPERU_ROOT", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / ROOT_MARKER).is_file():
            raise FileNotFoundError(f"MODPERU_ROOT no contiene {ROOT_MARKER}: {root}")
        return root

    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ROOT_MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"No se encontró {ROOT_MARKER} desde {current}")


def artifact_root(project_root: str | Path | None = None) -> Path:
    override = os.getenv("MODPERU_ARTIFACT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return find_project_root(project_root) / "artefactos"


def relative_to_root(path: str | Path, root: str | Path | None = None) -> str:
    base = find_project_root(root)
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return str(resolved)

