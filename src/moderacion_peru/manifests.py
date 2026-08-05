from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from .io import sha256_file, write_json_atomic
from .paths import find_project_root, relative_to_root
from .schemas import ArtifactReference, RunManifest
from .taxonomy import load_taxonomy


def artifact_reference(path: str | Path, role: str, *, required: bool = True) -> ArtifactReference:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    return ArtifactReference(
        path=relative_to_root(target),
        sha256=sha256_file(target),
        bytes=target.stat().st_size,
        role=role,
        required=required,
    )


def git_revision(root: str | Path | None = None) -> str | None:
    project_root = find_project_root(root)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    *,
    run_id: str,
    stage: str,
    inputs: Iterable[ArtifactReference] = (),
    outputs: Iterable[ArtifactReference] = (),
    **kwargs,
) -> RunManifest:
    taxonomy = load_taxonomy()
    return RunManifest(
        run_id=run_id,
        stage=stage,
        taxonomy_contract=taxonomy.contract_id,
        taxonomy_version=taxonomy.version,
        code_revision=git_revision(),
        inputs=list(inputs),
        outputs=list(outputs),
        **kwargs,
    )


def save_manifest(path: str | Path, manifest: RunManifest) -> None:
    write_json_atomic(path, manifest.model_dump(mode="json"))

