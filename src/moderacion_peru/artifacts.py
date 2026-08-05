from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import sha256_file
from .paths import find_project_root


def artifact_status(root: str | Path | None = None) -> dict[str, Any]:
    project = find_project_root(root)
    config = json.loads((project / "config" / "artifacts.json").read_text(encoding="utf-8"))
    records = []
    for specification in config["artifacts"]:
        path = project / specification["path"]
        record = dict(specification)
        record["available"] = path.is_file()
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256_file(path)
        records.append(record)
    return {
        "schema_version": config["schema_version"],
        "project_root": str(project),
        "available": sum(item["available"] for item in records),
        "missing": sum(not item["available"] for item in records),
        "artifacts": records,
    }

