from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl_atomic


DEFAULT_PRECEDENCE = {
    "human_modified": 50,
    "human": 45,
    "human_accepted": 40,
    "llm_remote_review": 30,
    "llm_remote": 20,
    "ollama_local": 10,
    "migration": 5,
}


def consolidate_annotations(
    sources: Iterable[str | Path],
    destination: str | Path,
    *,
    precedence: dict[str, int] | None = None,
) -> dict[str, int]:
    priorities = precedence or DEFAULT_PRECEDENCE
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        for row in read_jsonl(source):
            candidates[row["chunk_id"]].append(row)

    selected = []
    conflicts = 0
    for chunk_id, rows in candidates.items():
        rows.sort(
            key=lambda row: (
                priorities.get(str(row.get("label_source", "")), 0),
                str(row.get("created_at", "")),
            ),
            reverse=True,
        )
        winner = dict(rows[0])
        top_priority = priorities.get(str(winner.get("label_source", "")), 0)
        tied = [row for row in rows if priorities.get(str(row.get("label_source", "")), 0) == top_priority]
        decisions = {tuple(row.get("coarse_labels", [])) for row in tied}
        if len(decisions) > 1:
            conflicts += 1
            winner["coarse_labels"] = []
            winner["needs_review"] = True
            winner["training_eligible"] = False
            winner["decision_status"] = "needs_review"
            winner["consolidation_warning"] = "conflicting_top_priority_decisions"
        winner["consolidated_sources"] = [row.get("label_source") for row in rows]
        selected.append(winner)
    selected.sort(key=lambda row: row["chunk_id"])
    write_jsonl_atomic(destination, selected)
    return {"chunks": len(selected), "conflicts": conflicts}

