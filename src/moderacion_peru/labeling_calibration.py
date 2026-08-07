from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from typing import Any, Iterable


SAFE_LABEL = "SEGURO"


def select_calibration_panel(
    records: Iterable[dict[str, Any]],
    *,
    panel_size: int = 400,
    seed: int = 42,
    max_per_video: int = 1,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """Selecciona un panel reproducible, distribuido por canal y video."""

    if panel_size < 1 or max_per_video < 1:
        raise ValueError("panel_size y max_per_video deben ser positivos")
    by_channel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    if progress_callback is not None:
        progress_callback({"status": "started", "phase": "panel_scan", "total": None})
    for record in records:
        by_channel[str(record.get("channel_title") or "SIN_CANAL")].append(record)
        scanned += 1
        if progress_callback is not None and scanned % 1000 == 0:
            progress_callback(
                {"status": "progress", "phase": "panel_scan", "advance": 1000, "scanned": scanned}
            )
    rng = random.Random(seed)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for channel, rows in sorted(by_channel.items()):
        rng.shuffle(rows)
        queues[channel] = deque(rows)
    selected: list[dict[str, Any]] = []
    video_counts: Counter[str] = Counter()
    channels = list(queues)
    while channels and len(selected) < panel_size:
        next_channels = []
        for channel in channels:
            queue = queues[channel]
            chosen = None
            while queue:
                candidate = queue.popleft()
                video_id = str(candidate.get("video_id") or candidate["chunk_id"])
                if video_counts[video_id] < max_per_video:
                    chosen = candidate
                    video_counts[video_id] += 1
                    break
            if chosen is not None:
                selected.append(chosen)
            if queue:
                next_channels.append(channel)
            if len(selected) >= panel_size:
                break
        channels = next_channels
    if len(selected) < panel_size:
        raise ValueError(
            f"Solo se pudieron seleccionar {len(selected)} registros con max_per_video={max_per_video}"
        )
    if progress_callback is not None:
        remainder = scanned % 1000
        if remainder:
            progress_callback(
                {"status": "progress", "phase": "panel_scan", "advance": remainder, "scanned": scanned}
            )
        progress_callback(
            {"status": "finished", "phase": "panel_scan", "advance": 0, "selected": len(selected)}
        )
    return selected


def _is_damage(row: dict[str, Any]) -> bool:
    labels = set(row.get("coarse_labels") or row.get("labels") or [])
    return bool(labels - {SAFE_LABEL})


def _wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, (centre - spread) / denominator)


def calibrate_primary_against_reviewer(
    primary_rows: Iterable[dict[str, Any]],
    reviewer_rows: Iterable[dict[str, Any]],
    *,
    thresholds: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    minimum_exact_lower: float = 0.90,
    minimum_binary_lower: float = 0.95,
    minimum_auto_count: int = 30,
    bootstrap_replicates: int = 500,
    bootstrap_seed: int = 20260807,
) -> dict[str, Any]:
    """Calibra el score declarado; el revisor es comparador, no verdad humana."""

    primary = {str(row["chunk_id"]): row for row in primary_rows}
    reviewer = {str(row["chunk_id"]): row for row in reviewer_rows}
    paired_ids = sorted(set(primary) & set(reviewer))
    if not paired_ids:
        raise ValueError("No hay chunk_id pareados para calibrar")
    pairs = [(primary[cid], reviewer[cid]) for cid in paired_ids]
    comparisons = []
    for threshold in thresholds:
        accepted = [
            (left, right)
            for left, right in pairs
            if not left.get("needs_review")
            and not _is_damage(left)
            and float(left.get("score_confianza", 0.0)) >= threshold
        ]
        exact = sum(
            set(left.get("coarse_labels", [])) == set(right.get("coarse_labels", []))
            for left, right in accepted
        )
        binary = sum(_is_damage(left) == _is_damage(right) for left, right in accepted)
        n = len(accepted)
        comparisons.append(
            {
                "threshold": threshold,
                "auto_accepted": n,
                "coverage": n / len(pairs),
                "exact_agreement": exact / n if n else None,
                "exact_lower_one_sided_95": _wilson_lower(exact, n),
                "binary_agreement": binary / n if n else None,
                "binary_lower_one_sided_95": _wilson_lower(binary, n),
            }
        )
    eligible = [
        row
        for row in comparisons
        if row["auto_accepted"] >= minimum_auto_count
        and row["exact_lower_one_sided_95"] >= minimum_exact_lower
        and row["binary_lower_one_sided_95"] >= minimum_binary_lower
    ]
    selected = min(eligible, key=lambda row: row["threshold"]) if eligible else comparisons[-1]
    status = "calibrated" if eligible else "inconclusive_conservative_threshold"

    confidence_errors = []
    for left, right in pairs:
        exact = float(
            set(left.get("coarse_labels", [])) == set(right.get("coarse_labels", []))
        )
        confidence_errors.append(abs(float(left.get("score_confianza", 0.0)) - exact))

    accepted_selected = [
        (left, right)
        for left, right in pairs
        if not left.get("needs_review")
        and not _is_damage(left)
        and float(left.get("score_confianza", 0.0)) >= selected["threshold"]
    ]
    by_video: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for left, right in accepted_selected:
        by_video[str(left.get("video_id") or left["chunk_id"])].append((left, right))
    rng = random.Random(bootstrap_seed)
    exact_bootstrap: list[float] = []
    binary_bootstrap: list[float] = []
    videos = list(by_video)
    if videos and bootstrap_replicates:
        for _ in range(bootstrap_replicates):
            sample = []
            for _ in videos:
                sample.extend(by_video[rng.choice(videos)])
            exact_bootstrap.append(
                sum(
                    set(left.get("coarse_labels", [])) == set(right.get("coarse_labels", []))
                    for left, right in sample
                )
                / len(sample)
            )
            binary_bootstrap.append(
                sum(_is_damage(left) == _is_damage(right) for left, right in sample) / len(sample)
            )

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]

    return {
        "schema_version": "1.0.0",
        "reference_kind": "stronger_llm_not_human_ground_truth",
        "paired_chunks": len(pairs),
        "threshold_status": status,
        "selected_threshold": selected["threshold"],
        "selection_criteria": {
            "minimum_exact_lower": minimum_exact_lower,
            "minimum_binary_lower": minimum_binary_lower,
            "minimum_auto_count": minimum_auto_count,
        },
        "comparisons": comparisons,
        "mean_absolute_calibration_error_exact": sum(confidence_errors) / len(confidence_errors),
        "selected_threshold_cluster_bootstrap_95": {
            "replicates": bootstrap_replicates,
            "exact_low": percentile(exact_bootstrap, 0.025),
            "exact_high": percentile(exact_bootstrap, 0.975),
            "binary_low": percentile(binary_bootstrap, 0.025),
            "binary_high": percentile(binary_bootstrap, 0.975),
        },
    }


def build_directed_review_queue(
    chunks: Iterable[dict[str, Any]],
    primary_rows: Iterable[dict[str, Any]],
    *,
    confidence_threshold: float,
    safe_control_rate: float = 0.10,
    seed: int = 42,
    progress_callback=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enruta daño, dudas, baja confianza y un control seguro reproducible."""

    if not 0 <= safe_control_rate <= 1:
        raise ValueError("safe_control_rate debe estar entre 0 y 1")
    annotations = {str(row["chunk_id"]): row for row in primary_rows}
    chunk_rows = list(chunks)
    if progress_callback is not None:
        progress_callback(
            {"status": "started", "phase": "review_queue", "total": len(chunk_rows), "advance": 0}
        )
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        by_video[str(row.get("video_id") or "")].append(row)
    enriched: dict[str, dict[str, Any]] = {}
    for rows in by_video.values():
        rows.sort(key=lambda row: (float(row.get("start_seconds") or 0), str(row["chunk_id"])))
        for index, row in enumerate(rows):
            item = dict(row)
            if index:
                item["contexto_anterior"] = rows[index - 1]["text"]
            if index + 1 < len(rows):
                item["contexto_posterior"] = rows[index + 1]["text"]
            enriched[str(row["chunk_id"])] = item

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, chunk in enumerate(chunk_rows, start=1):
        chunk_id = str(chunk["chunk_id"])
        annotation = annotations.get(chunk_id)
        if annotation is None:
            reason = "missing_primary"
        elif _is_damage(annotation):
            reason = "damage"
        elif annotation.get("needs_review"):
            reason = "needs_review"
        elif float(annotation.get("score_confianza", 0.0)) < confidence_threshold:
            reason = "low_confidence"
        elif rng.random() < safe_control_rate:
            reason = "safe_control"
        else:
            reason = ""
        if reason:
            item = enriched[chunk_id]
            item["routing_reason"] = reason
            selected.append(item)
            reasons[reason] += 1
        if progress_callback is not None and (index % 1000 == 0 or index == len(chunk_rows)):
            progress_callback(
                {
                    "status": "progress",
                    "phase": "review_queue",
                    "advance": 1000 if index % 1000 == 0 else index % 1000,
                    "scanned": index,
                    "selected": len(selected),
                }
            )
    if progress_callback is not None:
        progress_callback(
            {"status": "finished", "phase": "review_queue", "advance": 0, "selected": len(selected)}
        )
    return selected, {
        "source_chunks": len(chunk_rows),
        "primary_annotations": len(annotations),
        "selected": len(selected),
        "confidence_threshold": confidence_threshold,
        "safe_control_rate": safe_control_rate,
        "seed": seed,
        "routing_reasons": dict(reasons),
    }
