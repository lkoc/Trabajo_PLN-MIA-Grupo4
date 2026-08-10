from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import mimetypes
import os
import threading
import urllib.parse
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import ValidationError

from .incremental import TranscriptSegment, chunk_transcript
from .io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .paths import find_project_root
from .registry import ProductionPredictor
from .schemas import ModelRegistryEntry, ReviewEvent
from .taxonomy import load_taxonomy
from .training import resolve_prediction

PRODUCTION_SLOTS = ("classical", "transformer", "qwen")
PRODUCTION_MODES = (*PRODUCTION_SLOTS, "compare", "consensus")
RETRAIN_MINIMUM_TOTAL = 500
RETRAIN_MINIMUM_SAFE = 200
RETRAIN_MINIMUM_PER_DAMAGE = 100
MAX_REQUEST_BYTES = 2 * 1024 * 1024
LABELING_BULK_SCOPES = {"video", "channel"}
LABELING_BULK_ACTIONS = {"accept", "modify", "reject"}
LABELING_URGENT_WARNING = "conflicting_top_priority_decisions"


def _is_labeling_priority(
    row: dict[str, Any], latest_review: dict[str, Any] | None = None
) -> bool:
    """Prioriza salidas Pro efectivamente pendientes o con daño vigente."""

    model = str(row.get("annotator_model") or "").casefold()
    if "pro" not in model:
        return False

    action = str((latest_review or {}).get("action") or "")
    if action == "reject" or (
        latest_review is None and str(row.get("decision_status") or "") == "excluded"
    ):
        return False
    if action in {"accept", "modify"}:
        unresolved = False
        labels = (latest_review or {}).get("final_labels", [])
    else:
        unresolved = (
            action == "defer"
            or bool(row.get("needs_review"))
            or str(row.get("decision_status") or "") == "needs_review"
        )
        labels = row.get("coarse_labels") or []
    has_damage = any(str(label).upper() != "SEGURO" for label in labels)
    return unresolved or has_damage


def _is_labeling_urgent(row: dict[str, Any]) -> bool:
    """Cola corta: la consolidación mantuvo propuestas máximas incompatibles."""

    return str(row.get("consolidation_warning") or "") == LABELING_URGENT_WARNING


def _is_labeling_excluded(
    row: dict[str, Any], latest_reviews: dict[str, dict[str, Any]]
) -> bool:
    """Indica si la decisión vigente mantiene el chunk fuera del dataset."""

    review = latest_reviews.get(str(row.get("chunk_id")))
    if review is not None:
        return str(review.get("action") or "") == "reject"
    return str(row.get("decision_status") or "") == "excluded"


def _requires_labeling_action(
    row: dict[str, Any], latest_reviews: dict[str, dict[str, Any]]
) -> bool:
    """Indica si falta una decisión final o el revisor decidió diferirla."""

    return _labeling_filter_values(row, latest_reviews)[2] in {"pending", "deferred"}


def _labeling_filter_values(
    row: dict[str, Any], latest_reviews: dict[str, dict[str, Any]]
) -> tuple[set[str], set[str], str]:
    """Devuelve categorías, flags y estado efectivos para filtros combinables."""

    review = latest_reviews.get(str(row.get("chunk_id")))
    action = str((review or {}).get("action") or "")
    if action == "reject" or (
        review is None and row.get("decision_status") == "excluded"
    ):
        status = "excluded"
    elif action == "defer":
        status = "deferred"
    elif action in {"accept", "modify"}:
        status = "resolved"
    elif bool(row.get("needs_review")) or row.get("decision_status") == "needs_review":
        status = "pending"
    else:
        status = "resolved"

    if action in {"accept", "modify"}:
        labels = {str(value) for value in (review or {}).get("final_labels", [])}
        flags = {str(value) for value in (review or {}).get("flags", [])}
    else:
        # Un diferimiento o una exclusión no borra la propuesta útil para auditar.
        labels = {str(value) for value in (row.get("coarse_labels") or [])}
        flags = {str(value) for value in (row.get("flags") or [])}
    return labels, flags, status


def _matches_labeling_filters(
    row: dict[str, Any],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    filter_labels: set[str] | None = None,
    filter_flags: set[str] | None = None,
    filter_statuses: set[str] | None = None,
    filter_labeling: set[str] | None = None,
    match_all: bool = False,
) -> bool:
    labels, flags, status = _labeling_filter_values(row, latest_reviews)
    if filter_labels:
        if match_all and not filter_labels.issubset(labels):
            return False
        if not match_all and labels.isdisjoint(filter_labels):
            return False
    if filter_flags:
        if match_all and not filter_flags.issubset(flags):
            return False
        if not match_all and flags.isdisjoint(filter_flags):
            return False
    labeling_state = "labeled" if labels else "unlabeled"
    return (not filter_statuses or status in filter_statuses) and (
        not filter_labeling or labeling_state in filter_labeling
    )


def _labeling_campaign_page(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    offset: int,
    limit: int,
    cohort: str = "",
    only_pending: bool = False,
    priority_only: bool = False,
    urgent_only: bool = False,
    excluded_only: bool = False,
    filter_labels: set[str] | None = None,
    filter_flags: set[str] | None = None,
    filter_statuses: set[str] | None = None,
    filter_labeling: set[str] | None = None,
    match_all: bool = False,
) -> dict[str, Any]:
    """Pagina la campaña sin copiar el corpus completo para cada solicitud."""

    page_indices: list[int] = []
    view_positions: list[int] = []
    page_rows: list[dict[str, Any]] = []
    matching = 0
    view_position = 0
    for index, row in enumerate(campaign_rows):
        row_cohort = str(row.get("cohort") or row.get("label_source") or "sin_cohorte")
        if cohort and cohort != "all" and row_cohort != cohort:
            continue
        review = latest_reviews.get(str(row.get("chunk_id")))
        if excluded_only and not _is_labeling_excluded(row, latest_reviews):
            continue
        if not _matches_labeling_filters(
            row,
            latest_reviews,
            filter_labels=filter_labels,
            filter_flags=filter_flags,
            filter_statuses=filter_statuses,
            filter_labeling=filter_labeling,
            match_all=match_all,
        ):
            continue
        if urgent_only and not _is_labeling_urgent(row):
            continue
        if priority_only and not urgent_only and not _is_labeling_priority(row, review):
            continue
        current_view_position = view_position
        view_position += 1
        effective_status = _labeling_filter_values(row, latest_reviews)[2]
        if (
            only_pending
            and not excluded_only
            and effective_status
            not in {
                "pending",
                "deferred",
            }
        ):
            continue
        if offset <= matching < offset + limit:
            page_indices.append(index)
            view_positions.append(current_view_position)
            page_rows.append(row)
        matching += 1
    return {
        "total": matching,
        "offset": offset,
        "indices": page_indices,
        "view_positions": view_positions,
        "rows": page_rows,
        "reviews": {
            str(row["chunk_id"]): latest_reviews[str(row["chunk_id"])]
            for row in page_rows
            if str(row.get("chunk_id")) in latest_reviews
        },
    }


def _labeling_progress(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    priority_only: bool = False,
    urgent_only: bool = False,
    action_only: bool = False,
    excluded_only: bool = False,
    cohort: str = "",
    filter_labels: set[str] | None = None,
    filter_flags: set[str] | None = None,
    filter_statuses: set[str] | None = None,
    filter_labeling: set[str] | None = None,
    match_all: bool = False,
) -> dict[str, Any]:
    cohort_rows = [
        row
        for row in campaign_rows
        if not cohort
        or cohort == "all"
        or str(row.get("cohort") or row.get("label_source") or "sin_cohorte") == cohort
        if _matches_labeling_filters(
            row,
            latest_reviews,
            filter_labels=filter_labels,
            filter_flags=filter_flags,
            filter_statuses=filter_statuses,
            filter_labeling=filter_labeling,
            match_all=match_all,
        )
    ]
    if excluded_only:
        selected_rows = [
            row for row in cohort_rows if _is_labeling_excluded(row, latest_reviews)
        ]
    elif urgent_only:
        selected_rows = [row for row in cohort_rows if _is_labeling_urgent(row)]
    elif action_only:
        selected_rows = [
            row for row in cohort_rows if _requires_labeling_action(row, latest_reviews)
        ]
    elif priority_only:
        selected_rows = [
            row
            for row in cohort_rows
            if _is_labeling_priority(row, latest_reviews.get(str(row.get("chunk_id"))))
        ]
    else:
        selected_rows = cohort_rows
    selected_ids = {str(row.get("chunk_id")) for row in selected_rows}
    total = len(selected_rows)
    events = [
        event
        for chunk_id, event in latest_reviews.items()
        if str(chunk_id) in selected_ids
    ]
    excluded_total = sum(
        _is_labeling_excluded(row, latest_reviews) for row in campaign_rows
    )
    if excluded_only:
        return {
            "total": total,
            "reviewed": len(events),
            "resolved": total,
            "deferred": 0,
            "pending": 0,
            "excluded_total": excluded_total,
            "progress_pct": 100.0 if total else 0.0,
        }
    effective_statuses = [
        _labeling_filter_values(row, latest_reviews)[2] for row in selected_rows
    ]
    resolved = sum(status in {"resolved", "excluded"} for status in effective_statuses)
    deferred = effective_statuses.count("deferred")
    pending = effective_statuses.count("pending")
    return {
        "total": total,
        "reviewed": len(events),
        "resolved": resolved,
        "deferred": deferred,
        "pending": pending,
        "excluded_total": excluded_total,
        "progress_pct": 100 * resolved / max(1, total),
    }


def _labeling_dashboard(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    taxonomy: Any,
    *,
    audit_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume el estado efectivo del etiquetado para el dashboard operativo."""

    total = len(campaign_rows)
    safe_label = str(taxonomy.safe_label)
    damage_labels = tuple(str(label) for label in taxonomy.damage_labels)
    damage_set = set(damage_labels)
    label_counts: Counter[str] = Counter({label: 0 for label in damage_labels})
    flag_counts: Counter[str] = Counter({str(flag): 0 for flag in taxonomy.flags})
    status_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    channel_harm: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reviewer_counts: Counter[str] = Counter()
    activity: dict[str, Counter[str]] = defaultdict(Counter)
    queue_counts: dict[str, Counter[str]] = {
        queue: Counter() for queue in ("urgent", "priority", "all", "excluded")
    }
    valid_flags = {str(flag) for flag in taxonomy.flags}
    excluded = safe = harm = unlabeled = invalid_safe_harm = assignments = 0
    intermediate_pro_needs_review = intermediate_overridden = 0

    for row in campaign_rows:
        chunk_id = str(row.get("chunk_id"))
        review = latest_reviews.get(chunk_id)
        model = str(row.get("annotator_model") or "Modelo no indicado").strip()
        is_intermediate_pro_review = "pro" in model.casefold() and (
            bool(row.get("needs_review"))
            or str(row.get("decision_status") or "") == "needs_review"
        )
        if is_intermediate_pro_review:
            intermediate_pro_needs_review += 1
            intermediate_overridden += int(
                str((review or {}).get("action") or "")
                in {"accept", "modify", "reject"}
            )
        labels, flags, status = _labeling_filter_values(row, latest_reviews)
        status_counts[status] += 1
        selected_queues = ["all"]
        if _is_labeling_urgent(row):
            selected_queues.append("urgent")
        if _is_labeling_priority(row, review):
            selected_queues.append("priority")
        if status == "excluded":
            selected_queues.append("excluded")
        for queue in selected_queues:
            queue_counts[queue]["total"] += 1
            if review is not None:
                queue_counts[queue]["reviewed"] += 1
                if str(review.get("action") or "") == "defer":
                    queue_counts[queue]["deferred"] += 1
                else:
                    queue_counts[queue]["resolved"] += 1

        if review is not None:
            action = str(review.get("action") or "sin_accion")
            action_counts[action] += 1
            reviewer = str(review.get("reviewer") or "Sin iniciales").strip()
            reviewer_counts[reviewer or "Sin iniciales"] += 1
            created_at = str(review.get("created_at") or "")
            if len(created_at) >= 13:
                hour = created_at[:13] + ":00Z"
                activity[hour]["total"] += 1
                activity[hour][action] += 1

        if status == "excluded":
            excluded += 1
            continue

        known_damage = labels & damage_set
        if safe_label in labels and known_damage:
            invalid_safe_harm += 1
        if known_damage:
            harm += 1
            label_counts.update(known_damage)
            assignments += len(known_damage)
        elif safe_label in labels:
            safe += 1
        else:
            unlabeled += 1
        flag_counts.update(flags & valid_flags)

        model_counts[model or "Modelo no indicado"] += 1
        channel = str(
            row.get("channel_title") or row.get("channel_id") or "Canal no indicado"
        ).strip()
        channel_counts[channel] += 1
        channel_harm[channel] += bool(known_damage)
        video_id = str(row.get("video_id") or "").strip()
        video_title = str(row.get("video_title") or "Video no indicado").strip()
        video_key = video_id or f"{channel}\u241f{video_title}"
        video_counts[video_key] += 1

    eligible = max(0, total - excluded)
    labeled = safe + harm
    channels = len(channel_counts)
    videos = len(video_counts)
    category_values = [label_counts[label] for label in damage_labels]
    category_mean = sum(category_values) / max(1, len(category_values))
    category_cv = (
        math.sqrt(
            sum((value - category_mean) ** 2 for value in category_values)
            / max(1, len(category_values))
        )
        / category_mean
        if category_mean
        else 0.0
    )
    category_total = sum(category_values)
    normalized_entropy = (
        -sum(
            (value / category_total) * math.log(value / category_total)
            for value in category_values
            if value
        )
        / math.log(len(category_values))
        if category_total and len(category_values) > 1
        else 0.0
    )
    positive_values = [value for value in category_values if value]
    ratio = max(positive_values) / min(positive_values) if positive_values else 0.0
    display_names = {
        label: str(taxonomy.categories[label].display_name)
        for label in taxonomy.target_labels
    }

    def pct(value: float, denominator: float) -> float:
        return 100.0 * value / denominator if denominator else 0.0

    def queue_payload(queue: str) -> dict[str, Any]:
        counts = queue_counts[queue]
        queue_total = counts["total"]
        if queue == "excluded":
            return {
                "total": queue_total,
                "reviewed": counts["reviewed"],
                "resolved": queue_total,
                "deferred": 0,
                "pending": 0,
                "excluded_total": excluded,
                "progress_pct": 100.0 if queue_total else 0.0,
            }
        resolved = counts["resolved"]
        return {
            "total": queue_total,
            "reviewed": counts["reviewed"],
            "resolved": resolved,
            "deferred": counts["deferred"],
            "pending": max(0, queue_total - resolved),
            "excluded_total": excluded,
            "progress_pct": pct(resolved, queue_total),
        }

    status_names = {
        "resolved": "Resuelto",
        "pending": "Pendiente",
        "deferred": "Diferido",
        "excluded": "Excluido",
    }
    action_names = {
        "accept": "Sugerencia aceptada",
        "modify": "Decisi\u00f3n ajustada",
        "reject": "Excluido",
        "defer": "Diferido",
        "sin_accion": "Sin acci\u00f3n",
    }

    channel_rows = [
        {
            "name": name,
            "chunks": count,
            "harm": channel_harm[name],
            "harm_pct": pct(channel_harm[name], count),
        }
        for name, count in channel_counts.items()
    ]
    top_volume = sorted(
        channel_rows, key=lambda item: (-item["chunks"], item["name"].casefold())
    )[:10]
    top_harm = sorted(
        (item for item in channel_rows if item["chunks"] >= 20),
        key=lambda item: (-item["harm_pct"], -item["harm"], item["name"].casefold()),
    )[:10]

    audit_payload: dict[str, Any] = {"available": False}
    if audit_metrics:
        system_names = {
            "cascada_flash_pro_consolidada": "Cascada Flash/Pro",
            "deepseek_v4_flash": "DeepSeek V4 Flash",
            "deepseek_v4_pro": "DeepSeek V4 Pro",
        }
        audit_systems: list[dict[str, Any]] = []
        for system_id in (
            "cascada_flash_pro_consolidada",
            "deepseek_v4_flash",
            "deepseek_v4_pro",
        ):
            system = (audit_metrics.get("systems") or {}).get(system_id)
            if not isinstance(system, dict):
                continue
            point = system.get("point") or {}
            confidence = system.get("confidence") or {}
            audit_systems.append(
                {
                    "id": system_id,
                    "label": system_names[system_id],
                    "answered": system.get("answered", 0),
                    "coverage": system.get("coverage_over_sample", 0.0),
                    "abstention": system.get("abstention_over_sample", 0.0),
                    "exact_agreement": point.get("exact_agreement"),
                    "exact_ci95": system.get("exact_agreement_wilson_95", []),
                    "binary_f1": point.get("binary_f1"),
                    "binary_mcc": point.get("binary_mcc"),
                    "micro_f1": point.get("multilabel_micro_f1"),
                    "hamming_loss": point.get("hamming_loss"),
                    "mean_confidence": confidence.get("mean"),
                    "brier": confidence.get("brier_for_exact_correctness"),
                    "ece": confidence.get("ece_10_equal_width"),
                    "confidence_bands": confidence.get("bands", []),
                    "per_label": system.get("per_label", {}),
                }
            )
        audit_payload = {
            "available": True,
            "generated_by": audit_metrics.get("generated_by"),
            "sample": audit_metrics.get("sample", {}),
            "reference": audit_metrics.get("reference", {}),
            "systems": audit_systems,
            "paired": audit_metrics.get("paired_flash_vs_pro_on_common_answered", {}),
            "inference": audit_metrics.get("inference", {}),
        }

    insights = [
        {
            "tone": "success" if unlabeled == 0 else "warning",
            "title": "Cobertura efectiva",
            "body": (
                f"{pct(labeled, eligible):.2f}% de los chunks elegibles tiene una "
                f"decisi\u00f3n gruesa; quedan {unlabeled:,} sin etiqueta."
            ),
        },
        {
            "tone": "info",
            "title": "Prevalencia de da\u00f1o",
            "body": (
                f"{harm:,} chunks ({pct(harm, eligible):.2f}% de los elegibles) "
                "tienen al menos una categor\u00eda de da\u00f1o vigente."
            ),
        },
        {
            "tone": "success" if status_counts["pending"] == 0 else "warning",
            "title": "Jerarqu\u00eda de decisi\u00f3n",
            "body": (
                f"{intermediate_overridden:,} de {intermediate_pro_needs_review:,} "
                "estados intermedios needs_review de Pro tienen una decisi\u00f3n "
                f"superior materializada; quedan {status_counts['pending']:,} "
                "decisiones finales pendientes."
            ),
        },
        {
            "tone": "warning" if ratio >= 2 else "info",
            "title": "Balance entre da\u00f1os",
            "body": (
                f"La raz\u00f3n m\u00e1ximo/m\u00ednimo es {ratio:.2f}\u00d7 y el CV es "
                f"{category_cv:.3f}; conviene vigilar las clases menos representadas."
            ),
        },
        {
            "tone": (
                "success" if min(category_values, default=0) >= 2_000 else "warning"
            ),
            "title": "Meta de soporte total",
            "body": (
                "La condici\u00f3n vigente suma train, validation y test: "
                f"la clase menos representada tiene {min(category_values, default=0):,} "
                "chunks frente a la meta de 2,000."
            ),
        },
        {
            "tone": "info",
            "title": "Lectura cualitativa",
            "body": (
                "La auditor\u00eda mantiene vigilancia dirigida sobre citas o denuncias "
                "sin respaldo del narrador, usos sexuales peruanos (p. ej., cachar), "
                "humor amistoso y condescendencia que puede encubrir clasismo, racismo "
                "o sexismo."
            ),
        },
    ]
    paired = audit_payload.get("paired") or {}
    if paired:
        difference = float(paired.get("pro_minus_flash_exact_agreement") or 0.0)
        insights.append(
            {
                "tone": "success" if difference > 0 else "warning",
                "title": "Comparaci\u00f3n pareada Flash/Pro",
                "body": (
                    f"En {int(paired.get('n') or 0):,} respuestas comunes, Pro supera "
                    f"a Flash en {difference * 100:.2f} puntos porcentuales de acuerdo "
                    "exacto; el panel es dirigido y no representa una muestra aleatoria."
                ),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": {
            "corpus": {
                "total": total,
                "eligible": eligible,
                "excluded": excluded,
                "excluded_pct": pct(excluded, total),
                "safe": safe,
                "harm": harm,
                "unlabeled": unlabeled,
                "labeled": labeled,
                "coverage_pct": pct(labeled, eligible),
                "harm_prevalence_pct": pct(harm, eligible),
                "channels": channels,
                "videos": videos,
                "avg_chunks_per_channel": eligible / channels if channels else 0.0,
                "median_chunks_per_channel": (
                    float(median(channel_counts.values())) if channels else 0.0
                ),
                "avg_chunks_per_video": eligible / videos if videos else 0.0,
                "median_chunks_per_video": (
                    float(median(video_counts.values())) if videos else 0.0
                ),
                "multilabel_assignments": assignments,
                "invalid_safe_harm": invalid_safe_harm,
                "intermediate_pro_needs_review": intermediate_pro_needs_review,
                "intermediate_review_overridden": intermediate_overridden,
                "final_pending": status_counts["pending"],
            },
            "labels": [
                {
                    "id": label,
                    "label": display_names[label],
                    "count": label_counts[label],
                    "pct_eligible": pct(label_counts[label], eligible),
                    "pct_harm": pct(label_counts[label], harm),
                }
                for label in damage_labels
            ],
            "flags": [
                {
                    "id": flag,
                    "label": flag.replace("_", " ").capitalize(),
                    "count": flag_counts[flag],
                    "pct_eligible": pct(flag_counts[flag], eligible),
                }
                for flag in taxonomy.flags
            ],
            "status": [
                {
                    "id": status,
                    "label": status_names[status],
                    "count": status_counts[status],
                    "pct_total": pct(status_counts[status], total),
                }
                for status in ("resolved", "pending", "deferred", "excluded")
            ],
            "models": [
                {
                    "id": model,
                    "label": model,
                    "count": count,
                    "pct_eligible": pct(count, eligible),
                }
                for model, count in model_counts.most_common()
            ],
            "queues": {
                queue: queue_payload(queue)
                for queue in ("urgent", "priority", "all", "excluded")
            },
            "imbalance": {
                "max_min_ratio": ratio,
                "coefficient_of_variation": category_cv,
                "normalized_shannon_entropy": normalized_entropy,
                "minimum_total_per_damage_target": 2_000,
                "minimum_total_support": min(category_values, default=0),
                "minimum_total_target_met": min(category_values, default=0) >= 2_000,
                "most_represented": (
                    damage_labels[category_values.index(max(category_values))]
                    if category_values
                    else None
                ),
                "least_represented": (
                    damage_labels[category_values.index(min(category_values))]
                    if category_values
                    else None
                ),
            },
            "top_channels_volume": top_volume,
            "top_channels_harm": top_harm,
            "actions": [
                {
                    "id": action,
                    "label": action_names.get(action, action),
                    "count": count,
                }
                for action, count in action_counts.most_common()
            ],
            "reviewers": [
                {"reviewer": reviewer, "count": count}
                for reviewer, count in reviewer_counts.most_common(10)
            ],
            "activity": [
                {"hour": hour, **dict(activity[hour])}
                for hour in sorted(activity)[-24:]
            ],
        },
        "audit": audit_payload,
        "insights": insights,
    }


def _normalised_scope_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _labeling_scope_rows(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    anchor_chunk_id: str,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resuelve un video/canal desde un chunk y resume su estado revisable."""

    if scope not in LABELING_BULK_SCOPES:
        raise ValueError("El alcance debe ser video o channel")
    anchor = next(
        (
            row
            for row in campaign_rows
            if str(row.get("chunk_id")) == str(anchor_chunk_id)
        ),
        None,
    )
    if anchor is None:
        raise ValueError("El chunk de referencia no pertenece a la campaña")

    if scope == "video":
        raw_key = str(anchor.get("video_id") or "").strip()
        if not raw_key:
            raise ValueError(
                "El chunk no tiene video_id; no se puede aplicar una acción masiva"
            )
        scope_key = f"video:{raw_key}"
        display_name = str(anchor.get("video_title") or raw_key)
        rows = [
            row
            for row in campaign_rows
            if str(row.get("video_id") or "").strip() == raw_key
        ]
    else:
        channel_id = str(anchor.get("channel_id") or "").strip()
        channel_title = str(anchor.get("channel_title") or "").strip()
        normalised_title = _normalised_scope_text(channel_title)
        if not channel_id and not normalised_title:
            raise ValueError(
                "El chunk no tiene canal identificable; no se puede aplicar una acción masiva"
            )
        scope_key = (
            f"channel-id:{channel_id}"
            if channel_id
            else f"channel-title:{normalised_title}"
        )
        display_name = channel_title or channel_id

        def same_channel(row: dict[str, Any]) -> bool:
            row_channel_id = str(row.get("channel_id") or "").strip()
            row_title = _normalised_scope_text(row.get("channel_title"))
            if channel_id and row_channel_id:
                return row_channel_id == channel_id
            return bool(normalised_title and row_title == normalised_title)

        rows = [row for row in campaign_rows if same_channel(row)]

    deferred = resolved = pending = acceptable_pending = acceptable_total = 0
    for row in rows:
        acceptable_total += int(bool(row.get("coarse_labels")))
        review = latest_reviews.get(str(row.get("chunk_id")))
        if review is None or review.get("action") == "defer":
            pending += 1
            deferred += int(review is not None and review.get("action") == "defer")
            acceptable_pending += int(bool(row.get("coarse_labels")))
        else:
            resolved += 1
    summary = {
        "scope": scope,
        "scope_key": scope_key,
        "display_name": display_name,
        "total": len(rows),
        "pending": pending,
        "resolved": resolved,
        "deferred": deferred,
        "acceptable_total": acceptable_total,
        "acceptable_pending": acceptable_pending,
        "without_proposal_total": len(rows) - acceptable_total,
        "without_proposal_pending": pending - acceptable_pending,
    }
    return summary, rows


def _labeling_bulk_events(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    anchor_chunk_id: str,
    scope: str,
    action: str,
    include_resolved: bool,
    reviewer: str,
    notes: str,
    batch_id: str,
    final_labels: list[str] | None = None,
    flags: list[str] | None = None,
) -> tuple[dict[str, Any], list[ReviewEvent]]:
    """Construye eventos idempotentes para una revisión masiva trazable."""

    if action not in LABELING_BULK_ACTIONS:
        raise ValueError("La acción masiva debe ser accept, modify o reject")
    taxonomy = load_taxonomy()
    raw_labels = final_labels or []
    raw_flags = flags or []
    if not all(isinstance(value, str) for value in (*raw_labels, *raw_flags)):
        raise ValueError("Las categorías y flags deben ser cadenas de texto")
    common_labels = list(taxonomy.normalize_categories(raw_labels))
    common_flags = list(dict.fromkeys(raw_flags))
    unknown_flags = set(common_flags) - set(taxonomy.flags)
    if unknown_flags:
        raise ValueError(f"Flags desconocidos: {sorted(unknown_flags)}")
    if action == "modify" and not common_labels:
        raise ValueError("La clasificación masiva requiere una categoría final")
    if common_flags and not set(common_labels).intersection(taxonomy.damage_labels):
        raise ValueError("Los flags requieren al menos una categoría final de daño")
    clean_batch_id = str(batch_id).strip()
    if not clean_batch_id or len(clean_batch_id) > 128:
        raise ValueError("batch_id debe contener entre 1 y 128 caracteres")
    summary, scoped_rows = _labeling_scope_rows(
        campaign_rows,
        latest_reviews,
        anchor_chunk_id=anchor_chunk_id,
        scope=scope,
    )
    selected_rows = [
        row
        for row in scoped_rows
        if include_resolved
        or latest_reviews.get(str(row.get("chunk_id"))) is None
        or latest_reviews[str(row.get("chunk_id"))].get("action") == "defer"
    ]
    events: list[ReviewEvent] = []
    skipped_without_proposal = 0
    skipped_invalid: list[dict[str, str]] = []
    for row in selected_rows:
        proposed_labels = list(row.get("coarse_labels") or [])
        if action == "accept" and not proposed_labels:
            skipped_without_proposal += 1
            continue
        event_labels = (
            proposed_labels
            if action == "accept"
            else common_labels if action == "modify" else []
        )
        event_flags = (
            list(row.get("flags") or [])
            if action == "accept"
            else common_flags if action == "modify" else []
        )
        event_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "moderacion-peru|"
                f"{clean_batch_id}|{scope}|{action}|{','.join(event_labels)}|"
                f"{','.join(event_flags)}|{row['chunk_id']}",
            )
        )
        try:
            events.append(
                ReviewEvent(
                    event_id=event_id,
                    chunk_id=str(row["chunk_id"]),
                    action=action,
                    proposed_labels=proposed_labels,
                    final_labels=event_labels,
                    flags=event_flags,
                    reviewer=reviewer,
                    model_id=row.get("annotator_model"),
                    notes=notes,
                    decision_scope=scope,
                    decision_scope_key=summary["scope_key"],
                    batch_id=clean_batch_id,
                    batch_target_count=len(selected_rows),
                )
            )
        except (KeyError, ValueError, ValidationError) as exc:
            skipped_invalid.append(
                {"chunk_id": str(row.get("chunk_id") or ""), "error": str(exc)}
            )
    summary = {
        **summary,
        "include_resolved": include_resolved,
        "selected": len(selected_rows),
        "events_ready": len(events),
        "skipped_without_proposal": skipped_without_proposal,
        "applied_labels": common_labels if action == "modify" else [],
        "applied_flags": common_flags if action == "modify" else [],
        "skipped_invalid": skipped_invalid[:20],
        "skipped_invalid_count": len(skipped_invalid),
    }
    return summary, events


def _model_slot(model_family: str) -> str:
    family = model_family.casefold()
    if family.startswith("classical:"):
        return "classical"
    if family.startswith("qwen"):
        return "qwen"
    return "transformer"


def _production_registry_paths(
    registry_path: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = ModelRegistryEntry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    ).model_dump(mode="json")
    references = payload.get("comparison_registries") or {}
    paths: dict[str, Path] = {}
    for slot, reference in references.items():
        path = Path(reference["path"])
        path = path if path.is_absolute() else root / path
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise ValueError(
                f"Registro productivo ausente o alterado para {slot}: {path}"
            )
        paths[slot] = path
    if not paths:
        paths[_model_slot(str(payload.get("model_family", "")))] = registry_path
    return payload, paths


def _production_feedback(
    inference_path: Path,
    review_path: Path,
    ready_path: Path,
) -> dict[str, Any]:
    """Materializa retroalimentación humana deduplicada y estadísticas auditables."""

    taxonomy = load_taxonomy()
    inferences = list(read_jsonl(inference_path)) if inference_path.is_file() else []
    reviews = list(read_jsonl(review_path)) if review_path.is_file() else []
    by_event = {
        str(row.get("event_id")): row for row in inferences if row.get("event_id")
    }
    by_chunk_model: dict[tuple[str, str], dict[str, Any]] = {}
    for row in inferences:
        by_chunk_model[(str(row.get("chunk_id", "")), str(row.get("model_id", "")))] = (
            row
        )

    linked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for review in reviews:
        inference = by_event.get(str(review.get("source_event_id", "")))
        if inference is None:
            inference = by_chunk_model.get(
                (str(review.get("chunk_id", "")), str(review.get("model_id", "")))
            )
        if inference is not None:
            linked.append((review, inference))

    by_model: dict[str, dict[str, Any]] = {}
    reviews_by_event = {
        str(review.get("source_event_id")): review
        for review in reviews
        if review.get("source_event_id")
    }
    for event in inferences:
        slot = str(
            event.get("model_slot") or _model_slot(str(event.get("model_family", "")))
        )
        bucket = by_model.setdefault(
            slot,
            {
                "model_id": event.get("model_id"),
                "model_label": event.get("model_label") or slot,
                "inference_chunks": 0,
                "requires_review": 0,
                "reviews_completed": 0,
                "actions": {"accept": 0, "reject": 0, "modify": 0, "defer": 0},
                "categories": {
                    label: {"predicted": 0, "human_final": 0}
                    for label in taxonomy.target_labels
                },
            },
        )
        bucket["inference_chunks"] += 1
        bucket["requires_review"] += int(bool(event.get("requires_review")))
        for label in event.get("labels", []):
            if label in bucket["categories"]:
                bucket["categories"][label]["predicted"] += 1
        review = reviews_by_event.get(str(event.get("event_id")))
        if review:
            bucket["reviews_completed"] += 1
            action = str(review.get("action", ""))
            if action in bucket["actions"]:
                bucket["actions"][action] += 1
            for label in review.get("final_labels", []):
                if label in bucket["categories"]:
                    bucket["categories"][label]["human_final"] += 1

    decisions: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for review, inference in linked:
        if review.get("action") == "defer" or not review.get("final_labels"):
            continue
        identity_payload = "|".join(
            [
                str(inference.get("video_id") or ""),
                str(
                    inference.get("start_seconds")
                    if inference.get("start_seconds") is not None
                    else ""
                ),
                " ".join(str(inference.get("text", "")).casefold().split()),
            ]
        )
        identity = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        decisions.setdefault(identity, {})[str(review.get("reviewer", ""))] = (
            review,
            inference,
        )

    records: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for identity, reviewer_decisions in decisions.items():
        values = list(reviewer_decisions.values())
        label_sets = {tuple(sorted(review["final_labels"])) for review, _ in values}
        if len(label_sets) != 1:
            conflicts.append(identity)
            continue
        review, inference = values[-1]
        records.append(
            {
                "schema_version": "2.1.0",
                "chunk_id": f"prod_{identity[:24]}",
                "video_id": inference.get("video_id")
                or f"production_text_{identity[:16]}",
                "text": inference["text"],
                "coarse_labels": list(next(iter(label_sets))),
                "flags": review.get("flags", []),
                "label_source": "human_production_review_adjudicated",
                "sample_weight": 1.0,
                "source_ref": inference.get("source_url"),
                "start_seconds": inference.get("start_seconds"),
                "end_seconds": inference.get("end_seconds"),
                "reviewed_at": review.get("created_at"),
                "reviewer": review.get("reviewer"),
                "notes": review.get("notes", ""),
                "source_event_id": review.get("source_event_id"),
                "exclude_from_existing_validation_test": True,
            }
        )
    write_jsonl_atomic(ready_path, records)
    counts = {
        label: sum(label in row["coarse_labels"] for row in records)
        for label in taxonomy.target_labels
    }
    checks = {
        "unique_human_reviewed_at_least_500": len(records) >= RETRAIN_MINIMUM_TOTAL,
        "safe_at_least_200": counts[taxonomy.safe_label] >= RETRAIN_MINIMUM_SAFE,
        **{
            f"{label}_at_least_100": counts[label] >= RETRAIN_MINIMUM_PER_DAMAGE
            for label in taxonomy.damage_labels
        },
    }
    readiness = {
        "unique_adjudicated_chunks": len(records),
        "conflicting_chunks_excluded": len(conflicts),
        "category_counts": counts,
        "checks": checks,
        "ready_for_retraining_review": all(checks.values()),
        "rule_is_advisory": True,
        "output": str(ready_path),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_events": len(inferences),
        "total_human_reviews": len(reviews),
        "unlinked_human_reviews": len(reviews) - len(linked),
        "by_model": by_model,
        "retraining_export": str(review_path),
        "retraining_ready_dataset": str(ready_path),
        "retraining_readiness": readiness,
    }


def _consensus_result(
    events: list[dict[str, Any]],
    minimum: int = 2,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    if len(events) != 3 or minimum != 2:
        raise ValueError("El contrato productivo vigente exige consenso 2-de-3")
    votes = {
        label: sum(label in event["labels"] for event in events)
        for label in taxonomy.target_labels
    }
    labels = [label for label in taxonomy.target_labels if votes[label] >= minimum]
    disagreement = any(count not in {0, len(events)} for count in votes.values())
    reasons = []
    if disagreement:
        reasons.append("desacuerdo_entre_modelos")
    if any(event["requires_review"] for event in events):
        reasons.append("algun_modelo_activa_revision")
    if not labels:
        reasons.append("sin_mayoria_2_de_3")
    if taxonomy.safe_label in labels and len(labels) > 1:
        labels.remove(taxonomy.safe_label)
        reasons.append("conflicto_seguro_dano_en_votacion")
    event_id = str(uuid.uuid4())
    return {
        "schema_version": "2.1.0",
        "event_id": event_id,
        "chunk_id": (metadata or {}).get("chunk_id") or f"production-{event_id}",
        "text": events[0]["text"],
        "model_id": "consensus_2_of_3",
        "model_family": "ensemble_majority_vote",
        "model_slot": "consensus",
        "model_label": "Consenso mayoritario de los tres modelos",
        "taxonomy_contract": taxonomy.contract_id,
        "scores": {
            label: sum(event["scores"][label] for event in events) / len(events)
            for label in taxonomy.target_labels
        },
        "thresholds": None,
        "labels": labels,
        "confidence": "baja" if reasons else "alta",
        "requires_review": bool(reasons),
        "review_reasons": reasons,
        "votes": votes,
        "consensus_min_votes": minimum,
        "member_event_ids": [event["event_id"] for event in events],
        "created_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _youtube_video_id(value: str) -> str | None:
    try:
        parsed = urllib.parse.urlparse(value.strip())
    except ValueError:
        return None
    host = parsed.netloc.lower().split(":", 1)[0]
    if host in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/", 1)[0] or None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "embed"}:
            return parts[1]
    return None


def serve(
    *,
    mode: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    campaign: str | Path | None = None,
    reviews: str | Path | None = None,
    registry: str | Path | None = None,
    inferences: str | Path | None = None,
    retraining: str | Path | None = None,
) -> None:
    if mode not in {"labeling", "production"}:
        raise ValueError(mode)
    root = find_project_root()
    taxonomy = load_taxonomy()
    auth_user = os.getenv("MODERATOR_ACCESS_USER", "moderador").strip() or "moderador"
    auth_password = os.getenv("MODERATOR_ACCESS_PASSWORD", "").strip()
    if host not in {"127.0.0.1", "localhost", "::1"} and not auth_password:
        raise ValueError(
            "Escuchar fuera de loopback requiere MODERATOR_ACCESS_PASSWORD; "
            "use 127.0.0.1 para operación exclusivamente local"
        )
    html_path = (
        root
        / "flujo"
        / ("02_etiquetado" if mode == "labeling" else "04_produccion")
        / "frontend"
        / ("validacion_humana.html" if mode == "labeling" else "produccion.html")
    )
    dashboard_path = (
        root / "flujo" / "02_etiquetado" / "frontend" / "dashboard_etiquetado.html"
    )
    audit_metrics_path = (
        root / "docs" / "artefactos" / "auditoria_16k_panel_actual_v3_2_metrics.json"
    )
    campaign_path = Path(campaign).resolve() if campaign else None
    if reviews:
        review_path = Path(reviews).resolve()
    elif mode == "labeling":
        review_path = (
            root / "datos" / "etiquetado" / "humano" / "labeling_events_v2.jsonl"
        )
    else:
        review_path = root / "datos" / "produccion" / "review_events_v2.jsonl"
    registry_path = (
        Path(registry).resolve()
        if registry
        else root / "modelos" / "registro_modelos_5_salidas.json"
    )
    predictors: dict[str, ProductionPredictor] = {}
    prediction_lock = threading.Lock()
    persistence_lock = threading.RLock()
    inference_path = (
        Path(inferences).resolve()
        if inferences
        else root / "datos" / "produccion" / "inference_events_v2.jsonl"
    )
    ready_path = (
        Path(retraining).resolve()
        if retraining
        else root / "datos" / "produccion" / "retraining_ready_v2.jsonl"
    )
    campaign_rows = (
        list(read_jsonl(campaign_path))
        if campaign_path and campaign_path.is_file()
        else []
    )
    for index, row in enumerate(campaign_rows):
        previous = campaign_rows[index - 1] if index else None
        following = campaign_rows[index + 1] if index + 1 < len(campaign_rows) else None
        row["previous_text"] = (
            previous.get("text")
            if previous and previous.get("video_id") == row.get("video_id")
            else None
        )
        row["next_text"] = (
            following.get("text")
            if following and following.get("video_id") == row.get("video_id")
            else None
        )
    labeling_reviews = (
        {row["chunk_id"]: row for row in read_jsonl(review_path)}
        if mode == "labeling" and review_path.is_file()
        else {}
    )
    labeling_cohorts = sorted(
        {
            str(row.get("cohort") or row.get("label_source") or "sin_cohorte")
            for row in campaign_rows
        }
    )
    labeling_priority_total = sum(
        _is_labeling_priority(row, labeling_reviews.get(str(row.get("chunk_id"))))
        for row in campaign_rows
    )
    labeling_urgent_total = sum(_is_labeling_urgent(row) for row in campaign_rows)
    labeling_action_total = sum(
        _requires_labeling_action(row, labeling_reviews) for row in campaign_rows
    )
    labeling_pro_intermediate_total = sum(
        "pro" in str(row.get("annotator_model") or "").casefold()
        and (
            bool(row.get("needs_review"))
            or str(row.get("decision_status") or "") == "needs_review"
        )
        for row in campaign_rows
    )
    labeling_pro_unresolved_total = sum(
        "pro" in str(row.get("annotator_model") or "").casefold()
        and _labeling_filter_values(row, labeling_reviews)[2] == "pending"
        for row in campaign_rows
    )
    labeling_pro_damage_total = sum(
        "pro" in str(row.get("annotator_model") or "").casefold()
        and _labeling_filter_values(row, labeling_reviews)[2] != "excluded"
        and any(
            str(label).upper() != "SEGURO"
            for label in _labeling_filter_values(row, labeling_reviews)[0]
        )
        for row in campaign_rows
    )

    def registry_state() -> tuple[dict[str, Any], dict[str, Path]]:
        if not registry_path.is_file():
            raise FileNotFoundError("No existe un registro productivo validado")
        return _production_registry_paths(registry_path, root)

    def default_production_mode() -> str:
        _, paths = registry_state()
        if set(paths) == set(PRODUCTION_SLOTS):
            return "consensus"
        return next(slot for slot in PRODUCTION_SLOTS if slot in paths)

    def predict_slot(
        text: str,
        slot: str,
        registry_paths: dict[str, Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if slot not in registry_paths:
            raise ValueError(f"No hay un modelo validado para el slot {slot}")
        with prediction_lock:
            predictor = predictors.get(slot)
            if predictor is None:
                predictor = ProductionPredictor(registry_paths[slot])
                predictors[slot] = predictor
            scores = predictor.scores(text)
            decision = resolve_prediction(scores, predictor.entry.thresholds)
        event_id = str(uuid.uuid4())
        model_labels = {
            "classical": "Mejor modelo clásico",
            "transformer": "Mejor Transformer",
            "qwen": "Mejor Qwen ajustado",
        }
        event = {
            "schema_version": "2.1.0",
            "event_id": event_id,
            "chunk_id": (metadata or {}).get("chunk_id") or f"production-{event_id}",
            "text": text,
            "model_id": predictor.entry.model_id,
            "model_family": predictor.entry.model_family,
            "model_slot": slot,
            "model_label": model_labels[slot],
            "taxonomy_contract": predictor.entry.taxonomy_contract,
            "scores": scores,
            "thresholds": predictor.entry.thresholds,
            "labels": list(decision.labels),
            "confidence": "baja" if decision.requires_review else "alta",
            "requires_review": decision.requires_review,
            "review_reasons": list(decision.review_reasons),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        return event

    def predict_modes(
        text: str,
        mode_name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        registry_payload, registry_paths = registry_state()
        mode_name = mode_name.casefold().strip()
        if mode_name not in PRODUCTION_MODES:
            raise ValueError(f"Modo productivo no válido: {mode_name}")
        if mode_name in PRODUCTION_SLOTS:
            slots = [mode_name]
        elif mode_name == "compare":
            if len(registry_paths) < 2:
                raise ValueError("Comparar requiere al menos dos familias validadas")
            slots = [slot for slot in PRODUCTION_SLOTS if slot in registry_paths]
        else:
            missing = set(PRODUCTION_SLOTS) - set(registry_paths)
            if missing:
                raise ValueError(
                    "El consenso 2-de-3 requiere clásico, Transformer y Qwen; faltan: "
                    + ", ".join(sorted(missing))
                )
            slots = list(PRODUCTION_SLOTS)
        events = [
            predict_slot(text, slot, registry_paths, metadata=metadata)
            for slot in slots
        ]
        if mode_name == "consensus":
            minimum = int(registry_payload.get("consensus_min_votes", 2))
            events.append(_consensus_result(events, minimum, metadata=metadata))
        with persistence_lock:
            append_jsonl_once(inference_path, events, id_field="event_id")
        return events

    class Handler(BaseHTTPRequestHandler):
        server_version = "ModeracionPeru/2.1"

        def authorized(self) -> bool:
            if not auth_password:
                return True
            header = self.headers.get("Authorization", "")
            if not header.startswith("Basic "):
                return False
            try:
                supplied = base64.b64decode(header[6:], validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return False
            return hmac.compare_digest(supplied, f"{auth_user}:{auth_password}")

        def require_authorization(self) -> bool:
            if self.authorized():
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header(
                "WWW-Authenticate", 'Basic realm="Moderación Perú", charset="UTF-8"'
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def send_json(self, payload: Any, status: int = 200) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_payload(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Content-Length inválido") from exc
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError(
                    f"La solicitud debe ocupar entre 1 y {MAX_REQUEST_BYTES} bytes"
                )
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("El cuerpo JSON debe ser un objeto")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/health" and not self.require_authorization():
                return
            if parsed.path == "/api/health":
                self.send_json(
                    {"status": "ok", "mode": mode, "taxonomy": taxonomy.contract_id}
                )
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/api/config":
                registry_payload = None
                registry_models: dict[str, Any] = {}
                available_modes: list[str] = []
                registry_available = mode == "production" and registry_path.is_file()
                if registry_available:
                    try:
                        registry_payload, registry_paths = registry_state()
                        for slot, path in registry_paths.items():
                            member = ModelRegistryEntry.model_validate_json(
                                path.read_text(encoding="utf-8")
                            )
                            registry_models[slot] = {
                                "model_id": member.model_id,
                                "model_family": member.model_family,
                                "selection_metrics": member.selection_metrics,
                            }
                    except (OSError, ValueError, ValidationError) as exc:
                        self.send_json(
                            {"error": f"Registro productivo inválido: {exc}"},
                            HTTPStatus.CONFLICT,
                        )
                        return
                    available_modes = [
                        slot for slot in PRODUCTION_SLOTS if slot in registry_paths
                    ]
                    if len(registry_paths) >= 2:
                        available_modes.append("compare")
                    if set(registry_paths) == set(PRODUCTION_SLOTS):
                        available_modes.append("consensus")
                self.send_json(
                    {
                        "mode": mode,
                        "taxonomy": taxonomy.model_dump(),
                        "campaign_available": bool(
                            campaign_path and campaign_path.is_file()
                        ),
                        "campaign_total": len(campaign_rows),
                        "campaign_priority_total": labeling_priority_total,
                        "campaign_urgent_total": labeling_urgent_total,
                        "campaign_action_total": labeling_action_total,
                        "campaign_excluded_total": sum(
                            _is_labeling_excluded(row, labeling_reviews)
                            for row in campaign_rows
                        ),
                        "campaign_pro_unresolved_total": labeling_pro_unresolved_total,
                        "campaign_pro_intermediate_total": labeling_pro_intermediate_total,
                        "campaign_pro_damage_total": labeling_pro_damage_total,
                        "campaign_priority_rule": (
                            "Salida de DeepSeek Pro cuya decisión efectiva sigue pendiente "
                            "o conserva al menos una categoría de daño; un needs_review "
                            "histórico superado por CODEX o por una persona no cuenta como "
                            "pendiente final."
                        ),
                        "campaign_urgent_rule": (
                            "Conflicto entre propuestas de máxima prioridad que la "
                            "consolidación no pudo resolver automáticamente."
                        ),
                        "campaign_action_rule": (
                            "Chunks sin decisión final o diferidos expresamente. "
                            "Una decisión CODEX o humana superior cierra un "
                            "needs_review intermedio de Pro."
                        ),
                        "campaign_excluded_rule": (
                            "Chunks cuya decisión vigente los deja fuera del dataset "
                            "entrenable; pueden abrirse y reclasificarse."
                        ),
                        "campaign_cohorts": (
                            labeling_cohorts if mode == "labeling" else []
                        ),
                        "registry_available": registry_available,
                        "registry": registry_payload,
                        "models": registry_models,
                        "available_modes": available_modes,
                        "default_production_mode": (
                            "consensus"
                            if "consensus" in available_modes
                            else available_modes[0] if available_modes else None
                        ),
                        "reviews": str(review_path),
                    }
                )
                return
            if parsed.path == "/api/campaign":
                if not campaign_path or not campaign_path.is_file():
                    self.send_json(
                        {"error": "campaign_not_available"}, HTTPStatus.NOT_FOUND
                    )
                    return
                query = urllib.parse.parse_qs(parsed.query)
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(1000, max(1, int(query.get("limit", [50])[0])))
                cohort = query.get("cohort", [""])[0]
                only_pending = query.get("pending", ["0"])[0] == "1"
                queue = query.get("queue", ["all"])[0]
                priority_only = (
                    queue == "priority" or query.get("priority", ["0"])[0] == "1"
                )
                urgent_only = queue == "urgent"
                action_only = queue == "action"
                excluded_only = queue == "excluded"
                filter_labels = {value for value in query.get("label", []) if value}
                filter_flags = {value for value in query.get("flag", []) if value}
                filter_statuses = {value for value in query.get("status", []) if value}
                filter_labeling = {
                    value for value in query.get("labeling", []) if value
                }
                match_all = query.get("match", ["any"])[0] == "all"
                with persistence_lock:
                    page = _labeling_campaign_page(
                        campaign_rows,
                        labeling_reviews,
                        offset=offset,
                        limit=limit,
                        cohort=cohort,
                        only_pending=only_pending or action_only,
                        priority_only=priority_only,
                        urgent_only=urgent_only,
                        excluded_only=excluded_only,
                        filter_labels=filter_labels,
                        filter_flags=filter_flags,
                        filter_statuses=filter_statuses,
                        filter_labeling=filter_labeling,
                        match_all=match_all,
                    )
                self.send_json(page)
                return
            if parsed.path == "/api/review-scope" and mode == "labeling":
                query = urllib.parse.parse_qs(parsed.query)
                try:
                    with persistence_lock:
                        summary, _ = _labeling_scope_rows(
                            campaign_rows,
                            labeling_reviews,
                            anchor_chunk_id=str(query.get("chunk_id", [""])[0]),
                            scope=str(query.get("scope", ["video"])[0]),
                        )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(summary)
                return
            if parsed.path == "/api/progress" and mode == "labeling":
                query = urllib.parse.parse_qs(parsed.query)
                queue = query.get("queue", ["all"])[0]
                priority_only = (
                    queue == "priority" or query.get("priority", ["0"])[0] == "1"
                )
                urgent_only = queue == "urgent"
                action_only = queue == "action"
                excluded_only = queue == "excluded"
                cohort = query.get("cohort", [""])[0]
                filter_labels = {value for value in query.get("label", []) if value}
                filter_flags = {value for value in query.get("flag", []) if value}
                filter_statuses = {value for value in query.get("status", []) if value}
                filter_labeling = {
                    value for value in query.get("labeling", []) if value
                }
                match_all = query.get("match", ["any"])[0] == "all"
                with persistence_lock:
                    progress = _labeling_progress(
                        campaign_rows,
                        labeling_reviews,
                        priority_only=priority_only,
                        urgent_only=urgent_only,
                        action_only=action_only,
                        excluded_only=excluded_only,
                        cohort=cohort,
                        filter_labels=filter_labels,
                        filter_flags=filter_flags,
                        filter_statuses=filter_statuses,
                        filter_labeling=filter_labeling,
                        match_all=match_all,
                    )
                self.send_json(progress)
                return
            if parsed.path == "/api/reviews" and mode == "labeling":
                rows = list(read_jsonl(review_path)) if review_path.is_file() else []
                self.send_json({"total": len(rows), "rows": rows})
                return
            if parsed.path == "/api/dashboard" and mode == "labeling":
                audit_metrics: dict[str, Any] | None = None
                if audit_metrics_path.is_file():
                    try:
                        audit_metrics = json.loads(
                            audit_metrics_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        audit_metrics = None
                with persistence_lock:
                    dashboard = _labeling_dashboard(
                        campaign_rows,
                        labeling_reviews,
                        taxonomy,
                        audit_metrics=audit_metrics,
                    )
                self.send_json(dashboard)
                return
            if parsed.path == "/api/stats" and mode == "production":
                with persistence_lock:
                    statistics = _production_feedback(
                        inference_path, review_path, ready_path
                    )
                self.send_json(statistics)
                return
            if parsed.path == "/api/export":
                query = urllib.parse.parse_qs(parsed.query)
                export_path = (
                    ready_path
                    if query.get("kind", [""])[0] == "retraining"
                    else review_path
                )
                if export_path == ready_path:
                    with persistence_lock:
                        _production_feedback(inference_path, review_path, ready_path)
                with persistence_lock:
                    body = export_path.read_bytes() if export_path.is_file() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{export_path.name}"'
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/dashboard", "/dashboard.html"} and mode == "labeling":
                if not dashboard_path.is_file():
                    self.send_json(
                        {"error": f"frontend_missing:{dashboard_path}"},
                        HTTPStatus.NOT_FOUND,
                    )
                    return
                body = dashboard_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/", "/index.html"}:
                if not html_path.is_file():
                    self.send_json(
                        {"error": f"frontend_missing:{html_path}"}, HTTPStatus.NOT_FOUND
                    )
                    return
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(html_path.name)[0] + "; charset=utf-8",
                )
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if not self.require_authorization():
                return
            request_path = urllib.parse.urlparse(self.path).path
            try:
                payload = self.read_payload()
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if request_path == "/api/predict" and mode == "production":
                try:
                    text = str(payload.get("text", "")).strip()
                    if not text:
                        raise ValueError("El texto no puede estar vacío")
                    mode_name = str(payload.get("mode") or default_production_mode())
                    events = predict_modes(
                        text,
                        mode_name,
                        metadata={"chunk_id": f"production-text-{uuid.uuid4()}"},
                    )
                    self.send_json({"mode": mode_name, "results": events})
                except (
                    FileNotFoundError,
                    ValueError,
                    RuntimeError,
                    ImportError,
                ) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if request_path == "/api/analyze" and mode == "production":
                try:
                    value = str(payload.get("input", "")).strip()
                    if not value:
                        raise ValueError("La entrada está vacía")
                    forced = payload.get("input_type", "auto")
                    video_id = _youtube_video_id(value) if forced != "text" else None
                    if forced == "youtube" and not video_id:
                        raise ValueError("No se reconoció un enlace de YouTube")
                    max_chunks = min(1000, max(1, int(payload.get("max_chunks", 300))))
                    mode_name = (
                        str(payload.get("mode") or default_production_mode())
                        .casefold()
                        .strip()
                    )
                    if video_id:
                        from .acquisition import fetch_youtube_subtitles

                        cache_path = (
                            root
                            / "datos"
                            / "produccion"
                            / "transcript_cache"
                            / f"{video_id}.json"
                        )
                        if cache_path.is_file():
                            transcript = json.loads(
                                cache_path.read_text(encoding="utf-8")
                            )
                            subtitle_status = "reused_cache"
                        else:
                            transcript = fetch_youtube_subtitles(
                                {"video_id": video_id, "url": value}
                            )
                            write_json_atomic(cache_path, transcript)
                            subtitle_status = "fetched_subtitles_only"
                        segments = [
                            TranscriptSegment(
                                float(item["start"]),
                                float(item["duration"]),
                                str(item["text"]),
                            )
                            for item in transcript["segments"]
                        ]
                        chunks = chunk_transcript(video_id, segments)
                        if len(chunks) > max_chunks:
                            raise ValueError(
                                f"La entrada produjo {len(chunks)} chunks; "
                                f"el límite configurado es {max_chunks}"
                            )
                        results = []
                        for chunk in chunks:
                            metadata = {
                                "chunk_id": chunk.chunk_id,
                                "video_id": video_id,
                                "video_title": transcript.get("title"),
                                "channel_title": transcript.get("channel"),
                                "source_url": transcript.get("url"),
                                "start_seconds": chunk.start_seconds,
                                "end_seconds": chunk.end_seconds,
                            }
                            results.append(
                                {
                                    **metadata,
                                    "text": chunk.text,
                                    "results": predict_modes(
                                        chunk.text,
                                        mode_name,
                                        metadata=metadata,
                                    ),
                                }
                            )
                        alert_chunks = sum(
                            any(
                                event["labels"] != [taxonomy.safe_label]
                                for event in chunk["results"]
                                if mode_name == "compare"
                                or event["model_slot"] == mode_name
                            )
                            for chunk in results
                        )
                        self.send_json(
                            {
                                "mode": mode_name,
                                "input_type": "youtube",
                                "video_id": video_id,
                                "embed_url": f"https://www.youtube-nocookie.com/embed/{video_id}",
                                "subtitle_status": subtitle_status,
                                "subtitle_language": transcript.get("language"),
                                "subtitle_kind": transcript.get("subtitle_source"),
                                "chunks": results,
                                "summary": {
                                    "chunks": len(results),
                                    "alert_chunks": alert_chunks,
                                    "models_executed": sorted(
                                        {
                                            event["model_slot"]
                                            for chunk in results
                                            for event in chunk["results"]
                                        }
                                    ),
                                },
                            }
                        )
                    else:
                        chunk_id = f"production-text-{uuid.uuid4()}"
                        events = predict_modes(
                            value,
                            mode_name,
                            metadata={"chunk_id": chunk_id},
                        )
                        relevant = [
                            event
                            for event in events
                            if mode_name == "compare"
                            or event["model_slot"] == mode_name
                        ]
                        self.send_json(
                            {
                                "mode": mode_name,
                                "input_type": "text",
                                "chunks": [
                                    {
                                        "chunk_id": chunk_id,
                                        "text": value,
                                        "results": events,
                                    }
                                ],
                                "summary": {
                                    "chunks": 1,
                                    "alert_chunks": int(
                                        any(
                                            event["labels"] != [taxonomy.safe_label]
                                            for event in relevant
                                        )
                                    ),
                                    "models_executed": [
                                        event["model_slot"] for event in events
                                    ],
                                },
                            }
                        )
                except (
                    FileNotFoundError,
                    ValueError,
                    RuntimeError,
                    ImportError,
                ) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if request_path == "/api/review/bulk" and mode == "labeling":
                try:
                    if payload.get("confirm") is not True:
                        raise ValueError(
                            "La acción masiva requiere confirmación explícita"
                        )
                    include_resolved = payload.get("include_resolved", False)
                    if not isinstance(include_resolved, bool):
                        raise ValueError("include_resolved debe ser booleano")
                    final_labels = payload.get("final_labels", [])
                    flags = payload.get("flags", [])
                    if not isinstance(final_labels, list) or not isinstance(
                        flags, list
                    ):
                        raise ValueError("final_labels y flags deben ser listas")
                    reviewer = str(payload.get("reviewer", "ANON")).strip() or "ANON"
                    salt = os.getenv("MODPERU_REVIEW_SALT", taxonomy.contract_id)
                    pseudonym = (
                        "reviewer-"
                        + hashlib.sha256(
                            f"{salt}|{reviewer}".encode("utf-8")
                        ).hexdigest()[:16]
                    )
                    with persistence_lock:
                        summary, events = _labeling_bulk_events(
                            campaign_rows,
                            labeling_reviews,
                            anchor_chunk_id=str(payload.get("chunk_id", "")),
                            scope=str(payload.get("scope", "")),
                            action=str(payload.get("action", "")),
                            include_resolved=include_resolved,
                            reviewer=pseudonym,
                            notes=str(payload.get("notes", "")).strip(),
                            batch_id=str(payload.get("batch_id", "")),
                            final_labels=final_labels,
                            flags=flags,
                        )
                        if events:
                            event_rows = [
                                event.model_dump(mode="json") for event in events
                            ]
                            added, skipped = append_jsonl_once(
                                review_path, event_rows, id_field="event_id"
                            )
                            if skipped:
                                labeling_reviews.clear()
                                labeling_reviews.update(
                                    {
                                        row["chunk_id"]: row
                                        for row in read_jsonl(review_path)
                                    }
                                )
                            else:
                                labeling_reviews.update(
                                    {row["chunk_id"]: row for row in event_rows}
                                )
                        else:
                            added = skipped = 0
                except (KeyError, ValueError, ValidationError) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(
                    {
                        "saved": added,
                        "duplicates": skipped,
                        "batch_id": str(payload.get("batch_id", "")),
                        "summary": summary,
                    }
                )
                return
            if request_path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                reviewer = str(payload.get("reviewer", "ANON")).strip() or "ANON"
                salt = os.getenv("MODPERU_REVIEW_SALT", taxonomy.contract_id)
                pseudonym = (
                    "reviewer-"
                    + hashlib.sha256(f"{salt}|{reviewer}".encode("utf-8")).hexdigest()[
                        :16
                    ]
                )
                final_labels = payload.get("final_labels", [])
                if mode == "production" and payload.get("action") == "reject":
                    final_labels = [taxonomy.safe_label]
                event = ReviewEvent(
                    event_id=str(payload.get("event_id") or uuid.uuid4()),
                    chunk_id=payload["chunk_id"],
                    action=payload["action"],
                    proposed_labels=payload.get("proposed_labels", []),
                    final_labels=final_labels,
                    flags=payload.get("flags", []),
                    reviewer=pseudonym,
                    model_id=payload.get("model_id"),
                    source_event_id=payload.get("source_event_id"),
                    notes=payload.get("notes", ""),
                )
                with persistence_lock:
                    added, skipped = append_jsonl_once(
                        review_path,
                        [event.model_dump(mode="json")],
                        id_field="event_id",
                    )
                    if mode == "production":
                        _production_feedback(inference_path, review_path, ready_path)
                    elif added:
                        labeling_reviews[event.chunk_id] = event.model_dump(mode="json")
            except (KeyError, ValueError, ValidationError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json(
                {
                    "saved": bool(added),
                    "duplicate": bool(skipped),
                    "event": event.model_dump(mode="json"),
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Moderación Perú ({mode}): http://{host}:{server.server_port}")
    print(f"Eventos: {review_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
