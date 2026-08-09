from __future__ import annotations

import base64
import json
import hashlib
import hmac
import mimetypes
import os
import threading
import urllib.parse
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .io import append_jsonl_once, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .incremental import TranscriptSegment, chunk_transcript
from .paths import find_project_root
from .schemas import ModelRegistryEntry, ReviewEvent
from .registry import ProductionPredictor
from .taxonomy import load_taxonomy
from .training import resolve_prediction


PRODUCTION_SLOTS = ("classical", "transformer", "qwen")
PRODUCTION_MODES = (*PRODUCTION_SLOTS, "compare", "consensus")
RETRAIN_MINIMUM_TOTAL = 500
RETRAIN_MINIMUM_SAFE = 200
RETRAIN_MINIMUM_PER_DAMAGE = 100
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def _labeling_campaign_page(
    campaign_rows: list[dict[str, Any]],
    latest_reviews: dict[str, dict[str, Any]],
    *,
    offset: int,
    limit: int,
    cohort: str = "",
    only_pending: bool = False,
) -> dict[str, Any]:
    """Pagina la campaña sin copiar el corpus completo para cada solicitud."""

    page_indices: list[int] = []
    page_rows: list[dict[str, Any]] = []
    matching = 0
    for index, row in enumerate(campaign_rows):
        row_cohort = str(row.get("cohort") or row.get("label_source") or "sin_cohorte")
        if cohort and cohort != "all" and row_cohort != cohort:
            continue
        review = latest_reviews.get(str(row.get("chunk_id")))
        if only_pending and review is not None and review.get("action") != "defer":
            continue
        if offset <= matching < offset + limit:
            page_indices.append(index)
            page_rows.append(row)
        matching += 1
    return {
        "total": matching,
        "offset": offset,
        "indices": page_indices,
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
) -> dict[str, Any]:
    total = len(campaign_rows)
    events = list(latest_reviews.values())
    resolved = sum(event.get("action") != "defer" for event in events)
    deferred = sum(event.get("action") == "defer" for event in events)
    return {
        "total": total,
        "reviewed": len(events),
        "resolved": resolved,
        "deferred": deferred,
        "pending": max(0, total - resolved),
        "progress_pct": 100 * resolved / max(1, total),
    }


def _model_slot(model_family: str) -> str:
    family = model_family.casefold()
    if family.startswith("classical:"):
        return "classical"
    if family.startswith("qwen"):
        return "qwen"
    return "transformer"


def _production_registry_paths(registry_path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    payload = ModelRegistryEntry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    ).model_dump(mode="json")
    references = payload.get("comparison_registries") or {}
    paths: dict[str, Path] = {}
    for slot, reference in references.items():
        path = Path(reference["path"])
        path = path if path.is_absolute() else root / path
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise ValueError(f"Registro productivo ausente o alterado para {slot}: {path}")
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
    by_event = {str(row.get("event_id")): row for row in inferences if row.get("event_id")}
    by_chunk_model: dict[tuple[str, str], dict[str, Any]] = {}
    for row in inferences:
        by_chunk_model[(str(row.get("chunk_id", "")), str(row.get("model_id", "")))] = row

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
        slot = str(event.get("model_slot") or _model_slot(str(event.get("model_family", ""))))
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
                str(inference.get("start_seconds") if inference.get("start_seconds") is not None else ""),
                " ".join(str(inference.get("text", "")).casefold().split()),
            ]
        )
        identity = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        decisions.setdefault(identity, {})[str(review.get("reviewer", ""))] = (review, inference)

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
                "video_id": inference.get("video_id") or f"production_text_{identity[:16]}",
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
    html_path = root / "flujo" / ("02_etiquetado" if mode == "labeling" else "04_produccion") / "frontend" / ("validacion_humana.html" if mode == "labeling" else "produccion.html")
    campaign_path = Path(campaign).resolve() if campaign else None
    if reviews:
        review_path = Path(reviews).resolve()
    elif mode == "labeling":
        review_path = root / "datos" / "etiquetado" / "humano" / "labeling_events_v2.jsonl"
    else:
        review_path = root / "datos" / "produccion" / "review_events_v2.jsonl"
    registry_path = Path(registry).resolve() if registry else root / "modelos" / "registro_modelos_5_salidas.json"
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
    campaign_rows = list(read_jsonl(campaign_path)) if campaign_path and campaign_path.is_file() else []
    for index, row in enumerate(campaign_rows):
        previous = campaign_rows[index - 1] if index else None
        following = campaign_rows[index + 1] if index + 1 < len(campaign_rows) else None
        row["previous_text"] = previous.get("text") if previous and previous.get("video_id") == row.get("video_id") else None
        row["next_text"] = following.get("text") if following and following.get("video_id") == row.get("video_id") else None
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
            self.send_header("WWW-Authenticate", 'Basic realm="Moderación Perú", charset="UTF-8"')
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
                self.send_json({"status": "ok", "mode": mode, "taxonomy": taxonomy.contract_id})
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
                    available_modes = [slot for slot in PRODUCTION_SLOTS if slot in registry_paths]
                    if len(registry_paths) >= 2:
                        available_modes.append("compare")
                    if set(registry_paths) == set(PRODUCTION_SLOTS):
                        available_modes.append("consensus")
                self.send_json({
                    "mode": mode,
                    "taxonomy": taxonomy.model_dump(),
                    "campaign_available": bool(campaign_path and campaign_path.is_file()),
                    "campaign_total": len(campaign_rows),
                    "campaign_cohorts": labeling_cohorts if mode == "labeling" else [],
                    "registry_available": registry_available,
                    "registry": registry_payload,
                    "models": registry_models,
                    "available_modes": available_modes,
                    "default_production_mode": (
                        "consensus" if "consensus" in available_modes
                        else available_modes[0] if available_modes else None
                    ),
                    "reviews": str(review_path),
                })
                return
            if parsed.path == "/api/campaign":
                if not campaign_path or not campaign_path.is_file():
                    self.send_json({"error": "campaign_not_available"}, HTTPStatus.NOT_FOUND)
                    return
                query = urllib.parse.parse_qs(parsed.query)
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(1000, max(1, int(query.get("limit", [50])[0])))
                cohort = query.get("cohort", [""])[0]
                only_pending = query.get("pending", ["0"])[0] == "1"
                with persistence_lock:
                    page = _labeling_campaign_page(
                        campaign_rows,
                        labeling_reviews,
                        offset=offset,
                        limit=limit,
                        cohort=cohort,
                        only_pending=only_pending,
                    )
                self.send_json(page)
                return
            if parsed.path == "/api/progress" and mode == "labeling":
                with persistence_lock:
                    progress = _labeling_progress(campaign_rows, labeling_reviews)
                self.send_json(progress)
                return
            if parsed.path == "/api/reviews" and mode == "labeling":
                rows = list(read_jsonl(review_path)) if review_path.is_file() else []
                self.send_json({"total": len(rows), "rows": rows})
                return
            if parsed.path == "/api/stats" and mode == "production":
                with persistence_lock:
                    statistics = _production_feedback(inference_path, review_path, ready_path)
                self.send_json(statistics)
                return
            if parsed.path == "/api/export":
                query = urllib.parse.parse_qs(parsed.query)
                export_path = ready_path if query.get("kind", [""])[0] == "retraining" else review_path
                if export_path == ready_path:
                    with persistence_lock:
                        _production_feedback(inference_path, review_path, ready_path)
                with persistence_lock:
                    body = export_path.read_bytes() if export_path.is_file() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{export_path.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/", "/index.html"}:
                if not html_path.is_file():
                    self.send_json({"error": f"frontend_missing:{html_path}"}, HTTPStatus.NOT_FOUND)
                    return
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(html_path.name)[0] + "; charset=utf-8")
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
                except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
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
                    mode_name = str(payload.get("mode") or default_production_mode()).casefold().strip()
                    if video_id:
                        from .acquisition import fetch_youtube_subtitles

                        cache_path = root / "datos" / "produccion" / "transcript_cache" / f"{video_id}.json"
                        if cache_path.is_file():
                            transcript = json.loads(cache_path.read_text(encoding="utf-8"))
                            subtitle_status = "reused_cache"
                        else:
                            transcript = fetch_youtube_subtitles({"video_id": video_id, "url": value})
                            write_json_atomic(cache_path, transcript)
                            subtitle_status = "fetched_subtitles_only"
                        segments = [TranscriptSegment(float(item["start"]), float(item["duration"]), str(item["text"])) for item in transcript["segments"]]
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
                                if mode_name == "compare" or event["model_slot"] == mode_name
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
                                        {event["model_slot"] for chunk in results for event in chunk["results"]}
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
                            event for event in events
                            if mode_name == "compare" or event["model_slot"] == mode_name
                        ]
                        self.send_json(
                            {
                                "mode": mode_name,
                                "input_type": "text",
                                "chunks": [{"chunk_id": chunk_id, "text": value, "results": events}],
                                "summary": {
                                    "chunks": 1,
                                    "alert_chunks": int(
                                        any(event["labels"] != [taxonomy.safe_label] for event in relevant)
                                    ),
                                    "models_executed": [event["model_slot"] for event in events],
                                },
                            }
                        )
                except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if request_path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                reviewer = str(payload.get("reviewer", "ANON")).strip() or "ANON"
                salt = os.getenv("MODPERU_REVIEW_SALT", taxonomy.contract_id)
                pseudonym = "reviewer-" + hashlib.sha256(f"{salt}|{reviewer}".encode("utf-8")).hexdigest()[:16]
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
                        review_path, [event.model_dump(mode="json")], id_field="event_id"
                    )
                    if mode == "production":
                        _production_feedback(inference_path, review_path, ready_path)
                    elif added:
                        labeling_reviews[event.chunk_id] = event.model_dump(mode="json")
            except (KeyError, ValueError, ValidationError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"saved": bool(added), "duplicate": bool(skipped), "event": event.model_dump(mode="json")})

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
