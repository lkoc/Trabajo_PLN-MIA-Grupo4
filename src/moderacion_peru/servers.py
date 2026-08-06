from __future__ import annotations

import json
import hashlib
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

from .io import append_jsonl_once, read_jsonl, write_json_atomic
from .incremental import TranscriptSegment, chunk_transcript
from .paths import find_project_root
from .schemas import ReviewEvent
from .registry import ProductionPredictor
from .taxonomy import load_taxonomy
from .training import resolve_prediction


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
) -> None:
    if mode not in {"labeling", "production"}:
        raise ValueError(mode)
    root = find_project_root()
    taxonomy = load_taxonomy()
    html_path = root / "flujo" / ("02_etiquetado" if mode == "labeling" else "04_produccion") / "frontend" / ("validacion_humana.html" if mode == "labeling" else "produccion.html")
    campaign_path = Path(campaign).resolve() if campaign else None
    if reviews:
        review_path = Path(reviews).resolve()
    elif mode == "labeling":
        review_path = root / "datos" / "etiquetado" / "humano" / "labeling_events_v2.jsonl"
    else:
        review_path = root / "datos" / "produccion" / "review_events_v2.jsonl"
    registry_path = Path(registry).resolve() if registry else root / "modelos" / "registro_modelos_5_salidas.json"
    predictor: ProductionPredictor | None = None
    prediction_lock = threading.Lock()
    inference_path = root / "datos" / "produccion" / "inference_events_v2.jsonl"
    campaign_rows = list(read_jsonl(campaign_path)) if campaign_path and campaign_path.is_file() else []
    for index, row in enumerate(campaign_rows):
        previous = campaign_rows[index - 1] if index else None
        following = campaign_rows[index + 1] if index + 1 < len(campaign_rows) else None
        row["previous_text"] = previous.get("text") if previous and previous.get("video_id") == row.get("video_id") else None
        row["next_text"] = following.get("text") if following and following.get("video_id") == row.get("video_id") else None

    def predict_one(text: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal predictor
        if not registry_path.is_file():
            raise FileNotFoundError("No existe un registro productivo validado")
        with prediction_lock:
            predictor = predictor or ProductionPredictor(registry_path)
            scores = predictor.scores(text)
            decision = resolve_prediction(scores, predictor.entry.thresholds)
        event_id = str(uuid.uuid4())
        event = {
            "schema_version": "2.1.0",
            "event_id": event_id,
            "chunk_id": (metadata or {}).get("chunk_id") or f"production-{event_id}",
            "text": text,
            "model_id": predictor.entry.model_id,
            "taxonomy_contract": predictor.entry.taxonomy_contract,
            "scores": scores,
            "thresholds": predictor.entry.thresholds,
            "labels": list(decision.labels),
            "requires_review": decision.requires_review,
            "review_reasons": list(decision.review_reasons),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        append_jsonl_once(inference_path, [event], id_field="event_id")
        return event

    class Handler(BaseHTTPRequestHandler):
        server_version = "ModeracionPeru/2.1"

        def send_json(self, payload: Any, status: int = 200) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                self.send_json({"status": "ok", "mode": mode, "taxonomy": taxonomy.contract_id})
                return
            if parsed.path == "/api/config":
                registry_payload = None
                if registry_path.is_file():
                    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
                self.send_json({
                    "mode": mode,
                    "taxonomy": taxonomy.model_dump(),
                    "campaign_available": bool(campaign_path and campaign_path.is_file()),
                    "registry_available": registry_path.is_file(),
                    "registry": registry_payload,
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
                rows = campaign_rows
                if query.get("pending", ["0"])[0] == "1" and review_path.is_file():
                    reviewed = {row["chunk_id"] for row in read_jsonl(review_path)}
                    rows = [row for row in rows if row.get("chunk_id") not in reviewed]
                self.send_json({"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]})
                return
            if parsed.path == "/api/progress" and mode == "labeling":
                total = len(campaign_rows)
                events = list(read_jsonl(review_path)) if review_path.is_file() else []
                latest = {event["chunk_id"]: event for event in events}
                resolved = sum(event.get("action") != "defer" for event in latest.values())
                self.send_json(
                    {
                        "total": total,
                        "reviewed": len(latest),
                        "resolved": resolved,
                        "deferred": sum(event.get("action") == "defer" for event in latest.values()),
                        "pending": max(0, total - len(latest)),
                        "progress_pct": 100 * len(latest) / max(1, total),
                    }
                )
                return
            if parsed.path == "/api/reviews" and mode == "labeling":
                rows = list(read_jsonl(review_path)) if review_path.is_file() else []
                self.send_json({"total": len(rows), "rows": rows})
                return
            if parsed.path == "/api/stats" and mode == "production":
                inferences = list(read_jsonl(inference_path)) if inference_path.is_file() else []
                reviews_done = list(read_jsonl(review_path)) if review_path.is_file() else []
                by_label = {label: 0 for label in taxonomy.target_labels}
                for event in inferences:
                    for label in event.get("labels", []):
                        if label in by_label:
                            by_label[label] += 1
                self.send_json(
                    {
                        "inference_events": len(inferences),
                        "requires_review": sum(bool(event.get("requires_review")) for event in inferences),
                        "human_reviews": len(reviews_done),
                        "by_label": by_label,
                        "retraining_export": str(review_path),
                    }
                )
                return
            if parsed.path == "/api/export":
                body = review_path.read_bytes() if review_path.is_file() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{review_path.name}"')
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
            nonlocal predictor
            if self.path == "/api/predict" and mode == "production":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    self.send_json(predict_one(str(payload.get("text", ""))))
                except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if self.path == "/api/analyze" and mode == "production":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    value = str(payload.get("input", "")).strip()
                    if not value:
                        raise ValueError("La entrada está vacía")
                    forced = payload.get("input_type", "auto")
                    video_id = _youtube_video_id(value) if forced != "text" else None
                    if forced == "youtube" and not video_id:
                        raise ValueError("No se reconoció un enlace de YouTube")
                    max_chunks = min(1000, max(1, int(payload.get("max_chunks", 300))))
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
                        chunks = chunk_transcript(video_id, segments)[:max_chunks]
                        results = [
                            predict_one(
                                chunk.text,
                                metadata={
                                    "chunk_id": chunk.chunk_id,
                                    "video_id": video_id,
                                    "video_title": transcript.get("title"),
                                    "channel_title": transcript.get("channel"),
                                    "source_url": transcript.get("url"),
                                    "start_seconds": chunk.start_seconds,
                                    "end_seconds": chunk.end_seconds,
                                },
                            )
                            for chunk in chunks
                        ]
                        self.send_json(
                            {
                                "input_type": "youtube",
                                "video_id": video_id,
                                "embed_url": f"https://www.youtube-nocookie.com/embed/{video_id}",
                                "subtitle_status": subtitle_status,
                                "subtitle_language": transcript.get("language"),
                                "subtitle_kind": transcript.get("subtitle_source"),
                                "chunks": results,
                                "summary": {"chunks": len(results), "alert_chunks": sum(event["labels"] != [taxonomy.safe_label] for event in results)},
                            }
                        )
                    else:
                        result = predict_one(value)
                        self.send_json(
                            {
                                "input_type": "text",
                                "chunks": [result],
                                "summary": {"chunks": 1, "alert_chunks": int(result["labels"] != [taxonomy.safe_label])},
                            }
                        )
                except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if self.path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                reviewer = str(payload.get("reviewer", "ANON")).strip() or "ANON"
                salt = os.getenv("MODPERU_REVIEW_SALT", taxonomy.contract_id)
                pseudonym = "reviewer-" + hashlib.sha256(f"{salt}|{reviewer}".encode("utf-8")).hexdigest()[:16]
                event = ReviewEvent(
                    event_id=str(payload.get("event_id") or uuid.uuid4()),
                    chunk_id=payload["chunk_id"],
                    action=payload["action"],
                    proposed_labels=payload.get("proposed_labels", []),
                    final_labels=payload.get("final_labels", []),
                    flags=payload.get("flags", []),
                    reviewer=pseudonym,
                    model_id=payload.get("model_id"),
                    notes=payload.get("notes", ""),
                )
                added, skipped = append_jsonl_once(
                    review_path, [event.model_dump(mode="json")], id_field="event_id"
                )
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
