from __future__ import annotations

import json
import mimetypes
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .io import append_jsonl_once, read_jsonl
from .paths import find_project_root
from .schemas import ReviewEvent
from .taxonomy import load_taxonomy


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def serve(
    *,
    mode: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    campaign: str | Path | None = None,
    reviews: str | Path | None = None,
) -> None:
    if mode not in {"labeling", "production"}:
        raise ValueError(mode)
    root = find_project_root()
    taxonomy = load_taxonomy()
    html_path = root / "flujo" / ("02_etiquetado" if mode == "labeling" else "04_produccion") / "frontend" / ("validacion_humana.html" if mode == "labeling" else "produccion.html")
    campaign_path = Path(campaign).resolve() if campaign else None
    review_path = Path(reviews).resolve() if reviews else root / "datos" / "etiquetado" / "humano" / f"{mode}_events_v2.jsonl"

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
                self.send_json({
                    "mode": mode,
                    "taxonomy": taxonomy.model_dump(),
                    "campaign_available": bool(campaign_path and campaign_path.is_file()),
                    "reviews": str(review_path),
                })
                return
            if parsed.path == "/api/campaign":
                if not campaign_path or not campaign_path.is_file():
                    self.send_json({"error": "campaign_not_available"}, HTTPStatus.NOT_FOUND)
                    return
                query = urllib.parse.parse_qs(parsed.query)
                offset = max(0, int(query.get("offset", [0])[0]))
                limit = min(200, max(1, int(query.get("limit", [50])[0])))
                rows = list(read_jsonl(campaign_path))
                self.send_json({"total": len(rows), "offset": offset, "rows": rows[offset:offset + limit]})
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
            if self.path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                event = ReviewEvent(
                    event_id=str(payload.get("event_id") or uuid.uuid4()),
                    chunk_id=payload["chunk_id"],
                    action=payload["action"],
                    proposed_labels=payload.get("proposed_labels", []),
                    final_labels=payload.get("final_labels", []),
                    flags=payload.get("flags", []),
                    reviewer=payload.get("reviewer", "ANON"),
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
