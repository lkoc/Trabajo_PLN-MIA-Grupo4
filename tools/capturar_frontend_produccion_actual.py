from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "flujo/04_produccion/frontend/produccion.html"
FREEZE = ROOT / "resultados/modelos/seleccion_congelada.json"
TAXONOMY = ROOT / "config/taxonomia_v2.json"


def preview_config() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    members = freeze["members"]
    return {
        "registry_available": True,
        "taxonomy": taxonomy,
        "registry": {
            "taxonomy_contract": "moderacion_peru_5_salidas_v2",
            "model_id": freeze["selected_id"],
            "model_family": "ensemble:soft_mean",
            "winner_status": freeze["winner_status"],
        },
        "models": {
            "classical": {"model_id": members[0]},
            "transformer": {"model_id": members[1]},
            "qwen": {"model_id": members[2]},
            "ensemble": {"model_id": freeze["selected_id"]},
        },
        "available_modes": [
            "classical",
            "transformer",
            "qwen",
            "ensemble",
            "compare",
        ],
        "default_production_mode": "ensemble",
    }


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            body = json.dumps(preview_config(), ensure_ascii=False).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif self.path in {"/", "/produccion.html"}:
            body = HTML.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            body = b'{"error":"Vista previa: inferencia deshabilitada"}'
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--chrome",
        type=Path,
        default=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), PreviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="modperu_frontend_preview_") as profile:
            subprocess.run(
                [
                    str(args.chrome),
                    "--headless=new",
                    "--hide-scrollbars",
                    "--disable-gpu",
                    "--no-first-run",
                    f"--user-data-dir={profile}",
                    "--window-size=1680,1050",
                    "--virtual-time-budget=2500",
                    f"--screenshot={args.output.resolve()}",
                    f"http://127.0.0.1:{server.server_port}/",
                ],
                check=True,
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
