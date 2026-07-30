"""Genera las cuatro capturas reales del panel operativo del Apéndice C.

El servidor 05 debe estar disponible en ``http://127.0.0.1:8765``. El guion
usa la interfaz pública, espera la respuesta real de ``/api/analyze`` y sólo
aplica una hoja de estilo de impresión para recortar navegación, video y
controles de revisión. No inserta ni modifica predicciones en el DOM.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
SERVER_URL = "http://127.0.0.1:8765"
FIGURES = ROOT / "Documento_final_paper" / "figuras"
EVIDENCE_PATH = FIGURES / "captura_panel_operacion.evidence.json"

CHUNK_ID = "jSOLkn7q83Y_0050"
CHUNK_SPLIT = "validation"
CHUNK_TEXT = (
    "casi esculcándola a ver si tiene algún objeto extraño, como un seguridad de "
    "aeropuerto más o menos. Entonces yo cuando duermo con ella o tengo la mano en "
    "su poto. Qué rico. Tú también. Yo también. O cuchare o quitándole carquita "
    "debajo de la teta también. Las chicas tienen carcas. Ay, no sean, no sean nada. "
    "Por favor, amigo, llega hoy día a tu casa. Hola, Holanda. Levántale la teta a tu "
    "mujer y huélele. Y huélele. Ahí. Mi ombligo huele más rico."
)

VIDEO_ID = "mTLmFx4SyH8"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
VIDEO_TITLE = "Alias ‘El Wichi’: extorsionaba a sus víctimas bajo amenaza de muerte"
VIDEO_CHANNEL = "Panorama"

RUNS = (
    {
        "key": "texto_qwen",
        "mode": "qwen",
        "input": CHUNK_TEXT,
        "input_type": "text",
        "max_chunks": 1,
        "output": "captura_operacion_texto_qwen.png",
    },
    {
        "key": "texto_consenso",
        "mode": "consensus",
        "input": CHUNK_TEXT,
        "input_type": "text",
        "max_chunks": 1,
        "output": "captura_operacion_texto_consenso.png",
    },
    {
        "key": "youtube_qwen",
        "mode": "qwen",
        "input": VIDEO_URL,
        "input_type": "youtube",
        "max_chunks": 30,
        "output": "captura_operacion_youtube_qwen.png",
    },
    {
        "key": "youtube_consenso",
        "mode": "consensus",
        "input": VIDEO_URL,
        "input_type": "youtube",
        "max_chunks": 30,
        "output": "captura_operacion_youtube_consenso.png",
    },
)


def api_json(path: str) -> dict:
    with urlopen(f"{SERVER_URL}{path}", timeout=30) as response:
        return json.load(response)


def wait_until_finished(driver: webdriver.Chrome) -> str:
    def finished(browser: webdriver.Chrome) -> bool:
        status = browser.find_element(By.ID, "status")
        return status.text.strip().startswith("Análisis terminado") or (
            status.get_attribute("class") == "error"
        )

    WebDriverWait(driver, 900, poll_frequency=1).until(finished)
    status = driver.find_element(By.ID, "status").text.strip()
    if not status.startswith("Análisis terminado"):
        raise RuntimeError(f"El servidor devolvió un error: {status}")
    return status


def prepare_textual_capture(driver: webdriver.Chrome, *, consensus: bool) -> None:
    """Conserva formulario, resumen y un chunk relevante en un recorte legible."""
    driver.execute_script(
        """
        const chunks = [...document.querySelectorAll('.chunk')];
        const consensusMode = arguments[0];
        const target = (consensusMode
          ? chunks.find(c => {
              const cards = [...c.querySelectorAll('.model-result')];
              const finalCard = cards[cards.length - 1];
              return finalCard && finalCard.querySelector('.labels .label:not(.safe)');
            })
          : chunks.find(c => c.querySelector('.chunk-head > .badge.alert'))) || chunks[0];
        chunks.forEach(c => { if (c !== target) c.classList.add('capture-hidden'); });
        document.querySelectorAll('.score-details').forEach(d => d.open = true);
        document.body.classList.toggle('capture-consensus', consensusMode);
        const style = document.createElement('style');
        style.id = 'capture-style';
        style.textContent = `
          nav, .hero, details.settings, #video, #help, #stats, footer,
          .review-actions, .review-form, .review-state { display:none !important; }
          body { background:#fff !important; }
          main { max-width:1180px !important; padding:12px 18px 18px !important;
                 font-size:18px !important; }
          .composer { margin-top:0 !important; box-shadow:none !important; }
          .composer textarea { min-height:54px !important; max-height:66px !important;
                               font-size:18px !important; line-height:1.2 !important; }
          .select, .primary { font-size:16px !important; }
          #status { margin:8px 2px 0 !important; font-size:15px !important; }
          .summary { margin:8px 0 !important; padding:9px 12px !important;
                     font-size:17px !important; }
          .chunk { margin:8px 0 0 !important; border-radius:11px !important; }
          .chunk-head, .chunk-text, .model-result { padding:9px 11px !important; }
          .chunk-head { font-size:16px !important; }
          .chunk-text { font-size:17px !important; line-height:1.25 !important;
                        max-height:52px !important; overflow:hidden !important; }
          .model-name { font-size:18px !important; }
          .badge, .label { font-size:14px !important; }
          .muted, .score-details { font-size:15px !important; }
          .score-details { margin-top:5px !important; }
          .score-row { margin:3px 0 !important; }
          body.capture-consensus .chunk { display:grid !important;
                                          grid-template-columns:1fr 1fr !important; }
          body.capture-consensus .chunk-head,
          body.capture-consensus .chunk-text { grid-column:1 / -1 !important; }
          body.capture-consensus .model-result { border-top:1px solid var(--line) !important; }
          body.capture-consensus .model-result:nth-of-type(even) {
            border-left:1px solid var(--line) !important;
          }
          body.capture-consensus .model-result:not(:last-child) .score-details,
          body.capture-consensus .model-result:not(:last-child) .muted {
            display:none !important;
          }
          body.capture-consensus .composer,
          body.capture-consensus #status { display:none !important; }
          body.capture-consensus .chunk-text {
            max-height:34px !important;
            white-space:nowrap !important;
            text-overflow:ellipsis !important;
          }
          body .chunk.capture-hidden { display:none !important; }
        `;
        document.head.appendChild(style);
        window.scrollTo(0, 0);
        """,
        consensus,
    )


def capture_main(driver: webdriver.Chrome, output_path: Path) -> None:
    rect = driver.execute_script(
        """
        const r = document.querySelector('main').getBoundingClientRect();
        return {x:r.left + scrollX, y:r.top + scrollY,
                width:r.width, height:document.querySelector('main').scrollHeight};
        """
    )
    result = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {
            "format": "png",
            "captureBeyondViewport": True,
            "clip": {
                "x": max(0, rect["x"]),
                "y": max(0, rect["y"]),
                "width": rect["width"],
                "height": rect["height"],
                "scale": 1,
            },
        },
    )
    output_path.write_bytes(base64.b64decode(result["data"]))


def execute_run(driver: webdriver.Chrome, config: dict) -> dict:
    driver.get(SERVER_URL)
    WebDriverWait(driver, 30).until(
        lambda browser: browser.find_element(By.ID, "run").is_enabled()
    )
    Select(driver.find_element(By.ID, "mode")).select_by_value(config["mode"])
    driver.execute_script("document.querySelector('details.settings').open = true;")
    Select(driver.find_element(By.ID, "inputType")).select_by_value(config["input_type"])
    max_chunks = driver.find_element(By.ID, "maxChunks")
    max_chunks.clear()
    max_chunks.send_keys(str(config["max_chunks"]))
    driver.find_element(By.ID, "input").send_keys(config["input"])
    driver.find_element(By.ID, "run").click()
    status = wait_until_finished(driver)

    prepare_textual_capture(driver, consensus=config["mode"] == "consensus")
    output_path = FIGURES / config["output"]
    capture_main(driver, output_path)

    cards = driver.find_elements(By.CSS_SELECTOR, ".chunk:not(.capture-hidden) .model-result")
    return {
        "key": config["key"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": config["mode"],
        "input_type": config["input_type"],
        "max_chunks": config["max_chunks"],
        "input": config["input"] if config["input_type"] == "youtube" else None,
        "source_chunk_id": CHUNK_ID if config["input_type"] == "text" else None,
        "source_split": CHUNK_SPLIT if config["input_type"] == "text" else None,
        "status_visible": status,
        "summary_visible": driver.find_element(By.ID, "summary").text.strip(),
        "results_visible": [card.text.strip() for card in cards],
        "screenshot": str(output_path.relative_to(ROOT)),
        "screenshot_pixels": {
            "width": output_path.stat().st_size and driver.execute_script(
                "return Math.round(document.querySelector('main').getBoundingClientRect().width)"
            ),
            "height": driver.execute_script(
                "return Math.round(document.querySelector('main').scrollHeight)"
            ),
        },
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    health = api_json("/api/health")
    stats_before = api_json("/api/stats")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1360,1050")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")

    driver = webdriver.Chrome(options=options)
    captures = []
    try:
        for config in RUNS:
            print(f"Ejecutando {config['key']}...", flush=True)
            captures.append(execute_run(driver, config))
    finally:
        driver.quit()

    evidence = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_url": SERVER_URL,
        "server_health": health,
        "interaction": "cuatro formularios enviados mediante Selenium; predicciones reales del servidor",
        "text_source": {"chunk_id": CHUNK_ID, "split": CHUNK_SPLIT},
        "youtube_source": {
            "video_id": VIDEO_ID,
            "url": VIDEO_URL,
            "title": VIDEO_TITLE,
            "channel": VIDEO_CHANNEL,
        },
        "events_before": stats_before.get("total_events"),
        "events_after": api_json("/api/stats").get("total_events"),
        "captures": captures,
    }
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
