"""Ejecuta el moderador 05 en el navegador y captura su respuesta real.

El servidor debe estar disponible en ``http://127.0.0.1:8765``. La captura no
inyecta resultados en el DOM: completa el formulario, pulsa ``Analizar`` y
espera la respuesta de ``/api/analyze`` que procesa la propia interfaz.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait


ROOT = Path(__file__).resolve().parents[1]
SERVER_URL = "http://127.0.0.1:8765"
OUTPUT_PATH = ROOT / "Documento_final_paper" / "figuras" / "captura_entorno_operacion.png"
EVIDENCE_PATH = OUTPUT_PATH.with_suffix(".evidence.json")

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


def api_json(path: str) -> dict:
    with urlopen(f"{SERVER_URL}{path}", timeout=20) as response:
        return json.load(response)


def main() -> None:
    health = api_json("/api/health")
    stats_before = api_json("/api/stats")

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1684,1049")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(SERVER_URL)
        WebDriverWait(driver, 30).until(
            lambda browser: browser.find_element(By.ID, "run").is_enabled()
        )
        driver.find_element(By.ID, "input").send_keys(CHUNK_TEXT)
        Select(driver.find_element(By.ID, "mode")).select_by_value("qwen")
        driver.find_element(By.ID, "run").click()

        def finished(browser: webdriver.Chrome) -> bool:
            status = browser.find_element(By.ID, "status").text.strip()
            return status.startswith("Análisis terminado") or bool(
                browser.find_element(By.ID, "status").get_attribute("class") == "error"
            )

        WebDriverWait(driver, 600, poll_frequency=1).until(finished)
        status = driver.find_element(By.ID, "status").text.strip()
        if not status.startswith("Análisis terminado"):
            raise RuntimeError(f"El servidor devolvió un error: {status}")

        score_summary = driver.find_element(By.CSS_SELECTOR, ".score-details > summary")
        driver.execute_script("arguments[0].click();", score_summary)
        WebDriverWait(driver, 10).until(
            lambda browser: browser.find_element(By.CSS_SELECTOR, ".score-row").is_displayed()
        )
        # Mantiene visible el formulario y permite incluir el resultado completo
        # en una captura apaisada, igual que al reducir el zoom del navegador.
        driver.execute_script("document.documentElement.style.zoom='90%';")
        driver.execute_script("window.scrollTo(0, 0);")
        driver.save_screenshot(str(OUTPUT_PATH))

        result_card = driver.find_element(By.CSS_SELECTOR, ".model-result")
        evidence = {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "server_url": SERVER_URL,
            "server_health": health,
            "interaction": "formulario completado y enviado mediante Selenium",
            "mode": "qwen",
            "input_type": "text",
            "source_chunk_id": CHUNK_ID,
            "source_split": CHUNK_SPLIT,
            "status_visible": status,
            "summary_visible": driver.find_element(By.ID, "summary").text.strip(),
            "result_visible": result_card.text.strip(),
            "events_before": stats_before.get("total_events"),
            "events_after": api_json("/api/stats").get("total_events"),
            "screenshot": str(OUTPUT_PATH.relative_to(ROOT)),
        }
        EVIDENCE_PATH.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
