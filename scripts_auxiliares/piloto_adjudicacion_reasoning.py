"""Ejecuta el piloto prerregistrado de adjudicación V4-Pro reasoning.

El piloto es ciego a Flash, Pro y humano durante inferencia. Conserva dos
pasadas aisladas, calcula consenso selectivo y nunca integra etiquetas al
dataset de entrenamiento.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import re
import threading
import time

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from sklearn.metrics import hamming_loss, precision_recall_fscore_support


SEED = 26072027
MODEL = "deepseek-v4-pro"
PILOT_SIZE = 200
NEW_SAMPLE_SIZE = 61
MAX_WORKERS = 16
MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 8_000
CONFIDENCE_GATE = 0.80
BUDGET_USD = 5.00
PRICES = {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87}
COARSE_ORDER = [
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ACOSO_GENERO_IDENTIDAD",
    "ACOSO_PERSONAL",
    "AMENAZA_DIRECTA",
    "CONTENIDO_SEXUAL",
]
FLAGS = ["ironia_ambigua", "humor_encubridor", "contexto_necesario"]


def find_root() -> Path:
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "datos" / "processed" / "taxonomia_moderacion.csv").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto")


ROOT = find_root()
ENV_PATH = ROOT / "03_2_etiquetado_llm_api" / ".env"
load_dotenv(ENV_PATH, override=False)
API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
CAMPAIGN_PATH = ROOT / "datos" / "etiquetado" / "humano" / "revision_humana_sospechosos_139.campaign.json"
PROGRESS_PATH = ROOT / "datos" / "etiquetado" / "humano" / "revision_humana_sospechosos_139.progress.json"
NEW_PENDING_PATH = ROOT / "datos" / "ampliacion" / "ampliacion_dano_20260726" / "processed" / "pendientes_revision_humana.jsonl"
NEW_CHUNKS_PATH = ROOT / "datos" / "ampliacion" / "ampliacion_dano_20260726" / "processed" / "chunks_para_etiquetar.jsonl"
OUT_DIR = ROOT / "datos" / "etiquetado" / "reasoning"
RESULTS_DIR = ROOT / "resultados"
METRICS_DIR = RESULTS_DIR / "metricas"
FIGURES_DIR = RESULTS_DIR / "figuras"
REPORT_PATH = RESULTS_DIR / "INFORME_PILOTO_ADJUDICACION_DEEPSEEK_REASONING.md"
PROMPT_OPERACIONAL_PATH = ROOT / "03_2_etiquetado_llm_api" / "prompt_operacional_compacto.md"
TAXONOMY_PATH = ROOT / "datos" / "processed" / "taxonomia_moderacion.csv"
SELECTION_PATH = OUT_DIR / "piloto_reasoning_200_seleccion.jsonl"
HUMAN_SNAPSHOT_PATH = OUT_DIR / "piloto_reasoning_200_human_snapshot.jsonl"
PASS_PATHS = {
    "A": OUT_DIR / "piloto_reasoning_200_pasada_a.jsonl",
    "B": OUT_DIR / "piloto_reasoning_200_pasada_b.jsonl",
}
CONSENSUS_PATH = OUT_DIR / "piloto_reasoning_200_consenso.jsonl"
MANIFEST_PATH = OUT_DIR / "piloto_reasoning_200.manifest.json"
METRICS_PATH = METRICS_DIR / "piloto_reasoning_200_metricas.json"
FIGURE_PATH = FIGURES_DIR / "piloto_reasoning_200_resultados.png"
EXCLUDED_TECHNICAL_PATHS = {
    # La continuación de 128 contiene las primeras 48; no se suman ambas para evitar doble conteo.
    "cap2000_continuacion": OUT_DIR / "piloto_reasoning_200_pasada_a_cap2000_continuacion_128_fallida.jsonl",
    "calibracion_cap8000_adaptador": OUT_DIR / "piloto_reasoning_calibracion8_cap8000.jsonl",
    "calibracion_contrato03_2": OUT_DIR / "piloto_reasoning_calibracion8_cap8000_contrato03_2.jsonl",
    "calibracion_final": OUT_DIR / "piloto_reasoning_calibracion8_final.jsonl",
}
for directory in (OUT_DIR, METRICS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FINE_TO_COARSE = {
    "seguro": "SEGURO",
    "seguro_ironia_marcada": "SEGURO",
    "racismo_etnico_explicito": "RACISMO_DISCRIMINACION",
    "racismo_linguistico": "RACISMO_DISCRIMINACION",
    "clasismo_racial": "RACISMO_DISCRIMINACION",
    "discriminacion_regional": "RACISMO_DISCRIMINACION",
    "racismo_encubierto": "RACISMO_DISCRIMINACION",
    "misoginia_acoso_genero": "ACOSO_GENERO_IDENTIDAD",
    "homofobia_transfobia": "ACOSO_GENERO_IDENTIDAD",
    "acoso_personal": "ACOSO_PERSONAL",
    "amenaza_directa": "AMENAZA_DIRECTA",
    "sexual_explicito": "CONTENIDO_SEXUAL",
    "sexual_cosificacion": "CONTENIDO_SEXUAL",
    "sexual_no_consensual": "CONTENIDO_SEXUAL",
}
FINE_ORDER = list(FINE_TO_COARSE)


VARIANTS = {
    "A": (
        "PASADA A — CRITERIO NORMATIVO. Aplica directamente las definiciones. "
        "No infieras daño sin evidencia textual o contextual suficiente."
    ),
    "B": (
        "PASADA B — CONTRAEVIDENCIA. Antes de decidir, contrasta internamente la mejor "
        "interpretación segura (cita, noticia, crítica, ficción o broma no dañina) con la "
        "mejor interpretación de daño. Devuelve solo la decisión que sobreviva ese contraste."
    ),
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def stable_rank(chunk_id: str) -> str:
    return sha256_bytes(f"{SEED}|{chunk_id}".encode("utf-8"))


def build_new_contexts() -> dict[str, tuple[str, str]]:
    chunks = read_jsonl(NEW_CHUNKS_PATH)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in chunks:
        by_video[str(row["video_id"])].append(row)
    contexts = {}
    for rows in by_video.values():
        rows.sort(key=lambda row: (float(row.get("start_seconds", 0)), str(row["chunk_id"])))
        for index, row in enumerate(rows):
            previous = rows[index - 1]["text"] if index else ""
            following = rows[index + 1]["text"] if index + 1 < len(rows) else ""
            contexts[row["chunk_id"]] = (previous, following)
    return contexts


def prepare_selection() -> tuple[list[dict], list[dict], dict]:
    required = [
        CAMPAIGN_PATH, PROGRESS_PATH, NEW_PENDING_PATH, NEW_CHUNKS_PATH,
        REPORT_PATH, PROMPT_OPERACIONAL_PATH, TAXONOMY_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan entradas:\n- " + "\n- ".join(missing))
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    old_records = campaign["records"]
    if len(old_records) != 139:
        raise ValueError(f"Se esperaban 139 casos originales; hay {len(old_records)}")

    new_pending = read_jsonl(NEW_PENDING_PATH)
    if len(new_pending) != 1_779:
        raise ValueError(f"Se esperaban 1.779 dudas nuevas; hay {len(new_pending)}")
    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    order_index = {label: index for index, label in enumerate(COARSE_ORDER)}
    for row in new_pending:
        labels = tuple(sorted(row["pro_coarse_labels"], key=order_index.__getitem__))
        strata[labels].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: stable_rank(str(row["chunk_id"])))
    chosen_new, cursors = [], {key: 0 for key in strata}
    stratum_order = sorted(strata, key=lambda values: (len(values), values))
    while len(chosen_new) < NEW_SAMPLE_SIZE:
        advanced = False
        for key in stratum_order:
            position = cursors[key]
            if position < len(strata[key]):
                chosen_new.append(strata[key][position])
                cursors[key] += 1
                advanced = True
                if len(chosen_new) == NEW_SAMPLE_SIZE:
                    break
        if not advanced:
            raise RuntimeError("No fue posible completar la muestra estratificada")

    new_contexts = build_new_contexts()
    selected = []
    for row in old_records:
        selected.append({
            "chunk_id": row["chunk_id"], "cohort": "original_139",
            "video_id": row["video_id"], "channel_title": row.get("channel_title", ""),
            "video_title": row.get("video_title", ""), "text": row["text"],
            "previous_text": row.get("previous_text", ""), "next_text": row.get("next_text", ""),
            "selection_rank": len(selected), "selection_seed": SEED,
        })
    for row in chosen_new:
        previous, following = new_contexts.get(row["chunk_id"], ("", ""))
        selected.append({
            "chunk_id": row["chunk_id"], "cohort": "ampliacion_61",
            "video_id": row["video_id"], "channel_title": row.get("channel_title", ""),
            "video_title": row.get("video_title", ""), "text": row["text"],
            "previous_text": previous, "next_text": following,
            "selection_rank": len(selected), "selection_seed": SEED,
            "selection_stratum": "|".join(row["pro_coarse_labels"]),
        })
    if len(selected) != PILOT_SIZE or len({row["chunk_id"] for row in selected}) != PILOT_SIZE:
        raise AssertionError("La selección no tiene 200 IDs únicos")

    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    human_snapshot = [
        {
            "chunk_id": row["chunk_id"], "coarse_labels": row["coarse_labels"],
            "flags": row.get("flags", []), "annotator_id": row.get("annotator_id", ""),
            "saved_at": row.get("saved_at", ""), "progress_revision": progress.get("revision"),
        }
        for row in progress.get("annotations", []) if row.get("status") == "completed"
    ]
    human_ids = {row["chunk_id"] for row in human_snapshot}
    if len(human_snapshot) != 23 or not human_ids <= {row["chunk_id"] for row in old_records}:
        raise ValueError(
            f"El prerregistro fijó 23 referencias humanas, pero al preparar hay {len(human_snapshot)}"
        )
    write_jsonl_atomic(SELECTION_PATH, selected)
    write_jsonl_atomic(HUMAN_SNAPSHOT_PATH, human_snapshot)
    selection_metadata = {
        "created_at": now_iso(), "seed": SEED, "rows": len(selected),
        "cohort_counts": {
            str(key): int(value)
            for key, value in pd.Series([row["cohort"] for row in selected]).value_counts().items()
        },
        "new_strata_available": len(strata),
        "new_strata_selected": len({row.get("selection_stratum") for row in selected if row["cohort"] == "ampliacion_61"}),
        "human_snapshot_rows": len(human_snapshot),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "human_snapshot_sha256": sha256_file(HUMAN_SNAPSHOT_PATH),
    }
    return selected, human_snapshot, selection_metadata


def system_prompt() -> str:
    authority = PROMPT_OPERACIONAL_PATH.read_text(encoding="utf-8")
    taxonomy = TAXONOMY_PATH.read_text(encoding="utf-8")
    projection = "\n".join(f"- {fine} -> {coarse}" for fine, coarse in FINE_TO_COARSE.items())
    return f"""Eres el adjudicador de mayor capacidad de este proyecto.
Las siguientes fuentes son la autoridad normativa completa. No uses una taxonomía externa.

=== PROMPT OPERACIONAL COMPACTO ===
{authority}

=== TAXONOMÍA CSV ===
{taxonomy}

=== ADAPTACIÓN TÉCNICA DEL PILOTO ===
- Analiza con las etiquetas finas del prompt operativo.
- Devuelve exactamente el contrato operativo original: `chunk_id`, `labels`, `flags`, `needs_review`, `notes`, `score_confianza` y `justificacion`.
- El programa proyectará `labels` a categorías gruesas mediante este mapa versionado:
{projection}
- No devuelvas `fine_labels`, `coarse_labels`, `evidence_excerpt`, `confidence`, `justification` ni campos adicionales.
- SEGURO es incompatible con cualquier etiqueta fina de daño.
- `contexto_necesario` implica `needs_review=true`.
- La confianza es una autoevaluación entre 0 y 1, no una probabilidad calibrada.
- Devuelve exclusivamente un objeto JSON con la clave `annotation` y no reveles tu cadena de razonamiento.
"""


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "annotation": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "chunk_id": {"type": "string"},
                "labels": {
                    "type": "array", "minItems": 1, "uniqueItems": True,
                    "items": {"type": "string", "enum": FINE_ORDER},
                },
                "flags": {
                    "type": "array", "uniqueItems": True,
                    "items": {"type": "string", "enum": FLAGS},
                },
                "needs_review": {"type": "boolean"},
                "notes": {"type": "string", "maxLength": 160},
                "score_confianza": {"type": "number", "minimum": 0, "maximum": 1},
                "justificacion": {"type": "string", "minLength": 1, "maxLength": 700},
            },
            "required": [
                "chunk_id", "labels", "flags", "needs_review", "notes",
                "score_confianza", "justificacion",
            ],
        }
    },
    "required": ["annotation"],
}


def api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Falta DEEPSEEK_API_KEY")
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


_thread_local = threading.local()


def api_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=0)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def usage_add(total: dict[str, int], usage: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        total[key] += int(usage.get(key, 0) or 0)
    details = usage.get("completion_tokens_details") or {}
    total["reasoning_tokens"] += int(details.get("reasoning_tokens", 0) or 0)


def usage_cost(usage: dict) -> float:
    prompt = int(usage.get("prompt_tokens", 0))
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    miss = int(usage.get("prompt_cache_miss_tokens", max(prompt - hit, 0)))
    output = int(usage.get("completion_tokens", 0))
    return (hit * PRICES["cache_hit"] + miss * PRICES["cache_miss"] + output * PRICES["output"]) / 1_000_000


def clean_json(content: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    parsed = json.loads(cleaned)
    operational_fields = {
        "chunk_id", "labels", "flags", "needs_review", "notes",
        "score_confianza", "justificacion",
    }
    if isinstance(parsed, dict) and set(parsed) == operational_fields:
        parsed = {"annotation": parsed}
    return parsed


def semantic_errors(annotation: dict, expected_id: str, central_text: str) -> list[str]:
    errors = []
    fine, flags = annotation.get("labels", []), annotation.get("flags", [])
    if annotation.get("chunk_id") != expected_id:
        errors.append("chunk_id no coincide")
    safe_fine = {"seguro", "seguro_ironia_marcada"} & set(fine)
    if safe_fine and len(fine) > 1:
        errors.append("una etiqueta fina segura es incompatible con daño")
    if "contexto_necesario" in flags and not annotation.get("needs_review"):
        errors.append("contexto_necesario exige needs_review")
    if flags and not annotation.get("needs_review"):
        errors.append("todo flag exige needs_review")
    if float(annotation.get("score_confianza", 0)) < 0.70 and not annotation.get("needs_review"):
        errors.append("score menor de 0.70 exige needs_review")
    if ({"ironia_ambigua", "contexto_necesario"} & set(flags)) and float(annotation.get("score_confianza", 0)) > 0.65:
        errors.append("ironía/contexto limitan score a 0.65")
    if safe_fine and flags:
        errors.append("SEGURO no admite flags")
    return errors


def input_payload(row: dict, variant: str, correction: str = "") -> str:
    payload = {
        "chunk_id": row["chunk_id"],
        "contexto_anterior": row.get("previous_text", ""),
        "chunk_central": row["text"],
        "contexto_posterior": row.get("next_text", ""),
    }
    message = VARIANTS[variant] + "\nAnaliza este único registro y devuelve JSON:\n"
    message += json.dumps(payload, ensure_ascii=False)
    if correction:
        message += "\nLa respuesta anterior fue inválida. Corrige exclusivamente estos problemas: " + correction
    return message


def adjudicate(row: dict, variant: str, system: str) -> dict:
    usage_total: dict[str, int] = defaultdict(int)
    correction = ""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": input_payload(row, variant, correction)},
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            response = api_session().post(
                f"{API_BASE}/chat/completions", headers=api_headers(), json=body,
                timeout=(20, 300),
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            response_json = response.json()
            usage_add(usage_total, response_json.get("usage", {}))
            choice = response_json["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("respuesta truncada por max_tokens")
            parsed = clean_json(choice["message"].get("content") or "")
            if isinstance(parsed, dict) and isinstance(parsed.get("annotation"), dict):
                # Misma normalización no sustantiva aplicada en el flujo 03_2.
                parsed["annotation"]["notes"] = str(
                    parsed["annotation"].get("notes") or ""
                )[:160]
                parsed["annotation"]["justificacion"] = str(
                    parsed["annotation"].get("justificacion") or ""
                )[:700]
            jsonschema.validate(parsed, SCHEMA)
            annotation = parsed["annotation"]
            errors = semantic_errors(annotation, row["chunk_id"], row["text"])
            if errors:
                raise ValueError("; ".join(errors))
            fine_labels = list(annotation.pop("labels"))
            projected_set = {FINE_TO_COARSE[value] for value in fine_labels}
            annotation["fine_labels"] = fine_labels
            annotation["coarse_labels"] = [
                value for value in COARSE_ORDER if value in projected_set
            ]
            return {
                **annotation, "variant": variant, "status": "ok", "model": MODEL,
                "thinking": "enabled", "reasoning_effort": "max", "attempts": attempt,
                "usage": dict(usage_total), "cost_usd": usage_cost(usage_total),
                "adjudicated_at": now_iso(),
            }
        except Exception as exc:
            last_error = str(exc)[:500]
            correction = last_error
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8) + random.random())
    return {
        "chunk_id": row["chunk_id"], "variant": variant, "status": "error",
        "model": MODEL, "thinking": "enabled", "reasoning_effort": "max",
        "attempts": MAX_RETRIES, "error": last_error, "usage": dict(usage_total),
        "cost_usd": usage_cost(usage_total), "adjudicated_at": now_iso(),
    }


def validate_existing(rows: list[dict], selected_ids: set[str], variant: str) -> dict[str, dict]:
    by_id = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if chunk_id not in selected_ids or row.get("variant") != variant or row.get("model") != MODEL:
            raise ValueError(f"Salida existente incompatible: {chunk_id}")
        if chunk_id in by_id:
            raise ValueError(f"ID duplicado en salida {variant}: {chunk_id}")
        by_id[chunk_id] = row
    return by_id


def run_pass(selected: list[dict], variant: str, system: str, prior_cost: float) -> tuple[list[dict], dict]:
    order = [row["chunk_id"] for row in selected]
    selected_by_id = {row["chunk_id"]: row for row in selected}
    existing = validate_existing(read_jsonl(PASS_PATHS[variant]), set(order), variant)
    # Los errores técnicos se vuelven a intentar al reanudar; los éxitos nunca se refacturan.
    complete = {key: value for key, value in existing.items() if value.get("status") == "ok"}
    pending_ids = [chunk_id for chunk_id in order if chunk_id not in complete]
    print(f"Pasada {variant}: existentes_ok={len(complete)}, pendientes={len(pending_ids)}", flush=True)
    total_cost = prior_cost + sum(float(row.get("cost_usd", 0)) for row in complete.values())
    processed = 0
    # Olas pequeñas permiten aplicar el techo de costo antes de enviar todo lo restante.
    for start in range(0, len(pending_ids), MAX_WORKERS):
        wave = pending_ids[start : start + MAX_WORKERS]
        if total_cost >= BUDGET_USD:
            raise RuntimeError(f"Presupuesto detenido antes de la siguiente ola: USD {total_cost:.4f}")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(adjudicate, selected_by_id[chunk_id], variant, system): chunk_id
                for chunk_id in wave
            }
            for future in as_completed(futures):
                chunk_id = futures[future]
                result = future.result()
                complete[chunk_id] = result
                total_cost += float(result.get("cost_usd", 0))
                processed += 1
        ordered_current = [complete[chunk_id] for chunk_id in order if chunk_id in complete]
        write_jsonl_atomic(PASS_PATHS[variant], ordered_current)
        print(
            f"Pasada {variant}: {len(complete)}/{len(order)}; costo acumulado piloto USD {total_cost:.4f}",
            flush=True,
        )
    final_rows = [complete[chunk_id] for chunk_id in order]
    write_jsonl_atomic(PASS_PATHS[variant], final_rows)
    usage = defaultdict(int)
    for row in final_rows:
        usage_add(usage, row.get("usage", {}))
    summary = {
        "rows": len(final_rows), "ok": sum(row.get("status") == "ok" for row in final_rows),
        "errors": sum(row.get("status") != "ok" for row in final_rows),
        "usage": dict(usage), "cost_usd": sum(float(row.get("cost_usd", 0)) for row in final_rows),
        "sha256": sha256_file(PASS_PATHS[variant]),
    }
    return final_rows, summary


def ordered_labels(values: list[str]) -> list[str]:
    present = set(values)
    return [label for label in COARSE_ORDER if label in present]


def build_consensus(selected: list[dict], pass_a: list[dict], pass_b: list[dict]) -> list[dict]:
    a_by_id = {row["chunk_id"]: row for row in pass_a}
    b_by_id = {row["chunk_id"]: row for row in pass_b}
    rows = []
    for source in selected:
        chunk_id = source["chunk_id"]
        a, b = a_by_id[chunk_id], b_by_id[chunk_id]
        both_ok = a.get("status") == "ok" and b.get("status") == "ok"
        exact = both_ok and set(a["coarse_labels"]) == set(b["coarse_labels"])
        accepted = bool(
            exact and not a["needs_review"] and not b["needs_review"]
            and "contexto_necesario" not in a["flags"]
            and "contexto_necesario" not in b["flags"]
            and float(a["score_confianza"]) >= CONFIDENCE_GATE
            and float(b["score_confianza"]) >= CONFIDENCE_GATE
        )
        labels_a = ordered_labels(a.get("coarse_labels", []))
        labels_b = ordered_labels(b.get("coarse_labels", []))
        intersection, union = set(labels_a) & set(labels_b), set(labels_a) | set(labels_b)
        rows.append({
            "chunk_id": chunk_id, "cohort": source["cohort"],
            "accepted_reasoning_consensus": accepted, "requires_human": not accepted,
            "consensus_coarse_labels": labels_a if accepted else [],
            "pass_a_coarse_labels": labels_a, "pass_b_coarse_labels": labels_b,
            "pass_a_fine_labels": a.get("fine_labels", []),
            "pass_b_fine_labels": b.get("fine_labels", []),
            "pass_a_flags": a.get("flags", []), "pass_b_flags": b.get("flags", []),
            "pass_a_needs_review": a.get("needs_review", True),
            "pass_b_needs_review": b.get("needs_review", True),
            "pass_a_confidence": a.get("score_confianza"),
            "pass_b_confidence": b.get("score_confianza"),
            "interpass_exact_set": exact,
            "interpass_jaccard": len(intersection) / len(union) if union else 0.0,
            "decision_source": "deepseek-v4-pro-thinking-max-dual-consensus" if accepted else "human_required",
            "training_eligible": False, "pilot_only": True,
        })
    write_jsonl_atomic(CONSENSUS_PATH, rows)
    return rows


def binary_matrix(rows: list[list[str]]) -> np.ndarray:
    return np.asarray([[int(label in values) for label in COARSE_ORDER] for values in rows], dtype=int)


def comparison_metrics(reference: list[list[str]], prediction: list[list[str]]) -> dict:
    if not reference:
        return {"n": 0}
    y_true, y_pred = binary_matrix(reference), binary_matrix(prediction)
    exact = np.all(y_true == y_pred, axis=1)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "n": len(reference), "exact_set_accuracy": float(exact.mean()),
        "exact_set_correct": int(exact.sum()), "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "precision_micro": float(p_micro), "recall_micro": float(r_micro), "f1_micro": float(f_micro),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
    }


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def excluded_technical_runs() -> dict:
    output = {}
    for name, path in EXCLUDED_TECHNICAL_PATHS.items():
        rows = read_jsonl(path)
        if not rows:
            raise FileNotFoundError(f"Falta la corrida técnica excluida: {path}")
        output[name] = {
            "path": str(path.relative_to(ROOT)), "rows": len(rows),
            "cost_usd": sum(float(row.get("cost_usd", 0)) for row in rows),
            "sha256": sha256_file(path),
        }
    return output


def calculate_metrics(
    selected: list[dict], human: list[dict], pass_a: list[dict], pass_b: list[dict],
    consensus: list[dict], pass_summaries: dict, selection_metadata: dict,
) -> dict:
    selected_by_id = {row["chunk_id"]: row for row in selected}
    a_by_id, b_by_id = ({row["chunk_id"]: row for row in values} for values in (pass_a, pass_b))
    c_by_id = {row["chunk_id"]: row for row in consensus}
    human_by_id = {row["chunk_id"]: row for row in human}
    human_ids = [row["chunk_id"] for row in human]
    references = [human_by_id[chunk_id]["coarse_labels"] for chunk_id in human_ids]
    pass_a_metrics = comparison_metrics(
        references, [a_by_id[x].get("coarse_labels", []) for x in human_ids]
    )
    pass_b_metrics = comparison_metrics(
        references, [b_by_id[x].get("coarse_labels", []) for x in human_ids]
    )
    accepted_human_ids = [x for x in human_ids if c_by_id[x]["accepted_reasoning_consensus"]]
    accepted_metrics = comparison_metrics(
        [human_by_id[x]["coarse_labels"] for x in accepted_human_ids],
        [c_by_id[x]["consensus_coarse_labels"] for x in accepted_human_ids],
    )
    accepted_metrics["coverage_of_human_snapshot"] = len(accepted_human_ids) / len(human_ids)
    accepted_metrics["wilson_95_exact_accuracy"] = wilson(
        accepted_metrics.get("exact_set_correct", 0), accepted_metrics.get("n", 0)
    )
    accepted_human_damage_ids = [
        chunk_id for chunk_id in accepted_human_ids
        if set(human_by_id[chunk_id]["coarse_labels"]) != {"SEGURO"}
    ]
    false_safe_ids = [
        chunk_id for chunk_id in accepted_human_damage_ids
        if set(c_by_id[chunk_id]["consensus_coarse_labels"]) == {"SEGURO"}
    ]
    accepted_metrics["human_damage_n"] = len(accepted_human_damage_ids)
    accepted_metrics["false_safe_count"] = len(false_safe_ids)
    accepted_metrics["false_safe_rate_on_human_damage"] = (
        len(false_safe_ids) / len(accepted_human_damage_ids) if accepted_human_damage_ids else None
    )
    cohort = {}
    for cohort_name in ("original_139", "ampliacion_61"):
        cohort_rows = [row for row in consensus if row["cohort"] == cohort_name]
        accepted = sum(row["accepted_reasoning_consensus"] for row in cohort_rows)
        cohort[cohort_name] = {
            "n": len(cohort_rows), "accepted": accepted,
            "requires_human": len(cohort_rows) - accepted,
            "acceptance_rate": accepted / len(cohort_rows),
        }
    interpass_exact = sum(row["interpass_exact_set"] for row in consensus)
    total_accepted = sum(row["accepted_reasoning_consensus"] for row in consensus)
    success_rate = min(pass_summaries["A"]["ok"], pass_summaries["B"]["ok"]) / PILOT_SIZE
    gates = {
        "technical_success_ge_98pct": success_rate >= 0.98,
        "acceptance_ge_50pct": total_accepted / PILOT_SIZE >= 0.50,
        "accepted_human_exact_point_ge_85pct": (
            accepted_metrics.get("n", 0) > 0 and accepted_metrics.get("exact_set_accuracy", 0) >= 0.85
        ),
    }
    technical_runs = excluded_technical_runs()
    technical_overhead = sum(row["cost_usd"] for row in technical_runs.values())
    analytic_cost = pass_summaries["A"]["cost_usd"] + pass_summaries["B"]["cost_usd"]
    metrics = {
        "created_at": now_iso(), "pilot_rows": PILOT_SIZE, "selection": selection_metadata,
        "pass_summaries": pass_summaries,
        "cost_usd_analytic_run": analytic_cost,
        "cost_usd_excluded_technical_overhead": technical_overhead,
        "cost_usd_total_process": analytic_cost + technical_overhead,
        "excluded_technical_runs": technical_runs,
        "technical_success_rate": success_rate,
        "interpass_exact_set": interpass_exact,
        "interpass_exact_set_rate": interpass_exact / PILOT_SIZE,
        "interpass_jaccard_mean": float(np.mean([row["interpass_jaccard"] for row in consensus])),
        "accepted": total_accepted, "requires_human": PILOT_SIZE - total_accepted,
        "acceptance_rate": total_accepted / PILOT_SIZE,
        "cohorts": cohort,
        "human_snapshot_n": len(human),
        "human_comparison": {
            "pass_a": pass_a_metrics, "pass_b": pass_b_metrics,
            "accepted_consensus": accepted_metrics,
        },
        "preregistered_gates": gates,
        "operationally_promising": all(gates.values()),
        "accepted_label_counts": {
            label: sum(label in row["consensus_coarse_labels"] for row in consensus)
            for label in COARSE_ORDER
        },
        "selected_ids_sha256": sha256_bytes(
            "\n".join(row["chunk_id"] for row in selected).encode("utf-8")
        ),
    }
    write_json_atomic(METRICS_PATH, metrics)
    return metrics


def create_figure(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cohort_names = ["original_139", "ampliacion_61"]
    display = ["Original 139", "Ampliación 61"]
    accepted = [metrics["cohorts"][name]["accepted"] for name in cohort_names]
    pending = [metrics["cohorts"][name]["requires_human"] for name in cohort_names]
    x = np.arange(2)
    axes[0].bar(x, accepted, label="Consenso aceptado", color="#54A24B")
    axes[0].bar(x, pending, bottom=accepted, label="Conserva revisión humana", color="#E45756")
    axes[0].set_xticks(x, display)
    axes[0].set_ylabel("Chunks")
    axes[0].set_title("Resultado selectivo por cohorte")
    axes[0].legend()
    for index in range(2):
        axes[0].text(index, accepted[index] / 2, str(accepted[index]), ha="center", va="center")
        axes[0].text(index, accepted[index] + pending[index] / 2, str(pending[index]), ha="center", va="center")

    comparisons = metrics["human_comparison"]
    labels = ["Pasada A", "Pasada B", "Consenso\naceptado"]
    values = [
        comparisons["pass_a"].get("exact_set_accuracy", 0),
        comparisons["pass_b"].get("exact_set_accuracy", 0),
        comparisons["accepted_consensus"].get("exact_set_accuracy", 0),
    ]
    ns = [
        comparisons["pass_a"].get("n", 0), comparisons["pass_b"].get("n", 0),
        comparisons["accepted_consensus"].get("n", 0),
    ]
    bars = axes[1].bar(np.arange(3), values, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Exactitud de conjunto")
    axes[1].set_title("Comparación con referencia humana congelada")
    for bar, value, n in zip(bars, values, ns):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}\n(n={n})", ha="center")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def update_report(metrics: dict, manifest: dict) -> None:
    human = metrics["human_comparison"]
    accepted_human = human["accepted_consensus"]
    wilson_ci = accepted_human.get("wilson_95_exact_accuracy")
    ci_text = (
        f"[{wilson_ci[0]:.1%}, {wilson_ci[1]:.1%}]" if wilson_ci else "no estimable"
    )
    gates = metrics["preregistered_gates"]
    lines = [
        "## 9. Ejecución y resultados",
        "",
        f"Ejecución cerrada: {metrics['created_at']}  ",
        f"Modelo: `{MODEL}` con `thinking=enabled` y `reasoning_effort=max`  ",
        f"Costo de la corrida analítica: **USD {metrics['cost_usd_analytic_run']:.4f}**  ",
        f"Costo total del proceso, incluidas calibraciones y corrida técnica excluida: **USD {metrics['cost_usd_total_process']:.4f}**",
        "",
        "### 9.1 Integridad técnica y consumo",
        "",
        "| Pasada | Éxitos | Errores | Entrada | Salida | Reasoning | Costo USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("A", "B"):
        summary = metrics["pass_summaries"][variant]
        usage = summary["usage"]
        lines.append(
            f"| {variant} | {summary['ok']:,} | {summary['errors']:,} | "
            f"{usage.get('prompt_tokens', 0):,} | {usage.get('completion_tokens', 0):,} | "
            f"{usage.get('reasoning_tokens', 0):,} | {summary['cost_usd']:.4f} |"
        )
    lines.extend([
        "",
        "La API no desglosó `reasoning_tokens` en estas respuestas; el cero de la tabla significa «no reportado por separado», no ausencia de razonamiento. Los tokens de razonamiento están incluidos en `completion_tokens` y en el costo.",
        "",
        "### 9.2 Consenso y reducción potencial de revisión humana",
        "",
        f"Las dos pasadas coincidieron exactamente en {metrics['interpass_exact_set']}/{PILOT_SIZE} "
        f"casos ({metrics['interpass_exact_set_rate']:.1%}); Jaccard medio multietiqueta "
        f"{metrics['interpass_jaccard_mean']:.3f}. La regla prerregistrada aceptó "
        f"**{metrics['accepted']}/{PILOT_SIZE} ({metrics['acceptance_rate']:.1%})** y conservó "
        f"{metrics['requires_human']} para humano.",
        "",
        "| Cohorte | N | Aceptados | Conservan humano | Tasa de aceptación |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in (("original_139", "Original 139"), ("ampliacion_61", "Ampliación 61")):
        row = metrics["cohorts"][name]
        lines.append(
            f"| {label} | {row['n']} | {row['accepted']} | {row['requires_human']} | {row['acceptance_rate']:.1%} |"
        )
    lines.extend([
        "",
        "![Resultados del piloto reasoning](figuras/piloto_reasoning_200_resultados.png)",
        "",
        "### 9.3 Comparación ciega con las 23 decisiones humanas congeladas",
        "",
        "| Evaluación | N | Exactitud de conjunto | Hamming loss | F1 micro | F1 macro |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Pasada A | {human['pass_a']['n']} | {human['pass_a']['exact_set_accuracy']:.1%} | "
        f"{human['pass_a']['hamming_loss']:.3f} | {human['pass_a']['f1_micro']:.3f} | {human['pass_a']['f1_macro']:.3f} |",
        f"| Pasada B | {human['pass_b']['n']} | {human['pass_b']['exact_set_accuracy']:.1%} | "
        f"{human['pass_b']['hamming_loss']:.3f} | {human['pass_b']['f1_micro']:.3f} | {human['pass_b']['f1_macro']:.3f} |",
        f"| Consenso aceptado | {accepted_human.get('n', 0)} | {accepted_human.get('exact_set_accuracy', 0):.1%} | "
        f"{accepted_human.get('hamming_loss', 0):.3f} | {accepted_human.get('f1_micro', 0):.3f} | "
        f"{accepted_human.get('f1_macro', 0):.3f} |",
        "",
        f"El consenso cubrió {accepted_human.get('coverage_of_human_snapshot', 0):.1%} de la referencia humana. "
        f"Su IC Wilson 95% para exactitud exacta es {ci_text}; el intervalo amplio refleja que solo hay 23 casos humanos.",
        "",
        f"Indicador de seguridad: entre {accepted_human.get('human_damage_n', 0)} casos aceptados que humano marcó como daño, "
        f"el consenso reasoning clasificó {accepted_human.get('false_safe_count', 0)} como `SEGURO` "
        f"({accepted_human.get('false_safe_rate_on_human_damage', 0):.1%}). Este patrón de falsos seguros impide usar el consenso para absolver automáticamente casos.",
        "",
        "### 9.4 Criterios prerregistrados y conclusión",
        "",
        f"- Éxito técnico ≥98%: **{'sí' if gates['technical_success_ge_98pct'] else 'no'}**.",
        f"- Aceptación ≥50%: **{'sí' if gates['acceptance_ge_50pct'] else 'no'}**.",
        f"- Exactitud puntual ≥85% contra humano entre aceptados: **{'sí' if gates['accepted_human_exact_point_ge_85pct'] else 'no'}**.",
        "",
        f"Conclusión prerregistrada: **{'piloto operacionalmente prometedor' if metrics['operationally_promising'] else 'el piloto no supera todos los criterios operativos'}**. "
        "Aunque los cumpla, las salidas siguen siendo pseudoetiquetas LLM y no reemplazan una validación humana académica. "
        "No se integró ningún resultado al entrenamiento y no se autoriza escalar este procedimiento a los 1.895 pendientes.",
        "",
        "La referencia humana de 23 casos corresponde a los casos completados disponibles, no a una muestra aleatoria nueva. Por ello la comparación es un control de seguridad del piloto y no una estimación poblacional definitiva; el mal desempeño observado, sin embargo, es suficiente para rechazar la automatización propuesta.",
        "",
        "### 9.5 Trazabilidad de salida",
        "",
        f"- Selección SHA-256: `{manifest['selection']['selection_sha256']}`.",
        f"- Prompt operacional SHA-256: `{manifest['prompt_operacional_sha256']}`.",
        f"- Taxonomía CSV SHA-256: `{manifest['taxonomy_sha256']}`.",
        f"- Referencia humana congelada SHA-256: `{manifest['selection']['human_snapshot_sha256']}`.",
        f"- Pasada A SHA-256: `{manifest['passes']['A']['sha256']}`.",
        f"- Pasada B SHA-256: `{manifest['passes']['B']['sha256']}`.",
        f"- Consenso SHA-256: `{manifest['consensus_sha256']}`.",
        f"- Métricas SHA-256: `{manifest['metrics_sha256']}`.",
    ])
    content = REPORT_PATH.read_text(encoding="utf-8")
    head = content.split("## 9. Ejecución y resultados", 1)[0].rstrip()
    head = re.sub(r"(?:\n---\s*)+$", "", head).rstrip()
    REPORT_PATH.write_text(head + "\n\n---\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def preflight() -> None:
    response = requests.get(f"{API_BASE}/models", headers=api_headers(), timeout=30)
    response.raise_for_status()
    models = [row["id"] for row in response.json().get("data", [])]
    if MODEL not in models:
        raise RuntimeError(f"{MODEL} no está visible: {models}")
    if not REPORT_PATH.read_text(encoding="utf-8").startswith("# Piloto prerregistrado"):
        raise RuntimeError("El informe prerregistrado no existe o cambió")
    print(f"Preflight correcto; modelos visibles={models}", flush=True)


def summarize_existing_pass(rows: list[dict], variant: str) -> dict:
    if len(rows) != PILOT_SIZE or len({row.get("chunk_id") for row in rows}) != PILOT_SIZE:
        raise ValueError(f"La pasada {variant} no contiene exactamente 200 IDs únicos")
    usage: dict[str, int] = defaultdict(int)
    for row in rows:
        usage_add(usage, row.get("usage", {}))
    return {
        "rows": len(rows), "ok": sum(row.get("status") == "ok" for row in rows),
        "errors": sum(row.get("status") != "ok" for row in rows),
        "usage": dict(usage),
        "cost_usd": sum(float(row.get("cost_usd", 0)) for row in rows),
        "sha256": sha256_file(PASS_PATHS[variant]),
    }


def analyze_existing() -> dict:
    selected = read_jsonl(SELECTION_PATH)
    human_snapshot = read_jsonl(HUMAN_SNAPSHOT_PATH)
    pass_a, pass_b = read_jsonl(PASS_PATHS["A"]), read_jsonl(PASS_PATHS["B"])
    if len(selected) != PILOT_SIZE or len(human_snapshot) != 23:
        raise ValueError("La selección o la referencia humana congelada cambió")
    summary_a = summarize_existing_pass(pass_a, "A")
    summary_b = summarize_existing_pass(pass_b, "B")
    selection_metadata = {
        "created_at": now_iso(), "seed": SEED, "rows": len(selected),
        "cohort_counts": {
            str(key): int(value)
            for key, value in pd.Series([row["cohort"] for row in selected]).value_counts().items()
        },
        "new_strata_selected": len({
            row.get("selection_stratum") for row in selected if row["cohort"] == "ampliacion_61"
        }),
        "human_snapshot_rows": len(human_snapshot),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "human_snapshot_sha256": sha256_file(HUMAN_SNAPSHOT_PATH),
    }
    consensus = build_consensus(selected, pass_a, pass_b)
    pass_summaries = {"A": summary_a, "B": summary_b}
    metrics = calculate_metrics(
        selected, human_snapshot, pass_a, pass_b, consensus, pass_summaries, selection_metadata
    )
    create_figure(metrics)
    system = system_prompt()
    prompt_sha = sha256_bytes(
        (system + "\n" + VARIANTS["A"] + "\n" + VARIANTS["B"]).encode("utf-8")
    )
    manifest = {
        "schema_version": "1.0", "created_at": now_iso(), "model": MODEL,
        "thinking": "enabled", "reasoning_effort": "max", "max_output_tokens": MAX_OUTPUT_TOKENS,
        "confidence_gate": CONFIDENCE_GATE, "budget_usd": BUDGET_USD,
        "prompt_sha256": prompt_sha, "selection": selection_metadata,
        "prompt_operacional_path": str(PROMPT_OPERACIONAL_PATH.relative_to(ROOT)),
        "prompt_operacional_sha256": sha256_file(PROMPT_OPERACIONAL_PATH),
        "taxonomy_path": str(TAXONOMY_PATH.relative_to(ROOT)),
        "taxonomy_sha256": sha256_file(TAXONOMY_PATH),
        "passes": pass_summaries, "consensus_sha256": sha256_file(CONSENSUS_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH), "figure_sha256": sha256_file(FIGURE_PATH),
        "excluded_technical_runs": metrics["excluded_technical_runs"],
        "training_modified": False, "human_campaign_modified": False,
        "analysis_only_after_completed_passes": True,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    update_report(metrics, manifest)
    print(json.dumps({
        "accepted": metrics["accepted"], "requires_human": metrics["requires_human"],
        "acceptance_rate": metrics["acceptance_rate"],
        "human_comparison": metrics["human_comparison"],
        "cost_usd_analytic_run": metrics["cost_usd_analytic_run"],
        "cost_usd_total_process": metrics["cost_usd_total_process"],
        "operationally_promising": metrics["operationally_promising"],
    }, ensure_ascii=False, indent=2))
    return metrics


def run() -> dict:
    preflight()
    selected, human_snapshot, selection_metadata = prepare_selection()
    system = system_prompt()
    prompt_sha = sha256_bytes(
        (system + "\n" + VARIANTS["A"] + "\n" + VARIANTS["B"]).encode("utf-8")
    )
    pass_a, summary_a = run_pass(selected, "A", system, prior_cost=0.0)
    pass_b, summary_b = run_pass(selected, "B", system, prior_cost=summary_a["cost_usd"])
    actual_cost = summary_a["cost_usd"] + summary_b["cost_usd"]
    if actual_cost > BUDGET_USD:
        raise RuntimeError(f"El costo final excedió el presupuesto: USD {actual_cost:.4f}")
    consensus = build_consensus(selected, pass_a, pass_b)
    pass_summaries = {"A": summary_a, "B": summary_b}
    metrics = calculate_metrics(
        selected, human_snapshot, pass_a, pass_b, consensus, pass_summaries, selection_metadata
    )
    create_figure(metrics)
    manifest = {
        "schema_version": "1.0", "created_at": now_iso(), "model": MODEL,
        "thinking": "enabled", "reasoning_effort": "max", "max_output_tokens": MAX_OUTPUT_TOKENS,
        "confidence_gate": CONFIDENCE_GATE, "budget_usd": BUDGET_USD,
        "prompt_sha256": prompt_sha, "selection": selection_metadata,
        "prompt_operacional_path": str(PROMPT_OPERACIONAL_PATH.relative_to(ROOT)),
        "prompt_operacional_sha256": sha256_file(PROMPT_OPERACIONAL_PATH),
        "taxonomy_path": str(TAXONOMY_PATH.relative_to(ROOT)),
        "taxonomy_sha256": sha256_file(TAXONOMY_PATH),
        "passes": pass_summaries, "consensus_sha256": sha256_file(CONSENSUS_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH), "figure_sha256": sha256_file(FIGURE_PATH),
        "training_modified": False, "human_campaign_modified": False,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    update_report(metrics, manifest)
    print(json.dumps({
        "accepted": metrics["accepted"], "requires_human": metrics["requires_human"],
        "acceptance_rate": metrics["acceptance_rate"],
        "human_comparison": metrics["human_comparison"],
        "cost_usd_analytic_run": metrics["cost_usd_analytic_run"],
        "cost_usd_total_process": metrics["cost_usd_total_process"],
        "operationally_promising": metrics["operationally_promising"],
    }, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run", "analyze"], default="run")
    args = parser.parse_args()
    if args.stage == "prepare":
        preflight()
        selected, human, metadata = prepare_selection()
        print(json.dumps({"selected": len(selected), "human_snapshot": len(human), **metadata}, ensure_ascii=False, indent=2))
    elif args.stage == "run":
        run()
    else:
        analyze_existing()


if __name__ == "__main__":
    main()
