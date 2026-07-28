"""Etiquetado incremental Flash→Pro del lote de ampliación dirigida.

La salida se mantiene aislada del corpus original. Flash etiqueta todos los
chunks; Pro revisa cualquier daño, alerta o score < 0.90, además de un control
aleatorio reproducible del 10% de los seguros confiables. El proceso es
reanudable, valida taxonomía y registra tokens, costo, hashes y selección.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time

import jsonschema
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter


BATCH_ID = os.getenv("AMPLIACION_BATCH_ID", "ampliacion_dano_20260726").strip()
SEED = int(os.getenv("AMPLIACION_SEED", "26072026"))
BATCH_SIZE = int(os.getenv("AMPLIACION_LLM_BATCH_SIZE", "5"))
MAX_WORKERS = int(os.getenv("AMPLIACION_LLM_WORKERS", "24"))
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 180
MAX_TOKENS_PER_RECORD = 512
MAX_TOKENS_OVERHEAD = 64
CONFIDENCE_THRESHOLD = 0.90
SAFE_CONTROL_RATE = 0.10
SAFE_LABELS = {"seguro", "seguro_ironia_marcada"}
MODEL_PRICING_USD_PER_MILLION = {
    "deepseek-v4-flash": {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
}


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "datos" / "processed" / "taxonomia_moderacion.csv").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto.")


ROOT = find_project_root()
BATCH_DIR = ROOT / "datos" / "ampliacion" / BATCH_ID
CHUNKS_PATH = BATCH_DIR / "processed" / "chunks_para_etiquetar.jsonl"
OUTPUT_DIR = BATCH_DIR / "etiquetado"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FLASH_PATH = OUTPUT_DIR / "deepseek-v4-flash.jsonl"
PRO_PATH = OUTPUT_DIR / "deepseek-v4-pro_revision.jsonl"
FLASH_MANIFEST_PATH = OUTPUT_DIR / "deepseek-v4-flash.manifest.json"
PRO_MANIFEST_PATH = OUTPUT_DIR / "deepseek-v4-pro_revision.manifest.json"
TAXONOMY_PATH = ROOT / "datos" / "processed" / "taxonomia_moderacion.csv"
SKILL_PATH = ROOT / "modelos" / "skills" / "clasificacion_moderacion_peru.md"
COMPACT_PROMPT_PATH = ROOT / "03_2_etiquetado_llm_api" / "prompt_operacional_compacto.md"
ENV_PATH = ROOT / "03_2_etiquetado_llm_api" / ".env"
load_dotenv(ENV_PATH, override=False)
API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
FLASH_MODEL = os.getenv("DEEPSEEK_PRIMARY_MODEL", "deepseek-v4-flash").strip()
PRO_MODEL = os.getenv("DEEPSEEK_REVIEW_MODEL", "deepseek-v4-pro").strip()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    output = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {line_number}: {exc}") from exc
    return output


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


taxonomy = pd.read_csv(TAXONOMY_PATH).fillna("")
LABEL_ORDER = taxonomy.loc[taxonomy["categoria"] != "FLAG", "label"].tolist()
FLAG_ORDER = taxonomy.loc[taxonomy["categoria"] == "FLAG", "label"].tolist()
ALLOWED_LABELS = set(LABEL_ORDER)
ALLOWED_FLAGS = set(FLAG_ORDER)
DAMAGE_LABELS = ALLOWED_LABELS - SAFE_LABELS
SEMANTIC_FIELDS = {
    "chunk_id", "labels", "flags", "needs_review", "notes",
    "score_confianza", "justificacion",
}
FINAL_FIELDS = SEMANTIC_FIELDS | {
    "annotator_type", "annotator_id", "annotator_model", "skill_file", "annotated_at",
}

authority_text = COMPACT_PROMPT_PATH.read_text(encoding="utf-8")
taxonomy_text = TAXONOMY_PATH.read_text(encoding="utf-8")
SYSTEM_PROMPT = f"""Eres el clasificador de este proyecto.
Las siguientes fuentes son la autoridad normativa completa. No uses una taxonomía externa.

=== REGLAS OPERATIVAS (compact) ===
{authority_text}

=== TAXONOMÍA CSV ===
{taxonomy_text}

ADAPTACIÓN TÉCNICA PARA API:
- Recibirás de 1 a {BATCH_SIZE} chunks por llamada.
- Devuelve exclusivamente JSON válido con el objeto raíz annotations.
- Conserva exactamente el orden y chunk_id de entrada; analiza cada chunk independientemente.
- Usa solo las categorías y flags literales de la taxonomía.
- ironia_ambigua, humor_encubridor y contexto_necesario van solo en flags.
- Si solo detectas flags, elimina los flags y usa seguro.
- notes siempre es texto y no supera 140 caracteres; justificacion no supera 450.
- No expongas razonamiento interno; justifica brevemente el criterio aplicado.
"""


def response_schema(batch_length: int) -> dict:
    annotation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chunk_id": {"type": "string", "minLength": 1},
            "labels": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(ALLOWED_LABELS)},
            },
            "flags": {
                "type": "array", "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(ALLOWED_FLAGS)},
            },
            "needs_review": {"type": "boolean"},
            "notes": {"type": "string", "maxLength": 160},
            "score_confianza": {"type": "number", "minimum": 0, "maximum": 1},
            "justificacion": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": sorted(SEMANTIC_FIELDS),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "annotations": {
                "type": "array", "minItems": batch_length, "maxItems": batch_length,
                "items": annotation,
            }
        },
        "required": ["annotations"],
    }


def normalize_response_structure(parsed: object) -> object:
    """Reubica valores válidos que el LLM puso en el campo taxonómico vecino.

    No corrige etiquetas desconocidas. Si la respuesta contiene únicamente un
    flag transversal y omite la categoría base, crea una decisión provisional
    ``seguro`` con ``needs_review=true`` para que la regla operativa la escale a
    Pro; así no se inventa una categoría de daño a partir de un flag.
    """
    if not isinstance(parsed, dict) or not isinstance(parsed.get("annotations"), list):
        return parsed
    for row in parsed["annotations"]:
        if not isinstance(row, dict):
            continue
        labels = row.get("labels")
        flags = row.get("flags")
        if not isinstance(labels, list) or not isinstance(flags, list):
            continue
        misplaced_flags = [value for value in labels if value in ALLOWED_FLAGS]
        misplaced_labels = [value for value in flags if value in ALLOWED_LABELS]
        if not misplaced_flags and not misplaced_labels:
            continue
        clean_labels = [value for value in labels if value not in ALLOWED_FLAGS]
        clean_flags = [value for value in flags if value not in ALLOWED_LABELS]
        clean_labels.extend(misplaced_labels)
        clean_flags.extend(misplaced_flags)
        row["labels"] = list(dict.fromkeys(clean_labels))
        row["flags"] = list(dict.fromkeys(clean_flags))
        if not row["labels"] and misplaced_flags:
            row["labels"] = ["seguro"]
            row["flags"] = []
            row["needs_review"] = True
            note = str(row.get("notes") or "")
            marker = "normalización: flag sin categoría base; escalar a Pro"
            row["notes"] = f"{note}; {marker}".strip("; ")[:160]
    return parsed


def validate_semantics(row: dict, expected_id: str) -> list[str]:
    errors = []
    if set(row) != SEMANTIC_FIELDS:
        errors.append(f"campos: {sorted(set(row) ^ SEMANTIC_FIELDS)}")
    if row.get("chunk_id") != expected_id:
        errors.append("chunk_id no coincide")
    labels, flags = row.get("labels", []), row.get("flags", [])
    if not isinstance(labels, list) or not labels or not set(labels) <= ALLOWED_LABELS:
        errors.append("labels inválidas")
    if not isinstance(flags, list) or not set(flags) <= ALLOWED_FLAGS:
        errors.append("flags inválidos")
    safe, damage = set(labels) & SAFE_LABELS, set(labels) & DAMAGE_LABELS
    if safe and damage or len(safe) > 1:
        errors.append("seguro incompatible con otras etiquetas")
    if flags and not damage:
        errors.append("flags sin daño")
    score = row.get("score_confianza")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        errors.append("score_confianza inválido")
    else:
        if ({"ironia_ambigua", "contexto_necesario"} & set(flags)) and score > 0.65:
            errors.append("score excede límite del flag")
        if (flags or score < 0.70) and row.get("needs_review") is not True:
            errors.append("needs_review debe ser true")
    if not isinstance(row.get("needs_review"), bool):
        errors.append("needs_review no booleano")
    if not isinstance(row.get("notes"), str):
        errors.append("notes no es texto")
    if not isinstance(row.get("justificacion"), str) or not row.get("justificacion", "").strip():
        errors.append("justificacion vacía")
    return errors


def normalize_semantics(row: dict) -> dict:
    output = dict(row)
    output["flags"] = output.get("flags") or []
    output["notes"] = str(output.get("notes") or "")[:160]
    output["justificacion"] = str(output.get("justificacion") or "")[:500]
    labels = output.get("labels") or []
    flags = output["flags"]
    if not (set(labels) & DAMAGE_LABELS):
        output["flags"] = []
    score = output.get("score_confianza")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        if {"ironia_ambigua", "contexto_necesario"} & set(output["flags"]):
            output["score_confianza"] = min(float(score), 0.65)
        if output["flags"] or float(output["score_confianza"]) < 0.70:
            output["needs_review"] = True
    return output


def api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Falta DEEPSEEK_API_KEY en 03_2_etiquetado_llm_api/.env")
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


def preflight() -> list[str]:
    response = requests.get(f"{API_BASE}/models", headers=api_headers(), timeout=30)
    response.raise_for_status()
    models = [str(row["id"]) for row in response.json().get("data", [])]
    missing = {FLASH_MODEL, PRO_MODEL} - set(models)
    if missing:
        raise RuntimeError(f"Modelos no visibles: {sorted(missing)}")
    return models


def input_message(records: list[dict]) -> str:
    payload = []
    for row in records:
        item = {
            "chunk_id": row["chunk_id"], "text": row["text"],
            "channel_title": row.get("channel_title"), "video_title": row.get("video_title"),
        }
        if row.get("contexto_anterior"):
            item["contexto_anterior"] = row["contexto_anterior"]
        if row.get("contexto_posterior"):
            item["contexto_posterior"] = row["contexto_posterior"]
        payload.append(item)
    return "Clasifica estos registros según las fuentes de autoridad:\n" + json.dumps(payload, ensure_ascii=False)


def call_api(records: list[dict], model: str, correction: str = "") -> tuple[list[dict], dict]:
    user_content = input_message(records)
    if correction:
        user_content += "\nCorrige estos errores de la respuesta anterior:\n" + correction
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_OVERHEAD + MAX_TOKENS_PER_RECORD * len(records),
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    response = api_session().post(
        f"{API_BASE}/chat/completions", headers=api_headers(), json=body,
        timeout=(15, REQUEST_TIMEOUT_SECONDS),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    response_json = response.json()
    choice = response_json["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("respuesta truncada por max_tokens")
    content = choice["message"].get("content") or ""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    parsed = normalize_response_structure(json.loads(clean))
    jsonschema.validate(parsed, response_schema(len(records)))
    annotations = parsed["annotations"]
    return annotations, response_json.get("usage", {})


def complete_row(row: dict, model: str, annotator_id: str) -> dict:
    return {
        "chunk_id": row["chunk_id"], "labels": row["labels"], "flags": row["flags"],
        "needs_review": bool(row["needs_review"]), "notes": row["notes"],
        "annotator_type": "llm", "annotator_id": annotator_id,
        "annotator_model": model, "skill_file": SKILL_PATH.name,
        "score_confianza": float(row["score_confianza"]),
        "justificacion": row["justificacion"].strip(), "annotated_at": now_iso(),
    }


def classify_batch(records: list[dict], model: str, annotator_id: str) -> tuple[list[dict], dict]:
    correction = ""
    for attempt in range(MAX_RETRIES):
        try:
            rows, usage = call_api(records, model, correction)
            normalized = []
            errors = []
            for expected, row in zip(records, rows):
                row = normalize_semantics(row)
                row_errors = validate_semantics(row, expected["chunk_id"])
                errors.extend(f"{expected['chunk_id']}: {error}" for error in row_errors)
                normalized.append(row)
            if errors:
                correction = "\n".join(errors[:20])
                raise ValueError(correction)
            return [complete_row(row, model, annotator_id) for row in normalized], usage
        except Exception as exc:
            if attempt + 1 >= MAX_RETRIES:
                if len(records) > 1:
                    recovered_rows: list[dict] = []
                    recovered_usage: dict[str, int] = defaultdict(int)
                    for record in records:
                        single_rows, single_usage = classify_batch([record], model, annotator_id)
                        recovered_rows.extend(single_rows)
                        for key, value in single_usage.items():
                            if isinstance(value, (int, float)):
                                recovered_usage[key] += int(value)
                    return recovered_rows, dict(recovered_usage)
                raise RuntimeError(
                    f"Falló lote {[row['chunk_id'] for row in records]} tras {MAX_RETRIES} intentos: {exc}"
                ) from exc
            correction = (
                "La salida no cumplió el contrato. Devuelve un objeto raíz con solo "
                f"annotations y exactamente {len(records)} anotaciones. Cada anotación debe "
                "contener solo chunk_id, labels, flags, needs_review, notes, "
                "score_confianza y justificacion; no copies text, channel_title ni video_title. "
                f"Error detectado: {str(exc)[:500]}"
            )
            time.sleep(min(2 ** attempt, 20) + random.random())
    raise AssertionError("inalcanzable")


def validate_existing(rows: list[dict], allowed_ids: set[str], model: str) -> set[str]:
    ids = []
    for index, row in enumerate(rows, 1):
        if set(row) != FINAL_FIELDS or row.get("chunk_id") not in allowed_ids:
            raise ValueError(f"Fila existente incompatible en posición {index}")
        if row.get("annotator_model") != model:
            raise ValueError(f"Modelo inesperado en fila {index}")
        errors = validate_semantics({key: row[key] for key in SEMANTIC_FIELDS}, row["chunk_id"])
        if errors:
            raise ValueError(f"Fila existente inválida {index}: {errors}")
        ids.append(row["chunk_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("La salida existente contiene IDs duplicados")
    return set(ids)


def accumulate_usage(target: dict, usage: dict) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            target[key] += value


def estimated_cost(usage: dict, model: str) -> float | None:
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if not pricing:
        return None
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", max(prompt - hit, 0)) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return round((hit * pricing["cache_hit"] + miss * pricing["cache_miss"] + completion * pricing["output"]) / 1_000_000, 6)


def metrics(rows: list[dict], target_rows: int, model: str, usage_new: dict, elapsed: float) -> dict:
    labels = Counter(label for row in rows for label in row["labels"])
    flags = Counter(flag for row in rows for flag in row["flags"])
    damage = sum(bool(set(row["labels"]) & DAMAGE_LABELS) for row in rows)
    return {
        "model": model, "completed": len(rows), "target_rows": target_rows,
        "pending": target_rows - len(rows), "damage_chunks": damage,
        "safe_chunks": len(rows) - damage,
        "needs_review": sum(bool(row["needs_review"]) for row in rows),
        "mean_confidence": sum(float(row["score_confianza"]) for row in rows) / max(len(rows), 1),
        "label_counts": {label: labels[label] for label in LABEL_ORDER},
        "flag_counts": {flag: flags[flag] for flag in FLAG_ORDER},
        "usage_new": dict(usage_new), "estimated_cost_usd_new": estimated_cost(usage_new, model),
        "elapsed_seconds_new": round(elapsed, 2),
    }


def execute(records: list[dict], output: Path, model: str, annotator_id: str) -> dict:
    allowed_ids = {row["chunk_id"] for row in records}
    existing = read_jsonl(output)
    completed_ids = validate_existing(existing, allowed_ids, model)
    pending = [row for row in records if row["chunk_id"] not in completed_ids]
    print(f"{model}: existentes={len(existing)}, pendientes={len(pending)}", flush=True)
    if not pending:
        return metrics(existing, len(records), model, {}, 0.0)
    batches = [pending[index : index + BATCH_SIZE] for index in range(0, len(pending), BATCH_SIZE)]
    usage_total: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as file, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        in_flight, ready = {}, {}
        next_submit = next_write = 0
        while next_submit < min(MAX_WORKERS, len(batches)):
            future = pool.submit(classify_batch, batches[next_submit], model, annotator_id)
            in_flight[future] = next_submit
            next_submit += 1
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                batch_index = in_flight.pop(future)
                ready[batch_index] = future.result()
                if next_submit < len(batches):
                    new_future = pool.submit(classify_batch, batches[next_submit], model, annotator_id)
                    in_flight[new_future] = next_submit
                    next_submit += 1
            while next_write in ready:
                rows, usage = ready.pop(next_write)
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
                existing.extend(rows)
                accumulate_usage(usage_total, usage)
                next_write += 1
                if next_write % 20 == 0 or next_write == len(batches):
                    print(
                        f"{model}: {len(existing)}/{len(records)} chunks; "
                        f"lotes={next_write}/{len(batches)}",
                        flush=True,
                    )
    return metrics(existing, len(records), model, usage_total, time.perf_counter() - started)


def neighbor_context(chunks: list[dict]) -> dict[str, dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in chunks:
        by_video[row["video_id"]].append(row)
    output = {}
    for rows in by_video.values():
        rows.sort(key=lambda row: (row.get("start_seconds", 0), row["chunk_id"]))
        for index, row in enumerate(rows):
            enriched = dict(row)
            if index:
                enriched["contexto_anterior"] = rows[index - 1]["text"]
            if index + 1 < len(rows):
                enriched["contexto_posterior"] = rows[index + 1]["text"]
            output[row["chunk_id"]] = enriched
    return output


def select_for_pro(chunks: list[dict], flash_rows: list[dict]) -> list[dict]:
    by_id = {row["chunk_id"]: row for row in flash_rows}
    if set(by_id) != {row["chunk_id"] for row in chunks}:
        raise ValueError("Flash no cubre exactamente el lote nuevo")
    rng = random.Random(SEED)
    selected_ids, reasons = [], {}
    for row in chunks:
        annotation = by_id[row["chunk_id"]]
        has_damage = bool(set(annotation["labels"]) & DAMAGE_LABELS)
        doubt = bool(annotation["needs_review"]) or float(annotation["score_confianza"]) < CONFIDENCE_THRESHOLD
        control = not has_damage and not doubt and rng.random() < SAFE_CONTROL_RATE
        if has_damage or doubt or control:
            selected_ids.append(row["chunk_id"])
            reasons[row["chunk_id"]] = {
                "flash_damage": has_damage, "flash_doubt": doubt, "safe_control": control,
            }
    contexts = neighbor_context(chunks)
    records = [contexts[chunk_id] for chunk_id in selected_ids]
    manifest = {
        "batch_id": BATCH_ID, "created_at": now_iso(), "source_flash": str(FLASH_PATH),
        "source_flash_sha256": sha256_file(FLASH_PATH),
        "rule": "Flash damage OR needs_review OR score_confianza < 0.90 OR 10% safe control",
        "threshold": CONFIDENCE_THRESHOLD, "safe_control_rate": SAFE_CONTROL_RATE,
        "seed": SEED, "selected_rows": len(records),
        "reason_counts": {
            key: sum(value[key] for value in reasons.values())
            for key in ("flash_damage", "flash_doubt", "safe_control")
        },
        "chunk_ids": selected_ids, "reasons": reasons,
    }
    write_json_atomic(PRO_MANIFEST_PATH, manifest)
    return records


def run(stage: str) -> None:
    chunks = read_jsonl(CHUNKS_PATH)
    if not chunks:
        raise FileNotFoundError("No existen chunks del lote de ampliación")
    models = preflight()
    print(f"API lista; modelos visibles={models}; chunks={len(chunks)}", flush=True)
    if stage in {"flash", "all"}:
        flash_stats = execute(chunks, FLASH_PATH, FLASH_MODEL, "DSF")
        write_json_atomic(
            FLASH_MANIFEST_PATH,
            {
                "batch_id": BATCH_ID, "created_at": now_iso(), "model": FLASH_MODEL,
                "chunks_sha256": sha256_file(CHUNKS_PATH), "output_sha256": sha256_file(FLASH_PATH),
                "prompt_sha256": sha256_file(COMPACT_PROMPT_PATH), "skill_sha256": sha256_file(SKILL_PATH),
                **flash_stats,
            },
        )
        print(json.dumps(flash_stats, ensure_ascii=False, indent=2), flush=True)
    if stage in {"pro", "all"}:
        flash_rows = read_jsonl(FLASH_PATH)
        if len(flash_rows) != len(chunks):
            raise RuntimeError("Flash debe terminar antes de ejecutar Pro")
        pro_records = select_for_pro(chunks, flash_rows)
        pro_stats = execute(pro_records, PRO_PATH, PRO_MODEL, "DSP")
        manifest = json.loads(PRO_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest.update(
            {
                "updated_at": now_iso(), "model": PRO_MODEL,
                "output_sha256": sha256_file(PRO_PATH), "prompt_sha256": sha256_file(COMPACT_PROMPT_PATH),
                "skill_sha256": sha256_file(SKILL_PATH), **pro_stats,
            }
        )
        write_json_atomic(PRO_MANIFEST_PATH, manifest)
        print(json.dumps(pro_stats, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "flash", "pro", "all"], default="all")
    args = parser.parse_args()
    if args.stage == "preflight":
        print(preflight())
    else:
        run(args.stage)


if __name__ == "__main__":
    main()
