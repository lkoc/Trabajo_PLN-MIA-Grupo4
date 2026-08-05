"""Revisión Pro reanudable de los 2.000 seguros sospechosos.

Este módulo replica el contrato de etiquetado de 03_2 sin ejecutar ni modificar
ese cuaderno. La anotación Pro se conserva con la taxonomía fina para mantener
trazabilidad, pero el entrenamiento posterior solo consume su proyección a las
categorías gruesas.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
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
from tqdm.auto import tqdm

from scripts_auxiliares.flujo_hibrido_moderador import (
    grouped_train_validation_test_split,
)


SAMPLE_SEED = 42
SPLIT_SEED = 131
BATCH_SIZE = 5
MAX_WORKERS = int(os.getenv("SUSPECT_REVIEW_MAX_WORKERS", "16"))
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 180
MAX_TOKENS_PER_RECORD = 512
MAX_TOKENS_OVERHEAD = 64
PROMPT_MODE = "compact"
PROMPT_BUNDLE_VERSION = "1.1"
REVIEW_ANNOTATOR_ID = "DSP"
SAFE_LABELS = {"seguro", "seguro_ironia_marcada"}
MODEL_PRICING_USD_PER_MILLION = {
    "deepseek-v4-pro": {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87}
}


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "datos" / "processed" / "chunks_para_etiquetar.jsonl").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto.")


ROOT = find_project_root()
MODULE_DIR = ROOT / "03_2_etiquetado_llm_api"
PROCESSED_DIR = ROOT / "datos" / "processed"
OUTPUT_DIR = ROOT / "datos" / "etiquetado" / "llm_api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(MODULE_DIR / ".env", override=False)

CHUNKS_FILE = PROCESSED_DIR / "chunks_para_etiquetar.jsonl"
TAXONOMY_FILE = PROCESSED_DIR / "taxonomia_moderacion.csv"
SKILL_FILE = ROOT / "modelos" / "skills" / "clasificacion_moderacion_peru.md"
OPERATIVE_PROMPT_FILE = ROOT / "para_equiquetado_LLM" / "PROMPT_ETIQUETADO_LLM.md"
COMPACT_PROMPT_FILE = MODULE_DIR / "prompt_operacional_compacto.md"
CANDIDATES_FILE = PROCESSED_DIR / "flash_seguros_dificiles_para_revision.csv"
CANDIDATES_MANIFEST = CANDIDATES_FILE.with_suffix(".manifest.json")
FLASH_FILE = OUTPUT_DIR / "deepseek-v4-flash_labeled_chunks_seed42.jsonl"
PREVIOUS_PRO_FILES = (
    OUTPUT_DIR / "deepseek-v4-pro_revision_de_deepseek-v4-flash_seed42.jsonl",
    OUTPUT_DIR / "deepseek-v4-pro_revision_umbral_recalibrado_t090_seed42.jsonl",
)
OUTPUT_FILE = OUTPUT_DIR / "deepseek-v4-pro_revision_sospechosos_gruesos_seed42.jsonl"
OUTPUT_MANIFEST = OUTPUT_FILE.with_suffix(".manifest.json")
OUTPUT_METRICS = OUTPUT_FILE.with_suffix(".metrics.json")

API_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
REVIEW_MODEL_ID = os.getenv("DEEPSEEK_REVIEW_MODEL", "deepseek-v4-pro").strip()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for lineno, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {lineno}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_by_id(rows: list[dict], source: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            raise ValueError(f"{source}: chunk_id vacío.")
        if chunk_id in result:
            raise ValueError(f"{source}: chunk_id duplicado: {chunk_id}")
        result[chunk_id] = row
    return result


taxonomy_df = pd.read_csv(TAXONOMY_FILE).fillna("")
ALLOWED_FLAGS = set(taxonomy_df.loc[taxonomy_df["categoria"] == "FLAG", "label"])
ALLOWED_LABELS = set(taxonomy_df.loc[taxonomy_df["categoria"] != "FLAG", "label"])
DAMAGE_LABELS = ALLOWED_LABELS - SAFE_LABELS
LABEL_ORDER = taxonomy_df.loc[taxonomy_df["categoria"] != "FLAG", "label"].tolist()
FLAG_ORDER = taxonomy_df.loc[taxonomy_df["categoria"] == "FLAG", "label"].tolist()

authority_text = COMPACT_PROMPT_FILE.read_text(encoding="utf-8")
taxonomy_text = TAXONOMY_FILE.read_text(encoding="utf-8")
SYSTEM_PROMPT = f"""Eres el clasificador de este proyecto.
Las siguientes fuentes son la autoridad normativa completa. No uses una taxonomía externa.

=== REGLAS OPERATIVAS ({PROMPT_MODE}) ===
{authority_text}

=== TAXONOMÍA CSV ===
{taxonomy_text}

ADAPTACIÓN TÉCNICA PARA API:
- Recibirás de 1 a {BATCH_SIZE} chunks por llamada.
- Devuelve exclusivamente JSON válido con el objeto raíz annotations exigido.
- Conserva exactamente el orden y chunk_id de entrada.
- Analiza cada chunk de forma independiente.
- Usa exclusivamente las categorías y flags literales incluidos en la taxonomía.
- ironia_ambigua, humor_encubridor y contexto_necesario van solo en flags.
- humor_encubridor acompaña la categoría de daño; nunca la reemplaza.
- Si solo detectas flags pero ninguna categoría de daño, elimina los flags y usa seguro.
- notes siempre debe ser texto: usa "" si no hay observación; máximo 140 caracteres.
- justificacion debe ser breve y no superar 450 caracteres.
- No expongas razonamiento interno; justifica brevemente el criterio aplicado.
- El programa añadirá los campos administrativos y escribirá una línea JSONL por anotación.
"""

SEMANTIC_FIELDS = {
    "chunk_id", "labels", "flags", "needs_review", "notes",
    "score_confianza", "justificacion",
}
FINAL_FIELDS = SEMANTIC_FIELDS | {
    "annotator_type", "annotator_id", "annotator_model", "skill_file", "annotated_at",
}


def response_schema(batch_length: int) -> dict:
    annotation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chunk_id": {"type": "string", "minLength": 1, "maxLength": 160},
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
        "type": "object", "additionalProperties": False,
        "properties": {
            "annotations": {
                "type": "array", "minItems": batch_length, "maxItems": batch_length,
                "items": annotation,
            }
        },
        "required": ["annotations"],
    }


def _api_headers() -> dict[str, str]:
    if not API_KEY:
        raise RuntimeError("Falta DEEPSEEK_API_KEY en el entorno o en 03_2_etiquetado_llm_api/.env.")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def api_preflight() -> dict:
    response = requests.get(f"{API_BASE}/models", headers=_api_headers(), timeout=30)
    response.raise_for_status()
    models = [str(item["id"]) for item in response.json().get("data", []) if item.get("id")]
    if REVIEW_MODEL_ID not in models:
        raise RuntimeError(f"{REVIEW_MODEL_ID!r} no aparece en /models: {models}")
    return {"api_base": API_BASE, "model": REVIEW_MODEL_ID, "models_visible": models}


def _build_neighbor_context(canonical: list[dict]) -> dict[str, dict]:
    by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for position, row in enumerate(canonical):
        by_video[str(row.get("video_id") or row["chunk_id"])].append((position, row))
    enriched: dict[str, dict] = {}
    for video_rows in by_video.values():
        video_rows.sort(key=lambda pair: (pair[1].get("start_seconds") or 0, pair[0]))
        for index, (_, row) in enumerate(video_rows):
            item = dict(row)
            if index > 0:
                item["contexto_anterior"] = video_rows[index - 1][1]["text"]
            if index + 1 < len(video_rows):
                item["contexto_posterior"] = video_rows[index + 1][1]["text"]
            enriched[row["chunk_id"]] = item
    return enriched


def load_and_audit_candidates() -> tuple[list[dict], dict]:
    required = [
        CHUNKS_FILE, TAXONOMY_FILE, SKILL_FILE, OPERATIVE_PROMPT_FILE,
        COMPACT_PROMPT_FILE, CANDIDATES_FILE, CANDIDATES_MANIFEST, FLASH_FILE,
        *PREVIOUS_PRO_FILES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan entradas:\n- " + "\n- ".join(missing))

    candidates = pd.read_csv(CANDIDATES_FILE, dtype={"chunk_id": str, "video_id": str})
    if len(candidates) != 2_000 or candidates["chunk_id"].nunique() != 2_000:
        raise ValueError("La selección debe contener exactamente 2.000 chunk_id únicos.")
    candidate_ids = candidates["chunk_id"].tolist()
    candidate_set = set(candidate_ids)
    manifest = json.loads(CANDIDATES_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("test_excluded") is not True or manifest.get("labels_changed") is not False:
        raise ValueError("El manifiesto de selección no conserva las garantías esperadas.")

    canonical = read_jsonl(CHUNKS_FILE)
    canonical_by_id = _unique_by_id(canonical, "canónico")
    if not candidate_set <= set(canonical_by_id):
        raise ValueError("La selección contiene IDs ajenos al canónico.")

    flash_by_id = _unique_by_id(read_jsonl(FLASH_FILE), "Flash")
    if set(flash_by_id) != set(canonical_by_id):
        raise ValueError("Flash no cubre exactamente el canónico.")
    unsafe = [cid for cid in candidate_ids if not set(flash_by_id[cid]["labels"]) <= SAFE_LABELS]
    routed = [
        cid for cid in candidate_ids
        if bool(flash_by_id[cid].get("needs_review"))
        or float(flash_by_id[cid].get("score_confianza", 0.0)) < 0.90
    ]
    if unsafe or routed:
        raise ValueError(
            f"La selección dejó de ser Flash-segura de alta confianza: daño={len(unsafe)}, "
            f"revisión/score<0.90={len(routed)}."
        )

    previous_pro_ids: set[str] = set()
    for path in PREVIOUS_PRO_FILES:
        previous_pro_ids.update(_unique_by_id(read_jsonl(path), path.name))
    overlap_previous_pro = candidate_set & previous_pro_ids
    if overlap_previous_pro:
        raise ValueError(f"Hay {len(overlap_previous_pro)} candidatos ya revisados por Pro.")

    human_path = PROCESSED_DIR / "dataset_etiquetado.jsonl"
    human_ids = set(_unique_by_id(read_jsonl(human_path), "humano")) if human_path.exists() else set()
    modeling_rows = [
        {"chunk_id": row["chunk_id"], "video_id": row.get("video_id") or row["chunk_id"]}
        for row in canonical if row["chunk_id"] not in human_ids
    ]
    modeling = pd.DataFrame(modeling_rows)
    split = grouped_train_validation_test_split(
        modeling, seed=SPLIT_SEED, test_size=0.15, validation_size=0.15
    )
    test_ids = set(modeling.iloc[split["test"]]["chunk_id"])
    test_videos = set(modeling.iloc[split["test"]]["video_id"])
    candidate_videos = set(candidates["video_id"])
    if candidate_set & test_ids or candidate_videos & test_videos:
        raise ValueError("La reproducción de la partición detectó contaminación con test.")

    enriched = _build_neighbor_context(canonical)
    records = [enriched[cid] for cid in candidate_ids]
    audit = {
        "candidate_rows": len(records),
        "unique_ids": len(candidate_set),
        "videos": len(candidate_videos),
        "flash_safe": len(records),
        "flash_high_confidence_no_review": len(records),
        "overlap_previous_pro": 0,
        "overlap_test_ids": 0,
        "overlap_test_videos": 0,
        "test_rows_reproduced": len(test_ids),
        "split_seed": SPLIT_SEED,
        "candidate_sha256": sha256_file(CANDIDATES_FILE),
        "canonical_sha256": sha256_file(CHUNKS_FILE),
    }
    return records, audit


def _validate_semantic(row: dict, expected_id: str) -> list[str]:
    errors: list[str] = []
    if set(row) != SEMANTIC_FIELDS:
        errors.append(f"campos inesperados/faltantes: {sorted(set(row) ^ SEMANTIC_FIELDS)}")
    if row.get("chunk_id") != expected_id:
        errors.append(f"chunk_id esperado {expected_id!r}, recibido {row.get('chunk_id')!r}")
    labels = row.get("labels", [])
    flags = row.get("flags", [])
    if not isinstance(labels, list) or not labels or not set(labels) <= ALLOWED_LABELS:
        errors.append("labels debe ser no vacío y pertenecer a la taxonomía")
    if not isinstance(flags, list) or not set(flags) <= ALLOWED_FLAGS:
        errors.append("flags debe pertenecer a la taxonomía")
    safe = set(labels) & SAFE_LABELS if isinstance(labels, list) else set()
    damage = set(labels) & DAMAGE_LABELS if isinstance(labels, list) else set()
    if safe and damage:
        errors.append("seguro no puede coexistir con daño")
    if len(safe) > 1:
        errors.append("las dos etiquetas seguras no pueden coexistir")
    if flags and not damage:
        errors.append("los flags requieren una categoría de daño")
    score = row.get("score_confianza")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        errors.append("score_confianza debe estar entre 0 y 1")
    else:
        if ({"ironia_ambigua", "contexto_necesario"} & set(flags)) and score > 0.65:
            errors.append("un flag ambiguo/contextual limita la confianza a 0.65")
        if (flags or score < 0.70) and row.get("needs_review") is not True:
            errors.append("flags o score<0.70 obligan needs_review=true")
    if not isinstance(row.get("needs_review"), bool):
        errors.append("needs_review debe ser booleano")
    if not isinstance(row.get("notes"), str):
        errors.append("notes debe ser texto")
    if not isinstance(row.get("justificacion"), str) or not row.get("justificacion", "").strip():
        errors.append("justificacion no puede estar vacía")
    return errors


def _normalize(row: dict) -> dict:
    normalized = dict(row)
    normalized["flags"] = normalized.get("flags") or []
    labels = normalized.get("labels")
    flags = normalized["flags"]
    if isinstance(labels, list) and isinstance(flags, list):
        misplaced = [value for value in labels if value in ALLOWED_FLAGS]
        if misplaced:
            normalized["labels"] = [value for value in labels if value not in ALLOWED_FLAGS]
            normalized["flags"] = list(dict.fromkeys([*flags, *misplaced]))
    normalized["notes"] = str(normalized.get("notes") or "").strip()[:160]
    if isinstance(normalized.get("justificacion"), str):
        normalized["justificacion"] = normalized["justificacion"].strip()[:500]
    score = normalized.get("score_confianza")
    flags = normalized["flags"]
    if ({"ironia_ambigua", "contexto_necesario"} & set(flags)) and isinstance(score, (int, float)):
        normalized["score_confianza"] = min(float(score), 0.65)
        score = normalized["score_confianza"]
    if flags or (isinstance(score, (int, float)) and score < 0.70):
        normalized["needs_review"] = True
    return normalized


def _complete(row: dict) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "labels": row["labels"],
        "flags": row["flags"],
        "needs_review": row["needs_review"],
        "notes": row["notes"],
        "annotator_type": "llm",
        "annotator_id": REVIEW_ANNOTATOR_ID,
        "annotator_model": REVIEW_MODEL_ID,
        "skill_file": SKILL_FILE.name,
        "score_confianza": float(row["score_confianza"]),
        "justificacion": row["justificacion"],
        "annotated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _build_input(records: list[dict]) -> str:
    payload = []
    for row in records:
        item = {
            "chunk_id": row["chunk_id"], "text": row["text"],
            "channel_title": row.get("channel_title"), "video_title": row.get("video_title"),
        }
        for key in ("contexto_anterior", "contexto_posterior"):
            if row.get(key):
                item[key] = row[key]
        payload.append(item)
    return "Clasifica estos registros según las fuentes de autoridad:\n" + json.dumps(
        payload, ensure_ascii=False
    )


_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return session


def _parse_json(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE)
    return json.loads(clean)


def _call_api(records: list[dict], correction: str = "", max_tokens: int | None = None):
    user_content = _build_input(records)
    if correction:
        user_content += "\nLa respuesta anterior fue inválida. Corrige estos errores:\n" + correction
    body = {
        "model": REVIEW_MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens or (MAX_TOKENS_OVERHEAD + MAX_TOKENS_PER_RECORD * len(records)),
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    response = _session().post(
        f"{API_BASE}/chat/completions", headers=_api_headers(), json=body,
        timeout=(15, REQUEST_TIMEOUT_SECONDS),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"API HTTP {response.status_code}: {response.text[:700]}")
    payload = response.json()
    choice = payload["choices"][0]
    reasoning_tokens = payload.get("usage", {}).get("completion_tokens_details", {}).get(
        "reasoning_tokens", 0
    )
    if reasoning_tokens:
        raise RuntimeError(f"El modelo usó {reasoning_tokens} tokens de razonamiento.")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("La respuesta agotó max_tokens antes de cerrar el JSON.")
    parsed = _parse_json(choice["message"]["content"])
    if not isinstance(parsed, dict) or set(parsed) != {"annotations"}:
        raise ValueError("La raíz debe contener exclusivamente annotations.")
    annotations = parsed["annotations"]
    if not isinstance(annotations, list) or len(annotations) != len(records):
        raise ValueError(f"Se esperaban {len(records)} anotaciones.")
    return annotations, payload.get("usage", {})


def _add_usage(target: defaultdict, usage: dict) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] += value


def classify_batch(records: list[dict]) -> tuple[list[dict], dict]:
    original_ids = [row["chunk_id"] for row in records]
    pending = list(records)
    completed: dict[str, dict] = {}
    usage_total: defaultdict = defaultdict(int)
    correction = ""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 2):
        token_budget = (
            MAX_TOKENS_OVERHEAD + MAX_TOKENS_PER_RECORD * len(pending)
        ) * (2 ** (attempt - 1))
        try:
            annotations, usage = _call_api(pending, correction, token_budget)
            _add_usage(usage_total, usage)
            expected = [row["chunk_id"] for row in pending]
            received = [row.get("chunk_id") if isinstance(row, dict) else None for row in annotations]
            if received != expected:
                raise ValueError(f"orden/IDs incorrectos: esperado={expected}, recibido={received}")
            next_pending: list[dict] = []
            all_errors: list[str] = []
            item_schema = response_schema(1)["properties"]["annotations"]["items"]
            for record, row in zip(pending, annotations):
                normalized = _normalize(row) if isinstance(row, dict) else {}
                try:
                    jsonschema.validate(normalized, item_schema)
                except jsonschema.ValidationError as exc:
                    errors = [f"JSON Schema: {exc.message}"]
                else:
                    errors = _validate_semantic(normalized, record["chunk_id"])
                if errors:
                    next_pending.append(record)
                    all_errors.extend(f"{record['chunk_id']}: {error}" for error in errors)
                else:
                    completed[record["chunk_id"]] = _complete(normalized)
            if not next_pending:
                return [completed[chunk_id] for chunk_id in original_ids], dict(usage_total)
            pending = next_pending
            last_error = ValueError("; ".join(all_errors))
            correction = str(last_error)[:3_000]
        except Exception as exc:
            last_error = exc
            correction = str(exc)[:3_000]
        if attempt <= MAX_RETRIES:
            delay = min(30.0, 2 ** (attempt - 1))
            time.sleep(delay + random.random() * min(1.0, delay * 0.25))
    raise RuntimeError(
        f"Fallaron {[row['chunk_id'] for row in pending]} tras {MAX_RETRIES + 1} intentos: "
        f"{last_error}"
    ) from last_error


def _validate_final(row: dict, allowed_ids: set[str]) -> list[str]:
    errors = []
    if set(row) != FINAL_FIELDS:
        errors.append(f"campos inesperados/faltantes: {sorted(set(row) ^ FINAL_FIELDS)}")
    if row.get("chunk_id") not in allowed_ids:
        errors.append("chunk_id fuera de la selección")
    semantic = {key: row.get(key) for key in SEMANTIC_FIELDS}
    errors.extend(_validate_semantic(semantic, row.get("chunk_id")))
    if row.get("annotator_type") != "llm" or row.get("annotator_id") != REVIEW_ANNOTATOR_ID:
        errors.append("metadatos del anotador incorrectos")
    if row.get("annotator_model") != REVIEW_MODEL_ID or row.get("skill_file") != SKILL_FILE.name:
        errors.append("modelo o skill incorrecto")
    return errors


def _estimate_cost(usage: dict) -> float | None:
    prices = MODEL_PRICING_USD_PER_MILLION.get(REVIEW_MODEL_ID)
    if not prices:
        return None
    prompt_total = int(usage.get("prompt_tokens", 0) or 0)
    cache_hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(usage.get("prompt_cache_miss_tokens", max(prompt_total - cache_hit, 0)) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return round(
        (cache_hit * prices["cache_hit"] + cache_miss * prices["cache_miss"]
         + completion * prices["output"]) / 1_000_000,
        6,
    )


def _summary(rows: list[dict], target: int) -> dict:
    label_counts = Counter(label for row in rows for label in set(row["labels"]))
    flag_counts = Counter(flag for row in rows for flag in set(row["flags"]))
    return {
        "completed": len(rows),
        "pending": target - len(rows),
        "progress_pct": round(100 * len(rows) / target, 3) if target else 100.0,
        "safe_chunks": sum(bool(set(row["labels"]) & SAFE_LABELS) for row in rows),
        "damage_chunks": sum(bool(set(row["labels"]) & DAMAGE_LABELS) for row in rows),
        "needs_review": sum(bool(row["needs_review"]) for row in rows),
        "mean_confidence": round(
            sum(float(row["score_confianza"]) for row in rows) / len(rows), 4
        ) if rows else None,
        "label_counts": {label: int(label_counts[label]) for label in LABEL_ORDER},
        "flag_counts": {flag: int(flag_counts[flag]) for flag in FLAG_ORDER},
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_manifest(records: list[dict], audit: dict) -> None:
    payload = {
        "purpose": "independent Pro review of 2,000 Flash-safe hard negatives",
        "training_target": "coarse categories only; fine labels retained only for traceability",
        "source_candidates": str(CANDIDATES_FILE),
        "output": str(OUTPUT_FILE),
        "api_provider": "deepseek",
        "api_base": API_BASE,
        "review_model": REVIEW_MODEL_ID,
        "review_annotator_id": REVIEW_ANNOTATOR_ID,
        "prompt_mode": PROMPT_MODE,
        "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
        "skill_sha256": sha256_file(SKILL_FILE),
        "operative_prompt_sha256": sha256_file(OPERATIVE_PROMPT_FILE),
        "compact_prompt_sha256": sha256_file(COMPACT_PROMPT_FILE),
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
        "sample_seed": SAMPLE_SEED,
        "split_seed": SPLIT_SEED,
        "selected_size": len(records),
        "chunk_ids": [row["chunk_id"] for row in records],
        "preflight_audit": audit,
    }
    if OUTPUT_MANIFEST.exists():
        previous = json.loads(OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        if previous.get("chunk_ids") != payload["chunk_ids"]:
            raise ValueError("El manifiesto existente corresponde a otra selección.")
    _write_json_atomic(OUTPUT_MANIFEST, payload)


def audit_output(records: list[dict]) -> dict:
    allowed_ids = {row["chunk_id"] for row in records}
    rows = read_jsonl(OUTPUT_FILE) if OUTPUT_FILE.exists() else []
    ids = [row.get("chunk_id") for row in rows]
    expected_ids = [row["chunk_id"] for row in records]
    errors: list[str] = []
    if len(ids) != len(set(ids)):
        errors.append("chunk_id duplicados")
    for index, row in enumerate(rows, 1):
        errors.extend(f"fila {index}: {error}" for error in _validate_final(row, allowed_ids))
    if ids != expected_ids[: len(ids)]:
        errors.append("el orden no coincide con el prefijo reproducible de la selección")
    return {**_summary(rows, len(records)), "valid": not errors, "errors": errors[:50]}


def run_review(records: list[dict], limit: int | None = None) -> dict:
    allowed_ids = {row["chunk_id"] for row in records}
    existing = read_jsonl(OUTPUT_FILE) if OUTPUT_FILE.exists() else []
    existing_ids = [row.get("chunk_id") for row in existing]
    if existing_ids != [row["chunk_id"] for row in records[: len(existing)]]:
        raise ValueError("La salida parcial no es un prefijo válido de la selección.")
    for row in existing:
        errors = _validate_final(row, allowed_ids)
        if errors:
            raise ValueError(f"Salida parcial inválida para {row.get('chunk_id')}: {errors}")
    pending = records[len(existing):]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        result = {"new_rows": 0, **audit_output(records)}
        # Una auditoría posterior no debe borrar tokens, costo y tiempo de la
        # ejecución que produjo la salida. El resultado liviano se devuelve al
        # notebook, mientras el archivo de métricas histórico se conserva.
        if not OUTPUT_METRICS.exists():
            _write_json_atomic(OUTPUT_METRICS, result)
        return result

    batches = [pending[start:start + BATCH_SIZE] for start in range(0, len(pending), BATCH_SIZE)]
    started = time.perf_counter()
    usage_total: defaultdict = defaultdict(int)
    new_rows = 0
    all_rows = list(existing)
    with (
        OUTPUT_FILE.open("a", encoding="utf-8", newline="\n") as file,
        ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor,
        tqdm(total=len(records), initial=len(existing), unit="chunk", desc=OUTPUT_FILE.stem) as progress,
    ):
        in_flight = {}
        ready = {}
        next_submit = 0
        next_write = 0
        while next_submit < min(MAX_WORKERS, len(batches)):
            future = executor.submit(classify_batch, batches[next_submit])
            in_flight[future] = next_submit
            next_submit += 1
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                batch_index = in_flight.pop(future)
                ready[batch_index] = future.result()
                if next_submit < len(batches):
                    next_future = executor.submit(classify_batch, batches[next_submit])
                    in_flight[next_future] = next_submit
                    next_submit += 1
            while next_write in ready:
                rows, usage = ready.pop(next_write)
                for row in rows:
                    errors = _validate_final(row, allowed_ids)
                    if errors:
                        raise ValueError(f"Salida final inválida para {row.get('chunk_id')}: {errors}")
                    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                file.flush()
                os.fsync(file.fileno())
                all_rows.extend(rows)
                new_rows += len(rows)
                _add_usage(usage_total, usage)
                live = {
                    "model": REVIEW_MODEL_ID,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "new_rows_this_run": new_rows,
                    "usage_this_run": dict(usage_total),
                    "estimated_cost_usd_this_run": _estimate_cost(usage_total),
                    **_summary(all_rows, len(records)),
                }
                _write_json_atomic(OUTPUT_METRICS, live)
                progress.update(len(rows))
                progress.set_postfix(dano=live["damage_chunks"], revision=live["needs_review"])
                next_write += 1
    elapsed = time.perf_counter() - started
    audit = audit_output(records)
    result = {
        "output": str(OUTPUT_FILE),
        "model": REVIEW_MODEL_ID,
        "new_rows": new_rows,
        "elapsed_seconds": round(elapsed, 2),
        "chunks_per_minute": round(new_rows / elapsed * 60, 3) if elapsed else None,
        "usage": dict(usage_total),
        "estimated_cost_usd_new_rows": _estimate_cost(usage_total),
        **audit,
    }
    _write_json_atomic(OUTPUT_METRICS, result)
    return result
