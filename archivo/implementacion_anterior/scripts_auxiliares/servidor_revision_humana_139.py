"""Servidor local para adjudicar la cola humana combinada Flash -> Pro -> humano.

La campaña reúne los 139 casos difíciles del corpus original y todas las colas
de ampliación descubiertas bajo ``datos/ampliacion``. La propuesta Pro se
muestra desde el inicio y admite tres decisiones explícitas: aceptarla,
rechazarla y excluir el chunk, o modificarla. Las categorías finas de Pro son
solo referencia; el objetivo humano es grueso.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd

from scripts_auxiliares.flujo_hibrido_moderador import grouped_train_validation_test_split
from scripts_auxiliares.modelos_gruesos_moderador import (
    COARSE_ORDER,
    DAMAGE_ORDER,
    FINE_TO_COARSE,
)


CAMPAIGN_ID = "revision_humana_combinada_1918_v2"
CAMPAIGN_SEED = 13942
EXPANSION_SEED = 177942
SPLIT_SEED = 131
EXPANSION_SPLIT_SEED = 26072026
ORIGINAL_ROWS = 139
EXPANSION_ROWS = 1_779
EXPECTED_ROWS = ORIGINAL_ROWS + EXPANSION_ROWS
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ALLOWED_FLAGS = ["ironia_ambigua", "humor_encubridor", "contexto_necesario"]
ALLOWED_ACTIONS = {
    "accept_llm",
    "reject_llm",
    "modify_llm",
    "legacy_human_decision",
}
COHORTS = {
    "original_139": {
        "label": "Corrida original (139)",
        "rows": ORIGINAL_ROWS,
        "source_run": "auditoria_dirigida_pro_2000",
    },
    "ampliacion_1779": {
        "label": "Segunda corrida / ampliación (1.779)",
        "rows": EXPANSION_ROWS,
        "source_run": "ampliacion_dirigida_21991",
    },
}
COARSE_DEFINITIONS = {
    "SEGURO": "No hay ataque, amenaza ni sexualización dañina; lo citado, informativo o crítico sin daño también es seguro.",
    "RACISMO_DISCRIMINACION": "Inferiorización o exclusión por etnia, rasgos racializados, acento, clase asociada o procedencia regional.",
    "ACOSO_GENERO_IDENTIDAD": "Ataque o degradación por género, misoginia, orientación sexual o identidad de género.",
    "ACOSO_PERSONAL": "Insulto, hostigamiento o degradación dirigido a una persona identificable.",
    "AMENAZA_DIRECTA": "Intención explícita o inequívoca de causar daño físico, sexual, patrimonial o equivalente.",
    "CONTENIDO_SEXUAL": "Contenido sexual explícito dañino, cosificación sexual o material no consensual.",
}
FLAG_DEFINITIONS = {
    "ironia_ambigua": "La ironía no permite decidir con seguridad si hay daño genuino.",
    "humor_encubridor": "El humor funciona como cobertura para normalizar o minimizar el daño.",
    "contexto_necesario": "Aun con los vecinos mostrados, se requiere revisar más contexto del video.",
}


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "datos" / "processed" / "chunks_para_etiquetar.jsonl").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto.")


ROOT = find_project_root()
PROCESSED_DIR = ROOT / "datos" / "processed"
LLM_DIR = ROOT / "datos" / "etiquetado" / "llm_api"
HUMAN_DIR = ROOT / "datos" / "etiquetado" / "humano"
FRONTEND_DIR = ROOT / "Cuadernos" / "frontend"
RESULTS_DIR = ROOT / "resultados"
EXPANSION_DIR = ROOT / "datos" / "ampliacion" / "ampliacion_dano_20260726"
HUMAN_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_PATH = PROCESSED_DIR / "chunks_para_etiquetar.jsonl"
CANDIDATES_PATH = PROCESSED_DIR / "flash_seguros_dificiles_para_revision.csv"
PRO_PATH = LLM_DIR / "deepseek-v4-pro_revision_sospechosos_gruesos_seed42.jsonl"
EXPANSION_CHUNKS_PATH = EXPANSION_DIR / "processed" / "chunks_para_etiquetar.jsonl"
EXPANSION_PENDING_PATH = EXPANSION_DIR / "processed" / "pendientes_revision_humana.jsonl"
EXPANSION_USABLE_PATH = EXPANSION_DIR / "processed" / "dataset_etiquetado_utilizable.jsonl"

LEGACY_CAMPAIGN_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.campaign.json"
LEGACY_PROGRESS_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.progress.json"
LEGACY_EVENTS_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.events.jsonl"
CAMPAIGN_PATH = HUMAN_DIR / "revision_humana_combinada_1918.campaign.json"
CAMPAIGN_MANIFEST_PATH = HUMAN_DIR / "revision_humana_combinada_1918.campaign.manifest.json"
PROGRESS_PATH = HUMAN_DIR / "revision_humana_combinada_1918.progress.json"
EVENTS_PATH = HUMAN_DIR / "revision_humana_combinada_1918.events.jsonl"
COMBINED_FINAL_PATH = HUMAN_DIR / "revision_humana_combinada_1918.jsonl"
COMBINED_FINAL_MANIFEST_PATH = HUMAN_DIR / "revision_humana_combinada_1918.manifest.json"
ORIGINAL_FINAL_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.jsonl"
ORIGINAL_FINAL_MANIFEST_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.manifest.json"
EXPANSION_FINAL_PATH = HUMAN_DIR / "revision_humana_ampliacion_1779.jsonl"
EXPANSION_FINAL_MANIFEST_PATH = HUMAN_DIR / "revision_humana_ampliacion_1779.manifest.json"
HTML_PATH = FRONTEND_DIR / "revision_humana_sospechosos_139.html"
REPORT_PATH = RESULTS_DIR / "INFORME_ADJUDICACION_HUMANA_COMBINADA_1918.md"
PID_PATH = HUMAN_DIR / "revision_humana_sospechosos_139.server.json"

_write_lock = Lock()


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _discover_expansion_specs() -> list[dict]:
    """Descubre colas Pro dudosas de todas las campañas de ampliación."""
    specs = []
    for pending_path in sorted(
        (ROOT / "datos" / "ampliacion").glob(
            "*/processed/pendientes_revision_humana.jsonl"
        )
    ):
        batch_dir = pending_path.parents[1]
        batch_id = batch_dir.name
        chunks_path = batch_dir / "processed" / "chunks_para_etiquetar.jsonl"
        usable_path = batch_dir / "processed" / "dataset_etiquetado_utilizable.jsonl"
        manifest_path = usable_path.with_suffix(".manifest.json")
        required = [chunks_path, usable_path, manifest_path]
        if any(not path.exists() for path in required):
            continue
        rows = read_jsonl(pending_path)
        if not rows:
            continue
        if batch_id == "ampliacion_dano_20260726":
            cohort = "ampliacion_1779"
            final_path = EXPANSION_FINAL_PATH
            final_manifest_path = EXPANSION_FINAL_MANIFEST_PATH
            label = "Segunda corrida / ampliación (1.779)"
            seed = EXPANSION_SEED
        else:
            cohort = batch_id
            final_path = batch_dir / "processed" / "revision_humana.jsonl"
            final_manifest_path = batch_dir / "processed" / "revision_humana.manifest.json"
            label = f"Ampliación {batch_id} ({len(rows):,})"
            seed = _stable_seed(batch_id)
        specs.append(
            {
                "batch_id": batch_id,
                "cohort": cohort,
                "label": label,
                "rows": len(rows),
                "seed": seed,
                "chunks_path": chunks_path,
                "pending_path": pending_path,
                "usable_path": usable_path,
                "manifest_path": manifest_path,
                "final_path": final_path,
                "final_manifest_path": final_manifest_path,
            }
        )
    # Una campaña ya publicada fija el orden. Nuevos directorios pueden tener
    # nombres que ordenarían antes alfabéticamente; siempre se anexan al final
    # para preservar chunk_ids y decisiones humanas de manera append-only.
    if CAMPAIGN_PATH.exists():
        previous_campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
        previous_order = [
            cohort
            for cohort in previous_campaign.get("cohorts", {})
            if cohort != "original_139"
        ]
        rank = {cohort: index for index, cohort in enumerate(previous_order)}
        specs.sort(
            key=lambda spec: (
                0 if spec["cohort"] in rank else 1,
                rank.get(spec["cohort"], 0),
                spec["batch_id"],
            )
        )
    return specs


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {line_number}: {exc}") from exc
    return rows


EXPANSION_SPECS = _discover_expansion_specs()
COHORTS = {
    "original_139": {
        "label": "Corrida original (139)",
        "rows": ORIGINAL_ROWS,
        "source_run": "auditoria_dirigida_pro_2000",
    },
    **{
        spec["cohort"]: {
            "label": spec["label"],
            "rows": spec["rows"],
            "source_run": spec["batch_id"],
        }
        for spec in EXPANSION_SPECS
    },
}
EXPECTED_ROWS = ORIGINAL_ROWS + sum(spec["rows"] for spec in EXPANSION_SPECS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _unique_by_id(rows: list[dict], source: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in rows:
        chunk_id = str(row.get("chunk_id") or "")
        if not chunk_id:
            raise ValueError(f"{source}: chunk_id vacío.")
        if chunk_id in output:
            raise ValueError(f"{source}: chunk_id duplicado: {chunk_id}")
        output[chunk_id] = row
    return output


def _coarse_from_fine(labels: list[str]) -> list[str]:
    unknown = set(labels) - set(FINE_TO_COARSE)
    if unknown:
        raise ValueError(f"Etiquetas finas fuera del mapeo grueso: {sorted(unknown)}")
    mapped = {FINE_TO_COARSE[label] for label in labels}
    return [label for label in COARSE_ORDER if label in mapped]


def _neighbor_context(canonical: list[dict]) -> dict[str, dict]:
    by_video: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for position, row in enumerate(canonical):
        video_id = str(row.get("video_id") or row["chunk_id"])
        by_video[video_id].append((position, row))
    output: dict[str, dict] = {}
    for video_rows in by_video.values():
        video_rows.sort(key=lambda pair: (pair[1].get("start_seconds") or 0, pair[0]))
        for index, (_, row) in enumerate(video_rows):
            output[row["chunk_id"]] = {
                "previous_text": video_rows[index - 1][1].get("text", "") if index else "",
                "next_text": (
                    video_rows[index + 1][1].get("text", "")
                    if index + 1 < len(video_rows)
                    else ""
                ),
            }
    return output


def _original_split_assignment(canonical: list[dict]) -> dict[str, str]:
    human_holdout_path = PROCESSED_DIR / "dataset_etiquetado.jsonl"
    human_ids = (
        set(_unique_by_id(read_jsonl(human_holdout_path), "holdout humano"))
        if human_holdout_path.exists()
        else set()
    )
    frame = pd.DataFrame(
        [
            {"chunk_id": row["chunk_id"], "video_id": row.get("video_id") or row["chunk_id"]}
            for row in canonical
            if row["chunk_id"] not in human_ids
        ]
    )
    split = grouped_train_validation_test_split(
        frame, seed=SPLIT_SEED, test_size=0.15, validation_size=0.15
    )
    return {
        str(chunk_id): split_name
        for split_name, indices in split.items()
        for chunk_id in frame.iloc[indices]["chunk_id"]
    }


def _expansion_split_assignment(
    pending: list[dict], usable_path: Path, split_seed: int
) -> dict[str, str]:
    usable = read_jsonl(usable_path)
    video_split: dict[str, str] = {}
    for row in usable:
        video_id, split = str(row["video_id"]), str(row["split"])
        if video_id in video_split and video_split[video_id] != split:
            raise ValueError(f"El video {video_id} aparece en dos particiones de ampliación.")
        video_split[video_id] = split
    assignment: dict[str, str] = {}
    for row in pending:
        video_id = str(row["video_id"])
        split = video_split.get(video_id)
        if split is None:
            digest = hashlib.sha256(f"{split_seed}:{video_id}".encode()).hexdigest()
            split = "validation" if int(digest[:16], 16) / 16**16 < 0.20 else "train"
        if split not in {"train", "validation"}:
            raise ValueError(f"Partición inválida en ampliación: {split}")
        assignment[str(row["chunk_id"])] = split
    return assignment


def _prepare_original_records() -> tuple[list[dict], dict]:
    canonical = read_jsonl(CANONICAL_PATH)
    canonical_by_id = _unique_by_id(canonical, "canónico original")
    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"chunk_id": str, "video_id": str})
    candidate_by_id = candidates.set_index("chunk_id").to_dict("index")
    pro_rows = read_jsonl(PRO_PATH)
    difficult = [row for row in pro_rows if bool(row.get("needs_review"))]
    difficult_by_id = _unique_by_id(difficult, "Pro original pendiente")
    if len(difficult_by_id) != ORIGINAL_ROWS:
        raise ValueError(f"Se esperaban 139 pendientes originales; hay {len(difficult_by_id)}.")
    if not set(difficult_by_id) <= set(canonical_by_id) or not set(difficult_by_id) <= set(candidate_by_id):
        raise ValueError("La cola original contiene IDs ajenos a sus fuentes.")
    contexts = _neighbor_context(canonical)
    split_by_id = _original_split_assignment(canonical)
    if any(split_by_id[chunk_id] == "test" for chunk_id in difficult_by_id):
        raise ValueError("La cola original contiene un caso de test.")
    ordered_ids = list(difficult_by_id)
    random.Random(CAMPAIGN_SEED).shuffle(ordered_ids)
    records: list[dict] = []
    for chunk_id in ordered_ids:
        chunk, pro, candidate = (
            canonical_by_id[chunk_id],
            difficult_by_id[chunk_id],
            candidate_by_id[chunk_id],
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "video_id": chunk.get("video_id") or chunk_id,
                "channel_title": chunk.get("channel_title") or "",
                "video_title": chunk.get("video_title") or "",
                "start_seconds": chunk.get("start_seconds"),
                "end_seconds": chunk.get("end_seconds"),
                "text": chunk["text"],
                **contexts[chunk_id],
                "split": split_by_id[chunk_id],
                "cohort": "original_139",
                "source_run": COHORTS["original_139"]["source_run"],
                "pro_reference": {
                    "fine_labels": pro["labels"],
                    "coarse_labels": _coarse_from_fine(pro["labels"]),
                    "flags": pro.get("flags", []),
                    "score_confianza": pro.get("score_confianza"),
                    "notes": pro.get("notes", ""),
                    "justificacion": pro.get("justificacion", ""),
                    "selection_context": (
                        f"Moderador: {candidate['categoria_sospechada']} · "
                        f"score daño {float(candidate['score_dano_maximo']):.4f}"
                    ),
                },
            }
        )
    return records, {
        "rows": len(records),
        "split_counts": dict(Counter(row["split"] for row in records)),
    }


def _prepare_expansion_records(spec: dict) -> tuple[list[dict], dict]:
    chunks = read_jsonl(spec["chunks_path"])
    chunks_by_id = _unique_by_id(chunks, f"chunks de {spec['batch_id']}")
    pending = read_jsonl(spec["pending_path"])
    pending_by_id = _unique_by_id(pending, f"pendientes de {spec['batch_id']}")
    if len(pending_by_id) != int(spec["rows"]):
        raise ValueError(
            f"Se esperaban {spec['rows']} pendientes de {spec['batch_id']}; "
            f"hay {len(pending_by_id)}."
        )
    if not set(pending_by_id) <= set(chunks_by_id):
        raise ValueError("La cola de ampliación contiene IDs ajenos a sus chunks.")
    contexts = _neighbor_context(chunks)
    split_by_id = _expansion_split_assignment(
        pending, spec["usable_path"], int(spec["seed"])
    )
    ordered_ids = list(pending_by_id)
    random.Random(int(spec["seed"])).shuffle(ordered_ids)
    records: list[dict] = []
    for chunk_id in ordered_ids:
        chunk, pro = chunks_by_id[chunk_id], pending_by_id[chunk_id]
        records.append(
            {
                "chunk_id": chunk_id,
                "video_id": chunk.get("video_id") or chunk_id,
                "channel_title": chunk.get("channel_title") or pro.get("channel_title") or "",
                "video_title": chunk.get("video_title") or pro.get("video_title") or "",
                "start_seconds": chunk.get("start_seconds"),
                "end_seconds": chunk.get("end_seconds"),
                "text": chunk["text"],
                **contexts[chunk_id],
                "split": split_by_id[chunk_id],
                "cohort": spec["cohort"],
                "source_run": COHORTS[spec["cohort"]]["source_run"],
                "pro_reference": {
                    "fine_labels": pro["pro_labels"],
                    "coarse_labels": pro["pro_coarse_labels"],
                    "flags": pro.get("pro_flags", []),
                    "score_confianza": pro.get("pro_score_confianza"),
                    "notes": pro.get("pro_notes", ""),
                    "justificacion": pro.get("pro_justificacion", ""),
                    "selection_context": (
                        f"Adquisición: {pro.get('discovery_type', '—')} · "
                        f"objetivo: {pro.get('target_category', '—')}"
                    ),
                },
            }
        )
    return records, {
        "rows": len(records),
        "videos": len({row["video_id"] for row in records}),
        "split_counts": dict(Counter(row["split"] for row in records)),
    }


def _migrate_legacy_progress(campaign: dict) -> dict:
    annotations: list[dict] = []
    revision = 0
    updated_at = None
    original_ids = {
        row["chunk_id"] for row in campaign["records"] if row["cohort"] == "original_139"
    }
    if LEGACY_PROGRESS_PATH.exists():
        legacy = json.loads(LEGACY_PROGRESS_PATH.read_text(encoding="utf-8"))
        revision = int(legacy.get("revision", 0))
        updated_at = legacy.get("updated_at")
        for old in legacy.get("annotations", []):
            if old.get("chunk_id") not in original_ids:
                continue
            migrated = dict(old)
            migrated["campaign_id"] = CAMPAIGN_ID
            migrated["pro_revealed"] = True
            if migrated.get("status") == "completed":
                migrated["review_action"] = "legacy_human_decision"
                migrated["training_eligible"] = True
                migrated["label_origin"] = "legacy_human_decision"
            else:
                migrated["review_action"] = "modify_llm"
                migrated["training_eligible"] = False
                migrated["label_origin"] = None
            annotations.append(migrated)
    return {
        "campaign_id": CAMPAIGN_ID,
        "updated_at": updated_at,
        "revision": revision,
        "migration": {
            "source_campaign_id": "revision_humana_sospechosos_139_v1",
            "source_progress": str(LEGACY_PROGRESS_PATH.relative_to(ROOT)),
            "migrated_annotations": len(annotations),
            "migrated_at": now_iso(),
        },
        "annotations": annotations,
    }


def prepare_campaign() -> dict:
    required = [
        CANONICAL_PATH,
        CANDIDATES_PATH,
        PRO_PATH,
        HTML_PATH,
    ]
    required.extend(
        path
        for spec in EXPANSION_SPECS
        for path in (
            spec["chunks_path"],
            spec["pending_path"],
            spec["usable_path"],
            spec["manifest_path"],
        )
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan archivos de campaña:\n- " + "\n- ".join(missing))
    original, original_meta = _prepare_original_records()
    records = list(original)
    cohort_metadata = {"original_139": original_meta}
    seen_ids = {row["chunk_id"] for row in records}
    for spec in EXPANSION_SPECS:
        expansion, expansion_meta = _prepare_expansion_records(spec)
        overlap = seen_ids & {row["chunk_id"] for row in expansion}
        if overlap:
            raise ValueError(
                f"{spec['cohort']} se solapa con cohortes previas en {len(overlap)} IDs."
            )
        seen_ids.update(row["chunk_id"] for row in expansion)
        records.extend(expansion)
        cohort_metadata[spec["cohort"]] = expansion_meta
    if len(records) != EXPECTED_ROWS:
        raise AssertionError(f"La campaña combinada tiene {len(records)} filas, no {EXPECTED_ROWS}.")
    campaign = {
        "campaign_id": CAMPAIGN_ID,
        "purpose": "adjudicación humana gruesa de dudas persistentes de Pro",
        "created_at": now_iso(),
        "expected_rows": EXPECTED_ROWS,
        "cohorts": COHORTS,
        "order_method": "cohortes consecutivas; barajado determinístico dentro de cada cohorte",
        "order_seeds": {
            "original_139": CAMPAIGN_SEED,
            **{spec["cohort"]: spec["seed"] for spec in EXPANSION_SPECS},
        },
        "test_excluded": True,
        "decision_policy": {
            "accept_llm": "incluye exactamente categorías gruesas y flags de Pro",
            "reject_llm": "excluye el chunk del entrenamiento",
            "modify_llm": "incluye categorías gruesas y flags seleccionados por humano",
        },
        "llm_visible_before_decision": True,
        "coarse_order": COARSE_ORDER,
        "coarse_definitions": COARSE_DEFINITIONS,
        "flags": ALLOWED_FLAGS,
        "flag_definitions": FLAG_DEFINITIONS,
        "records": records,
    }
    manifest = {
        "schema_version": "2.0",
        "campaign_id": CAMPAIGN_ID,
        "rows": len(records),
        "videos": len({(row["cohort"], row["video_id"]) for row in records}),
        "cohort_counts": dict(Counter(row["cohort"] for row in records)),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "cohort_metadata": cohort_metadata,
        "test_id_overlap": 0,
        "source_sha256": {
            "canonical_original": sha256_file(CANONICAL_PATH),
            "candidates_original": sha256_file(CANDIDATES_PATH),
            "pro_original_2000": sha256_file(PRO_PATH),
            **{
                f"{spec['batch_id']}__{kind}": sha256_file(path)
                for spec in EXPANSION_SPECS
                for kind, path in (
                    ("chunks", spec["chunks_path"]),
                    ("pending", spec["pending_path"]),
                    ("usable", spec["usable_path"]),
                    ("manifest", spec["manifest_path"]),
                )
            },
            "frontend": sha256_file(HTML_PATH),
        },
        "chunk_ids": [row["chunk_id"] for row in records],
    }
    if CAMPAIGN_MANIFEST_PATH.exists():
        previous = json.loads(CAMPAIGN_MANIFEST_PATH.read_text(encoding="utf-8"))
        previous_ids = previous.get("chunk_ids", [])
        if previous_ids != manifest["chunk_ids"][: len(previous_ids)]:
            raise ValueError(
                "La actualización no es append-only: cambió la selección u orden previo."
            )
    _write_json_atomic(CAMPAIGN_PATH, campaign)
    _write_json_atomic(CAMPAIGN_MANIFEST_PATH, manifest)
    if not PROGRESS_PATH.exists():
        _write_json_atomic(PROGRESS_PATH, _migrate_legacy_progress(campaign))
    write_trace_report(campaign, load_progress(), manifest)
    return {"campaign": campaign, "manifest": manifest}


def load_campaign() -> dict:
    if not CAMPAIGN_PATH.exists():
        return prepare_campaign()["campaign"]
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    if campaign.get("campaign_id") != CAMPAIGN_ID or len(campaign.get("records", [])) != EXPECTED_ROWS:
        raise ValueError("El archivo de campaña no coincide con la versión combinada esperada.")
    return campaign


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return _migrate_legacy_progress(load_campaign())
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    if progress.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("El progreso pertenece a otra campaña.")
    if not isinstance(progress.get("annotations"), list):
        raise ValueError("annotations del progreso debe ser una lista.")
    _unique_by_id(progress["annotations"], "progreso humano combinado")
    return progress


def validate_annotation(row: dict, campaign: dict, allow_draft: bool = True) -> dict:
    if not isinstance(row, dict):
        raise ValueError("La anotación debe ser un objeto JSON.")
    record_by_id = {record["chunk_id"]: record for record in campaign["records"]}
    chunk_id = str(row.get("chunk_id") or "")
    if chunk_id not in record_by_id:
        raise ValueError("chunk_id fuera de la campaña.")
    record = record_by_id[chunk_id]
    status = str(row.get("status") or "draft")
    allowed_statuses = {"draft", "completed", "deferred"} if allow_draft else {"completed"}
    if status not in allowed_statuses:
        raise ValueError(f"status inválido: {status}")
    action = str(row.get("review_action") or "") or None
    if action is not None and action not in ALLOWED_ACTIONS:
        raise ValueError(f"review_action inválida: {action}")
    if status == "completed" and action not in ALLOWED_ACTIONS:
        raise ValueError("Una decisión completa exige aceptar, rechazar o modificar la propuesta.")

    labels = row.get("coarse_labels") or []
    flags = row.get("flags") or []
    if action == "accept_llm" and status == "completed":
        labels = list(record["pro_reference"]["coarse_labels"])
        flags = list(record["pro_reference"]["flags"])
    elif action == "reject_llm" and status == "completed":
        labels, flags = [], []
    if not isinstance(labels, list) or len(labels) != len(set(labels)):
        raise ValueError("coarse_labels debe ser una lista sin duplicados.")
    if not set(labels) <= set(COARSE_ORDER):
        raise ValueError("coarse_labels contiene valores fuera de taxonomía.")
    if not isinstance(flags, list) or len(flags) != len(set(flags)):
        raise ValueError("flags debe ser una lista sin duplicados.")
    if not set(flags) <= set(ALLOWED_FLAGS):
        raise ValueError("flags contiene valores fuera de taxonomía.")
    safe = "SEGURO" in labels
    damage = bool(set(labels) & set(DAMAGE_ORDER))
    if safe and damage:
        raise ValueError("SEGURO no puede coexistir con daño.")
    if flags and not damage:
        raise ValueError("Los flags transversales requieren una categoría de daño.")
    if status == "completed" and action != "reject_llm" and not labels:
        raise ValueError("Aceptar o modificar requiere al menos una categoría base.")
    annotator_id = str(row.get("annotator_id") or "").strip().upper()[:3]
    if status in {"completed", "deferred"} and not re.fullmatch(r"[A-Z0-9]{3}", annotator_id):
        raise ValueError("Las iniciales deben tener exactamente tres caracteres A-Z/0-9.")
    notes = str(row.get("notes") or "").strip()[:1000]
    if status == "deferred" and not notes:
        raise ValueError("Explica en notas por qué el caso queda pendiente.")
    training_eligible = status == "completed" and action != "reject_llm"
    label_origin = {
        "accept_llm": "pro_llm_accepted_by_human",
        "reject_llm": "excluded_after_human_rejection",
        "modify_llm": "human_modified",
        "legacy_human_decision": "legacy_human_decision",
    }.get(action) if status == "completed" else None
    return {
        "campaign_id": CAMPAIGN_ID,
        "chunk_id": chunk_id,
        "coarse_labels": [label for label in COARSE_ORDER if label in labels],
        "flags": [flag for flag in ALLOWED_FLAGS if flag in flags],
        "status": status,
        "review_action": action,
        "training_eligible": training_eligible,
        "label_origin": label_origin,
        "needs_review": status != "completed",
        "notes": notes,
        "annotator_type": "human",
        "annotator_id": annotator_id,
        "pro_revealed": True,
        "annotated_at": str(row.get("annotated_at") or now_iso()),
    }


def _materialize_output(
    campaign: dict,
    annotations: list[dict],
    output_path: Path,
    manifest_path: Path,
    output_id: str,
) -> None:
    record_by_id = {row["chunk_id"]: row for row in campaign["records"]}
    final_rows: list[dict] = []
    for annotation in annotations:
        row = validate_annotation(annotation, campaign, allow_draft=False)
        record = record_by_id[row["chunk_id"]]
        row.update(
            {
                "cohort": record["cohort"],
                "source_run": record["source_run"],
                "split": record["split"],
                "source_annotation": row["label_origin"],
            }
        )
        final_rows.append(row)
    _write_jsonl_atomic(output_path, final_rows)
    manifest = {
        "schema_version": "2.0",
        "campaign_id": CAMPAIGN_ID,
        "output_id": output_id,
        "completed": len(final_rows),
        "included_for_training": sum(row["training_eligible"] for row in final_rows),
        "excluded_from_training": sum(not row["training_eligible"] for row in final_rows),
        "action_counts": dict(Counter(row["review_action"] for row in final_rows)),
        "cohort_counts": dict(Counter(row["cohort"] for row in final_rows)),
        "split_counts": dict(Counter(row["split"] for row in final_rows)),
        "created_at": now_iso(),
        "coarse_labels_only": True,
        "fine_labels_trained": False,
        "transversal_flags_separate": True,
        "source_campaign_sha256": sha256_file(CAMPAIGN_PATH),
        "source_events_sha256": sha256_file(EVENTS_PATH) if EVENTS_PATH.exists() else None,
        "legacy_events_sha256": sha256_file(LEGACY_EVENTS_PATH) if LEGACY_EVENTS_PATH.exists() else None,
        "output_sha256": sha256_file(output_path),
        "annotator_ids": sorted({row["annotator_id"] for row in final_rows}),
    }
    _write_json_atomic(manifest_path, manifest)


def _finalize_ready_outputs(campaign: dict, progress: dict) -> dict[str, bool]:
    by_id = _unique_by_id(progress["annotations"], "progreso humano combinado")
    records_by_cohort = {
        cohort: [row for row in campaign["records"] if row["cohort"] == cohort]
        for cohort in COHORTS
    }
    ready: dict[str, bool] = {}
    destinations = {
        "original_139": (ORIGINAL_FINAL_PATH, ORIGINAL_FINAL_MANIFEST_PATH),
        **{
            spec["cohort"]: (spec["final_path"], spec["final_manifest_path"])
            for spec in EXPANSION_SPECS
        },
    }
    for cohort, records in records_by_cohort.items():
        annotations = [by_id.get(record["chunk_id"]) for record in records]
        cohort_ready = all(row is not None and row.get("status") == "completed" for row in annotations)
        ready[cohort] = cohort_ready
        if cohort_ready:
            output_path, manifest_path = destinations[cohort]
            _materialize_output(campaign, annotations, output_path, manifest_path, cohort)
    combined_annotations = [by_id.get(record["chunk_id"]) for record in campaign["records"]]
    ready["combined"] = all(
        row is not None and row.get("status") == "completed" for row in combined_annotations
    )
    if ready["combined"]:
        _materialize_output(
            campaign,
            combined_annotations,
            COMBINED_FINAL_PATH,
            COMBINED_FINAL_MANIFEST_PATH,
            f"combined_{len(campaign['records'])}",
        )
    return ready


def save_annotation(annotation: dict) -> dict:
    campaign = load_campaign()
    validated = validate_annotation(annotation, campaign, allow_draft=True)
    with _write_lock:
        progress = load_progress()
        by_id = _unique_by_id(progress["annotations"], "progreso humano combinado")
        previous = by_id.get(validated["chunk_id"])
        revision = int(progress.get("revision", 0)) + 1
        saved_at = now_iso()
        validated["revision"] = int(previous.get("revision", 0)) + 1 if previous else 1
        validated["saved_at"] = saved_at
        by_id[validated["chunk_id"]] = validated
        order = [row["chunk_id"] for row in campaign["records"]]
        progress = {
            **{key: value for key, value in progress.items() if key not in {"annotations", "revision", "updated_at"}},
            "campaign_id": CAMPAIGN_ID,
            "updated_at": saved_at,
            "revision": revision,
            "annotations": [by_id[chunk_id] for chunk_id in order if chunk_id in by_id],
        }
        _write_json_atomic(PROGRESS_PATH, progress)
        record = next(row for row in campaign["records"] if row["chunk_id"] == validated["chunk_id"])
        event = {
            "campaign_id": CAMPAIGN_ID,
            "event_revision": revision,
            "saved_at": saved_at,
            "chunk_id": validated["chunk_id"],
            "cohort": record["cohort"],
            "previous_status": previous.get("status") if previous else None,
            "previous_action": previous.get("review_action") if previous else None,
            "new_status": validated["status"],
            "review_action": validated["review_action"],
            "training_eligible": validated["training_eligible"],
            "label_origin": validated["label_origin"],
            "annotation_revision": validated["revision"],
            "coarse_labels": validated["coarse_labels"],
            "flags": validated["flags"],
            "annotator_id": validated["annotator_id"],
        }
        with EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            file.flush()
            os.fsync(file.fileno())
        finalized = _finalize_ready_outputs(campaign, progress)
        manifest = json.loads(CAMPAIGN_MANIFEST_PATH.read_text(encoding="utf-8"))
        write_trace_report(campaign, progress, manifest)
    return {
        "saved": validated,
        "summary": progress_summary(progress, campaign),
        "finalized": finalized,
    }


def progress_summary(progress: dict, campaign: dict | None = None) -> dict:
    campaign = campaign or load_campaign()
    annotations = progress.get("annotations", [])
    by_id = {row["chunk_id"]: row for row in annotations}
    statuses = Counter(row.get("status", "draft") for row in annotations)
    actions = Counter(
        row.get("review_action") for row in annotations if row.get("status") == "completed"
    )
    cohort_summary = {}
    campaign_cohorts = campaign.get("cohorts", COHORTS)
    for cohort, definition in campaign_cohorts.items():
        ids = [row["chunk_id"] for row in campaign["records"] if row["cohort"] == cohort]
        completed_rows = [by_id[chunk_id] for chunk_id in ids if by_id.get(chunk_id, {}).get("status") == "completed"]
        cohort_summary[cohort] = {
            "label": definition["label"],
            "total": len(ids),
            "completed": len(completed_rows),
            "remaining": len(ids) - len(completed_rows),
            "included": sum(row.get("training_eligible") is True for row in completed_rows),
            "excluded": sum(row.get("training_eligible") is False for row in completed_rows),
            "action_counts": dict(Counter(row.get("review_action") for row in completed_rows)),
        }
    completed = statuses["completed"]
    total = len(campaign["records"])
    return {
        "total": total,
        "completed": completed,
        "deferred": statuses["deferred"],
        "draft": statuses["draft"],
        "untouched": total - len(annotations),
        "included": sum(row.get("training_eligible") is True for row in annotations if row.get("status") == "completed"),
        "excluded": sum(row.get("training_eligible") is False for row in annotations if row.get("status") == "completed"),
        "action_counts": dict(actions),
        "remaining_for_final": total - completed,
        "progress_pct": round(100 * completed / total, 2),
        "final_ready": completed == total,
        "cohorts": cohort_summary,
        "revision": int(progress.get("revision", 0)),
        "updated_at": progress.get("updated_at"),
    }


def write_trace_report(campaign: dict, progress: dict, manifest: dict) -> None:
    summary = progress_summary(progress, campaign)
    status = "COMPLETADA" if summary["final_ready"] else "EN CURSO"
    cohort_table_rows = []
    for cohort, values in summary["cohorts"].items():
        cohort_table_rows.append(
            f"| {values['label']} | {values['total']:,} | {values['completed']:,} | "
            f"{values['remaining']:,} | {values['included']:,} | {values['excluded']:,} |"
        )
    expansion_artifact_rows = []
    for spec in EXPANSION_SPECS:
        pending_key = f"{spec['batch_id']}__pending"
        expansion_artifact_rows.append(
            f"- Cola `{spec['batch_id']}`: `{spec['pending_path'].relative_to(ROOT)}`; "
            f"SHA-256 `{manifest['source_sha256'][pending_key]}`."
        )
    lines = [
        f"# Informe de adjudicación humana combinada: {summary['total']:,} casos",
        "",
        f"Estado: **{status}**  ",
        f"Campaña: `{CAMPAIGN_ID}`  ",
        f"Actualización: {summary['updated_at'] or 'sin decisiones nuevas'}  ",
        "",
        "## Alcance y procedencia",
        "",
        f"La campaña integra de forma append-only la cola original y {len(EXPANSION_SPECS)} colas de ampliación descubiertas automáticamente. Los manifiestos verifican procedencia, orden y hashes; ningún caso de ampliación pertenece a test. Total actual: **{summary['total']:,}**.",
        "",
        "| Cohorte | Total | Resueltos | Faltan | Incluidos | Excluidos |",
        "|---|---:|---:|---:|---:|---:|",
        *cohort_table_rows,
        f"| **Total** | **{summary['total']:,}** | **{summary['completed']:,}** | **{summary['remaining_for_final']:,}** | **{summary['included']:,}** | **{summary['excluded']:,}** |",
        "",
        "## Regla operativa y trazabilidad",
        "",
        "La propuesta de `deepseek-v4-pro` se presenta antes de decidir para acelerar la adjudicación. Cada clic queda registrado con fecha, iniciales, cohorte, revisión y acción:",
        "",
        "- `accept_llm`: copia en servidor las categorías gruesas y flags de Pro; `training_eligible=true`.",
        "- `reject_llm`: vacía categorías y flags; `training_eligible=false`; el chunk queda fuera del entrenamiento.",
        "- `modify_llm`: conserva la versión gruesa elegida por el humano y sus flags; `training_eligible=true`.",
        "- `legacy_human_decision`: preserva como tal una decisión realizada antes de introducir los botones rápidos; no se afirma retrospectivamente que fue un clic de aceptación.",
        "",
        "Las etiquetas finas de Pro se muestran solo como contexto y nunca son objetivos de entrenamiento. Los flags transversales se mantienen separados de las categorías base.",
        "",
        "## Estado de las acciones",
        "",
        f"- Aceptaciones explícitas: {summary['action_counts'].get('accept_llm', 0):,}.",
        f"- Rechazos/exclusiones: {summary['action_counts'].get('reject_llm', 0):,}.",
        f"- Modificaciones humanas: {summary['action_counts'].get('modify_llm', 0):,}.",
        f"- Decisiones humanas migradas: {summary['action_counts'].get('legacy_human_decision', 0):,}.",
        f"- Borradores: {summary['draft']:,}; diferidos: {summary['deferred']:,}; sin abrir: {summary['untouched']:,}.",
        "",
        "## Integración con entrenamiento",
        "",
        f"Los archivos finales incluyen todas las decisiones para conservar la auditoría. Los pipelines filtran explícitamente `training_eligible=false`: aceptar usa la versión Pro, modificar usa la versión humana y rechazar elimina el chunk. Cada cohorte se materializa cuando termina; la salida combinada se genera al completar las {summary['total']:,} decisiones vigentes.",
        "",
        "## Artefactos reproducibles",
        "",
        f"- Campaña: `{CAMPAIGN_PATH.relative_to(ROOT)}`.",
        f"- Manifiesto: `{CAMPAIGN_MANIFEST_PATH.relative_to(ROOT)}`.",
        f"- Progreso: `{PROGRESS_PATH.relative_to(ROOT)}`.",
        f"- Eventos: `{EVENTS_PATH.relative_to(ROOT)}`.",
        f"- Salida original: `{ORIGINAL_FINAL_PATH.relative_to(ROOT)}`.",
        f"- Salida combinada: `{COMBINED_FINAL_PATH.relative_to(ROOT)}`.",
        f"- Frontend: `{HTML_PATH.relative_to(ROOT)}`.",
        f"- SHA-256 cola original Pro: `{manifest['source_sha256']['pro_original_2000']}`.",
        *expansion_artifact_rows,
        "",
        "Inicio desde la raíz:",
        "",
        "```powershell",
        "python -m scripts_auxiliares.servidor_revision_humana_139 --host 127.0.0.1 --port 8765",
        "```",
        "",
        "## Limitación metodológica",
        "",
        "Mostrar la propuesta antes de la decisión aumenta velocidad, pero puede producir anclaje. Por ello estas decisiones sirven para depurar y construir entrenamiento, no para estimar de manera ciega e independiente el error de Pro. Una medición de acuerdo humano–LLM requiere una submuestra ciega o doble anotación independiente.",
        "",
        "## Referencias (APA 7)",
        "",
        "Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics, 34*(4), 555–596. https://doi.org/10.1162/coli.07-034-R2",
        "",
        "Brodley, C. E., & Friedl, M. A. (1999). Identifying mislabeled training data. *Journal of Artificial Intelligence Research, 11*, 131–167. https://doi.org/10.1613/jair.606",
        "",
        "Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison. https://research.cs.wisc.edu/techreports/2009/TR1648.pdf",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "HumanReviewCombined/2.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _send_bytes(
        self,
        payload: bytes,
        content_type: str,
        status: int = 200,
        filename: str | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # El navegador puede cancelar una descarga grande al recargar o
            # cerrar una pestaña; no es un fallo de campaña ni de persistencia.
            return

    def _send_json(self, value: object, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self._send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._send_json(
                    {"ok": True, "campaign_id": CAMPAIGN_ID, "pid": os.getpid(), "rows": EXPECTED_ROWS}
                )
            elif path == "/api/campaign":
                self._send_json(load_campaign())
            elif path == "/api/progress":
                campaign, progress = load_campaign(), load_progress()
                self._send_json({"progress": progress, "summary": progress_summary(progress, campaign)})
            elif path == "/api/export" and COMBINED_FINAL_PATH.exists():
                self._send_bytes(
                    COMBINED_FINAL_PATH.read_bytes(),
                    "application/x-ndjson; charset=utf-8",
                    filename=COMBINED_FINAL_PATH.name,
                )
            elif path == "/api/export/original" and ORIGINAL_FINAL_PATH.exists():
                self._send_bytes(
                    ORIGINAL_FINAL_PATH.read_bytes(),
                    "application/x-ndjson; charset=utf-8",
                    filename=ORIGINAL_FINAL_PATH.name,
                )
            elif path == "/api/export/expansion" and EXPANSION_FINAL_PATH.exists():
                self._send_bytes(
                    EXPANSION_FINAL_PATH.read_bytes(),
                    "application/x-ndjson; charset=utf-8",
                    filename=EXPANSION_FINAL_PATH.name,
                )
            elif path.startswith("/api/export"):
                self._send_json({"error": "La salida solicitada aún no está completa."}, HTTPStatus.CONFLICT)
            else:
                self._send_json({"error": "Ruta no encontrada."}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self._send_json({"error": "Ruta no encontrada."}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64 * 1024:
                raise ValueError("Tamaño de solicitud inválido.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send_json(save_annotation(payload.get("annotation", payload)))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    prepared = prepare_campaign()
    server_state = {
        "pid": os.getpid(),
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}/",
        "campaign_id": CAMPAIGN_ID,
        "started_at": now_iso(),
        "rows": len(prepared["campaign"]["records"]),
        "cohort_counts": prepared["manifest"]["cohort_counts"],
    }
    _write_json_atomic(PID_PATH, server_state)
    print(json.dumps(server_state, ensure_ascii=False, indent=2))
    server = ThreadingHTTPServer((host, port), ReviewHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if args.prepare_only:
        result = prepare_campaign()
        print(
            json.dumps(
                {"campaign": str(CAMPAIGN_PATH), **result["manifest"]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if args.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("Por seguridad, esta campaña solo puede enlazarse a localhost.")
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
