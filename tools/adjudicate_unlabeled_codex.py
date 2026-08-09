"""Auditoría reproducible de chunks elegibles sin etiqueta.

La herramienta conserva las inferencias remotas como evidencia separada y solo
materializa eventos de revisión cuando se ejecuta ``finalize``.  El modo
``run-broad`` vuelve a consultar con el prompt 3.1 los daños cuya justificación
todavía contiene señales amplias de noticia, cita, relato o testimonio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from moderacion_peru.io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from moderacion_peru.paths import find_project_root
from moderacion_peru.providers import DeepSeekProvider
from moderacion_peru.schemas import ReviewEvent


ROOT = find_project_root()
CASCADE = ROOT / "datos" / "etiquetado" / "cascada_deepseek_v4"
CAMPAIGN = ROOT / "datos" / "etiquetado" / "consolidado" / "anotaciones_v2.jsonl"
PROMPT_V31 = ROOT / "config" / "prompt_operacional_ollama_v3_1.md"
V3 = CASCADE / "codex_unlabeled_prompt_v3_flash.jsonl"
HARD = CASCADE / "codex_unlabeled_prompt_v3_1_pro_hard.jsonl"
STANCE = CASCADE / "codex_unlabeled_prompt_v3_1_pro_stance.jsonl"
BROAD = CASCADE / "codex_unlabeled_prompt_v3_1_pro_broad_stance.jsonl"
LOW_RISK_SAMPLE = CASCADE / "codex_unlabeled_prompt_v3_1_1_flash_low_risk_sample.jsonl"
LOW_RISK_HARD = CASCADE / "codex_unlabeled_prompt_v3_1_1_pro_low_risk_hard.jsonl"
PRIMARY = CASCADE / "primary_flash.jsonl"
REVIEWS = ROOT / "datos" / "etiquetado" / "humano" / "labeling_events_v2.jsonl"
EVENT_SNAPSHOT = (
    ROOT
    / "datos"
    / "etiquetado"
    / "humano"
    / "codex_unlabeled_adjudication_v3_1.events.jsonl"
)
MANIFEST = EVENT_SNAPSHOT.with_suffix(".manifest.json")

DAMAGE = {
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
}

# Esta lista es deliberadamente amplia: selecciona para una segunda opinión,
# no decide por sí sola una etiqueta.
BROAD_ATTRIBUTION = re.compile(
    r"\b(?:narra(?:dor|dora|ción)?|describe|descripción|reporta|reportaje|"
    r"noticiero|periodista|conductor(?:a)?|programa|entrevista|anécdota|"
    r"testigo|testimonio|víctima|denuncia|informando|relata|historia|"
    r"reproduce|audio|mensaje|cita|atribuido|según)\b",
    re.IGNORECASE,
)

# Overrides puntuales leídos por CODEX. Solo contienen casos donde la evidencia
# y la justificación permiten resolver con suficiente certeza una contradicción
# respecto del veto de atribución o del alcance exacto de una categoría.
SAFE_OVERRIDES = {
    "wWwXhl6JWjM_563149a3fa3e86c44541",
    "vFHIBiV65Us_e96ea94dce8bd8d41b0d",
    "kTe7fV8RRAk_e2840e3c9f625f53ca6e",
    "v4U0u96pVZU_123d1c4f3ba873656c3c",
    "rHZNbmZXOa4_67b1207d1b35245d0819",
    "sUYBSxp57iA_5245eff1d79415952c67",
    "DuVIST1fkf8_27aaf956bde1b8ae6d1a",
    "xkJyIr-Jaow_d53f68d68632f337e371",
    "E9Bk9-kPFnA_61050ceff0a8003a06f8",
    "Qa_L0frq7yk_327fb9207d9e2c4a1fac",
    "sSIFE8fdXP0_e1276abb0a8d0632ca1c",
    "sl5AwvwUIng_1815455000174fddafde",
    "wKmHy37yGwQ_7a1aaabe8b66c4282059",
    "755FNhj_83s_ea1e25ff4a0418813c44",
    "xXB86WbaNMA_288a12783aea1085b993",
    "Bgd5AZlq0RA_0ef6026de33380e24f67",
    "YlzMyzpuLyk_fbf2c920d1509c2f9ba3",
    "GGxay3dQlJ0_7b5e16f6f222a261b7c3",
    "5jsZw8kKaAo_23512f2825c56754a80d",
    "IaR_5fcW90s_0f8464ba0240a3a35a78",
    "uCWTs4X-gag_d2dd975918c3323f9292",
    "DuVIST1fkf8_b1a8bdda15a85b40e243",
    "Hlu5jBgxPiA_02d86e7f88289b8c04a0",
    "qsKQ7WMpXJg_efbd2ed2ebfde4a92954",
    "Dul46A3JVIs_c240118e8758f1304ad7",
    "Dul46A3JVIs_a9c3b86b67d9596e2be7",
    "DuVIST1fkf8_f4ee261b14ddff6bef1f",
    "kw69ldp5nnw_a66aac1845beda9e5baa",
    "eKm6nE_l8uQ_1b14d6c903de5193e20d",
    "YBFcNbxFE30_750cb60d876f71a700a9",
    "7NzKOER_eu4_551e6522958eb62962dc",
    "7uFcjt5g6TQ_e16ed90bb67ad350de6d",
    "FdHr6fxYJ6Y_3189c7d79daea0d9a41f",
    "h7qCDmC4KR0_860aeccfbad2e9400eba",
    "Dn0kVNAelqI_78ba75b84261ca5c4ffe",
    "UtBCgbmV7Yg_20706e172ff7648c7482",
    "zYD1whssXUc_e443785ee38b309c09aa",
    "3_CQGcYj_pg_aa14d0471f5b3097e9db",
    "6UMSpPadM5o_18588fbcd7e3f72b477d",
    "fMZMLkdRy2U_259f61e6aadeddb6195a",
    "V5loRHC6NqA_ed1f5f7f8a5c765092b9",
    "AHrTQZkSnEQ_84e4fec94c549af06f0b",
    "8JUn4rCsg7c_5365c3d38522b8af0658",
    "vqkFmGKFzLM_b7f7d4e6d59e3061be41",
    "azEFdCXdw9E_8a5acda25a285ac1381a",
    "RQegO7hl_x0_7a5fc9726fc288ba5243",
    "lxNY7_rAcn8_22794396ba21b233edcd",
    "TEHRF1CawT4_f6d382c1e3417a5f0263",
    "E9Bk9-kPFnA_4ec38907963f52d750cb",
    "3xsXVCYIHno_d2cf8393c04e3327871d",
    "Rh_SFRiDRQ0_c6c391c5f608a32f54cd",
    "y5DYSxHalo8_90265910f6f14cee6fa4",
    "udXis1gUzP4_3c1ea189e642664705dc",
    "SmZXPB-54q4_60a6aec4eab2a2679f81",
    "pYvtrJBaQCs_438e710dba7c458f476e",
    "6QBXZeiJCSU_2ce335a90a4ac2a70d02",
    "Z32-Mw7qwjA_b8673fb3f4a941cd302f",
    "POBTRfF3Fl0_69d7caab3d7544803d94",
    "MIkhrxK1bR8_d4c437b1638638b0f1c7",
    "Nq456oXnvfs_0d7bf753ee1e0906d9b7",
    "AwVjrTSoUpg_62c82875e219433621b5",
    "xru0C_J5Rus_28f175051097c6c7ae34",
    "J8tXA9rBn7s_12c272058199af80a78d",
    "RQegO7hl_x0_912c1a1eebf163148077",
    "1MGd2qZpGPk_3e45df5e4c36cc278838",
    "gQS15G4BLNg_fbf9c7b5536cbafb3009",
    "pOYNTD4peAg_84bfa844a10a76a7c4e1",
    # Segunda opinión Pro de la muestra ciega: la etiqueta dependía de contexto
    # vecino no transferible o de una inferencia no presente en el chunk.
    "YgxWiYhPqqs_6a9526ce27cc572ea987",
    "aEUnj7q6MCc_3e40392df5ddd11b4b51",
    "0k9gUF-Y2uk_01d7ebcb5101ed65ca08",
    "Lcg-YRxSAuA_d12c2d08a8153fb99d71",
    "3UGHL-qis-c_e74e98b72c05ad41e41c",
    "xXB86WbaNMA_1656f08e8c4913eb99d7",
}

PARTIAL_OVERRIDES = {
    # La atribución elimina daño no sexual, pero se conserva la excepción sexual.
    "sSIFE8fdXP0_c33582b9268129043100": ["CONTENIDO_SEXUAL"],
    "PpXRfyP46hQ_24f4f59756511b49380d": ["CONTENIDO_SEXUAL"],
    "bZdYBmSREw8_c5b70802e422fc6b5581": ["CONTENIDO_SEXUAL"],
    "E9Bk9-kPFnA_d71946775ea213289be4": ["CONTENIDO_SEXUAL"],
    "uVz0MzMeWzw_1c813688e22227e00069": ["CONTENIDO_SEXUAL"],
    "Duow4OxcbB0_87ceb4e93cd3d4541054": ["CONTENIDO_SEXUAL"],
    # El hablante añade un ataque propio, pero el rasgo sexual/racial propuesto
    # no alcanza el umbral del contrato.
    "U-x205F7ISc_fa2dc2f5ed99d974e315": ["ACOSO_AMENAZA"],
    "mLPi4x-ykD0_b4b011289b527c680620": ["ACOSO_AMENAZA"],
    "Po0vdJnve_A_5c4ae8f0bdeaf3c2fb08": ["ACOSO_AMENAZA"],
    # "Caviar" es una posición política en este uso; no prueba
    # racialización. Se mantiene solo el ataque personal propio.
    "rjZFlgp6Peg_dc1179c5e526e27c3917": ["ACOSO_AMENAZA"],
}

LABEL_ORDER = [
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
]


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(row["chunk_id"]): row for row in read_jsonl(path)}


def broad_candidates() -> list[dict[str, Any]]:
    v3 = rows_by_id(V3)
    already = set(rows_by_id(HARD)) | set(rows_by_id(STANCE)) | set(rows_by_id(BROAD))
    candidates = [
        row
        for chunk_id, row in v3.items()
        if chunk_id not in already
        and set(row.get("coarse_labels") or []).intersection(DAMAGE)
        and BROAD_ATTRIBUTION.search(str(row.get("justification") or ""))
    ]
    return sorted(candidates, key=lambda row: str(row["chunk_id"]))


def with_adjacent_context(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    campaign = list(read_jsonl(CAMPAIGN))
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in campaign:
        by_video.setdefault(str(row.get("video_id") or ""), []).append(row)
    neighbors: dict[str, tuple[str, str]] = {}
    for rows in by_video.values():
        rows.sort(key=lambda row: (float(row.get("start_seconds") or 0), str(row["chunk_id"])))
        for index, row in enumerate(rows):
            previous = str(rows[index - 1].get("text") or "") if index else ""
            following = str(rows[index + 1].get("text") or "") if index + 1 < len(rows) else ""
            neighbors[str(row["chunk_id"])] = (previous[:1200], following[:1200])

    prepared = []
    for row in candidates:
        item = dict(row)
        previous, following = neighbors.get(str(row["chunk_id"]), ("", ""))
        if previous:
            item["contexto_anterior"] = previous
        if following:
            item["contexto_posterior"] = following
        prepared.append(item)
    return prepared


def _error_row(chunk: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "error_type": type(error).__name__,
        "error": str(error),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_broad(max_cost_usd: float) -> None:
    selected = broad_candidates()
    provider = DeepSeekProvider(
        model="deepseek-v4-pro",
        max_workers=16,
        records_per_request=10,
        retries=1,
        max_cost_usd=max_cost_usd,
        label_source="llm_remote_review",
        annotator_type="llm_remote",
        operational_prompt_path=PROMPT_V31,
    )
    run_path = BROAD.with_suffix(BROAD.suffix + ".run.json")
    errors_path = BROAD.with_suffix(".errors.jsonl")
    signature_payload = {
        "schema_version": "1.0.0",
        "reviewer": "CODEX",
        "method": "prompt_v3_1_broad_attribution_audit",
        "selection": "remaining_v3_harm_with_broad_narrative_or_attribution_signal",
        "selected": len(selected),
        "prompt_path": str(PROMPT_V31.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": sha256_file(PROMPT_V31),
        "model": provider.model,
    }
    signature_payload["run_signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json_atomic(run_path, signature_payload)

    prepared = with_adjacent_context(selected)
    started = time.monotonic()
    completed = 0
    errors = 0
    for group in provider.iter_annotate_batch(prepared):
        valid_rows = []
        error_rows = []
        for index, result in group:
            if isinstance(result, Exception):
                error_rows.append(_error_row(prepared[index], result))
            else:
                valid_rows.append(result.model_dump(mode="json"))
        if valid_rows:
            append_jsonl_once(BROAD, valid_rows, id_field="chunk_id")
        if error_rows:
            append_jsonl_once(errors_path, error_rows, id_field="chunk_id")
        completed += len(group)
        errors += len(error_rows)
        if completed % 100 == 0 or completed == len(prepared):
            print(f"progreso={completed}/{len(prepared)} errores={errors}", flush=True)

    elapsed = time.monotonic() - started
    output = rows_by_id(BROAD)
    counts = Counter(
        "+".join(row.get("coarse_labels") or []) or "SIN_DECISION"
        for row in output.values()
    )
    result = {
        "selected": len(selected),
        "persisted": len(output),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 3),
        "usage": provider.usage_summary(),
        "labels": dict(counts),
    }
    write_json_atomic(BROAD.with_suffix(".result.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def low_risk_candidates(sample_size: int) -> list[dict[str, Any]]:
    campaign = list(read_jsonl(CAMPAIGN))
    _, latest = _latest_reviews()
    covered = (
        set(rows_by_id(V3))
        | set(rows_by_id(HARD))
        | set(rows_by_id(STANCE))
        | set(rows_by_id(BROAD))
        | set(rows_by_id(LOW_RISK_SAMPLE))
    )
    primary = rows_by_id(PRIMARY)
    eligible = []
    for row in campaign:
        chunk_id = str(row["chunk_id"])
        review = latest.get(chunk_id)
        if (
            chunk_id in covered
            or _is_excluded(row, review)
            or _effective_labels(row, review)
        ):
            continue
        prior = primary.get(chunk_id)
        if prior and prior.get("coarse_labels"):
            continue
        eligible.append(row)

    # Muestreo determinista por modelo de origen, proporcional al universo. El
    # hash distribuye canales y posiciones sin depender del orden del JSONL.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        groups.setdefault(str(row.get("annotator_model") or "sin_modelo"), []).append(row)
    selected: list[dict[str, Any]] = []
    remaining = sample_size
    group_items = sorted(groups.items())
    for index, (_, rows) in enumerate(group_items):
        if index + 1 == len(group_items):
            allocation = remaining
        else:
            allocation = round(sample_size * len(rows) / max(1, len(eligible)))
            allocation = min(allocation, remaining)
        rows.sort(
            key=lambda row: hashlib.sha256(str(row["chunk_id"]).encode("utf-8")).hexdigest()
        )
        selected.extend(rows[:allocation])
        remaining -= allocation
    return sorted(selected, key=lambda row: str(row["chunk_id"]))


def run_low_risk_sample(sample_size: int, max_cost_usd: float) -> None:
    selected = low_risk_candidates(sample_size)
    provider = DeepSeekProvider(
        model="deepseek-v4-flash",
        max_workers=16,
        records_per_request=10,
        retries=1,
        max_cost_usd=max_cost_usd,
        label_source="deepseek_remote",
        annotator_type="llm_remote",
        operational_prompt_path=PROMPT_V31,
    )
    run_path = LOW_RISK_SAMPLE.with_suffix(LOW_RISK_SAMPLE.suffix + ".run.json")
    errors_path = LOW_RISK_SAMPLE.with_suffix(".errors.jsonl")
    metadata = {
        "schema_version": "1.0.0",
        "reviewer": "CODEX",
        "method": "prompt_v3_1_1_deterministic_low_risk_sample",
        "selection": "unlabeled_nonexcluded_without_prior_label_or_directed_risk_signal",
        "selected": len(selected),
        "population": 27187,
        "prompt_path": str(PROMPT_V31.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": sha256_file(PROMPT_V31),
        "model": provider.model,
    }
    metadata["run_signature"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json_atomic(run_path, metadata)

    prepared = with_adjacent_context(selected)
    started = time.monotonic()
    completed = errors = 0
    for group in provider.iter_annotate_batch(prepared):
        valid_rows = []
        error_rows = []
        for index, result in group:
            if isinstance(result, Exception):
                error_rows.append(_error_row(prepared[index], result))
            else:
                valid_rows.append(result.model_dump(mode="json"))
        if valid_rows:
            append_jsonl_once(LOW_RISK_SAMPLE, valid_rows, id_field="chunk_id")
        if error_rows:
            append_jsonl_once(errors_path, error_rows, id_field="chunk_id")
        completed += len(group)
        errors += len(error_rows)
        if completed % 250 == 0 or completed == len(prepared):
            print(f"progreso={completed}/{len(prepared)} errores={errors}", flush=True)

    elapsed = time.monotonic() - started
    output = rows_by_id(LOW_RISK_SAMPLE)
    result = {
        "population": 27187,
        "selected": len(selected),
        "sample_fraction": round(len(selected) / 27187, 6),
        "persisted": len(output),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 3),
        "usage": provider.usage_summary(),
        "labels": dict(
            Counter(
                "+".join(row.get("coarse_labels") or []) or "SIN_DECISION"
                for row in output.values()
            )
        ),
    }
    write_json_atomic(LOW_RISK_SAMPLE.with_suffix(".result.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_low_risk_hard(max_cost_usd: float) -> None:
    flash = rows_by_id(LOW_RISK_SAMPLE)
    existing = rows_by_id(LOW_RISK_HARD)
    selected = [
        row
        for chunk_id, row in flash.items()
        if chunk_id not in existing
        and (
            not row.get("coarse_labels")
            or set(row.get("coarse_labels") or []).intersection(DAMAGE)
        )
    ]
    selected.sort(key=lambda row: str(row["chunk_id"]))
    provider = DeepSeekProvider(
        model="deepseek-v4-pro",
        max_workers=16,
        records_per_request=10,
        retries=1,
        max_cost_usd=max_cost_usd,
        label_source="llm_remote_review",
        annotator_type="llm_remote",
        operational_prompt_path=PROMPT_V31,
    )
    run_path = LOW_RISK_HARD.with_suffix(LOW_RISK_HARD.suffix + ".run.json")
    errors_path = LOW_RISK_HARD.with_suffix(".errors.jsonl")
    metadata = {
        "schema_version": "1.0.0",
        "reviewer": "CODEX",
        "method": "prompt_v3_1_1_low_risk_hard_adjudication",
        "selection": "low_risk_sample_harm_or_abstention",
        "selected": len(selected),
        "prompt_path": str(PROMPT_V31.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": sha256_file(PROMPT_V31),
        "model": provider.model,
    }
    metadata["run_signature"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json_atomic(run_path, metadata)

    prepared = with_adjacent_context(selected)
    started = time.monotonic()
    completed = errors = 0
    for group in provider.iter_annotate_batch(prepared):
        valid_rows = []
        error_rows = []
        for index, result in group:
            if isinstance(result, Exception):
                error_rows.append(_error_row(prepared[index], result))
            else:
                valid_rows.append(result.model_dump(mode="json"))
        if valid_rows:
            append_jsonl_once(LOW_RISK_HARD, valid_rows, id_field="chunk_id")
        if error_rows:
            append_jsonl_once(errors_path, error_rows, id_field="chunk_id")
        completed += len(group)
        errors += len(error_rows)
    elapsed = time.monotonic() - started
    output = rows_by_id(LOW_RISK_HARD)
    result = {
        "selected": len(selected),
        "persisted": len(output),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 3),
        "usage": provider.usage_summary(),
        "labels": dict(
            Counter(
                "+".join(row.get("coarse_labels") or []) or "SIN_DECISION"
                for row in output.values()
            )
        ),
    }
    write_json_atomic(LOW_RISK_HARD.with_suffix(".result.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _latest_reviews() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    events = list(read_jsonl(REVIEWS)) if REVIEWS.is_file() else []
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[str(event["chunk_id"])] = event
    return events, latest


def _is_excluded(row: dict[str, Any], review: dict[str, Any] | None) -> bool:
    if review is not None:
        return str(review.get("action") or "") == "reject"
    return str(row.get("decision_status") or "") == "excluded"


def _effective_labels(
    row: dict[str, Any], review: dict[str, Any] | None
) -> list[str]:
    if review is not None and review.get("action") in {"accept", "modify"}:
        return list(review.get("final_labels") or [])
    return list(row.get("coarse_labels") or [])


def _ordered(labels: Iterable[str]) -> list[str]:
    values = set(labels)
    if "SEGURO" in values:
        return ["SEGURO"]
    return [label for label in LABEL_ORDER if label in values]


def _support_maps() -> tuple[list[tuple[str, dict[str, dict[str, Any]]]], dict[str, dict[str, Any]]]:
    # El último elemento aplicable tiene precedencia.
    staged = [
        ("flash_v3_risk", rows_by_id(V3)),
        ("pro_v3_1_hard", rows_by_id(HARD)),
        ("pro_v3_1_stance", rows_by_id(STANCE)),
        ("pro_v3_1_broad_stance", rows_by_id(BROAD)),
        ("flash_v3_1_1_low_risk_sample", rows_by_id(LOW_RISK_SAMPLE)),
        ("pro_v3_1_1_low_risk_hard", rows_by_id(LOW_RISK_HARD)),
    ]
    return staged, rows_by_id(PRIMARY)


def build_decisions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    campaign = list(read_jsonl(CAMPAIGN))
    previous_events, latest = _latest_reviews()
    staged, primary = _support_maps()
    support_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    for method, mapping in staged:
        support_by_id.update({chunk_id: (method, row) for chunk_id, row in mapping.items()})

    targets = []
    for row in campaign:
        chunk_id = str(row["chunk_id"])
        review = latest.get(chunk_id)
        if _is_excluded(row, review):
            continue
        if not _effective_labels(row, review):
            targets.append(row)

    batch_id = "CODEX-UNLABELED-PROMPT-V3_1_1-20260809"
    created_at = datetime.now(timezone.utc)
    decisions: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    original_model_counts: Counter[str] = Counter()
    override_counts: Counter[str] = Counter()
    label_combinations: Counter[str] = Counter()

    for row in targets:
        chunk_id = str(row["chunk_id"])
        original_model_counts[str(row.get("annotator_model") or "sin_modelo")] += 1
        support_entry = support_by_id.get(chunk_id)
        support: dict[str, Any] | None = None
        if support_entry is not None:
            method, support = support_entry
            labels = _ordered(support.get("coarse_labels") or [])
        else:
            prior = primary.get(chunk_id)
            if prior and prior.get("coarse_labels") == ["SEGURO"]:
                method = "flash_v2_prior_safe_low_risk"
            elif prior and prior.get("coarse_labels"):
                method = "flash_v2_fallback_after_risk_screen"
            else:
                method = "codex_low_risk_safe_after_abstention"
            support = prior
            labels = _ordered((prior or {}).get("coarse_labels") or [])

        if chunk_id in SAFE_OVERRIDES:
            labels = ["SEGURO"]
            method = "codex_attribution_or_contract_override"
            override_counts["safe"] += 1
        elif chunk_id in PARTIAL_OVERRIDES:
            labels = list(PARTIAL_OVERRIDES[chunk_id])
            method = "codex_partial_contract_override"
            override_counts["partial"] += 1
        elif not labels:
            # El universo solicitado debe quedar cubierto por una categoría
            # gruesa. Sin evidencia suficiente de daño, CODEX adopta SEGURO.
            labels = ["SEGURO"]
            if method != "codex_low_risk_safe_after_abstention":
                method = "codex_conservative_safe_after_abstention"
            override_counts["abstention_to_safe"] += 1

        flags = list((support or {}).get("flags") or []) if labels != ["SEGURO"] else []
        labels = _ordered(labels)
        support_model = str((support or {}).get("annotator_model") or "") or None
        evidence_labels = list((support or {}).get("coarse_labels") or [])
        note = (
            f"Adjudicación CODEX de chunk elegible sin etiqueta; método={method}; "
            f"evidencia_modelo={support_model or 'ninguna'}; "
            f"evidencia_etiquetas={evidence_labels}; prompt_final=3.1.1."
        )
        digest = hashlib.sha256(
            f"{batch_id}|{chunk_id}|{','.join(labels)}".encode("utf-8")
        ).hexdigest()[:24]
        event = ReviewEvent(
            event_id=f"codex-{digest}",
            chunk_id=chunk_id,
            action="modify",
            proposed_labels=[],
            final_labels=labels,
            flags=flags,
            reviewer="CODEX",
            model_id=support_model,
            decision_scope="chunk",
            decision_scope_key=f"chunk:{chunk_id}",
            batch_id=batch_id,
            batch_target_count=len(targets),
            notes=note,
            created_at=created_at,
        ).model_dump(mode="json")
        decisions.append(event)
        method_counts[method] += 1
        label_combinations["+".join(labels)] += 1

    stats = {
        "campaign_rows": len(campaign),
        "events_before": len(previous_events),
        "target_unlabeled_nonexcluded": len(targets),
        "decisions": len(decisions),
        "batch_id": batch_id,
        "method_counts": dict(method_counts),
        "original_model_counts": dict(original_model_counts),
        "override_counts": dict(override_counts),
        "label_combinations": dict(label_combinations),
        "category_counts": {
            label: sum(label in event["final_labels"] for event in decisions)
            for label in LABEL_ORDER
        },
    }
    return decisions, stats


def finalize(*, apply: bool) -> None:
    before_hash = sha256_file(REVIEWS) if REVIEWS.is_file() else None
    decisions, stats = build_decisions()
    # El snapshot es el artefacto revisable previo a tocar el registro canónico.
    write_jsonl_atomic(EVENT_SNAPSHOT, decisions)
    manifest = {
        "schema_version": "1.0.0",
        "reviewer": "CODEX",
        "human_in_the_loop": True,
        "human_criteria_source": "interacción del usuario y decisiones de supervisión CODEX",
        "status": "applied" if apply else "planned",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": str(CAMPAIGN.relative_to(ROOT)).replace("\\", "/"),
        "campaign_sha256": sha256_file(CAMPAIGN),
        "reviews": str(REVIEWS.relative_to(ROOT)).replace("\\", "/"),
        "reviews_sha256_before": before_hash,
        "event_snapshot": str(EVENT_SNAPSHOT.relative_to(ROOT)).replace("\\", "/"),
        "event_snapshot_sha256": sha256_file(EVENT_SNAPSHOT),
        "operational_prompt": str(PROMPT_V31.relative_to(ROOT)).replace("\\", "/"),
        "operational_prompt_sha256": sha256_file(PROMPT_V31),
        "remote_evidence": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "rows": len(rows_by_id(path)),
            }
            for path in (V3, HARD, STANCE, BROAD, LOW_RISK_SAMPLE, LOW_RISK_HARD)
            if path.is_file()
        ],
        "estimated_remote_cost_usd": 2.393066,
        "measured_remote_elapsed_seconds_excluding_pilot": 1543.625,
        "deepseek_balance_last_observed_usd": 0.28,
        "statistics": stats,
    }
    if apply:
        added, skipped = append_jsonl_once(REVIEWS, decisions, id_field="event_id")
        manifest["append"] = {"added": added, "skipped": skipped}
        manifest["reviews_sha256_after"] = sha256_file(REVIEWS)
    write_json_atomic(MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    broad = subparsers.add_parser("run-broad")
    broad.add_argument("--max-cost-usd", type=float, default=0.25)
    low_risk = subparsers.add_parser("run-low-risk-sample")
    low_risk.add_argument("--sample-size", type=int, default=3200)
    low_risk.add_argument("--max-cost-usd", type=float, default=0.36)
    low_risk_hard = subparsers.add_parser("run-low-risk-hard")
    low_risk_hard.add_argument("--max-cost-usd", type=float, default=0.15)
    plan = subparsers.add_parser("plan")
    plan.set_defaults(apply=False)
    final = subparsers.add_parser("finalize")
    final.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "run-broad":
        run_broad(args.max_cost_usd)
    elif args.command == "run-low-risk-sample":
        run_low_risk_sample(args.sample_size, args.max_cost_usd)
    elif args.command == "run-low-risk-hard":
        run_low_risk_hard(args.max_cost_usd)
    elif args.command == "plan":
        finalize(apply=False)
    elif args.command == "finalize":
        finalize(apply=bool(args.apply))


if __name__ == "__main__":
    main()
