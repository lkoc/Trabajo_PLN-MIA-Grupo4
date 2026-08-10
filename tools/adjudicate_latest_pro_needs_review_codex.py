"""Cierra los ``needs_review`` de la última pasada Pro mediante supervisión CODEX.

La propuesta Pro siempre contiene una etiqueta gruesa. El protocolo conserva
esa propuesta salvo cuando la revisión semántica identifica una contradicción
clara con el prompt operacional 3.2: veto de atribución, negación, ausencia de
blanco, polisemia peruana o categoría protegida inexistente. Las decisiones se
guardan como eventos append-only; las inferencias remotas no se reescriben.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from moderacion_peru.io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from moderacion_peru.paths import find_project_root
from moderacion_peru.schemas import ReviewEvent

ROOT = find_project_root()
PRO = ROOT / "datos/etiquetado/cascada_deepseek_v4/review_pro_v3_2.jsonl"
FLASH = ROOT / "datos/etiquetado/cascada_deepseek_v4/primary_flash_v3_2.jsonl"
REVIEWS = ROOT / "datos/etiquetado/humano/labeling_events_v2.jsonl"
PROMPT = ROOT / "config/prompt_operacional_ollama_v3_2.md"
SNAPSHOT = (
    ROOT / "datos/etiquetado/humano" / "codex_latest_pro_v3_2_reviewed.events.jsonl"
)
MANIFEST = SNAPSHOT.with_suffix(".manifest.json")
BATCH_ID = "CODEX-PRO-V3_2-FINAL-20260809"

LABEL_ORDER = (
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
)


def _safe(reason: str) -> tuple[list[str], str]:
    return ["SEGURO"], reason


# Overrides puntuales leídos por CODEX. Solo se cambia Pro cuando el propio
# fragmento, su atribución recuperable o la justificación de Pro contradicen
# de forma suficientemente clara el contrato 3.2.
OVERRIDES: dict[str, tuple[list[str], str]] = {
    # Anatomía o topónimo, no descripción sexual explícita.
    "PyJZ71zkU0c_45bae80483ff87eddb3b": _safe(
        "'concha de San Andrés' es un topónimo y la reducción mamaria no es sexual; "
        "la justificación Pro añadió un contexto de penetración ausente."
    ),
    # Cita insinuante, pero no explícita ni gráfica: aplica veto de atribución.
    "hN0kJIXdC6o_0ba2187a878d9678a500": _safe(
        "La frase insinuante está atribuida a una tercera persona y no describe un "
        "acto sexual explícito; aplica el veto de atribución."
    ),
    # ASR corrupto sin blanco ni proposición dañina recuperable.
    "bKfbWk4zBZc_38ec63758560a551f0e5": _safe(
        "No hay insulto, blanco ni amenaza observables; mencionar madre e hijo no "
        "basta para inferir acoso."
    ),
    "wM5l94LBkCE_93d5ce875472e233fd63": _safe(
        "Insulto fosilizado aislado en ASR corrupto, sin blanco ni atribución "
        "recuperables."
    ),
    "wM5l94LBkCE_762042d3e94ac013a203": _safe(
        "Expresión censurada y fragmentaria sin blanco identificable; el contrato "
        "prohíbe inventar el ataque omitido."
    ),
    # El enunciado desmiente el estereotipo; se conserva la excepción sexual.
    "gJtXP-CTKc0_676950231167b2114ec6": (
        ["CONTENIDO_SEXUAL"],
        (
            "El hablante desmiente el mito racial y no inferioriza al grupo; "
            "permanece la descripción sexual explícita independiente."
        ),
    ),
    # Referencia política y juego de palabras, no atributos protegidos.
    "V5dyjL_43RA_968b68e11072cb6060e2": _safe(
        "'Videos chinos/Vladichín' es un juego de palabras sobre videovigilancia; "
        "no hay inferiorización racial, regional o nacional."
    ),
    "oy169TBaS_o_818e0d5c268f0c209a4c": (
        ["ACOSO_AMENAZA"],
        (
            "'Caviares' designa una posición política en este uso y no prueba "
            "racialización; se conserva el ataque colectivo propio."
        ),
    ),
    "oy169TBaS_o_85478670133917626597": (
        ["ACOSO_AMENAZA"],
        (
            "'Caviares' designa una posición política en este uso y no prueba "
            "racialización; se conserva la imputación hostil al grupo."
        ),
    ),
    # Denuncias, testimonios y audios atribuidos: no heredan daño no sexual.
    "6sWQXDx0lvY_d1e3c98e05dc62f71cac": _safe(
        "La víctima denuncia arañazos y golpes; no profiere un ataque propio."
    ),
    "hLiZRkcKnUM_e1971cea16779c3dfce4": _safe(
        "El programa introduce y reproduce un audio atribuido; no hay ataque nuevo "
        "del narrador."
    ),
    "hLiZRkcKnUM_575e7fa35b18e257e284": _safe(
        "Es continuación de un audio atribuido y el chunk no contiene un insulto "
        "propio inequívoco del narrador."
    ),
    "hLiZRkcKnUM_8494ca6405510ea95599": _safe(
        "La víctima pide que la dejen en paz y pregunta si la matarán o golpearán; "
        "no emite la amenaza."
    ),
    "cWe6VKDEzvk_1d4000ce594bdbe77803": _safe(
        "Testimonio explícito de amenazas recibidas en el programa, sin adopción "
        "ni ataque propio."
    ),
    "cWe6VKDEzvk_26479e1c6c74877216b1": _safe(
        "La hablante denuncia que ella y sus hijos fueron amenazados; aplica el "
        "veto de atribución."
    ),
    "cWe6VKDEzvk_8424836cd6d3cf7ea783": _safe(
        "La conductora resume amenazas denunciadas y fotos de armas; no vuelve a "
        "dirigir la amenaza."
    ),
    "B8RyfJpAQVk_22b496f2d504c30e3ea9": _safe(
        "El chunk introduce con 'Escuchen' una grabación de una niña; los insultos "
        "raciales y personales son evidencia reproducida, no habla adoptada."
    ),
    "gMJF-yqQKwM_b3bbc1a1d27667bd1cff": _safe(
        "La conductora presenta un audio de una niña; la amenaza e insulto quedan "
        "claramente atribuidos."
    ),
    "gMJF-yqQKwM_7c8d067ad558fd72c508": _safe(
        "Continuación del audio infantil reproducido por el reportaje, sin adopción "
        "del narrador."
    ),
    "gMJF-yqQKwM_3774eb3d897e05c66402": _safe(
        "Continuación del audio infantil atribuido; denunciar a alguien no es aquí "
        "una amenaza proferida por el programa."
    ),
    "gMJF-yqQKwM_61fbde33eced5f5368ce": _safe(
        "El propio texto anuncia 'Vamos con el primer audio'; los insultos raciales "
        "son cita atribuida y no son adoptados."
    ),
    "o3xPbl3v-NA_8536606e08a1129fc995": _safe(
        "La hablante cita una amenaza recibida y la denuncia; la justificación Pro "
        "reconoce que corresponde el veto de atribución."
    ),
    "DXsTLFy79bw_d6f06f2f436549753b30": _safe(
        "Se responde defensivamente a amenazas previas; el chunk no profiere una "
        "amenaza, insulto ni hostigamiento propio."
    ),
    "HNBaiKo92nw_c7d3e9789e9d20435b5d": _safe(
        "Acusación truncada de una amenaza recibida, sin hablante, blanco ni ataque "
        "propio recuperables."
    ),
    # Negación o confrontación sin insulto, humillación ni amenaza observable.
    "hXTM_nshBOo_397d5029226dee00a464": _safe(
        "La frase repite 'no va a subir' y niega matar/violar; Pro convirtió la "
        "negación en amenaza."
    ),
    "F8fqLppdwUU_eca306a41d8f3e6d55b6": _safe(
        "Discusión sobre responsabilidad y una acusación citada, sin insulto, "
        "humillación ni amenaza propia."
    ),
    "F8fqLppdwUU_24b0f520187387c980cd": _safe(
        "Relato de maltrato y engaño; el enojo y la confrontación por sí solos no "
        "cumplen el umbral de acoso."
    ),
    "Iaa2LSEaANk_e5c9e510cba9dc808659": _safe(
        "'Ten cuidado' funciona como advertencia defensiva; no hay insulto ni "
        "amenaza plausible proferida."
    ),
    "QhO6lNeNSwo_3df171fe97c5eb7225cd": _safe(
        "Misma advertencia defensiva: tono hostil no equivale por sí solo a acoso."
    ),
    "QhO6lNeNSwo_1cb80d087d48ef75da4b": _safe(
        "Acusación de violencia dentro de una disputa, sin insulto, amenaza o "
        "humillación observable."
    ),
    "o3xPbl3v-NA_eada273108db0b793386": _safe(
        "La hablante se defiende de acusaciones; no profiere un insulto ni amenaza "
        "contra un blanco."
    ),
    "OWlryMLElpY_75349318b301711f2985": _safe(
        "Fragmento sobre custodia y espera judicial; no contiene el ataque que la "
        "justificación Pro trasladó desde otro contexto."
    ),
}


def _latest_events(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in rows:
        chunk_id = str(event["chunk_id"])
        key = (str(event.get("created_at") or ""), str(event["event_id"]))
        previous = latest.get(chunk_id)
        previous_key = (
            (str(previous.get("created_at") or ""), str(previous["event_id"]))
            if previous
            else None
        )
        if previous_key is None or key > previous_key:
            latest[chunk_id] = event
    return latest


def _ordered(labels: Iterable[str]) -> list[str]:
    values = set(labels)
    if "SEGURO" in values:
        return ["SEGURO"]
    return [label for label in LABEL_ORDER if label in values]


def build_events() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    review_rows = list(read_jsonl(REVIEWS)) if REVIEWS.is_file() else []
    latest = _latest_events(review_rows)
    pro_rows = list(read_jsonl(PRO))
    targets = [
        row
        for row in pro_rows
        if bool(row.get("needs_review"))
        or str(row.get("decision_status") or "") == "needs_review"
        if (
            str(row["chunk_id"]) not in latest
            or str(latest[str(row["chunk_id"])].get("batch_id") or "") == BATCH_ID
        )
    ]
    targets.sort(key=lambda row: str(row["chunk_id"]))

    created_at = datetime.now(UTC)
    events: list[dict[str, Any]] = []
    before_labels: Counter[str] = Counter()
    after_labels: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    overrides_detail: list[dict[str, Any]] = []

    for row in targets:
        chunk_id = str(row["chunk_id"])
        proposed = _ordered(row.get("coarse_labels") or [])
        if not proposed:
            raise RuntimeError(f"Pro no emitió etiqueta gruesa para {chunk_id}")
        final, reason = OVERRIDES.get(
            chunk_id,
            (
                proposed,
                (
                    "La revisión CODEX no encontró una contradicción "
                    "suficientemente clara; prevalece la propuesta Pro conforme "
                    "al protocolo."
                ),
            ),
        )
        final = _ordered(final)
        action = "accept" if final == proposed else "modify"
        transition = f"{'+'.join(proposed)} -> {'+'.join(final)}"
        digest = hashlib.sha256(
            f"{BATCH_ID}|{chunk_id}|{','.join(final)}".encode()
        ).hexdigest()[:24]
        notes = (
            "Supervisión semántica CODEX con prompt operacional 3.2.0. "
            f"{reason} El estado needs_review de Pro queda resuelto y no se "
            "conservan flags intermedios."
        )
        existing = latest.get(chunk_id)
        if existing and str(existing.get("batch_id") or "") == BATCH_ID:
            event = ReviewEvent.model_validate(existing).model_dump(mode="json")
            if event["action"] != action or event["final_labels"] != final:
                raise RuntimeError(
                    f"El evento existente de {chunk_id} contradice la adjudicación"
                )
        else:
            event = ReviewEvent(
                event_id=f"codex-pro-v3-2-{digest}",
                chunk_id=chunk_id,
                action=action,
                proposed_labels=proposed,
                final_labels=final,
                flags=[],
                reviewer="CODEX",
                model_id="codex",
                decision_scope="chunk",
                decision_scope_key=f"chunk:{chunk_id}",
                batch_id=BATCH_ID,
                batch_target_count=len(targets),
                notes=notes,
                created_at=created_at,
            ).model_dump(mode="json")
        events.append(event)
        before_labels.update(proposed)
        after_labels.update(final)
        transitions[transition] += 1
        channels[str(row.get("channel_title") or "sin_canal")] += 1
        if action == "modify":
            overrides_detail.append(
                {
                    "chunk_id": chunk_id,
                    "channel_title": row.get("channel_title"),
                    "proposed_labels": proposed,
                    "final_labels": final,
                    "reason": reason,
                }
            )

    stats = {
        "pro_rows": len(pro_rows),
        "review_events_before": len(review_rows),
        "latest_reviewed_chunks_before": len(latest),
        "targets": len(targets),
        "accepted_pro": sum(event["action"] == "accept" for event in events),
        "modified": sum(event["action"] == "modify" for event in events),
        "labels_before": dict(sorted(before_labels.items())),
        "labels_after": dict(sorted(after_labels.items())),
        "transitions": dict(sorted(transitions.items())),
        "channels": dict(sorted(channels.items())),
        "override_decisions": overrides_detail,
    }
    return events, stats


def finalize(*, apply: bool) -> dict[str, Any]:
    reviews_hash_before = sha256_file(REVIEWS) if REVIEWS.is_file() else None
    events, statistics = build_events()
    write_jsonl_atomic(SNAPSHOT, events)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "applied" if apply else "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "batch_id": BATCH_ID,
        "reviewer": "CODEX",
        "human_supervisor_simulation": True,
        "decision_rule": (
            "conservar Pro salvo contradicción semántica clara con prompt 3.2; "
            "needs_review queda superado por la decisión CODEX"
        ),
        "reasoning_profile": "revisión dirigida de alta rigurosidad",
        "prompt": str(PROMPT.relative_to(ROOT)).replace("\\", "/"),
        "prompt_sha256": sha256_file(PROMPT),
        "flash": str(FLASH.relative_to(ROOT)).replace("\\", "/"),
        "flash_sha256": sha256_file(FLASH),
        "pro": str(PRO.relative_to(ROOT)).replace("\\", "/"),
        "pro_sha256": sha256_file(PRO),
        "reviews": str(REVIEWS.relative_to(ROOT)).replace("\\", "/"),
        "reviews_sha256_before": reviews_hash_before,
        "event_snapshot": str(SNAPSHOT.relative_to(ROOT)).replace("\\", "/"),
        "event_snapshot_sha256": sha256_file(SNAPSHOT),
        "statistics": statistics,
    }
    if apply:
        added, skipped = append_jsonl_once(REVIEWS, events, id_field="event_id")
        manifest["append"] = {"added": added, "skipped": skipped}
        manifest["reviews_sha256_after"] = sha256_file(REVIEWS)
    write_json_atomic(MANIFEST, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(finalize(apply=arguments.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
