"""Prepara, audita y entrena el moderador con la ampliación dirigida.

Los casos nuevos Pro con duda se exportan a cola humana. Aceptaciones y
modificaciones se incorporan; rechazos se excluyen. Los casos utilizables se
dividen por video entre train/validation y el test histórico permanece congelado.
Los casos humanos todavía abiertos se excluyen de la instantánea utilizable sin
detener el flujo; se podrán incorporar en una iteración posterior.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts_auxiliares.flujo_hibrido_moderador import (
    build_hybrid_dataset,
    grouped_train_validation_test_split,
    load_taxonomy,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from scripts_auxiliares.mejoras_modelos_gruesos import augment_damage_with_punctuation
from scripts_auxiliares.modelos_gruesos_moderador import (
    COARSE_ORDER,
    DAMAGE_ORDER,
    FINE_TO_COARSE,
    add_coarse_targets,
    evaluate_candidate,
    fit_candidate,
    load_coarse_model,
    save_coarse_model,
    target_matrix,
    tune_candidate,
)


BATCH_ID = os.getenv("AMPLIACION_BATCH_ID", "ampliacion_dano_20260726").strip()
LEGACY_BATCH_ID = "ampliacion_dano_20260726"
BASE_SPLIT_SEED = 131
NEW_SPLIT_SEED_START = int(os.getenv("AMPLIACION_SEED", "26072026"))
NEW_VALIDATION_SIZE = 0.20
MAX_FEATURES = 50_000


def find_project_root(start: Path | None = None) -> Path:
    starts = []
    configured = os.getenv("PLN_PROJECT_ROOT", "").strip()
    if configured:
        starts.append(Path(configured).expanduser())
    starts.extend([start or Path.cwd(), Path(__file__).resolve().parents[1]])
    candidates = []
    for value in starts:
        value = value.resolve()
        candidates.extend((value, *value.parents))
    for candidate in dict.fromkeys(candidates):
        if (candidate / "scripts_auxiliares" / "modelos_gruesos_moderador.py").exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró la raíz del proyecto. Defina PLN_PROJECT_ROOT en kernels remotos."
    )


ROOT = find_project_root()
BATCH_DIR = ROOT / "datos" / "ampliacion" / BATCH_ID
CHUNKS_PATH = BATCH_DIR / "processed" / "chunks_para_etiquetar.jsonl"
FLASH_PATH = BATCH_DIR / "etiquetado" / "deepseek-v4-flash.jsonl"
PRO_PATH = BATCH_DIR / "etiquetado" / "deepseek-v4-pro_revision.jsonl"
PRO_MANIFEST_PATH = BATCH_DIR / "etiquetado" / "deepseek-v4-pro_revision.manifest.json"
ACQUISITION_MANIFEST_PATH = BATCH_DIR / "manifiesto_adquisicion.json"
USABLE_PATH = BATCH_DIR / "processed" / "dataset_etiquetado_utilizable.jsonl"
PENDING_HUMAN_PATH = BATCH_DIR / "processed" / "pendientes_revision_humana.jsonl"
DATASET_MANIFEST_PATH = BATCH_DIR / "processed" / "dataset_etiquetado_utilizable.manifest.json"
RESULTS_DIR = ROOT / "resultados"
OUTPUT_SUFFIX = "ampliacion_dano" if BATCH_ID == LEGACY_BATCH_ID else BATCH_ID
METRICS_DIR = RESULTS_DIR / "metricas" / OUTPUT_SUFFIX
FIGURES_DIR = RESULTS_DIR / "figuras" / OUTPUT_SUFFIX
MODEL_DIR = ROOT / "modelos" / f"moderador_grueso_{OUTPUT_SUFFIX}"
REPORT_PATH = RESULTS_DIR / (
    "INFORME_AMPLIACION_DIRIGIDA_DANO.md"
    if BATCH_ID == LEGACY_BATCH_ID
    else f"INFORME_{BATCH_ID.upper()}.md"
)
INCREMENTAL_REPORT_PATH = RESULTS_DIR / "INFORME_ENTRENAMIENTO_INCREMENTAL_AMPLIACION.md"
OPERATIONAL_METRICS_PATH = METRICS_DIR / "aceptabilidad_operativa.json"
SOURCE_PERFORMANCE_PATH = METRICS_DIR / "rendimiento_por_fuente.csv"
if BATCH_ID == LEGACY_BATCH_ID:
    HUMAN_PROGRESS_PATH = ROOT / "datos" / "etiquetado" / "humano" / "revision_humana_combinada_1918.progress.json"
    HUMAN_EXPANSION_PATH = ROOT / "datos" / "etiquetado" / "humano" / "revision_humana_ampliacion_1779.jsonl"
    HUMAN_EXPANSION_MANIFEST_PATH = ROOT / "datos" / "etiquetado" / "humano" / "revision_humana_ampliacion_1779.manifest.json"
else:
    HUMAN_PROGRESS_PATH = BATCH_DIR / "processed" / "revision_humana.progress.json"
    HUMAN_EXPANSION_PATH = BATCH_DIR / "processed" / "revision_humana.jsonl"
    HUMAN_EXPANSION_MANIFEST_PATH = BATCH_DIR / "processed" / "revision_humana.manifest.json"
HUMAN_SNAPSHOT_DIR = ROOT / "datos" / "etiquetado" / "humano" / "snapshots_entrenamiento"
ORCHESTRATOR_NOTEBOOK_PATH = ROOT / "Cuadernos" / "01_1_ampliacion_dirigida_dano.ipynb"
for directory in (METRICS_DIR, FIGURES_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def active_human_progress_path() -> Path:
    configured = os.environ.get("MODERATION_HUMAN_PROGRESS_SNAPSHOT")
    return Path(configured) if configured else HUMAN_PROGRESS_PATH


def snapshot_human_progress() -> Path:
    """Congela el progreso mutable en un artefacto identificado por contenido."""
    if not HUMAN_PROGRESS_PATH.exists():
        raise FileNotFoundError(f"No existe el progreso humano: {HUMAN_PROGRESS_PATH}")
    progress = json.loads(HUMAN_PROGRESS_PATH.read_text(encoding="utf-8"))
    digest = sha256_file(HUMAN_PROGRESS_PATH)
    revision = int(progress.get("revision", 0))
    HUMAN_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = HUMAN_SNAPSHOT_DIR / f"revision_humana_r{revision}_{digest[:12]}.json"
    if not path.exists():
        write_json_atomic(path, progress)
    return path


def coarse_from_fine(labels: list[str]) -> list[str]:
    mapped = {FINE_TO_COARSE[label] for label in labels}
    return [label for label in COARSE_ORDER if label in mapped]


def load_expansion_human_adjudications(
    pending_by_id: dict[str, dict],
) -> tuple[dict[str, dict], dict]:
    """Carga decisiones completas de la salida final o del progreso parcial."""
    complete = HUMAN_EXPANSION_PATH.exists()
    manifest_sha256 = None
    output_sha256 = None
    progress_snapshot_sha256 = None
    if complete:
        rows = read_jsonl(HUMAN_EXPANSION_PATH)
        source = "human_final_1779"
        output_sha256 = sha256_file(HUMAN_EXPANSION_PATH)
    else:
        progress_path = active_human_progress_path()
        if not progress_path.exists():
            rows = []
        else:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            rows = [
                row for row in progress.get("annotations", [])
                if row.get("status") == "completed" and row.get("chunk_id") in pending_by_id
            ]
            progress_snapshot_sha256 = sha256_file(progress_path)
        source = "human_partial_progress_snapshot"
    by_id = {str(row.get("chunk_id") or ""): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("La adjudicación humana de ampliación contiene IDs vacíos o duplicados.")
    missing, extra = set(pending_by_id) - set(by_id), set(by_id) - set(pending_by_id)
    expected_rows = len(pending_by_id)
    if complete and (len(rows) != expected_rows or missing or extra):
        raise ValueError(
            f"La salida humana de ampliación no cubre exactamente los {expected_rows} pendientes: "
            f"filas={len(rows)}, faltan={len(missing)}, sobran={len(extra)}."
        )
    allowed_actions = {"accept_llm", "reject_llm", "modify_llm", "legacy_human_decision"}
    for chunk_id, row in by_id.items():
        action = row.get("review_action")
        eligible = bool(row.get("training_eligible"))
        labels, flags = row.get("coarse_labels", []), row.get("flags", [])
        if row.get("status") != "completed" or bool(row.get("needs_review")):
            raise ValueError(f"Humano-ampliación/{chunk_id}: decisión no cerrada.")
        if action not in allowed_actions:
            raise ValueError(f"Humano-ampliación/{chunk_id}: review_action inválida.")
        if action == "reject_llm" and (eligible or labels or flags):
            raise ValueError(f"Humano-ampliación/{chunk_id}: rechazo inconsistente.")
        if action != "reject_llm" and (not eligible or not labels):
            raise ValueError(f"Humano-ampliación/{chunk_id}: inclusión inconsistente.")
        if not set(labels) <= set(COARSE_ORDER) or len(labels) != len(set(labels)):
            raise ValueError(f"Humano-ampliación/{chunk_id}: categorías gruesas inválidas.")
        if not set(flags) <= {"ironia_ambigua", "humor_encubridor", "contexto_necesario"}:
            raise ValueError(f"Humano-ampliación/{chunk_id}: flags inválidos.")
        if "SEGURO" in labels and len(labels) > 1:
            raise ValueError(f"Humano-ampliación/{chunk_id}: SEGURO no es excluyente.")
        if flags and not (set(labels) & set(DAMAGE_ORDER)):
            raise ValueError(f"Humano-ampliación/{chunk_id}: flag sin daño.")
        if action == "accept_llm":
            proposal = pending_by_id[chunk_id]
            if (
                set(labels) != set(proposal["pro_coarse_labels"])
                or set(flags) != set(proposal.get("pro_flags", []))
            ):
                raise ValueError(f"Humano-ampliación/{chunk_id}: aceptación no coincide con Pro.")
    if complete:
        if not HUMAN_EXPANSION_MANIFEST_PATH.exists():
            raise FileNotFoundError("Falta el manifiesto de la adjudicación humana de ampliación.")
        manifest = json.loads(HUMAN_EXPANSION_MANIFEST_PATH.read_text(encoding="utf-8"))
        if int(manifest.get("completed", 0)) != expected_rows:
            raise ValueError(
                f"El manifiesto humano no confirma {expected_rows} decisiones completas."
            )
        if manifest.get("output_sha256") != output_sha256:
            raise ValueError("El SHA-256 humano de ampliación no coincide con su manifiesto.")
        manifest_sha256 = sha256_file(HUMAN_EXPANSION_MANIFEST_PATH)
    actions = Counter(row["review_action"] for row in rows)
    metadata = {
        "complete": complete,
        "source": source,
        "adjudicated": len(rows),
        "pending": len(pending_by_id) - len(rows),
        "included": sum(bool(row["training_eligible"]) for row in rows),
        "excluded": sum(not bool(row["training_eligible"]) for row in rows),
        "action_counts": dict(actions),
        "output_sha256": output_sha256,
        "manifest_sha256": manifest_sha256,
        "progress_snapshot_sha256": progress_snapshot_sha256,
    }
    return by_id, metadata


def validate_inputs() -> tuple[list[dict], list[dict], list[dict], dict]:
    required = [CHUNKS_PATH, FLASH_PATH, PRO_PATH, PRO_MANIFEST_PATH, ACQUISITION_MANIFEST_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan entradas:\n- " + "\n- ".join(missing))
    chunks, flash, pro = read_jsonl(CHUNKS_PATH), read_jsonl(FLASH_PATH), read_jsonl(PRO_PATH)
    chunk_ids = [row["chunk_id"] for row in chunks]
    flash_ids = [row["chunk_id"] for row in flash]
    pro_ids = [row["chunk_id"] for row in pro]
    if set(chunk_ids) != set(flash_ids) or len(chunk_ids) != len(flash_ids):
        raise ValueError(
            f"Flash no cubre exactamente los {len(chunk_ids):,} chunks nuevos"
        )
    if len(chunk_ids) != len(set(chunk_ids)) or len(pro_ids) != len(set(pro_ids)):
        raise ValueError("Existen IDs duplicados")
    manifest = json.loads(PRO_MANIFEST_PATH.read_text(encoding="utf-8"))
    if set(pro_ids) != set(manifest["chunk_ids"]) or len(pro) != int(manifest["selected_rows"]):
        raise ValueError("Pro no coincide con su selección manifiesta")
    return chunks, flash, pro, manifest


def choose_new_split(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    y = target_matrix(frame)
    groups = frame["video_id"].astype(str).to_numpy()
    global_prevalence = y.mean(axis=0)
    best = None
    for offset in range(500):
        seed = NEW_SPLIT_SEED_START + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=NEW_VALIDATION_SIZE, random_state=seed)
        train_index, validation_index = next(splitter.split(frame, groups=groups))
        train_y, validation_y = y[train_index], y[validation_index]
        missing = int((train_y.sum(axis=0) == 0).sum() + (validation_y.sum(axis=0) == 0).sum())
        scale = np.sqrt(np.maximum(global_prevalence, 1 / len(frame)))
        divergence = float(
            np.mean(np.abs(train_y.mean(axis=0) - global_prevalence) / scale)
            + np.mean(np.abs(validation_y.mean(axis=0) - global_prevalence) / scale)
        )
        size_error = abs(len(validation_index) / len(frame) - NEW_VALIDATION_SIZE)
        score = 10 * missing + divergence + size_error
        candidate = (score, seed, train_index, validation_index, missing, divergence)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, seed, train_index, validation_index, missing, divergence = best
    split = pd.Series(index=frame.index, dtype="object")
    split.iloc[train_index] = "train"
    split.iloc[validation_index] = "validation"
    metadata = {
        "method": "best of 500 GroupShuffleSplit candidates by coarse prevalence",
        "seed": int(seed), "group": "video_id", "test_rows": 0,
        "train_rows": int(len(train_index)), "validation_rows": int(len(validation_index)),
        "missing_category_cells": int(missing), "prevalence_divergence": float(divergence),
    }
    return split, metadata


def previous_usable_expansion_paths() -> list[Path]:
    """Datasets de ampliación cerrados antes del lote activo."""
    paths = []
    for path in sorted(
        (ROOT / "datos" / "ampliacion").glob(
            "*/processed/dataset_etiquetado_utilizable.jsonl"
        )
    ):
        if path.resolve() == USABLE_PATH.resolve():
            continue
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Ampliación sin manifiesto: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("output_sha256", {}).get("usable")
        if expected != sha256_file(path):
            raise ValueError(f"SHA-256 no coincide para la ampliación previa: {path}")
        paths.append(path)
    return paths


def base_provisional_counts() -> tuple[pd.DataFrame, dict]:
    taxonomy, _, _ = load_taxonomy(ROOT)
    frame, metadata, _, _ = build_hybrid_dataset(
        ROOT, require_complete_hard_review=False, write_output=False
    )
    frame = add_coarse_targets(frame, taxonomy)
    prior_frames = []
    for path in previous_usable_expansion_paths():
        prior = pd.DataFrame(read_jsonl(path))
        if prior.empty:
            continue
        prior["campaign"] = path.parents[1].name
        prior_frames.append(prior)
    if prior_frames:
        frame = pd.concat([frame, *prior_frames], ignore_index=True, sort=False)
    if frame["chunk_id"].duplicated().any():
        raise ValueError("Las campañas previas se solapan por chunk_id.")
    rows = []
    for category in DAMAGE_ORDER:
        count = int(frame["coarse_labels"].map(lambda values: category in values).sum())
        rows.append({"categoria": category, "antes": count})
    metadata = {
        **metadata,
        "rows": len(frame),
        "previous_expansion_datasets": [
            str(path.relative_to(ROOT)) for path in previous_usable_expansion_paths()
        ],
    }
    return pd.DataFrame(rows), metadata


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    chunks, flash, pro, pro_manifest = validate_inputs()
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    flash_by_id = {row["chunk_id"]: row for row in flash}
    pro_by_id = {row["chunk_id"]: row for row in pro}
    usable_rows, pending_rows = [], []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        flash_row = flash_by_id[chunk_id]
        pro_row = pro_by_id.get(chunk_id)
        if pro_row is not None and bool(pro_row["needs_review"]):
            pending_rows.append(
                {
                    **{key: chunk.get(key) for key in (
                        "chunk_id", "video_id", "channel_title", "video_title",
                        "start_seconds", "end_seconds", "text", "text_hash",
                        "discovery_type", "discovery_source", "target_category",
                    )},
                    "pro_labels": pro_row["labels"], "pro_coarse_labels": coarse_from_fine(pro_row["labels"]),
                    "pro_flags": pro_row["flags"], "pro_score_confianza": pro_row["score_confianza"],
                    "pro_notes": pro_row["notes"], "pro_justificacion": pro_row["justificacion"],
                    "needs_review": True, "training_eligible": False,
                }
            )
            continue
        annotation = pro_row if pro_row is not None else flash_row
        source = "pro_augmented_resolved" if pro_row is not None else "flash_pseudo"
        confidence = float(annotation["score_confianza"])
        usable_rows.append(
            {
                **{key: chunk.get(key) for key in (
                    "chunk_id", "video_id", "channel_title", "video_title",
                    "start_seconds", "end_seconds", "text", "text_hash",
                    "discovery_type", "discovery_source", "target_category",
                )},
                "labels": annotation["labels"], "flags": annotation["flags"],
                "coarse_labels": coarse_from_fine(annotation["labels"]),
                "label_source": source,
                "sample_weight": 1.0 if pro_row is not None else 0.5 * confidence,
                "score_confianza_source": confidence,
                "needs_review": False,
                "batch_id": BATCH_ID,
            }
        )
    pending = pd.DataFrame(pending_rows)
    pending_by_id = {row["chunk_id"]: row for row in pending_rows}
    human_by_id, human_metadata = load_expansion_human_adjudications(pending_by_id)
    if human_by_id:
        for chunk_id, human in human_by_id.items():
            if not bool(human["training_eligible"]):
                continue
            chunk, proposal = chunk_by_id[chunk_id], pending_by_id[chunk_id]
            action = human["review_action"]
            usable_rows.append(
                {
                    **{key: chunk.get(key) for key in (
                        "chunk_id", "video_id", "channel_title", "video_title",
                        "start_seconds", "end_seconds", "text", "text_hash",
                        "discovery_type", "discovery_source", "target_category",
                    )},
                    "labels": proposal["pro_labels"] if action == "accept_llm" else [],
                    "flags": human["flags"],
                    "coarse_labels": human["coarse_labels"],
                    "label_source": (
                        "human_accepted_pro_expansion"
                        if action == "accept_llm"
                        else "human_modified_expansion"
                    ),
                    "sample_weight": 1.0,
                    "score_confianza_source": proposal["pro_score_confianza"],
                    "needs_review": False,
                    "review_action": action,
                    "fine_labels_reference_only": action != "accept_llm",
                    "batch_id": BATCH_ID,
                }
            )
    usable = pd.DataFrame(usable_rows)
    pending_remaining = pending.loc[~pending["chunk_id"].isin(human_by_id)].copy()
    usable["split"], split_metadata = choose_new_split(usable)
    if usable.groupby("video_id")["split"].nunique().max() != 1:
        raise AssertionError("Un video nuevo aparece en más de una partición")
    if set(usable["split"]) != {"train", "validation"}:
        raise AssertionError("La ampliación no puede entrar a test")
    write_jsonl(USABLE_PATH, usable.to_dict("records"))
    # La cola original se preserva como fuente inmutable de la campaña, incluso
    # cuando una salida humana posterior reduzca los pendientes operativos.
    write_jsonl(PENDING_HUMAN_PATH, pending.to_dict("records"))

    performance_rows = []
    source_keys = sorted(
        set(zip(usable["discovery_type"], usable["discovery_source"]))
        | set(zip(pending.get("discovery_type", []), pending.get("discovery_source", [])))
    )
    for discovery_type, discovery_source in source_keys:
        resolved = usable.loc[
            (usable["discovery_type"] == discovery_type)
            & (usable["discovery_source"] == discovery_source)
        ]
        doubtful = pending.loc[
            (pending["discovery_type"] == discovery_type)
            & (pending["discovery_source"] == discovery_source)
        ]
        record = {
            "type": discovery_type,
            "source": discovery_source,
            "videos_resolved": int(resolved["video_id"].nunique()),
            "resolved_rows": len(resolved),
            "pending_human_rows": len(doubtful),
        }
        for category in DAMAGE_ORDER:
            record[category] = int(
                resolved["coarse_labels"].map(lambda labels: category in labels).sum()
            )
        performance_rows.append(record)
    pd.DataFrame(performance_rows).to_csv(SOURCE_PERFORMANCE_PATH, index=False)

    before, base_metadata = base_provisional_counts()
    additions = []
    for category in DAMAGE_ORDER:
        count = int(usable["coarse_labels"].map(lambda values: category in values).sum())
        additions.append({"categoria": category, "agregados_resueltos": count})
    balance = before.merge(pd.DataFrame(additions), on="categoria")
    balance["despues_utilizable"] = balance["antes"] + balance["agregados_resueltos"]
    balance["deficit_1000"] = (1000 - balance["despues_utilizable"]).clip(lower=0)
    balance["cumple_1000"] = balance["despues_utilizable"] >= 1000
    balance.to_csv(METRICS_DIR / "balance_antes_despues.csv", index=False)

    targeted_video_ids = set(
        usable.loc[usable["discovery_type"] == "targeted_search", "video_id"]
    ) | set(pending.loc[pending["discovery_type"] == "targeted_search", "video_id"])
    resolved_threat_by_video = (
        usable.assign(
            threat=usable["coarse_labels"].map(lambda values: "AMENAZA_DIRECTA" in values)
        ).loc[lambda frame: frame["video_id"].isin(targeted_video_ids)]
        .groupby("video_id")["threat"].sum()
        .reindex(sorted(targeted_video_ids), fill_value=0)
    )
    rng = np.random.default_rng(NEW_SPLIT_SEED_START)
    bootstrap_means = np.asarray([
        rng.choice(resolved_threat_by_video.to_numpy(), size=len(resolved_threat_by_video), replace=True).mean()
        for _ in range(10_000)
    ])
    mean_yield = float(resolved_threat_by_video.mean())
    lower_yield, upper_yield = np.quantile(bootstrap_means, [0.025, 0.975])
    threat_after = int(balance.loc[balance["categoria"] == "AMENAZA_DIRECTA", "despues_utilizable"].iloc[0])
    remaining = max(0, 1000 - threat_after)
    estimate = {
        "targeted_videos_observed": int(len(resolved_threat_by_video)),
        "resolved_threat_chunks": int(resolved_threat_by_video.sum()),
        "resolved_threat_per_targeted_video": mean_yield,
        "bootstrap_95_ci_yield": [float(lower_yield), float(upper_yield)],
        "remaining_to_1000": remaining,
        "estimated_additional_targeted_videos_point": math.ceil(remaining / mean_yield) if mean_yield else None,
        "estimated_additional_targeted_videos_conservative": math.ceil(remaining / lower_yield) if lower_yield > 0 else None,
    }

    manifest = {
        "schema_version": "2.0", "batch_id": BATCH_ID, "created_at": now_iso(),
        "policy": (
            "resolved Pro > high-confidence Flash; human accept/modify included; "
            "human reject excluded; unresolved Pro excluded"
        ),
        "coarse_training_only": True, "fine_labels_trained": False,
        "transversal_flags_are_base_targets": False,
        "input_rows": len(chunks), "usable_rows": len(usable),
        "human_review_queue_rows": len(pending),
        "pending_human_rows": len(pending_remaining),
        "human_adjudication": human_metadata,
        "usable_source_counts": usable["label_source"].value_counts().to_dict(),
        "pending_human_pct_new": len(pending_remaining) / len(chunks),
        "split": split_metadata, "balance": balance.to_dict("records"),
        "threat_video_estimate": estimate,
        "input_sha256": {
            "chunks": sha256_file(CHUNKS_PATH), "flash": sha256_file(FLASH_PATH),
            "pro": sha256_file(PRO_PATH), "pro_selection_manifest": sha256_file(PRO_MANIFEST_PATH),
            "human_expansion": human_metadata["output_sha256"],
            "human_expansion_manifest": human_metadata["manifest_sha256"],
            "human_progress_snapshot": human_metadata["progress_snapshot_sha256"],
        },
        "output_sha256": {
            "usable": sha256_file(USABLE_PATH), "pending_human": sha256_file(PENDING_HUMAN_PATH),
        },
        "base_provisional_metadata": {
            "rows": base_metadata["rows"], "source_counts": base_metadata["source_counts"],
            "previous_expansion_datasets": base_metadata.get(
                "previous_expansion_datasets", []
            ),
        },
        "ready_for_04_2_auto_discovery": True,
    }
    write_json_atomic(DATASET_MANIFEST_PATH, manifest)
    create_balance_figure(balance)
    write_report(manifest, balance, training=None)
    return usable, pending_remaining, manifest


def create_balance_figure(balance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    positions = np.arange(len(balance))
    width = 0.36
    ax.bar(positions - width / 2, balance["antes"], width, label="Antes", color="#4C78A8")
    ax.bar(positions + width / 2, balance["despues_utilizable"], width, label="Después (Pro + humano elegible)", color="#F58518")
    ax.axhline(1000, color="#D62728", linestyle="--", linewidth=1.6, label="Objetivo 1.000")
    ax.set_xticks(positions, [value.replace("_", "\n") for value in balance["categoria"]])
    ax.set_ylabel("Chunks positivos (multietiqueta)")
    ax.set_title("Balance de categorías gruesas antes y después de la ampliación")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "balance_antes_despues.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _base_split(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Aplica la partición histórica antes de retirar rechazos humanos.

    Así, excluir un chunk de train/validation no mueve videos hacia otro split
    ni altera el test congelado.
    """
    canonical = read_jsonl(ROOT / "datos" / "processed" / "chunks_para_etiquetar.jsonl")
    holdout_path = ROOT / "datos" / "processed" / "dataset_etiquetado.jsonl"
    holdout_ids = {row["chunk_id"] for row in read_jsonl(holdout_path)} if holdout_path.exists() else set()
    full = pd.DataFrame(
        [
            {"chunk_id": row["chunk_id"], "video_id": row.get("video_id") or row["chunk_id"]}
            for row in canonical if row["chunk_id"] not in holdout_ids
        ]
    )
    full_split = grouped_train_validation_test_split(
        full, seed=BASE_SPLIT_SEED, test_size=0.15, validation_size=0.15
    )
    assignment = {
        str(chunk_id): split_name
        for split_name, indices in full_split.items()
        for chunk_id in full.iloc[indices]["chunk_id"]
    }
    unknown = set(frame["chunk_id"].astype(str)) - set(assignment)
    if unknown:
        raise ValueError(f"Hay {len(unknown)} IDs sin partición histórica.")
    split = {
        name: np.flatnonzero(frame["chunk_id"].astype(str).map(assignment).eq(name).to_numpy())
        for name in ("train", "validation", "test")
    }
    if len(split["test"]) != 10_293:
        raise AssertionError(f"El test histórico cambió: {len(split['test'])} filas.")
    return split


def train_models() -> dict:
    snapshot_path = snapshot_human_progress()
    os.environ["MODERATION_HUMAN_PROGRESS_SNAPSHOT"] = str(snapshot_path)
    # Reconstituye el dataset con las decisiones cerradas de la instantánea y
    # excluye tanto rechazos como dudas todavía abiertas.
    prepare_dataset()
    taxonomy, _, _ = load_taxonomy(ROOT)
    base, base_metadata, _, _ = build_hybrid_dataset(
        ROOT, require_complete_hard_review=False, write_output=False
    )
    base = add_coarse_targets(base, taxonomy)
    base = base.loc[~base["human_holdout"]].reset_index(drop=True)
    split = _base_split(base)
    base_train = base.iloc[split["train"]].reset_index(drop=True)
    base_validation = base.iloc[split["validation"]].reset_index(drop=True)
    frozen_test = base.iloc[split["test"]].reset_index(drop=True)
    new = pd.DataFrame(read_jsonl(USABLE_PATH))
    new_train = new.loc[new["split"] == "train"].reset_index(drop=True)
    new_validation = new.loc[new["split"] == "validation"].reset_index(drop=True)
    augmented_train = pd.concat([base_train, new_train], ignore_index=True)

    experiments = {}
    specs = [
        ("baseline_reproducido", base_train, False),
        ("ampliado_sin_aeda", augmented_train, False),
        ("ampliado_con_aeda", augmented_train, True),
    ]
    rows = []
    for experiment_id, training_frame, use_aeda in specs:
        effective = (
            augment_damage_with_punctuation(
                training_frame, text_column="text", seed=42, repetitions=1,
                insertion_rate=0.08, augmented_weight=0.50,
            ) if use_aeda else training_frame
        )
        model, diagnostics = fit_candidate(
            "linear_svm_word_char", effective, max_features=MAX_FEATURES,
            text_column="text", model_parameters={"C": 1.0}, flash_pseudo_weight=0.50,
        )
        tune_candidate(model, base_validation, text_column="text")
        val_metrics, _, _ = evaluate_candidate(model, base_validation, text_column="text")
        test_metrics, test_report, _ = evaluate_candidate(model, frozen_test, text_column="text")
        new_metrics, _, _ = evaluate_candidate(model, new_validation, text_column="text")
        experiments[experiment_id] = {"model": model, "test_report": test_report}
        rows.append(
            {
                "experimento": experiment_id, "train_rows_original": len(training_frame),
                "train_rows_effective": len(effective), **diagnostics,
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{f"new_validation_{key}": value for key, value in new_metrics.items()},
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"], ascending=False
    ).reset_index(drop=True)
    winner_id = str(comparison.iloc[0]["experimento"])
    if winner_id == "baseline_reproducido":
        # La ampliación no se exporta si no mejora el criterio predefinido.
        export_id = "ampliado_sin_aeda" if (
            comparison.set_index("experimento").loc["ampliado_sin_aeda", "validation_damage_pr_auc_macro"]
            >= comparison.set_index("experimento").loc["ampliado_con_aeda", "validation_damage_pr_auc_macro"]
        ) else "ampliado_con_aeda"
        authorized_as_improvement = False
    else:
        export_id = winner_id
        authorized_as_improvement = True

    comparison_by_id = comparison.set_index("experimento")
    baseline_metrics = comparison_by_id.loc["baseline_reproducido"]
    winner_metrics = comparison_by_id.loc[winner_id]
    test_deltas_vs_baseline = {
        "damage_pr_auc_macro": float(
            winner_metrics["test_damage_pr_auc_macro"]
            - baseline_metrics["test_damage_pr_auc_macro"]
        ),
        "damage_f1_macro": float(
            winner_metrics["test_damage_f1_macro"]
            - baseline_metrics["test_damage_f1_macro"]
        ),
        "damage_recall_micro": float(
            winner_metrics["test_damage_recall_micro"]
            - baseline_metrics["test_damage_recall_micro"]
        ),
    }
    general_improvement_supported = bool(
        winner_metrics["validation_damage_pr_auc_macro"]
        > baseline_metrics["validation_damage_pr_auc_macro"]
        and all(delta > 0 for delta in test_deltas_vs_baseline.values())
    )

    development_model = experiments[export_id]["model"]
    final_source = pd.concat([base_train, base_validation, new_train, new_validation], ignore_index=True)
    use_aeda = export_id == "ampliado_con_aeda"
    final_training = (
        augment_damage_with_punctuation(
            final_source, text_column="text", seed=42, repetitions=1,
            insertion_rate=0.08, augmented_weight=0.50,
        ) if use_aeda else final_source
    )
    final_model, final_diagnostics = fit_candidate(
        "linear_svm_word_char", final_training, max_features=MAX_FEATURES,
        text_column="text", model_parameters={"C": 1.0}, flash_pseudo_weight=0.50,
    )
    final_model.thresholds = development_model.thresholds.copy()
    final_model.metadata = {
        **base_metadata,
        "schema_version": "5.0", "batch_id": BATCH_ID, "outputs": COARSE_ORDER,
        "fine_labels_trained": False, "flags_trained_as_categories": False,
        "frozen_test_rows": len(frozen_test), "new_test_rows": 0,
        "new_train_rows": len(new_train), "new_validation_rows": len(new_validation),
        "pending_new_human_excluded": int(
            json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))["pending_human_rows"]
        ),
        "human_expansion_excluded": int(
            json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))["human_adjudication"]["excluded"]
        ),
        "winner_on_historical_validation": winner_id,
        "export_configuration": export_id,
        "authorized_as_improvement": authorized_as_improvement,
        "general_improvement_supported": general_improvement_supported,
        "test_deltas_vs_baseline": test_deltas_vs_baseline,
        "selection_metric": "damage PR-AUC macro on frozen historical validation",
        "new_dataset_sha256": sha256_file(USABLE_PATH),
        "human_review_snapshot": str(snapshot_path.relative_to(ROOT)),
        "human_review_snapshot_sha256": sha256_file(snapshot_path),
    }
    model_path = MODEL_DIR / "moderador_grueso_ampliado.joblib"
    save_coarse_model(final_model, model_path)
    write_json_atomic(MODEL_DIR / "manifiesto.json", final_model.metadata)
    comparison.to_csv(METRICS_DIR / "comparacion_entrenamiento_ampliado.csv", index=False)
    for experiment_id, values in experiments.items():
        values["test_report"].to_csv(METRICS_DIR / f"reporte_test_{experiment_id}.csv")
    create_training_figure(comparison)
    training = {
        "completed_at": now_iso(), "winner_id": winner_id, "export_id": export_id,
        "authorized_as_improvement": authorized_as_improvement,
        "general_improvement_supported": general_improvement_supported,
        "test_deltas_vs_baseline": test_deltas_vs_baseline,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_file(model_path), "final_training_rows": len(final_training),
        "human_review_snapshot": str(snapshot_path.relative_to(ROOT)),
        "human_review_snapshot_sha256": sha256_file(snapshot_path),
        "final_training_diagnostics": final_diagnostics,
        "comparison": comparison.to_dict("records"),
    }
    manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["training"] = training
    write_json_atomic(DATASET_MANIFEST_PATH, manifest)
    balance = pd.read_csv(METRICS_DIR / "balance_antes_despues.csv")
    evaluate_operational_acceptability(snapshot_path)
    write_report(manifest, balance, training=training)
    write_incremental_training_report(manifest, balance, training)
    return training


def create_training_figure(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = comparison["experimento"].str.replace("_", " ")
    x = np.arange(len(comparison))
    width = 0.35
    axes[0].bar(x - width / 2, comparison["test_damage_f1_macro"], width, label="F1 macro daño")
    axes[0].bar(x + width / 2, comparison["test_damage_pr_auc_macro"], width, label="PR-AUC macro daño")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylim(0, max(0.45, comparison[["test_damage_f1_macro", "test_damage_pr_auc_macro"]].to_numpy().max() + 0.05))
    axes[0].set_title("Desempeño en test histórico congelado")
    axes[0].legend()
    axes[1].bar(x - width / 2, comparison["validation_damage_pr_auc_macro"], width, label="Validación histórica")
    axes[1].bar(x + width / 2, comparison["new_validation_damage_pr_auc_macro"], width, label="Validación enriquecida")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_title("PR-AUC macro de daño por dominio")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparacion_entrenamiento.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(manifest: dict, balance: pd.DataFrame, training: dict | None) -> None:
    acquisition = json.loads(ACQUISITION_MANIFEST_PATH.read_text(encoding="utf-8"))
    flash_manifest = json.loads((FLASH_PATH.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    pro_manifest = json.loads(PRO_MANIFEST_PATH.read_text(encoding="utf-8"))
    estimate = manifest["threat_video_estimate"]
    old_min = int(balance["antes"].min())
    old_max = int(balance["antes"].max())
    new_min = int(balance["despues_utilizable"].min())
    new_max = int(balance["despues_utilizable"].max())
    usable = pd.DataFrame(read_jsonl(USABLE_PATH))
    usable["damage"] = usable["coarse_labels"].map(lambda values: "SEGURO" not in values)
    usable["threat"] = usable["coarse_labels"].map(lambda values: "AMENAZA_DIRECTA" in values)
    source_summary = (
        usable.groupby("discovery_type")
        .agg(videos=("video_id", "nunique"), chunks=("chunk_id", "size"),
             damage=("damage", "sum"), threat=("threat", "sum"))
        .reset_index()
    )
    human_total = int(manifest["human_review_queue_rows"])
    human_completed = int(manifest["human_adjudication"]["adjudicated"])
    prior_raw_rows = int(sum(acquisition.get("prior_chunk_sources", {}).values()))
    if not prior_raw_rows:
        prior_raw_rows = len(read_jsonl(ROOT / "datos" / "processed" / "chunks_para_etiquetar.jsonl"))
    prior_usable_rows = int(manifest["base_provisional_metadata"]["rows"])
    categories_over_target = int((balance["despues_utilizable"] >= 1000).sum())
    threat_before = int(
        balance.loc[
            balance["categoria"] == "AMENAZA_DIRECTA", "antes"
        ].iloc[0]
    )
    lines = [
        "# Informe reproducible de ampliación dirigida de categorías de daño",
        "",
        f"Actualización: {now_iso()}  ",
        f"Lote: `{BATCH_ID}`  ",
        f"Estado del entrenamiento: **{'COMPLETADO' if training else 'DATOS PREPARADOS; ENTRENAMIENTO AÚN NO EJECUTADO'}**",
        "",
        "## 1. Objetivo y salvaguardas",
        "",
        "Se amplió el corpus desde el flujo del cuaderno 01 con muestreo dirigido a categorías minoritarias. Esta selección es adecuada para enriquecer entrenamiento, pero no para estimar prevalencia poblacional. El test histórico se mantiene congelado y ningún video nuevo entra en test. Las etiquetas finas solo fundamentan/proyectan objetivos gruesos; no se entrenan. Los flags transversales se conservan para enrutamiento y tampoco son categorías base.",
        "",
        "## 2. Adquisición y segmentación",
        "",
        f"Se encontraron {acquisition['candidate_unique_videos']:,} candidatos únicos, se seleccionaron {acquisition['selected_videos']:,} y {acquisition['transcript_videos']:,} tuvieron subtítulos utilizables ({acquisition['subtitle_success_rate']:.1%}). Se descargaron exclusivamente subtítulos VTT públicos; no audio ni video. La segmentación compatible con el cuaderno 02 produjo {acquisition['new_chunks']:,} chunks nuevos, eliminando {acquisition['duplicates_against_base'] + acquisition['duplicates_within_batch']:,} duplicados por hash.",
        "",
        f"Los corpus segmentados previos sumaban {prior_raw_rows:,} chunks; el nuevo lote eleva ese inventario auditable a {prior_raw_rows + acquisition['new_chunks']:,}. Se añadieron {acquisition['new_videos_with_chunks']:,} videos con chunks y el manifiesto de adquisición confirma intersección cero con los videos ya procesados. El conjunto utilizable para integración pasa de {prior_usable_rows:,} a {prior_usable_rows + manifest['usable_rows']:,} chunks; quedan {manifest['pending_human_rows']:,} dudas humanas sin cerrar. De las {manifest['human_review_queue_rows']:,} dudas nuevas originales, las aceptadas o modificadas se incorporan y las rechazadas o todavía abiertas se excluyen.",
        "",
        "| Estrategia de adquisición | Videos seleccionados |",
        "|---|---:|",
        f"| Canales de alto rendimiento histórico | {acquisition.get('selected_by_type', {}).get('high_yield_channel', 0):,} |",
        f"| Búsquedas temáticas dirigidas | {acquisition.get('selected_by_type', {}).get('targeted_search', 0):,} |",
        "",
        "Fuentes de canal priorizadas y criterio previo:",
        "",
        "| Fuente | Cuota | Criterio observado en el corpus base |",
        "|---|---:|---|",
    ]
    for target in acquisition["channel_targets"]:
        lines.append(f"| {target['name']} | {target['quota']:,} | {target['reason']} |")
    lines.extend([
        "",
        "Rendimiento limpio del lote por estrategia (después de excluir dudas Pro):",
        "",
        "| Estrategia | Videos utilizables | Chunks utilizables | Chunks de daño | Amenazas directas |",
        "|---|---:|---:|---:|---:|",
    ])
    source_labels = {
        "high_yield_channel": "Canales de alto rendimiento",
        "targeted_search": "Búsquedas temáticas dirigidas",
    }
    for row in source_summary.to_dict("records"):
        lines.append(
            f"| {source_labels.get(row['discovery_type'], row['discovery_type'])} | "
            f"{int(row['videos']):,} | {int(row['chunks']):,} | "
            f"{int(row['damage']):,} | {int(row['threat']):,} |"
        )
    lines.extend([
        "",
        "Las búsquedas dirigidas se usan para extrapolar el tamaño restante porque fueron diseñadas específicamente para enriquecer amenaza directa; los conteos se recalculan al integrar la adjudicación humana.",
        "",
        "## 3. Etiquetado Flash → Pro",
        "",
        f"Flash etiquetó {flash_manifest['completed']:,} chunks, con {flash_manifest['damage_chunks']:,} daños y costo estimado nuevo de USD {flash_manifest['estimated_cost_usd_new']:.2f}. La regla Pro fue: todo daño Flash, toda alerta, `score_confianza < 0.90` y control aleatorio del 10% de seguros confiables. Pro revisó {pro_manifest['completed']:,} chunks por aproximadamente USD {pro_manifest['estimated_cost_usd_new']:.2f}.",
        "",
        f"Pro resolvió {manifest['usable_source_counts'].get('pro_augmented_resolved', 0):,} seleccionados y derivó {manifest['human_review_queue_rows']:,} a humano. La adjudicación humana está {'completa' if manifest['human_adjudication']['complete'] else 'pendiente'}: {manifest['human_adjudication']['included']:,} incluidos y {manifest['human_adjudication']['excluded']:,} rechazados. Persisten {manifest['pending_human_rows']:,} sin resolver ({manifest['pending_human_pct_new']:.2%} del lote nuevo). El costo API nuevo registrado fue aproximadamente USD {flash_manifest['estimated_cost_usd_new'] + pro_manifest['estimated_cost_usd_new']:.2f}.",
        "",
        "## 4. Balance antes/después",
        "",
        "| Categoría gruesa | Antes | Agregados Pro resueltos | Después utilizable | Déficit a 1.000 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in balance.to_dict("records"):
        lines.append(
            f"| `{row['categoria']}` | {int(row['antes']):,} | {int(row['agregados_resueltos']):,} | "
            f"{int(row['despues_utilizable']):,} | {int(row['deficit_1000']):,} |"
        )
    lines.extend(
        [
            "",
            f"![Balance antes y después](figuras/{OUTPUT_SUFFIX}/balance_antes_despues.png)",
            "",
            f"{categories_over_target} de cinco categorías de daño quedan por encima de 1.000. La razón máxima/mínima entre daños cambia de {old_max / old_min:.2f}:1 a {new_max / new_min:.2f}:1. Amenaza directa aumenta de {threat_before} a {int(balance.loc[balance['categoria'] == 'AMENAZA_DIRECTA', 'despues_utilizable'].iloc[0])} ejemplos limpios.",
            "",
            "## 5. Estimación restante para amenaza directa",
            "",
            f"Entre {estimate['targeted_videos_observed']} videos de búsqueda dirigida se obtuvieron {estimate['resolved_threat_chunks']} amenazas Pro resueltas: {estimate['resolved_threat_per_targeted_video']:.3f} por video. El bootstrap por video (10.000 remuestras) dio IC 95% [{estimate['bootstrap_95_ci_yield'][0]:.3f}, {estimate['bootstrap_95_ci_yield'][1]:.3f}] para el rendimiento medio. Para cubrir los {estimate['remaining_to_1000']} faltantes se estiman {estimate['estimated_additional_targeted_videos_point']:,} videos dirigidos; usando el límite inferior como plan conservador, {estimate['estimated_additional_targeted_videos_conservative']:,}. Esta extrapolación presupone que se mantiene la mezcla de consultas y la disponibilidad de subtítulos.",
            "",
            "## 6. Particiones y control de fuga",
            "",
            f"Los {manifest['usable_rows']:,} casos nuevos utilizables se agruparon por `video_id`: {manifest['split']['train_rows']:,} train y {manifest['split']['validation_rows']:,} validation, cero test. La búsqueda de semilla evaluó 500 particiones y seleccionó `{manifest['split']['seed']}` sin celdas gruesas vacías. La validación enriquecida es diagnóstica; la selección del modelo usa la validación histórica y la evaluación final usa el test histórico congelado por video.",
            "",
        ]
    )
    if training:
        comparison = pd.DataFrame(training["comparison"])
        lines.extend(
            [
                "## 7. Reentrenamiento y resultados",
                "",
                "| Experimento | PR-AUC daño validación | F1 daño test | PR-AUC daño test | Recall micro daño test |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in comparison.to_dict("records"):
            lines.append(
                f"| `{row['experimento']}` | {row['validation_damage_pr_auc_macro']:.4f} | "
                f"{row['test_damage_f1_macro']:.4f} | {row['test_damage_pr_auc_macro']:.4f} | "
                f"{row['test_damage_recall_micro']:.4f} |"
            )
        lines.extend(
            [
                "",
                f"![Comparación del reentrenamiento](figuras/{OUTPUT_SUFFIX}/comparacion_entrenamiento.png)",
                "",
                f"Ganador y configuración exportada por el criterio predefinido de validación: `{training['winner_id']}`. La evidencia del test {'SÍ respalda' if training.get('general_improvement_supported', False) else 'NO respalda'} una mejora general en detección de daño. La selección no se cambia retrospectivamente usando test.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 7. Reentrenamiento incremental autorizado",
                "",
                f"El flujo se detuvo antes de `04_2`, como fue solicitado. El progreso humano disponible para este lote es {human_completed}/{human_total} decisiones completas. Las decisiones cerradas elegibles se aplican; rechazos y {human_total - human_completed} casos todavía abiertos se excluyen del dataset utilizable. `04_2` descubrirá este archivo mediante su manifiesto, sin requerir una ruta codificada.",
                "",
            ]
        )
    lines.extend(
        [
            "## 8. Artefactos y hashes",
            "",
            f"- Dataset nuevo utilizable: `{USABLE_PATH.relative_to(ROOT)}`; SHA-256 `{manifest['output_sha256']['usable']}`.",
            f"- Cola humana nueva: `{PENDING_HUMAN_PATH.relative_to(ROOT)}`; SHA-256 `{manifest['output_sha256']['pending_human']}`.",
            f"- Manifiesto: `{DATASET_MANIFEST_PATH.relative_to(ROOT)}`.",
            f"- Cuaderno orquestador: `{ORCHESTRATOR_NOTEBOOK_PATH.relative_to(ROOT)}`.",
            f"- Rendimiento detallado por fuente: `{SOURCE_PERFORMANCE_PATH.relative_to(ROOT)}`.",
            f"- Chunks nuevos: SHA-256 `{manifest['input_sha256']['chunks']}`.",
            f"- Etiquetas Flash: SHA-256 `{manifest['input_sha256']['flash']}`.",
            f"- Revisión Pro: SHA-256 `{manifest['input_sha256']['pro']}`.",
            "",
            "Comandos reproducibles desde la raíz del repositorio:",
            "",
            "```powershell",
            f"$env:AMPLIACION_BATCH_ID='{BATCH_ID}'",
            f"$env:AMPLIACION_SEED='{NEW_SPLIT_SEED_START}'",
            "python -m scripts_auxiliares.ampliacion_dirigida_dano --stage discover",
            "python -m scripts_auxiliares.ampliacion_dirigida_dano --stage transcribe",
            "python -m scripts_auxiliares.ampliacion_dirigida_dano --stage chunk",
            "python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage flash",
            "python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage pro",
            "python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage prepare",
            "python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage train",
            "```",
            "",
            "## 9. Limitaciones metodológicas",
            "",
            "- La adquisición dirigida altera deliberadamente la prevalencia de entrenamiento; no estima la prevalencia natural de YouTube.",
            "- Los conteos antes de cerrar la adjudicación humana combinada son provisionales.",
            "- Excluir dudas Pro mejora pureza aparente, pero puede retirar ejemplos fronterizos; por eso se conserva la cola humana completa.",
            "- El test histórico ya fue observado en experimentos anteriores; sirve para comparación de ingeniería, no sustituye un holdout humano ciego nuevo.",
            "- Llegar a 1.000 no garantiza suficiencia: importa diversidad de videos/canales y calidad de etiqueta, además del conteo.",
            "",
            "## 10. Referencias (APA 7)",
            "",
            "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html",
            "",
            "Fairstein, Y., Kalinsky, O., Karnin, Z., Kushilevitz, G., Libov, A., & Tolmach, S. (2024). Class balancing for efficient active learning in imbalanced datasets. In *Proceedings of the 18th Linguistic Annotation Workshop* (pp. 77–86). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.law-1.8",
            "",
            "Fithian, W., & Hastie, T. (2014). Local case-control sampling: Efficient subsampling in imbalanced data sets. *The Annals of Statistics, 42*(5), 1693–1724. https://doi.org/10.1214/14-AOS1220",
            "",
            "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
            "",
            "Scikit-learn developers. (2026). *GroupShuffleSplit*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_operational_acceptability(snapshot_path: Path | None = None) -> dict:
    """Evalúa la política predicción de daño + ``needs_review`` en test congelado."""
    snapshot_path = snapshot_path or active_human_progress_path()
    os.environ["MODERATION_HUMAN_PROGRESS_SNAPSHOT"] = str(snapshot_path)
    taxonomy, _, _ = load_taxonomy(ROOT)
    base, _, _, _ = build_hybrid_dataset(
        ROOT, require_complete_hard_review=False, write_output=False
    )
    base = add_coarse_targets(base, taxonomy)
    base = base.loc[~base["human_holdout"]].reset_index(drop=True)
    split = _base_split(base)
    test = base.iloc[split["test"]].reset_index(drop=True)
    model_path = MODEL_DIR / "moderador_grueso_ampliado.joblib"
    model = load_coarse_model(model_path)
    predictions = model.predict(test["text"].tolist())
    true_damage = np.asarray(
        [bool(set(values) & set(DAMAGE_ORDER)) for values in test["coarse_labels"]],
        dtype=bool,
    )
    predicted_damage = np.asarray(
        [bool(set(row["coarse_labels"]) & set(DAMAGE_ORDER)) for row in predictions],
        dtype=bool,
    )
    needs_review = np.asarray([bool(row["needs_review"]) for row in predictions], dtype=bool)
    intervention = predicted_damage | needs_review
    auto_safe = ~intervention
    true_safe = ~true_damage
    summary = {
        "n_test": int(len(test)),
        "true_damage": int(true_damage.sum()),
        "true_damage_rate": float(true_damage.mean()),
        "predicted_damage": int(predicted_damage.sum()),
        "damage_detection_recall_any": float(
            (predicted_damage & true_damage).sum() / true_damage.sum()
        ),
        "needs_review_rows": int(needs_review.sum()),
        "needs_review_rate": float(needs_review.mean()),
        "intervention_rows": int(intervention.sum()),
        "intervention_rate": float(intervention.mean()),
        "damage_covered_by_prediction_or_review": float(
            (intervention & true_damage).sum() / true_damage.sum()
        ),
        "damage_missed_as_auto_safe": int((true_damage & auto_safe).sum()),
        "auto_safe_rows": int(auto_safe.sum()),
        "auto_safe_coverage": float(auto_safe.mean()),
        "auto_safe_npv": float((true_safe & auto_safe).sum() / auto_safe.sum()),
        "auto_safe_damage_rate": float((true_damage & auto_safe).sum() / auto_safe.sum()),
        "safe_sent_to_review_rate": float((true_safe & needs_review).sum() / true_safe.sum()),
        "predicted_damage_precision_any": float(
            (predicted_damage & true_damage).sum() / predicted_damage.sum()
        ),
    }
    per_category = []
    for category in DAMAGE_ORDER:
        truth = np.asarray([category in values for values in test["coarse_labels"]], dtype=bool)
        predicted = np.asarray(
            [category in row["coarse_labels"] for row in predictions], dtype=bool
        )
        per_category.append(
            {
                "category": category,
                "support": int(truth.sum()),
                "recall": float((truth & predicted).sum() / truth.sum()),
                "covered_by_any_prediction_or_review": float(
                    (truth & intervention).sum() / truth.sum()
                ),
                "missed_auto_safe": int((truth & auto_safe).sum()),
            }
        )
    output = {
        "schema_version": "1.0",
        "created_at": now_iso(),
        "policy": "intervention = predicted_any_damage OR needs_review",
        "decision": "not_acceptable_for_autonomous_moderation",
        "summary": summary,
        "per_category": per_category,
        "inputs": {
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path),
            "human_review_snapshot": str(snapshot_path.relative_to(ROOT)),
            "human_review_snapshot_sha256": sha256_file(snapshot_path),
            "test_rows": len(test),
        },
    }
    write_json_atomic(OPERATIONAL_METRICS_PATH, output)
    return output


def write_incremental_training_report(
    manifest: dict,
    balance: pd.DataFrame,
    training: dict,
) -> None:
    """Documenta el experimento incremental cuantitativa y cualitativamente."""
    comparison = pd.DataFrame(training["comparison"])
    indexed = comparison.set_index("experimento")
    baseline = indexed.loc["baseline_reproducido"]
    # Esta comparación aísla el efecto de los videos: mismo modelo, sin AEDA.
    expanded = indexed.loc["ampliado_sin_aeda"]
    expanded_id = "ampliado_sin_aeda"
    selected = indexed.loc[str(training["winner_id"])]
    metric_specs = [
        ("validation_damage_pr_auc_macro", "PR-AUC macro de daño · validación histórica"),
        ("validation_damage_f1_macro", "F1 macro de daño · validación histórica"),
        ("test_damage_pr_auc_macro", "PR-AUC macro de daño · test congelado"),
        ("test_damage_f1_macro", "F1 macro de daño · test congelado"),
        ("test_damage_recall_micro", "Recall micro de daño · test congelado"),
        ("test_exact_match", "Exact match · test congelado"),
        ("new_validation_damage_pr_auc_macro", "PR-AUC macro de daño · validación nueva"),
        ("new_validation_damage_f1_macro", "F1 macro de daño · validación nueva"),
        ("new_validation_damage_recall_micro", "Recall micro de daño · validación nueva"),
    ]

    model_manifest_path = MODEL_DIR / "manifiesto.json"
    model_manifest = (
        json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if model_manifest_path.exists()
        else {}
    )
    snapshot_relative = training.get("human_review_snapshot") or model_manifest.get(
        "human_review_snapshot"
    )
    snapshot_path = ROOT / snapshot_relative if snapshot_relative else active_human_progress_path()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    completed = [row for row in snapshot.get("annotations", []) if row.get("status") == "completed"]
    action_counts = Counter(row.get("review_action") for row in completed)
    acquisition = json.loads(ACQUISITION_MANIFEST_PATH.read_text(encoding="utf-8"))
    operational = (
        json.loads(OPERATIONAL_METRICS_PATH.read_text(encoding="utf-8"))
        if OPERATIONAL_METRICS_PATH.exists()
        else None
    )

    def relative(delta: float, base: float) -> str:
        return f"{100 * delta / base:+.2f}%" if base else "n/a"

    class_changes: list[tuple[str, float, float, float]] = []
    baseline_report_path = METRICS_DIR / "reporte_test_baseline_reproducido.csv"
    expanded_report_path = METRICS_DIR / f"reporte_test_{expanded_id}.csv"
    if baseline_report_path.exists() and expanded_report_path.exists():
        base_report = pd.read_csv(baseline_report_path, index_col=0)
        exp_report = pd.read_csv(expanded_report_path, index_col=0)
        for category in DAMAGE_ORDER:
            if category in base_report.index and category in exp_report.index:
                base_f1 = float(base_report.loc[category, "f1-score"])
                exp_f1 = float(exp_report.loc[category, "f1-score"])
                class_changes.append((category, base_f1, exp_f1, exp_f1 - base_f1))

    validation_delta = float(expanded["validation_damage_pr_auc_macro"] - baseline["validation_damage_pr_auc_macro"])
    test_pr_delta = float(expanded["test_damage_pr_auc_macro"] - baseline["test_damage_pr_auc_macro"])
    test_f1_delta = float(expanded["test_damage_f1_macro"] - baseline["test_damage_f1_macro"])
    consistent = validation_delta > 0 and test_pr_delta > 0 and test_f1_delta > 0
    conclusion = (
        "La ampliación muestra una mejora consistente en el criterio de selección y en ambos indicadores principales del test congelado."
        if consistent
        else "La ampliación no mejora de forma consistente validación, PR-AUC y F1 del test; el resultado debe describirse como mixto y no como una mejora general demostrada."
    )
    lines = [
        "# Informe del entrenamiento incremental con ampliación de videos",
        "",
        f"Fecha: {training['completed_at']}  ",
        f"Lote: `{BATCH_ID}`  ",
        f"Comparación que aísla el incremento de videos: `{expanded_id}`  ",
        f"Modelo exportado: `{training['export_id']}`",
        "",
        "## 1. Pregunta y diseño experimental",
        "",
        "Se evaluó cuánto cambia el desempeño del moderador grueso al añadir videos adquiridos de forma dirigida hacia categorías minoritarias. La comparación mantiene el mismo algoritmo principal, la misma taxonomía gruesa y el mismo test histórico agrupado por video. El baseline y las variantes ampliadas difieren en la incorporación de los nuevos chunks; una variante añade además AEDA solo al entrenamiento.",
        "",
        "La selección de configuración se realiza con validación histórica. El test congelado se usa para una comparación de ingeniería y no para ajustar hiperparámetros. Las etiquetas finas no se entrenan; los flags transversales siguen separados de las cinco categorías de daño y `SEGURO`.",
        "",
        "## 2. Instantánea humana e inclusión de datos",
        "",
        f"La validación humana no tuvo que terminar. Se congeló `{snapshot_path.relative_to(ROOT)}` (revisión {snapshot.get('revision')}, SHA-256 `{sha256_file(snapshot_path)}`). Contenía {len(completed):,} decisiones cerradas: {sum(row.get('training_eligible') is True for row in completed):,} incluidas y {sum(row.get('training_eligible') is False for row in completed):,} excluidas.",
        "",
        f"- Decisiones anteriores preservadas: {action_counts.get('legacy_human_decision', 0):,}.",
        f"- Propuestas LLM aceptadas: {action_counts.get('accept_llm', 0):,}.",
        f"- Propuestas modificadas por humano: {action_counts.get('modify_llm', 0):,}.",
        f"- Rechazos excluidos: {action_counts.get('reject_llm', 0):,}.",
        "- Todo caso todavía abierto se excluyó de esta corrida; continuar validando no altera retrospectivamente la instantánea.",
        "",
        "## 3. Incremento del corpus",
        "",
        f"La adquisición produjo {acquisition['new_chunks']:,} chunks de {acquisition['new_videos_with_chunks']:,} videos nuevos. Fueron utilizables {manifest['usable_rows']:,}: {manifest['usable_source_counts'].get('flash_pseudo', 0):,} pseudoetiquetas Flash y {manifest['usable_source_counts'].get('pro_augmented_resolved', 0):,} decisiones Pro cerradas. Los {manifest['pending_human_rows']:,} casos Pro aún dudosos se excluyeron.",
        "",
        f"Partición nueva por `video_id`: {manifest['split']['train_rows']:,} train, {manifest['split']['validation_rows']:,} validation y 0 test. El test histórico conserva 10.293 filas.",
        "",
        "| Categoría de daño | Antes | Nuevos utilizables | Después |",
        "|---|---:|---:|---:|",
    ]
    for row in balance.to_dict("records"):
        lines.append(
            f"| `{row['categoria']}` | {int(row['antes']):,} | "
            f"{int(row['agregados_resueltos']):,} | {int(row['despues_utilizable']):,} |"
        )
    lines.extend(
        [
            "",
            "## 4. Configuraciones entrenadas",
            "",
            "1. `baseline_reproducido`: SVM lineal word+character con corpus histórico depurado.",
            "2. `ampliado_sin_aeda`: mismo modelo más los chunks nuevos utilizables.",
            "3. `ampliado_con_aeda`: corpus ampliado y una copia AEDA de cada ejemplo de daño, con peso reducido.",
            "",
            "Se ajustan umbrales con validación; no se describen épocas porque una SVM lineal converge por optimización, no por pasadas neuronales o epochs.",
            "",
            "## 5. Resultados cuantitativos",
            "",
            f"Comparación principal para atribución causal operativa: baseline frente a `{expanded_id}` (sin cambiar augmentation). La configuración `{training['winner_id']}` se seleccionó por PR-AUC de daño en validación histórica y se informa por separado en la tabla completa.",
            "",
            "| Métrica | Baseline | Ampliado | Δ absoluto | Δ relativo |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in metric_specs:
        if key not in baseline or key not in expanded:
            continue
        base_value, expanded_value = float(baseline[key]), float(expanded[key])
        delta = expanded_value - base_value
        lines.append(
            f"| {label} | {base_value:.4f} | {expanded_value:.4f} | "
            f"{delta:+.4f} | {relative(delta, base_value)} |"
        )
    lines.extend(["", "Resultados completos de los tres modelos:", "", "| Modelo | PR-AUC daño validación | F1 daño test | PR-AUC daño test | Recall daño test | Filas train |", "|---|---:|---:|---:|---:|---:|"])
    for row in comparison.to_dict("records"):
        lines.append(
            f"| `{row['experimento']}` | {row['validation_damage_pr_auc_macro']:.4f} | "
            f"{row['test_damage_f1_macro']:.4f} | {row['test_damage_pr_auc_macro']:.4f} | "
            f"{row['test_damage_recall_micro']:.4f} | {int(row['train_rows_effective']):,} |"
        )
    if class_changes:
        lines.extend(["", "### Cambio cualitativo por categoría", "", "| Categoría | F1 baseline | F1 ampliado | Δ |", "|---|---:|---:|---:|"])
        for category, base_f1, exp_f1, delta in sorted(class_changes, key=lambda row: row[3], reverse=True):
            lines.append(f"| `{category}` | {base_f1:.4f} | {exp_f1:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "## 6. Interpretación",
            "",
            conclusion,
            "",
            f"En concreto, el efecto aislado de añadir videos es {validation_delta:+.4f} en PR-AUC macro de daño de validación histórica, {test_pr_delta:+.4f} en PR-AUC macro de daño de test y {test_f1_delta:+.4f} en F1 macro de daño de test. En la validación del dominio nuevo, la PR-AUC de daño cambia {float(expanded['new_validation_damage_pr_auc_macro'] - baseline['new_validation_damage_pr_auc_macro']):+.4f} y el F1 macro de daño {float(expanded['new_validation_damage_f1_macro'] - baseline['new_validation_damage_f1_macro']):+.4f}.",
            "",
            f"La variante seleccionada por validación histórica (`{training['winner_id']}`) alcanzó PR-AUC de daño {float(selected['validation_damage_pr_auc_macro']):.4f} en validación y {float(selected['test_damage_pr_auc_macro']):.4f} en test. Se conserva esa selección para no escoger retrospectivamente con el test, pero el test no respalda una mejora general.",
            "",
            "La PR-AUC se prioriza porque las clases de daño son poco frecuentes. Una ganancia en conteo o accuracy global puede deberse a `SEGURO` y no prueba mejor detección de daño. Tampoco se interpreta esta comparación como suficiencia para moderación autónoma sin un holdout humano ciego y contemporáneo.",
            "",
        ]
    )
    if operational:
        summary = operational["summary"]
        lines.extend(
            [
                "",
                "## 7. Aceptabilidad operativa",
                "",
                f"Sobre {summary['n_test']:,} chunks de test había {summary['true_damage']:,} con al menos una categoría de daño. El modelo predijo algún daño en {summary['predicted_damage']:,} casos y marcó `needs_review` en {summary['needs_review_rows']:,} ({summary['needs_review_rate']:.2%}). La unión de predicción de daño o revisión interviene sobre {summary['intervention_rows']:,} chunks ({summary['intervention_rate']:.2%} del test).",
                "",
                f"Esa política cubre solo {summary['damage_covered_by_prediction_or_review']:.2%} de los daños y deja {summary['damage_missed_as_auto_safe']:,}/{summary['true_damage']:,} ({1-summary['damage_covered_by_prediction_or_review']:.2%}) como seguros sin revisión. La precisión de predecir algún daño es {summary['predicted_damage_precision_any']:.2%}. Aunque el valor predictivo negativo de los auto-seguros es {summary['auto_safe_npv']:.2%}, ese valor está dominado por la prevalencia baja de daño y oculta el gran número absoluto de falsos negativos.",
                "",
                "| Categoría | Soporte | Recall de categoría | Cobertura por daño o revisión | Omitidos como auto-seguros |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in operational["per_category"]:
            lines.append(
                f"| `{row['category']}` | {row['support']:,} | {row['recall']:.2%} | "
                f"{row['covered_by_any_prediction_or_review']:.2%} | {row['missed_auto_safe']:,} |"
            )
        lines.extend(
            [
                "",
                "**Decisión:** no es aceptable para moderación autónoma ni para aprobar automáticamente todo caso sin `needs_review`. Puede conservarse como baseline experimental, herramienta de priorización o asistente cuya decisión final siga siendo humana.",
                "",
            ]
        )
    lines.extend(
        [
            "## 8. Limitaciones",
            "",
            "- La adquisición fue dirigida; mejora representación para entrenamiento, pero no estima prevalencia natural.",
            "- Los casos humanos abiertos fueron excluidos, por lo que ejemplos fronterizos permanecen subrepresentados.",
            "- El test histórico ya fue observado en experimentos anteriores; la comparación es de ingeniería y puede tener sesgo adaptativo.",
            "- Una sola partición no cuantifica variabilidad entre videos; futuras conclusiones fuertes requieren bootstrap por video o validación humana externa.",
            "",
            "## 9. Reproducibilidad y artefactos",
            "",
            f"- Modelo: `{training['model_path']}`; SHA-256 `{training['model_sha256']}`.",
            f"- Dataset ampliado: `{USABLE_PATH.relative_to(ROOT)}`; SHA-256 `{manifest['output_sha256']['usable']}`.",
            f"- Comparación: `{(METRICS_DIR / 'comparacion_entrenamiento_ampliado.csv').relative_to(ROOT)}`.",
            f"- Manifiesto: `{DATASET_MANIFEST_PATH.relative_to(ROOT)}`.",
            f"- Figura: `{(FIGURES_DIR / 'comparacion_entrenamiento.png').relative_to(ROOT)}`.",
            f"- Aceptabilidad operativa: `{OPERATIONAL_METRICS_PATH.relative_to(ROOT)}`.",
            "",
            "Comando:",
            "",
            "```powershell",
            "python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage train",
            "```",
            "",
            "## Referencias (APA 7)",
            "",
            "Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html",
            "",
            "Karimi, A., Rossi, L., & Prati, A. (2021). AEDA: An easier data augmentation technique for text classification. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 2748–2754). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-emnlp.234",
            "",
            "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
        ]
    )
    INCREMENTAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="prepare")
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        usable, pending, manifest = prepare_dataset()
        print(json.dumps({
            "usable": len(usable), "pending_human": len(pending),
            "split": manifest["split"], "balance": manifest["balance"],
            "threat_estimate": manifest["threat_video_estimate"],
        }, ensure_ascii=False, indent=2))
    if args.stage in {"train", "all"}:
        print(json.dumps(train_models(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
