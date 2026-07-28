"""Utilidades reproducibles para el flujo híbrido del cuaderno 04.

El flujo usa anotaciones Pro cuando existen y Flash como pseudoetiqueta en el
resto. Cualquier consenso humano se reserva como evaluación externa. Las
categorías semánticas y los flags transversales se modelan con cabezas separadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit


SAFE_LABELS = {"seguro", "seguro_ironia_marcada"}
DEFAULT_RECALIBRATED_THRESHOLD = 0.90
HUMAN_COARSE_ORDER = [
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ACOSO_GENERO_IDENTIDAD",
    "ACOSO_PERSONAL",
    "AMENAZA_DIRECTA",
    "CONTENIDO_SEXUAL",
]
HUMAN_DAMAGE_ORDER = HUMAN_COARSE_ORDER[1:]
EXPECTED_HARD_REVIEW_ROWS = 139


def find_project_root(start: Path | None = None) -> Path:
    """Encuentra la raíz sin depender del directorio desde el que abre Jupyter."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "datos" / "processed" / "chunks_para_etiquetar.jsonl").exists()
            and (candidate / "datos" / "processed" / "taxonomia_moderacion.csv").exists()
        ):
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto ni los datos canónicos.")


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


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_taxonomy(root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    path = root / "datos" / "processed" / "taxonomia_moderacion.csv"
    taxonomy = pd.read_csv(path).fillna("")
    required = {"label", "categoria"}
    if not required <= set(taxonomy.columns):
        raise ValueError(f"La taxonomía no contiene {sorted(required)}: {path}")
    if taxonomy["label"].duplicated().any():
        raise ValueError("La taxonomía contiene labels duplicados.")
    label_order = taxonomy.loc[taxonomy["categoria"] != "FLAG", "label"].tolist()
    flag_order = taxonomy.loc[taxonomy["categoria"] == "FLAG", "label"].tolist()
    if len(label_order) != 14 or len(flag_order) != 3:
        raise ValueError(
            f"Se esperaban 14 categorías y 3 flags; se obtuvieron {len(label_order)} y {len(flag_order)}."
        )
    return taxonomy, label_order, flag_order


def _validate_annotation(
    row: dict,
    allowed_labels: set[str],
    allowed_flags: set[str],
    source: str,
) -> None:
    chunk_id = row.get("chunk_id")
    labels = row.get("labels")
    flags = row.get("flags", [])
    if not chunk_id:
        raise ValueError(f"{source}: chunk_id vacío.")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{source}/{chunk_id}: labels debe ser una lista no vacía.")
    if not isinstance(flags, list):
        raise ValueError(f"{source}/{chunk_id}: flags debe ser una lista.")
    unknown_labels = set(labels) - allowed_labels
    unknown_flags = set(flags) - allowed_flags
    if unknown_labels or unknown_flags:
        raise ValueError(
            f"{source}/{chunk_id}: fuera de taxonomía; labels={unknown_labels}, flags={unknown_flags}."
        )
    safe = set(labels) & SAFE_LABELS
    damage = set(labels) - SAFE_LABELS
    if safe and damage:
        raise ValueError(f"{source}/{chunk_id}: seguro no puede coexistir con daño.")
    if len(safe) > 1:
        raise ValueError(f"{source}/{chunk_id}: las dos etiquetas seguras no pueden coexistir.")
    if flags and not damage:
        raise ValueError(f"{source}/{chunk_id}: un flag transversal requiere una categoría de daño.")


def _unique_by_id(rows: list[dict], source: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        chunk_id = row.get("chunk_id")
        if chunk_id in result:
            raise ValueError(f"{source}: chunk_id duplicado: {chunk_id}")
        result[chunk_id] = row
    return result


def _validate_human_coarse_annotation(
    row: dict,
    allowed_ids: set[str],
    source: str,
) -> None:
    """Valida una adjudicación gruesa o una exclusión humana explícita."""
    chunk_id = str(row.get("chunk_id") or "")
    labels = row.get("coarse_labels", [])
    flags = row.get("flags", [])
    if chunk_id not in allowed_ids:
        raise ValueError(f"{source}: chunk_id fuera de la campaña: {chunk_id}")
    if row.get("status") != "completed" or bool(row.get("needs_review")):
        raise ValueError(f"{source}/{chunk_id}: la decisión humana no está cerrada.")
    action = row.get("review_action", "legacy_human_decision")
    training_eligible = bool(row.get("training_eligible", True))
    allowed_actions = {
        "accept_llm", "reject_llm", "modify_llm", "legacy_human_decision"
    }
    if action not in allowed_actions:
        raise ValueError(f"{source}/{chunk_id}: review_action inválida: {action}")
    if action == "reject_llm" and training_eligible:
        raise ValueError(f"{source}/{chunk_id}: un rechazo no puede entrar al entrenamiento.")
    if action != "reject_llm" and not training_eligible:
        raise ValueError(f"{source}/{chunk_id}: una decisión incluida no puede marcarse inelegible.")
    if not isinstance(labels, list) or len(labels) != len(set(labels)):
        raise ValueError(f"{source}/{chunk_id}: coarse_labels debe ser una lista sin duplicados.")
    if training_eligible and not labels:
        raise ValueError(f"{source}/{chunk_id}: una decisión incluida requiere categoría gruesa.")
    if not training_eligible and (labels or flags):
        raise ValueError(f"{source}/{chunk_id}: una exclusión debe tener categorías y flags vacíos.")
    if not set(labels) <= set(HUMAN_COARSE_ORDER):
        raise ValueError(f"{source}/{chunk_id}: coarse_labels fuera de taxonomía.")
    if not isinstance(flags, list) or len(flags) != len(set(flags)):
        raise ValueError(f"{source}/{chunk_id}: flags debe ser una lista sin duplicados.")
    if "SEGURO" in labels and len(labels) > 1:
        raise ValueError(f"{source}/{chunk_id}: SEGURO no puede coexistir con daño.")
    if flags and not (set(labels) & set(HUMAN_DAMAGE_ORDER)):
        raise ValueError(f"{source}/{chunk_id}: un flag requiere una categoría gruesa de daño.")


def _load_hard_review_stage(
    root: Path,
    allowed_labels: set[str],
    allowed_flags: set[str],
    require_complete: bool,
) -> tuple[dict[str, dict], dict[str, dict], set[str], dict[str, str | None]]:
    """Carga Pro-2000 y adjudicaciones humanas finales o parciales.

    Si la campaña aún no terminó y ``require_complete`` es falso, aplica solo
    decisiones cerradas de una instantánea de progreso y devuelve los IDs aún
    pendientes para excluirlos del entrenamiento.
    """
    api_dir = root / "datos" / "etiquetado" / "llm_api"
    human_dir = root / "datos" / "etiquetado" / "humano"
    hard_pro_path = api_dir / "deepseek-v4-pro_revision_sospechosos_gruesos_seed42.jsonl"
    human_final_path = human_dir / "revision_humana_sospechosos_139.jsonl"
    human_manifest_path = human_dir / "revision_humana_sospechosos_139.manifest.json"
    hashes: dict[str, str | None] = {
        "pro_hard_2000": None,
        "human_hard_139": None,
        "human_hard_139_manifest": None,
        "human_review_progress_snapshot": None,
    }
    if not hard_pro_path.exists():
        return {}, {}, set(), hashes

    hard_rows = read_jsonl(hard_pro_path)
    hard_by_id = _unique_by_id(hard_rows, "Pro-2000")
    if len(hard_by_id) != 2_000:
        raise ValueError(
            f"Pro-2000 debe contener exactamente 2.000 IDs; contiene {len(hard_by_id)}."
        )
    for row in hard_rows:
        _validate_annotation(row, allowed_labels, allowed_flags, "Pro-2000")
    persistent_ids = {
        chunk_id for chunk_id, row in hard_by_id.items() if bool(row.get("needs_review"))
    }
    if len(persistent_ids) != EXPECTED_HARD_REVIEW_ROWS:
        raise ValueError(
            "La revisión Pro-2000 no reproduce los 139 casos de duda persistente: "
            f"se encontraron {len(persistent_ids)}."
        )
    hashes["pro_hard_2000"] = sha256_file(hard_pro_path)

    if not human_final_path.exists():
        if require_complete:
            raise RuntimeError(
                "REENTRENAMIENTO BLOQUEADO: faltan las 139 adjudicaciones humanas finales. "
                "Complete la campaña en http://127.0.0.1:8765/; el servidor creará "
                f"{human_final_path.relative_to(root)} solo al cerrar todos los casos."
            )
        snapshot_env = os.environ.get("MODERATION_HUMAN_PROGRESS_SNAPSHOT")
        progress_path = (
            Path(snapshot_env)
            if snapshot_env
            else human_dir / "revision_humana_combinada_1918.progress.json"
        )
        if not progress_path.exists():
            legacy_progress = human_dir / "revision_humana_sospechosos_139.progress.json"
            progress_path = legacy_progress if legacy_progress.exists() else progress_path
        human_by_id: dict[str, dict] = {}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_rows = [
                row for row in progress.get("annotations", [])
                if row.get("status") == "completed" and row.get("chunk_id") in persistent_ids
            ]
            human_by_id = _unique_by_id(completed_rows, "humano-139-parcial")
            for row in completed_rows:
                _validate_human_coarse_annotation(row, persistent_ids, "humano-139-parcial")
            hashes["human_review_progress_snapshot"] = sha256_file(progress_path)
        return hard_by_id, human_by_id, persistent_ids - set(human_by_id), hashes

    human_rows = read_jsonl(human_final_path)
    human_by_id = _unique_by_id(human_rows, "humano-139")
    missing = persistent_ids - set(human_by_id)
    extra = set(human_by_id) - persistent_ids
    if len(human_by_id) != EXPECTED_HARD_REVIEW_ROWS or missing or extra:
        raise ValueError(
            "La salida humana no cubre exactamente las 139 dudas de Pro: "
            f"filas={len(human_by_id)}, faltan={len(missing)}, sobran={len(extra)}."
        )
    for row in human_rows:
        _validate_human_coarse_annotation(row, persistent_ids, "humano-139")
    split_counts = pd.Series([row.get("split") for row in human_rows]).value_counts().to_dict()
    if split_counts != {"train": 114, "validation": 25}:
        raise ValueError(
            "La adjudicación humana no conserva la partición auditada "
            f"(train=114, validation=25, test=0): {split_counts}."
        )
    if not human_manifest_path.exists():
        raise FileNotFoundError(
            "Existe la salida humana, pero falta su manifiesto de trazabilidad: "
            f"{human_manifest_path}"
        )
    manifest = json.loads(human_manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("completed", 0)) != EXPECTED_HARD_REVIEW_ROWS:
        raise ValueError("El manifiesto humano no confirma 139 decisiones completas.")
    if manifest.get("output_sha256") != sha256_file(human_final_path):
        raise ValueError("El SHA-256 de la salida humana no coincide con su manifiesto.")
    hashes["human_hard_139"] = sha256_file(human_final_path)
    hashes["human_hard_139_manifest"] = sha256_file(human_manifest_path)
    return hard_by_id, human_by_id, set(), hashes


def build_hybrid_dataset(
    root: Path,
    recalibrated_threshold: float = DEFAULT_RECALIBRATED_THRESHOLD,
    flash_weight: float = 0.50,
    include_hard_review: bool = True,
    require_complete_hard_review: bool = True,
    write_output: bool = True,
) -> tuple[pd.DataFrame, dict, list[str], list[str]]:
    """Construye humano-grueso > Pro > Flash y reserva el holdout externo.

    Un rechazo humano se conserva en la salida de auditoría, pero se omite por
    completo del DataFrame destinado a desarrollo y entrenamiento.
    """
    _, label_order, flag_order = load_taxonomy(root)
    allowed_labels, allowed_flags = set(label_order), set(flag_order)
    processed = root / "datos" / "processed"
    api_dir = root / "datos" / "etiquetado" / "llm_api"
    chunks_path = processed / "chunks_para_etiquetar.jsonl"
    flash_path = api_dir / "deepseek-v4-flash_labeled_chunks_seed42.jsonl"
    original_pro_path = api_dir / "deepseek-v4-pro_revision_de_deepseek-v4-flash_seed42.jsonl"
    recalibrated_pro_path = api_dir / "deepseek-v4-pro_revision_umbral_recalibrado_t090_seed42.jsonl"
    human_path = processed / "dataset_etiquetado.jsonl"

    required = [chunks_path, flash_path, original_pro_path, recalibrated_pro_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan entradas del flujo híbrido:\n- " + "\n- ".join(missing))

    chunks = read_jsonl(chunks_path)
    chunk_by_id = _unique_by_id(chunks, "canónico")
    flash_rows = read_jsonl(flash_path)
    flash_by_id = _unique_by_id(flash_rows, "Flash")
    if set(chunk_by_id) != set(flash_by_id):
        missing_flash = set(chunk_by_id) - set(flash_by_id)
        extra_flash = set(flash_by_id) - set(chunk_by_id)
        raise ValueError(
            f"Flash no cubre exactamente el canónico: faltan={len(missing_flash)}, sobran={len(extra_flash)}."
        )

    pro_by_id: dict[str, dict] = {}
    pro_paths = [original_pro_path, recalibrated_pro_path]
    for pro_path in pro_paths:
        for row in read_jsonl(pro_path):
            chunk_id = row.get("chunk_id")
            if chunk_id in pro_by_id:
                raise ValueError(f"Pro aparece más de una vez para {chunk_id}.")
            pro_by_id[chunk_id] = row
    if not set(pro_by_id) <= set(chunk_by_id):
        raise ValueError("Pro contiene IDs ajenos al canónico.")

    for row in flash_rows:
        _validate_annotation(row, allowed_labels, allowed_flags, "Flash")
    for row in pro_by_id.values():
        _validate_annotation(row, allowed_labels, allowed_flags, "Pro")

    hard_pro_by_id: dict[str, dict] = {}
    human_hard_by_id: dict[str, dict] = {}
    pending_human_hard_ids: set[str] = set()
    hard_hashes: dict[str, str | None] = {}
    if include_hard_review:
        hard_pro_by_id, human_hard_by_id, pending_human_hard_ids, hard_hashes = _load_hard_review_stage(
            root,
            allowed_labels,
            allowed_flags,
            require_complete=require_complete_hard_review,
        )
        overlap = set(pro_by_id) & set(hard_pro_by_id)
        if overlap:
            raise ValueError(f"Pro previo y Pro-2000 se solapan en {len(overlap)} IDs.")
        if not set(hard_pro_by_id) <= set(chunk_by_id):
            raise ValueError("Pro-2000 contiene IDs ajenos al canónico.")

    routed_ids = {
        row["chunk_id"]
        for row in flash_rows
        if bool(row.get("needs_review"))
        or float(row.get("score_confianza", 0.0)) < recalibrated_threshold
    }
    pending_routed = routed_ids - set(pro_by_id)
    if pending_routed:
        raise ValueError(
            f"Hay {len(pending_routed)} casos derivados sin Pro. Complete la sección 13.4.5 del 03_2."
        )

    human_rows = read_jsonl(human_path) if human_path.exists() else []
    human_by_id = _unique_by_id(human_rows, "humano") if human_rows else {}
    if human_rows:
        for row in human_rows:
            _validate_annotation(row, allowed_labels, allowed_flags, "humano")
        if not set(human_by_id) <= set(chunk_by_id):
            raise ValueError("El consenso humano contiene IDs ajenos al canónico.")

    hybrid_rows: list[dict] = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        flash = flash_by_id[chunk_id]
        if chunk_id in pending_human_hard_ids:
            continue
        if chunk_id in human_hard_by_id:
            if not bool(human_hard_by_id[chunk_id].get("training_eligible", True)):
                continue
            annotation = hard_pro_by_id[chunk_id]
            source = "human_coarse"
            coarse_override = human_hard_by_id[chunk_id]["coarse_labels"]
            output_flags = human_hard_by_id[chunk_id].get("flags", [])
        elif chunk_id in hard_pro_by_id:
            annotation = hard_pro_by_id[chunk_id]
            source = "pro_hard_mining"
            coarse_override = None
            output_flags = annotation.get("flags", [])
        elif chunk_id in pro_by_id:
            annotation = pro_by_id[chunk_id]
            source = "pro"
            coarse_override = None
            output_flags = annotation.get("flags", [])
        else:
            annotation = flash
            source = "flash_pseudo"
            coarse_override = None
            output_flags = annotation.get("flags", [])
        confidence = float(annotation.get("score_confianza", flash.get("score_confianza", 0.0)))
        weight = 1.0 if source != "flash_pseudo" else flash_weight * confidence
        hybrid_rows.append(
            {
                "chunk_id": chunk_id,
                "video_id": chunk.get("video_id") or chunk_id,
                "text": str(chunk.get("text") or ""),
                "labels": sorted(set(annotation.get("labels", []))),
                "flags": sorted(set(output_flags)),
                "coarse_labels_override": coarse_override,
                "fine_labels_reference_only": source == "human_coarse",
                "hard_review_split": (
                    human_hard_by_id[chunk_id].get("split")
                    if chunk_id in human_hard_by_id
                    else None
                ),
                "label_source": source,
                "sample_weight": float(weight),
                "score_confianza_source": confidence,
                "needs_review_flash": bool(flash.get("needs_review")),
                "needs_review_recalibrado": bool(flash.get("needs_review"))
                or float(flash.get("score_confianza", 0.0)) < recalibrated_threshold,
                "human_holdout": chunk_id in human_by_id,
            }
        )

    frame = pd.DataFrame(hybrid_rows)
    if frame["text"].str.strip().eq("").any():
        raise ValueError("El dataset híbrido contiene textos vacíos.")

    output_path = processed / "dataset_pseudoetiquetado_hibrido.jsonl"
    manifest_path = output_path.with_suffix(".manifest.json")
    metadata = {
        "schema_version": "3.0",
        "policy": (
            "humano grueso aceptado/modificado reemplaza Pro dudoso; rechazo humano excluye; "
            "Pro reemplaza Flash; consenso humano externo se reserva como holdout"
        ),
        "recalibrated_threshold": recalibrated_threshold,
        "flash_pseudo_base_weight": flash_weight,
        "rows": len(frame),
        "videos": int(frame["video_id"].nunique()),
        "source_counts": {key: int(value) for key, value in frame["label_source"].value_counts().items()},
        "human_holdout_rows": int(frame["human_holdout"].sum()),
        "routed_rows": len(routed_ids),
        "pro_rows": len(pro_by_id),
        "pro_hard_mining_rows": len(hard_pro_by_id),
        "human_hard_rows": sum(
            bool(row.get("training_eligible", True)) for row in human_hard_by_id.values()
        ),
        "human_hard_adjudicated_rows": len(human_hard_by_id),
        "human_hard_excluded_rows": sum(
            not bool(row.get("training_eligible", True)) for row in human_hard_by_id.values()
        ),
        "human_hard_pending_excluded_rows": len(pending_human_hard_ids),
        "hard_review_complete": bool(hard_pro_by_id) and (
            not pending_human_hard_ids
            and len(human_hard_by_id) == EXPECTED_HARD_REVIEW_ROWS
        ),
        "label_order": label_order,
        "flag_order": flag_order,
        "input_sha256": {
            "chunks": sha256_file(chunks_path),
            "flash": sha256_file(flash_path),
            "pro_original": sha256_file(original_pro_path),
            "pro_recalibrated": sha256_file(recalibrated_pro_path),
            "human": sha256_file(human_path) if human_path.exists() else None,
            **hard_hashes,
        },
    }
    if write_output:
        export_columns = [
            "chunk_id",
            "video_id",
            "labels",
            "flags",
            "coarse_labels_override",
            "fine_labels_reference_only",
            "hard_review_split",
            "label_source",
            "sample_weight",
            "score_confianza_source",
            "needs_review_flash",
            "needs_review_recalibrado",
            "human_holdout",
        ]
        write_jsonl(output_path, frame[export_columns].to_dict("records"))
        manifest_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return frame, metadata, label_order, flag_order


def load_human_holdout(
    root: Path,
    label_order: list[str],
    flag_order: list[str],
) -> pd.DataFrame:
    """Carga el consenso humano con el texto canónico para evaluación externa.

    Devuelve un DataFrame vacío cuando la validación humana aún no existe. Los
    IDs se validan contra el corpus canónico y nunca se mezclan con las
    pseudoetiquetas de entrenamiento.
    """
    processed = root / "datos" / "processed"
    human_path = processed / "dataset_etiquetado.jsonl"
    if not human_path.exists():
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "video_id",
                "text",
                "labels",
                "flags",
                "label_source",
                "sample_weight",
                "human_holdout",
            ]
        )

    chunks = read_jsonl(processed / "chunks_para_etiquetar.jsonl")
    chunk_by_id = _unique_by_id(chunks, "canónico")
    human_rows = read_jsonl(human_path)
    human_by_id = _unique_by_id(human_rows, "humano")
    allowed_labels, allowed_flags = set(label_order), set(flag_order)
    unknown_ids = set(human_by_id) - set(chunk_by_id)
    if unknown_ids:
        raise ValueError(f"El consenso humano contiene {len(unknown_ids)} IDs ajenos al canónico.")

    output: list[dict] = []
    for chunk_id, annotation in human_by_id.items():
        _validate_annotation(annotation, allowed_labels, allowed_flags, "humano")
        chunk = chunk_by_id[chunk_id]
        text = str(chunk.get("text") or "")
        if not text.strip():
            raise ValueError(f"El texto canónico de {chunk_id} está vacío.")
        output.append(
            {
                "chunk_id": chunk_id,
                "video_id": chunk.get("video_id") or chunk_id,
                "text": text,
                "labels": sorted(set(annotation["labels"])),
                "flags": sorted(set(annotation.get("flags", []))),
                "label_source": "humano",
                "sample_weight": 1.0,
                "human_holdout": True,
            }
        )
    return pd.DataFrame(output)


def targets_from_frame(
    frame: pd.DataFrame, label_order: list[str], flag_order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [[int(label in values) for label in label_order] for values in frame["labels"]],
        dtype=np.int8,
    )
    flags = np.asarray(
        [[int(flag in values) for flag in flag_order] for values in frame["flags"]],
        dtype=np.int8,
    )
    return labels, flags


def grouped_train_validation_test_split(
    frame: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
) -> dict[str, np.ndarray]:
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("Los tamaños de validación y prueba deben sumar menos de uno.")
    groups = frame["video_id"].astype(str).to_numpy()
    indices = np.arange(len(frame))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_validation_idx, test_idx = next(outer.split(indices, groups=groups))
    remaining_groups = groups[train_validation_idx]
    relative_validation = validation_size / (1 - test_size)
    inner = GroupShuffleSplit(
        n_splits=1, test_size=relative_validation, random_state=seed + 1
    )
    train_rel, validation_rel = next(
        inner.split(train_validation_idx, groups=remaining_groups)
    )
    split = {
        "train": train_validation_idx[train_rel],
        "validation": train_validation_idx[validation_rel],
        "test": test_idx,
    }
    group_sets = {name: set(groups[idx]) for name, idx in split.items()}
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise AssertionError("Hay fuga de video entre particiones.")
    return split


def _fit_binary_estimators(
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    class_names: list[str],
    max_iter: int = 1_000,
) -> list:
    estimators = []
    for index, class_name in enumerate(class_names):
        target = y[:, index]
        if np.unique(target).size < 2:
            estimator = DummyClassifier(strategy="constant", constant=int(target[0]))
            estimator.fit(np.zeros((len(target), 1)), target, sample_weight=sample_weight)
        else:
            estimator = LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )
            estimator.fit(X, target, sample_weight=sample_weight)
        estimators.append(estimator)
    return estimators


def _positive_probabilities(estimators: list, X) -> np.ndarray:
    columns = []
    for estimator in estimators:
        classes = np.asarray(estimator.classes_)
        if classes.size == 1:
            probability = np.full(X.shape[0], float(classes[0]))
        else:
            positive_index = int(np.where(classes == 1)[0][0])
            probability = estimator.predict_proba(X)[:, positive_index]
        columns.append(probability)
    return np.column_stack(columns)


def tune_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
    minimum_positives: int = 10,
) -> np.ndarray:
    grid = grid if grid is not None else np.linspace(0.10, 0.90, 33)
    thresholds = np.full(y_true.shape[1], 0.50, dtype=float)
    for column in range(y_true.shape[1]):
        if y_true[:, column].sum() < minimum_positives:
            continue
        candidates = []
        for threshold in grid:
            prediction = probabilities[:, column] >= threshold
            score = f1_score(y_true[:, column], prediction, zero_division=0)
            candidates.append((score, -abs(threshold - 0.50), threshold))
        thresholds[column] = max(candidates)[2]
    return thresholds


def apply_semantic_constraints(
    label_probabilities: np.ndarray,
    flag_probabilities: np.ndarray,
    label_thresholds: np.ndarray,
    flag_thresholds: np.ndarray,
    label_order: list[str],
    review_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_labels = label_probabilities >= label_thresholds
    raw_flags = flag_probabilities >= flag_thresholds
    labels = raw_labels.copy()
    flags = raw_flags.copy()
    safe_indices = [index for index, label in enumerate(label_order) if label in SAFE_LABELS]
    damage_indices = [index for index, label in enumerate(label_order) if label not in SAFE_LABELS]
    empty_before_constraints = ~raw_labels.any(axis=1)

    for row in range(len(labels)):
        if labels[row, damage_indices].any():
            labels[row, safe_indices] = False
        elif labels[row, safe_indices].sum() > 1:
            best_safe = safe_indices[int(np.argmax(label_probabilities[row, safe_indices]))]
            labels[row, safe_indices] = False
            labels[row, best_safe] = True
        elif not labels[row].any():
            labels[row, int(np.argmax(label_probabilities[row]))] = True
        if not labels[row, damage_indices].any():
            flags[row] = False

    label_near_boundary = np.min(
        np.abs(label_probabilities - label_thresholds), axis=1
    ) < review_margin
    flag_near_boundary = np.min(
        np.abs(flag_probabilities - flag_thresholds), axis=1
    ) < review_margin
    needs_review = (
        raw_flags.any(axis=1)
        | label_near_boundary
        | flag_near_boundary
        | empty_before_constraints
    )
    return labels.astype(np.int8), flags.astype(np.int8), needs_review


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    try:
        pr_auc_macro = float(average_precision_score(y_true, probabilities, average="macro"))
    except ValueError:
        pr_auc_macro = math.nan
    summary = {
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "jaccard_samples": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=1)
        ),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc_macro": pr_auc_macro,
        "n": int(len(y_true)),
    }
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
    ).T
    return summary, report


@dataclass
class MultiHeadModerationModel:
    vectorizer: TfidfVectorizer
    label_estimators: list
    flag_estimators: list
    label_order: list[str]
    flag_order: list[str]
    label_thresholds: np.ndarray
    flag_thresholds: np.ndarray
    review_margin: float = 0.05
    metadata: dict | None = None

    def predict_proba(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        text_list = list(texts)
        X = self.vectorizer.transform(text_list)
        return {
            "labels": _positive_probabilities(self.label_estimators, X),
            "flags": _positive_probabilities(self.flag_estimators, X),
        }

    def predict(self, texts: Iterable[str]) -> list[dict]:
        text_list = list(texts)
        probabilities = self.predict_proba(text_list)
        label_matrix, flag_matrix, review = apply_semantic_constraints(
            probabilities["labels"],
            probabilities["flags"],
            self.label_thresholds,
            self.flag_thresholds,
            self.label_order,
            self.review_margin,
        )
        results = []
        for row in range(len(text_list)):
            results.append(
                {
                    "labels": [
                        label for label, selected in zip(self.label_order, label_matrix[row]) if selected
                    ],
                    "flags": [
                        flag for flag, selected in zip(self.flag_order, flag_matrix[row]) if selected
                    ],
                    "needs_review": bool(review[row]),
                    "label_probabilities": {
                        label: float(value)
                        for label, value in zip(self.label_order, probabilities["labels"][row])
                    },
                    "flag_probabilities": {
                        flag: float(value)
                        for flag, value in zip(self.flag_order, probabilities["flags"][row])
                    },
                }
            )
        return results


def fit_multihead_model(
    frame: pd.DataFrame,
    label_order: list[str],
    flag_order: list[str],
    max_features: int = 50_000,
    min_df: int = 2,
    label_thresholds: np.ndarray | None = None,
    flag_thresholds: np.ndarray | None = None,
    review_margin: float = 0.05,
    metadata: dict | None = None,
) -> MultiHeadModerationModel:
    y_labels, y_flags = targets_from_frame(frame, label_order, flag_order)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(frame["text"])
    weights = frame["sample_weight"].to_numpy(dtype=float)
    label_estimators = _fit_binary_estimators(X, y_labels, weights, label_order)
    flag_estimators = _fit_binary_estimators(X, y_flags, weights, flag_order)
    return MultiHeadModerationModel(
        vectorizer=vectorizer,
        label_estimators=label_estimators,
        flag_estimators=flag_estimators,
        label_order=label_order,
        flag_order=flag_order,
        label_thresholds=(
            np.asarray(label_thresholds, dtype=float)
            if label_thresholds is not None
            else np.full(len(label_order), 0.50)
        ),
        flag_thresholds=(
            np.asarray(flag_thresholds, dtype=float)
            if flag_thresholds is not None
            else np.full(len(flag_order), 0.50)
        ),
        review_margin=review_margin,
        metadata=metadata or {},
    )


def tune_model_thresholds(
    model: MultiHeadModerationModel,
    validation_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    y_labels, y_flags = targets_from_frame(
        validation_frame, model.label_order, model.flag_order
    )
    probabilities = model.predict_proba(validation_frame["text"])
    return {
        "labels": tune_thresholds(y_labels, probabilities["labels"]),
        "flags": tune_thresholds(y_flags, probabilities["flags"]),
    }


def evaluate_model(
    model: MultiHeadModerationModel,
    frame: pd.DataFrame,
) -> dict:
    y_labels, y_flags = targets_from_frame(frame, model.label_order, model.flag_order)
    probabilities = model.predict_proba(frame["text"])
    pred_labels, pred_flags, needs_review = apply_semantic_constraints(
        probabilities["labels"],
        probabilities["flags"],
        model.label_thresholds,
        model.flag_thresholds,
        model.label_order,
        model.review_margin,
    )
    category_summary, category_report = multilabel_metrics(
        y_labels, pred_labels, probabilities["labels"], model.label_order
    )
    flag_summary, flag_report = multilabel_metrics(
        y_flags, pred_flags, probabilities["flags"], model.flag_order
    )
    return {
        "categories": category_summary,
        "flags": flag_summary,
        "needs_review_rate": float(needs_review.mean()),
        "category_report": category_report,
        "flag_report": flag_report,
    }


def export_lexicon(model: MultiHeadModerationModel, top_k: int = 80) -> dict:
    vocabulary = np.asarray(model.vectorizer.get_feature_names_out())

    def head_terms(names: list[str], estimators: list) -> dict[str, list[dict]]:
        output: dict[str, list[dict]] = {}
        for name, estimator in zip(names, estimators):
            if not hasattr(estimator, "coef_"):
                output[name] = []
                continue
            coefficients = estimator.coef_.ravel()
            indices = np.argsort(coefficients)[-top_k:][::-1]
            output[name] = [
                {"term": str(vocabulary[index]), "weight": float(coefficients[index])}
                for index in indices
                if coefficients[index] > 0
            ]
        return output

    return {
        "schema_version": "2.0",
        "categories": model.label_order,
        "transversal_flags": model.flag_order,
        "category_thresholds": {
            name: float(value) for name, value in zip(model.label_order, model.label_thresholds)
        },
        "flag_thresholds": {
            name: float(value) for name, value in zip(model.flag_order, model.flag_thresholds)
        },
        "category_terms": head_terms(model.label_order, model.label_estimators),
        "flag_terms": head_terms(model.flag_order, model.flag_estimators),
        "review_margin": model.review_margin,
    }


def save_model(model: MultiHeadModerationModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> MultiHeadModerationModel:
    return joblib.load(path)
