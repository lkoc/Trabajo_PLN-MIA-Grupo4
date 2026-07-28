"""Fine-tuning reproducible de dos encoders compactos para etiquetas gruesas.

El módulo usa únicamente las cinco categorías gruesas de daño. SEGURO se
deriva cuando ninguna salida de daño supera su umbral. Las etiquetas finas y
los flags transversales no se usan como objetivos ni como predictores.

Diseño:
1. integra todas las campañas canónicas y aplica la precedencia de revisiones;
2. conserva todos los casos con daño y submuestrea SEGURO a razón 4:1;
3. divide aleatoriamente 70/15/15, agrupando por video;
4. reentrena los cinco clásicos previos con sus hiperparámetros ya usados y
   añade fastText supervisado, tomado del material del profesor;
5. usa linear probing para escoger BCE normal o ponderación raíz;
6. hace fine-tuning completo, máximo tres épocas, con parada temprana;
7. selecciona por validación y evalúa una sola vez en test;
8. compara el Transformer ganador con el mejor clásico mediante bootstrap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from scripts_auxiliares.flujo_hibrido_moderador import (
    build_hybrid_dataset,
    load_taxonomy,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from scripts_auxiliares.modelos_gruesos_moderador import (
    COARSE_ORDER,
    DAMAGE_ORDER,
    MODEL_LABELS,
    MODEL_ORDER,
    add_coarse_targets,
    coarse_metrics,
    constrained_coarse_predictions,
    evaluate_candidate,
    fit_candidate,
    save_coarse_model,
    target_matrix,
    tune_candidate,
    tune_thresholds,
)
from scripts_auxiliares.preparar_entrenamiento_ampliado import (
    DATASET_MANIFEST_PATH,
    USABLE_PATH,
)


SEED = 20260727
SAFE_TO_DAMAGE_RATIO = 4
MAX_LENGTH = 128
ENCODE_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32
MAX_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
BOOTSTRAP_REPLICATES = 1_000

# Puertas operativas declaradas antes de comparar los modelos. No son normas
# universales: expresan la tolerancia de riesgo de este prototipo.
ALERT_VALIDATION_RECALL_TARGET = 0.95
AUTONOMOUS_MIN_PRECISION = 0.90
AUTONOMOUS_MIN_RECALL = 0.90
AUTONOMOUS_MIN_WILSON_LOWER = 0.85
AUTONOMOUS_MIN_CATEGORY_RECALL = 0.80
AUTONOMOUS_MIN_DAMAGE_F1_MACRO = 0.75
HUMAN_ALERT_MIN_TEST_RECALL = 0.90
HUMAN_ALERT_MIN_WILSON_LOWER = 0.85
HUMAN_ALERT_MIN_NPV = 0.95
HUMAN_ALERT_MAX_REVIEW_RATE = 0.60


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_id: str
    revision: str
    prefix: str
    label: str


MODEL_SPECS = {
    "paraphrase_minilm": EncoderSpec(
        key="paraphrase_minilm",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        prefix="",
        label="Paraphrase Multilingual MiniLM-L12",
    ),
    "e5_small": EncoderSpec(
        key="e5_small",
        model_id="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        prefix="query: ",
        label="Multilingual E5-small (linaje MiniLM)",
    ),
}

QWEN_LORA_SPEC = EncoderSpec(
    key="qwen3_06b_lora",
    model_id="Qwen/Qwen3-0.6B-Base",
    revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    prefix="",
    label="Qwen3-0.6B-Base + LoRA",
)
QWEN_TRAIN_BATCH_SIZE = 2
QWEN_EVAL_BATCH_SIZE = 4
QWEN_GRADIENT_ACCUMULATION = 4
QWEN_MAX_EPOCHS = 2
QWEN_LORA_RANK = 8
QWEN_LORA_ALPHA = 16
QWEN_LORA_DROPOUT = 0.05
QWEN_LEARNING_RATE = 1e-4


FASTTEXT_KEY = "fasttext_supervised_ova"
CLASSICAL_MODEL_ORDER = (*MODEL_ORDER, FASTTEXT_KEY)
CLASSICAL_MODEL_LABELS = {
    **MODEL_LABELS,
    FASTTEXT_KEY: "fastText supervisado OVA (sesión 4)",
}

# Configuraciones iniciales cerradas antes de ver el nuevo test. Los cinco
# primeros reproducen la pasada 04/04_1 y sirven para el screening; fastText
# usa la receta multietiqueta oficial que amplía ``train_supervised`` de
# PLN_clases/sesión 4. Solo las tres mejores familias pasan al GroupKFold.
PRIOR_CLASSICAL_CONFIGS = {
    "dummy_prior": {},
    "complement_nb": {"alpha": 0.5},
    "logistic_regression": {"C": 1.0},
    "linear_svm_word_char": {"C": 1.0},
    "hist_gradient_boosting_svd": {},
    FASTTEXT_KEY: {
        "lr": 0.5,
        "epoch": 25,
        "wordNgrams": 2,
        "bucket": 200_000,
        "dim": 50,
        "loss": "ova",
    },
}

# Búsqueda deliberadamente acotada para los tres mejores clásicos del
# screening. La validación externa no se usa durante estas combinaciones: los
# hiperparámetros se eligen con GroupKFold por video dentro de ``train``.
CLASSICAL_TUNING_FOLDS = 3
CLASSICAL_TUNING_TOP_K = 3
CLASSICAL_TUNING_GRIDS = {
    "complement_nb": [
        {"alpha": alpha, "min_df": min_df, "max_features": 50_000}
        for alpha in (0.10, 0.30, 0.50, 1.00)
        for min_df in (1, 2)
    ],
    "logistic_regression": [
        {"C": regularization, "min_df": min_df, "max_features": 50_000}
        for regularization in (0.25, 0.50, 1.00, 2.00)
        for min_df in (1, 2)
    ],
    "linear_svm_word_char": [
        {"C": regularization, "min_df": min_df, "max_features": 50_000}
        for regularization in (0.25, 0.50, 1.00, 2.00)
        for min_df in (1, 2)
    ],
    "hist_gradient_boosting_svd": [
        {
            "svd_components": components,
            "learning_rate": learning_rate,
            "max_iter": iterations,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "min_df": 2,
            "max_features": 30_000,
        }
        for components, learning_rate, iterations in (
            (64, 0.05, 120),
            (64, 0.08, 100),
            (96, 0.05, 160),
            (96, 0.08, 100),
            (128, 0.05, 120),
            (128, 0.08, 100),
            (160, 0.05, 120),
            (160, 0.08, 100),
        )
    ],
    FASTTEXT_KEY: [
        {
            "lr": learning_rate,
            "epoch": epochs,
            "wordNgrams": 2,
            "bucket": 200_000,
            "dim": dimension,
            "loss": "ova",
        }
        for learning_rate in (0.20, 0.50)
        for epochs in (15, 25)
        for dimension in (50, 100)
    ],
}


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
CACHE_DIR = ROOT / "datos" / "model_ready" / "transformer_grueso"
METRICS_DIR = ROOT / "resultados" / "metricas" / "transformer_grueso"
FIGURES_DIR = ROOT / "resultados" / "figuras" / "transformer_grueso"
LOG_DIR = ROOT / "resultados" / "logs" / "transformer_grueso"
MODEL_DIR = ROOT / "modelos" / "moderador_transformer_grueso"
REPORT_PATH = ROOT / "resultados" / "INFORME_ENTRENAMIENTO_TRANSFORMER_GRUESO.md"
INTEGRATED_DATASET_PATH = CACHE_DIR / "dataset_integrado_todas_pasadas.jsonl"
INTEGRATED_MANIFEST_PATH = CACHE_DIR / "dataset_integrado_todas_pasadas.manifest.json"
BALANCED_DATASET_PATH = CACHE_DIR / "dataset_balanceado_4a1_particionado.jsonl"
BALANCED_TRAIN_PATH = CACHE_DIR / "dataset_entrenamiento_transformer_4a1.jsonl"
BALANCED_TRAIN_MANIFEST_PATH = CACHE_DIR / "dataset_entrenamiento_transformer_4a1.manifest.json"
HUMAN_DIR = ROOT / "datos" / "etiquetado" / "humano"
HUMAN_CAMPAIGN_PATH = HUMAN_DIR / "revision_humana_combinada_1918.campaign.json"
HUMAN_PROGRESS_PATH = HUMAN_DIR / "revision_humana_combinada_1918.progress.json"
HUMAN_FINAL_PATH = HUMAN_DIR / "revision_humana_combinada_1918.jsonl"
HUMAN_FINAL_MANIFEST_PATH = HUMAN_DIR / "revision_humana_combinada_1918.manifest.json"
HUMAN_SNAPSHOT_DIR = HUMAN_DIR / "snapshots_entrenamiento"
USABLE_REFRESH_MANIFEST_PATH = CACHE_DIR / "actualizacion_datasets_utilizables.manifest.json"
for directory in (CACHE_DIR, METRICS_DIR, FIGURES_DIR, LOG_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_log(event: str, **values: object) -> None:
    payload = {"timestamp": now_iso(), "event": event, **values}
    with (LOG_DIR / "progreso.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True, warn_only=True)


def ids_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def training_content_sha256(frame: pd.DataFrame) -> str:
    """Firma IDs, texto y objetivos; invalida caché si cambia una adjudicación."""
    digest = hashlib.sha256()
    for row in frame.to_dict("records"):
        payload = {
            "chunk_id": str(row.get("chunk_id") or ""),
            "video_id": str(row.get("video_id") or ""),
            "text": str(row.get("text") or ""),
            "coarse_labels": list(row.get("coarse_labels") or []),
            "flags": list(row.get("flags") or []),
            "label_source": str(row.get("label_source") or ""),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "campaign": str(row.get("campaign") or ""),
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def damage_targets(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [[int(label in labels) for label in DAMAGE_ORDER] for labels in frame["coarse_labels"]],
        dtype=np.float32,
    )


def _verify_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    names = list(parts)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            chunk_overlap = set(parts[left]["chunk_id"]) & set(parts[right]["chunk_id"])
            video_overlap = set(parts[left]["video_id"].astype(str)) & set(
                parts[right]["video_id"].astype(str)
            )
            if chunk_overlap or video_overlap:
                raise AssertionError(
                    f"Fuga entre {left} y {right}: chunks={len(chunk_overlap)}, "
                    f"videos={len(video_overlap)}."
                )


def validate_and_snapshot_complete_human_review() -> dict:
    """Valida la campaña cerrada y congela su progreso para el entrenamiento.

    La salida humana combinada es la autoridad para reemplazar las propuestas
    Pro de los casos ``need_review``. Esta función falla antes de construir el
    dataset si faltan decisiones, hay IDs duplicados o los hashes no coinciden.
    """
    required = (
        HUMAN_CAMPAIGN_PATH,
        HUMAN_PROGRESS_PATH,
        HUMAN_FINAL_PATH,
        HUMAN_FINAL_MANIFEST_PATH,
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos de adjudicación humana final:\n- " + "\n- ".join(missing)
        )

    campaign = json.loads(HUMAN_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    progress = json.loads(HUMAN_PROGRESS_PATH.read_text(encoding="utf-8"))
    final_rows = read_jsonl(HUMAN_FINAL_PATH)
    manifest = json.loads(HUMAN_FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    campaign_ids = [str(row["chunk_id"]) for row in campaign.get("records", [])]
    annotations = progress.get("annotations", [])
    progress_ids = [str(row.get("chunk_id") or "") for row in annotations]
    final_ids = [str(row.get("chunk_id") or "") for row in final_rows]
    total = len(campaign_ids)
    if not total or len(set(campaign_ids)) != total:
        raise ValueError("La campaña humana está vacía o contiene chunk_id duplicados.")
    if (
        len(annotations) != total
        or len(final_rows) != total
        or len(set(progress_ids)) != total
        or len(set(final_ids)) != total
        or set(progress_ids) != set(campaign_ids)
        or set(final_ids) != set(campaign_ids)
    ):
        raise ValueError("Campaña, progreso y salida humana final no cubren los mismos IDs.")
    if any(row.get("status") != "completed" for row in annotations):
        raise ValueError("La campaña humana todavía contiene decisiones sin completar.")
    if int(manifest.get("completed", -1)) != total:
        raise ValueError("El manifiesto humano no confirma la campaña completa.")
    if manifest.get("output_sha256") != sha256_file(HUMAN_FINAL_PATH):
        raise ValueError("El hash de la salida humana final no coincide con su manifiesto.")
    if manifest.get("source_campaign_sha256") != sha256_file(HUMAN_CAMPAIGN_PATH):
        raise ValueError("El hash de la campaña humana no coincide con su manifiesto.")

    revision = int(progress.get("revision", 0))
    source_sha = sha256_file(HUMAN_PROGRESS_PATH)
    HUMAN_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = HUMAN_SNAPSHOT_DIR / (
        f"revision_humana_final_r{revision}_{source_sha[:12]}.json"
    )
    if not snapshot_path.exists():
        write_json(snapshot_path, progress)
    snapshot_sha = sha256_file(snapshot_path)
    os.environ["MODERATION_HUMAN_PROGRESS_SNAPSHOT"] = str(snapshot_path)
    return {
        "campaign_rows": total,
        "completed": total,
        "included": sum(bool(row.get("training_eligible")) for row in final_rows),
        "excluded": sum(not bool(row.get("training_eligible")) for row in final_rows),
        "revision": revision,
        "campaign": str(HUMAN_CAMPAIGN_PATH.relative_to(ROOT)),
        "campaign_sha256": sha256_file(HUMAN_CAMPAIGN_PATH),
        "final_output": str(HUMAN_FINAL_PATH.relative_to(ROOT)),
        "final_output_sha256": sha256_file(HUMAN_FINAL_PATH),
        "final_manifest": str(HUMAN_FINAL_MANIFEST_PATH.relative_to(ROOT)),
        "final_manifest_sha256": sha256_file(HUMAN_FINAL_MANIFEST_PATH),
        "snapshot": str(snapshot_path.relative_to(ROOT)),
        "snapshot_sha256": snapshot_sha,
    }


def _cohort_human_output(batch_id: str) -> tuple[str, Path, Path]:
    if batch_id == "ampliacion_dano_20260726":
        return (
            "ampliacion_1779",
            HUMAN_DIR / "revision_humana_ampliacion_1779.jsonl",
            HUMAN_DIR / "revision_humana_ampliacion_1779.manifest.json",
        )
    processed = ROOT / "datos" / "ampliacion" / batch_id / "processed"
    return batch_id, processed / "revision_humana.jsonl", processed / "revision_humana.manifest.json"


def refresh_usable_datasets_from_human_review(force: bool = False) -> dict:
    """Regenera entradas utilizables desactualizadas antes de integrar 04_2.

    No llama a ningún LLM ni entrena modelos: aplica las decisiones finales,
    reconstruye cada ``dataset_etiquetado_utilizable.jsonl`` y verifica hashes.
    Es idempotente; una cohorte vigente se valida y se omite.
    """
    human = validate_and_snapshot_complete_human_review()
    campaign = json.loads(HUMAN_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    campaign_counts = {
        cohort: sum(row.get("cohort") == cohort for row in campaign["records"])
        for cohort in campaign.get("cohorts", {})
    }
    batch_dirs = sorted(
        path.parents[1]
        for path in (ROOT / "datos" / "ampliacion").glob(
            "*/processed/pendientes_revision_humana.jsonl"
        )
    )
    if not batch_dirs:
        raise FileNotFoundError("No se encontraron ampliaciones con cola de revisión humana.")
    legacy = "ampliacion_dano_20260726"
    batch_dirs.sort(key=lambda path: (path.name != legacy, path.name))
    entries = []
    for batch_dir in batch_dirs:
        batch_id = batch_dir.name
        cohort, final_path, final_manifest_path = _cohort_human_output(batch_id)
        usable_path = batch_dir / "processed" / "dataset_etiquetado_utilizable.jsonl"
        usable_manifest_path = usable_path.with_suffix(".manifest.json")
        chunks_path = batch_dir / "processed" / "chunks_para_etiquetar.jsonl"
        for required in (final_path, final_manifest_path, chunks_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"{batch_id}: falta {required.relative_to(ROOT)} para actualizar 04_2."
                )
        final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        expected = int(campaign_counts.get(cohort, -1))
        if expected < 0 or int(final_manifest.get("completed", -2)) != expected:
            raise ValueError(f"{batch_id}: la salida humana no cubre su cohorte completa.")
        final_sha = sha256_file(final_path)
        final_manifest_sha = sha256_file(final_manifest_path)
        if final_manifest.get("output_sha256") != final_sha:
            raise ValueError(f"{batch_id}: hash humano final inválido.")

        current = (
            json.loads(usable_manifest_path.read_text(encoding="utf-8"))
            if usable_manifest_path.exists()
            else {}
        )
        adjudication = current.get("human_adjudication") or {}
        input_hashes = current.get("input_sha256") or {}
        current_ready = (
            bool(adjudication.get("complete"))
            and int(adjudication.get("adjudicated", -1)) == expected
            and int(adjudication.get("pending", -1)) == 0
            and int(current.get("pending_human_rows", -1)) == 0
            and input_hashes.get("human_expansion") == final_sha
            and input_hashes.get("human_expansion_manifest") == final_manifest_sha
            and usable_path.exists()
            and (current.get("output_sha256") or {}).get("usable") == sha256_file(usable_path)
        )
        refreshed = force or not current_ready
        if refreshed:
            environment = os.environ.copy()
            environment["AMPLIACION_BATCH_ID"] = batch_id
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts_auxiliares.preparar_entrenamiento_ampliado",
                    "--stage",
                    "prepare",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout or "sin detalle").strip()
                raise RuntimeError(f"No se pudo actualizar {batch_id}:\n{detail}")
            current = json.loads(usable_manifest_path.read_text(encoding="utf-8"))
            adjudication = current.get("human_adjudication") or {}

        usable_rows = read_jsonl(usable_path)
        raw_rows = read_jsonl(chunks_path)
        excluded = int(final_manifest.get("excluded_from_training", 0))
        if (
            not bool(adjudication.get("complete"))
            or int(adjudication.get("adjudicated", -1)) != expected
            or int(current.get("pending_human_rows", -1)) != 0
            or len(usable_rows) != len(raw_rows) - excluded
            or len({row["chunk_id"] for row in usable_rows}) != len(usable_rows)
            or any(bool(row.get("needs_review")) for row in usable_rows)
            or (current.get("output_sha256") or {}).get("usable") != sha256_file(usable_path)
        ):
            raise ValueError(f"{batch_id}: dataset utilizable regenerado inconsistente.")
        entries.append(
            {
                "batch_id": batch_id,
                "cohort": cohort,
                "refreshed": refreshed,
                "raw_rows": len(raw_rows),
                "human_review_rows": expected,
                "human_included": int(adjudication.get("included", 0)),
                "human_excluded": int(adjudication.get("excluded", 0)),
                "usable_rows": len(usable_rows),
                "pending_human_rows": 0,
                "usable": str(usable_path.relative_to(ROOT)),
                "usable_sha256": sha256_file(usable_path),
                "manifest": str(usable_manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256_file(usable_manifest_path),
            }
        )
    report = {
        "schema_version": "1.0",
        "created_at": now_iso(),
        "purpose": "preflight reproducible de adjudicación humana antes de 04_2",
        "llm_calls": 0,
        "model_training": False,
        "human_review": human,
        "batches": entries,
        "all_pending_resolved": all(item["pending_human_rows"] == 0 for item in entries),
    }
    write_json(USABLE_REFRESH_MANIFEST_PATH, report)
    return report


def resolve_training_human_snapshot() -> tuple[Path, str]:
    configured = os.environ.get("MODERATION_HUMAN_PROGRESS_SNAPSHOT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"No existe la instantánea humana configurada: {path}")
        return path, sha256_file(path)
    if USABLE_REFRESH_MANIFEST_PATH.exists():
        refresh = json.loads(USABLE_REFRESH_MANIFEST_PATH.read_text(encoding="utf-8"))
        review = refresh.get("human_review") or {}
        path = ROOT / Path(review.get("snapshot", ""))
        expected = review.get("snapshot_sha256")
        if path.exists() and expected == sha256_file(path):
            return path, expected
    manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    training = manifest.get("training") or {}
    relative = training.get("human_review_snapshot")
    expected = training.get("human_review_snapshot_sha256")
    if not relative:
        raise ValueError(
            "No hay una instantánea humana vigente. Ejecute primero "
            "refresh_usable_datasets_from_human_review()."
        )
    path = ROOT / Path(relative)
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError("La instantánea humana registrada falta o tiene un hash distinto.")
    return path, expected


def discover_usable_expansion_datasets() -> list[dict]:
    """Descubre y verifica todas las ampliaciones utilizables presentes/futuras."""
    entries = []
    pattern = "*/processed/dataset_etiquetado_utilizable.jsonl"
    for path in sorted((ROOT / "datos" / "ampliacion").glob(pattern)):
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset de ampliación sin manifiesto: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest.get("output_sha256", {}).get("usable")
        actual_sha = sha256_file(path)
        if expected_sha != actual_sha:
            raise ValueError(f"SHA-256 inválido para la ampliación: {path}")
        if not bool(manifest.get("coarse_training_only")):
            raise ValueError(f"La ampliación no declara entrenamiento grueso: {path}")
        rows = read_jsonl(path)
        if int(manifest.get("usable_rows", -1)) != len(rows):
            raise ValueError(f"El manifiesto no coincide en número de filas: {path}")
        frame = pd.DataFrame(rows)
        required = {"chunk_id", "video_id", "text", "coarse_labels", "needs_review"}
        if frame.empty or not required <= set(frame.columns):
            raise ValueError(f"Esquema utilizable incompleto: {path}")
        if frame["chunk_id"].duplicated().any() or frame["needs_review"].astype(bool).any():
            raise ValueError(f"La ampliación tiene duplicados o dudas sin excluir: {path}")
        batch_id = str(manifest.get("batch_id") or path.parents[1].name)
        entries.append(
            {
                "batch_id": batch_id,
                "path": path,
                "manifest_path": manifest_path,
                "sha256": actual_sha,
                "rows": len(frame),
                "frame": frame,
                "pending_human_rows_excluded": int(
                    manifest.get("pending_human_rows", 0)
                ),
            }
        )
    if not entries:
        raise FileNotFoundError("No hay ampliaciones utilizables para integrar.")
    return entries


def _balanced_group_split(frame: pd.DataFrame, candidates: int = 500) -> tuple[dict, dict]:
    """Busca una partición aleatoria 70/15/15 agrupada por video.

    Se generan candidatos con GroupShuffleSplit y se elige el que más aproxima
    tamaños y prevalencias gruesas. El test nunca interviene en decisiones del
    modelo posteriores; usar etiquetas aquí sólo estratifica la partición.
    """
    y = damage_targets(frame)
    groups = frame["video_id"].astype(str).to_numpy()
    all_indices = np.arange(len(frame))
    global_prevalence = y.mean(axis=0)
    scale = np.sqrt(np.maximum(global_prevalence, 1 / len(frame)))
    best = None
    for candidate in range(candidates):
        seed = SEED + 2 * candidate
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_validation, test = next(outer.split(all_indices, groups=groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15 / 0.85,
            random_state=seed + 1,
        )
        train_relative, validation_relative = next(
            inner.split(train_validation, groups=groups[train_validation])
        )
        split = {
            "train": train_validation[train_relative],
            "validation": train_validation[validation_relative],
            "test": test,
        }
        score = 0.0
        missing_cells = 0
        for name, target_fraction in (("train", 0.70), ("validation", 0.15), ("test", 0.15)):
            indices = split[name]
            score += 5.0 * abs(len(indices) / len(frame) - target_fraction)
            score += float(np.mean(np.abs(y[indices].mean(axis=0) - global_prevalence) / scale))
            missing_cells += int((y[indices].sum(axis=0) == 0).sum())
        score += 10.0 * missing_cells
        candidate_value = (score, seed, split, missing_cells)
        if best is None or candidate_value[0] < best[0]:
            best = candidate_value
    assert best is not None
    score, seed, split, missing_cells = best
    group_sets = {name: set(groups[indices]) for name, indices in split.items()}
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise AssertionError("La búsqueda produjo fuga de videos.")
    metadata = {
        "method": "best of 500 random GroupShuffleSplit candidates",
        "base_seed": SEED,
        "selected_seed": seed,
        "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "score": score,
        "missing_category_cells": missing_cells,
    }
    return split, metadata


def load_experiment_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    """Integra todo, submuestrea SEGURO y recién entonces divide 70/15/15."""
    snapshot_path, expected_snapshot_sha = resolve_training_human_snapshot()
    os.environ["MODERATION_HUMAN_PROGRESS_SNAPSHOT"] = str(snapshot_path)

    taxonomy, _, _ = load_taxonomy(ROOT)
    base, base_metadata, _, _ = build_hybrid_dataset(
        ROOT, require_complete_hard_review=False, write_output=False
    )
    base = add_coarse_targets(base, taxonomy)
    base = base.loc[~base["human_holdout"]].reset_index(drop=True)
    base = base.copy()
    base["campaign"] = "historico_2026"
    expansion_entries = discover_usable_expansion_datasets()
    expansion_frames = []
    campaign_video_ids: dict[str, set[str]] = {
        "historico_2026": set(base["video_id"].astype(str))
    }
    for entry in expansion_entries:
        frame = entry["frame"].copy()
        frame["campaign"] = entry["batch_id"]
        videos = set(frame["video_id"].astype(str))
        for prior_campaign, prior_videos in campaign_video_ids.items():
            overlap = videos & prior_videos
            if overlap:
                raise AssertionError(
                    f"{entry['batch_id']} comparte {len(overlap)} videos con {prior_campaign}."
                )
        campaign_video_ids[entry["batch_id"]] = videos
        expansion_frames.append(frame)
    integrated = pd.concat([base, *expansion_frames], ignore_index=True, sort=False)
    if integrated["chunk_id"].duplicated().any():
        raise AssertionError("Las campañas canónicas se solapan por chunk_id.")
    balanced_all, _, balance = moderate_safe_undersample(integrated)
    split, split_metadata = _balanced_group_split(balanced_all)
    frames = {
        "integrated": integrated,
        "balanced_all": balanced_all,
        "train": balanced_all.iloc[split["train"]].reset_index(drop=True),
        "validation": balanced_all.iloc[split["validation"]].reset_index(drop=True),
        "test": balanced_all.iloc[split["test"]].reset_index(drop=True),
    }
    _verify_disjoint({name: frames[name] for name in ("train", "validation", "test")})
    for name, frame in frames.items():
        if frame["chunk_id"].duplicated().any() or frame["text"].str.strip().eq("").any():
            raise ValueError(f"{name} contiene IDs duplicados o texto vacío.")
        invalid = frame["coarse_labels"].map(
            lambda values: not set(values) <= set(COARSE_ORDER) or (
                "SEGURO" in values and len(values) != 1
            )
        )
        if invalid.any():
            raise ValueError(f"{name} contiene categorías gruesas inválidas.")

    audit = {
        "created_at": now_iso(),
        "seed": SEED,
        "snapshot": str(snapshot_path.relative_to(ROOT)),
        "snapshot_sha256": sha256_file(snapshot_path),
        "expansion_datasets": [
            {
                "batch_id": entry["batch_id"],
                "path": str(entry["path"].relative_to(ROOT)),
                "manifest": str(entry["manifest_path"].relative_to(ROOT)),
                "rows": entry["rows"],
                "sha256": entry["sha256"],
                "pending_human_rows_excluded": entry[
                    "pending_human_rows_excluded"
                ],
            }
            for entry in expansion_entries
        ],
        "base_metadata": base_metadata,
        "parts": {},
        "fine_labels_trained": False,
        "transversal_flags_trained": False,
        "targets": DAMAGE_ORDER,
        "safe_is_derived": True,
        "video_leakage": 0,
        "ordering": "integrate_all -> safe_undersample_4to1 -> grouped_random_split",
        "global_balance": balance,
        "split_design": split_metadata,
    }
    for name, frame in frames.items():
        y = damage_targets(frame)
        any_damage = y.any(axis=1)
        audit["parts"][name] = {
            "rows": len(frame),
            "videos": int(frame["video_id"].nunique()),
            "chunk_ids_sha256": ids_sha256(frame["chunk_id"]),
            "safe_rows": int((~any_damage).sum()),
            "damage_rows": int(any_damage.sum()),
            "damage_pct": float(100 * any_damage.mean()),
            "category_counts": {
                label: int(y[:, index].sum()) for index, label in enumerate(DAMAGE_ORDER)
            },
        }
    return frames, audit


def materialize_integrated_dataset(
    frames: dict[str, pd.DataFrame],
    audit: dict,
    force: bool = False,
) -> dict:
    """Une todas las pasadas útiles y conserva el origen/split sin duplicar.

    Los archivos de revisión Pro, hard mining, Qwen y reasoning no se apilan
    como nuevas observaciones porque contienen nuevas decisiones sobre chunks
    canónicos ya presentes. Su efecto entra mediante la precedencia aplicada
    por ``build_hybrid_dataset``. El piloto reasoning no se incorpora porque
    su manifiesto declara ``training_modified=false`` y el control humano lo
    rechazó como mecanismo de absolución automática.
    """
    integrated = frames["integrated"].reset_index(drop=True)
    content_sha256 = training_content_sha256(integrated)
    if INTEGRATED_DATASET_PATH.exists() and INTEGRATED_MANIFEST_PATH.exists() and not force:
        existing = json.loads(INTEGRATED_MANIFEST_PATH.read_text(encoding="utf-8"))
        if (
            int(existing.get("usable_unique_rows", 0)) == len(integrated)
            and existing.get("chunk_ids_sha256") == ids_sha256(integrated["chunk_id"])
            and existing.get("training_content_sha256") == content_sha256
            and existing.get("contains_split_assignment") is False
        ):
            return existing
    rows = [
        {
            "chunk_id": str(row["chunk_id"]),
            "video_id": str(row.get("video_id") or row["chunk_id"]),
            "text": str(row["text"]),
            "coarse_labels": list(row["coarse_labels"]),
            "flags_reference_only": list(row.get("flags") or []),
            "label_source": str(row.get("label_source") or "unknown"),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "campaign": str(row["campaign"]),
        }
        for row in integrated.to_dict("records")
    ]
    combined = pd.DataFrame(rows)
    if combined["chunk_id"].duplicated().any():
        duplicates = int(combined["chunk_id"].duplicated(keep=False).sum())
        raise AssertionError(f"La integración produjo {duplicates} IDs duplicados.")
    damage = combined["coarse_labels"].map(lambda values: "SEGURO" not in values)
    category_counts = {
        label: int(combined["coarse_labels"].map(lambda values: label in values).sum())
        for label in DAMAGE_ORDER
    }
    multilabel_rows = int(
        combined["coarse_labels"].map(
            lambda values: sum(label in values for label in DAMAGE_ORDER) > 1
        ).sum()
    )
    write_jsonl(INTEGRATED_DATASET_PATH, combined.to_dict("records"))
    raw_campaigns = {
        "historical": {
            "path": "datos/processed/chunks_para_etiquetar.jsonl",
            "rows": len(read_jsonl(ROOT / "datos" / "processed" / "chunks_para_etiquetar.jsonl")),
        }
    }
    for entry in discover_usable_expansion_datasets():
        raw_path = entry["path"].parent / "chunks_para_etiquetar.jsonl"
        raw_campaigns[entry["batch_id"]] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(read_jsonl(raw_path)),
            "usable_path": str(entry["path"].relative_to(ROOT)),
            "usable_rows": entry["rows"],
        }
    raw_union_rows = sum(int(value["rows"]) for value in raw_campaigns.values())
    manifest = {
        "schema_version": "2.0",
        "created_at": now_iso(),
        "purpose": "integración de todas las pasadas antes del fine-tuning Transformer",
        "raw_canonical_campaigns": raw_campaigns,
        "raw_union_unique_rows": raw_union_rows,
        "usable_unique_rows": len(combined),
        "excluded_unresolved_or_rejected_rows": raw_union_rows - len(combined),
        "safe_rows": int((~damage).sum()),
        "damage_unique_rows": int(damage.sum()),
        "damage_label_incidences": int(sum(category_counts.values())),
        "multilabel_damage_rows": multilabel_rows,
        "category_counts": category_counts,
        "campaign_counts": {
            str(key): int(value) for key, value in combined["campaign"].value_counts().items()
        },
        "contains_split_assignment": False,
        "ordering": "this artifact precedes safe undersampling and splitting",
        "label_precedence": "human accepted/modified > Pro hard mining > Pro > Flash",
        "review_passes_are_replacements_not_new_rows": True,
        "reasoning_pilot_included": False,
        "reasoning_exclusion": (
            "analysis-only; training_modified=false; failed preregistered human agreement criteria"
        ),
        "fine_labels_trained": False,
        "transversal_flags_trained": False,
        "chunk_ids_sha256": ids_sha256(combined["chunk_id"]),
        "training_content_sha256": content_sha256,
        "output": str(INTEGRATED_DATASET_PATH.relative_to(ROOT)),
        "output_sha256": sha256_file(INTEGRATED_DATASET_PATH),
        "human_snapshot": audit["snapshot"],
        "human_snapshot_sha256": audit["snapshot_sha256"],
    }
    write_json(INTEGRATED_MANIFEST_PATH, manifest)
    return manifest


def materialize_balanced_training(
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    sampled = frames["balanced_all"].reset_index(drop=True)
    train = frames["train"].reset_index(drop=True)
    _, _, balance = moderate_safe_undersample(frames["integrated"])
    split_by_id = {
        str(chunk_id): split_name
        for split_name in ("train", "validation", "test")
        for chunk_id in frames[split_name]["chunk_id"]
    }
    export_all = sampled.copy()
    export_all["flags_reference_only"] = export_all["flags"].map(list)
    export_all["split"] = export_all["chunk_id"].astype(str).map(split_by_id)
    if export_all["split"].isna().any():
        raise AssertionError("Hay chunks balanceados sin partición.")
    columns = [
        "chunk_id",
        "video_id",
        "text",
        "coarse_labels",
        "flags_reference_only",
        "label_source",
        "sample_weight",
        "campaign",
        "split",
    ]
    write_jsonl(BALANCED_DATASET_PATH, export_all[columns].to_dict("records"))
    write_jsonl(
        BALANCED_TRAIN_PATH,
        export_all.loc[export_all["split"].eq("train"), columns].to_dict("records"),
    )
    manifest = {
        **balance,
        "created_at": now_iso(),
        "ordering": "integrate -> safe undersample 4:1 -> grouped random 70/15/15 split",
        "input_integrated_dataset": str(INTEGRATED_DATASET_PATH.relative_to(ROOT)),
        "input_integrated_sha256": sha256_file(INTEGRATED_DATASET_PATH),
        "balanced_dataset_output": str(BALANCED_DATASET_PATH.relative_to(ROOT)),
        "balanced_dataset_sha256": sha256_file(BALANCED_DATASET_PATH),
        "training_output": str(BALANCED_TRAIN_PATH.relative_to(ROOT)),
        "training_output_sha256": sha256_file(BALANCED_TRAIN_PATH),
        "split_counts": {
            name: int(len(frames[name])) for name in ("train", "validation", "test")
        },
        "split_damage_counts": {
            name: int(damage_targets(frames[name]).any(axis=1).sum())
            for name in ("train", "validation", "test")
        },
        "split_video_counts": {
            name: int(frames[name]["video_id"].nunique())
            for name in ("train", "validation", "test")
        },
        "fine_labels_trained": False,
        "transversal_flags_trained": False,
    }
    write_json(BALANCED_TRAIN_MANIFEST_PATH, manifest)
    return train, manifest


def moderate_safe_undersample(
    frame: pd.DataFrame,
    ratio: int = SAFE_TO_DAMAGE_RATIO,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Conserva todo daño y toma SEGURO por hash estable hasta ratio:1."""
    y = damage_targets(frame)
    damage_mask = y.any(axis=1)
    damage_indices = np.flatnonzero(damage_mask)
    safe_indices = np.flatnonzero(~damage_mask)
    safe_limit = min(len(safe_indices), ratio * len(damage_indices))
    ranked_safe = sorted(
        safe_indices,
        key=lambda index: hashlib.sha256(
            f"{seed}|{frame.iloc[index]['chunk_id']}".encode("utf-8")
        ).hexdigest(),
    )[:safe_limit]
    selected = np.asarray(
        sorted(
            [*damage_indices.tolist(), *ranked_safe],
            key=lambda index: hashlib.sha256(
                f"orden|{seed}|{frame.iloc[index]['chunk_id']}".encode("utf-8")
            ).hexdigest(),
        ),
        dtype=np.int64,
    )
    sampled = frame.iloc[selected].reset_index(drop=True)
    sampled_y = damage_targets(sampled)
    information = {
        "method": "deterministic_hash_safe_undersampling",
        "seed": seed,
        "safe_to_damage_ratio": ratio,
        "rows_before": len(frame),
        "rows_after": len(sampled),
        "damage_before": int(damage_mask.sum()),
        "damage_after": int(sampled_y.any(axis=1).sum()),
        "safe_before": int((~damage_mask).sum()),
        "safe_after": int((~sampled_y.any(axis=1)).sum()),
        "damage_pct_before": float(100 * damage_mask.mean()),
        "damage_pct_after": float(100 * sampled_y.any(axis=1).mean()),
        "selected_chunk_ids_sha256": ids_sha256(sampled["chunk_id"]),
        "category_counts_after": {
            label: int(sampled_y[:, index].sum()) for index, label in enumerate(DAMAGE_ORDER)
        },
    }
    return sampled, selected, information


def positive_weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "plain_bce":
        return np.ones(y.shape[1], dtype=np.float32)
    if mode == "sqrt_positive_weight":
        positives = y.sum(axis=0)
        negatives = len(y) - positives
        if (positives == 0).any():
            raise ValueError("No se pueden ponderar categorías sin positivos.")
        return np.sqrt(negatives / positives).astype(np.float32)
    raise ValueError(f"Modo de ponderación desconocido: {mode}")


def source_weights(frame: pd.DataFrame) -> np.ndarray:
    if "sample_weight" not in frame:
        return np.ones(len(frame), dtype=np.float32)
    return frame["sample_weight"].astype(float).to_numpy(dtype=np.float32)


def _six_class_arrays(
    y_damage: np.ndarray,
    damage_scores: np.ndarray,
    damage_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_safe = (~y_damage.astype(bool).any(axis=1)).astype(np.int8)[:, None]
    y_six = np.column_stack([true_safe, y_damage.astype(np.int8)])
    safe_score = (1.0 - damage_scores.max(axis=1))[:, None]
    scores_six = np.column_stack([safe_score, damage_scores])
    thresholds_six = np.concatenate([[0.50], damage_thresholds])
    pred_six = constrained_coarse_predictions(scores_six, thresholds_six)
    return y_six, scores_six, pred_six


def evaluate_damage_scores(
    y_damage: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y_six, scores_six, pred_six = _six_class_arrays(y_damage, scores, thresholds)
    summary, report = coarse_metrics(y_six, pred_six, scores_six)
    pred_damage = pred_six[:, 1:]
    true_any = y_damage.astype(bool).any(axis=1)
    pred_any = pred_damage.astype(bool).any(axis=1)
    summary.update(
        {
            "any_damage_prevalence": float(true_any.mean()),
            "any_damage_prediction_rate": float(pred_any.mean()),
            "any_damage_precision": float(
                precision_score(true_any, pred_any, zero_division=0)
            ),
            "any_damage_recall": float(recall_score(true_any, pred_any, zero_division=0)),
            "any_damage_f1": float(f1_score(true_any, pred_any, zero_division=0)),
            "missed_damage_as_safe": int((true_any & ~pred_any).sum()),
        }
    )
    return summary, report, pred_damage


def _embedding_paths(spec: EncoderSpec, split_name: str) -> tuple[Path, Path]:
    model_cache = CACHE_DIR / spec.key
    model_cache.mkdir(parents=True, exist_ok=True)
    return model_cache / f"{split_name}.npy", model_cache / f"{split_name}.manifest.json"


def load_encoder(spec: EncoderSpec):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(spec.model_id, revision=spec.revision, device="cpu")
    encoder.max_seq_length = MAX_LENGTH
    return encoder


def encode_frame(
    encoder,
    spec: EncoderSpec,
    frame: pd.DataFrame,
    split_name: str,
    force: bool = False,
) -> np.ndarray:
    data_path, manifest_path = _embedding_paths(spec, split_name)
    expected_ids = ids_sha256(frame["chunk_id"])
    if data_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("chunk_ids_sha256") == expected_ids
            and manifest.get("model_revision") == spec.revision
            and int(manifest.get("max_length", 0)) == MAX_LENGTH
        ):
            return np.load(data_path)
    texts = [spec.prefix + str(text) for text in frame["text"]]
    append_log("encoding_started", model=spec.key, split=split_name, rows=len(frame))
    start = perf_counter()
    embeddings = encoder.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = perf_counter() - start
    np.save(data_path, embeddings)
    write_json(
        manifest_path,
        {
            "created_at": now_iso(),
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "prefix": spec.prefix,
            "max_length": MAX_LENGTH,
            "rows": len(frame),
            "dimensions": int(embeddings.shape[1]),
            "chunk_ids_sha256": expected_ids,
            "elapsed_seconds": elapsed,
            "rows_per_second": len(frame) / elapsed,
        },
    )
    append_log(
        "encoding_completed",
        model=spec.key,
        split=split_name,
        rows=len(frame),
        elapsed_seconds=elapsed,
    )
    return embeddings


def fit_linear_head(
    X: np.ndarray,
    y: np.ndarray,
    base_sample_weight: np.ndarray,
    pos_weights: np.ndarray,
) -> list[LogisticRegression]:
    estimators: list[LogisticRegression] = []
    for column, label in enumerate(DAMAGE_ORDER):
        estimator = LogisticRegression(
            C=1.0,
            max_iter=2_000,
            solver="liblinear",
            random_state=SEED,
        )
        weights = base_sample_weight * np.where(y[:, column] == 1, pos_weights[column], 1.0)
        estimator.fit(X, y[:, column].astype(np.int8), sample_weight=weights)
        estimators.append(estimator)
        append_log("linear_head_fitted", category=label, iterations=int(estimator.n_iter_[0]))
    return estimators


def estimator_scores(estimators: list[LogisticRegression], X: np.ndarray) -> np.ndarray:
    return np.column_stack([estimator.predict_proba(X)[:, 1] for estimator in estimators])


def run_linear_screening(
    spec: EncoderSpec,
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> dict:
    output_path = METRICS_DIR / f"screening_{spec.key}.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    sampled = frames["train"]
    global_y = damage_targets(frames["balanced_all"])
    sampled_y = damage_targets(sampled)
    balance = {
        "ordering": "global 4:1 undersampling before split",
        "balanced_all_rows": len(frames["balanced_all"]),
        "balanced_all_damage": int(global_y.any(axis=1).sum()),
        "training_rows": len(sampled),
        "training_damage": int(sampled_y.any(axis=1).sum()),
        "training_safe": int((~sampled_y.any(axis=1)).sum()),
    }
    encoder = load_encoder(spec)
    X_train = encode_frame(encoder, spec, sampled, "train", force=force)
    X_validation = encode_frame(
        encoder, spec, frames["validation"], "validation", force=force
    )
    y_train = damage_targets(sampled)
    y_validation = damage_targets(frames["validation"])
    rows = []
    models = {}
    for mode in ("plain_bce", "sqrt_positive_weight"):
        weights = positive_weights(y_train, mode)
        start = perf_counter()
        estimators = fit_linear_head(
            X_train, y_train, source_weights(sampled), weights
        )
        fit_seconds = perf_counter() - start
        validation_scores = estimator_scores(estimators, X_validation)
        thresholds = tune_thresholds(y_validation.astype(np.int8), validation_scores)
        metrics, _, _ = evaluate_damage_scores(y_validation, validation_scores, thresholds)
        rows.append(
            {
                "mode": mode,
                "fit_seconds": fit_seconds,
                "positive_weights": weights.tolist(),
                "thresholds": thresholds.tolist(),
                **metrics,
            }
        )
        models[mode] = estimators
    comparison = pd.DataFrame(rows).sort_values(
        ["damage_pr_auc_macro", "damage_f1_macro"], ascending=False
    )
    selected_mode = str(comparison.iloc[0]["mode"])
    joblib.dump(models[selected_mode], MODEL_DIR / f"linear_probe_{spec.key}.joblib")
    comparison.to_csv(METRICS_DIR / f"screening_{spec.key}.csv", index=False)
    result = {
        "completed_at": now_iso(),
        "model": asdict(spec),
        "balance": balance,
        "selection_partition": "validation_grouped_by_video",
        "selection_metric": "damage_pr_auc_macro",
        "selected_loss_mode": selected_mode,
        "comparison": comparison.to_dict("records"),
    }
    write_json(output_path, result)
    del encoder
    return result


class TextDamageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, prefix: str):
        self.texts = [prefix + str(text) for text in frame["text"]]
        self.targets = damage_targets(frame)
        self.weights = source_weights(frame)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.targets[index], self.weights[index]


class BatchCollator:
    def __init__(self, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts, targets, weights = zip(*examples)
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return tokens, torch.tensor(np.asarray(targets), dtype=torch.float32), torch.tensor(
            weights, dtype=torch.float32
        )


class TransformerDamageClassifier(nn.Module):
    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
        hidden_size = int(self.backbone.config.hidden_size)
        dropout = float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(DAMAGE_ORDER))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.backbone(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


def make_loader(
    frame: pd.DataFrame,
    tokenizer,
    prefix: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TextDamageDataset(frame, prefix),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=BatchCollator(tokenizer),
        generator=generator,
        pin_memory=False,
    )


@torch.inference_mode()
def predict_loader(
    model: TransformerDamageClassifier,
    loader: DataLoader,
    description: str,
) -> np.ndarray:
    model.eval()
    outputs = []
    for tokens, _, _ in tqdm(loader, desc=description, unit="lote"):
        logits = model({key: value for key, value in tokens.items()})
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(outputs)


def scheduler_for(optimizer, total_steps: int) -> LambdaLR:
    warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, multiplier)


def save_checkpoint(
    path: Path,
    model: TransformerDamageClassifier,
    spec: EncoderSpec,
    epoch: int,
    thresholds: np.ndarray,
    history: list[dict],
    loss_mode: str,
    balance: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_spec": asdict(spec),
            "epoch": epoch,
            "thresholds": thresholds.tolist(),
            "history": history,
            "loss_mode": loss_mode,
            "balance": balance,
            "targets": DAMAGE_ORDER,
            "max_length": MAX_LENGTH,
            "seed": SEED,
        },
        path,
    )


def load_best_model(spec: EncoderSpec) -> tuple[TransformerDamageClassifier, dict]:
    checkpoint_path = MODEL_DIR / spec.key / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_spec"]["revision"] != spec.revision:
        raise ValueError("La revisión del checkpoint no coincide con el modelo solicitado.")
    model = TransformerDamageClassifier(spec)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_finetuning(
    spec: EncoderSpec,
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> dict:
    result_path = METRICS_DIR / f"finetuning_{spec.key}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    set_reproducibility()
    screening = run_linear_screening(spec, frames, force=False)
    loss_mode = screening["selected_loss_mode"]
    train_frame = frames["train"]
    train_y_for_balance = damage_targets(train_frame)
    balance = {
        "ordering": "global 4:1 undersampling before split",
        "training_rows": len(train_frame),
        "training_damage": int(train_y_for_balance.any(axis=1).sum()),
        "training_safe": int((~train_y_for_balance.any(axis=1)).sum()),
    }
    y_train = damage_targets(train_frame)
    pos_weights_np = positive_weights(y_train, loss_mode)
    pos_weights_t = torch.tensor(pos_weights_np, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    model = TransformerDamageClassifier(spec)
    train_loader = make_loader(
        train_frame, tokenizer, spec.prefix, TRAIN_BATCH_SIZE, shuffle=True
    )
    validation_loader = make_loader(
        frames["validation"], tokenizer, spec.prefix, EVAL_BATCH_SIZE, shuffle=False
    )
    y_validation = damage_targets(frames["validation"])
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = scheduler_for(optimizer, total_steps)
    history: list[dict] = []
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    training_start = perf_counter()
    model_output_dir = MODEL_DIR / spec.key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    append_log(
        "finetuning_started",
        model=spec.key,
        rows=len(train_frame),
        loss_mode=loss_mode,
        epochs=MAX_EPOCHS,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_start = perf_counter()
        cumulative_loss = 0.0
        seen = 0
        progress = tqdm(
            train_loader,
            desc=f"{spec.key} · época {epoch}/{MAX_EPOCHS}",
            unit="lote",
        )
        for batch_index, (tokens, targets, weights) in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model({key: value for key, value in tokens.items()})
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weights_t, reduction="none"
            )
            per_sample = element_loss.mean(dim=1)
            loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_rows = len(targets)
            cumulative_loss += float(loss.detach()) * batch_rows
            seen += batch_rows
            progress.set_postfix(
                loss=f"{cumulative_loss / seen:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if batch_index % 200 == 0:
                append_log(
                    "training_progress",
                    model=spec.key,
                    epoch=epoch,
                    batch=batch_index,
                    batches=len(train_loader),
                    rows_seen=seen,
                    mean_loss=cumulative_loss / seen,
                )

        validation_scores = predict_loader(
            model, validation_loader, f"{spec.key} · validación época {epoch}"
        )
        thresholds = tune_thresholds(y_validation.astype(np.int8), validation_scores)
        metrics, _, _ = evaluate_damage_scores(
            y_validation, validation_scores, thresholds
        )
        epoch_record = {
            "epoch": epoch,
            "training_loss": cumulative_loss / seen,
            "epoch_seconds": perf_counter() - epoch_start,
            "thresholds": thresholds.tolist(),
            **metrics,
        }
        history.append(epoch_record)
        tqdm.write(
            f"{spec.label} · época {epoch}: "
            f"loss={epoch_record['training_loss']:.4f}, "
            f"PR-AUC val={metrics['damage_pr_auc_macro']:.4f}, "
            f"F1 macro val={metrics['damage_f1_macro']:.4f}, "
            f"recall daño={metrics['damage_recall_micro']:.4f}, "
            f"tiempo={epoch_record['epoch_seconds'] / 60:.1f} min"
        )
        pd.DataFrame(history).to_csv(
            METRICS_DIR / f"historial_{spec.key}.csv", index=False
        )
        score = float(metrics["damage_pr_auc_macro"])
        append_log("epoch_completed", model=spec.key, **epoch_record)
        save_checkpoint(
            model_output_dir / "last_checkpoint.pt",
            model,
            spec,
            epoch,
            thresholds,
            history,
            loss_mode,
            balance,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                model_output_dir / "best_checkpoint.pt",
                model,
                spec,
                epoch,
                thresholds,
                history,
                loss_mode,
                balance,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            append_log("early_stopping", model=spec.key, epoch=epoch, best_epoch=best_epoch)
            break

    tokenizer.save_pretrained(model_output_dir / "tokenizer")
    result = {
        "completed_at": now_iso(),
        "model": asdict(spec),
        "training_rows": len(train_frame),
        "loss_mode": loss_mode,
        "positive_weights": pos_weights_np.tolist(),
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_metric": "validation_damage_pr_auc_macro",
        "best_validation_damage_pr_auc_macro": best_score,
        "training_seconds": perf_counter() - training_start,
        "balance": balance,
        "history": history,
        "checkpoint": str((model_output_dir / "best_checkpoint.pt").relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(model_output_dir / "best_checkpoint.pt"),
    }
    write_json(result_path, result)
    return result


def evaluate_finetuned_model(
    spec: EncoderSpec,
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> dict:
    result_path = METRICS_DIR / f"evaluacion_{spec.key}.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    model, checkpoint = load_best_model(spec)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    outputs = {}
    for split_name in ("validation", "test"):
        frame = frames[split_name]
        loader = make_loader(
            frame, tokenizer, spec.prefix, EVAL_BATCH_SIZE, shuffle=False
        )
        start = perf_counter()
        scores = predict_loader(model, loader, f"{spec.key} · {split_name}")
        elapsed = perf_counter() - start
        metrics, report, _ = evaluate_damage_scores(
            damage_targets(frame), scores, thresholds
        )
        metrics["inference_seconds"] = elapsed
        outputs[split_name] = metrics
        np.save(METRICS_DIR / f"scores_{spec.key}_{split_name}.npy", scores)
        report.to_csv(METRICS_DIR / f"reporte_{spec.key}_{split_name}.csv")
    result = {
        "completed_at": now_iso(),
        "model": asdict(spec),
        "best_epoch": int(checkpoint["epoch"]),
        "thresholds": thresholds.tolist(),
        "metrics": outputs,
    }
    write_json(result_path, result)
    return result


def _qwen_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_LORA_SPEC.model_id, revision=QWEN_LORA_SPEC.revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def qwen_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _qwen_base_model():
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        QWEN_LORA_SPEC.model_id,
        revision=QWEN_LORA_SPEC.revision,
        num_labels=len(DAMAGE_ORDER),
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    return model


def build_qwen_lora_model(device: torch.device | None = None):
    from peft import LoraConfig, TaskType, get_peft_model

    base = _qwen_base_model()
    configuration = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=QWEN_LORA_RANK,
        lora_alpha=QWEN_LORA_ALPHA,
        lora_dropout=QWEN_LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(base, configuration)
    return model.to(device or qwen_device())


@torch.inference_mode()
def predict_qwen_loader(model, loader: DataLoader, description: str) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for tokens, _, _ in tqdm(loader, desc=description, unit="lote"):
        logits = model(
            **{key: value.to(device) for key, value in tokens.items()}
        ).logits
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(outputs)


def _save_qwen_adapter(
    directory: Path,
    model,
    epoch: int,
    thresholds: np.ndarray,
    history: list[dict],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    write_json(
        directory / "training_state.json",
        {
            "model_spec": asdict(QWEN_LORA_SPEC),
            "epoch": epoch,
            "thresholds": thresholds.tolist(),
            "history": history,
            "targets": DAMAGE_ORDER,
            "max_length": MAX_LENGTH,
            "seed": SEED,
            "method": "LoRA sequence classification",
        },
    )


def run_qwen_lora_finetuning(
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> dict:
    """Fine-tuning PEFT de Qwen3-0.6B sobre todo el train seleccionado."""
    result_path = METRICS_DIR / "finetuning_qwen3_06b_lora.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        adapter = ROOT / existing.get("adapter", "")
        state_path = adapter / "training_state.json"
        if (
            existing.get("training_chunk_ids_sha256")
            == ids_sha256(frames["train"]["chunk_id"])
            and state_path.exists()
            and existing.get("training_state_sha256") == sha256_file(state_path)
        ):
            return existing
    set_reproducibility()
    device = qwen_device()
    tokenizer = _qwen_tokenizer()
    model = build_qwen_lora_model(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    train_frame = frames["train"]
    validation_frame = frames["validation"]
    train_loader = make_loader(
        train_frame,
        tokenizer,
        QWEN_LORA_SPEC.prefix,
        QWEN_TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_frame,
        tokenizer,
        QWEN_LORA_SPEC.prefix,
        QWEN_EVAL_BATCH_SIZE,
        shuffle=False,
    )
    y_validation = damage_targets(validation_frame)
    pos_weights_np = positive_weights(damage_targets(train_frame), "sqrt_positive_weight")
    pos_weights_t = torch.tensor(
        pos_weights_np, dtype=torch.float32, device=device
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable, lr=QWEN_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / QWEN_GRADIENT_ACCUMULATION
    )
    scheduler = scheduler_for(
        optimizer, optimizer_steps_per_epoch * QWEN_MAX_EPOCHS
    )
    output_dir = MODEL_DIR / QWEN_LORA_SPEC.key
    tokenizer.save_pretrained(output_dir / "tokenizer")
    history = []
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    training_start = perf_counter()
    append_log(
        "qwen_lora_started",
        rows=len(train_frame),
        device=str(device),
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )
    for epoch in range(1, QWEN_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = perf_counter()
        cumulative_loss = 0.0
        seen = 0
        optimizer_steps = 0
        progress = tqdm(
            train_loader,
            desc=f"{QWEN_LORA_SPEC.key} · época {epoch}/{QWEN_MAX_EPOCHS}",
            unit="lote",
        )
        for batch_index, (tokens, targets, weights) in enumerate(progress, start=1):
            tokens = {key: value.to(device) for key, value in tokens.items()}
            targets = targets.to(device)
            weights = weights.to(device)
            logits = model(**tokens).logits
            element_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weights_t, reduction="none"
            )
            per_sample = element_loss.mean(dim=1)
            loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            (loss / QWEN_GRADIENT_ACCUMULATION).backward()
            batch_rows = len(targets)
            cumulative_loss += float(loss.detach()) * batch_rows
            seen += batch_rows
            should_step = (
                batch_index % QWEN_GRADIENT_ACCUMULATION == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            progress.set_postfix(
                loss=f"{cumulative_loss / seen:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                pasos=optimizer_steps,
            )
            if batch_index % 200 == 0:
                append_log(
                    "qwen_training_progress",
                    epoch=epoch,
                    batch=batch_index,
                    batches=len(train_loader),
                    rows_seen=seen,
                    mean_loss=cumulative_loss / seen,
                )
        validation_scores = predict_qwen_loader(
            model,
            validation_loader,
            f"{QWEN_LORA_SPEC.key} · validación época {epoch}",
        )
        thresholds = tune_thresholds(
            y_validation.astype(np.int8), validation_scores
        )
        metrics, _, _ = evaluate_damage_scores(
            y_validation, validation_scores, thresholds
        )
        epoch_record = {
            "epoch": epoch,
            "training_loss": cumulative_loss / seen,
            "epoch_seconds": perf_counter() - epoch_start,
            "optimizer_steps": optimizer_steps,
            "thresholds": thresholds.tolist(),
            **metrics,
        }
        history.append(epoch_record)
        tqdm.write(
            f"{QWEN_LORA_SPEC.label} · época {epoch}: "
            f"loss={epoch_record['training_loss']:.4f}, "
            f"PR-AUC val={metrics['damage_pr_auc_macro']:.4f}, "
            f"F1 macro val={metrics['damage_f1_macro']:.4f}, "
            f"recall daño={metrics['damage_recall_micro']:.4f}, "
            f"tiempo={epoch_record['epoch_seconds'] / 60:.1f} min"
        )
        _save_qwen_adapter(
            output_dir / "last_adapter", model, epoch, thresholds, history
        )
        score = float(metrics["damage_pr_auc_macro"])
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _save_qwen_adapter(
                output_dir / "best_adapter", model, epoch, thresholds, history
            )
        else:
            stale_epochs += 1
        append_log("qwen_epoch_completed", **epoch_record)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    state_path = output_dir / "best_adapter" / "training_state.json"
    result = {
        "completed_at": now_iso(),
        "model": asdict(QWEN_LORA_SPEC),
        "method": "LoRA parameter-efficient fine-tuning",
        "device": str(device),
        "training_rows": len(train_frame),
        "training_chunk_ids_sha256": ids_sha256(train_frame["chunk_id"]),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "trainable_fraction": trainable_parameters / total_parameters,
        "lora": {
            "rank": QWEN_LORA_RANK,
            "alpha": QWEN_LORA_ALPHA,
            "dropout": QWEN_LORA_DROPOUT,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "batch_size": QWEN_TRAIN_BATCH_SIZE,
        "gradient_accumulation": QWEN_GRADIENT_ACCUMULATION,
        "effective_batch_size": (
            QWEN_TRAIN_BATCH_SIZE * QWEN_GRADIENT_ACCUMULATION
        ),
        "max_epochs": QWEN_MAX_EPOCHS,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_damage_pr_auc_macro": best_score,
        "training_seconds": perf_counter() - training_start,
        "history": history,
        "adapter": str((output_dir / "best_adapter").relative_to(ROOT)),
        "training_state_sha256": sha256_file(state_path),
    }
    write_json(result_path, result)
    return result


def load_qwen_lora_best(device: torch.device | None = None):
    from peft import PeftModel

    adapter_dir = MODEL_DIR / QWEN_LORA_SPEC.key / "best_adapter"
    state = json.loads(
        (adapter_dir / "training_state.json").read_text(encoding="utf-8")
    )
    base = _qwen_base_model()
    model = PeftModel.from_pretrained(base, adapter_dir).to(device or qwen_device())
    return model, state


def evaluate_qwen_lora_model(
    frames: dict[str, pd.DataFrame],
    force: bool = False,
) -> dict:
    result_path = METRICS_DIR / "evaluacion_qwen3_06b_lora.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    tokenizer = _qwen_tokenizer()
    device = qwen_device()
    model, state = load_qwen_lora_best(device)
    thresholds = np.asarray(state["thresholds"], dtype=float)
    outputs = {}
    classification_reports = {}
    for split_name in ("validation", "test"):
        frame = frames[split_name]
        loader = make_loader(
            frame,
            tokenizer,
            QWEN_LORA_SPEC.prefix,
            QWEN_EVAL_BATCH_SIZE,
            shuffle=False,
        )
        start = perf_counter()
        scores = predict_qwen_loader(
            model, loader, f"{QWEN_LORA_SPEC.key} · {split_name}"
        )
        elapsed = perf_counter() - start
        metrics, report, _ = evaluate_damage_scores(
            damage_targets(frame), scores, thresholds
        )
        metrics["inference_seconds"] = elapsed
        outputs[split_name] = metrics
        np.save(METRICS_DIR / f"scores_qwen3_06b_lora_{split_name}.npy", scores)
        classification_reports[split_name] = report.reset_index().rename(
            columns={"index": "category"}
        ).to_dict("records")
    result = {
        "completed_at": now_iso(),
        "model": asdict(QWEN_LORA_SPEC),
        "best_epoch": int(state["epoch"]),
        "thresholds": thresholds.tolist(),
        "device": str(device),
        "metrics": outputs,
        "classification_reports": classification_reports,
    }
    write_json(result_path, result)
    return result


def analyze_qwen_operational(
    frames: dict[str, pd.DataFrame],
    evaluation: dict | None = None,
) -> dict:
    evaluation = evaluation or evaluate_qwen_lora_model(frames)
    return analyze_operational_scores(
        model_key=QWEN_LORA_SPEC.key,
        model_label=QWEN_LORA_SPEC.label,
        frames=frames,
        validation_scores=np.load(
            METRICS_DIR / "scores_qwen3_06b_lora_validation.npy"
        ),
        test_scores=np.load(METRICS_DIR / "scores_qwen3_06b_lora_test.npy"),
        damage_thresholds=np.asarray(evaluation["thresholds"], dtype=float),
        output_path=METRICS_DIR / "operacion_qwen3_06b_lora.json",
    )


def _fasttext_text(value: object) -> str:
    """Normaliza una observación al formato de una línea exigido por fastText."""
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _write_fasttext_training_file(frame: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in frame.to_dict("records"):
        labels = " ".join(f"__label__{label}" for label in row["coarse_labels"])
        lines.append(f"{labels} {_fasttext_text(row['text'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_fasttext_cli() -> Path:
    """Obtiene el CLI oficial v0.9.2; en Windows lo compila con MinGW."""
    executable = MODEL_DIR / "tools" / "fasttext.exe"
    if executable.exists():
        return executable
    source_dir = CACHE_DIR / "fasttext" / "source_v0.9.2"
    source_executable = source_dir / "fasttext.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.9.2",
                "https://github.com/facebookresearch/fastText.git",
                str(source_dir),
            ],
            check=True,
        )
    if not source_executable.exists():
        make = shutil.which("mingw32-make") or shutil.which("make")
        if make is None:
            raise RuntimeError(
                "fastText requiere `mingw32-make`/`g++` en Windows. "
                "No se encontró el compilador necesario."
            )
        cxxflags = (
            "CXXFLAGS=-pthread -std=c++11 -march=native -O3 "
            "-funroll-loops -DNDEBUG -include cstdint"
        )
        subprocess.run(
            [make, "-C", str(source_dir), "-j", str(min(8, os.cpu_count() or 1)), cxxflags],
            check=True,
        )
    shutil.copy2(source_executable, executable)
    return executable


def _fit_fasttext(
    frame: pd.DataFrame,
    experiment_id: str,
    model_parameters: dict | None = None,
) -> tuple[Path, dict]:
    parameters = {
        **PRIOR_CLASSICAL_CONFIGS[FASTTEXT_KEY],
        **(model_parameters or {}),
    }
    training_path = CACHE_DIR / "fasttext" / f"{experiment_id}.train.txt"
    _write_fasttext_training_file(frame, training_path)
    executable = ensure_fasttext_cli()
    output_prefix = MODEL_DIR / "fasttext_candidates" / experiment_id
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    subprocess.run(
        [
            str(executable),
            "supervised",
            "-input",
            str(training_path),
            "-output",
            str(output_prefix),
            "-lr",
            str(parameters["lr"]),
            "-epoch",
            str(parameters["epoch"]),
            "-wordNgrams",
            str(parameters["wordNgrams"]),
            "-bucket",
            str(parameters["bucket"]),
            "-dim",
            str(parameters["dim"]),
            "-loss",
            str(parameters["loss"]),
            "-thread",
            str(min(8, os.cpu_count() or 1)),
            "-verbose",
            "2",
        ],
        check=True,
    )
    elapsed = perf_counter() - start
    model_path = output_prefix.with_suffix(".bin")
    return model_path, {
        "model": FASTTEXT_KEY,
        "model_label": CLASSICAL_MODEL_LABELS[FASTTEXT_KEY],
        "features": int(parameters["dim"]),
        "feature_seconds": 0.0,
        "fit_seconds": elapsed,
        "training_seconds": elapsed,
        "text_column": "text",
        "flash_pseudo_weight": None,
        "sample_weights_supported": False,
        "model_parameters": parameters,
        "configuration_origin": (
            "PLN_clases sesión 4 + receta multietiqueta OVA oficial de fastText"
        ),
        "fasttext_cli": str(executable.relative_to(ROOT)),
        "fasttext_cli_sha256": sha256_file(executable),
    }


def _fasttext_scores(model: Path, frame: pd.DataFrame) -> np.ndarray:
    input_path = CACHE_DIR / "fasttext" / f"predict_{os.getpid()}.txt"
    input_path.write_text(
        "\n".join(_fasttext_text(value) for value in frame["text"]) + "\n",
        encoding="utf-8",
    )
    executable = ensure_fasttext_cli()
    completed = subprocess.run(
        [
            str(executable),
            "predict-prob",
            str(model),
            str(input_path),
            str(len(COARSE_ORDER)),
            "0.0",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output_lines = completed.stdout.splitlines()
    input_path.unlink(missing_ok=True)
    if len(output_lines) != len(frame):
        raise RuntimeError(
            f"fastText devolvió {len(output_lines)} filas para {len(frame)} entradas."
        )
    scores = np.zeros((len(frame), len(COARSE_ORDER)), dtype=np.float32)
    column_by_label = {
        f"__label__{label}": index for index, label in enumerate(COARSE_ORDER)
    }
    for row_index, line in enumerate(output_lines):
        tokens = line.split()
        for token_index in range(0, len(tokens), 2):
            if token_index + 1 >= len(tokens):
                break
            column = column_by_label.get(tokens[token_index])
            if column is not None:
                scores[row_index, column] = float(tokens[token_index + 1])
    return scores


def _evaluate_fasttext(
    model: Path,
    thresholds: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    start = perf_counter()
    scores = _fasttext_scores(model, frame)
    elapsed = perf_counter() - start
    predictions = constrained_coarse_predictions(scores, thresholds)
    summary, report = coarse_metrics(target_matrix(frame), predictions, scores)
    summary["inference_seconds"] = elapsed
    summary["milliseconds_per_1000"] = 1_000_000 * elapsed / len(frame)
    return summary, report, scores


def run_classical_benchmarks(
    frames: dict[str, pd.DataFrame],
    force: bool = False,
    validation_only: bool = False,
) -> dict:
    """Entrena seis clásicos sobre el único train 4:1 seleccionado.

    Todos los candidatos usan exactamente las particiones creadas al inicio
    del cuaderno. La selección usa validación y el test se consulta después de
    fijar el ganador.
    """
    output_path = METRICS_DIR / (
        "screening_modelos_clasicos.json"
        if validation_only
        else "comparacion_modelos_clasicos.json"
    )
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("training_chunk_ids_sha256")
            == ids_sha256(frames["train"]["chunk_id"])
            and existing.get("training_content_sha256")
            == training_content_sha256(frames["train"])
            and len(existing.get("comparison", [])) == len(CLASSICAL_MODEL_ORDER)
            and bool(existing.get("validation_only")) == validation_only
        ):
            return existing
    train_frame = frames["train"].reset_index(drop=True)
    validation = frames["validation"]
    test = frames["test"]
    fitted = {}
    validation_scores_by_id = {}
    validation_rows = []
    append_log(
        "classical_benchmark_started",
        train_rows=len(train_frame),
        models=len(CLASSICAL_MODEL_ORDER),
    )
    progress = tqdm(
        total=len(CLASSICAL_MODEL_ORDER),
        desc="Modelos clásicos",
        unit="modelo",
    )
    for model_name in CLASSICAL_MODEL_ORDER:
        experiment_id = model_name
        progress.set_description_str(
            f"Clásicos · {CLASSICAL_MODEL_LABELS[model_name]}"
        )
        candidate_start = perf_counter()
        if model_name == FASTTEXT_KEY:
            model, diagnostics = _fit_fasttext(train_frame, experiment_id)
            validation_scores = _fasttext_scores(model, validation)
            thresholds = tune_thresholds(target_matrix(validation), validation_scores)
            metrics, _, _ = _evaluate_fasttext(model, thresholds, validation)
            fitted[experiment_id] = {
                "kind": "fasttext",
                "model": model,
                "thresholds": thresholds,
            }
        else:
            parameters = PRIOR_CLASSICAL_CONFIGS[model_name]
            model, diagnostics = fit_candidate(
                model_name,
                train_frame,
                max_features=50_000,
                text_column="text",
                model_parameters=parameters,
                flash_pseudo_weight=0.50,
            )
            validation_scores = tune_candidate(model, validation, text_column="text")
            metrics, _, _ = evaluate_candidate(
                model, validation, text_column="text"
            )
            diagnostics["configuration_origin"] = (
                "Hiperparámetros fijados antes de esta ejecución"
            )
            fitted[experiment_id] = {
                "kind": "sklearn",
                "model": model,
                "thresholds": model.thresholds,
            }
        validation_scores_by_id[experiment_id] = validation_scores
        validation_rows.append(
            {
                "experiment_id": experiment_id,
                "model": model_name,
                "model_label": CLASSICAL_MODEL_LABELS[model_name],
                "train_rows": len(train_frame),
                **diagnostics,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        append_log(
            "classical_candidate_validated",
            experiment_id=experiment_id,
            validation_damage_pr_auc_macro=metrics["damage_pr_auc_macro"],
        )
        progress.update(1)
        progress.set_postfix(
            {
                "PR-AUC val": f"{metrics['damage_pr_auc_macro']:.4f}",
                "seg": f"{perf_counter() - candidate_start:.1f}",
            },
            refresh=True,
        )
    progress.close()
    validation_comparison = pd.DataFrame(validation_rows)
    ranked_candidates = validation_comparison.sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"],
        ascending=False,
    )
    winner_id = str(ranked_candidates.iloc[0]["experiment_id"])
    if validation_only:
        eligible = ranked_candidates.loc[
            ranked_candidates["model"].ne("dummy_prior")
        ].head(CLASSICAL_TUNING_TOP_K)
        result = {
            "schema_version": "2.1",
            "completed_at": now_iso(),
            "dataset": "balanced_4to1_grouped_split",
            "selected_dataset_rows": len(frames["balanced_all"]),
            "training_rows": len(train_frame),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "training_chunk_ids_sha256": ids_sha256(train_frame["chunk_id"]),
            "training_content_sha256": training_content_sha256(train_frame),
            "validation_chunk_ids_sha256": ids_sha256(validation["chunk_id"]),
            "selection_partition": "validation_screening_only",
            "selection_metric": "damage_pr_auc_macro",
            "validation_only": True,
            "test_evaluated": False,
            "screening_winner_id": winner_id,
            "top_candidates": eligible["model"].astype(str).tolist(),
            "fixed_configurations": PRIOR_CLASSICAL_CONFIGS,
            "hyperparameter_search_performed": False,
            "comparison": ranked_candidates.to_dict("records"),
        }
        ranked_candidates.to_csv(
            METRICS_DIR / "screening_modelos_clasicos.csv", index=False
        )
        write_json(output_path, result)
        append_log(
            "classical_screening_completed",
            top_candidates=result["top_candidates"],
        )
        return result

    final_rows = []
    reports = {}
    scores_by_id = {}
    for row in validation_comparison.to_dict("records"):
        experiment_id = row["experiment_id"]
        fitted_entry = fitted[experiment_id]
        if fitted_entry["kind"] == "fasttext":
            metrics, report, scores = _evaluate_fasttext(
                fitted_entry["model"], fitted_entry["thresholds"], test
            )
        else:
            metrics, report, scores = evaluate_candidate(
                fitted_entry["model"], test, text_column="text"
            )
        reports[experiment_id] = report
        scores_by_id[experiment_id] = scores
        final_rows.append(
            {**row, **{f"test_{key}": value for key, value in metrics.items()}}
        )
    comparison = pd.DataFrame(final_rows).sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"],
        ascending=False,
    )
    comparison.to_csv(METRICS_DIR / "comparacion_modelos_clasicos.csv", index=False)
    for experiment_id, report in reports.items():
        report.to_csv(METRICS_DIR / f"reporte_test_{experiment_id}.csv")

    winner_entry = fitted[winner_id]
    winner_model = winner_entry["model"]
    classical_dir = MODEL_DIR / "baseline_clasico_mismo_split"
    classical_dir.mkdir(parents=True, exist_ok=True)
    if winner_entry["kind"] == "fasttext":
        classical_model_path = classical_dir / "mejor_modelo_clasico.bin"
        shutil.copy2(winner_model, classical_model_path)
        write_json(
            classical_dir / "umbrales_fasttext.json",
            {
                label: float(value)
                for label, value in zip(COARSE_ORDER, winner_entry["thresholds"])
            },
        )
    else:
        classical_model_path = classical_dir / "mejor_modelo_clasico.joblib"
        save_coarse_model(winner_model, classical_model_path)
    winner_scores = scores_by_id[winner_id][:, 1:]
    winner_validation_scores = validation_scores_by_id[winner_id][:, 1:]
    np.save(METRICS_DIR / "scores_clasico_ganador_test.npy", winner_scores)
    np.save(
        METRICS_DIR / "scores_clasico_ganador_validation.npy",
        winner_validation_scores,
    )
    winner_row = comparison.set_index("experiment_id").loc[winner_id]

    result = {
        "schema_version": "2.0",
        "completed_at": now_iso(),
        "dataset": "balanced_4to1_grouped_split",
        "selected_dataset_rows": len(frames["balanced_all"]),
        "training_rows": len(train_frame),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "training_chunk_ids_sha256": ids_sha256(train_frame["chunk_id"]),
        "training_content_sha256": training_content_sha256(train_frame),
        "selection_partition": "validation",
        "selection_metric": "damage_pr_auc_macro",
        "validation_only": False,
        "test_evaluated": True,
        "winner_id": winner_id,
        "winner_model": str(winner_row["model"]),
        "winner_label": str(winner_row["model_label"]),
        "winner_thresholds": winner_entry["thresholds"].tolist(),
        "winner_model_path": str(classical_model_path.relative_to(ROOT)),
        "winner_model_sha256": sha256_file(classical_model_path),
        "winner_validation_metrics": {
            key.removeprefix("validation_"): float(value)
            for key, value in winner_row.items()
            if key.startswith("validation_") and isinstance(value, (int, float, np.number))
        },
        "winner_test_metrics": {
            key.removeprefix("test_"): float(value)
            for key, value in winner_row.items()
            if key.startswith("test_") and isinstance(value, (int, float, np.number))
        },
        "fixed_configurations": PRIOR_CLASSICAL_CONFIGS,
        "hyperparameter_search_performed": False,
        "professor_model_source": (
            "PLN_clases/clase4/Cuadernos/nlp_sesion4_1_FastText_Intro.ipynb"
        ),
        "comparison": comparison.to_dict("records"),
    }
    write_json(output_path, result)
    append_log("classical_benchmark_completed", winner=winner_id)
    return result


def _fit_tuned_classical_candidate(
    model_name: str,
    frame: pd.DataFrame,
    experiment_id: str,
    parameters: dict,
) -> tuple[dict, dict]:
    """Ajusta un candidato clásico con una interfaz común sklearn/fastText."""
    if model_name == FASTTEXT_KEY:
        model, diagnostics = _fit_fasttext(
            frame, experiment_id, model_parameters=parameters
        )
        return {"kind": "fasttext", "model": model}, diagnostics
    max_features = int(parameters.get("max_features", 50_000))
    model_parameters = {
        key: value for key, value in parameters.items() if key != "max_features"
    }
    model, diagnostics = fit_candidate(
        model_name,
        frame,
        max_features=max_features,
        text_column="text",
        model_parameters=model_parameters,
        flash_pseudo_weight=0.50,
    )
    return {"kind": "sklearn", "model": model}, diagnostics


def _tuned_candidate_scores(entry: dict, frame: pd.DataFrame) -> np.ndarray:
    if entry["kind"] == "fasttext":
        return _fasttext_scores(entry["model"], frame)
    return entry["model"].predict_scores(frame["text"].tolist())


def _remove_temporary_fasttext_candidate(model_path: Path, experiment_id: str) -> None:
    model_path.unlink(missing_ok=True)
    model_path.with_suffix(".vec").unlink(missing_ok=True)
    (CACHE_DIR / "fasttext" / f"{experiment_id}.train.txt").unlink(missing_ok=True)


def tune_top_classical_models(
    frames: dict[str, pd.DataFrame],
    screening: dict,
    top_k: int = CLASSICAL_TUNING_TOP_K,
    folds: int = CLASSICAL_TUNING_FOLDS,
    force: bool = False,
) -> dict:
    """Optimiza los mejores clásicos sin usar test para ajustar o seleccionar.

    El screening fija candidatos con validación. La búsqueda se realiza dentro
    de ``train`` mediante GroupKFold por ``video_id``. Después cada candidato
    se reentrena en todo train, sus umbrales se calibran en validación y el
    ganador se congela antes de evaluar test.
    """
    if not bool(screening.get("validation_only")):
        raise ValueError("La búsqueda requiere el screening clásico sin acceso a test.")
    ranked = pd.DataFrame(screening.get("comparison", [])).sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"],
        ascending=False,
    )
    selected_models = (
        ranked.loc[ranked["model"].ne("dummy_prior"), "model"]
        .astype(str)
        .drop_duplicates()
        .head(top_k)
        .tolist()
    )
    if len(selected_models) != top_k:
        raise ValueError(f"El screening no produjo {top_k} clásicos ajustables.")
    missing_grids = [name for name in selected_models if name not in CLASSICAL_TUNING_GRIDS]
    if missing_grids:
        raise ValueError(f"Faltan espacios de búsqueda para {missing_grids}.")

    search_definition = {
        "top_k": top_k,
        "folds": folds,
        "selection_metric": "mean_group_cv_damage_pr_auc_macro",
        "selected_models": selected_models,
        "grids": {name: CLASSICAL_TUNING_GRIDS[name] for name in selected_models},
    }
    search_sha = hashlib.sha256(
        json.dumps(search_definition, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_path = METRICS_DIR / "comparacion_modelos_clasicos_optimizados.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        artifact_value = str(existing.get("winner_model_path") or "")
        artifact = ROOT / Path(artifact_value)
        if (
            existing.get("training_content_sha256")
            == training_content_sha256(frames["train"])
            and existing.get("validation_chunk_ids_sha256")
            == ids_sha256(frames["validation"]["chunk_id"])
            and existing.get("test_chunk_ids_sha256")
            == ids_sha256(frames["test"]["chunk_id"])
            and existing.get("search_definition_sha256") == search_sha
            and bool(artifact_value)
            and artifact.is_file()
            and existing.get("winner_model_sha256") == sha256_file(artifact)
        ):
            return existing

    train = frames["train"].reset_index(drop=True)
    validation = frames["validation"].reset_index(drop=True)
    test = frames["test"].reset_index(drop=True)
    groups = train["video_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    fold_indices = list(splitter.split(train, groups=groups))
    for train_indices, validation_indices in fold_indices:
        if set(groups[train_indices]) & set(groups[validation_indices]):
            raise AssertionError("GroupKFold produjo fuga de video.")

    total_cv_fits = sum(len(CLASSICAL_TUNING_GRIDS[name]) for name in selected_models) * folds
    progress = tqdm(total=total_cv_fits + len(selected_models), unit="ajuste")
    cv_rows = []
    best_parameters = {}
    append_log(
        "classical_tuning_started",
        selected_models=selected_models,
        folds=folds,
        candidate_configurations=total_cv_fits // folds,
    )
    for model_name in selected_models:
        model_rows = []
        for configuration_index, parameters in enumerate(
            CLASSICAL_TUNING_GRIDS[model_name], start=1
        ):
            fold_scores = []
            fold_seconds = []
            for fold_index, (fit_indices, score_indices) in enumerate(fold_indices, start=1):
                experiment_id = (
                    f"cv_{model_name}_c{configuration_index:02d}_f{fold_index}"
                )
                progress.set_description_str(
                    f"CV clásico · {CLASSICAL_MODEL_LABELS[model_name]} "
                    f"{configuration_index}/{len(CLASSICAL_TUNING_GRIDS[model_name])} "
                    f"fold {fold_index}/{folds}"
                )
                started = perf_counter()
                entry, _ = _fit_tuned_classical_candidate(
                    model_name,
                    train.iloc[fit_indices].reset_index(drop=True),
                    experiment_id,
                    dict(parameters),
                )
                fold_frame = train.iloc[score_indices].reset_index(drop=True)
                scores = _tuned_candidate_scores(entry, fold_frame)
                value = float(
                    average_precision_score(
                        target_matrix(fold_frame)[:, 1:], scores[:, 1:], average="macro"
                    )
                )
                elapsed = perf_counter() - started
                fold_scores.append(value)
                fold_seconds.append(elapsed)
                if entry["kind"] == "fasttext":
                    _remove_temporary_fasttext_candidate(entry["model"], experiment_id)
                progress.update(1)
                progress.set_postfix(
                    {"PR-AUC CV": f"{value:.4f}", "seg": f"{elapsed:.1f}"},
                    refresh=True,
                )
            record = {
                "model": model_name,
                "model_label": CLASSICAL_MODEL_LABELS[model_name],
                "configuration_index": configuration_index,
                "parameters": dict(parameters),
                "fold_damage_pr_auc_macro": fold_scores,
                "mean_cv_damage_pr_auc_macro": float(np.mean(fold_scores)),
                "std_cv_damage_pr_auc_macro": float(np.std(fold_scores, ddof=1)),
                "cv_training_seconds": float(sum(fold_seconds)),
            }
            model_rows.append(record)
            cv_rows.append(record)
        best = sorted(
            model_rows,
            key=lambda row: (
                -row["mean_cv_damage_pr_auc_macro"],
                row["std_cv_damage_pr_auc_macro"],
                row["configuration_index"],
            ),
        )[0]
        best_parameters[model_name] = dict(best["parameters"])

    fitted = {}
    validation_rows = []
    validation_scores_by_model = {}
    for model_name in selected_models:
        experiment_id = f"tuned__{model_name}"
        progress.set_description_str(
            f"Ajuste final · {CLASSICAL_MODEL_LABELS[model_name]}"
        )
        entry, diagnostics = _fit_tuned_classical_candidate(
            model_name, train, experiment_id, best_parameters[model_name]
        )
        validation_scores = _tuned_candidate_scores(entry, validation)
        thresholds = tune_thresholds(target_matrix(validation), validation_scores)
        entry["thresholds"] = thresholds
        if entry["kind"] == "sklearn":
            entry["model"].thresholds = thresholds
        validation_predictions = constrained_coarse_predictions(
            validation_scores, thresholds
        )
        validation_metrics, _ = coarse_metrics(
            target_matrix(validation), validation_predictions, validation_scores
        )
        fitted[model_name] = entry
        validation_scores_by_model[model_name] = validation_scores
        cv_best = next(
            row
            for row in cv_rows
            if row["model"] == model_name
            and row["parameters"] == best_parameters[model_name]
        )
        validation_rows.append(
            {
                "experiment_id": experiment_id,
                "model": model_name,
                "model_label": CLASSICAL_MODEL_LABELS[model_name],
                "train_rows": len(train),
                "best_parameters": best_parameters[model_name],
                "mean_cv_damage_pr_auc_macro": cv_best[
                    "mean_cv_damage_pr_auc_macro"
                ],
                "std_cv_damage_pr_auc_macro": cv_best[
                    "std_cv_damage_pr_auc_macro"
                ],
                **diagnostics,
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        progress.update(1)
        progress.set_postfix(
            {"PR-AUC val": f"{validation_metrics['damage_pr_auc_macro']:.4f}"},
            refresh=True,
        )
    progress.close()

    # El ganador queda fijado exclusivamente con validación antes de esta línea.
    validation_comparison = pd.DataFrame(validation_rows).sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"],
        ascending=False,
    )
    winner_model_name = str(validation_comparison.iloc[0]["model"])
    winner_id = str(validation_comparison.iloc[0]["experiment_id"])

    final_rows = []
    reports = {}
    test_scores_by_model = {}
    for row in validation_comparison.to_dict("records"):
        model_name = str(row["model"])
        entry = fitted[model_name]
        test_scores = _tuned_candidate_scores(entry, test)
        predictions = constrained_coarse_predictions(test_scores, entry["thresholds"])
        test_metrics, report = coarse_metrics(
            target_matrix(test), predictions, test_scores
        )
        reports[model_name] = report
        test_scores_by_model[model_name] = test_scores
        final_rows.append(
            {**row, **{f"test_{key}": value for key, value in test_metrics.items()}}
        )
    comparison = pd.DataFrame(final_rows).sort_values(
        ["validation_damage_pr_auc_macro", "validation_damage_f1_macro"],
        ascending=False,
    )
    comparison.to_csv(
        METRICS_DIR / "comparacion_modelos_clasicos_optimizados.csv", index=False
    )
    pd.DataFrame(cv_rows).to_json(
        METRICS_DIR / "busqueda_hiperparametros_clasicos.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    for model_name, report in reports.items():
        report.to_csv(METRICS_DIR / f"reporte_test_ajustado_{model_name}.csv")

    winner_entry = fitted[winner_model_name]
    classical_dir = MODEL_DIR / "baseline_clasico_mismo_split"
    classical_dir.mkdir(parents=True, exist_ok=True)
    if winner_entry["kind"] == "fasttext":
        classical_model_path = classical_dir / "mejor_modelo_clasico.bin"
        shutil.copy2(winner_entry["model"], classical_model_path)
        write_json(
            classical_dir / "umbrales_fasttext.json",
            {
                label: float(value)
                for label, value in zip(COARSE_ORDER, winner_entry["thresholds"])
            },
        )
    else:
        classical_model_path = classical_dir / "mejor_modelo_clasico.joblib"
        save_coarse_model(winner_entry["model"], classical_model_path)
    np.save(
        METRICS_DIR / "scores_clasico_ganador_validation.npy",
        validation_scores_by_model[winner_model_name][:, 1:],
    )
    np.save(
        METRICS_DIR / "scores_clasico_ganador_test.npy",
        test_scores_by_model[winner_model_name][:, 1:],
    )
    winner_row = comparison.set_index("experiment_id").loc[winner_id]
    result = {
        "schema_version": "2.1",
        "completed_at": now_iso(),
        "dataset": "balanced_4to1_grouped_split",
        "selected_dataset_rows": len(frames["balanced_all"]),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "training_chunk_ids_sha256": ids_sha256(train["chunk_id"]),
        "training_content_sha256": training_content_sha256(train),
        "validation_chunk_ids_sha256": ids_sha256(validation["chunk_id"]),
        "test_chunk_ids_sha256": ids_sha256(test["chunk_id"]),
        "selection_partition": "validation_after_group_cv_tuning",
        "selection_metric": "damage_pr_auc_macro",
        "test_access_policy": "test evaluated only after winner frozen on validation",
        "validation_only": False,
        "test_evaluated": True,
        "top_candidates": selected_models,
        "group_cv_folds": folds,
        "group_variable": "video_id",
        "search_definition_sha256": search_sha,
        "hyperparameter_search_performed": True,
        "winner_id": winner_id,
        "winner_model": winner_model_name,
        "winner_label": str(winner_row["model_label"]),
        "winner_parameters": best_parameters[winner_model_name],
        "winner_thresholds": winner_entry["thresholds"].tolist(),
        "winner_model_path": str(classical_model_path.relative_to(ROOT)),
        "winner_model_sha256": sha256_file(classical_model_path),
        "winner_validation_metrics": {
            key.removeprefix("validation_"): float(value)
            for key, value in winner_row.items()
            if key.startswith("validation_") and isinstance(value, (int, float, np.number))
        },
        "winner_test_metrics": {
            key.removeprefix("test_"): float(value)
            for key, value in winner_row.items()
            if key.startswith("test_") and isinstance(value, (int, float, np.number))
        },
        "best_parameters_by_model": best_parameters,
        "cv_results": cv_rows,
        "comparison": comparison.to_dict("records"),
    }
    write_json(output_path, result)
    append_log(
        "classical_tuning_completed",
        winner=winner_id,
        winner_parameters=best_parameters[winner_model_name],
    )
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Intervalo Wilson bilateral 95 % para una proporción binomial."""
    if trials <= 0:
        return [math.nan, math.nan]
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)
        )
        / denominator
    )
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _binary_routing_metrics(y_true: np.ndarray, routed: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    routed = np.asarray(routed, dtype=bool)
    tp = int((y_true & routed).sum())
    fn = int((y_true & ~routed).sum())
    fp = int((~y_true & routed).sum())
    tn = int((~y_true & ~routed).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    return {
        "n": int(len(y_true)),
        "true_damage": int(y_true.sum()),
        "routed_to_human": int(routed.sum()),
        "review_rate": float(routed.mean()),
        "automatic_coverage": float((~routed).mean()),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": float(precision),
        "recall": float(recall),
        "negative_predictive_value": float(npv),
        "precision_wilson_95": wilson_interval(tp, tp + fp),
        "recall_wilson_95": wilson_interval(tp, tp + fn),
        "npv_wilson_95": wilson_interval(tn, tn + fn),
    }


def tune_human_alert_cutoff(
    y_damage: np.ndarray,
    damage_scores: np.ndarray,
    decision_thresholds: np.ndarray,
    recall_target: float = ALERT_VALIDATION_RECALL_TARGET,
) -> tuple[float, dict]:
    """Maximiza cobertura automática sujeto a recall de daño en validación."""
    true_any = y_damage.astype(bool).any(axis=1)
    risk_margin = np.max(damage_scores - decision_thresholds, axis=1)
    candidates = np.unique(risk_margin)[::-1]
    selected_cutoff = float(np.nextafter(risk_margin.min(), -np.inf))
    for cutoff in candidates:
        routed = risk_margin >= cutoff
        recall = (true_any & routed).sum() / true_any.sum()
        if recall >= recall_target:
            selected_cutoff = float(cutoff)
            break
    metrics = _binary_routing_metrics(true_any, risk_margin >= selected_cutoff)
    metrics["risk_margin_cutoff"] = selected_cutoff
    metrics["validation_recall_target"] = recall_target
    return selected_cutoff, metrics


def _autonomous_metrics(
    y_damage: np.ndarray,
    damage_scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    predictions = damage_scores >= thresholds
    true_any = y_damage.astype(bool).any(axis=1)
    predicted_any = predictions.any(axis=1)
    binary = _binary_routing_metrics(true_any, predicted_any)
    category_recall = {
        label: float(
            recall_score(
                y_damage[:, index], predictions[:, index], zero_division=0
            )
        )
        for index, label in enumerate(DAMAGE_ORDER)
    }
    damage_f1_macro = float(
        f1_score(y_damage, predictions, average="macro", zero_division=0)
    )
    checks = {
        "precision_point_at_least_0_90": binary["precision"]
        >= AUTONOMOUS_MIN_PRECISION,
        "recall_point_at_least_0_90": binary["recall"] >= AUTONOMOUS_MIN_RECALL,
        "precision_wilson_lower_at_least_0_85": binary["precision_wilson_95"][0]
        >= AUTONOMOUS_MIN_WILSON_LOWER,
        "recall_wilson_lower_at_least_0_85": binary["recall_wilson_95"][0]
        >= AUTONOMOUS_MIN_WILSON_LOWER,
        "minimum_category_recall_at_least_0_80": min(category_recall.values())
        >= AUTONOMOUS_MIN_CATEGORY_RECALL,
        "damage_f1_macro_at_least_0_75": damage_f1_macro
        >= AUTONOMOUS_MIN_DAMAGE_F1_MACRO,
    }
    return {
        **binary,
        "damage_f1_macro": damage_f1_macro,
        "category_recall": category_recall,
        "minimum_category_recall": float(min(category_recall.values())),
        "performance_gate_checks": checks,
        "performance_gate_passed": bool(all(checks.values())),
    }


def _human_alert_gate(metrics: dict) -> dict:
    checks = {
        "recall_point_at_least_0_90": metrics["recall"]
        >= HUMAN_ALERT_MIN_TEST_RECALL,
        "recall_wilson_lower_at_least_0_85": metrics["recall_wilson_95"][0]
        >= HUMAN_ALERT_MIN_WILSON_LOWER,
        "negative_predictive_value_at_least_0_95": metrics[
            "negative_predictive_value"
        ]
        >= HUMAN_ALERT_MIN_NPV,
        "review_rate_at_most_0_60": metrics["review_rate"]
        <= HUMAN_ALERT_MAX_REVIEW_RATE,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def analyze_operational_scores(
    model_key: str,
    model_label: str,
    frames: dict[str, pd.DataFrame],
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    damage_thresholds: np.ndarray,
    output_path: Path | None = None,
) -> dict:
    """Evalúa autonomía y triage humano sin recalibrar con el test."""
    y_validation = damage_targets(frames["validation"]).astype(np.int8)
    y_test = damage_targets(frames["test"]).astype(np.int8)
    cutoff, validation_alert = tune_human_alert_cutoff(
        y_validation,
        validation_scores,
        damage_thresholds,
    )
    test_margin = np.max(test_scores - damage_thresholds, axis=1)
    test_alert = _binary_routing_metrics(
        y_test.astype(bool).any(axis=1), test_margin >= cutoff
    )
    test_alert["risk_margin_cutoff"] = cutoff
    test_alert_gate = _human_alert_gate(test_alert)
    validation_autonomous = _autonomous_metrics(
        y_validation, validation_scores, damage_thresholds
    )
    test_autonomous = _autonomous_metrics(y_test, test_scores, damage_thresholds)
    human_gold_rows = int(frames["test"]["label_source"].eq("human_coarse").sum())
    evidence_checks = {
        "independent_human_gold_test": False,
        "natural_prevalence_test": False,
        "prospective_production_pilot": False,
    }
    autonomy_performance_passed = bool(
        validation_autonomous["performance_gate_passed"]
        and test_autonomous["performance_gate_passed"]
    )
    autonomy_authorized = bool(
        autonomy_performance_passed and all(evidence_checks.values())
    )
    alert_supported = bool(test_alert_gate["passed"])
    if autonomy_authorized:
        operating_mode = "autonomous"
    elif alert_supported:
        operating_mode = "human_review_alert_pilot"
    else:
        operating_mode = "research_only"
    result = {
        "completed_at": now_iso(),
        "model_key": model_key,
        "model_label": model_label,
        "decision_thresholds": damage_thresholds.tolist(),
        "validation_damage_pr_auc_macro": float(
            average_precision_score(y_validation, validation_scores, average="macro")
        ),
        "test_damage_pr_auc_macro": float(
            average_precision_score(y_test, test_scores, average="macro")
        ),
        "autonomous": {
            "validation": validation_autonomous,
            "test": test_autonomous,
            "performance_gate_passed": autonomy_performance_passed,
            "evidence_gate_checks": evidence_checks,
            "authorized": autonomy_authorized,
            "interpretation": (
                "La autonomía exige superar desempeño y evidencia externa. El test actual "
                "está balanceado y contiene etiquetas mayormente LLM, no un gold standard "
                "humano independiente."
            ),
        },
        "human_review_alert": {
            "selection_partition": "validation",
            "risk_definition": "max(score_categoria - umbral_decision_categoria)",
            "validation": validation_alert,
            "test": test_alert,
            "test_gate": test_alert_gate,
            "supported": alert_supported,
        },
        "test_human_gold_rows": human_gold_rows,
        "test_human_gold_fraction": human_gold_rows / len(frames["test"]),
        "recommended_operating_mode": operating_mode,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def analyze_classical_winner(
    classical: dict,
    frames: dict[str, pd.DataFrame],
) -> dict:
    return analyze_operational_scores(
        model_key=f"classical__{classical['winner_model']}",
        model_label=classical["winner_label"],
        frames=frames,
        validation_scores=np.load(
            METRICS_DIR / "scores_clasico_ganador_validation.npy"
        ),
        test_scores=np.load(METRICS_DIR / "scores_clasico_ganador_test.npy"),
        damage_thresholds=np.asarray(classical["winner_thresholds"][1:], dtype=float),
        output_path=METRICS_DIR / "operacion_mejor_clasico.json",
    )


def analyze_transformer_operational(
    spec: EncoderSpec,
    frames: dict[str, pd.DataFrame],
    evaluation: dict | None = None,
) -> dict:
    evaluation = evaluation or evaluate_finetuned_model(spec, frames)
    return analyze_operational_scores(
        model_key=spec.key,
        model_label=spec.label,
        frames=frames,
        validation_scores=np.load(METRICS_DIR / f"scores_{spec.key}_validation.npy"),
        test_scores=np.load(METRICS_DIR / f"scores_{spec.key}_test.npy"),
        damage_thresholds=np.asarray(evaluation["thresholds"], dtype=float),
        output_path=METRICS_DIR / f"operacion_{spec.key}.json",
    )


def operational_comparison_frame(analyses: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        autonomous = analysis["autonomous"]["test"]
        alert = analysis["human_review_alert"]["test"]
        rows.append(
            {
                "modelo": analysis["model_label"],
                "model_key": analysis["model_key"],
                "PR-AUC daño validación": analysis[
                    "validation_damage_pr_auc_macro"
                ],
                "PR-AUC daño test": analysis["test_damage_pr_auc_macro"],
                "precisión autónoma test": autonomous["precision"],
                "recall autónomo test": autonomous["recall"],
                "recall mínimo categoría": autonomous["minimum_category_recall"],
                "F1 macro daño test": autonomous["damage_f1_macro"],
                "puerta desempeño autónomo": analysis["autonomous"][
                    "performance_gate_passed"
                ],
                "autonomía autorizada": analysis["autonomous"]["authorized"],
                "tasa revisión humana": alert["review_rate"],
                "recall alerta test": alert["recall"],
                "límite inferior recall 95%": alert["recall_wilson_95"][0],
                "VPN auto-paso": alert["negative_predictive_value"],
                "daños no alertados": alert["false_negatives"],
                "alerta humana respaldada": analysis["human_review_alert"][
                    "supported"
                ],
                "modo recomendado": analysis["recommended_operating_mode"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["PR-AUC daño validación", "recall alerta test"], ascending=False
    )


def select_production_candidate(analyses: Iterable[dict]) -> dict:
    analyses = list(analyses)
    autonomous = [item for item in analyses if item["autonomous"]["authorized"]]
    alert = [
        item
        for item in analyses
        if item["human_review_alert"]["supported"]
    ]
    if autonomous:
        eligible = autonomous
        mode = "autonomous"
        status = "candidate_requires_external_governance_approval"
    elif alert:
        eligible = alert
        mode = "human_review_alert_pilot"
        status = "recommended_for_controlled_human_in_the_loop_pilot"
    else:
        eligible = []
        mode = "none"
        status = "no_model_is_ready_for_production"
    selected = (
        max(
            eligible,
            key=lambda item: item["validation_damage_pr_auc_macro"],
        )
        if eligible
        else None
    )
    result = {
        "completed_at": now_iso(),
        "selection_uses_validation_metric": "damage_pr_auc_macro",
        "test_is_acceptance_gate_not_ranking_metric": True,
        "selected_model_key": selected["model_key"] if selected else None,
        "selected_model_label": selected["model_label"] if selected else None,
        "operating_mode": mode,
        "status": status,
        "autonomous_deployment_supported": mode == "autonomous",
        "human_review_alert_supported": mode == "human_review_alert_pilot",
        "mandatory_limitations": [
            "El test 4:1 no reproduce la prevalencia natural de producción.",
            "El test no es un gold standard humano independiente.",
            "Se requiere piloto prospectivo con decisiones humanas y auditoría por categoría.",
        ],
    }
    write_json(METRICS_DIR / "seleccion_operativa_produccion.json", result)
    return result


MODEL_REGISTRY_PATH = MODEL_DIR / "registro_modelos_comparables.json"
OPERATIONAL_REPORT_PATH = ROOT / "resultados" / "INFORME_DECISION_OPERATIVA_MODERADOR.md"
QWEN_REPORT_PATH = ROOT / "resultados" / "INFORME_FINETUNING_QWEN3_LORA.md"


def _artifact_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def write_model_registry(
    classical: dict,
    trainings: dict[str, dict],
    evaluations: dict[str, dict],
    production_selection: dict,
    qwen_training: dict | None = None,
    qwen_evaluation: dict | None = None,
) -> dict:
    classical_evaluation_path = (
        METRICS_DIR / "comparacion_modelos_clasicos_optimizados.json"
        if (METRICS_DIR / "comparacion_modelos_clasicos_optimizados.json").exists()
        else METRICS_DIR / "comparacion_modelos_clasicos.json"
    )
    records = [
        {
            "model_key": f"classical__{classical['winner_model']}",
            "model_label": classical["winner_label"],
            "family": "classical",
            "artifact": _artifact_record(ROOT / classical["winner_model_path"]),
            "thresholds": classical["winner_thresholds"][1:],
            "evaluation": _artifact_record(classical_evaluation_path),
        }
    ]
    for key, training in trainings.items():
        checkpoint = ROOT / training["checkpoint"]
        records.append(
            {
                "model_key": key,
                "model_label": MODEL_SPECS[key].label,
                "family": "transformer_full_finetuning",
                "artifact": _artifact_record(checkpoint),
                "thresholds": evaluations[key]["thresholds"],
                "evaluation": _artifact_record(
                    METRICS_DIR / f"evaluacion_{key}.json"
                ),
            }
        )
    if qwen_training is not None and qwen_evaluation is not None:
        adapter_dir = ROOT / qwen_training["adapter"]
        adapter_files = [
            _artifact_record(path)
            for path in sorted(adapter_dir.rglob("*"))
            if path.is_file()
        ]
        records.append(
            {
                "model_key": QWEN_LORA_SPEC.key,
                "model_label": QWEN_LORA_SPEC.label,
                "family": "causal_lm_lora_sequence_classification",
                "artifact_directory": str(adapter_dir.relative_to(ROOT)),
                "artifact_files": adapter_files,
                "thresholds": qwen_evaluation["thresholds"],
                "evaluation": _artifact_record(
                    METRICS_DIR / "evaluacion_qwen3_06b_lora.json"
                ),
            }
        )
    registry = {
        "schema_version": "1.1",
        "created_at": now_iso(),
        "dataset": str(BALANCED_DATASET_PATH.relative_to(ROOT)),
        "dataset_sha256": sha256_file(BALANCED_DATASET_PATH),
        "split_manifest": str(BALANCED_TRAIN_MANIFEST_PATH.relative_to(ROOT)),
        "split_manifest_sha256": sha256_file(BALANCED_TRAIN_MANIFEST_PATH),
        "selection": production_selection,
        "models": records,
    }
    write_json(MODEL_REGISTRY_PATH, registry)
    return registry


def load_model_registry() -> dict:
    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Falta el registro de modelos. Ejecute la comparación final del 04_2."
        )
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("dataset_sha256") != sha256_file(BALANCED_DATASET_PATH):
        raise ValueError("El registro no corresponde al dataset 4:1 actual.")
    manifest_path = ROOT / registry["split_manifest"]
    if (
        not manifest_path.exists()
        or registry.get("split_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("El manifiesto de particiones está ausente o fue alterado.")
    for model in registry["models"]:
        artifact = model.get("artifact")
        if artifact:
            path = ROOT / artifact["path"]
            if not path.exists() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"Artefacto ausente o alterado: {path}")
        for artifact_file in model.get("artifact_files", []):
            path = ROOT / artifact_file["path"]
            if not path.exists() or sha256_file(path) != artifact_file["sha256"]:
                raise ValueError(f"Artefacto ausente o alterado: {path}")
        evaluation = model.get("evaluation")
        if evaluation:
            path = ROOT / evaluation["path"]
            if not path.exists() or sha256_file(path) != evaluation["sha256"]:
                raise ValueError(f"Resultado ausente o alterado: {path}")
    return registry


def registered_artifact_frame(registry: dict | None = None) -> pd.DataFrame:
    """Tabla auditable de artefactos que pueden volver a cargarse."""
    registry = registry or load_model_registry()
    rows = []
    for item in registry["models"]:
        artifact = item.get("artifact")
        path = artifact["path"] if artifact else item["artifact_directory"]
        evaluation = item.get("evaluation", {})
        rows.append(
            {
                "model_key": item["model_key"],
                "modelo": item["model_label"],
                "familia": item["family"],
                "artefacto": path,
                "resultado": evaluation.get("path"),
            }
        )
    return pd.DataFrame(rows)


def load_registered_model(
    model_key: str,
    device: torch.device | None = None,
    registry: dict | None = None,
) -> tuple[object, dict]:
    """Carga un modelo registrado y devuelve también sus metadatos verificables.

    Los modelos fastText se devuelven como ruta ``.bin`` porque este proyecto
    usa el ejecutable CLI compilado; los modelos joblib y PyTorch se cargan.
    """
    registry = registry or load_model_registry()
    try:
        item = next(row for row in registry["models"] if row["model_key"] == model_key)
    except StopIteration as exc:
        raise KeyError(f"Modelo no registrado: {model_key}") from exc
    family = item["family"]
    if family == "classical":
        path = ROOT / item["artifact"]["path"]
        model = path if path.suffix.lower() == ".bin" else joblib.load(path)
        return model, item
    if family == "transformer_full_finetuning":
        model, checkpoint = load_best_model(MODEL_SPECS[model_key])
        if device is not None:
            model = model.to(device)
        return model, {**item, "checkpoint": checkpoint}
    if family == "causal_lm_lora_sequence_classification":
        model, state = load_qwen_lora_best(device)
        return model, {**item, "training_state": state}
    raise ValueError(f"Familia no soportada: {family}")


def write_operational_decision_report(
    analyses: Iterable[dict],
    selection: dict,
) -> None:
    analyses = list(analyses)
    rows = []
    for analysis in analyses:
        autonomous = analysis["autonomous"]["test"]
        alert = analysis["human_review_alert"]["test"]
        rows.append(
            "| "
            + " | ".join(
                [
                    analysis["model_label"],
                    f"{analysis['validation_damage_pr_auc_macro']:.4f}",
                    f"{analysis['test_damage_pr_auc_macro']:.4f}",
                    f"{autonomous['precision']:.4f}",
                    f"{autonomous['recall']:.4f}",
                    "sí" if analysis["autonomous"]["authorized"] else "no",
                    f"{alert['review_rate']:.4f}",
                    f"{alert['recall']:.4f}",
                    f"{alert['negative_predictive_value']:.4f}",
                    "sí" if analysis["human_review_alert"]["supported"] else "no",
                ]
            )
            + " |"
        )
    selected = selection.get("selected_model_label") or "ninguno"
    report = f"""# Decisión operativa del moderador de contenido

Fecha: {now_iso()}

## Método

La evaluación distingue dos usos. Para autonomía se exigen simultáneamente precisión y recall de daño ≥ 0,90, límites inferiores Wilson 95 % ≥ 0,85, recall mínimo por categoría ≥ 0,80 y F1 macro de daño ≥ 0,75. Además se exige evidencia externa: test humano independiente, prevalencia natural y piloto prospectivo.

Para asistencia humana se calibra **sólo en validación** el mayor umbral de margen `max(score - umbral_de_categoría)` que capture al menos {ALERT_VALIDATION_RECALL_TARGET:.0%} del daño. En test se exige recall ≥ {HUMAN_ALERT_MIN_TEST_RECALL:.0%}, límite inferior Wilson ≥ {HUMAN_ALERT_MIN_WILSON_LOWER:.0%}, VPN ≥ {HUMAN_ALERT_MIN_NPV:.0%} y revisión ≤ {HUMAN_ALERT_MAX_REVIEW_RATE:.0%}. Esto formaliza el intercambio cobertura–riesgo de la clasificación selectiva (Geifman & El-Yaniv, 2017) y el intercambio costo–recall de moderación humana (Tonneau et al., 2024).

Los puntos de corte son criterios del proyecto, declarados para hacer auditable la decisión; no son estándares universales.

## Resultados

| Modelo | PR-AUC val. | PR-AUC test | Precisión autónoma | Recall autónomo | Autonomía | Tasa revisión | Recall alerta | VPN auto-paso | Alerta respaldada |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|
{os.linesep.join(rows)}

## Decisión

- Modelo seleccionado: **{selected}**.
- Modo: `{selection['operating_mode']}`.
- Estado: `{selection['status']}`.
- Moderación autónoma respaldada: **{'sí' if selection['autonomous_deployment_supported'] else 'no'}**.
- Alerta con revisión humana respaldada: **{'sí' if selection['human_review_alert_supported'] else 'no'}**.

Aunque un modelo supere la puerta numérica, el test actual fue construido con prevalencia 4:1 y está etiquetado mayormente por LLM. Por eso no autoriza sanciones, eliminación o bloqueo autónomos. La salida defendible, si supera la puerta de alerta, es un piloto controlado: el modelo prioriza casos y una persona toma la decisión final. Antes de producción se requiere un test aleatorio de prevalencia natural con gold standard humano, análisis por subgrupo y monitoreo de deriva.

## Referencias (APA 7)

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

Tonneau, M., Quinta de Castro, P. V., Lasri, K., Farouq, I., Subramanian, L., Orozco-Olvera, V., & Fraiberger, S. P. (2024). NAIJAHATE: Evaluating hate speech detection on Nigerian Twitter using representative data. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 9020–9040). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.488/
"""
    OPERATIONAL_REPORT_PATH.write_text(report, encoding="utf-8")


def write_qwen_report(
    training: dict,
    evaluation: dict,
    analyses: Iterable[dict],
    selection: dict,
    registry: dict,
) -> None:
    comparison = operational_comparison_frame(analyses)
    rows = []
    for row in comparison.to_dict("records"):
        rows.append(
            f"| {row['modelo']} | {row['PR-AUC daño validación']:.4f} | "
            f"{row['PR-AUC daño test']:.4f} | {row['recall autónomo test']:.4f} | "
            f"{row['tasa revisión humana']:.4f} | {row['recall alerta test']:.4f} | "
            f"{row['modo recomendado']} |"
        )
    report = f"""# Fine-tuning Qwen3-0.6B con LoRA para moderación gruesa

Fecha: {now_iso()}

## Elección del modelo

Se eligió `Qwen/Qwen3-0.6B-Base` (revisión `{QWEN_LORA_SPEC.revision}`) porque es un checkpoint abierto Apache-2.0 de 0,6B parámetros, declara preentrenamiento en 119 idiomas y Transformers ofrece `Qwen3ForSequenceClassification`. GPT-3 no es un checkpoint abierto local. La alternativa abierta de OpenAI, `gpt-oss-20b`, tiene 20B parámetros y su guía oficial de fine-tuning presupone una H100 de 80 GB. DeepSeek-V2-Lite y los checkpoints generales Kimi/Moonlight más pequeños parten de 16B. Qwen se ajustó con LoRA para mantener el experimento reproducible en el hardware disponible.

## Configuración reproducible

- Dataset: {registry['dataset']} (`SHA-256 {registry['dataset_sha256']}`).
- Train: {training['training_rows']:,} chunks; etiquetas: cinco categorías gruesas.
- Dispositivo: `{training['device']}`.
- Parámetros totales: {training['total_parameters']:,}; entrenables: {training['trainable_parameters']:,} ({training['trainable_fraction']:.4%}).
- LoRA: rango {QWEN_LORA_RANK}, alpha {QWEN_LORA_ALPHA}, dropout {QWEN_LORA_DROPOUT}; módulos Q/K/V/O.
- Batch físico {QWEN_TRAIN_BATCH_SIZE}; acumulación {QWEN_GRADIENT_ACCUMULATION}; batch efectivo {training['effective_batch_size']}.
- Learning rate {QWEN_LEARNING_RATE}; longitud {MAX_LENGTH}; máximo {QWEN_MAX_EPOCHS} épocas.
- Mejor época: {training['best_epoch']}; tiempo: {training['training_seconds'] / 3600:.2f} h.
- Adaptador: `{training['adapter']}`.

No se entrenaron etiquetas finas ni flags transversales.

## Comparación con artefactos del 04_2

| Modelo | PR-AUC val. | PR-AUC test | Recall autónomo | Tasa revisión | Recall alerta | Modo |
|---|---:|---:|---:|---:|---:|---|
{os.linesep.join(rows)}

## Decisión final

- Seleccionado: **{selection.get('selected_model_label') or 'ninguno'}**.
- Modo defendible: `{selection['operating_mode']}`.
- Estado: `{selection['status']}`.

El test balanceado y mayormente pseudoetiquetado impide respaldar moderación autónoma. Si la puerta de alerta se supera, el uso defendible es un piloto human-in-the-loop: priorización automática y decisión final humana.

## Referencias (APA 7)

DeepSeek-AI. (2024). *DeepSeek-V2-Lite* [Modelo de lenguaje]. Hugging Face. https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Moonshot AI. (n.d.). *Modelos publicados por Moonshot AI* [Colección de modelos]. Hugging Face. https://huggingface.co/moonshotai/models

OpenAI. (n.d.). *Fine-tuning a multilingual reasoner with Hugging Face*. https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers

Qwen Team. (2025). *Qwen3 technical report*. arXiv. https://doi.org/10.48550/arXiv.2505.09388

Qwen Team. (2025). *Qwen3-0.6B-Base* [Modelo de lenguaje]. Hugging Face. https://huggingface.co/Qwen/Qwen3-0.6B-Base

Tonneau, M., Quinta de Castro, P. V., Lasri, K., Farouq, I., Subramanian, L., Orozco-Olvera, V., & Fraiberger, S. P. (2024). NAIJAHATE: Evaluating hate speech detection on Nigerian Twitter using representative data. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 9020–9040). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.488/
"""
    QWEN_REPORT_PATH.write_text(report, encoding="utf-8")


def _bootstrap_metric_triplet(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, float]:
    predictions = scores >= thresholds
    valid_columns = y.sum(axis=0) > 0
    pr_auc = float(
        average_precision_score(y[:, valid_columns], scores[:, valid_columns], average="macro")
    )
    macro_f1 = float(f1_score(y, predictions, average="macro", zero_division=0))
    micro_recall = float(recall_score(y, predictions, average="micro", zero_division=0))
    return pr_auc, macro_f1, micro_recall


def paired_video_bootstrap(
    winner: EncoderSpec,
    frames: dict[str, pd.DataFrame],
    replicates: int = BOOTSTRAP_REPLICATES,
    force: bool = False,
) -> dict:
    output_path = METRICS_DIR / "bootstrap_transformer_vs_mejor_clasico.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))
    frame = frames["test"].reset_index(drop=True)
    y = damage_targets(frame).astype(np.int8)
    transformer_scores = np.load(METRICS_DIR / f"scores_{winner.key}_test.npy")
    classical_scores = np.load(METRICS_DIR / "scores_clasico_ganador_test.npy")
    transformer_eval = json.loads(
        (METRICS_DIR / f"evaluacion_{winner.key}.json").read_text(encoding="utf-8")
    )
    classical_eval = json.loads(
        (METRICS_DIR / "comparacion_modelos_clasicos.json").read_text(encoding="utf-8")
    )
    transformer_thresholds = np.asarray(transformer_eval["thresholds"], dtype=float)
    classical_thresholds = np.asarray(
        classical_eval["winner_thresholds"][1:], dtype=float
    )
    groups = frame["video_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(SEED + 99)
    differences = np.empty((replicates, 3), dtype=float)
    progress = tqdm(range(replicates), desc="Bootstrap pareado por video", unit="réplica")
    for replicate in progress:
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        transformer_metrics = _bootstrap_metric_triplet(
            y[indices], transformer_scores[indices], transformer_thresholds
        )
        classical_metrics = _bootstrap_metric_triplet(
            y[indices], classical_scores[indices], classical_thresholds
        )
        differences[replicate] = (
            np.asarray(transformer_metrics) - np.asarray(classical_metrics)
        )
    names = ("damage_pr_auc_macro", "damage_f1_macro", "damage_recall_micro")
    summaries = {}
    for column, name in enumerate(names):
        values = differences[:, column]
        summaries[name] = {
            "mean_difference": float(values.mean()),
            "ci_95_percentile": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_difference_gt_zero": float((values > 0).mean()),
        }
    np.save(METRICS_DIR / "bootstrap_differences.npy", differences)
    result = {
        "completed_at": now_iso(),
        "method": "paired percentile bootstrap resampling video clusters with replacement",
        "seed": SEED + 99,
        "replicates": replicates,
        "test_rows": len(frame),
        "test_videos": int(len(unique_groups)),
        "winner_selected_on_validation": winner.key,
        "comparison": "winner_transformer_minus_best_classical_same_split",
        "best_classical": classical_eval["winner_id"],
        "metrics": summaries,
    }
    write_json(output_path, result)
    return result


def create_figures(
    audit: dict,
    evaluations: dict[str, dict],
    trainings: dict[str, dict],
    classical: dict,
) -> None:
    integrated = audit["parts"]["integrated"]
    balanced = audit["parts"]["balanced_all"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = ["Integrado", "Balanceado 4:1"]
    safe = [integrated["safe_rows"], balanced["safe_rows"]]
    damage = [integrated["damage_rows"], balanced["damage_rows"]]
    axes[0].bar(labels, safe, label="SEGURO", color="#4C78A8")
    axes[0].bar(labels, damage, bottom=safe, label="Algún daño", color="#E45756")
    axes[0].set_title("Submuestreo global antes de la partición")
    axes[0].set_ylabel("Chunks")
    axes[0].legend()

    display_keys = [*evaluations, "best_classical"]
    display_labels = [
        *[MODEL_SPECS[key].label for key in evaluations],
        classical["winner_label"],
    ]
    validation_values = [
        *[
            evaluations[key]["metrics"]["validation"]["damage_pr_auc_macro"]
            for key in evaluations
        ],
        classical["winner_validation_metrics"]["damage_pr_auc_macro"],
    ]
    test_values = [
        *[
            evaluations[key]["metrics"]["test"]["damage_pr_auc_macro"]
            for key in evaluations
        ],
        classical["winner_test_metrics"]["damage_pr_auc_macro"],
    ]
    x = np.arange(len(display_keys))
    width = 0.36
    axes[1].bar(x - width / 2, validation_values, width, label="Validación")
    axes[1].bar(x + width / 2, test_values, width, label="Test")
    axes[1].set_xticks(x, display_labels, rotation=15, ha="right")
    axes[1].set_title("Transformers frente al mejor clásico")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "balance_y_comparacion_transformers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, training in trainings.items():
        history = training["history"]
        ax.plot(
            [row["epoch"] for row in history],
            [row["damage_pr_auc_macro"] for row in history],
            marker="o",
            label=MODEL_SPECS[key].label,
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("PR-AUC macro de daño en validación")
    ax.set_title("Selección de época sin consultar el test")
    ax.set_xticks(range(1, MAX_EPOCHS + 1))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "curvas_validacion_transformers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    audit: dict,
    trainings: dict[str, dict],
    evaluations: dict[str, dict],
    classical: dict,
    bootstrap: dict,
    winner_key: str,
) -> None:
    transformer_rows = []
    for key, evaluation in evaluations.items():
        val = evaluation["metrics"]["validation"]
        test = evaluation["metrics"]["test"]
        transformer_rows.append(
            f"| {MODEL_SPECS[key].label} | {trainings[key]['best_epoch']} | "
            f"{_format_metric(val['damage_pr_auc_macro'])} | "
            f"{_format_metric(test['damage_pr_auc_macro'])} | "
            f"{_format_metric(test['damage_f1_macro'])} | "
            f"{_format_metric(test['damage_recall_micro'])} | "
            f"{_format_metric(test['any_damage_precision'])} | "
            f"{_format_metric(test['any_damage_recall'])} |"
        )
    classical_rows = []
    for row in classical["comparison"]:
        classical_rows.append(
            f"| {row['model_label']} | "
            f"{_format_metric(row['validation_damage_pr_auc_macro'])} | "
            f"{_format_metric(row['test_damage_pr_auc_macro'])} | "
            f"{_format_metric(row['test_damage_f1_macro'])} | "
            f"{_format_metric(row['test_damage_recall_micro'])} |"
        )
    classical_metrics = classical["winner_test_metrics"]
    winner_test = evaluations[winner_key]["metrics"]["test"]
    delta = (
        winner_test["damage_pr_auc_macro"]
        - classical_metrics["damage_pr_auc_macro"]
    )
    ci = bootstrap["metrics"]["damage_pr_auc_macro"]["ci_95_percentile"]
    if delta > 0 and ci[0] > 0:
        conclusion = (
            "El Transformer seleccionado mejora la PR-AUC macro de daño frente al mejor "
            "modelo clásico, "
            "y el intervalo bootstrap por video conserva una diferencia positiva."
        )
    elif delta > 0:
        conclusion = (
            "El Transformer seleccionado obtiene una PR-AUC puntual mayor, pero el intervalo "
            "bootstrap no permite afirmar todavía una mejora estable frente al mejor clásico."
        )
    else:
        conclusion = (
            "El Transformer seleccionado no mejora la PR-AUC macro de daño del mejor clásico; por "
            "tanto no debe reemplazarla con la evidencia actual."
        )
    integrated = audit["parts"]["integrated"]
    balanced = audit["parts"]["balanced_all"]
    report = f"""# Informe del fine-tuning Transformer para categorías gruesas

Fecha de ejecución: {now_iso()}  
Instantánea humana: `{audit['snapshot']}`  
SHA-256 de la instantánea: `{audit['snapshot_sha256']}`

## Resumen

Primero se reentrenaron seis baselines clásicos y después se compararon dos encoders compactos mediante fine-tuning completo. El ganador Transformer se fijó exclusivamente por PR-AUC macro de daño en validación: **{MODEL_SPECS[winner_key].label}**. El test no intervino en el ajuste de modelos, umbrales ni selección.

{conclusion}

Esta evaluación no autoriza moderación autónoma. La aceptabilidad operativa depende también de recall, falsos negativos, calibración y revisión humana por categoría.

## Datos y objetivos

- Unión útil antes de balancear: {integrated['rows']:,} chunks; {integrated['damage_rows']:,} con daño ({integrated['damage_pct']:.2f} %) y {integrated['safe_rows']:,} seguros.
- Muestra balanceada antes de dividir: {balanced['rows']:,} chunks; {balanced['damage_rows']:,} con daño y {balanced['safe_rows']:,} seguros.
- Entrenamiento: {audit['parts']['train']['rows']:,} chunks y {audit['parts']['train']['videos']:,} videos.
- Validación: {audit['parts']['validation']['rows']:,} chunks y {audit['parts']['validation']['videos']:,} videos.
- Test: {audit['parts']['test']['rows']:,} chunks y {audit['parts']['test']['videos']:,} videos.
- Objetivos: {', '.join(DAMAGE_ORDER)}. `SEGURO` se deriva si no se activa daño.
- Etiquetas finas entrenadas: no. Flags transversales entrenados como categorías: no.
- Fuga de videos entre particiones: {audit['video_leakage']}.

## Balanceo reproducible

Se conservaron los {integrated['damage_rows']:,} chunks únicos con daño y se seleccionaron por SHA-256 {SAFE_TO_DAMAGE_RATIO} seguros por cada chunk con daño. Solo después se aplicó la partición aleatoria agrupada por video 70/15/15. La muestra efectiva contiene {balanced['rows']:,} filas: 20 % con algún daño y 80 % `SEGURO` en el conjunto global. Al haber balanceado antes de dividir, validación y test miden comparación controlada bajo esa prevalencia; no estiman directamente el valor predictivo en la prevalencia natural de producción.

Para cada encoder, un linear probe sobre la validación eligió entre BCE normal y una ponderación positiva moderada `sqrt(N_neg/N_pos)`. Este paso no consultó el test.

## Modelos clásicos antes del fine-tuning

Los cinco modelos de los cuadernos 04/04_1 y **fastText supervisado OVA** se ejecutaron primero con configuraciones iniciales fijadas para un screening sin acceso a test. Las tres mejores familias no triviales por PR-AUC macro de daño en validación pasaron a una búsqueda acotada de ocho configuraciones cada una. Cada configuración se comparó con {classical['group_cv_folds']} folds de `GroupKFold` dentro de train, agrupando por `video_id`; por tanto ningún video estuvo a ambos lados de un fold.

Después del CV, cada familia seleccionada se reentrenó con los {audit['parts']['train']['rows']:,} chunks completos de entrenamiento. Los umbrales se calibraron en validación, el ganador se congeló por PR-AUC macro de daño y recién entonces se evaluó test. Los mejores parámetros fueron `{classical['best_parameters_by_model']}`. fastText procede de `PLN_clases/clase4/Cuadernos/nlp_sesion4_1_FastText_Intro.ipynb` y de la receta OVA oficial; a diferencia de los modelos scikit-learn, no admite los pesos por observación, limitación conservada en la comparación.

| Modelo clásico | PR-AUC validación | PR-AUC test | F1 macro test | Recall micro test |
|---|---:|---:|---:|---:|
{os.linesep.join(classical_rows)}

El mejor clásico seleccionado en validación fue **{classical['winner_label']}** (`{classical['winner_id']}`).

## Configuración

- Longitud máxima común: {MAX_LENGTH} tokens.
- Batch de entrenamiento: {TRAIN_BATCH_SIZE}; batch de evaluación: {EVAL_BATCH_SIZE}.
- Optimizador: AdamW; learning rate {LEARNING_RATE}; weight decay {WEIGHT_DECAY}.
- Máximo: {MAX_EPOCHS} épocas; parada temprana con paciencia {EARLY_STOPPING_PATIENCE}.
- Criterio: PR-AUC macro de las cinco categorías de daño en validación.
- Semilla: {SEED}.
- Hardware de esta ejecución: {platform.processor() or 'AMD Ryzen 7 8845HS'}, PyTorch {torch.__version__}, dispositivo CPU.

## Resultados

| Modelo | Mejor época | PR-AUC validación | PR-AUC test | F1 macro test | Recall micro test | Precisión algún daño | Recall algún daño |
|---|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(transformer_rows)}

El ganador Transformer por validación fue `{winner_key}`. Su diferencia de PR-AUC macro de daño frente al mejor clásico en test fue {delta:+.4f}. El bootstrap pareado de {bootstrap['replicates']:,} réplicas, remuestreando los {bootstrap['test_videos']} videos como conglomerados, produjo IC 95 % percentil [{ci[0]:+.4f}, {ci[1]:+.4f}]. Este intervalo cuantifica variación muestral entre videos, no variación entre semillas de entrenamiento.

## Interpretación

La reducción de `SEGURO` disminuye el tiempo de entrenamiento y expone con más frecuencia los positivos, pero cambia la prevalencia de los tres subconjuntos. Un mejor F1 acompañado de pérdida fuerte de precisión no se interpreta automáticamente como una mejora operativa. Antes de desplegar se requiere una evaluación adicional con prevalencia natural y revisión humana de falsos negativos.

La comparación usa una sola semilla por modelo. Para reporte académico definitivo se recomienda repetir ambos fine-tunings con al menos tres semillas y reportar media, desviación e intervalos; el bootstrap actual no reemplaza esa estimación de variabilidad de optimización.

## Figuras y artefactos

- `resultados/figuras/transformer_grueso/balance_y_comparacion_transformers.png`.
- `resultados/figuras/transformer_grueso/curvas_validacion_transformers.png`.
- `resultados/figuras/transformer_grueso/comparacion_modelos_clasicos_antes_transformers.png`.
- `resultados/metricas/transformer_grueso/` contiene auditoría, curvas, scores y reportes por clase.
- `modelos/moderador_transformer_grueso/` contiene checkpoints y tokenizadores.
- `resultados/logs/transformer_grueso/progreso.jsonl` conserva el progreso temporal.

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Huang, Y., Giledereli, B., Köksal, A., Özgür, A., & Ozkirimli, E. (2021). Balancing methods for multi-label text classification with long-tailed class distribution. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 8153–8161). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.643

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers* (pp. 427–431). Association for Computational Linguistics. https://aclanthology.org/E17-2068/

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. *arXiv*. https://doi.org/10.48550/arXiv.2402.05672

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained Transformers. In *Advances in Neural Information Processing Systems* (Vol. 33). https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_all(force: bool = False) -> dict:
    set_reproducibility()
    frames, audit = load_experiment_frames()
    integration = materialize_integrated_dataset(frames, audit, force=force)
    sampled, balance = materialize_balanced_training(frames, force=force)
    audit["integration"] = integration
    audit["moderate_undersampling"] = balance
    write_json(METRICS_DIR / "auditoria_dataset_transformer.json", audit)
    append_log("dataset_ready", rows=len(sampled), balance=balance)

    classical_screening = run_classical_benchmarks(
        frames, force=force, validation_only=True
    )
    classical = tune_top_classical_models(
        frames, classical_screening, force=force
    )
    screenings = {}
    trainings = {}
    evaluations = {}
    for key, spec in MODEL_SPECS.items():
        screenings[key] = run_linear_screening(spec, frames, force=force)
        trainings[key] = run_finetuning(spec, frames, force=force)
        evaluations[key] = evaluate_finetuned_model(spec, frames, force=force)
    winner_key = max(
        MODEL_SPECS,
        key=lambda key: (
            evaluations[key]["metrics"]["validation"]["damage_pr_auc_macro"],
            evaluations[key]["metrics"]["validation"]["damage_f1_macro"],
        ),
    )
    operational_analyses = [
        analyze_classical_winner(classical, frames),
        *[
            analyze_transformer_operational(MODEL_SPECS[key], frames, evaluations[key])
            for key in MODEL_SPECS
        ],
    ]
    production_selection = select_production_candidate(operational_analyses)
    registry = write_model_registry(
        classical, trainings, evaluations, production_selection
    )
    write_operational_decision_report(
        operational_analyses, production_selection
    )
    bootstrap = paired_video_bootstrap(
        MODEL_SPECS[winner_key], frames, force=force
    )
    create_figures(audit, evaluations, trainings, classical)
    write_report(audit, trainings, evaluations, classical, bootstrap, winner_key)
    summary = {
        "completed_at": now_iso(),
        "winner_selected_on_validation": winner_key,
        "screenings": screenings,
        "trainings": trainings,
        "evaluations": evaluations,
        "classical": classical,
        "operational_analyses": operational_analyses,
        "production_selection": production_selection,
        "model_registry": registry,
        "bootstrap": bootstrap,
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }
    write_json(METRICS_DIR / "resumen_experimento_transformer.json", summary)
    append_log("experiment_completed", winner=winner_key)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("audit", "classical", "screen", "finetune", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=(*MODEL_SPECS.keys(), "both"),
        default="both",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility()
    frames, audit = load_experiment_frames()
    integration = materialize_integrated_dataset(frames, audit, force=args.force)
    sampled, balance = materialize_balanced_training(frames, force=args.force)
    audit["integration"] = integration
    audit["moderate_undersampling"] = balance
    write_json(METRICS_DIR / "auditoria_dataset_transformer.json", audit)
    if args.phase == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    if args.phase == "classical":
        screening = run_classical_benchmarks(
            frames, force=args.force, validation_only=True
        )
        result = tune_top_classical_models(
            frames, screening, force=args.force
        )
        print(
            json.dumps(
                {"winner": result["winner_id"], "metrics": result["winner_test_metrics"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.phase == "all" and args.model == "both":
        result = run_all(force=args.force)
        print(json.dumps({"winner": result["winner_selected_on_validation"]}, indent=2))
        return
    selected = MODEL_SPECS if args.model == "both" else {args.model: MODEL_SPECS[args.model]}
    for key, spec in selected.items():
        if args.phase in ("screen", "all"):
            result = run_linear_screening(spec, frames, force=args.force)
        elif args.phase == "finetune":
            result = run_finetuning(spec, frames, force=args.force)
        elif args.phase == "evaluate":
            result = evaluate_finetuned_model(spec, frames, force=args.force)
        else:
            raise AssertionError(args.phase)
        print(json.dumps({"model": key, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
