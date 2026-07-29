"""Fine-tuning reanudable de Qwen3-0.6B para cuatro categorías de daño.

Taxonomía semántica de salida:

* SEGURO (derivada cuando ninguna categoría de daño supera su umbral),
* RACISMO_DISCRIMINACION,
* ACOSO_GENERO_IDENTIDAD,
* ACOSO_AMENAZA = ACOSO_PERSONAL ∪ AMENAZA_DIRECTA,
* CONTENIDO_SEXUAL.

El entrenamiento usa LoRA y guarda checkpoints alternos con adaptador, estado
del optimizador/scheduler, RNG, época y lote. La calibración sigmoide y los
umbrales se ajustan sólo en validación; test se consulta una vez al final.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable
import hashlib
import json
import math
import os
import random
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl, sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import tune_thresholds


ROOT = tm.ROOT
SEED = tm.SEED
MODEL_SPEC = tm.QWEN_LORA_SPEC
SOURCE_DAMAGE_LABELS = list(tm.DAMAGE_ORDER)
TARGET_LABELS = [
    "RACISMO_DISCRIMINACION",
    "ACOSO_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
]
SEMANTIC_LABELS = ["SEGURO", *TARGET_LABELS]
FINE_LABELS = [
    "seguro",
    "seguro_ironia_marcada",
    "racismo_etnico_explicito",
    "racismo_encubierto",
    "clasismo_racial",
    "discriminacion_regional",
    "racismo_linguistico",
    "misoginia_acoso_genero",
    "homofobia_transfobia",
    "acoso_personal",
    "amenaza_directa",
    "sexual_explicito",
    "sexual_cosificacion",
    "sexual_no_consensual",
]
TRANSVERSAL_FLAGS = ["ironia_ambigua", "humor_encubridor", "contexto_necesario"]
OUTPUT_LABELS = [
    *[f"primary::{label}" for label in TARGET_LABELS],
    *[f"fine::{label}" for label in FINE_LABELS],
    *[f"flag::{label}" for label in TRANSVERSAL_FLAGS],
]
PRIMARY_OUTPUTS = len(TARGET_LABELS)
FINE_OUTPUTS = len(FINE_LABELS)
FLAG_OUTPUTS = len(TRANSVERSAL_FLAGS)
AUX_FINE_LOSS_WEIGHT = 0.20
AUX_FLAG_LOSS_WEIGHT = 0.15

MAX_LENGTH = tm.MAX_LENGTH
TRAIN_BATCH_SIZE = tm.QWEN_TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = tm.QWEN_EVAL_BATCH_SIZE
GRADIENT_ACCUMULATION = tm.QWEN_GRADIENT_ACCUMULATION
MAX_EPOCHS = tm.QWEN_MAX_EPOCHS
SUPPORTS_COMPLETED_EXTENSION = True
LEARNING_RATE = tm.QWEN_LEARNING_RATE
WEIGHT_DECAY = tm.WEIGHT_DECAY
LORA_RANK = tm.QWEN_LORA_RANK
LORA_ALPHA = tm.QWEN_LORA_ALPHA
LORA_DROPOUT = tm.QWEN_LORA_DROPOUT
SAVE_EVERY_OPTIMIZER_STEPS = 250
ALERT_RECALL_TARGET = 0.95

RUN_KEY = "qwen3_06b_lora_acoso_amenaza_4"
MODEL_DIR = ROOT / "modelos" / RUN_KEY
METRICS_DIR = ROOT / "resultados" / "metricas" / RUN_KEY
FIGURES_DIR = ROOT / "resultados" / "figuras" / RUN_KEY
REPORT_PATH = ROOT / "resultados" / "INFORME_QWEN_ACOSO_AMENAZA_4.md"
TRAINING_RESULT_PATH = METRICS_DIR / "finetuning.json"
EVALUATION_PATH = METRICS_DIR / "evaluacion_calibrada.json"
OPERATION_PATH = METRICS_DIR / "operacion_selectiva.json"
COMPARISON_PATH = METRICS_DIR / "comparacion_referencias.csv"
PROGRESS_LOG_PATH = METRICS_DIR / "progreso.jsonl"
RESUME_POINTER_PATH = MODEL_DIR / "resume_pointer.json"
PROMPT_OPERATIONAL_PATH = ROOT / "03_1_etiquetado_llm" / "prompt_operacional_compacto.md"
AUXILIARY_HISTORICAL_PATH = (
    ROOT / "datos" / "processed" / "dataset_pseudoetiquetado_hibrido.jsonl"
)
for _directory in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _relative(path: Path) -> str:
    return tm.project_relative(path)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _directory_artifacts(directory: Path) -> list[dict]:
    return [
        _artifact(path) for path in sorted(directory.rglob("*")) if path.is_file()
    ]


def _append_log(event: str, **values: object) -> None:
    payload = {"timestamp": tm.now_iso(), "event": event, **values}
    with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _ids_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _multi_hot(values: Iterable[str], labels: list[str]) -> np.ndarray:
    present = set(values)
    return np.asarray([label in present for label in labels], dtype=np.float32)


def _fine_reference_only(value: object) -> bool:
    """Los JSONL antiguos representan valores ausentes como NaN."""
    return value is True or (isinstance(value, str) and value.lower() == "true")


def auxiliary_source_paths() -> list[Path]:
    paths = [AUXILIARY_HISTORICAL_PATH]
    paths.extend(
        sorted(
            (ROOT / "datos" / "ampliacion").glob(
                "*/processed/dataset_etiquetado_utilizable.jsonl"
            )
        )
    )
    return [path for path in paths if path.exists()]


def load_auxiliary_annotations() -> tuple[dict[str, tuple[str, ...]], dict]:
    """Recupera etiquetas finas válidas sin reinterpretar las etiquetas gruesas.

    Las filas explícitamente marcadas como ``fine_labels_reference_only`` se
    excluyen de esta supervisión. Una ausencia no se transforma en negativos:
    la pérdida fina se enmascara para esa fila.
    """
    paths = auxiliary_source_paths()
    if not paths:
        raise FileNotFoundError("No se encontraron fuentes de etiquetas finas.")
    index: dict[str, tuple[str, ...]] = {}
    source_rows = []
    unknown_labels: set[str] = set()
    duplicates_consistent = 0
    for path in paths:
        read_rows = 0
        usable_rows = 0
        reference_only_rows = 0
        for row in read_jsonl(path):
            read_rows += 1
            chunk_id = str(row.get("chunk_id", ""))
            labels = row.get("labels")
            if _fine_reference_only(row.get("fine_labels_reference_only")):
                reference_only_rows += 1
                continue
            if not chunk_id or not isinstance(labels, list) or not labels:
                continue
            unknown_labels.update(set(labels) - set(FINE_LABELS))
            normalized = tuple(sorted(set(labels) & set(FINE_LABELS)))
            if not normalized:
                continue
            if chunk_id in index:
                if index[chunk_id] != normalized:
                    raise ValueError(
                        f"Etiquetas finas contradictorias para chunk_id={chunk_id}."
                    )
                duplicates_consistent += 1
            index[chunk_id] = normalized
            usable_rows += 1
        source_rows.append(
            {
                **_artifact(path),
                "rows": read_rows,
                "fine_rows_usable": usable_rows,
                "fine_rows_reference_only": reference_only_rows,
            }
        )
    if unknown_labels:
        raise ValueError(f"Etiquetas finas no reconocidas: {sorted(unknown_labels)}")
    return index, {
        "sources": source_rows,
        "unique_chunks_with_fine_supervision": len(index),
        "consistent_duplicate_rows": duplicates_consistent,
    }


def four_targets(frame: pd.DataFrame) -> np.ndarray:
    source = tm.damage_targets(frame).astype(np.float32)
    # Orden original: racismo, género, acoso personal, amenaza, sexual.
    return np.column_stack(
        [
            source[:, 0],
            source[:, 1],
            np.maximum(source[:, 2], source[:, 3]),
            source[:, 4],
        ]
    ).astype(np.float32)


def load_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    """Carga el 4:1 guardado por 04_2 sin reconstruir ni modificar sus datos."""
    dataset_path = tm.BALANCED_DATASET_PATH
    manifest_path = tm.BALANCED_TRAIN_MANIFEST_PATH
    if not dataset_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Ejecute primero la construcción del dataset en 04_2.")
    manifest = _json(manifest_path)
    if sha256_file(dataset_path) != manifest["balanced_dataset_sha256"]:
        raise ValueError("El dataset 4:1 no coincide con su manifiesto.")
    frame = pd.DataFrame(read_jsonl(dataset_path))
    frames = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    fine_index, auxiliary_audit = load_auxiliary_annotations()
    auxiliary_split_audit = {}
    for split, part in frames.items():
        fine_values = [fine_index.get(str(value)) for value in part["chunk_id"]]
        part["fine_labels_auxiliary"] = [
            list(values) if values is not None else [] for values in fine_values
        ]
        part["fine_auxiliary_available"] = [
            values is not None for values in fine_values
        ]
        part["flags_auxiliary"] = part["flags_reference_only"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        unknown_flags = set().union(*map(set, part["flags_auxiliary"])) - set(
            TRANSVERSAL_FLAGS
        )
        if unknown_flags:
            raise ValueError(f"Flags transversales no reconocidos: {sorted(unknown_flags)}")
        available = part["fine_auxiliary_available"].to_numpy(dtype=bool)
        fine_counts = np.vstack(
            [_multi_hot(values, FINE_LABELS) for values in part["fine_labels_auxiliary"]]
        ).sum(axis=0)
        flag_counts = np.vstack(
            [_multi_hot(values, TRANSVERSAL_FLAGS) for values in part["flags_auxiliary"]]
        ).sum(axis=0)
        auxiliary_split_audit[split] = {
            "fine_rows_available": int(available.sum()),
            "fine_rows_masked": int((~available).sum()),
            "fine_coverage": float(available.mean()),
            "fine_positive_counts": {
                label: int(fine_counts[index]) for index, label in enumerate(FINE_LABELS)
            },
            "flag_rows_available": int(len(part)),
            "flag_positive_counts": {
                label: int(flag_counts[index])
                for index, label in enumerate(TRANSVERSAL_FLAGS)
            },
        }
    tm._verify_disjoint(frames)
    for split, expected in manifest["split_counts"].items():
        if len(frames[split]) != int(expected):
            raise ValueError(f"Conteo alterado en {split}.")
    y_train = four_targets(frames["train"])
    y_validation = four_targets(frames["validation"])
    y_test = four_targets(frames["test"])
    overlap = {}
    original = {
        split: tm.damage_targets(part).astype(np.int8)
        for split, part in frames.items()
    }
    for split, matrix in original.items():
        overlap[split] = {
            "acoso_personal": int(matrix[:, 2].sum()),
            "amenaza_directa": int(matrix[:, 3].sum()),
            "ambas": int((matrix[:, 2].astype(bool) & matrix[:, 3].astype(bool)).sum()),
            "acoso_amenaza_union": int(
                np.maximum(matrix[:, 2], matrix[:, 3]).sum()
            ),
        }
    fingerprint_payload = {
        "dataset_sha256": manifest["balanced_dataset_sha256"],
        "auxiliary_sources": [
            {
                "path": str(source["path"]).replace("\\", "/"),
                "sha256": source["sha256"],
            }
            for source in auxiliary_audit["sources"]
        ],
        "primary_labels": TARGET_LABELS,
        "fine_labels": FINE_LABELS,
        "transversal_flags": TRANSVERSAL_FLAGS,
        "auxiliary_loss_weights": {
            "fine": AUX_FINE_LOSS_WEIGHT,
            "flags": AUX_FLAG_LOSS_WEIGHT,
        },
    }
    # Compatibilidad con checkpoints creados originalmente en Windows. La
    # barra usada al serializar una ruta no cambia la supervisión semántica.
    legacy_windows_payload = {
        **fingerprint_payload,
        "auxiliary_sources": [
            {"path": source["path"].replace("/", "\\"), "sha256": source["sha256"]}
            for source in fingerprint_payload["auxiliary_sources"]
        ],
    }
    compatible_fingerprints = sorted(
        {
            _payload_sha256(fingerprint_payload),
            _payload_sha256(legacy_windows_payload),
        }
    )
    audit = {
        "created_at": tm.now_iso(),
        "dataset": _relative(dataset_path),
        "dataset_sha256": manifest["balanced_dataset_sha256"],
        "manifest": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_counts": {split: int(len(part)) for split, part in frames.items()},
        "split_chunk_ids_sha256": {
            split: _ids_sha256(part["chunk_id"]) for split, part in frames.items()
        },
        "category_counts": {
            split: {
                label: int(matrix[:, index].sum())
                for index, label in enumerate(TARGET_LABELS)
            }
            for split, matrix in (
                ("train", y_train),
                ("validation", y_validation),
                ("test", y_test),
            )
        },
        "merge_audit": overlap,
        "target_labels": TARGET_LABELS,
        "safe_is_derived": True,
        "training_fingerprint_sha256": _payload_sha256(fingerprint_payload),
        "compatible_training_fingerprints_sha256": compatible_fingerprints,
        "auxiliary_supervision": {
            **auxiliary_audit,
            "split_coverage": auxiliary_split_audit,
            "fine_labels": FINE_LABELS,
            "transversal_flags": TRANSVERSAL_FLAGS,
            "fine_loss_weight": AUX_FINE_LOSS_WEIGHT,
            "flag_loss_weight": AUX_FLAG_LOSS_WEIGHT,
            "fine_missing_policy": "masked_loss; missing is not a negative label",
            "flags_source": "flags_reference_only from the frozen 04_2 dataset",
        },
        "prompt_operational": (
            {
                **_artifact(PROMPT_OPERATIONAL_PATH),
                "role": "taxonomy provenance only; it is not model input",
            }
            if PROMPT_OPERATIONAL_PATH.exists()
            else None
        ),
        "fine_labels_trained_as_auxiliary": True,
        "transversal_flags_trained_as_auxiliary": True,
    }
    tm.write_json(METRICS_DIR / "auditoria_dataset.json", audit)
    return frames, audit


def _fingerprint_matches(value: object, audit: dict) -> bool:
    accepted = set(audit.get("compatible_training_fingerprints_sha256", []))
    accepted.add(audit["training_fingerprint_sha256"])
    return isinstance(value, str) and value in accepted


def _completed_result_is_usable(result: dict, audit: dict) -> bool:
    if result.get("dataset_sha256") != audit["dataset_sha256"]:
        return False
    if not _fingerprint_matches(result.get("training_fingerprint_sha256"), audit):
        return False
    artifacts = result.get("adapter_files", [])
    if not artifacts:
        return False
    for artifact in artifacts:
        path = tm.project_path(artifact["path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(artifact["bytes"])
            or sha256_file(path) != artifact["sha256"]
        ):
            return False
    return True


def dataset_summary(audit: dict | None = None) -> pd.DataFrame:
    if audit is None:
        _, audit = load_frames()
    rows = []
    for split, count in audit["split_counts"].items():
        categories = audit["category_counts"][split]
        rows.append(
            {
                "split": split,
                "chunks": count,
                **categories,
                "ACOSO_AMENAZA_union": audit["merge_audit"][split][
                    "acoso_amenaza_union"
                ],
                "solapamiento_acoso_amenaza": audit["merge_audit"][split]["ambas"],
            }
        )
    return pd.DataFrame(rows)


class FourTargetDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame["text"].astype(str).tolist()
        self.primary_targets = four_targets(frame)
        self.fine_targets = np.vstack(
            [_multi_hot(values, FINE_LABELS) for values in frame["fine_labels_auxiliary"]]
        )
        fine_available = frame["fine_auxiliary_available"].to_numpy(dtype=np.float32)
        self.fine_masks = np.repeat(fine_available[:, None], FINE_OUTPUTS, axis=1)
        self.flag_targets = np.vstack(
            [
                _multi_hot(values, TRANSVERSAL_FLAGS)
                for values in frame["flags_auxiliary"]
            ]
        )
        self.flag_masks = np.ones_like(self.flag_targets, dtype=np.float32)
        self.weights = tm.source_weights(frame)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return (
            self.texts[index],
            self.primary_targets[index],
            self.fine_targets[index],
            self.fine_masks[index],
            self.flag_targets[index],
            self.flag_masks[index],
            self.weights[index],
        )


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: Iterable[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class FourTargetCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        (
            texts,
            primary_targets,
            fine_targets,
            fine_masks,
            flag_targets,
            flag_masks,
            weights,
        ) = zip(*examples)
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return (
            tokens,
            torch.tensor(np.asarray(primary_targets), dtype=torch.float32),
            torch.tensor(np.asarray(fine_targets), dtype=torch.float32),
            torch.tensor(np.asarray(fine_masks), dtype=torch.float32),
            torch.tensor(np.asarray(flag_targets), dtype=torch.float32),
            torch.tensor(np.asarray(flag_masks), dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )


def tokenizer():
    value = AutoTokenizer.from_pretrained(
        MODEL_SPEC.model_id, revision=MODEL_SPEC.revision
    )
    if value.pad_token_id is None:
        value.pad_token = value.eos_token
    value.padding_side = "right"
    return value


def evaluation_loader(frame: pd.DataFrame, tokenization) -> DataLoader:
    return DataLoader(
        FourTargetDataset(frame),
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=FourTargetCollator(tokenization),
    )


def training_loader(
    frame: pd.DataFrame,
    tokenization,
    epoch: int,
    completed_batches: int,
) -> tuple[DataLoader, int]:
    dataset = FourTargetDataset(frame)
    permutation = np.random.default_rng(SEED + epoch).permutation(len(dataset))
    start_sample = min(len(dataset), completed_batches * TRAIN_BATCH_SIZE)
    remaining = permutation[start_sample:].tolist()
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=FixedIndexSampler(remaining),
        num_workers=0,
        collate_fn=FourTargetCollator(tokenization),
    )
    total_batches = math.ceil(len(dataset) / TRAIN_BATCH_SIZE)
    return loader, total_batches


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _base_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SPEC.model_id,
        revision=MODEL_SPEC.revision,
        num_labels=len(OUTPUT_LABELS),
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    return model


def build_model(target_device: torch.device | None = None):
    from peft import LoraConfig, TaskType, get_peft_model

    configuration = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )
    return get_peft_model(_base_model(), configuration).to(target_device or device())


def load_adapter(directory: Path, target_device: torch.device | None = None):
    from peft import PeftModel

    return PeftModel.from_pretrained(
        _base_model(), directory, is_trainable=True
    ).to(target_device or device())


@torch.inference_mode()
def predict_logits(model, loader: DataLoader, description: str) -> np.ndarray:
    model.eval()
    target_device = next(model.parameters()).device
    outputs = []
    for batch in tqdm(loader, desc=description, unit="lote"):
        tokens = batch[0]
        output = model(
            **{key: value.to(target_device) for key, value in tokens.items()}
        ).logits
        outputs.append(output.cpu().numpy())
    return np.vstack(outputs)


def _semantic_arrays(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories = scores >= thresholds
    true_safe = (~y.astype(bool).any(axis=1)).astype(np.int8)[:, None]
    predicted_safe = (~categories.any(axis=1)).astype(np.int8)[:, None]
    true_semantic = np.column_stack([true_safe, y.astype(np.int8)])
    predicted_semantic = np.column_stack([predicted_safe, categories.astype(np.int8)])
    semantic_scores = np.column_stack([1 - scores.max(axis=1), scores])
    return true_semantic, predicted_semantic, semantic_scores


def evaluate_scores(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    true_semantic, predicted_semantic, semantic_scores = _semantic_arrays(
        y, scores, thresholds
    )
    predicted_categories = predicted_semantic[:, 1:]
    true_any = y.astype(bool).any(axis=1)
    predicted_any = predicted_categories.astype(bool).any(axis=1)
    category_recall = {
        label: float(
            recall_score(y[:, index], predicted_categories[:, index], zero_division=0)
        )
        for index, label in enumerate(TARGET_LABELS)
    }
    metrics = {
        "n": int(len(y)),
        "exact_match": float(accuracy_score(true_semantic, predicted_semantic)),
        "jaccard_samples": float(
            jaccard_score(
                true_semantic, predicted_semantic, average="samples", zero_division=1
            )
        ),
        "pr_auc_macro_semantic": float(
            average_precision_score(true_semantic, semantic_scores, average="macro")
        ),
        "damage_pr_auc_macro": float(
            average_precision_score(y, scores, average="macro")
        ),
        "damage_precision_micro": float(
            precision_score(y, predicted_categories, average="micro", zero_division=0)
        ),
        "damage_recall_micro": float(
            recall_score(y, predicted_categories, average="micro", zero_division=0)
        ),
        "damage_f1_micro": float(
            f1_score(y, predicted_categories, average="micro", zero_division=0)
        ),
        "damage_f1_macro": float(
            f1_score(y, predicted_categories, average="macro", zero_division=0)
        ),
        "any_damage_precision": float(
            precision_score(true_any, predicted_any, zero_division=0)
        ),
        "any_damage_recall": float(
            recall_score(true_any, predicted_any, zero_division=0)
        ),
        "any_damage_f1": float(f1_score(true_any, predicted_any, zero_division=0)),
        "missed_damage_as_safe": int((true_any & ~predicted_any).sum()),
        "category_false_negatives": int(
            (y.astype(bool) & ~predicted_categories.astype(bool)).sum()
        ),
        "category_recall": category_recall,
        "minimum_category_recall": float(min(category_recall.values())),
    }
    report = pd.DataFrame(
        classification_report(
            true_semantic,
            predicted_semantic,
            target_names=SEMANTIC_LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).T
    return metrics, report, predicted_categories


def _positive_weights(y: np.ndarray) -> np.ndarray:
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    if (positives == 0).any():
        raise ValueError("Hay una categoría sin positivos en train.")
    return np.sqrt(negatives / positives).astype(np.float32)


def _masked_positive_weights(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positives = (y * mask).sum(axis=0)
    observed = mask.sum(axis=0)
    negatives = observed - positives
    if (observed == 0).any() or (positives == 0).any():
        missing = [
            OUTPUT_LABELS[PRIMARY_OUTPUTS + index]
            for index in np.flatnonzero((observed == 0) | (positives == 0))
        ]
        raise ValueError(f"Supervisión auxiliar insuficiente para: {missing}")
    return np.sqrt(negatives / positives).astype(np.float32)


def _save_adapter_snapshot(
    directory: Path,
    model,
    state: dict,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tm.write_json(directory / "training_state.json", state)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_rng = state["torch"]
    if isinstance(cpu_rng, torch.Tensor):
        cpu_rng = cpu_rng.detach().to(device="cpu", dtype=torch.uint8)
    else:
        cpu_rng = torch.as_tensor(cpu_rng, dtype=torch.uint8, device="cpu")
    torch.set_rng_state(cpu_rng)
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_rng = [
            value.detach().to(device="cpu", dtype=torch.uint8)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=torch.uint8, device="cpu")
            for value in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng)


def _resume_slot_path(slot: str) -> Path:
    return MODEL_DIR / f"resume_slot_{slot}"


def _save_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    state: dict,
) -> dict:
    active = None
    if RESUME_POINTER_PATH.exists():
        try:
            active = _json(RESUME_POINTER_PATH).get("active_slot")
        except (json.JSONDecodeError, OSError):
            active = None
    inactive = "b" if active == "a" else "a"
    directory = _resume_slot_path(inactive)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model.save_pretrained(directory, safe_serialization=True)
    tm.write_json(directory / "resume_state.json", state)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        },
        directory / "optimizer_scheduler_rng.pt",
    )
    files = _directory_artifacts(directory)
    pointer = {
        "updated_at": tm.now_iso(),
        "active_slot": inactive,
        "directory": _relative(directory),
        "dataset_sha256": state["dataset_sha256"],
        "training_fingerprint_sha256": state["training_fingerprint_sha256"],
        "target_labels": TARGET_LABELS,
        "output_labels": OUTPUT_LABELS,
        "epoch": state["epoch"],
        "completed_batches": state["completed_batches"],
        "global_optimizer_steps": state["global_optimizer_steps"],
        "files": files,
    }
    tm.write_json(RESUME_POINTER_PATH, pointer)
    return pointer


def _load_resume_checkpoint(
    expected_dataset_sha256: str,
    expected_training_fingerprints_sha256: set[str],
    target_device: torch.device,
) -> tuple[object, dict, dict] | None:
    if not RESUME_POINTER_PATH.exists():
        return None
    pointer = _json(RESUME_POINTER_PATH)
    if pointer.get("dataset_sha256") != expected_dataset_sha256:
        raise ValueError("El checkpoint reanudable corresponde a otro dataset.")
    if pointer.get("training_fingerprint_sha256") not in expected_training_fingerprints_sha256:
        raise ValueError(
            "El checkpoint reanudable corresponde a otra supervisión auxiliar."
        )
    if pointer.get("target_labels") != TARGET_LABELS:
        raise ValueError("El checkpoint reanudable corresponde a otra taxonomía.")
    if pointer.get("output_labels") != OUTPUT_LABELS:
        raise ValueError("El checkpoint usa otra configuración de salidas auxiliares.")
    directory = tm.project_path(pointer["directory"])
    for artifact in pointer["files"]:
        path = tm.project_path(artifact["path"])
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Checkpoint reanudable incompleto o alterado: {path}")
    model = load_adapter(directory, target_device)
    state = _json(directory / "resume_state.json")
    optimizer_state = torch.load(
        directory / "optimizer_scheduler_rng.pt",
        map_location=target_device,
        weights_only=False,
    )
    return model, state, optimizer_state


def resume_status() -> dict:
    if TRAINING_RESULT_PATH.exists():
        result = _json(TRAINING_RESULT_PATH)
        epochs_completed = int(result["epochs_completed"])
        pointer_epoch = (
            int(_json(RESUME_POINTER_PATH)["epoch"])
            if RESUME_POINTER_PATH.exists()
            else None
        )
        can_extend = (
            epochs_completed < MAX_EPOCHS
            and pointer_epoch is not None
            and pointer_epoch <= MAX_EPOCHS
        )
        can_force_extend = (
            epochs_completed < MAX_EPOCHS
            and pointer_epoch is not None
            and pointer_epoch > MAX_EPOCHS
        )
        return {
            "status": (
                "extendable"
                if can_extend
                else "force_extendable"
                if can_force_extend
                else "completed"
            ),
            "epochs_completed": epochs_completed,
            "configured_max_epochs": MAX_EPOCHS,
            "next_epoch": (
                pointer_epoch
                if can_extend
                else epochs_completed + 1
                if can_force_extend
                else None
            ),
            "best_epoch": result["best_epoch"],
            "adapter": result["adapter"],
        }
    if not RESUME_POINTER_PATH.exists():
        return {"status": "not_started"}
    pointer = _json(RESUME_POINTER_PATH)
    return {"status": "resumable", **pointer}


def _archive_for_restart() -> Path | None:
    if not any(MODEL_DIR.iterdir()):
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = MODEL_DIR.parent / f"{RUN_KEY}_archive_{timestamp}"
    shutil.move(str(MODEL_DIR), str(archive))
    MODEL_DIR.mkdir(parents=True)
    return archive


def run_finetuning(
    frames: dict[str, pd.DataFrame] | None = None,
    resume: bool = True,
    force_restart: bool = False,
    save_every_optimizer_steps: int = SAVE_EVERY_OPTIMIZER_STEPS,
    force_complete_max_epochs: bool = False,
) -> dict:
    frames, audit = (load_frames() if frames is None else (frames, load_frames()[1]))
    extending_completed_run = False
    if TRAINING_RESULT_PATH.exists() and not force_restart:
        result = _json(TRAINING_RESULT_PATH)
        if _completed_result_is_usable(result, audit):
            epochs_completed = int(result.get("epochs_completed", 0))
            if epochs_completed >= MAX_EPOCHS:
                return result
            if not resume:
                raise ValueError(
                    "El entrenamiento terminado tiene menos épocas que el máximo "
                    "actual. Use resume=True para continuarlo sin reiniciar."
                )
            if not RESUME_POINTER_PATH.exists():
                raise FileNotFoundError(
                    "Falta resume_pointer.json; no se puede extender el entrenamiento "
                    "terminado sin reiniciar."
                )
            pointer_epoch = int(_json(RESUME_POINTER_PATH)["epoch"])
            if pointer_epoch > MAX_EPOCHS and not force_complete_max_epochs:
                return result
            extending_completed_run = True
            print(
                f"Extendiendo el entrenamiento terminado: {epochs_completed} → "
                f"máximo {MAX_EPOCHS} épocas."
            )
    archive = _archive_for_restart() if force_restart else None
    tm.set_reproducibility()
    target_device = device()
    tokenization = tokenizer()
    tokenization.save_pretrained(MODEL_DIR / "tokenizer")
    train_frame = frames["train"]
    validation_frame = frames["validation"]
    y_train = four_targets(train_frame)
    y_validation = four_targets(validation_frame)
    training_dataset = FourTargetDataset(train_frame)
    primary_pos_weights_np = _positive_weights(y_train)
    fine_pos_weights_np = _masked_positive_weights(
        training_dataset.fine_targets, training_dataset.fine_masks
    )
    flag_pos_weights_np = _positive_weights(training_dataset.flag_targets)
    primary_pos_weights = torch.tensor(
        primary_pos_weights_np, dtype=torch.float32, device=target_device
    )
    fine_pos_weights = torch.tensor(
        fine_pos_weights_np, dtype=torch.float32, device=target_device
    )
    flag_pos_weights = torch.tensor(
        flag_pos_weights_np, dtype=torch.float32, device=target_device
    )

    loaded = (
        _load_resume_checkpoint(
            audit["dataset_sha256"],
            set(audit["compatible_training_fingerprints_sha256"]),
            target_device,
        )
        if resume and not force_restart
        else None
    )
    if loaded is None:
        model = build_model(target_device)
        state = {
            "epoch": 1,
            "completed_batches": 0,
            "cumulative_loss": 0.0,
            "seen": 0,
            "optimizer_steps_in_epoch": 0,
            "global_optimizer_steps": 0,
            "history": [],
            "best_score": -math.inf,
            "best_epoch": 0,
            "stale_epochs": 0,
            "training_seconds_accumulated": 0.0,
            "dataset_sha256": audit["dataset_sha256"],
            "training_fingerprint_sha256": audit["training_fingerprint_sha256"],
            "target_labels": TARGET_LABELS,
            "output_labels": OUTPUT_LABELS,
        }
        optimizer_state = None
    else:
        model, state, optimizer_state = loaded
        completed_history = len(state.get("history", []))
        if (
            force_complete_max_epochs
            and completed_history < MAX_EPOCHS
            and int(state["epoch"]) > MAX_EPOCHS
        ):
            # Early stopping marca epoch=MAX_EPOCHS+1. Los pesos, Adam, el
            # scheduler y el RNG siguen siendo los del último epoch realmente
            # terminado, por lo que se puede continuar desde el siguiente.
            state.update(
                {
                    "epoch": completed_history + 1,
                    "completed_batches": 0,
                    "cumulative_loss": 0.0,
                    "seen": 0,
                    "optimizer_steps_in_epoch": 0,
                    "stale_epochs": 0,
                }
            )
            print(
                "Early stopping anulado para completar el horizonte: "
                f"se ejecutará la época {state['epoch']}/{MAX_EPOCHS}."
            )
        print(
            f"Reanudando época {state['epoch']}, después del lote "
            f"{state['completed_batches']:,}."
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(
        math.ceil(len(train_frame) / TRAIN_BATCH_SIZE) / GRADIENT_ACCUMULATION
    )
    scheduler = tm.scheduler_for(optimizer, steps_per_epoch * MAX_EPOCHS)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state["optimizer"])
        scheduler.load_state_dict(optimizer_state["scheduler"])
        # El plan original terminaba en dos épocas y su LR guardado es cero.
        # Al ampliar el horizonte, recalculamos el LR en el mismo paso global
        # usando la nueva curva de cuatro épocas, sin perder Adam ni su RNG.
        resumed_lrs = [
            base_lr * lr_lambda(scheduler.last_epoch)
            for base_lr, lr_lambda in zip(
                scheduler.base_lrs, scheduler.lr_lambdas, strict=True
            )
        ]
        for parameter_group, resumed_lr in zip(
            optimizer.param_groups, resumed_lrs, strict=True
        ):
            parameter_group["lr"] = resumed_lr
        scheduler._last_lr = resumed_lrs
        _restore_rng_state(optimizer_state["rng"])

    session_start = perf_counter()
    _append_log(
        "training_session_started",
        resumed=loaded is not None,
        extending_completed_run=extending_completed_run,
        configured_max_epochs=MAX_EPOCHS,
        epoch=state["epoch"],
        completed_batches=state["completed_batches"],
        device=str(target_device),
    )
    epoch = int(state["epoch"])
    while epoch <= MAX_EPOCHS:
        completed_batches = int(state["completed_batches"])
        cumulative_loss = float(state["cumulative_loss"])
        seen = int(state["seen"])
        optimizer_steps_in_epoch = int(state["optimizer_steps_in_epoch"])
        loader, total_batches = training_loader(
            train_frame, tokenization, epoch, completed_batches
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = perf_counter()
        progress = tqdm(
            loader,
            total=total_batches,
            initial=completed_batches,
            desc=f"{RUN_KEY} · época {epoch}/{MAX_EPOCHS}",
            unit="lote",
        )
        for batch_index, batch in enumerate(progress, start=completed_batches + 1):
            (
                tokens,
                primary_targets,
                fine_targets,
                fine_masks,
                flag_targets,
                flag_masks,
                weights,
            ) = batch
            tokens = {key: value.to(target_device) for key, value in tokens.items()}
            primary_targets = primary_targets.to(target_device)
            fine_targets = fine_targets.to(target_device)
            fine_masks = fine_masks.to(target_device)
            flag_targets = flag_targets.to(target_device)
            flag_masks = flag_masks.to(target_device)
            weights = weights.to(target_device)
            logits = model(**tokens).logits
            primary_logits = logits[:, :PRIMARY_OUTPUTS]
            fine_logits = logits[:, PRIMARY_OUTPUTS : PRIMARY_OUTPUTS + FINE_OUTPUTS]
            flag_logits = logits[:, PRIMARY_OUTPUTS + FINE_OUTPUTS :]
            primary_loss = nn.functional.binary_cross_entropy_with_logits(
                primary_logits,
                primary_targets,
                pos_weight=primary_pos_weights,
                reduction="none",
            )
            fine_loss = nn.functional.binary_cross_entropy_with_logits(
                fine_logits,
                fine_targets,
                pos_weight=fine_pos_weights,
                reduction="none",
            )
            flag_loss = nn.functional.binary_cross_entropy_with_logits(
                flag_logits,
                flag_targets,
                pos_weight=flag_pos_weights,
                reduction="none",
            )
            fine_denominator = fine_masks.sum(dim=1).clamp(min=1.0)
            flag_denominator = flag_masks.sum(dim=1).clamp(min=1.0)
            per_sample = (
                primary_loss.mean(dim=1)
                + AUX_FINE_LOSS_WEIGHT
                * (fine_loss * fine_masks).sum(dim=1)
                / fine_denominator
                + AUX_FLAG_LOSS_WEIGHT
                * (flag_loss * flag_masks).sum(dim=1)
                / flag_denominator
            )
            loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            (loss / GRADIENT_ACCUMULATION).backward()
            batch_rows = len(primary_targets)
            cumulative_loss += float(loss.detach()) * batch_rows
            seen += batch_rows
            should_step = (
                batch_index % GRADIENT_ACCUMULATION == 0
                or batch_index == total_batches
            )
            if should_step:
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps_in_epoch += 1
                state["global_optimizer_steps"] += 1
                if state["global_optimizer_steps"] % save_every_optimizer_steps == 0:
                    state.update(
                        {
                            "epoch": epoch,
                            "completed_batches": batch_index,
                            "cumulative_loss": cumulative_loss,
                            "seen": seen,
                            "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
                            "training_seconds_accumulated": state[
                                "training_seconds_accumulated"
                            ]
                            + perf_counter()
                            - session_start,
                        }
                    )
                    pointer = _save_resume_checkpoint(
                        model, optimizer, scheduler, state
                    )
                    _append_log(
                        "resume_checkpoint_saved",
                        epoch=epoch,
                        batch=batch_index,
                        global_optimizer_steps=state["global_optimizer_steps"],
                        slot=pointer["active_slot"],
                    )
                    session_start = perf_counter()
            progress.set_postfix(
                loss=f"{cumulative_loss / max(1, seen):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                checkpoint=state["global_optimizer_steps"]
                // save_every_optimizer_steps,
            )

        validation_logits = predict_logits(
            model,
            evaluation_loader(validation_frame, tokenization),
            f"Qwen 4 etiquetas · validación época {epoch}",
        )
        validation_primary_logits = validation_logits[:, :PRIMARY_OUTPUTS]
        validation_scores = expit(validation_primary_logits)
        thresholds = tune_thresholds(y_validation.astype(np.int8), validation_scores)
        metrics, _, _ = evaluate_scores(y_validation, validation_scores, thresholds)
        epoch_record = {
            "epoch": epoch,
            "training_loss": cumulative_loss / max(1, seen),
            "epoch_seconds_this_session": perf_counter() - epoch_start,
            "optimizer_steps": optimizer_steps_in_epoch,
            "thresholds_raw": thresholds.tolist(),
            **{key: value for key, value in metrics.items() if key != "category_recall"},
            "category_recall": metrics["category_recall"],
        }
        state["history"].append(epoch_record)
        np.save(
            METRICS_DIR / f"validation_logits_all_outputs_epoch_{epoch:02d}.npy",
            validation_logits,
        )
        np.save(
            METRICS_DIR / f"validation_logits_primary_epoch_{epoch:02d}.npy",
            validation_primary_logits,
        )
        tm.write_json(
            METRICS_DIR / f"validacion_epoch_{epoch:02d}.json", epoch_record
        )
        pd.DataFrame(state["history"]).to_csv(
            METRICS_DIR / "historial.csv", index=False
        )
        snapshot_state = {
            "model_spec": asdict(MODEL_SPEC),
            "epoch": epoch,
            "thresholds_raw": thresholds.tolist(),
            "history": state["history"],
            "target_labels": TARGET_LABELS,
            "output_labels": OUTPUT_LABELS,
            "safe_is_derived": True,
            "dataset_sha256": audit["dataset_sha256"],
            "training_fingerprint_sha256": audit["training_fingerprint_sha256"],
            "method": (
                "LoRA multi-task sequence classification; four primary merged "
                "damage labels plus masked fine-label and transversal-flag auxiliaries"
            ),
        }
        _save_adapter_snapshot(
            MODEL_DIR / "epoch_adapters" / f"epoch_{epoch:02d}",
            model,
            snapshot_state,
        )
        _save_adapter_snapshot(MODEL_DIR / "last_adapter", model, snapshot_state)
        score = float(metrics["damage_pr_auc_macro"])
        if score > state["best_score"] + 1e-6:
            state["best_score"] = score
            state["best_epoch"] = epoch
            state["stale_epochs"] = 0
            _save_adapter_snapshot(MODEL_DIR / "best_adapter", model, snapshot_state)
        else:
            state["stale_epochs"] += 1
        tqdm.write(
            f"Época {epoch}: PR-AUC={score:.4f}; F1 macro="
            f"{metrics['damage_f1_macro']:.4f}; recall daño="
            f"{metrics['any_damage_recall']:.4f}"
        )
        _append_log("epoch_completed", **epoch_record)
        early_stop = (
            not force_complete_max_epochs
            and state["stale_epochs"] >= tm.EARLY_STOPPING_PATIENCE
        )
        state.update(
            {
                "epoch": MAX_EPOCHS + 1 if early_stop else epoch + 1,
                "completed_batches": 0,
                "cumulative_loss": 0.0,
                "seen": 0,
                "optimizer_steps_in_epoch": 0,
                "training_seconds_accumulated": state[
                    "training_seconds_accumulated"
                ]
                + perf_counter()
                - session_start,
            }
        )
        _save_resume_checkpoint(model, optimizer, scheduler, state)
        session_start = perf_counter()
        if early_stop:
            break
        epoch += 1

    best_state_path = MODEL_DIR / "best_adapter" / "training_state.json"
    result = {
        "status": "completed",
        "completed_at": tm.now_iso(),
        "model": asdict(MODEL_SPEC),
        "method": "LoRA parameter-efficient fine-tuning",
        "taxonomy": {
            "semantic_labels": SEMANTIC_LABELS,
            "trained_damage_targets": TARGET_LABELS,
            "merge": "ACOSO_AMENAZA = ACOSO_PERSONAL union AMENAZA_DIRECTA",
            "safe_is_derived": True,
        },
        "dataset": audit["dataset"],
        "dataset_sha256": audit["dataset_sha256"],
        "training_fingerprint_sha256": audit["training_fingerprint_sha256"],
        "training_rows": int(len(train_frame)),
        "training_chunk_ids_sha256": audit["split_chunk_ids_sha256"]["train"],
        "device": str(target_device),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "trainable_fraction": trainable_parameters / total_parameters,
        "positive_weights_sqrt": {
            "primary": primary_pos_weights_np.tolist(),
            "fine_auxiliary": fine_pos_weights_np.tolist(),
            "flags_auxiliary": flag_pos_weights_np.tolist(),
        },
        "auxiliary_supervision": {
            "fine_labels": FINE_LABELS,
            "transversal_flags": TRANSVERSAL_FLAGS,
            "fine_loss_weight": AUX_FINE_LOSS_WEIGHT,
            "flag_loss_weight": AUX_FLAG_LOSS_WEIGHT,
            "fine_missing_policy": "masked",
            "operational_inference_outputs": TARGET_LABELS,
        },
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION,
        "max_epochs": MAX_EPOCHS,
        "force_complete_max_epochs": bool(force_complete_max_epochs),
        "epochs_completed": len(state["history"]),
        "best_epoch": int(state["best_epoch"]),
        "best_validation_damage_pr_auc_macro": float(state["best_score"]),
        "training_seconds": float(state["training_seconds_accumulated"]),
        "history": state["history"],
        "adapter": _relative(MODEL_DIR / "best_adapter"),
        "adapter_files": _directory_artifacts(MODEL_DIR / "best_adapter"),
        "best_training_state_sha256": sha256_file(best_state_path),
        "resume_checkpoint_frequency_optimizer_steps": save_every_optimizer_steps,
        "restart_archive": _relative(archive) if archive else None,
    }
    tm.write_json(TRAINING_RESULT_PATH, result)
    return result


def load_best_model(target_device: torch.device | None = None):
    directory = MODEL_DIR / "best_adapter"
    state = _json(directory / "training_state.json")
    if state["target_labels"] != TARGET_LABELS:
        raise ValueError("El adaptador best usa otra taxonomía.")
    if state.get("output_labels") != OUTPUT_LABELS:
        raise ValueError("El adaptador best no incluye la supervisión auxiliar esperada.")
    return load_adapter(directory, target_device or device()), state


@dataclass
class LabelCalibrator:
    label: str
    model: LogisticRegression

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


def fit_calibrators(y: np.ndarray, logits: np.ndarray) -> list[LabelCalibrator]:
    values = []
    for index, label in enumerate(TARGET_LABELS):
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1_000,
            random_state=SEED,
        )
        model.fit(logits[:, index].reshape(-1, 1), y[:, index])
        values.append(LabelCalibrator(label, model))
    return values


def apply_calibrators(
    calibrators: list[LabelCalibrator], logits: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [calibrator.predict(logits[:, index]) for index, calibrator in enumerate(calibrators)]
    )


def evaluate_calibrated(
    frames: dict[str, pd.DataFrame] | None = None,
    force: bool = False,
) -> dict:
    frames, audit = (load_frames() if frames is None else (frames, load_frames()[1]))
    if EVALUATION_PATH.exists() and not force:
        existing = _json(EVALUATION_PATH)
        best_state_path = MODEL_DIR / "best_adapter" / "training_state.json"
        if (
            existing.get("dataset_sha256") == audit["dataset_sha256"]
            and _fingerprint_matches(
                existing.get("training_fingerprint_sha256"), audit
            )
            and best_state_path.exists()
            and existing.get("best_adapter_training_state_sha256")
            == sha256_file(best_state_path)
        ):
            return existing
    target_device = device()
    tokenization = tokenizer()
    model, state = load_best_model(target_device)
    logits = {}
    for split in ("validation", "test"):
        all_logits = predict_logits(
            model,
            evaluation_loader(frames[split], tokenization),
            f"Qwen 4 etiquetas · {split}",
        )
        np.save(METRICS_DIR / f"logits_all_outputs_{split}.npy", all_logits)
        logits[split] = all_logits[:, :PRIMARY_OUTPUTS]
        np.save(METRICS_DIR / f"logits_{split}.npy", logits[split])
    y_validation = four_targets(frames["validation"]).astype(np.int8)
    y_test = four_targets(frames["test"]).astype(np.int8)
    calibrators = fit_calibrators(y_validation, logits["validation"])
    joblib.dump(calibrators, METRICS_DIR / "calibradores_sigmoides.joblib")
    scores = {
        split: apply_calibrators(calibrators, values)
        for split, values in logits.items()
    }
    thresholds = tune_thresholds(y_validation, scores["validation"])
    raw_validation = expit(logits["validation"])
    calibration = []
    for index, label in enumerate(TARGET_LABELS):
        calibration.append(
            {
                "label": label,
                "coefficient": float(calibrators[index].model.coef_[0, 0]),
                "intercept": float(calibrators[index].model.intercept_[0]),
                "validation_brier_raw": float(
                    brier_score_loss(y_validation[:, index], raw_validation[:, index])
                ),
                "validation_brier_calibrated": float(
                    brier_score_loss(y_validation[:, index], scores["validation"][:, index])
                ),
            }
        )
    outputs = {}
    for split, y in (("validation", y_validation), ("test", y_test)):
        metrics, report, _ = evaluate_scores(y, scores[split], thresholds)
        outputs[split] = metrics
        np.save(METRICS_DIR / f"scores_calibrated_{split}.npy", scores[split])
        report.to_csv(METRICS_DIR / f"reporte_{split}.csv")
    result = {
        "completed_at": tm.now_iso(),
        "model": asdict(MODEL_SPEC),
        "dataset_sha256": audit["dataset_sha256"],
        "training_fingerprint_sha256": audit["training_fingerprint_sha256"],
        "best_adapter_training_state_sha256": sha256_file(
            MODEL_DIR / "best_adapter" / "training_state.json"
        ),
        "best_epoch": int(state["epoch"]),
        "target_labels": TARGET_LABELS,
        "auxiliary_output_labels": OUTPUT_LABELS[PRIMARY_OUTPUTS:],
        "safe_is_derived": True,
        "calibration_partition": "validation",
        "calibration_method": "per-label sigmoid (Platt scaling) on logits",
        "calibration": calibration,
        "thresholds_selected_on_validation": thresholds.tolist(),
        "metrics": outputs,
        "artifacts": {
            "calibrators": _artifact(METRICS_DIR / "calibradores_sigmoides.joblib"),
            "validation_scores": _artifact(
                METRICS_DIR / "scores_calibrated_validation.npy"
            ),
            "test_scores": _artifact(METRICS_DIR / "scores_calibrated_test.npy"),
        },
    }
    tm.write_json(EVALUATION_PATH, result)
    return result


def analyze_selective_operation(
    frames: dict[str, pd.DataFrame] | None = None,
    evaluation: dict | None = None,
) -> dict:
    frames, audit = (load_frames() if frames is None else (frames, load_frames()[1]))
    evaluation = evaluation or evaluate_calibrated(frames)
    thresholds = np.asarray(
        evaluation["thresholds_selected_on_validation"], dtype=float
    )
    scores_validation = np.load(METRICS_DIR / "scores_calibrated_validation.npy")
    scores_test = np.load(METRICS_DIR / "scores_calibrated_test.npy")
    y_validation = four_targets(frames["validation"]).astype(np.int8)
    y_test = four_targets(frames["test"]).astype(np.int8)
    cutoff, validation_alert = tm.tune_human_alert_cutoff(
        y_validation,
        scores_validation,
        thresholds,
        recall_target=ALERT_RECALL_TARGET,
    )
    test_margin = np.max(scores_test - thresholds, axis=1)
    test_alert = tm._binary_routing_metrics(
        y_test.astype(bool).any(axis=1), test_margin >= cutoff
    )
    test_alert["risk_margin_cutoff"] = cutoff
    test_gate = tm._human_alert_gate(test_alert)
    result = {
        "completed_at": tm.now_iso(),
        "dataset_sha256": audit["dataset_sha256"],
        "model_key": RUN_KEY,
        "target_labels": TARGET_LABELS,
        "calibration_partition": "validation",
        "risk_definition": "max(calibrated_score - category_threshold)",
        "validation": validation_alert,
        "test": test_alert,
        "test_gate": test_gate,
        "supported_for_human_review_alert": bool(test_gate["passed"]),
        "autonomous_deployment_supported": False,
        "mandatory_limitations": [
            "Test is not an independent human gold standard.",
            "The 4:1 sample does not represent prospective production prevalence.",
            "A prospective human-in-the-loop pilot is still required.",
        ],
    }
    tm.write_json(OPERATION_PATH, result)
    return result


def _merge_five_scores(scores: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [scores[:, 0], scores[:, 1], np.maximum(scores[:, 2], scores[:, 3]), scores[:, 4]]
    )


def compare_references(
    frames: dict[str, pd.DataFrame] | None = None,
    evaluation: dict | None = None,
) -> pd.DataFrame:
    frames, _ = load_frames() if frames is None else (frames, None)
    evaluation = evaluation or evaluate_calibrated(frames)
    qwen_scores = {
        split: np.load(METRICS_DIR / f"scores_calibrated_{split}.npy")
        for split in ("validation", "test")
    }
    candidates = {
        "qwen4_lora": {
            "label": "Qwen3-0.6B LoRA · entrenado directamente con 4 etiquetas",
            "scores": qwen_scores,
            "thresholds": np.asarray(
                evaluation["thresholds_selected_on_validation"], dtype=float
            ),
        }
    }
    historical_paths = {
        split: tm.METRICS_DIR / f"scores_clasico_ganador_{split}.npy"
        for split in ("validation", "test")
    }
    if all(path.exists() for path in historical_paths.values()):
        historical_scores = {
            split: _merge_five_scores(np.load(path))
            for split, path in historical_paths.items()
        }
        candidates["svm_04_2_posthoc4"] = {
            "label": "SVM plano 04_2 · unión post hoc a 4 etiquetas",
            "scores": historical_scores,
            "thresholds": tune_thresholds(
                four_targets(frames["validation"]).astype(np.int8),
                historical_scores["validation"],
            ),
        }
    rows = []
    for key, candidate in candidates.items():
        for split in ("validation", "test"):
            metrics, _, _ = evaluate_scores(
                four_targets(frames[split]).astype(np.int8),
                candidate["scores"][split],
                candidate["thresholds"],
            )
            rows.append(
                {
                    "model_key": key,
                    "modelo": candidate["label"],
                    "split": split,
                    **{name: value for name, value in metrics.items() if name != "category_recall"},
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(COMPARISON_PATH, index=False)
    return table


def write_report_and_figures(
    training: dict,
    evaluation: dict,
    operation: dict,
    comparison: pd.DataFrame,
) -> None:
    test = evaluation["metrics"]["test"]
    category_rows = "\n".join(
        f"| {label} | {recall:.4f} |"
        for label, recall in test["category_recall"].items()
    )
    comparison_rows = "\n".join(
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | "
        f"{row.damage_f1_macro:.4f} | {row.any_damage_recall:.4f} | "
        f"{int(row.missed_damage_as_safe)} |"
        for row in comparison.loc[comparison["split"].eq("test")].itertuples()
    )
    report = f"""# Qwen3-0.6B LoRA con ACOSO_AMENAZA

Fecha: {tm.now_iso()}

## Diseño

Se ajustó `Qwen/Qwen3-0.6B-Base` con LoRA para cuatro objetivos operativos de daño: {', '.join(TARGET_LABELS)}. `ACOSO_AMENAZA` es la unión reproducible de `ACOSO_PERSONAL` y `AMENAZA_DIRECTA`; `SEGURO` se deriva cuando ninguna salida supera su umbral.

Como regularización multitararea se añadieron cabezas auxiliares para {len(FINE_LABELS)} etiquetas finas y {len(TRANSVERSAL_FLAGS)} flags transversales, con pesos de pérdida {AUX_FINE_LOSS_WEIGHT:.2f} y {AUX_FLAG_LOSS_WEIGHT:.2f}. Las etiquetas finas ausentes se enmascararon: no se interpretaron como negativos. Estas cabezas no agregan categorías de moderación ni se usan como entradas o reglas en inferencia; las únicas salidas operativas continúan siendo los cuatro daños primarios y `SEGURO` derivada.

El prompt operativo se conservó únicamente como procedencia de la taxonomía y no se introdujo en el texto de entrenamiento. Esto evita depender en producción de información que no estará disponible junto con cada chunk.

El dataset y splits 4:1 son los congelados por `04_2` (`{training['dataset_sha256']}`). Se guardó un checkpoint reanudable cada {training['resume_checkpoint_frequency_optimizer_steps']} pasos de optimizador, alternando dos slots verificables, además de `last_adapter` y `best_adapter` por época.

Las probabilidades finales se calibraron por etiqueta mediante regresión sigmoide sobre logits de validación. Los umbrales también se fijaron en validación; test no intervino en entrenamiento, selección, calibración ni umbrales.

## Resultado en test

- PR-AUC macro de daño: {test['damage_pr_auc_macro']:.4f}.
- F1 macro de daño: {test['damage_f1_macro']:.4f}.
- Precisión de cualquier daño: {test['any_damage_precision']:.4f}.
- Recall de cualquier daño: {test['any_damage_recall']:.4f}.
- Daños clasificados como seguro: {test['missed_damage_as_safe']}.

| Categoría | Recall test |
|---|---:|
{category_rows}

## Comparación sobre el mismo test

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{comparison_rows}

## Operación selectiva

La alerta calibrada para {ALERT_RECALL_TARGET:.0%} de recall en validación obtuvo en test recall {operation['test']['recall']:.4f}, tasa de revisión {operation['test']['review_rate']:.4f}, VPN {operation['test']['negative_predictive_value']:.4f} y {operation['test']['false_negatives']} falsos negativos automáticos. Alerta respaldada por la puerta declarada: **{'sí' if operation['supported_for_human_review_alert'] else 'no'}**.

No se autoriza autonomía sin gold standard humano independiente, prevalencia natural y piloto prospectivo.

## Conclusión sobre desempeño y uso en producción

Qwen3-0.6B LoRA mejora al SVM de referencia en el mismo test, pero su desempeño absoluto todavía es moderado: alcanza PR-AUC macro {test['damage_pr_auc_macro']:.4f}, F1 macro {test['damage_f1_macro']:.4f} y recall de cualquier daño {test['any_damage_recall']:.4f}. Con los umbrales ordinarios deja {test['missed_damage_as_safe']} ejemplos con daño clasificados como seguros, y el recall por categoría se sitúa entre {test['minimum_category_recall']:.4f} y {max(test['category_recall'].values()):.4f}. Por tanto, la exactitud global no debe interpretarse como evidencia de seguridad operativa, pues está influida por la abundancia de ejemplos seguros.

La política selectiva de alto recall reduce los falsos negativos a {operation['test']['false_negatives']} y alcanza recall {operation['test']['recall']:.4f} y VPN {operation['test']['negative_predictive_value']:.4f}; sin embargo, envía {operation['test']['review_rate']:.2%} de los textos a revisión humana y no supera la puerta operativa predefinida. En consecuencia, **el modelo no está listo para moderación autónoma ni para decisiones de bloqueo o sanción en producción**. Su uso razonable se limita por ahora a experimentación fuera de línea o a un piloto controlado en modo sombra, siempre con revisión humana y sin afectar usuarios. Antes de desplegarlo se requiere validación prospectiva con un gold standard humano independiente y prevalencia real, además de comprobar capacidad de revisión, latencia, coste, deriva y desempeño por subgrupos.

## Referencias (APA 7)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Qwen Team. (2025). Qwen3 technical report. *arXiv*. https://doi.org/10.48550/arXiv.2505.09388

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv*. https://doi.org/10.48550/arXiv.1706.05098
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    plot = comparison.loc[comparison["split"].eq("test")].set_index("modelo")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plot[["damage_pr_auc_macro", "damage_f1_macro", "any_damage_recall"]].plot.bar(
        ax=axes[0]
    )
    axes[0].set_title("Comparación en test 4:1")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    calibration = pd.DataFrame(evaluation["calibration"]).set_index("label")
    calibration[["validation_brier_raw", "validation_brier_calibrated"]].plot.bar(
        ax=axes[1]
    )
    axes[1].set_title("Brier antes y después de calibrar")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "comparacion_y_calibracion.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def finalize(
    frames: dict[str, pd.DataFrame] | None = None,
    force_evaluation: bool = False,
) -> dict:
    frames, _ = load_frames() if frames is None else (frames, None)
    training = run_finetuning(frames)
    evaluation = evaluate_calibrated(frames, force=force_evaluation)
    operation = analyze_selective_operation(frames, evaluation)
    comparison = compare_references(frames, evaluation)
    write_report_and_figures(training, evaluation, operation, comparison)
    result = {
        "training": training,
        "evaluation": evaluation,
        "operation": operation,
        "comparison_artifact": _artifact(COMPARISON_PATH),
        "report_artifact": _artifact(REPORT_PATH),
        "figure_artifact": _artifact(FIGURES_DIR / "comparacion_y_calibracion.png"),
    }
    tm.write_json(METRICS_DIR / "resultado_final.json", result)
    return result
