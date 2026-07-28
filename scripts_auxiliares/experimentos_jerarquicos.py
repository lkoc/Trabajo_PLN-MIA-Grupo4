"""Experimentos jerárquicos comparables con el moderador plano de 04_2.

Este módulo implementa dos diseños sin alterar las particiones ya congeladas:

1. cascada de un detector binario de daño y un clasificador multietiqueta;
2. Transformer jerárquico conjunto con una cabeza binaria y cinco cabezas de daño.

La referencia plana se elige exclusivamente por PR-AUC macro de validación
entre los Transformers registrados por 04_2. Los umbrales también se fijan con
validación. El test se abre una sola vez para la comparación final pareada.
Las etiquetas finas y los flags transversales no son objetivos ni predictores.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Iterable
import hashlib
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl, sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import DAMAGE_ORDER, tune_thresholds


ROOT = tm.ROOT
SEED = tm.SEED
MAX_LENGTH = tm.MAX_LENGTH
TRAIN_BATCH_SIZE = tm.TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = tm.EVAL_BATCH_SIZE
MAX_EPOCHS = tm.MAX_EPOCHS
LEARNING_RATE = tm.LEARNING_RATE
WEIGHT_DECAY = tm.WEIGHT_DECAY
EARLY_STOPPING_PATIENCE = tm.EARLY_STOPPING_PATIENCE
BOOTSTRAP_REPLICATES = tm.BOOTSTRAP_REPLICATES

GATE_VALIDATION_RECALL_TARGET = 0.97
AUTO_DAMAGE_PRECISION_TARGET = 0.90
CASCADE_SAFE_TO_DAMAGE_STAGE2_RATIO = 1.0
JOINT_BINARY_LOSS_WEIGHT = 0.50
JOINT_CATEGORY_LOSS_WEIGHT = 1.00
JOINT_CONSISTENCY_LOSS_WEIGHT = 0.10

CASCADE_KEY = "cascada_binaria_multietiqueta"
CASCADE_EXTRA_SAFE_KEY = "cascada_binaria_multietiqueta_seguros_ampliados"
JOINT_KEY = "transformer_jerarquico_multitarea"
EXPERIMENT_LABELS = {
    CASCADE_KEY: "Cascada binaria → multietiqueta",
    CASCADE_EXTRA_SAFE_KEY: "Cascada con negativos SEGURO ampliados (ablación)",
    JOINT_KEY: "Transformer jerárquico multitarea",
}

METRICS_ROOT = ROOT / "resultados" / "metricas" / "experimentos_jerarquicos"
FIGURES_ROOT = ROOT / "resultados" / "figuras" / "experimentos_jerarquicos"
MODEL_ROOT = ROOT / "modelos" / "experimentos_jerarquicos"
REPORT_ROOT = ROOT / "resultados"
for _directory in (METRICS_ROOT, FIGURES_ROOT, MODEL_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)


def experiment_targets(frame: pd.DataFrame) -> np.ndarray:
    """Objetivos del experimento; el modo histórico conserva cinco daños."""
    return tm.damage_targets(frame)


def evaluate_experiment_scores(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Evaluación inyectable para variantes con otra taxonomía de salida."""
    return tm.evaluate_damage_scores(y, scores, thresholds)


def initialize_experiment_model(
    model: nn.Module,
    role: str,
    spec: tm.EncoderSpec,
    reference_key: str,
    dataset_sha256: str,
) -> dict:
    """Hook de inicialización; las variantes pueden reutilizar pesos previos."""
    return {
        "enabled": False,
        "role": role,
        "strategy": "hub_pretrained_encoder_and_random_task_head",
        "reference_key": reference_key,
    }


def joint_training_frames(context: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    """Hook para ampliar sólo la supervisión binaria en variantes multitarea."""
    frames = {split: frame.copy() for split, frame in context["frames"].items()}
    frames["train"]["category_loss_mask"] = 1.0
    return frames, {
        "purpose": "strict_architecture_comparison",
        "binary_training_rows": int(len(frames["train"])),
        "category_training_rows": int(len(frames["train"])),
        "additional_safe_binary_only_rows": 0,
        "validation_or_test_videos_used": False,
    }


def _experiment_paths(key: str) -> dict[str, Path]:
    return {
        "metrics": METRICS_ROOT / key,
        "figures": FIGURES_ROOT / key,
        "models": MODEL_ROOT / key,
        "result": METRICS_ROOT / key / "resultado.json",
        "comparison": METRICS_ROOT / key / "comparacion_modelo_plano.csv",
        "categories": METRICS_ROOT / key / "comparacion_recall_por_categoria.csv",
        "bootstrap": METRICS_ROOT / key / "bootstrap_pareado_por_video.csv",
        "report": REPORT_ROOT
        / (
            "INFORME_EXPERIMENTO_CASCADA_JERARQUICA.md"
            if key == CASCADE_KEY
            else (
                "INFORME_ABLACION_CASCADA_SEGUROS_AMPLIADOS.md"
                if key == CASCADE_EXTRA_SAFE_KEY
                else "INFORME_EXPERIMENTO_TRANSFORMER_JERARQUICO.md"
            )
        ),
    }


def _ensure_directories(paths: dict[str, Path]) -> None:
    for name in ("metrics", "figures", "models"):
        paths[name].mkdir(parents=True, exist_ok=True)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return tm.project_relative(path)


def _ids_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _artifact(path: Path) -> dict:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _metric_from_evaluation(evaluation: dict, split: str, metric: str) -> float:
    try:
        return float(evaluation["metrics"][split][metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"La evaluación plana no contiene metrics/{split}/{metric}."
        ) from exc


def _select_flat_reference(registry: dict, frames: dict[str, pd.DataFrame]) -> dict:
    """Elige referencia plana con validación, nunca con el test."""
    candidates = []
    for item in registry["models"]:
        if item.get("family") != "transformer_full_finetuning":
            continue
        key = item["model_key"]
        if key not in tm.MODEL_SPECS:
            continue
        evaluation_path = ROOT / item["evaluation"]["path"]
        evaluation = _json(evaluation_path)
        validation_score_path = tm.METRICS_DIR / f"scores_{key}_validation.npy"
        test_score_path = tm.METRICS_DIR / f"scores_{key}_test.npy"
        missing = [
            path for path in (validation_score_path, test_score_path) if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Faltan probabilidades del modelo plano: "
                + ", ".join(str(path) for path in missing)
                + ". Ejecute la evaluación y comparación final de 04_2."
            )
        validation_scores = np.load(validation_score_path)
        test_scores = np.load(test_score_path)
        if validation_scores.shape != (len(frames["validation"]), len(DAMAGE_ORDER)):
            raise ValueError(f"Dimensión incompatible en {validation_score_path}.")
        if test_scores.shape != (len(frames["test"]), len(DAMAGE_ORDER)):
            raise ValueError(f"Dimensión incompatible en {test_score_path}.")
        thresholds = np.asarray(evaluation["thresholds"], dtype=float)
        if thresholds.shape != (len(DAMAGE_ORDER),):
            raise ValueError(f"Umbrales incompatibles para {key}.")
        candidates.append(
            {
                "model_key": key,
                "model_label": item["model_label"],
                "family": item["family"],
                "spec": tm.MODEL_SPECS[key],
                "evaluation": evaluation,
                "evaluation_artifact": _artifact(evaluation_path),
                "validation_score_artifact": _artifact(validation_score_path),
                "test_score_artifact": _artifact(test_score_path),
                "validation_scores": validation_scores,
                "test_scores": test_scores,
                "thresholds": thresholds,
                "validation_damage_pr_auc_macro": _metric_from_evaluation(
                    evaluation, "validation", "damage_pr_auc_macro"
                ),
            }
        )
    if not candidates:
        raise RuntimeError(
            "04_2 todavía no registró ningún Transformer plano evaluado. "
            "Ejecute sus celdas de evaluación y comparación final."
        )
    candidates.sort(
        key=lambda row: (row["validation_damage_pr_auc_macro"], row["model_key"]),
        reverse=True,
    )
    return candidates[0]


def load_frozen_context() -> dict:
    """Carga y audita el dataset/splits registrados por 04_2."""
    registry = tm.load_model_registry()
    dataset_path = ROOT / registry["dataset"]
    manifest_path = ROOT / registry["split_manifest"]
    if registry["dataset_sha256"] != sha256_file(dataset_path):
        raise ValueError("El hash del dataset no coincide con el registro de 04_2.")
    if registry["split_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("El hash del manifiesto no coincide con el registro de 04_2.")
    frame = pd.DataFrame(read_jsonl(dataset_path))
    required = {"chunk_id", "video_id", "text", "coarse_labels", "split"}
    if not required <= set(frame.columns):
        raise ValueError(f"Dataset sin columnas requeridas: {sorted(required - set(frame))}")
    if frame["chunk_id"].duplicated().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("El dataset congelado tiene IDs duplicados o texto vacío.")
    frames = {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    if any(part.empty for part in frames.values()):
        raise ValueError("Falta alguna partición train/validation/test.")
    manifest = _json(manifest_path)
    for split, expected in manifest["split_counts"].items():
        if len(frames[split]) != int(expected):
            raise ValueError(f"Conteo alterado en {split}: {len(frames[split])} != {expected}.")
    tm._verify_disjoint(frames)
    reference = _select_flat_reference(registry, frames)
    return {
        "registry": registry,
        "dataset_path": dataset_path,
        "manifest_path": manifest_path,
        "dataset_sha256": registry["dataset_sha256"],
        "manifest_sha256": registry["split_manifest_sha256"],
        "frames": frames,
        "reference": reference,
    }


def context_summary(context: dict | None = None) -> pd.DataFrame:
    """Resumen visible para la primera celda de ambos cuadernos."""
    context = context or load_frozen_context()
    rows = []
    for split, frame in context["frames"].items():
        y = experiment_targets(frame)
        rows.append(
            {
                "split": split,
                "chunks": len(frame),
                "videos": frame["video_id"].astype(str).nunique(),
                "con_daño": int(y.any(axis=1).sum()),
                "seguros": int((~y.any(axis=1)).sum()),
                "chunk_ids_sha256": _ids_sha256(frame["chunk_id"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["flat_reference"] = context["reference"]["model_label"]
    summary.attrs["flat_selection_metric"] = context["reference"][
        "validation_damage_pr_auc_macro"
    ]
    return summary


def expanded_safe_gate_training_frame(context: dict) -> tuple[pd.DataFrame, dict]:
    """Añade sólo SEGURO de videos de train para una ablación sin fuga.

    Los videos que no recibieron una partición en el dataset 4:1 no se usan,
    aunque estén disponibles en el integrado completo.
    """
    manifest = _json(context["manifest_path"])
    integrated_path = ROOT / manifest["input_integrated_dataset"]
    expected_hash = manifest.get("input_integrated_sha256")
    if not integrated_path.exists() or sha256_file(integrated_path) != expected_hash:
        raise ValueError("El dataset integrado completo está ausente o cambió de hash.")
    integrated = pd.DataFrame(read_jsonl(integrated_path))
    train = context["frames"]["train"].copy()
    train_videos = set(train["video_id"].astype(str))
    frozen_ids = set(
        pd.concat(list(context["frames"].values()), ignore_index=True)["chunk_id"].astype(str)
    )
    # La función inyectable conserva la semántica cuando una variante fusiona
    # etiquetas cuyos nombres ya no aparecen literalmente en coarse_labels.
    is_safe = ~experiment_targets(integrated).astype(bool).any(axis=1)
    candidates = integrated.loc[
        is_safe
        & integrated["video_id"].astype(str).isin(train_videos)
        & ~integrated["chunk_id"].astype(str).isin(frozen_ids)
    ].copy()
    candidates["split"] = "train"
    expanded = pd.concat([train, candidates], ignore_index=True, sort=False)
    if expanded["chunk_id"].duplicated().any():
        raise AssertionError("La ampliación de SEGURO produjo chunks duplicados.")
    if set(expanded["video_id"].astype(str)) - train_videos:
        raise AssertionError("La ampliación introdujo videos fuera de train.")
    metadata = {
        "purpose": "optional_negative_diversity_ablation_for_binary_gate",
        "integrated_dataset": _relative(integrated_path),
        "integrated_dataset_sha256": expected_hash,
        "matched_train_rows": int(len(train)),
        "additional_safe_rows": int(len(candidates)),
        "expanded_gate_train_rows": int(len(expanded)),
        "expanded_gate_train_videos": int(expanded["video_id"].astype(str).nunique()),
        "additional_chunk_ids_sha256": _ids_sha256(candidates["chunk_id"]),
        "validation_or_test_videos_used": False,
    }
    return expanded, metadata


class HierarchicalTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, prefix: str):
        self.texts = [prefix + str(text) for text in frame["text"]]
        self.categories = experiment_targets(frame)
        self.any_damage = self.categories.any(axis=1).astype(np.float32)
        self.weights = tm.source_weights(frame)
        self.category_loss_mask = (
            frame["category_loss_mask"].astype(float).to_numpy(dtype=np.float32)
            if "category_loss_mask" in frame
            else np.ones(len(frame), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return (
            self.texts[index],
            self.any_damage[index],
            self.categories[index],
            self.weights[index],
            self.category_loss_mask[index],
        )


class HierarchicalCollator:
    def __init__(self, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts, any_damage, categories, weights, category_loss_mask = zip(*examples)
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return (
            tokens,
            torch.tensor(any_damage, dtype=torch.float32),
            torch.tensor(np.asarray(categories), dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(category_loss_mask, dtype=torch.float32),
        )


def _loader(
    frame: pd.DataFrame,
    tokenizer,
    prefix: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        HierarchicalTextDataset(frame, prefix),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=HierarchicalCollator(tokenizer),
        generator=generator,
        pin_memory=False,
    )


def _pool(backbone, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = backbone(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class BinaryGateClassifier(nn.Module):
    def __init__(self, spec: tm.EncoderSpec):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
        hidden = int(self.backbone.config.hidden_size)
        dropout = float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.dropout(_pool(self.backbone, inputs))).squeeze(-1)


class ConditionalCategoryClassifier(nn.Module):
    def __init__(self, spec: tm.EncoderSpec):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
        hidden = int(self.backbone.config.hidden_size)
        dropout = float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, len(DAMAGE_ORDER))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.dropout(_pool(self.backbone, inputs)))


class JointHierarchicalClassifier(nn.Module):
    def __init__(self, spec: tm.EncoderSpec):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
        hidden = int(self.backbone.config.hidden_size)
        dropout = float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout)
        self.damage_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, len(DAMAGE_ORDER))

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.dropout(_pool(self.backbone, inputs))
        return self.damage_head(pooled).squeeze(-1), self.category_head(pooled)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(tokens: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in tokens.items()}


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (losses * weights).sum() / weights.sum().clamp(min=1e-6)


def _sqrt_pos_weight(binary_targets: np.ndarray) -> float:
    positives = float(np.asarray(binary_targets).sum())
    negatives = float(len(binary_targets) - positives)
    if positives <= 0:
        raise ValueError("No hay positivos para calcular pos_weight.")
    return math.sqrt(negatives / positives)


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def _predict_gate(
    model: BinaryGateClassifier,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> np.ndarray:
    model.eval()
    output = []
    for tokens, _, _, _, _ in tqdm(loader, desc=description, unit="lote"):
        logits = model(_to_device(tokens, device))
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def _predict_categories(
    model: ConditionalCategoryClassifier,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> np.ndarray:
    model.eval()
    output = []
    for tokens, _, _, _, _ in tqdm(loader, desc=description, unit="lote"):
        logits = model(_to_device(tokens, device))
        output.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(output)


@torch.inference_mode()
def _predict_joint(
    model: JointHierarchicalClassifier,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gates, categories = [], []
    for tokens, _, _, _, _ in tqdm(loader, desc=description, unit="lote"):
        gate_logits, category_logits = model(_to_device(tokens, device))
        gates.append(torch.sigmoid(gate_logits).cpu().numpy())
        categories.append(torch.sigmoid(category_logits).cpu().numpy())
    return np.concatenate(gates), np.vstack(categories)


def _threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    y = np.asarray(y_true, dtype=bool)
    if not y.any():
        raise ValueError("Validación sin daño para calibrar recall.")
    selected = float(np.nextafter(np.min(scores), -np.inf))
    for threshold in np.unique(scores)[::-1]:
        if recall_score(y, scores >= threshold, zero_division=0) >= target:
            selected = float(threshold)
            break
    return selected


def _threshold_for_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    minimum_threshold: float,
    target: float,
) -> float:
    y = np.asarray(y_true, dtype=bool)
    eligible = []
    for threshold in np.unique(scores):
        if threshold < minimum_threshold:
            continue
        prediction = scores >= threshold
        if prediction.any() and precision_score(y, prediction, zero_division=0) >= target:
            eligible.append(float(threshold))
    return min(eligible) if eligible else math.inf


def _train_binary_gate(
    spec: tm.EncoderSpec,
    frames: dict[str, pd.DataFrame],
    tokenizer,
    paths: dict[str, Path],
    dataset_sha256: str,
    reference_key: str,
    full_prevalence_weight: bool = False,
) -> tuple[dict, BinaryGateClassifier]:
    tm.set_reproducibility()
    device = _device()
    model = BinaryGateClassifier(spec).to(device)
    initialization = initialize_experiment_model(
        model, "binary_gate", spec, reference_key, dataset_sha256
    )
    train_loader = _loader(frames["train"], tokenizer, spec.prefix, TRAIN_BATCH_SIZE, True)
    validation_loader = _loader(
        frames["validation"], tokenizer, spec.prefix, EVAL_BATCH_SIZE, False
    )
    y_train = experiment_targets(frames["train"]).any(axis=1).astype(np.float32)
    y_validation = experiment_targets(frames["validation"]).any(axis=1).astype(np.int8)
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = (
        negatives / positives
        if full_prevalence_weight
        else _sqrt_pos_weight(y_train)
    )
    pos_weight_mode = (
        "full_negative_to_positive_ratio"
        if full_prevalence_weight
        else "sqrt_negative_to_positive_ratio"
    )
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = tm.scheduler_for(optimizer, len(train_loader) * MAX_EPOCHS)
    best_score, stale, history = -math.inf, 0, []
    best_path = paths["models"] / "gate_best.pt"
    start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        cumulative, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"puerta binaria · época {epoch}/{MAX_EPOCHS}", unit="lote")
        for tokens, any_damage, _, weights, _ in progress:
            any_damage = any_damage.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(_to_device(tokens, device))
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, any_damage, pos_weight=pos_weight, reduction="none"
            )
            loss = _weighted_mean(losses, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch = len(any_damage)
            cumulative += float(loss.detach()) * batch
            seen += batch
            progress.set_postfix(loss=f"{cumulative / seen:.4f}")
        validation_scores = _predict_gate(
            model, validation_loader, device, f"puerta · validación época {epoch}"
        )
        pr_auc = float(average_precision_score(y_validation, validation_scores))
        recall_threshold = _threshold_for_recall(
            y_validation, validation_scores, GATE_VALIDATION_RECALL_TARGET
        )
        record = {
            "epoch": epoch,
            "training_loss": cumulative / seen,
            "validation_binary_pr_auc": pr_auc,
            "validation_recall_threshold": recall_threshold,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(paths["metrics"] / "historial_puerta.csv", index=False)
        tqdm.write(
            f"Puerta época {epoch}: PR-AUC={pr_auc:.4f}; "
            f"umbral recall {GATE_VALIDATION_RECALL_TARGET:.0%}={recall_threshold:.4f}"
        )
        if pr_auc > best_score + 1e-6:
            best_score, stale = pr_auc, 0
            _save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "model_spec": asdict(spec),
                    "epoch": epoch,
                    "history": history,
                    "dataset_sha256": dataset_sha256,
                    "flat_reference_key": reference_key,
                    "pos_weight": pos_weight_value,
                    "pos_weight_mode": pos_weight_mode,
                },
            )
        else:
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_binary_pr_auc": float(best_score),
        "epochs_completed": len(history),
        "training_seconds": perf_counter() - start,
        "positive_weight": pos_weight_value,
        "positive_weight_mode": pos_weight_mode,
        "initialization": initialization,
        "history": history,
        "checkpoint": _artifact(best_path),
    }
    return result, model


def _stage2_training_frame(
    train: pd.DataFrame, gate_scores: np.ndarray
) -> tuple[pd.DataFrame, dict]:
    y = experiment_targets(train)
    damage_indices = np.flatnonzero(y.any(axis=1))
    safe_indices = np.flatnonzero(~y.any(axis=1))
    safe_budget = min(
        len(safe_indices),
        int(round(len(damage_indices) * CASCADE_SAFE_TO_DAMAGE_STAGE2_RATIO)),
    )
    hard_budget = safe_budget // 2
    safe_ranked = safe_indices[np.argsort(gate_scores[safe_indices])[::-1]]
    hard = safe_ranked[:hard_budget]
    hard_set = set(hard.tolist())
    remaining = [index for index in safe_indices if int(index) not in hard_set]
    remaining.sort(
        key=lambda index: hashlib.sha256(
            f"{SEED}|{train.iloc[index]['chunk_id']}".encode("utf-8")
        ).hexdigest()
    )
    random_safe = np.asarray(remaining[: safe_budget - len(hard)], dtype=int)
    selected = np.concatenate([damage_indices, hard, random_safe])
    selected.sort()
    frame = train.iloc[selected].reset_index(drop=True)
    metadata = {
        "method": "all_damage_plus_50pct_gate_hard_negatives_plus_50pct_hash_safe",
        "all_damage_rows": int(len(damage_indices)),
        "safe_rows": int(safe_budget),
        "hard_safe_rows": int(len(hard)),
        "hash_random_safe_rows": int(len(random_safe)),
        "total_rows": int(len(frame)),
        "selected_chunk_ids_sha256": _ids_sha256(frame["chunk_id"]),
    }
    return frame, metadata


def _train_stage2(
    spec: tm.EncoderSpec,
    frames: dict[str, pd.DataFrame],
    tokenizer,
    gate_model: BinaryGateClassifier,
    gate_scores: dict[str, np.ndarray],
    paths: dict[str, Path],
    dataset_sha256: str,
    reference_key: str,
) -> tuple[dict, ConditionalCategoryClassifier]:
    tm.set_reproducibility()
    device = _device()
    stage_frame, selection = _stage2_training_frame(frames["train"], gate_scores["train"])
    model = ConditionalCategoryClassifier(spec).to(device)
    model.backbone.load_state_dict(gate_model.backbone.state_dict())
    initialization = initialize_experiment_model(
        model, "conditional_categories", spec, reference_key, dataset_sha256
    )
    train_loader = _loader(stage_frame, tokenizer, spec.prefix, TRAIN_BATCH_SIZE, True)
    validation_loader = _loader(
        frames["validation"], tokenizer, spec.prefix, EVAL_BATCH_SIZE, False
    )
    y_train = experiment_targets(stage_frame)
    y_validation = experiment_targets(frames["validation"])
    pos_weights_np = tm.positive_weights(y_train, "sqrt_positive_weight")
    pos_weights = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = tm.scheduler_for(optimizer, len(train_loader) * MAX_EPOCHS)
    best_score, stale, history = -math.inf, 0, []
    best_path = paths["models"] / "categorias_best.pt"
    start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        cumulative, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"categorías condicionales · época {epoch}/{MAX_EPOCHS}", unit="lote")
        for tokens, _, categories, weights, _ in progress:
            categories = categories.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(_to_device(tokens, device))
            element = nn.functional.binary_cross_entropy_with_logits(
                logits, categories, pos_weight=pos_weights, reduction="none"
            )
            loss = _weighted_mean(element.mean(dim=1), weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch = len(categories)
            cumulative += float(loss.detach()) * batch
            seen += batch
            progress.set_postfix(loss=f"{cumulative / seen:.4f}")
        conditional = _predict_categories(
            model, validation_loader, device, f"categorías · validación época {epoch}"
        )
        combined = gate_scores["validation"][:, None] * conditional
        thresholds = tune_thresholds(y_validation.astype(np.int8), combined)
        metrics, _, _ = evaluate_experiment_scores(y_validation, combined, thresholds)
        record = {
            "epoch": epoch,
            "training_loss": cumulative / seen,
            "validation_damage_pr_auc_macro": metrics["damage_pr_auc_macro"],
            "validation_damage_f1_macro": metrics["damage_f1_macro"],
            "thresholds": thresholds.tolist(),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(paths["metrics"] / "historial_categorias.csv", index=False)
        tqdm.write(
            f"Categorías época {epoch}: PR-AUC macro={metrics['damage_pr_auc_macro']:.4f}; "
            f"F1 macro={metrics['damage_f1_macro']:.4f}"
        )
        score = float(metrics["damage_pr_auc_macro"])
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            _save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "model_spec": asdict(spec),
                    "epoch": epoch,
                    "history": history,
                    "thresholds": thresholds.tolist(),
                    "dataset_sha256": dataset_sha256,
                    "flat_reference_key": reference_key,
                    "stage2_selection": selection,
                    "positive_weights": pos_weights_np.tolist(),
                    "score_definition": "sigmoid(gate) * sigmoid(category_given_gate)",
                },
            )
        else:
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_damage_pr_auc_macro": float(best_score),
        "epochs_completed": len(history),
        "training_seconds": perf_counter() - start,
        "positive_weights_sqrt": pos_weights_np.tolist(),
        "training_selection": selection,
        "initialization": {
            **initialization,
            "backbone_source": "trained_binary_gate_from_same_experiment",
        },
        "history": history,
        "checkpoint": _artifact(best_path),
    }
    return result, model


def _train_joint(
    spec: tm.EncoderSpec,
    frames: dict[str, pd.DataFrame],
    tokenizer,
    paths: dict[str, Path],
    dataset_sha256: str,
    reference_key: str,
) -> tuple[dict, JointHierarchicalClassifier]:
    tm.set_reproducibility()
    device = _device()
    model = JointHierarchicalClassifier(spec).to(device)
    initialization = initialize_experiment_model(
        model, "joint_hierarchical", spec, reference_key, dataset_sha256
    )
    train_loader = _loader(frames["train"], tokenizer, spec.prefix, TRAIN_BATCH_SIZE, True)
    validation_loader = _loader(
        frames["validation"], tokenizer, spec.prefix, EVAL_BATCH_SIZE, False
    )
    y_train = experiment_targets(frames["train"])
    y_validation = experiment_targets(frames["validation"])
    binary_pos_weight_value = _sqrt_pos_weight(y_train.any(axis=1))
    binary_pos_weight = torch.tensor(
        binary_pos_weight_value, dtype=torch.float32, device=device
    )
    category_mask_np = frames["train"].get(
        "category_loss_mask", pd.Series(np.ones(len(frames["train"])))
    ).astype(bool).to_numpy()
    category_pos_weights_np = tm.positive_weights(
        y_train[category_mask_np], "sqrt_positive_weight"
    )
    category_pos_weights = torch.tensor(
        category_pos_weights_np, dtype=torch.float32, device=device
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = tm.scheduler_for(optimizer, len(train_loader) * MAX_EPOCHS)
    best_score, stale, history = -math.inf, 0, []
    best_path = paths["models"] / "jerarquico_best.pt"
    start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        cumulative, seen = 0.0, 0
        progress = tqdm(train_loader, desc=f"jerárquico conjunto · época {epoch}/{MAX_EPOCHS}", unit="lote")
        for tokens, any_damage, categories, weights, category_loss_mask in progress:
            any_damage = any_damage.to(device)
            categories = categories.to(device)
            weights = weights.to(device)
            category_loss_mask = category_loss_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            gate_logits, category_logits = model(_to_device(tokens, device))
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                gate_logits, any_damage, pos_weight=binary_pos_weight, reduction="none"
            )
            category_loss = nn.functional.binary_cross_entropy_with_logits(
                category_logits,
                categories,
                pos_weight=category_pos_weights,
                reduction="none",
            ).mean(dim=1)
            gate_probability = torch.sigmoid(gate_logits)
            maximum_category = torch.sigmoid(category_logits).max(dim=1).values
            consistency_loss = torch.relu(maximum_category - gate_probability).square()
            per_sample = (
                JOINT_BINARY_LOSS_WEIGHT * binary_loss
                + JOINT_CATEGORY_LOSS_WEIGHT * category_loss * category_loss_mask
                + JOINT_CONSISTENCY_LOSS_WEIGHT * consistency_loss
            )
            loss = _weighted_mean(per_sample, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch = len(categories)
            cumulative += float(loss.detach()) * batch
            seen += batch
            progress.set_postfix(loss=f"{cumulative / seen:.4f}")
        gates, conditional = _predict_joint(
            model, validation_loader, device, f"jerárquico · validación época {epoch}"
        )
        combined = gates[:, None] * conditional
        thresholds = tune_thresholds(y_validation.astype(np.int8), combined)
        metrics, _, _ = evaluate_experiment_scores(y_validation, combined, thresholds)
        binary_pr_auc = float(
            average_precision_score(y_validation.any(axis=1), gates)
        )
        record = {
            "epoch": epoch,
            "training_loss": cumulative / seen,
            "validation_binary_pr_auc": binary_pr_auc,
            "validation_damage_pr_auc_macro": metrics["damage_pr_auc_macro"],
            "validation_damage_f1_macro": metrics["damage_f1_macro"],
            "thresholds": thresholds.tolist(),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(paths["metrics"] / "historial_jerarquico.csv", index=False)
        tqdm.write(
            f"Jerárquico época {epoch}: PR-AUC macro={metrics['damage_pr_auc_macro']:.4f}; "
            f"F1 macro={metrics['damage_f1_macro']:.4f}; PR-AUC binaria={binary_pr_auc:.4f}"
        )
        score = float(metrics["damage_pr_auc_macro"])
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            _save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "model_spec": asdict(spec),
                    "epoch": epoch,
                    "history": history,
                    "thresholds": thresholds.tolist(),
                    "dataset_sha256": dataset_sha256,
                    "flat_reference_key": reference_key,
                    "category_positive_weights": category_pos_weights_np.tolist(),
                    "binary_positive_weight": binary_pos_weight_value,
                    "loss_weights": {
                        "binary": JOINT_BINARY_LOSS_WEIGHT,
                        "category": JOINT_CATEGORY_LOSS_WEIGHT,
                        "consistency": JOINT_CONSISTENCY_LOSS_WEIGHT,
                    },
                    "score_definition": "sigmoid(binary_head) * sigmoid(category_head)",
                },
            )
        else:
            stale += 1
        if stale >= EARLY_STOPPING_PATIENCE:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_damage_pr_auc_macro": float(best_score),
        "epochs_completed": len(history),
        "training_seconds": perf_counter() - start,
        "binary_positive_weight_sqrt": binary_pos_weight_value,
        "category_positive_weights_sqrt": category_pos_weights_np.tolist(),
        "loss_weights": {
            "binary": JOINT_BINARY_LOSS_WEIGHT,
            "category": JOINT_CATEGORY_LOSS_WEIGHT,
            "consistency": JOINT_CONSISTENCY_LOSS_WEIGHT,
        },
        "initialization": initialization,
        "history": history,
        "checkpoint": _artifact(best_path),
    }
    return result, model


def _model_summary(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    metrics, report, predictions = evaluate_experiment_scores(y, scores, thresholds)
    metrics["category_false_negatives"] = int(
        (y.astype(bool) & ~predictions.astype(bool)).sum()
    )
    metrics["minimum_category_recall"] = float(
        min(
            recall_score(y[:, index], predictions[:, index], zero_division=0)
            for index in range(y.shape[1])
        )
    )
    return metrics, report, predictions


def _per_category_comparison(
    y: np.ndarray,
    flat_scores: np.ndarray,
    flat_thresholds: np.ndarray,
    experiment_scores: np.ndarray,
    experiment_thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(DAMAGE_ORDER):
        flat_pred = flat_scores[:, index] >= flat_thresholds[index]
        experiment_pred = experiment_scores[:, index] >= experiment_thresholds[index]
        flat_p, flat_r, flat_f, _ = precision_recall_fscore_support(
            y[:, index], flat_pred, average="binary", zero_division=0
        )
        exp_p, exp_r, exp_f, _ = precision_recall_fscore_support(
            y[:, index], experiment_pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "categoria": label,
                "positivos_test": int(y[:, index].sum()),
                "precision_plano": float(flat_p),
                "recall_plano": float(flat_r),
                "f1_plano": float(flat_f),
                "pr_auc_plano": float(average_precision_score(y[:, index], flat_scores[:, index])),
                "falsos_negativos_plano": int((y[:, index].astype(bool) & ~flat_pred).sum()),
                "precision_experimento": float(exp_p),
                "recall_experimento": float(exp_r),
                "f1_experimento": float(exp_f),
                "pr_auc_experimento": float(
                    average_precision_score(y[:, index], experiment_scores[:, index])
                ),
                "falsos_negativos_experimento": int(
                    (y[:, index].astype(bool) & ~experiment_pred).sum()
                ),
                "delta_recall": float(exp_r - flat_r),
                "delta_pr_auc": float(
                    average_precision_score(y[:, index], experiment_scores[:, index])
                    - average_precision_score(y[:, index], flat_scores[:, index])
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_metrics(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> dict[str, float]:
    predictions = scores >= thresholds
    true_any = y.astype(bool).any(axis=1)
    predicted_any = predictions.any(axis=1)
    return {
        "pr_auc_macro": float(average_precision_score(y, scores, average="macro")),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "any_damage_recall": float(recall_score(true_any, predicted_any, zero_division=0)),
        "false_negative_rate": float(
            (true_any & ~predicted_any).sum() / max(1, true_any.sum())
        ),
    }


def paired_cluster_bootstrap(
    y: np.ndarray,
    flat_scores: np.ndarray,
    flat_thresholds: np.ndarray,
    experiment_scores: np.ndarray,
    experiment_thresholds: np.ndarray,
    video_ids: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> pd.DataFrame:
    """IC percentil pareado, remuestreando videos para preservar clústeres."""
    groups = np.asarray(video_ids, dtype=str)
    unique_groups = np.unique(groups)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    observed_flat = _bootstrap_metrics(y, flat_scores, flat_thresholds)
    observed_experiment = _bootstrap_metrics(y, experiment_scores, experiment_thresholds)
    rng = np.random.default_rng(seed)
    deltas = {metric: [] for metric in observed_flat}
    progress = tqdm(range(replicates), desc="bootstrap pareado por video", unit="réplica")
    for _ in progress:
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        # Omite réplicas degeneradas sin positivos en alguna categoría.
        if (y[indices].sum(axis=0) == 0).any():
            continue
        flat = _bootstrap_metrics(y[indices], flat_scores[indices], flat_thresholds)
        experiment = _bootstrap_metrics(
            y[indices], experiment_scores[indices], experiment_thresholds
        )
        for metric in deltas:
            deltas[metric].append(experiment[metric] - flat[metric])
    rows = []
    for metric, values in deltas.items():
        values_array = np.asarray(values, dtype=float)
        if len(values_array) < max(100, int(0.80 * replicates)):
            raise RuntimeError("Demasiadas réplicas bootstrap degeneradas.")
        observed_delta = observed_experiment[metric] - observed_flat[metric]
        rows.append(
            {
                "metrica": metric,
                "plano_test": observed_flat[metric],
                "experimento_test": observed_experiment[metric],
                "delta_experimento_menos_plano": observed_delta,
                "ic95_inferior": float(np.quantile(values_array, 0.025)),
                "ic95_superior": float(np.quantile(values_array, 0.975)),
                "replicas_validas": int(len(values_array)),
            }
        )
    return pd.DataFrame(rows)


def _selective_gate_metrics(
    y: np.ndarray,
    gate_scores: np.ndarray,
    combined_scores: np.ndarray,
    category_thresholds: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict:
    true_any = y.astype(bool).any(axis=1)
    auto_safe = gate_scores < low_threshold
    auto_damage_candidate = gate_scores >= high_threshold
    category_prediction = combined_scores >= category_thresholds
    no_category = ~category_prediction.any(axis=1)
    review = (~auto_safe & ~auto_damage_candidate) | (auto_damage_candidate & no_category)
    auto_damage = auto_damage_candidate & ~no_category
    return {
        "n": int(len(y)),
        "auto_safe_rows": int(auto_safe.sum()),
        "auto_damage_rows": int(auto_damage.sum()),
        "review_rows": int(review.sum()),
        "automatic_coverage": float((auto_safe | auto_damage).mean()),
        "review_rate": float(review.mean()),
        "damage_sent_to_review_or_auto_damage_recall": float(
            ((review | auto_damage) & true_any).sum() / max(1, true_any.sum())
        ),
        "damage_automatic_safe_false_negatives": int((true_any & auto_safe).sum()),
        "damage_automatic_safe_false_negative_rate": float(
            (true_any & auto_safe).sum() / max(1, true_any.sum())
        ),
        "safe_sent_to_review": int((~true_any & review).sum()),
        "low_auto_safe_threshold": float(low_threshold),
        "high_auto_damage_threshold": (
            float(high_threshold) if math.isfinite(high_threshold) else None
        ),
    }


def _decision_from_bootstrap(bootstrap: pd.DataFrame) -> dict:
    pr = bootstrap.loc[bootstrap["metrica"].eq("pr_auc_macro")].iloc[0]
    fnr = bootstrap.loc[bootstrap["metrica"].eq("false_negative_rate")].iloc[0]
    if pr["ic95_inferior"] > 0 and fnr["ic95_superior"] <= 0:
        status = "mejora_respaldada_en_pr_auc_sin_aumento_de_falsos_negativos"
    elif pr["ic95_superior"] < 0:
        status = "modelo_plano_superior_en_pr_auc_macro"
    else:
        status = "diferencia_inconclusa_con_este_test"
    return {
        "rule": (
            "Mejora sólo si el IC95% del delta PR-AUC macro es > 0 y el límite "
            "superior del delta de tasa de falsos negativos es <= 0."
        ),
        "status": status,
        "replace_flat_model": status.startswith("mejora_respaldada"),
        "autonomous_deployment_supported": False,
        "reason_autonomy": (
            "El test conserva etiquetas mayormente LLM y prevalencia 4:1; se requiere "
            "un gold standard humano independiente y un piloto prospectivo."
        ),
    }


def _comparison_frame(
    reference: dict,
    experiment_key: str,
    validation_metrics: dict,
    test_metrics: dict,
    flat_validation_metrics: dict,
    flat_test_metrics: dict,
) -> pd.DataFrame:
    fields = [
        "damage_pr_auc_macro",
        "damage_f1_macro",
        "damage_recall_micro",
        "any_damage_recall",
        "minimum_category_recall",
        "missed_damage_as_safe",
        "category_false_negatives",
    ]
    rows = []
    for split, flat, experiment in (
        ("validation", flat_validation_metrics, validation_metrics),
        ("test", flat_test_metrics, test_metrics),
    ):
        for label, values in (
            (reference["model_label"], flat),
            (EXPERIMENT_LABELS[experiment_key], experiment),
        ):
            rows.append(
                {"split": split, "modelo": label, **{field: values[field] for field in fields}}
            )
    return pd.DataFrame(rows)


def _plots(
    key: str,
    comparison: pd.DataFrame,
    categories: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paths: dict[str, Path],
) -> list[dict]:
    artifacts = []
    test = comparison.loc[comparison["split"].eq("test")].set_index("modelo")
    metrics = ["damage_pr_auc_macro", "damage_f1_macro", "any_damage_recall"]
    labels = ["PR-AUC macro", "F1 macro", "Recall daño"]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.36
    for offset, (model, row) in zip((-width / 2, width / 2), test.iterrows()):
        axis.bar(x + offset, [row[m] for m in metrics], width, label=model)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Puntuación en el mismo test")
    axis.set_title(f"{EXPERIMENT_LABELS[key]} frente al modelo plano")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = paths["figures"] / "comparacion_global_test.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))

    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    axis.bar(x - width / 2, categories["recall_plano"], width, label="Plano")
    axis.bar(
        x + width / 2, categories["recall_experimento"], width, label="Jerárquico"
    )
    axis.set_xticks(x, categories["categoria"], rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall por categoría en test")
    axis.set_title("Falsos negativos por categoría")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = paths["figures"] / "recall_por_categoria_test.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))

    forest = bootstrap.copy()
    figure, axis = plt.subplots(figsize=(9, 5))
    y_positions = np.arange(len(forest))
    centers = forest["delta_experimento_menos_plano"].to_numpy()
    lower = centers - forest["ic95_inferior"].to_numpy()
    upper = forest["ic95_superior"].to_numpy() - centers
    axis.errorbar(centers, y_positions, xerr=[lower, upper], fmt="o", capsize=4)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y_positions, forest["metrica"])
    axis.set_xlabel("Delta jerárquico − plano (IC 95% bootstrap por video)")
    axis.set_title("Incertidumbre de la diferencia pareada")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = paths["figures"] / "bootstrap_deltas_test.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))
    return artifacts


def _write_report(
    result: dict,
    comparison: pd.DataFrame,
    categories: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    key = result["experiment_key"]
    test_rows = comparison.loc[comparison["split"].eq("test")]
    table_rows = []
    for _, row in test_rows.iterrows():
        table_rows.append(
            f"| {row['modelo']} | {row['damage_pr_auc_macro']:.4f} | "
            f"{row['damage_f1_macro']:.4f} | {row['any_damage_recall']:.4f} | "
            f"{int(row['missed_damage_as_safe'])} |"
        )
    category_rows = []
    for _, row in categories.iterrows():
        category_rows.append(
            f"| {row['categoria']} | {int(row['positivos_test'])} | "
            f"{row['recall_plano']:.4f} | {row['recall_experimento']:.4f} | "
            f"{row['delta_recall']:+.4f} |"
        )
    bootstrap_rows = []
    for _, row in bootstrap.iterrows():
        bootstrap_rows.append(
            f"| {row['metrica']} | {row['delta_experimento_menos_plano']:+.4f} | "
            f"[{row['ic95_inferior']:+.4f}, {row['ic95_superior']:+.4f}] | "
            f"{int(row['replicas_validas'])} |"
        )
    selective = result.get("selective_operation", {})
    selective_text = (
        f"La zona de abstención calibrada sólo en validación usa `p(daño) < "
        f"{selective.get('low_auto_safe_threshold', math.nan):.4f}` para auto-paso seguro. "
        f"En test envía {selective.get('review_rate', math.nan):.2%} a revisión y deja "
        f"{selective.get('damage_automatic_safe_false_negatives', 'NA')} daños como auto-seguros."
        if selective
        else "No se calculó una política selectiva."
    )
    training_control = result["architecture_control"]["same_training_rows_as_flat"]
    training_design_text = (
        "Ambos modelos usan exactamente los mismos IDs de `train`, `validation` y `test`."
        if training_control
        else (
            "La ablación conserva exactamente `validation/test`, pero amplía sólo la puerta "
            "binaria con chunks SEGURO pertenecientes a videos de `train`; por ello no aísla "
            "el efecto de arquitectura del efecto de datos adicionales."
        )
    )
    report = f"""# {EXPERIMENT_LABELS[key]} frente al modelo plano

Fecha: {tm.now_iso()}

## Pregunta y diseño

Se evalúa si **{EXPERIMENT_LABELS[key]}** mejora la referencia plana **{result['flat_reference']['model_label']}**. {training_design_text} El dataset y el manifiesto se verificaron por SHA-256. Sólo las cinco etiquetas gruesas de daño son objetivos. Las etiquetas finas y los flags transversales no ingresan como variables predictoras.

El encoder base se igualó al del Transformer plano seleccionado por **PR-AUC macro de validación**. Época, umbrales y reglas de abstención se fijaron sin consultar test. Esto reduce el sesgo de selección descrito por Cawley y Talbot (2010). PR-AUC macro es primaria porque las categorías son desbalanceadas y una curva precisión–recall es más informativa que ROC bajo esa condición (Saito & Rehmsmeier, 2015).

## Inferencia estadística

La diferencia se calcula de forma pareada sobre los mismos chunks. El intervalo de confianza percentil 95 % usa {result['bootstrap_replicates']} remuestreos de **videos completos**, no chunks aislados, preservando la dependencia intravideo. La regla declarada antes de la interpretación exige que todo el IC 95 % de Δ PR-AUC macro sea positivo y que el límite superior de Δ tasa de falsos negativos no sea mayor que cero. El test no se usa para escoger entre configuraciones.

## Resultados en el mismo test

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños predichos como seguro |
|---|---:|---:|---:|---:|
{os.linesep.join(table_rows)}

| Categoría | Positivos | Recall plano | Recall jerárquico | Δ recall |
|---|---:|---:|---:|---:|
{os.linesep.join(category_rows)}

| Métrica | Δ experimento − plano | IC 95 % por video | Réplicas |
|---|---:|---:|---:|
{os.linesep.join(bootstrap_rows)}

{selective_text}

## Conclusión reproducible

Resultado de la regla: **`{result['decision']['status']}`**. Reemplazar el modelo plano: **{'sí' if result['decision']['replace_flat_model'] else 'no'}**.

Esto sólo establece desempeño retrospectivo relativo. No autoriza moderación autónoma: las etiquetas de test son mayormente generadas o adjudicadas con apoyo de LLM y el muestreo 4:1 no representa prevalencia natural. Antes de producción se requiere un gold standard humano independiente y un piloto prospectivo.

## Artefactos

- Resultado JSON: `{_relative(paths['result'])}`
- Comparación: `{_relative(paths['comparison'])}`
- Recall por categoría: `{_relative(paths['categories'])}`
- Bootstrap: `{_relative(paths['bootstrap'])}`
- Checkpoints: `{_relative(paths['models'])}`
- Figuras: `{_relative(paths['figures'])}`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *International Conference on Learning Representations*. https://openreview.net/forum?id=ryMhqj0ct7

Naik, A., & Rangwala, H. (2017). Filter based taxonomy modification for improving hierarchical classification. *arXiv*. https://doi.org/10.48550/arXiv.1706.01214

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Zhou, J., Ma, C., Long, D., Xu, G., Ding, N., Zhang, H., Xie, P., & Liu, G. (2020). Hierarchy-aware global model for hierarchical text classification. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1106–1117). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.104
"""
    paths["report"].write_text(report, encoding="utf-8")


def _finalize(
    key: str,
    context: dict,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    thresholds: np.ndarray,
    training: dict,
    paths: dict[str, Path],
    bootstrap_replicates: int,
    selective_operation: dict | None = None,
    extra_artifacts: list[dict] | None = None,
    same_training_rows_as_flat: bool = True,
) -> dict:
    reference = context["reference"]
    y_validation = experiment_targets(context["frames"]["validation"])
    y_test = experiment_targets(context["frames"]["test"])
    validation_metrics, validation_report, _ = _model_summary(
        y_validation, validation_scores, thresholds
    )
    test_metrics, test_report, _ = _model_summary(y_test, test_scores, thresholds)
    flat_validation_metrics, _, _ = _model_summary(
        y_validation, reference["validation_scores"], reference["thresholds"]
    )
    flat_test_metrics, _, _ = _model_summary(
        y_test, reference["test_scores"], reference["thresholds"]
    )
    comparison = _comparison_frame(
        reference,
        key,
        validation_metrics,
        test_metrics,
        flat_validation_metrics,
        flat_test_metrics,
    )
    categories = _per_category_comparison(
        y_test,
        reference["test_scores"],
        reference["thresholds"],
        test_scores,
        thresholds,
    )
    bootstrap = paired_cluster_bootstrap(
        y_test,
        reference["test_scores"],
        reference["thresholds"],
        test_scores,
        thresholds,
        context["frames"]["test"]["video_id"].astype(str).to_numpy(),
        replicates=bootstrap_replicates,
    )
    comparison.to_csv(paths["comparison"], index=False)
    categories.to_csv(paths["categories"], index=False)
    bootstrap.to_csv(paths["bootstrap"], index=False)
    validation_report.to_csv(paths["metrics"] / "reporte_validation.csv")
    test_report.to_csv(paths["metrics"] / "reporte_test.csv")
    validation_score_path = paths["metrics"] / "scores_validation.npy"
    test_score_path = paths["metrics"] / "scores_test.npy"
    np.save(validation_score_path, validation_scores)
    np.save(test_score_path, test_scores)
    decision = _decision_from_bootstrap(bootstrap)
    result = {
        "schema_version": "1.0",
        "created_at": tm.now_iso(),
        "experiment_key": key,
        "experiment_label": EXPERIMENT_LABELS[key],
        "dataset": {
            "path": _relative(context["dataset_path"]),
            "sha256": context["dataset_sha256"],
            "split_manifest": _relative(context["manifest_path"]),
            "split_manifest_sha256": context["manifest_sha256"],
            "split_counts": {
                split: int(len(frame)) for split, frame in context["frames"].items()
            },
            "split_chunk_ids_sha256": {
                split: _ids_sha256(frame["chunk_id"])
                for split, frame in context["frames"].items()
            },
            "targets": list(DAMAGE_ORDER),
            "fine_labels_trained": False,
            "transversal_flags_trained": False,
        },
        "flat_reference": {
            "model_key": reference["model_key"],
            "model_label": reference["model_label"],
            "selection_partition": "validation",
            "selection_metric": "damage_pr_auc_macro",
            "validation_damage_pr_auc_macro": reference[
                "validation_damage_pr_auc_macro"
            ],
            "thresholds": reference["thresholds"].tolist(),
            "evaluation": reference["evaluation_artifact"],
            "validation_scores": reference["validation_score_artifact"],
            "test_scores": reference["test_score_artifact"],
        },
        "architecture_control": {
            "same_encoder_model_id": reference["spec"].model_id,
            "same_encoder_revision": reference["spec"].revision,
            "same_splits": True,
            "same_training_rows_as_flat": same_training_rows_as_flat,
            "test_used_for_training_selection_or_thresholds": False,
        },
        "training": training,
        "thresholds_selected_on_validation": thresholds.tolist(),
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "flat_metrics_recomputed": {
            "validation": flat_validation_metrics,
            "test": flat_test_metrics,
        },
        "selective_operation": selective_operation,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_cluster": "video_id",
        "decision": decision,
        "score_artifacts": [_artifact(validation_score_path), _artifact(test_score_path)],
        "extra_artifacts": extra_artifacts or [],
    }
    tm.write_json(paths["result"], result)
    figures = _plots(key, comparison, categories, bootstrap, paths)
    result["figure_artifacts"] = figures
    tm.write_json(paths["result"], result)
    _write_report(result, comparison, categories, bootstrap, paths)
    result["report_artifact"] = _artifact(paths["report"])
    tm.write_json(paths["result"], result)
    return result


def _cached_result(
    key: str,
    context: dict,
    force: bool,
    bootstrap_replicates: int,
) -> dict | None:
    path = _experiment_paths(key)["result"]
    if force or not path.exists():
        return None
    result = _json(path)
    if result.get("dataset", {}).get("sha256") != context["dataset_sha256"]:
        raise ValueError(
            f"El resultado previo de {key} usa otro dataset; ejecute con force=True."
        )
    if result.get("flat_reference", {}).get("model_key") != context["reference"]["model_key"]:
        raise ValueError(
            f"Cambió la referencia plana de {key}; ejecute con force=True."
        )
    if int(result.get("bootstrap_replicates", -1)) != int(bootstrap_replicates):
        raise ValueError(
            "Cambió el número de réplicas bootstrap; ejecute con force=True."
        )
    if key == CASCADE_EXTRA_SAFE_KEY:
        current_manifest = _json(context["manifest_path"])
        cached_integrated_hash = (
            result.get("training", {})
            .get("gate", {})
            .get("training_data", {})
            .get("integrated_dataset_sha256")
        )
        if cached_integrated_hash != current_manifest.get("input_integrated_sha256"):
            raise ValueError(
                "Cambió el dataset integrado usado por la puerta; ejecute con force=True."
            )
    return result


def run_cascade_experiment(
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    expanded_safe_gate: bool = True,
) -> dict:
    """Entrena, evalúa y documenta la cascada completa."""
    key = CASCADE_EXTRA_SAFE_KEY if expanded_safe_gate else CASCADE_KEY
    context = load_frozen_context()
    cached = _cached_result(key, context, force, bootstrap_replicates)
    if cached is not None:
        return cached
    paths = _experiment_paths(key)
    _ensure_directories(paths)
    reference = context["reference"]
    spec = reference["spec"]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    tokenizer.save_pretrained(paths["models"] / "tokenizer")
    gate_frames = context["frames"]
    gate_data = {
        "purpose": "strict_architecture_comparison",
        "matched_train_rows": int(len(context["frames"]["train"])),
        "additional_safe_rows": 0,
        "expanded_gate_train_rows": int(len(context["frames"]["train"])),
        "validation_or_test_videos_used": False,
    }
    if expanded_safe_gate:
        expanded_train, gate_data = expanded_safe_gate_training_frame(context)
        gate_frames = {**context["frames"], "train": expanded_train}
    gate_training, gate_model = _train_binary_gate(
        spec,
        gate_frames,
        tokenizer,
        paths,
        context["dataset_sha256"],
        reference["model_key"],
        full_prevalence_weight=expanded_safe_gate,
    )
    gate_training["training_data"] = gate_data
    device = _device()
    gate_scores = {}
    for split, frame in context["frames"].items():
        gate_scores[split] = _predict_gate(
            gate_model,
            _loader(frame, tokenizer, spec.prefix, EVAL_BATCH_SIZE, False),
            device,
            f"puerta · {split}",
        )
        np.save(paths["metrics"] / f"scores_puerta_{split}.npy", gate_scores[split])
    stage_training, stage_model = _train_stage2(
        spec,
        context["frames"],
        tokenizer,
        gate_model,
        gate_scores,
        paths,
        context["dataset_sha256"],
        reference["model_key"],
    )
    combined_scores = {}
    for split in ("validation", "test"):
        conditional = _predict_categories(
            stage_model,
            _loader(
                context["frames"][split], tokenizer, spec.prefix, EVAL_BATCH_SIZE, False
            ),
            device,
            f"categorías condicionales · {split}",
        )
        np.save(paths["metrics"] / f"scores_condicionales_{split}.npy", conditional)
        combined_scores[split] = gate_scores[split][:, None] * conditional
    y_validation = experiment_targets(context["frames"]["validation"])
    thresholds = tune_thresholds(y_validation.astype(np.int8), combined_scores["validation"])
    low = _threshold_for_recall(
        y_validation.any(axis=1),
        gate_scores["validation"],
        GATE_VALIDATION_RECALL_TARGET,
    )
    high = _threshold_for_precision(
        y_validation.any(axis=1),
        gate_scores["validation"],
        low,
        AUTO_DAMAGE_PRECISION_TARGET,
    )
    validation_selective = _selective_gate_metrics(
        y_validation,
        gate_scores["validation"],
        combined_scores["validation"],
        thresholds,
        low,
        high,
    )
    test_selective = _selective_gate_metrics(
        experiment_targets(context["frames"]["test"]),
        gate_scores["test"],
        combined_scores["test"],
        thresholds,
        low,
        high,
    )
    selective = {
        "calibration_partition": "validation",
        "validation_recall_target_for_auto_safe": GATE_VALIDATION_RECALL_TARGET,
        "validation_precision_target_for_auto_damage": AUTO_DAMAGE_PRECISION_TARGET,
        "low_auto_safe_threshold": low,
        "high_auto_damage_threshold": float(high) if math.isfinite(high) else None,
        "validation": validation_selective,
        **test_selective,
    }
    artifacts = [
        _artifact(paths["metrics"] / f"scores_puerta_{split}.npy")
        for split in context["frames"]
    ] + [
        _artifact(paths["metrics"] / f"scores_condicionales_{split}.npy")
        for split in ("validation", "test")
    ]
    return _finalize(
        key,
        context,
        combined_scores["validation"],
        combined_scores["test"],
        thresholds,
        {"gate": gate_training, "conditional_categories": stage_training},
        paths,
        bootstrap_replicates,
        selective_operation=selective,
        extra_artifacts=artifacts,
        same_training_rows_as_flat=not expanded_safe_gate,
    )


def run_joint_experiment(
    force: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Entrena, evalúa y documenta el Transformer jerárquico conjunto."""
    context = load_frozen_context()
    cached = _cached_result(JOINT_KEY, context, force, bootstrap_replicates)
    if cached is not None:
        return cached
    paths = _experiment_paths(JOINT_KEY)
    _ensure_directories(paths)
    reference = context["reference"]
    spec = reference["spec"]
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    tokenizer.save_pretrained(paths["models"] / "tokenizer")
    training_frames, training_data = joint_training_frames(context)
    training, model = _train_joint(
        spec,
        training_frames,
        tokenizer,
        paths,
        context["dataset_sha256"],
        reference["model_key"],
    )
    training["training_data"] = training_data
    device = _device()
    gate_scores, combined_scores, artifacts = {}, {}, []
    for split in ("validation", "test"):
        gate, conditional = _predict_joint(
            model,
            _loader(
                context["frames"][split], tokenizer, spec.prefix, EVAL_BATCH_SIZE, False
            ),
            device,
            f"jerárquico conjunto · {split}",
        )
        combined = gate[:, None] * conditional
        gate_scores[split] = gate
        combined_scores[split] = combined
        for name, array in (
            (f"scores_puerta_{split}.npy", gate),
            (f"scores_condicionales_{split}.npy", conditional),
        ):
            path = paths["metrics"] / name
            np.save(path, array)
            artifacts.append(_artifact(path))
    y_validation = experiment_targets(context["frames"]["validation"])
    thresholds = tune_thresholds(y_validation.astype(np.int8), combined_scores["validation"])
    # La cabeza binaria también produce una política de abstención comparable.
    low = _threshold_for_recall(
        y_validation.any(axis=1), gate_scores["validation"], GATE_VALIDATION_RECALL_TARGET
    )
    high = _threshold_for_precision(
        y_validation.any(axis=1),
        gate_scores["validation"],
        low,
        AUTO_DAMAGE_PRECISION_TARGET,
    )
    validation_selective = _selective_gate_metrics(
        y_validation,
        gate_scores["validation"],
        combined_scores["validation"],
        thresholds,
        low,
        high,
    )
    test_selective = _selective_gate_metrics(
        experiment_targets(context["frames"]["test"]),
        gate_scores["test"],
        combined_scores["test"],
        thresholds,
        low,
        high,
    )
    selective = {
        "calibration_partition": "validation",
        "validation_recall_target_for_auto_safe": GATE_VALIDATION_RECALL_TARGET,
        "validation_precision_target_for_auto_damage": AUTO_DAMAGE_PRECISION_TARGET,
        "low_auto_safe_threshold": low,
        "high_auto_damage_threshold": float(high) if math.isfinite(high) else None,
        "validation": validation_selective,
        **test_selective,
    }
    return _finalize(
        JOINT_KEY,
        context,
        combined_scores["validation"],
        combined_scores["test"],
        thresholds,
        training,
        paths,
        bootstrap_replicates,
        selective_operation=selective,
        extra_artifacts=artifacts,
    )


def load_experiment_tables(key: str) -> dict[str, pd.DataFrame]:
    """Recupera resultados guardados sin reentrenar."""
    if key not in EXPERIMENT_LABELS:
        raise KeyError(f"Experimento desconocido: {key}")
    paths = _experiment_paths(key)
    required = (paths["result"], paths["comparison"], paths["categories"], paths["bootstrap"])
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan resultados: " + ", ".join(str(path) for path in missing))
    return {
        "comparison": pd.read_csv(paths["comparison"]),
        "categories": pd.read_csv(paths["categories"]),
        "bootstrap": pd.read_csv(paths["bootstrap"]),
    }


def result_path(key: str) -> Path:
    return _experiment_paths(key)["result"]


def report_path(key: str) -> Path:
    return _experiment_paths(key)["report"]
