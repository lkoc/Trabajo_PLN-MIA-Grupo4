"""Variantes de 04_4 y 04_5 con cuatro categorías de daño.

La implementación conserva los splits y el protocolo estadístico originales,
pero fusiona ACOSO_PERSONAL y AMENAZA_DIRECTA. Por defecto reutiliza pesos de
un Transformer ya ajustado: primero un checkpoint jerárquico compatible, si
existe; en caso contrario, el encoder plano de 04_2 seleccionado sólo con
validación. Las nuevas cabezas se optimizan de nuevo.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl, sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import tune_thresholds


ROOT = tm.ROOT
SOURCE_DAMAGE_LABELS = list(tm.DAMAGE_ORDER)
TARGET_LABELS = [
    "RACISMO_DISCRIMINACION",
    "ACOSO_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
]
SEMANTIC_LABELS = ["SEGURO", *TARGET_LABELS]

CASCADE_KEY = "cascada_binaria_multietiqueta_4"
CASCADE_EXTRA_SAFE_KEY = "cascada_binaria_multietiqueta_4_seguros_ampliados"
JOINT_KEY = "transformer_jerarquico_multitarea_4"
EXPERIMENT_LABELS = {
    CASCADE_KEY: "Cascada binaria → cuatro daños",
    CASCADE_EXTRA_SAFE_KEY: "Cascada binaria → cuatro daños, SEGURO ampliado",
    JOINT_KEY: "Transformer jerárquico multitarea → cuatro daños",
}

METRICS_ROOT = ROOT / "resultados" / "metricas" / "experimentos_jerarquicos_4"
FIGURES_ROOT = ROOT / "resultados" / "figuras" / "experimentos_jerarquicos_4"
MODEL_ROOT = ROOT / "modelos" / "experimentos_jerarquicos_4"
REPORT_ROOT = ROOT / "resultados"
for _directory in (METRICS_ROOT, FIGURES_ROOT, MODEL_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)


def _load_isolated_runtime():
    """Carga el motor original bajo otro nombre para no alterar 04_4/04_5."""
    name = "scripts_auxiliares._experimentos_jerarquicos_4_runtime"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("experimentos_jerarquicos.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_runtime = _load_isolated_runtime()


def four_targets(frame: pd.DataFrame) -> np.ndarray:
    source = tm.damage_targets(frame).astype(np.float32)
    if source.shape[1] != 5:
        raise ValueError("La taxonomía fuente debe contener cinco daños.")
    return np.column_stack(
        [source[:, 0], source[:, 1], np.maximum(source[:, 2], source[:, 3]), source[:, 4]]
    ).astype(np.float32)


def merge_five_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"Se esperaban scores (n, 5), no {values.shape}.")
    # No se conoce la dependencia entre ambas probabilidades. El máximo evita
    # imponer una independencia no demostrada; el umbral se recalibra después.
    merged = np.maximum(values[:, 2], values[:, 3])
    return np.column_stack([values[:, 0], values[:, 1], merged, values[:, 4]])


def evaluate_four_scores(
    y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if y.shape != scores.shape or y.shape[1] != len(TARGET_LABELS):
        raise ValueError("Targets y scores no corresponden a cuatro categorías.")
    categories = scores >= thresholds
    true_safe = (~y.astype(bool).any(axis=1)).astype(np.int8)[:, None]
    predicted_safe = (~categories.any(axis=1)).astype(np.int8)[:, None]
    true_semantic = np.column_stack([true_safe, y])
    predicted_semantic = np.column_stack([predicted_safe, categories.astype(np.int8)])
    semantic_scores = np.column_stack([1.0 - scores.max(axis=1), scores])
    true_any = y.astype(bool).any(axis=1)
    predicted_any = categories.any(axis=1)
    category_recall = {
        label: float(recall_score(y[:, index], categories[:, index], zero_division=0))
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
        "damage_pr_auc_macro": float(average_precision_score(y, scores, average="macro")),
        "damage_precision_micro": float(
            precision_score(y, categories, average="micro", zero_division=0)
        ),
        "damage_recall_micro": float(
            recall_score(y, categories, average="micro", zero_division=0)
        ),
        "damage_f1_micro": float(
            f1_score(y, categories, average="micro", zero_division=0)
        ),
        "damage_f1_macro": float(
            f1_score(y, categories, average="macro", zero_division=0)
        ),
        "any_damage_precision": float(
            precision_score(true_any, predicted_any, zero_division=0)
        ),
        "any_damage_recall": float(recall_score(true_any, predicted_any, zero_division=0)),
        "any_damage_f1": float(f1_score(true_any, predicted_any, zero_division=0)),
        "missed_damage_as_safe": int((true_any & ~predicted_any).sum()),
        "category_recall": category_recall,
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
    return metrics, report, categories.astype(np.int8)


def _experiment_paths(key: str) -> dict[str, Path]:
    if key not in EXPERIMENT_LABELS:
        raise KeyError(f"Experimento desconocido: {key}")
    report_name = (
        "INFORME_04_203_CASCADA_4_ETIQUETAS.md"
        if key in (CASCADE_KEY, CASCADE_EXTRA_SAFE_KEY)
        else "INFORME_04_204_JERARQUICO_MULTITAREA_4_ETIQUETAS.md"
    )
    return {
        "metrics": METRICS_ROOT / key,
        "figures": FIGURES_ROOT / key,
        "models": MODEL_ROOT / key,
        "result": METRICS_ROOT / key / "resultado.json",
        "comparison": METRICS_ROOT / key / "comparacion_modelo_plano.csv",
        "categories": METRICS_ROOT / key / "comparacion_recall_por_categoria.csv",
        "bootstrap": METRICS_ROOT / key / "bootstrap_pareado_por_video.csv",
        "report": REPORT_ROOT / report_name,
    }


def expanded_safe_gate_training_frame(context: dict) -> tuple[pd.DataFrame, dict]:
    """Usa todo SEGURO cuyo video no pertenezca a validation/test.

    A diferencia del control histórico, incorpora también videos seguros que
    quedaron sin mapa en el submuestreo 4:1. Son train-only y por ello no
    contaminan las particiones de evaluación.
    """
    manifest = json.loads(context["manifest_path"].read_text(encoding="utf-8"))
    integrated_path = ROOT / manifest["input_integrated_dataset"]
    expected_hash = manifest.get("input_integrated_sha256")
    if not integrated_path.exists() or sha256_file(integrated_path) != expected_hash:
        raise ValueError("El dataset integrado completo está ausente o cambió de hash.")
    integrated = pd.DataFrame(read_jsonl(integrated_path))
    integrated["video_id"] = integrated["video_id"].astype(str)
    train = context["frames"]["train"].copy()
    train["video_id"] = train["video_id"].astype(str)
    evaluation_videos = set(
        pd.concat(
            [context["frames"]["validation"], context["frames"]["test"]],
            ignore_index=True,
        )["video_id"].astype(str)
    )
    train_videos = set(train["video_id"])
    frozen_ids = set(
        pd.concat(list(context["frames"].values()), ignore_index=True)["chunk_id"].astype(str)
    )
    safe = ~four_targets(integrated).astype(bool).any(axis=1)
    candidates = integrated.loc[
        safe
        & ~integrated["video_id"].isin(evaluation_videos)
        & ~integrated["chunk_id"].astype(str).isin(frozen_ids)
    ].copy()
    candidates["split"] = "train"
    expanded = pd.concat([train, candidates], ignore_index=True, sort=False)
    if expanded["chunk_id"].duplicated().any():
        raise AssertionError("La ampliación SEGURO produjo chunks duplicados.")
    if set(expanded["video_id"].astype(str)) & evaluation_videos:
        raise AssertionError("La ampliación introdujo videos de validation/test.")
    if four_targets(candidates).astype(bool).any():
        raise AssertionError("La ampliación de la puerta contiene daño.")
    original_safe = int((~four_targets(train).astype(bool).any(axis=1)).sum())
    expanded_safe = int((~four_targets(expanded).astype(bool).any(axis=1)).sum())
    metadata = {
        "purpose": "binary_gate_negative_diversity_without_evaluation_video_leakage",
        "selection": "all integrated SEGURO excluding validation/test videos",
        "integrated_dataset": _runtime._relative(integrated_path),
        "integrated_dataset_sha256": expected_hash,
        "matched_train_rows": int(len(train)),
        "matched_train_safe_rows": original_safe,
        "additional_safe_rows": int(len(candidates)),
        "additional_safe_from_mapped_train_videos": int(
            candidates["video_id"].isin(train_videos).sum()
        ),
        "additional_safe_from_unassigned_train_only_videos": int(
            (~candidates["video_id"].isin(train_videos)).sum()
        ),
        "expanded_gate_train_rows": int(len(expanded)),
        "expanded_gate_safe_rows": expanded_safe,
        "expanded_gate_damage_rows": int(len(expanded) - expanded_safe),
        "expanded_gate_train_videos": int(expanded["video_id"].nunique()),
        "additional_chunk_ids_sha256": _runtime._ids_sha256(candidates["chunk_id"]),
        "validation_or_test_videos_used": False,
    }
    return expanded, metadata


def joint_training_frames(
    context: dict,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Amplía la cabeza binaria y enmascara categorías en SEGURO adicional."""
    expanded, metadata = expanded_safe_gate_training_frame(context)
    base_rows = len(context["frames"]["train"])
    expanded = expanded.copy()
    expanded["category_loss_mask"] = 0.0
    expanded.loc[: base_rows - 1, "category_loss_mask"] = 1.0
    if int(expanded["category_loss_mask"].sum()) != base_rows:
        raise AssertionError("Máscara de la pérdida temática inconsistente.")
    frames = {split: frame.copy() for split, frame in context["frames"].items()}
    frames["train"] = expanded
    return frames, {
        **metadata,
        "purpose": "expanded_binary_supervision_with_masked_category_loss",
        "binary_training_rows": int(len(expanded)),
        "binary_training_safe_rows": metadata["expanded_gate_safe_rows"],
        "category_training_rows": int(base_rows),
        "additional_safe_binary_only_rows": int(len(expanded) - base_rows),
        "category_loss_on_additional_safe": False,
    }


def _select_flat_reference(registry: dict, frames: dict[str, pd.DataFrame]) -> dict:
    """Selecciona el Transformer plano por PR-AUC de cuatro daños en validación."""
    candidates = []
    y_validation = four_targets(frames["validation"]).astype(np.int8)
    for item in registry["models"]:
        if item.get("family") != "transformer_full_finetuning":
            continue
        key = item["model_key"]
        if key not in tm.MODEL_SPECS:
            continue
        evaluation_path = ROOT / item["evaluation"]["path"]
        validation_path = tm.METRICS_DIR / f"scores_{key}_validation.npy"
        test_path = tm.METRICS_DIR / f"scores_{key}_test.npy"
        checkpoint_path = ROOT / item["artifact"]["path"]
        for path in (evaluation_path, validation_path, test_path, checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(f"Falta artefacto de 04_2: {path}")
        if sha256_file(checkpoint_path) != item["artifact"]["sha256"]:
            raise ValueError(f"Cambió el checkpoint de 04_2: {checkpoint_path}")
        validation_scores = merge_five_scores(np.load(validation_path))
        test_scores = merge_five_scores(np.load(test_path))
        if validation_scores.shape != (len(frames["validation"]), 4):
            raise ValueError(f"Dimensión incompatible en {validation_path}.")
        if test_scores.shape != (len(frames["test"]), 4):
            raise ValueError(f"Dimensión incompatible en {test_path}.")
        thresholds = tune_thresholds(y_validation, validation_scores)
        metrics, _, _ = evaluate_four_scores(y_validation, validation_scores, thresholds)
        candidates.append(
            {
                "model_key": key,
                "model_label": f"{item['model_label']} · unión post hoc a cuatro daños",
                "family": item["family"],
                "spec": tm.MODEL_SPECS[key],
                "evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
                "evaluation_artifact": _runtime._artifact(evaluation_path),
                "validation_score_artifact": _runtime._artifact(validation_path),
                "test_score_artifact": _runtime._artifact(test_path),
                "warm_start_artifact": _runtime._artifact(checkpoint_path),
                "validation_scores": validation_scores,
                "test_scores": test_scores,
                "thresholds": thresholds,
                "validation_damage_pr_auc_macro": metrics["damage_pr_auc_macro"],
            }
        )
    if not candidates:
        raise RuntimeError("04_2 no tiene un Transformer plano evaluado.")
    candidates.sort(
        key=lambda row: (row["validation_damage_pr_auc_macro"], row["model_key"]),
        reverse=True,
    )
    return candidates[0]


def _mapped_head(source_state: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    weights = source_state[f"{prefix}.weight"]
    bias = source_state[f"{prefix}.bias"]
    if weights.shape[0] != 5 or bias.shape[0] != 5:
        raise ValueError("La cabeza fuente no tiene cinco daños.")
    mapped_weights = torch.stack(
        [weights[0], weights[1], (weights[2] + weights[3]) / 2.0, weights[4]]
    )
    mapped_bias = torch.stack([bias[0], bias[1], (bias[2] + bias[3]) / 2.0, bias[4]])
    return mapped_weights, mapped_bias


def _copy_head(target, source_state: dict, prefix: str) -> None:
    weights, bias = _mapped_head(source_state, prefix)
    with torch.no_grad():
        target.weight.copy_(weights.to(device=target.weight.device, dtype=target.weight.dtype))
        target.bias.copy_(bias.to(device=target.bias.device, dtype=target.bias.dtype))


WARM_START_ENABLED = True


def _old_checkpoint_candidates(role: str) -> list[Path]:
    old_root = ROOT / "modelos" / "experimentos_jerarquicos"
    if role == "binary_gate":
        return [
            old_root / "cascada_binaria_multietiqueta_seguros_ampliados" / "gate_best.pt",
            old_root / "cascada_binaria_multietiqueta" / "gate_best.pt",
        ]
    if role == "conditional_categories":
        return [
            old_root
            / "cascada_binaria_multietiqueta_seguros_ampliados"
            / "categorias_best.pt",
            old_root / "cascada_binaria_multietiqueta" / "categorias_best.pt",
        ]
    if role == "joint_hierarchical":
        return [old_root / "transformer_jerarquico_multitarea" / "jerarquico_best.pt"]
    return []


def _valid_old_checkpoint(
    path: Path, spec: tm.EncoderSpec, dataset_sha256: str
) -> dict | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_spec = checkpoint.get("model_spec", {})
    if (
        checkpoint.get("dataset_sha256") != dataset_sha256
        or source_spec.get("model_id") != spec.model_id
        or source_spec.get("revision") != spec.revision
    ):
        return None
    return checkpoint


def _flat_checkpoint(reference_key: str, spec: tm.EncoderSpec) -> tuple[Path, dict]:
    path = tm.MODEL_DIR / reference_key / "best_checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint plano: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_spec = checkpoint.get("model_spec", {})
    if (
        source_spec.get("model_id") != spec.model_id
        or source_spec.get("revision") != spec.revision
        or checkpoint.get("targets") != SOURCE_DAMAGE_LABELS
    ):
        raise ValueError("El checkpoint plano no coincide con encoder/taxonomía esperados.")
    return path, checkpoint


def _initialize_model(
    model,
    role: str,
    spec: tm.EncoderSpec,
    reference_key: str,
    dataset_sha256: str,
) -> dict:
    if not WARM_START_ENABLED:
        return {
            "enabled": False,
            "role": role,
            "strategy": "hub_pretrained_encoder_and_random_task_head",
        }
    for path in _old_checkpoint_candidates(role):
        checkpoint = _valid_old_checkpoint(path, spec, dataset_sha256)
        if checkpoint is None:
            continue
        state = checkpoint["model_state"]
        if role == "binary_gate":
            model.load_state_dict(state, strict=True)
            copied = ["backbone", "binary_head"]
        elif role == "conditional_categories":
            _copy_head(model.classifier, state, "classifier")
            copied = ["mapped_category_head"]
        else:
            compatible = {
                key: value for key, value in state.items() if not key.startswith("category_head.")
            }
            model.load_state_dict(compatible, strict=False)
            _copy_head(model.category_head, state, "category_head")
            copied = ["backbone", "binary_head", "mapped_category_head"]
        return {
            "enabled": True,
            "role": role,
            "strategy": "warm_start_from_five_label_hierarchical_checkpoint",
            "source": _runtime._artifact(path),
            "copied": copied,
            "head_merge": "mean of ACOSO_PERSONAL and AMENAZA_DIRECTA rows",
        }

    path, checkpoint = _flat_checkpoint(reference_key, spec)
    state = checkpoint["model_state"]
    copied = []
    if role in ("binary_gate", "joint_hierarchical"):
        backbone = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone, strict=True)
        copied.append("domain_finetuned_backbone")
    if role == "conditional_categories":
        _copy_head(model.classifier, state, "classifier")
        copied.append("mapped_category_head")
    elif role == "joint_hierarchical":
        _copy_head(model.category_head, state, "classifier")
        copied.append("mapped_category_head")
    return {
        "enabled": True,
        "role": role,
        "strategy": "warm_start_from_04_2_flat_transformer",
        "source": _runtime._artifact(path),
        "source_epoch": int(checkpoint["epoch"]),
        "copied": copied,
        "head_merge": "mean of ACOSO_PERSONAL and AMENAZA_DIRECTA rows",
        "random_components": (
            ["binary_head"] if role in ("binary_gate", "joint_hierarchical") else []
        ),
    }


def warm_start_plan(context: dict | None = None) -> dict:
    context = context or load_frozen_context()
    reference = context["reference"]
    return {
        "enabled_by_default": True,
        "selected_by_validation": reference["model_key"],
        "selected_model_label": reference["model_label"],
        "validation_damage_pr_auc_macro_four_labels": reference[
            "validation_damage_pr_auc_macro"
        ],
        "checkpoint": reference["warm_start_artifact"],
        "reuse": {
            "encoder": "copied from the selected 04_2 checkpoint",
            "unchanged_category_rows": [0, 1, 4],
            "ACOSO_AMENAZA_row": "mean of former rows 2 and 3; retrained afterwards",
            "binary_head": "random unless a compatible old hierarchical checkpoint exists",
            "thresholds": "never reused; recalibrated on validation",
        },
        "old_hierarchical_checkpoints_present": {
            role: [str(path.relative_to(ROOT)) for path in _old_checkpoint_candidates(role) if path.exists()]
            for role in ("binary_gate", "conditional_categories", "joint_hierarchical")
        },
    }


def _write_report(
    result: dict,
    comparison: pd.DataFrame,
    categories: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    key = result["experiment_key"]
    test_rows = "\n".join(
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | {row.damage_f1_macro:.4f} | "
        f"{row.any_damage_recall:.4f} | {int(row.missed_damage_as_safe)} |"
        for row in comparison.loc[comparison["split"].eq("test")].itertuples()
    )
    category_rows = "\n".join(
        f"| {row.categoria} | {int(row.positivos_test)} | {row.recall_plano:.4f} | "
        f"{row.recall_experimento:.4f} | {row.delta_recall:+.4f} |"
        for row in categories.itertuples()
    )
    bootstrap_rows = "\n".join(
        f"| {row.metrica} | {row.delta_experimento_menos_plano:+.4f} | "
        f"[{row.ic95_inferior:+.4f}, {row.ic95_superior:+.4f}] |"
        for row in bootstrap.itertuples()
    )
    training = result["training"]
    initialization = (
        training["gate"]["initialization"]
        if "gate" in training
        else training["initialization"]
    )
    selective = result.get("selective_operation") or {}
    report = f"""# {EXPERIMENT_LABELS[key]}

Fecha: {tm.now_iso()}

## Diseño y transferencia

El experimento conserva los splits de `04_2` y entrena cuatro daños: {', '.join(TARGET_LABELS)}. `SEGURO` se deriva si no se activa daño. `ACOSO_AMENAZA` es la unión de `ACOSO_PERSONAL` y `AMENAZA_DIRECTA`. Las etiquetas finas y flags no son predictores ni objetivos de este experimento.

Inicialización: `{initialization['strategy']}`. Se reutilizan representaciones del Transformer previamente afinado, pero las cabezas nuevas se optimizan con la taxonomía de cuatro daños. Para inicializar `ACOSO_AMENAZA` se promedian las antiguas filas de acoso y amenaza; esto es sólo un punto inicial, no la decisión final. Los umbrales anteriores no se reutilizan: se estiman exclusivamente en validación.

## Resultado en el mismo test

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{test_rows}

| Categoría | Positivos | Recall plano | Recall jerárquico | Δ recall |
|---|---:|---:|---:|---:|
{category_rows}

## Inferencia pareada por video

| Métrica | Δ experimento − plano | IC 95 % bootstrap |
|---|---:|---:|
{bootstrap_rows}

La regla predeclarada concluyó `{result['decision']['status']}`. Reemplazar la referencia plana: **{'sí' if result['decision']['replace_flat_model'] else 'no'}**. En test, la política selectiva envía {selective.get('review_rate', math.nan):.2%} a revisión y deja {selective.get('damage_automatic_safe_false_negatives', 'NA')} daños como auto-seguros. Esto no autoriza operación autónoma sin gold standard humano independiente y piloto prospectivo.

## Artefactos

- Resultado: `{_runtime._relative(paths['result'])}`
- Checkpoints: `{_runtime._relative(paths['models'])}`
- Comparación: `{_runtime._relative(paths['comparison'])}`
- Bootstrap: `{_runtime._relative(paths['bootstrap'])}`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Zhou, J., Ma, C., Long, D., Xu, G., Ding, N., Zhang, H., Xie, P., & Liu, G. (2020). Hierarchy-aware global model for hierarchical text classification. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1106–1117). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.104
"""
    paths["report"].write_text(report, encoding="utf-8")


# Configura el runtime aislado. Las funciones originales quedan intactas en su módulo.
_runtime.DAMAGE_ORDER = TARGET_LABELS
_runtime.CASCADE_KEY = CASCADE_KEY
_runtime.CASCADE_EXTRA_SAFE_KEY = CASCADE_EXTRA_SAFE_KEY
_runtime.JOINT_KEY = JOINT_KEY
_runtime.EXPERIMENT_LABELS = EXPERIMENT_LABELS
_runtime.METRICS_ROOT = METRICS_ROOT
_runtime.FIGURES_ROOT = FIGURES_ROOT
_runtime.MODEL_ROOT = MODEL_ROOT
_runtime.REPORT_ROOT = REPORT_ROOT
_runtime.experiment_targets = four_targets
_runtime.evaluate_experiment_scores = evaluate_four_scores
_runtime.initialize_experiment_model = _initialize_model
_runtime._select_flat_reference = _select_flat_reference
_runtime._experiment_paths = _experiment_paths
_runtime._write_report = _write_report
_runtime.expanded_safe_gate_training_frame = expanded_safe_gate_training_frame
_runtime.joint_training_frames = joint_training_frames

_base_cached_result = _runtime._cached_result


def _cached_result(
    key: str, context: dict, force: bool, bootstrap_replicates: int
) -> dict | None:
    result = _base_cached_result(key, context, force, bootstrap_replicates)
    if result is None:
        return None
    if result.get("dataset", {}).get("targets") != TARGET_LABELS:
        raise ValueError("El resultado cacheado no corresponde a cuatro etiquetas.")
    training = result["training"]
    initialization = (
        training["gate"]["initialization"]
        if "gate" in training
        else training["initialization"]
    )
    if bool(initialization.get("enabled")) != bool(WARM_START_ENABLED):
        raise ValueError("Cambió warm_start; ejecute con force=True.")
    return result


_runtime._cached_result = _cached_result


def load_frozen_context() -> dict:
    return _runtime.load_frozen_context()


def context_summary(context: dict | None = None) -> pd.DataFrame:
    return _runtime.context_summary(context)


def run_cascade_experiment(
    force: bool = False,
    bootstrap_replicates: int = tm.BOOTSTRAP_REPLICATES,
    expanded_safe_gate: bool = True,
    warm_start: bool = True,
) -> dict:
    global WARM_START_ENABLED
    WARM_START_ENABLED = bool(warm_start)
    return _runtime.run_cascade_experiment(
        force=force,
        bootstrap_replicates=bootstrap_replicates,
        expanded_safe_gate=expanded_safe_gate,
    )


def run_joint_experiment(
    force: bool = False,
    bootstrap_replicates: int = tm.BOOTSTRAP_REPLICATES,
    warm_start: bool = True,
) -> dict:
    global WARM_START_ENABLED
    WARM_START_ENABLED = bool(warm_start)
    return _runtime.run_joint_experiment(
        force=force, bootstrap_replicates=bootstrap_replicates
    )


def load_experiment_tables(key: str) -> dict[str, pd.DataFrame]:
    return _runtime.load_experiment_tables(key)


def result_path(key: str) -> Path:
    return _experiment_paths(key)["result"]


def report_path(key: str) -> Path:
    return _experiment_paths(key)["report"]


def device() -> torch.device:
    return _runtime._device()
