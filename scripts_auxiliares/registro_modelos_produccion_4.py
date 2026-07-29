"""Registro verificable de los tres modelos consumidos por el cuaderno 05."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from scripts_auxiliares import analizar_auxiliares_modelos_4 as audit4
from scripts_auxiliares import entrenar_qwen_acoso_amenaza as q4
from scripts_auxiliares import entrenar_transformers_planos_4 as t4
from scripts_auxiliares import experimentos_jerarquicos_4 as h4
from scripts_auxiliares import experimentos_jerarquicos_clasicos_4 as c4
from scripts_auxiliares.flujo_hibrido_moderador import sha256_file


ROOT = q4.ROOT
REGISTRY_PATH = (
    ROOT
    / "resultados"
    / "metricas"
    / "comparacion_final_4"
    / "registro_modelos_desplegables.json"
)
TARGET_LABELS = list(q4.TARGET_LABELS)


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Falta artefacto desplegable: {path}")
    return {
        "path": q4.tm.project_relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _directory_artifacts(path: Path) -> list[dict]:
    if not path.is_dir():
        raise FileNotFoundError(f"Falta directorio desplegable: {path}")
    return [_artifact(item) for item in sorted(path.rglob("*")) if item.is_file()]


def _review_policy(
    validation_scores: np.ndarray,
    thresholds: np.ndarray,
    validation_targets: np.ndarray,
) -> dict:
    scores = np.asarray(validation_scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    risk_cutoff, routing = q4.tm.tune_human_alert_cutoff(
        validation_targets.astype(np.int8),
        scores,
        thresholds,
        recall_target=q4.ALERT_RECALL_TARGET,
    )
    distances = np.min(np.abs(scores - thresholds), axis=1)
    uncertainty_cutoff = float(np.quantile(distances, 0.20))
    return {
        "selection_partition": "validation",
        "test_used": False,
        "damage_recall_target": q4.ALERT_RECALL_TARGET,
        "risk_definition": "max(score - category_threshold)",
        "risk_margin_cutoff": float(risk_cutoff),
        "validation_routing": routing,
        "uncertainty_definition": "min(abs(score - category_threshold))",
        "uncertainty_margin_cutoff": uncertainty_cutoff,
        "requires_review_when": (
            "risk_margin >= risk_margin_cutoff OR uncertainty_margin < "
            "uncertainty_margin_cutoff"
        ),
        "meaning_of_high_confidence": (
            "no activa la ruta de alto recall y no está en el 20% de márgenes "
            "más cercanos a los umbrales de validation"
        ),
    }


def _ensure_transformer_config(model_key: str, model_directory: Path) -> dict | None:
    config_path = model_directory / "config.json"
    if not config_path.is_file():
        try:
            from transformers import AutoConfig

            spec = q4.tm.MODEL_SPECS[model_key]
            try:
                config = AutoConfig.from_pretrained(
                    spec.model_id,
                    revision=spec.revision,
                    local_files_only=True,
                )
            except Exception:
                config = AutoConfig.from_pretrained(
                    spec.model_id,
                    revision=spec.revision,
                )
            config.save_pretrained(model_directory)
        except Exception:
            return None
    return _artifact(config_path) if config_path.is_file() else None


def build_registry() -> dict:
    """Selecciona por validation y publica rutas/hashes para el servidor 05."""
    required = [c4.RESULT_PATH, c4.COMMON_4A1_PATH, t4.RESULT_PATH]
    missing = [q4.tm.project_relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "No se puede publicar el registro; faltan resultados 04:\n"
            + "\n".join(missing)
        )

    frames, dataset_audit = q4.load_frames()
    validation_targets = h4.four_targets(frames["validation"]).astype(np.int8)
    dataset_sha256 = dataset_audit["dataset_sha256"]

    classic_result = json.loads(c4.RESULT_PATH.read_text(encoding="utf-8"))
    common = pd.read_csv(c4.COMMON_4A1_PATH)
    classic_key = classic_result["common_4a1_evaluation"]["winner_key"]
    classic_row = common.loc[
        common["model_key"].eq(classic_key) & common["split"].eq("validation")
    ].iloc[0]
    classic_record = classic_result["models"][classic_key]
    classic_short = classic_key.split("__", 1)[0]
    classic_dir = c4.MODEL_DIR / classic_short
    classic_thresholds = np.asarray(
        json.loads(classic_row["thresholds_selected_on_common_validation"]),
        dtype=float,
    )
    classic_scores_path = c4.OUTPUT_DIR / f"scores_{classic_key}_validation_4a1.npy"

    transformer_result = json.loads(t4.RESULT_PATH.read_text(encoding="utf-8"))
    transformer_key = transformer_result["selection"]["winner_key"]
    transformer_record = transformer_result["models"][transformer_key]
    transformer_dir = t4.MODEL_DIR / transformer_key
    transformer_thresholds = np.asarray(
        transformer_record["thresholds_selected_on_validation"], dtype=float
    )
    transformer_scores_path = ROOT / transformer_record["score_artifacts"]["validation"]["path"]
    transformer_spec = transformer_record["model"]
    config_artifact = _ensure_transformer_config(transformer_key, transformer_dir)

    qwen = q4.load_operational_evaluation(load_scores=True, require_test=True)
    qwen_calibrator = ROOT / qwen["selected_candidate"]["calibrator"]
    qwen_tokenizer = q4.MODEL_DIR / "tokenizer"
    qwen_thresholds = np.asarray(qwen["thresholds_selected_on_validation"], dtype=float)

    models = {
        "classical": {
            "family": "classical_ml",
            "model_key": classic_key,
            "label": classic_record["label"],
            "selected_by": "maximum validation damage_pr_auc_macro within classical family",
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "design": classic_record["design"],
            "thresholds": classic_thresholds.tolist(),
            "validation_damage_pr_auc_macro": float(classic_row["damage_pr_auc_macro"]),
            "artifacts": {
                "vectorizer": _artifact(classic_dir / "vectorizador.joblib"),
                "model_bundle": _artifact(classic_dir / "modelos_calibrados.joblib"),
                "validation_scores": _artifact(classic_scores_path),
            },
            "review_policy": _review_policy(
                np.load(classic_scores_path), classic_thresholds, validation_targets
            ),
        },
        "transformer": {
            "family": "minilm_transformer",
            "model_key": transformer_key,
            "label": transformer_record["model"]["label"],
            "selected_by": "maximum validation damage_pr_auc_macro within 04_202",
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "model_spec": transformer_spec,
            "thresholds": transformer_thresholds.tolist(),
            "validation_damage_pr_auc_macro": float(
                transformer_record["metrics"]["validation"]["damage_pr_auc_macro"]
            ),
            "artifacts": {
                "checkpoint": transformer_record["checkpoint"],
                "tokenizer": _directory_artifacts(transformer_dir / "tokenizer"),
                "config": config_artifact,
                "validation_scores": transformer_record["score_artifacts"]["validation"],
            },
            "review_policy": _review_policy(
                np.load(transformer_scores_path),
                transformer_thresholds,
                validation_targets,
            ),
        },
        "qwen": {
            "family": "finetuned_qwen",
            "model_key": "qwen4_flat",
            "label": f"Qwen3-0.6B LoRA · época operativa {qwen['selected_epoch']}",
            "selected_by": qwen["selection"]["selection_rule"],
            "selection_partition": "validation",
            "test_used_for_selection": False,
            "model_spec": qwen["training"]["model"],
            "selected_epoch": qwen["selected_epoch"],
            "thresholds": qwen_thresholds.tolist(),
            "validation_damage_pr_auc_macro": float(
                qwen["metrics"]["validation"]["damage_pr_auc_macro"]
            ),
            "artifacts": {
                # PEFT necesita adapter_config.json además de los pesos. Se
                # registra el directorio completo para que un bundle offline
                # no dependa de archivos implícitos del workspace original.
                "adapter": _directory_artifacts(
                    ROOT / qwen["selected_adapter"]
                ),
                "adapter_state": qwen["artifacts"]["adapter_training_state"],
                "calibrator": _artifact(qwen_calibrator),
                "tokenizer": _directory_artifacts(qwen_tokenizer),
                "validation_scores": qwen["artifacts"]["validation_scores"],
                "selection": qwen["artifacts"]["selection"],
            },
            "selected_adapter": qwen["selected_adapter"],
            "review_policy": _review_policy(
                qwen["scores"]["validation"], qwen_thresholds, validation_targets
            ),
        },
    }
    registry = {
        "schema_version": "1.0",
        "generated_at": q4.tm.now_iso(),
        "purpose": "verified local inference registry for notebook 05",
        "dataset_sha256": dataset_sha256,
        "target_labels": TARGET_LABELS,
        "safe_is_derived": True,
        "global_selection_partition": "validation",
        "test_used_for_model_selection": False,
        "chunking": {
            "target_seconds": 30,
            "max_characters": 600,
            "minimum_characters_for_video": 90,
            "rolling_caption_overlap_words": 12,
            "matches_pretraining_pipeline": True,
        },
        "models": models,
        "comparison_mode": {
            "returns_each_model_separately": True,
            "consensus_rule": "category accepted when at least two of three models activate it",
            "requires_review_on_any_model_disagreement": True,
        },
        "audit_result": q4.tm.project_relative(
            audit4.OUTPUT_DIR / "resultado.json"
        ),
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    q4.tm.write_json(REGISTRY_PATH, registry)
    return registry


def load_registry(*, verify_hashes: bool = True) -> dict:
    if not REGISTRY_PATH.is_file():
        return build_registry()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if verify_hashes:
        for model in registry["models"].values():
            for artifact in model["artifacts"].values():
                values = artifact if isinstance(artifact, list) else [artifact]
                for item in values:
                    if item is None:
                        continue
                    path = ROOT / item["path"]
                    if not path.is_file() or sha256_file(path) != item["sha256"]:
                        raise RuntimeError(
                            f"Artefacto desplegable ausente o modificado: {item['path']}"
                        )
    return registry
