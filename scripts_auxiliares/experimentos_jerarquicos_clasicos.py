"""Modelos clásicos planos y jerárquicos sobre el corpus SEGURO ampliado.

El módulo conserva el mapa de videos train/validation/test congelado por 04_2
y propaga ese mapa al dataset integrado completo. Así usa 109 mil chunks
SEGURO sin introducir videos de validación o test en el entrenamiento.

Para cada familia ganadora del experimento plano (SVM lineal y regresión
logística) entrena tres diseños sobre el mismo TF-IDF y los mismos datos:

* plano: cinco cabezas independientes;
* cascada: puerta daño/seguro y cinco cabezas entrenadas con daño + negativos
  difíciles;
* jerárquico compartido: puerta y cinco cabezas sobre todo train, con puntaje
  final p(daño) * p(categoría).

Los márgenes se calibran mediante predicciones out-of-fold con GroupKFold por
video. Test no participa en ajuste, calibración, selección ni umbrales.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable
import hashlib
import json
import math
import os
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares import experimentos_jerarquicos as hj
from scripts_auxiliares import modelos_gruesos_moderador as mg
from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl, sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import DAMAGE_ORDER, tune_thresholds


ROOT = tm.ROOT
SEED = tm.SEED
FAMILIES = ("linear_svm_word_char", "logistic_regression")
FAMILY_SHORT = {
    "linear_svm_word_char": "svm",
    "logistic_regression": "logistic",
}
FAMILY_LABEL = {
    "linear_svm_word_char": "SVM lineal calibrado palabra+carácter",
    "logistic_regression": "Regresión logística calibrada",
    "fasttext_supervised_ova": "fastText supervisado OVA",
}
DESIGN_LABEL = {
    "flat": "Plano",
    "cascade": "Cascada binaria → multietiqueta",
    "shared_hierarchy": "Jerárquico clásico con TF-IDF compartido",
}
CALIBRATION_FOLDS = 3
GATE_RECALL_TARGET = hj.GATE_VALIDATION_RECALL_TARGET
AUTO_DAMAGE_PRECISION_TARGET = hj.AUTO_DAMAGE_PRECISION_TARGET
DEFAULT_BOOTSTRAP_REPLICATES = tm.BOOTSTRAP_REPLICATES

OUTPUT_DIR = ROOT / "resultados" / "metricas" / "jerarquico_clasico"
FIGURE_DIR = ROOT / "resultados" / "figuras" / "jerarquico_clasico"
MODEL_DIR = ROOT / "modelos" / "jerarquico_clasico"
REPORT_PATH = ROOT / "resultados" / "INFORME_EXPERIMENTO_JERARQUICO_CLASICO.md"
RESULT_PATH = OUTPUT_DIR / "resultado.json"
COMPARISON_PATH = OUTPUT_DIR / "comparacion_modelos.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "bootstrap_pareado_por_video.csv"
CATEGORY_PATH = OUTPUT_DIR / "recall_por_categoria.csv"
DATA_AUDIT_PATH = OUTPUT_DIR / "auditoria_dataset_ampliado.json"
HISTORICAL_BOOTSTRAP_PATH = OUTPUT_DIR / "bootstrap_ganador_vs_plano_04_2_4a1.csv"
for _directory in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def experiment_targets(frame: pd.DataFrame) -> np.ndarray:
    """Objetivos inyectables; el experimento histórico conserva cinco daños."""
    return tm.damage_targets(frame)


def _relative(path: Path) -> str:
    return tm.project_relative(path)


def _ids_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _config_sha256(config: object) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_expanded_context() -> dict:
    """Propaga el split por video de 04_2 al dataset integrado completo."""
    manifest_path = tm.BALANCED_TRAIN_MANIFEST_PATH
    balanced_path = tm.BALANCED_DATASET_PATH
    if not manifest_path.exists() or not balanced_path.exists():
        raise FileNotFoundError("Ejecute primero la construcción del dataset en 04_2.")
    manifest = _json(manifest_path)
    if sha256_file(balanced_path) != manifest["balanced_dataset_sha256"]:
        raise ValueError("El dataset 4:1 no coincide con su manifiesto.")
    integrated_path = ROOT / manifest["input_integrated_dataset"]
    if sha256_file(integrated_path) != manifest["input_integrated_sha256"]:
        raise ValueError("El dataset integrado no coincide con su manifiesto.")
    balanced = pd.DataFrame(read_jsonl(balanced_path))
    integrated = pd.DataFrame(read_jsonl(integrated_path))
    video_map_frame = balanced[["video_id", "split"]].copy()
    video_map_frame["video_id"] = video_map_frame["video_id"].astype(str)
    ambiguity = video_map_frame.groupby("video_id")["split"].nunique()
    if (ambiguity > 1).any():
        raise AssertionError("Un video aparece en más de un split congelado.")
    video_to_split = dict(
        video_map_frame.drop_duplicates().itertuples(index=False, name=None)
    )
    integrated = integrated.copy()
    integrated["video_id"] = integrated["video_id"].astype(str)
    integrated["split"] = integrated["video_id"].map(video_to_split)
    unmapped = integrated.loc[integrated["split"].isna()].copy()
    mapped = integrated.loc[integrated["split"].notna()].reset_index(drop=True)
    frames = {
        split: mapped.loc[mapped["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    hj.tm._verify_disjoint(frames)
    if mapped["chunk_id"].duplicated().any():
        raise AssertionError("El dataset ampliado contiene chunks duplicados.")
    unmapped_y = (
        experiment_targets(unmapped)
        if len(unmapped)
        else np.zeros((0, len(DAMAGE_ORDER)))
    )
    if unmapped_y.any():
        raise AssertionError("Se excluiría daño por carecer de split; revise 04_2.")
    balanced_frames = {
        split: balanced.loc[balanced["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    audit = {
        "created_at": tm.now_iso(),
        "method": "propagate_frozen_04_2_video_split_to_full_integrated_dataset",
        "integrated_dataset": _relative(integrated_path),
        "integrated_dataset_sha256": manifest["input_integrated_sha256"],
        "balanced_dataset": _relative(balanced_path),
        "balanced_dataset_sha256": manifest["balanced_dataset_sha256"],
        "integrated_rows": int(len(integrated)),
        "mapped_rows": int(len(mapped)),
        "unmapped_safe_rows_excluded": int(len(unmapped)),
        "unmapped_videos_excluded": int(unmapped["video_id"].nunique()),
        "all_damage_retained": True,
        "split_counts": {},
        "targets": list(DAMAGE_ORDER),
        "fine_labels_trained": False,
        "transversal_flags_trained": False,
    }
    for split, frame in frames.items():
        y = experiment_targets(frame)
        audit["split_counts"][split] = {
            "rows": int(len(frame)),
            "videos": int(frame["video_id"].nunique()),
            "safe": int((~y.any(axis=1)).sum()),
            "damage": int(y.any(axis=1).sum()),
            "chunk_ids_sha256": _ids_sha256(frame["chunk_id"]),
        }
    audit["usable_safe_total"] = int(
        sum(item["safe"] for item in audit["split_counts"].values())
    )
    audit["usable_damage_total"] = int(
        sum(item["damage"] for item in audit["split_counts"].values())
    )
    tm.write_json(DATA_AUDIT_PATH, audit)
    return {
        "frames": frames,
        "balanced_frames": balanced_frames,
        "audit": audit,
        "manifest": manifest,
        "signature": _config_sha256(
            {
                "integrated": manifest["input_integrated_sha256"],
                "balanced": manifest["balanced_dataset_sha256"],
                "split_ids": {
                    split: audit["split_counts"][split]["chunk_ids_sha256"]
                    for split in frames
                },
            }
        ),
    }


def expanded_dataset_summary(context: dict | None = None) -> pd.DataFrame:
    context = context or load_expanded_context()
    return pd.DataFrame(
        [
            {"split": split, **counts}
            for split, counts in context["audit"]["split_counts"].items()
        ]
    )


def load_prior_hyperparameters() -> dict:
    path = tm.METRICS_DIR / "comparacion_modelos_clasicos_optimizados.json"
    if not path.exists():
        raise FileNotFoundError("Ejecute la sección 2.2 de 04_2 antes de 04_6.")
    prior = _json(path)
    if prior.get("winner_model") != "linear_svm_word_char":
        raise ValueError(
            "El ganador clásico cambió; revise explícitamente el diseño de 04_6."
        )
    configurations = {
        family: prior["best_parameters_by_model"][family] for family in FAMILIES
    }
    return {
        "source": _artifact(path),
        "winner_model": prior["winner_model"],
        "winner_label": prior["winner_label"],
        "configurations": configurations,
        "fasttext_parameters": prior["best_parameters_by_model"].get(
            tm.FASTTEXT_KEY,
            tm.PRIOR_CLASSICAL_CONFIGS[tm.FASTTEXT_KEY],
        ),
        "historical_thresholds": prior["winner_thresholds"][1:],
        "historical_test_metrics": prior["winner_test_metrics"],
    }


@dataclass
class CalibratedBinaryHead:
    family: str
    estimator: object
    calibrator: LogisticRegression
    calibration_folds: int

    def predict_score(self, features) -> np.ndarray:
        raw = np.asarray(self.estimator.decision_function(features), dtype=float)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _new_estimator(family: str, parameters: dict):
    return mg._make_estimator(family, model_parameters=parameters)


def fit_calibrated_head(
    features,
    target: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray,
    family: str,
    parameters: dict,
    description: str,
) -> tuple[CalibratedBinaryHead, dict]:
    """Calibración sigmoide OOF agrupada; después ajusta la cabeza final."""
    target = np.asarray(target, dtype=np.int8)
    groups = np.asarray(groups, dtype=str)
    if np.unique(target).size != 2:
        raise ValueError(f"{description}: objetivo sin ambas clases.")
    folds = list(GroupKFold(n_splits=CALIBRATION_FOLDS).split(features, target, groups))
    oof_raw = np.full(len(target), np.nan, dtype=float)
    start = perf_counter()
    for fold_index, (fit_indices, calibration_indices) in enumerate(folds, 1):
        estimator = _new_estimator(family, parameters)
        estimator.fit(
            features[fit_indices],
            target[fit_indices],
            sample_weight=sample_weights[fit_indices],
        )
        oof_raw[calibration_indices] = estimator.decision_function(
            features[calibration_indices]
        )
        tqdm.write(
            f"{description} · calibración fold {fold_index}/{CALIBRATION_FOLDS}"
        )
    if np.isnan(oof_raw).any():
        raise AssertionError("La calibración no cubrió todas las filas de train.")
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1_000,
        random_state=SEED,
    )
    calibrator.fit(
        oof_raw.reshape(-1, 1), target, sample_weight=sample_weights
    )
    final_estimator = _new_estimator(family, parameters)
    final_estimator.fit(features, target, sample_weight=sample_weights)
    head = CalibratedBinaryHead(
        family=family,
        estimator=final_estimator,
        calibrator=calibrator,
        calibration_folds=CALIBRATION_FOLDS,
    )
    diagnostics = {
        "description": description,
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "videos": int(pd.Series(groups).nunique()),
        "calibration": "sigmoid_on_out_of_fold_groupkfold_scores",
        "calibration_folds": CALIBRATION_FOLDS,
        "fit_seconds": perf_counter() - start,
    }
    return head, diagnostics


def _predict_heads(heads: list[CalibratedBinaryHead], features) -> np.ndarray:
    return np.column_stack([head.predict_score(features) for head in heads])


def _fit_head_collection(
    features,
    targets: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    family: str,
    parameters: dict,
    prefix: str,
) -> tuple[list[CalibratedBinaryHead], list[dict]]:
    heads, diagnostics = [], []
    progress = tqdm(enumerate(DAMAGE_ORDER), total=len(DAMAGE_ORDER), desc=prefix, unit="cabeza")
    for column, label in progress:
        head, detail = fit_calibrated_head(
            features,
            targets[:, column],
            groups,
            weights,
            family,
            parameters,
            f"{prefix}/{label}",
        )
        heads.append(head)
        diagnostics.append(detail)
    return heads, diagnostics


def _stage2_indices(train: pd.DataFrame, gate_scores: np.ndarray) -> tuple[np.ndarray, dict]:
    selected, metadata = hj._stage2_training_frame(train, gate_scores)
    index_by_id = {
        str(chunk_id): index for index, chunk_id in enumerate(train["chunk_id"].astype(str))
    }
    indices = np.asarray(
        [index_by_id[str(chunk_id)] for chunk_id in selected["chunk_id"]], dtype=int
    )
    return indices, metadata


def _evaluate_model(
    key: str,
    label: str,
    family: str,
    design: str,
    scores: dict[str, np.ndarray],
    thresholds: np.ndarray,
    context: dict,
) -> tuple[list[dict], list[dict]]:
    rows, category_rows = [], []
    for split in ("validation", "test"):
        y = experiment_targets(context["frames"][split])
        metrics, report, predictions = hj._model_summary(y, scores[split], thresholds)
        report.to_csv(OUTPUT_DIR / f"reporte_{key}_{split}.csv")
        np.save(OUTPUT_DIR / f"scores_{key}_{split}.npy", scores[split])
        rows.append(
            {
                "model_key": key,
                "modelo": label,
                "familia": family,
                "diseno": design,
                "split": f"{split}_expanded",
                **metrics,
            }
        )
        for column, category in enumerate(DAMAGE_ORDER):
            category_rows.append(
                {
                    "model_key": key,
                    "modelo": label,
                    "familia": family,
                    "diseno": design,
                    "split": f"{split}_expanded",
                    "categoria": category,
                    "positivos": int(y[:, column].sum()),
                    "recall": float(
                        recall_score(
                            y[:, column], predictions[:, column], zero_division=0
                        )
                    ),
                    "falsos_negativos": int(
                        (y[:, column].astype(bool) & ~predictions[:, column].astype(bool)).sum()
                    ),
                    "pr_auc": float(
                        average_precision_score(y[:, column], scores[split][:, column])
                    ),
                }
            )
    # Subconjunto 4:1: comparación directa con los resultados históricos de 04_2.
    for split in ("validation", "test"):
        expanded = context["frames"][split]
        balanced = context["balanced_frames"][split]
        position = {
            str(chunk_id): index
            for index, chunk_id in enumerate(expanded["chunk_id"].astype(str))
        }
        indices = np.asarray([position[str(value)] for value in balanced["chunk_id"]])
        y = experiment_targets(balanced)
        metrics, _, _ = hj._model_summary(y, scores[split][indices], thresholds)
        rows.append(
            {
                "model_key": key,
                "modelo": label,
                "familia": family,
                "diseno": design,
                "split": f"{split}_4a1",
                **metrics,
            }
        )
    return rows, category_rows


def _selective_metrics(
    y_validation: np.ndarray,
    y_test: np.ndarray,
    gate_validation: np.ndarray,
    gate_test: np.ndarray,
    scores_validation: np.ndarray,
    scores_test: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    low = hj._threshold_for_recall(
        y_validation.any(axis=1), gate_validation, GATE_RECALL_TARGET
    )
    high = hj._threshold_for_precision(
        y_validation.any(axis=1),
        gate_validation,
        low,
        AUTO_DAMAGE_PRECISION_TARGET,
    )
    return {
        "calibration_partition": "validation_expanded",
        "validation_recall_target_for_auto_safe": GATE_RECALL_TARGET,
        "validation_precision_target_for_auto_damage": AUTO_DAMAGE_PRECISION_TARGET,
        "low_auto_safe_threshold": low,
        "high_auto_damage_threshold": float(high) if math.isfinite(high) else None,
        "validation": hj._selective_gate_metrics(
            y_validation,
            gate_validation,
            scores_validation,
            thresholds,
            low,
            high,
        ),
        "test": hj._selective_gate_metrics(
            y_test, gate_test, scores_test, thresholds, low, high
        ),
    }


def _fit_family(
    family: str,
    parameters: dict,
    context: dict,
) -> tuple[dict[str, dict], dict]:
    short = FAMILY_SHORT[family]
    train = context["frames"]["train"]
    validation = context["frames"]["validation"]
    test = context["frames"]["test"]
    y_train = experiment_targets(train).astype(np.int8)
    y_validation = experiment_targets(validation).astype(np.int8)
    y_test = experiment_targets(test).astype(np.int8)
    groups = train["video_id"].astype(str).to_numpy()
    weights = tm.source_weights(train)
    family_dir = MODEL_DIR / short
    family_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    vectorizer = mg._make_featurizer(
        family,
        max_features=int(parameters.get("max_features", 50_000)),
        model_parameters=parameters,
    )
    print(f"[{FAMILY_LABEL[family]}] Ajustando TF-IDF con {len(train):,} chunks...")
    features = {
        "train": vectorizer.fit_transform(train["text"].astype(str)),
        "validation": vectorizer.transform(validation["text"].astype(str)),
        "test": vectorizer.transform(test["text"].astype(str)),
    }
    vectorizer_path = family_dir / "vectorizador.joblib"
    joblib.dump(vectorizer, vectorizer_path)

    gate, gate_diagnostics = fit_calibrated_head(
        features["train"],
        y_train.any(axis=1),
        groups,
        weights,
        family,
        parameters,
        f"{short}/puerta",
    )
    gate_scores = {
        split: gate.predict_score(features[split])
        for split in ("train", "validation", "test")
    }
    full_heads, full_diagnostics = _fit_head_collection(
        features["train"],
        y_train,
        groups,
        weights,
        family,
        parameters,
        f"{short}/cabezas completas",
    )
    full_scores = {
        split: _predict_heads(full_heads, features[split])
        for split in ("validation", "test")
    }
    stage_indices, stage_metadata = _stage2_indices(train, gate_scores["train"])
    cascade_heads, cascade_diagnostics = _fit_head_collection(
        features["train"][stage_indices],
        y_train[stage_indices],
        groups[stage_indices],
        weights[stage_indices],
        family,
        parameters,
        f"{short}/cabezas cascada",
    )
    conditional_scores = {
        split: _predict_heads(cascade_heads, features[split])
        for split in ("validation", "test")
    }
    designs = {
        "flat": full_scores,
        "shared_hierarchy": {
            split: gate_scores[split][:, None] * full_scores[split]
            for split in ("validation", "test")
        },
        "cascade": {
            split: gate_scores[split][:, None] * conditional_scores[split]
            for split in ("validation", "test")
        },
    }
    models = {}
    for design, scores in designs.items():
        key = f"{short}__{design}"
        label = f"{FAMILY_LABEL[family]} · {DESIGN_LABEL[design]}"
        thresholds = tune_thresholds(y_validation, scores["validation"])
        models[key] = {
            "key": key,
            "label": label,
            "family": family,
            "design": design,
            "scores": scores,
            "thresholds": thresholds,
        }
        if design != "flat":
            models[key]["selective_operation"] = _selective_metrics(
                y_validation,
                y_test,
                gate_scores["validation"],
                gate_scores["test"],
                scores["validation"],
                scores["test"],
                thresholds,
            )
    bundle_path = family_dir / "modelos_calibrados.joblib"
    joblib.dump(
        {
            "family": family,
            "parameters": parameters,
            "gate": gate,
            "full_heads": full_heads,
            "cascade_heads": cascade_heads,
            "stage2_chunk_ids": train.iloc[stage_indices]["chunk_id"].tolist(),
            "thresholds": {
                key: value["thresholds"].tolist() for key, value in models.items()
            },
            "dataset_signature": context["signature"],
        },
        bundle_path,
    )
    diagnostics = {
        "family": family,
        "label": FAMILY_LABEL[family],
        "parameters_reused_from_04_2": parameters,
        "feature_shape_train": list(features["train"].shape),
        "gate": gate_diagnostics,
        "full_heads": full_diagnostics,
        "cascade_heads": cascade_diagnostics,
        "cascade_stage2": stage_metadata,
        "training_seconds": perf_counter() - start,
        "vectorizer": _artifact(vectorizer_path),
        "model_bundle": _artifact(bundle_path),
    }
    return models, diagnostics


def _fit_fasttext_flat(
    context: dict,
    parameters: dict,
) -> tuple[dict, dict]:
    """Referencia plana adicional; no interviene en las cabezas TF-IDF."""
    train = context["frames"]["train"]
    start = perf_counter()
    candidate, diagnostics = tm._fit_fasttext(
        train,
        "jerarquico_clasico_flat_expanded",
        model_parameters=parameters,
    )
    destination = MODEL_DIR / "fasttext_plano_ampliado.bin"
    shutil.copy2(candidate, destination)
    scores = {
        split: tm._fasttext_scores(destination, context["frames"][split])[:, 1:]
        for split in ("validation", "test")
    }
    y_validation = experiment_targets(context["frames"]["validation"])
    thresholds = tune_thresholds(y_validation.astype(np.int8), scores["validation"])
    model = {
        "key": "fasttext__flat",
        "label": f"{FAMILY_LABEL[tm.FASTTEXT_KEY]} · Plano",
        "family": tm.FASTTEXT_KEY,
        "design": "flat",
        "scores": scores,
        "thresholds": thresholds,
    }
    diagnostics.update(
        {
            "parameters_reused_from_04_2": parameters,
            "training_seconds_total": perf_counter() - start,
            "model_artifact": _artifact(destination),
            "limitation": "fastText CLI does not consume per-row sample weights",
        }
    )
    return model, diagnostics


def _bootstrap_comparisons(
    models: dict[str, dict], context: dict, replicates: int
) -> tuple[pd.DataFrame, dict]:
    y = experiment_targets(context["frames"]["test"])
    groups = context["frames"]["test"]["video_id"].astype(str).to_numpy()
    frames, decisions = [], {}
    for family in FAMILIES:
        short = FAMILY_SHORT[family]
        flat = models[f"{short}__flat"]
        for design in ("cascade", "shared_hierarchy"):
            candidate = models[f"{short}__{design}"]
            table = hj.paired_cluster_bootstrap(
                y,
                flat["scores"]["test"],
                flat["thresholds"],
                candidate["scores"]["test"],
                candidate["thresholds"],
                groups,
                replicates=replicates,
                seed=SEED,
            )
            table.insert(0, "candidate_key", candidate["key"])
            table.insert(1, "flat_reference_key", flat["key"])
            frames.append(table)
            decisions[candidate["key"]] = hj._decision_from_bootstrap(table)
    return pd.concat(frames, ignore_index=True), decisions


def _historical_comparison(
    winner: dict,
    prior: dict,
    context: dict,
    replicates: int,
) -> tuple[pd.DataFrame, dict]:
    historical_path = tm.METRICS_DIR / "scores_clasico_ganador_test.npy"
    if not historical_path.exists():
        raise FileNotFoundError("Faltan scores del clásico ganador de 04_2.")
    historical_scores = np.load(historical_path)
    balanced_test = context["balanced_frames"]["test"]
    expanded_test = context["frames"]["test"]
    position = {
        str(chunk_id): index
        for index, chunk_id in enumerate(expanded_test["chunk_id"].astype(str))
    }
    indices = np.asarray([position[str(value)] for value in balanced_test["chunk_id"]])
    winner_scores = winner["scores"]["test"][indices]
    y = experiment_targets(balanced_test)
    table = hj.paired_cluster_bootstrap(
        y,
        historical_scores,
        np.asarray(prior["historical_thresholds"], dtype=float),
        winner_scores,
        winner["thresholds"],
        balanced_test["video_id"].astype(str).to_numpy(),
        replicates=replicates,
        seed=SEED + 1,
    )
    table.insert(0, "candidate_key", winner["key"])
    table.insert(1, "flat_reference_key", "svm_flat_historical_04_2")
    return table, hj._decision_from_bootstrap(table)


def _plots(
    comparison: pd.DataFrame,
    category: pd.DataFrame,
    bootstrap: pd.DataFrame,
    context: dict,
    winner_key: str,
    winner_flat_key: str,
) -> list[dict]:
    artifacts = []
    summary = expanded_dataset_summary(context).set_index("split")
    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    axis.bar(x, summary["safe"], label="SEGURO", color="#4C78A8")
    axis.bar(x, summary["damage"], bottom=summary["safe"], label="Daño", color="#E45756")
    axis.set_xticks(x, summary.index)
    axis.set_ylabel("Chunks")
    axis.set_title("Dataset ampliado por split de video")
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "dataset_ampliado_por_split.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))

    test = comparison.loc[comparison["split"].eq("test_expanded")].copy()
    test = test.sort_values("damage_pr_auc_macro", ascending=False)
    figure, axis = plt.subplots(figsize=(11, 6))
    x = np.arange(len(test))
    width = 0.25
    for offset, metric, label in (
        (-width, "damage_pr_auc_macro", "PR-AUC macro"),
        (0, "damage_f1_macro", "F1 macro"),
        (width, "any_damage_recall", "Recall daño"),
    ):
        axis.bar(x + offset, test[metric], width, label=label)
    axis.set_xticks(x, test["modelo"], rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Puntuación")
    axis.set_title("Modelos clásicos sobre el mismo test ampliado")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "comparacion_global_test_ampliado.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))

    subset = category.loc[
        category["model_key"].isin([winner_key, winner_flat_key])
        & category["split"].eq("test_expanded")
    ].copy()
    pivot = subset.pivot(index="categoria", columns="model_key", values="recall").loc[DAMAGE_ORDER]
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    axis.bar(x - 0.18, pivot[winner_flat_key], 0.36, label="Plano pareado")
    axis.bar(x + 0.18, pivot[winner_key], 0.36, label="Jerárquico ganador")
    axis.set_xticks(x, pivot.index, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_title("Recall por categoría: ganador vs. referencia plana")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "recall_ganador_vs_plano.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))

    pr = bootstrap.loc[bootstrap["metrica"].eq("pr_auc_macro")].copy()
    figure, axis = plt.subplots(figsize=(10, 5))
    y_positions = np.arange(len(pr))
    centers = pr["delta_experimento_menos_plano"].to_numpy()
    lower = centers - pr["ic95_inferior"].to_numpy()
    upper = pr["ic95_superior"].to_numpy() - centers
    axis.errorbar(centers, y_positions, xerr=[lower, upper], fmt="o", capsize=4)
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y_positions, pr["candidate_key"])
    axis.set_xlabel("Δ PR-AUC macro jerárquico − plano (IC 95%)")
    axis.set_title("Bootstrap pareado por video en test ampliado")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = FIGURE_DIR / "bootstrap_pr_auc_jerarquicos.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    artifacts.append(_artifact(path))
    return artifacts


def _write_report(result: dict, comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    audit = result["dataset"]
    test = comparison.loc[comparison["split"].eq("test_expanded")].sort_values(
        "damage_pr_auc_macro", ascending=False
    )
    rows = [
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | {row.damage_f1_macro:.4f} | "
        f"{row.any_damage_recall:.4f} | {int(row.missed_damage_as_safe)} |"
        for row in test.itertuples()
    ]
    boot_rows = [
        f"| {row.candidate_key} | {row.metrica} | "
        f"{row.delta_experimento_menos_plano:+.4f} | "
        f"[{row.ic95_inferior:+.4f}, {row.ic95_superior:+.4f}] |"
        for row in bootstrap.itertuples()
    ]
    report = f"""# Experimento jerárquico clásico con SEGURO ampliado

Fecha: {tm.now_iso()}

## Diseño y datos

El mapa `train/validation/test` de `04_2` se propagó por `video_id` al dataset integrado completo. Se utilizaron **{audit['mapped_rows']:,} chunks**, incluidos **{audit['usable_safe_total']:,} SEGURO** y **{audit['usable_damage_total']:,} con daño**. Train contiene {audit['split_counts']['train']['safe']:,} seguros y {audit['split_counts']['train']['damage']:,} daños. Se excluyeron {audit['unmapped_safe_rows_excluded']:,} seguros de {audit['unmapped_videos_excluded']:,} videos sin asignación; no se perdió ningún daño.

Se reutilizaron los hiperparámetros ganadores de `04_2`: SVM lineal palabra+carácter (`C=0,25`, `min_df=1`, 50.000 features) y regresión logística (`C=2`, `min_df=2`, 50.000 features). Cada familia compara, sobre el mismo TF-IDF y datos, un modelo plano, una cascada y una jerarquía de cabezas compartidas. fastText se reentrenó como referencia plana adicional.

Los márgenes de SVM y logística se calibraron con regresión sigmoide sobre predicciones out-of-fold de tres `GroupKFold` por video. Épocas no aplican a estos optimizadores convexos: cada ajuste converge según su tolerancia o `max_iter`. Umbrales, selección del ganador y abstención usan sólo validación.

## Resultados en test ampliado

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{os.linesep.join(rows)}

## Diferencias pareadas frente al plano de la misma familia

| Candidato | Métrica | Δ jerárquico − plano | IC 95 % por video |
|---|---|---:|---:|
{os.linesep.join(boot_rows)}

El ganador se fijó con validación: **{result['selection']['winner_label']}**. Decisión frente a su plano pareado: **`{result['selection']['paired_decision']['status']}`**. Comparación secundaria frente al SVM histórico de `04_2` sobre el mismo test 4:1: **`{result['historical_04_2_comparison']['decision']['status']}`**.

No se autoriza moderación autónoma: el test sigue siendo retrospectivo, con etiquetas mayormente asistidas por LLM y sin prevalencia prospectiva de producción.

## Artefactos

- Resultado: `{_relative(RESULT_PATH)}`
- Comparación: `{_relative(COMPARISON_PATH)}`
- Bootstrap: `{_relative(BOOTSTRAP_PATH)}`
- Modelos: `{_relative(MODEL_DIR)}`
- Figuras: `{_relative(FIGURE_DIR)}`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers* (pp. 427–431). Association for Computational Linguistics. https://aclanthology.org/E17-2068/

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_experiment(
    force: bool = False,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    include_fasttext: bool = True,
) -> dict:
    context = load_expanded_context()
    prior = load_prior_hyperparameters()
    run_config = {
        "dataset_signature": context["signature"],
        "families": list(FAMILIES),
        "parameters": prior["configurations"],
        "calibration_folds": CALIBRATION_FOLDS,
        "bootstrap_replicates": bootstrap_replicates,
        "include_fasttext": include_fasttext,
    }
    run_signature = _config_sha256(run_config)
    if RESULT_PATH.exists() and not force:
        cached = _json(RESULT_PATH)
        if cached.get("run_signature") != run_signature:
            raise ValueError("Cambió la configuración; use force=True para reentrenar.")
        return cached

    all_models: dict[str, dict] = {}
    training = {}
    for family in FAMILIES:
        models, diagnostics = _fit_family(
            family, prior["configurations"][family], context
        )
        all_models.update(models)
        training[family] = diagnostics
    if include_fasttext:
        fasttext, diagnostics = _fit_fasttext_flat(
            context, prior["fasttext_parameters"]
        )
        all_models[fasttext["key"]] = fasttext
        training[tm.FASTTEXT_KEY] = diagnostics

    comparison_rows, category_rows = [], []
    for model in all_models.values():
        rows, categories = _evaluate_model(
            model["key"],
            model["label"],
            model["family"],
            model["design"],
            model["scores"],
            model["thresholds"],
            context,
        )
        comparison_rows.extend(rows)
        category_rows.extend(categories)
    comparison = pd.DataFrame(comparison_rows)
    category = pd.DataFrame(category_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)
    category.to_csv(CATEGORY_PATH, index=False)

    bootstrap, decisions = _bootstrap_comparisons(
        all_models, context, bootstrap_replicates
    )
    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    hierarchical_keys = [
        key for key, model in all_models.items() if model["design"] != "flat"
    ]
    validation = comparison.loc[
        comparison["split"].eq("validation_expanded")
        & comparison["model_key"].isin(hierarchical_keys)
    ].sort_values(["damage_pr_auc_macro", "damage_f1_macro"], ascending=False)
    winner_key = str(validation.iloc[0]["model_key"])
    winner = all_models[winner_key]
    winner_flat_key = f"{FAMILY_SHORT[winner['family']]}__flat"
    historical_bootstrap, historical_decision = _historical_comparison(
        winner, prior, context, bootstrap_replicates
    )
    historical_bootstrap.to_csv(HISTORICAL_BOOTSTRAP_PATH, index=False)
    figures = _plots(
        comparison,
        category,
        bootstrap,
        context,
        winner_key,
        winner_flat_key,
    )
    model_records = {}
    for key, model in all_models.items():
        model_records[key] = {
            "label": model["label"],
            "family": model["family"],
            "design": model["design"],
            "thresholds": model["thresholds"].tolist(),
            "validation_score_artifact": _artifact(
                OUTPUT_DIR / f"scores_{key}_validation.npy"
            ),
            "test_score_artifact": _artifact(
                OUTPUT_DIR / f"scores_{key}_test.npy"
            ),
            "selective_operation": model.get("selective_operation"),
        }
    result = {
        "schema_version": "1.0",
        "created_at": tm.now_iso(),
        "run_signature": run_signature,
        "configuration": run_config,
        "dataset": context["audit"],
        "prior_hyperparameters": prior,
        "training": training,
        "models": model_records,
        "selection": {
            "partition": "validation_expanded",
            "metric": "damage_pr_auc_macro",
            "winner_key": winner_key,
            "winner_label": winner["label"],
            "matched_flat_key": winner_flat_key,
            "paired_decision": decisions[winner_key],
            "test_used_for_selection": False,
        },
        "paired_decisions": decisions,
        "historical_04_2_comparison": {
            "partition": "test_4a1",
            "flat_reference": "SVM lineal plano histórico optimizado en 04_2",
            "candidate": winner_key,
            "decision": historical_decision,
            "bootstrap_artifact": _artifact(HISTORICAL_BOOTSTRAP_PATH),
            "interpretation": (
                "Comparison is paired on the same 4:1 test, but training data differ; "
                "the delta can combine model-family, architecture, and expanded-safe "
                "training effects."
            ),
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "cluster": "video_id",
            "artifact": _artifact(BOOTSTRAP_PATH),
        },
        "figure_artifacts": figures,
        "limitations": [
            "The expanded split reflects the available corpus, not prospective production prevalence.",
            "Labels are mostly LLM-assisted rather than an independent human gold standard.",
            "fastText CLI does not use per-row sample weights.",
            "Classical shared hierarchy is post-hoc probabilistic coupling, not joint neural optimization.",
        ],
    }
    tm.write_json(RESULT_PATH, result)
    _write_report(result, comparison, bootstrap)
    result["report_artifact"] = _artifact(REPORT_PATH)
    tm.write_json(RESULT_PATH, result)
    return result


def load_tables() -> dict[str, pd.DataFrame]:
    required = (RESULT_PATH, COMPARISON_PATH, CATEGORY_PATH, BOOTSTRAP_PATH)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan resultados: " + ", ".join(map(str, missing)))
    return {
        "comparison": pd.read_csv(COMPARISON_PATH),
        "categories": pd.read_csv(CATEGORY_PATH),
        "bootstrap": pd.read_csv(BOOTSTRAP_PATH),
        "historical_bootstrap": pd.read_csv(HISTORICAL_BOOTSTRAP_PATH),
    }
