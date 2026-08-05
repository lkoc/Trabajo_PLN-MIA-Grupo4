"""SVM y regresión logística, planos y jerárquicos, con cuatro daños."""

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares import experimentos_jerarquicos_4 as h4
from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl, sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import tune_thresholds


ROOT = tm.ROOT
TARGET_LABELS = h4.TARGET_LABELS
OUTPUT_DIR = ROOT / "resultados" / "metricas" / "jerarquico_clasico_4"
FIGURE_DIR = ROOT / "resultados" / "figuras" / "jerarquico_clasico_4"
MODEL_DIR = ROOT / "modelos" / "jerarquico_clasico_4"
REPORT_PATH = ROOT / "resultados" / "INFORME_EXPERIMENTO_JERARQUICO_CLASICO_4.md"
for _directory in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _load_runtime():
    name = "scripts_auxiliares._experimentos_jerarquicos_clasicos_4_runtime"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("experimentos_jerarquicos_clasicos.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_runtime = _load_runtime()


def _ids_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _signature(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_expanded_context() -> dict:
    """Propaga splits por video y asigna SEGURO no mapeado sólo a train."""
    manifest_path = tm.BALANCED_TRAIN_MANIFEST_PATH
    balanced_path = tm.BALANCED_DATASET_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(balanced_path) != manifest["balanced_dataset_sha256"]:
        raise ValueError("El dataset 4:1 no coincide con su manifiesto.")
    integrated_path = ROOT / manifest["input_integrated_dataset"]
    if sha256_file(integrated_path) != manifest["input_integrated_sha256"]:
        raise ValueError("El dataset integrado no coincide con su manifiesto.")
    balanced = pd.DataFrame(read_jsonl(balanced_path))
    integrated = pd.DataFrame(read_jsonl(integrated_path))
    balanced["video_id"] = balanced["video_id"].astype(str)
    integrated["video_id"] = integrated["video_id"].astype(str)
    ambiguity = balanced.groupby("video_id")["split"].nunique()
    if (ambiguity > 1).any():
        raise AssertionError("Un video aparece en más de un split congelado.")
    video_to_split = dict(
        balanced[["video_id", "split"]].drop_duplicates().itertuples(index=False, name=None)
    )
    integrated["split"] = integrated["video_id"].map(video_to_split)
    unmapped = integrated.loc[integrated["split"].isna()].copy()
    if len(unmapped) and h4.four_targets(unmapped).astype(bool).any():
        raise AssertionError("Hay daño sin split; no puede asignarse automáticamente.")
    unmapped["split"] = "train"
    assigned = pd.concat(
        [integrated.loc[integrated["split"].notna()], unmapped],
        ignore_index=True,
        sort=False,
    )
    if assigned["chunk_id"].duplicated().any():
        raise AssertionError("El dataset ampliado contiene chunks duplicados.")
    frames = {
        split: assigned.loc[assigned["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    h4._runtime.tm._verify_disjoint(frames)
    balanced_frames = {
        split: balanced.loc[balanced["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    audit = {
        "created_at": tm.now_iso(),
        "method": (
            "propagate frozen 04_2 video split; assign only unmapped SEGURO videos to train"
        ),
        "integrated_dataset": str(integrated_path.relative_to(ROOT)),
        "integrated_dataset_sha256": manifest["input_integrated_sha256"],
        "balanced_dataset": str(balanced_path.relative_to(ROOT)),
        "balanced_dataset_sha256": manifest["balanced_dataset_sha256"],
        "integrated_rows": int(len(integrated)),
        "mapped_rows": int(len(assigned)),
        "unmapped_safe_rows_assigned_train": int(len(unmapped)),
        "unmapped_videos_assigned_train": int(unmapped["video_id"].nunique()),
        "all_damage_retained": True,
        "validation_or_test_video_leakage": False,
        "split_counts": {},
        "targets": TARGET_LABELS,
        "fine_labels_trained": False,
        "transversal_flags_trained": False,
    }
    for split, frame in frames.items():
        y = h4.four_targets(frame).astype(bool)
        audit["split_counts"][split] = {
            "rows": int(len(frame)),
            "videos": int(frame["video_id"].nunique()),
            "safe": int((~y.any(axis=1)).sum()),
            "damage": int(y.any(axis=1).sum()),
            "chunk_ids_sha256": _ids_sha256(frame["chunk_id"]),
        }
    audit["usable_safe_total"] = int(
        sum(row["safe"] for row in audit["split_counts"].values())
    )
    audit["usable_damage_total"] = int(
        sum(row["damage"] for row in audit["split_counts"].values())
    )
    tm.write_json(OUTPUT_DIR / "auditoria_dataset_ampliado.json", audit)
    signature = _signature(
        {
            "integrated": manifest["input_integrated_sha256"],
            "balanced": manifest["balanced_dataset_sha256"],
            "split_ids": {
                split: audit["split_counts"][split]["chunk_ids_sha256"]
                for split in frames
            },
            "targets": TARGET_LABELS,
        }
    )
    return {
        "frames": frames,
        "balanced_frames": balanced_frames,
        "audit": audit,
        "manifest": manifest,
        "signature": signature,
    }


def _historical_comparison(winner, prior, context, replicates):
    validation_path = tm.METRICS_DIR / "scores_clasico_ganador_validation.npy"
    test_path = tm.METRICS_DIR / "scores_clasico_ganador_test.npy"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError("Faltan scores del clásico histórico de 04_2.")
    historical_validation = h4.merge_five_scores(np.load(validation_path))
    historical_test = h4.merge_five_scores(np.load(test_path))
    balanced_validation = context["balanced_frames"]["validation"]
    balanced_test = context["balanced_frames"]["test"]
    historical_thresholds = tune_thresholds(
        h4.four_targets(balanced_validation).astype(np.int8), historical_validation
    )
    expanded_test = context["frames"]["test"]
    position = {
        str(chunk_id): index
        for index, chunk_id in enumerate(expanded_test["chunk_id"].astype(str))
    }
    indices = np.asarray([position[str(value)] for value in balanced_test["chunk_id"]])
    winner_scores = winner["scores"]["test"][indices]
    y = h4.four_targets(balanced_test)
    table = h4._runtime.paired_cluster_bootstrap(
        y,
        historical_test,
        historical_thresholds,
        winner_scores,
        winner["thresholds"],
        balanced_test["video_id"].astype(str).to_numpy(),
        replicates=replicates,
        seed=tm.SEED + 1,
    )
    table.insert(0, "candidate_key", winner["key"])
    table.insert(1, "flat_reference_key", "svm_flat_historical_04_2_posthoc4")
    return table, h4._runtime._decision_from_bootstrap(table)


def _write_report(result: dict, comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    audit = result["dataset"]
    test = comparison.loc[comparison["split"].eq("test_expanded")].sort_values(
        "damage_pr_auc_macro", ascending=False
    )
    rows = "\n".join(
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | {row.damage_f1_macro:.4f} | "
        f"{row.any_damage_recall:.4f} | {int(row.missed_damage_as_safe)} |"
        for row in test.itertuples()
    )
    common_path = OUTPUT_DIR / "comparacion_comun_4a1.csv"
    common_rows = ""
    if common_path.exists():
        common_test = pd.read_csv(common_path)
        common_test = common_test.loc[common_test["split"].eq("test")].sort_values(
            "damage_pr_auc_macro", ascending=False
        )
        common_rows = "\n".join(
            f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | "
            f"{row.damage_f1_macro:.4f} | {row.any_damage_recall:.4f} |"
            for row in common_test.itertuples()
        )
    report = f"""# Modelos clásicos planos y jerárquicos con cuatro daños

Fecha: {tm.now_iso()}

## Datos y diseño

Todos los experimentos proceden del dataset integrado `{audit['integrated_dataset']}` con hash `{audit['integrated_dataset_sha256']}`. Contiene {audit['usable_safe_total']:,} chunks SEGURO y {audit['usable_damage_total']:,} con daño. Train utiliza {audit['split_counts']['train']['safe']:,} SEGURO y {audit['split_counts']['train']['damage']:,} daños; ningún video de validation/test entra en train.

Se entrenan SVM lineal y regresión logística, cada uno como modelo plano, cascada binaria y jerarquía probabilística compartida. Las cuatro etiquetas son {', '.join(TARGET_LABELS)}. Los hiperparámetros provienen de la búsqueda de `04_2`; la calibración se realiza out-of-fold por video y los umbrales se fijan en validation. fastText queda fuera de esta variante porque su artefacto histórico codifica cinco salidas y no admite transferencia exacta de la cabeza.

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{rows}

## Comparación cruzada sobre el test 4:1 común

| Modelo | PR-AUC macro | F1 macro | Recall daño |
|---|---:|---:|---:|
{common_rows or '| Pendiente de completar el postproceso | NA | NA | NA |'}

Ganador por validation: **{result['selection']['winner_label']}**. Decisión pareada frente a su plano: `{result['selection']['paired_decision']['status']}`. No se autoriza moderación autónoma sin gold standard humano independiente y piloto prospectivo.

## Artefactos

- Resultado: `{_runtime._relative(_runtime.RESULT_PATH)}`
- Modelos: `{_runtime._relative(MODEL_DIR)}`
- Comparación: `{_runtime._relative(_runtime.COMPARISON_PATH)}`
- Bootstrap: `{_runtime._relative(_runtime.BOOTSTRAP_PATH)}`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


_runtime.DAMAGE_ORDER = TARGET_LABELS
_runtime.experiment_targets = h4.four_targets
_runtime.hj = h4._runtime
_runtime.OUTPUT_DIR = OUTPUT_DIR
_runtime.FIGURE_DIR = FIGURE_DIR
_runtime.MODEL_DIR = MODEL_DIR
_runtime.REPORT_PATH = REPORT_PATH
_runtime.RESULT_PATH = OUTPUT_DIR / "resultado.json"
_runtime.COMPARISON_PATH = OUTPUT_DIR / "comparacion_modelos.csv"
_runtime.BOOTSTRAP_PATH = OUTPUT_DIR / "bootstrap_pareado_por_video.csv"
_runtime.CATEGORY_PATH = OUTPUT_DIR / "recall_por_categoria.csv"
_runtime.DATA_AUDIT_PATH = OUTPUT_DIR / "auditoria_dataset_ampliado.json"
_runtime.HISTORICAL_BOOTSTRAP_PATH = (
    OUTPUT_DIR / "bootstrap_ganador_vs_plano_04_2_4a1.csv"
)
_runtime.load_expanded_context = load_expanded_context
_runtime._historical_comparison = _historical_comparison
_runtime._write_report = _write_report


RESULT_PATH = _runtime.RESULT_PATH
COMPARISON_PATH = _runtime.COMPARISON_PATH
BOOTSTRAP_PATH = _runtime.BOOTSTRAP_PATH
CATEGORY_PATH = _runtime.CATEGORY_PATH
COMMON_4A1_PATH = OUTPUT_DIR / "comparacion_comun_4a1.csv"


def expanded_dataset_summary(context: dict | None = None) -> pd.DataFrame:
    return _runtime.expanded_dataset_summary(context)


def run_experiment(
    force: bool = False,
    bootstrap_replicates: int = tm.BOOTSTRAP_REPLICATES,
) -> dict:
    result = _runtime.run_experiment(
        force=force,
        bootstrap_replicates=bootstrap_replicates,
        include_fasttext=False,
    )
    context = load_expanded_context()
    rows = []
    for key, record in result["models"].items():
        expanded_scores = {
            split: np.load(ROOT / record[f"{split}_score_artifact"]["path"])
            for split in ("validation", "test")
        }
        common_scores = {}
        for split in ("validation", "test"):
            expanded = context["frames"][split]
            balanced = context["balanced_frames"][split]
            position = {
                str(chunk_id): index
                for index, chunk_id in enumerate(expanded["chunk_id"].astype(str))
            }
            indices = np.asarray([position[str(value)] for value in balanced["chunk_id"]])
            common_scores[split] = expanded_scores[split][indices]
            np.save(OUTPUT_DIR / f"scores_{key}_{split}_4a1.npy", common_scores[split])
        thresholds = tune_thresholds(
            h4.four_targets(context["balanced_frames"]["validation"]).astype(np.int8),
            common_scores["validation"],
        )
        for split in ("validation", "test"):
            metrics, _, _ = h4.evaluate_four_scores(
                h4.four_targets(context["balanced_frames"][split]),
                common_scores[split],
                thresholds,
            )
            rows.append(
                {
                    "model_key": key,
                    "modelo": record["label"],
                    "split": split,
                    "thresholds_selected_on_common_validation": json.dumps(
                        thresholds.tolist()
                    ),
                    **{name: value for name, value in metrics.items() if name != "category_recall"},
                }
            )
    common = pd.DataFrame(rows)
    common.to_csv(COMMON_4A1_PATH, index=False)
    validation = common.loc[common["split"].eq("validation")].sort_values(
        ["damage_pr_auc_macro", "damage_f1_macro"], ascending=False
    )
    result["common_4a1_evaluation"] = {
        "purpose": "cross-family comparison with 04_202, 04_203, 04_204 and 04_205",
        "threshold_partition": "common 4:1 validation",
        "test_partition": "common 4:1 test",
        "winner_key": str(validation.iloc[0]["model_key"]),
        "winner_label": str(validation.iloc[0]["modelo"]),
        "artifact": _runtime._artifact(COMMON_4A1_PATH),
    }
    tm.write_json(RESULT_PATH, result)
    _write_report(
        result,
        pd.read_csv(COMPARISON_PATH),
        pd.read_csv(BOOTSTRAP_PATH),
    )
    result["report_artifact"] = _runtime._artifact(REPORT_PATH)
    tm.write_json(RESULT_PATH, result)
    return result


def load_tables() -> dict[str, pd.DataFrame]:
    tables = _runtime.load_tables()
    tables["common_4a1"] = pd.read_csv(COMMON_4A1_PATH)
    return tables
