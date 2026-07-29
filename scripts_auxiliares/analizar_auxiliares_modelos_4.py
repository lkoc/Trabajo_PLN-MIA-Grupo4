"""Auditoría común de etiquetas finas y flags para modelos de cuatro daños."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts_auxiliares import entrenar_qwen_acoso_amenaza as q4
from scripts_auxiliares import entrenar_transformers_planos_4 as t4
from scripts_auxiliares import experimentos_jerarquicos_4 as h4
from scripts_auxiliares import experimentos_jerarquicos_clasicos_4 as c4
from scripts_auxiliares import experimentos_qwen_jerarquico_4 as qh


ROOT = q4.ROOT
OUTPUT_DIR = ROOT / "resultados" / "metricas" / "auditoria_auxiliar_modelos_4"
FIGURE_DIR = ROOT / "resultados" / "figuras" / "auditoria_auxiliar_modelos_4"
REPORT_PATH = ROOT / "resultados" / "INFORME_AUDITORIA_FINAS_FLAGS_MODELOS_4.md"
for _directory in (OUTPUT_DIR, FIGURE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

FINE_TO_TARGET = {
    "seguro": None,
    "seguro_ironia_marcada": None,
    "racismo_etnico_explicito": 0,
    "racismo_encubierto": 0,
    "clasismo_racial": 0,
    "discriminacion_regional": 0,
    "racismo_linguistico": 0,
    "misoginia_acoso_genero": 1,
    "homofobia_transfobia": 1,
    "acoso_personal": 2,
    "amenaza_directa": 2,
    "sexual_explicito": 3,
    "sexual_cosificacion": 3,
    "sexual_no_consensual": 3,
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_models() -> tuple[list[dict], list[str]]:
    models, missing = [], []
    if q4.OPERATIONAL_SELECTION_PATH.exists() and q4.OPERATIONAL_TEST_PATH.exists():
        operational = q4.load_operational_evaluation(
            load_scores=True,
            require_test=True,
        )
        models.append(
            {
                "key": "qwen4_flat",
                "label": (
                    "Qwen 04_205 plano · época operativa "
                    f"{operational['selected_epoch']}"
                ),
                "regime": "coarse + fine/flag auxiliary multitask",
                "scores": operational["scores"]["test"],
                "thresholds": np.asarray(
                    operational["thresholds_selected_on_validation"]
                ),
                "dataset_sha256": operational["dataset_sha256"],
                "checkpoint_provenance": {
                    "selected_epoch": operational["selected_epoch"],
                    "selected_adapter": operational["selected_adapter"],
                    "selection_artifact": operational["artifacts"]["selection"],
                    "test_used_for_selection": False,
                },
            }
        )
    else:
        missing.extend(
            str(path.relative_to(ROOT))
            for path in (q4.OPERATIONAL_SELECTION_PATH, q4.OPERATIONAL_TEST_PATH)
            if not path.exists()
        )

    if t4.RESULT_PATH.exists():
        result = _json(t4.RESULT_PATH)
        for key, record in result["models"].items():
            models.append(
                {
                    "key": f"transformer_flat__{key}",
                    "label": f"{record['model']['label']} plano",
                    "regime": "coarse-only",
                    "scores": np.load(ROOT / record["score_artifacts"]["test"]["path"]),
                    "thresholds": np.asarray(record["thresholds_selected_on_validation"]),
                    "dataset_sha256": result["dataset"]["sha256"],
                }
            )
    else:
        missing.append(str(t4.RESULT_PATH.relative_to(ROOT)))

    for key in (h4.CASCADE_EXTRA_SAFE_KEY, h4.JOINT_KEY):
        path = h4.result_path(key)
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        result = _json(path)
        score_path = h4._experiment_paths(key)["metrics"] / "scores_test.npy"
        models.append(
            {
                "key": key,
                "label": result["experiment_label"],
                "regime": "coarse-only; expanded binary supervision",
                "scores": np.load(score_path),
                "thresholds": np.asarray(result["thresholds_selected_on_validation"]),
                "dataset_sha256": result["dataset"]["sha256"],
            }
        )

    if c4.RESULT_PATH.exists():
        result = _json(c4.RESULT_PATH)
        common = pd.read_csv(c4.COMMON_4A1_PATH)
        for key, group in common.groupby("model_key"):
            row = group.loc[group["split"].eq("test")].iloc[0]
            models.append(
                {
                    "key": f"classic__{key}",
                    "label": row["modelo"],
                    "regime": "coarse-only",
                    "scores": np.load(c4.OUTPUT_DIR / f"scores_{key}_test_4a1.npy"),
                    "thresholds": np.asarray(
                        json.loads(row["thresholds_selected_on_common_validation"])
                    ),
                    "dataset_sha256": result["dataset"]["balanced_dataset_sha256"],
                }
            )
    else:
        missing.append(str(c4.RESULT_PATH.relative_to(ROOT)))

    if qh.RESULT_PATH.exists():
        result = _json(qh.RESULT_PATH)
        for key, record in result["models"].items():
            models.append(
                {
                    "key": key,
                    "label": record["label"],
                    "regime": "frozen Qwen coarse + auxiliary logits",
                    "scores": np.load(qh.METRICS_DIR / f"scores_{key}_test.npy"),
                    "thresholds": np.asarray(record["thresholds"]),
                    "dataset_sha256": result["dataset"]["sha256"],
                }
            )
    else:
        missing.append(str(qh.RESULT_PATH.relative_to(ROOT)))
    return models, missing


def run_analysis(review_fractions=(0.10, 0.20)) -> dict:
    frames, audit = q4.load_frames()
    test = frames["test"]
    models, missing = collect_models()
    if not models:
        raise RuntimeError(
            "Todavía no hay resultados finales para auditar. Ejecute al menos uno de "
            "los cuadernos 04_201 a 04_206 y vuelva a correr 04_208."
        )
    bad = [model["key"] for model in models if model["dataset_sha256"] != audit["dataset_sha256"]]
    if bad:
        raise ValueError(f"Modelos con otro dataset: {bad}")
    fine_rows, flag_rows = [], []
    fine_values = test["fine_labels_auxiliary"]
    flags = test["flags_auxiliary"]
    y = h4.four_targets(test).astype(bool)
    for model in models:
        scores = model["scores"]
        thresholds = model["thresholds"]
        if scores.shape != (len(test), 4) or thresholds.shape != (4,):
            raise ValueError(f"Dimensión incompatible: {model['key']}")
        predictions = scores >= thresholds
        distance = np.min(np.abs(scores - thresholds), axis=1)
        uncertainty_order = np.argsort(distance)
        for fine_label, target in FINE_TO_TARGET.items():
            mask = fine_values.map(lambda values: fine_label in values).to_numpy()
            if not mask.any():
                continue
            if target is None:
                value = float((~predictions[mask].any(axis=1)).mean())
                metric = "safe_specificity"
            else:
                value = float(predictions[mask, target].mean())
                metric = "target_recall"
            fine_rows.append(
                {
                    "model_key": model["key"],
                    "modelo": model["label"],
                    "supervision_regime": model["regime"],
                    "fine_label": fine_label,
                    "coarse_target": "SEGURO" if target is None else h4.TARGET_LABELS[target],
                    "n": int(mask.sum()),
                    "metric": metric,
                    "value": value,
                }
            )
        for fraction in review_fractions:
            count = max(1, int(round(len(test) * fraction)))
            review = np.zeros(len(test), dtype=bool)
            review[uncertainty_order[:count]] = True
            missed = y.any(axis=1) & ~predictions.any(axis=1)
            for flag in q4.TRANSVERSAL_FLAGS:
                flag_mask = flags.map(lambda values: flag in values).to_numpy()
                flag_rows.append(
                    {
                        "model_key": model["key"],
                        "modelo": model["label"],
                        "supervision_regime": model["regime"],
                        "review_fraction": fraction,
                        "flag": flag,
                        "flag_n": int(flag_mask.sum()),
                        "flag_capture": float((review & flag_mask).sum() / max(1, flag_mask.sum())),
                        "missed_damage_n": int(missed.sum()),
                        "missed_damage_capture": float(
                            (review & missed).sum() / max(1, missed.sum())
                        ),
                    }
                )
    fine = pd.DataFrame(fine_rows)
    flag = pd.DataFrame(flag_rows)
    fine.to_csv(OUTPUT_DIR / "desempeno_por_etiqueta_fina.csv", index=False)
    flag.to_csv(OUTPUT_DIR / "captura_flags_por_incertidumbre.csv", index=False)
    _figures(fine, flag)
    result = {
        "completed_at": q4.tm.now_iso(),
        "dataset_sha256": audit["dataset_sha256"],
        "test_rows": len(test),
        "models_analyzed": [
            {
                "key": model["key"],
                "label": model["label"],
                "regime": model["regime"],
                "checkpoint_provenance": model.get("checkpoint_provenance"),
            }
            for model in models
        ],
        "missing_results": missing,
        "fine_rows": len(fine),
        "flag_rows": len(flag),
        "interpretation": (
            "Fine labels and flags are audit strata, never gold predictors. "
            "Differences between supervision regimes are descriptive unless an explicit ablation is run."
        ),
    }
    q4.tm.write_json(OUTPUT_DIR / "resultado.json", result)
    _write_report(result, fine, flag)
    return result


def _figures(fine: pd.DataFrame, flag: pd.DataFrame) -> None:
    if not fine.empty:
        subset = fine.loc[fine["n"].ge(10)]
        pivot = subset.pivot_table(index="fine_label", columns="modelo", values="value")
        axis = pivot.plot.bar(figsize=(15, 7))
        axis.set_ylim(0, 1)
        axis.set_title("Desempeño por etiqueta fina en test")
        axis.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "desempeno_fino.png", dpi=180, bbox_inches="tight")
        plt.close()
    if not flag.empty:
        subset = flag.loc[flag["review_fraction"].eq(0.20)]
        pivot = subset.pivot_table(index="flag", columns="modelo", values="flag_capture")
        axis = pivot.plot.bar(figsize=(15, 6))
        axis.set_ylim(0, 1)
        axis.set_title("Captura de flags al revisar 20% más incierto")
        axis.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "captura_flags.png", dpi=180, bbox_inches="tight")
        plt.close()


def _write_report(result: dict, fine: pd.DataFrame, flag: pd.DataFrame) -> None:
    qwen = next(
        (model for model in result["models_analyzed"] if model["key"] == "qwen4_flat"),
        None,
    )
    qwen_provenance = (
        f"La referencia Qwen plana corresponde a la época operativa "
        f"**{qwen['checkpoint_provenance']['selected_epoch']}**, elegida en validation "
        "sin consultar test."
        if qwen and qwen.get("checkpoint_provenance")
        else "La referencia Qwen plana no estaba disponible."
    )
    report = f"""# Auditoría de etiquetas finas y transversales

Fecha: {result['completed_at']}

Se analizaron {len(result['models_analyzed'])} modelos sobre el mismo test de {result['test_rows']:,} chunks. Las etiquetas finas se usaron para desagregar recall o especificidad de `SEGURO`; los flags transversales midieron cuánto captura una revisión ordenada por cercanía a los umbrales.

Las etiquetas finas y flags **no se usaron como predictores gold**. Qwen `04_205` sí las usa como supervisión auxiliar multitararea y `04_206` consume sus logits auxiliares; los demás son controles `coarse-only`. Por tanto, una diferencia entre esos regímenes no debe atribuirse exclusivamente a la arquitectura sin una ablación específica.

{qwen_provenance}

- Tabla fina: `resultados/metricas/auditoria_auxiliar_modelos_4/desempeno_por_etiqueta_fina.csv`
- Tabla de flags: `resultados/metricas/auditoria_auxiliar_modelos_4/captura_flags_por_incertidumbre.csv`
- Resultados pendientes: {', '.join(result['missing_results']) if result['missing_results'] else 'ninguno'}.

## Conclusión

Esta auditoría sirve para detectar debilidades por fenómeno fino y para estudiar qué casos capturaría una cola de revisión. Sus diferencias son descriptivas: no sustituyen la comparación principal de cuatro daños, no prueban causalmente el beneficio de la supervisión auxiliar y no autorizan despliegue autónomo.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
