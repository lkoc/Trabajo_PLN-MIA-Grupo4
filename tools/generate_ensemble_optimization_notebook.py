from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "flujo/03_entrenamiento/03_07b_optimizacion_ensembles.ipynb"


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


notebook = nbformat.v4.new_notebook(
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
)
notebook.cells = [
    markdown(
        r"""
# 03_07b · Optimización reproducible de ensembles

Este subpaso **actualiza la comparación 03_07 existente**; no crea un segundo
reporte ni consulta `test` durante la selección. Reconstituye el panel común de
10 600 filas desde las predicciones originales de clásico, Transformer y Qwen,
verifica SHA-256, calibra dentro de pliegues agrupados por vídeo y compara las
cinco reglas base más las mezclas suave y dura optimizadas.

La búsqueda convexa sigue la idea de *stacked generalization*/Super Learner
(Wolpert, 1992; van der Laan, Polley y Hubbard, 2007), pero el objetivo
lexicográfico BA → riesgo 0,67 → macro-AUPRC y el desempate hacia pesos iguales
son decisiones propias del proyecto. Unión e intersección no tienen
coeficientes y solo se recalibran/revalúan.
"""
    ),
    code(
        r"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path.cwd().resolve()
ROOT = next(
    path for path in [HERE, *HERE.parents]
    if (path / "pyproject.toml").is_file()
)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from moderacion_peru.ensemble_optimization import (
    DEFAULT_MEMBER_IDS,
    align_prediction_rows,
    build_optimization_report,
    build_updated_freeze,
    evaluate_optimized_test_from_archive,
    load_prediction_rows,
    load_prediction_rows_from_tar,
    nested_compare_ensembles,
    write_optimization_report,
    write_updated_selection_and_test,
)
from moderacion_peru.io import write_json_atomic

CLASSICAL = ROOT / "modelos/v2/clasicos/regularization_screen/runs/classical-54f7971c6000eae5/logistic_regression_c0p5/predictions_validation.jsonl"
TRANSFORMER_TAR = ROOT / "modelos/_downloads_04/transformer_03_03b_run_outputs.tar.gz"
QWEN_TAR = Path.home() / "Downloads/run_outputs-b.tar"
TEST_TAR = Path.home() / "Downloads/run_outputs-a.tar"
DATASET = ROOT / "datos/model_ready/v2/dataset_5_salidas.jsonl"
COMPARISON = ROOT / "resultados/modelos/comparacion_individual_ensemble_validation.json"
FREEZE = ROOT / "resultados/modelos/seleccion_congelada.json"
TEST_REPORT = ROOT / "resultados/modelos/test_final_abierto_una_vez.json"
OPTIMIZATION_REPORT = ROOT / "resultados/modelos/optimizacion_ensembles/optimizacion_ensembles_validation.json"

EXPECTED_SHA256 = {
    TRANSFORMER_TAR: "fff9f75ae381ec0123b57850afc72528bd27e4d6ae75b8e3a6aedf150bbab290",
    QWEN_TAR: "4a6242fa9a9c5c6e5182ebcf695dc44bdf32e224a7b93dd8b70cb7cf6eb1bf7b",
    TEST_TAR: "6ac7ac71d4173819a07d266001eb2eec9711fef5d4edbce41062d28a069bc7e2",
}
RUN_NESTED_OPTIMIZATION = True
UPDATE_CURRENT_ARTIFACTS = True
"""
    ),
    code(
        r"""
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

for path, expected in EXPECTED_SHA256.items():
    observed = sha256(path)
    assert observed == expected, (path, observed)

classical_rows = load_prediction_rows(CLASSICAL)
transformer_rows = load_prediction_rows_from_tar(
    TRANSFORMER_TAR,
    "runs/cascade_v2-af78eba77883921f/predictions_validation.jsonl",
)
qwen_rows = load_prediction_rows_from_tar(
    QWEN_TAR,
    "runs/qwen_lora-4aa5ce04df057144/predictions_validation.jsonl",
)
panel = align_prediction_rows(
    list(zip(DEFAULT_MEMBER_IDS, [classical_rows, transformer_rows, qwen_rows]))
)
assert panel.raw_scores.shape == (10_600, 3, 5)
assert len(set(panel.video_ids)) == 716
{"rows": len(panel.chunk_ids), "videos": len(set(panel.video_ids)), "shape": panel.raw_scores.shape}
"""
    ),
    code(
        r"""
if RUN_NESTED_OPTIMIZATION:
    nested = nested_compare_ensembles(
        panel, outer_folds=5, inner_folds=5, grid_step=0.025
    )
    report = build_optimization_report(
        panel=panel,
        nested_result=nested,
        original_comparison_path=COMPARISON,
        source_artifacts={
            "classical_predictions": CLASSICAL,
            "transformer_run_outputs": TRANSFORMER_TAR,
            "qwen_run_outputs": QWEN_TAR,
        },
    )
    write_optimization_report(OPTIMIZATION_REPORT, report)
else:
    report = json.loads(OPTIMIZATION_REPORT.read_text(encoding="utf-8"))

ranking = pd.DataFrame(report["optimization"]["ranking_nested_oof"])
ranking[[
    "rank", "candidate_id", "balanced_accuracy_any_damage_nested_oof",
    "risk_0_67_nested_oof", "macro_auprc_damage_nested_oof",
    "macro_f1_damage_nested_oof",
]]
"""
    ),
    code(
        r"""
# Solo después de terminar la selección con validation se congela la fórmula.
# El TAR de test aporta checkpoints de la apertura original: no se infiere de nuevo.
if UPDATE_CURRENT_ARTIFACTS:
    old_test = json.loads(TEST_REPORT.read_text(encoding="utf-8"))
    freeze = build_updated_freeze(
        original_freeze_path=FREEZE,
        optimization_report=report,
        dataset_path=DATASET,
    )
    test = evaluate_optimized_test_from_archive(
        freeze=freeze,
        dataset_path=DATASET,
        archive_path=TEST_TAR,
        original_test_report=old_test,
    )
    report["status"] = "comparison_updated_optimized_winner_test_reanalyzed"
    report["updated_selection"] = {
        "selected_id": freeze["selected_id"],
        "comparison_signature": freeze["comparison_signature"],
        "weights": freeze["ensemble_weights"],
        "thresholds": freeze["thresholds"],
        "any_damage_threshold": freeze["any_damage_threshold"],
        "winner_status": freeze["winner_status"],
    }
    report["updated_test"] = {
        "archive_path": str(TEST_TAR),
        "archive_bytes": TEST_TAR.stat().st_size,
        "archive_sha256": sha256(TEST_TAR),
        "test_rows_natural": test["test_rows_natural"],
        "test_rows_4_to_1": test["test_rows_4_to_1"],
        "primary_metrics_natural_prevalence": test["primary_metrics_natural_prevalence"],
        "secondary_metrics_4_to_1": test["secondary_metrics_4_to_1"],
        "new_inference_passes": 0,
        "test_open_count": 1,
    }
    report["interpretation"] = {
        "selection_scope": "validation_only_nested_grouped_cv",
        "test_reused": "verified_member_score_checkpoints_after_formula_freeze",
        "production_replaced": True,
        "why": "La optimización amplía 03_07; el ganador se congela con validation antes de aplicar la fórmula a checkpoints de la única apertura de test.",
        "union_intersection": "max/min no contienen coeficientes; se recalibran sin inventar parámetros.",
    }
    write_optimization_report(OPTIMIZATION_REPORT, report)
    write_updated_selection_and_test(
        freeze_path=FREEZE,
        test_report_path=TEST_REPORT,
        freeze=freeze,
        test_report=test,
    )

    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparison.update({
        "schema_version": "5.0.0",
        "parent_comparison_signature": comparison.get("comparison_signature"),
        "comparison_signature": freeze["comparison_signature"],
        "selected_for_freeze": freeze["selected_id"],
        "winner_status": freeze["winner_status"],
        "test_status": "evaluated_from_original_verified_score_checkpoints",
        "ensemble_optimization_update": {
            "protocol": report["optimization"]["protocol"],
            "report": str(OPTIMIZATION_REPORT.relative_to(ROOT)),
            "ranking_nested_oof": report["optimization"]["ranking_nested_oof"],
            "paired_video_bootstrap": report["optimization"]["paired_video_bootstrap"],
            "final_fit": report["optimization"]["final_fit_on_complete_validation"],
            "base_ranking_preserved_for_audit": True,
        },
    })
    write_json_atomic(COMPARISON, comparison)
"""
    ),
    code(
        r"""
winner = report["optimization"]["ranking_nested_oof"][0]
final = report["optimization"]["final_fit_on_complete_validation"]
test_any = report["updated_test"]["primary_metrics_natural_prevalence"]["binary_any_damage_frozen_gate"]
pd.DataFrame({
    "miembro": ["clásico", "Transformer", "Qwen"],
    "peso_suave_final": final["soft_optimized"]["weights"],
    "peso_duro_final": final["hard_optimized"]["weights"],
}), {
    "ganador": winner["candidate_id"],
    "BA_validation_anidada": winner["balanced_accuracy_any_damage_nested_oof"],
    "BA_test": test_any["balanced_accuracy"],
    "FNR_test": test_any["false_negative_rate"],
    "FPR_test": test_any["false_positive_rate"],
    "nuevas_inferencias_test": report["updated_test"]["new_inference_passes"],
}
"""
    ),
    markdown(
        r"""
## Lectura del resultado

`ensemble_soft_optimized` es el único seleccionado actual por la regla
lexicográfica. Su ventaja de BA sobre el ponderado heurístico es pequeña y el
IC bootstrap por vídeo incluye cero; además, los pesos suaves cambian entre
pliegues. Esta incertidumbre se reporta como limitación, sin reintroducir al
promedio simple como un segundo “ganador”.
"""
    ),
]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(notebook, DESTINATION)
print(DESTINATION)
