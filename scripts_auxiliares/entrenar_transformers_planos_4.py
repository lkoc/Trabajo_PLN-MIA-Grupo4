"""Fine-tuning plano de MiniLM y E5 para cuatro categorías de daño."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares import experimentos_jerarquicos_4 as h4
from scripts_auxiliares.flujo_hibrido_moderador import sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import tune_thresholds


ROOT = tm.ROOT
TARGET_LABELS = h4.TARGET_LABELS
MODEL_KEYS = tuple(tm.MODEL_SPECS)
METRICS_DIR = ROOT / "resultados" / "metricas" / "transformer_plano_4"
FIGURES_DIR = ROOT / "resultados" / "figuras" / "transformer_plano_4"
MODEL_DIR = ROOT / "modelos" / "transformer_plano_4"
REPORT_PATH = ROOT / "resultados" / "INFORME_TRANSFORMERS_PLANOS_4.md"
RESULT_PATH = METRICS_DIR / "resultado.json"
for _directory in (METRICS_DIR, FIGURES_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _fingerprint(context: dict) -> str:
    sources = {}
    for key in MODEL_KEYS:
        path = tm.MODEL_DIR / key / "best_checkpoint.pt"
        if not path.exists():
            raise FileNotFoundError(f"Falta checkpoint de 04_2: {path}")
        sources[key] = sha256_file(path)
    value = {
        "dataset": context["dataset_sha256"],
        "manifest": context["manifest_sha256"],
        "labels": TARGET_LABELS,
        "sources": sources,
        "max_epochs": tm.MAX_EPOCHS,
        "learning_rate": tm.LEARNING_RATE,
        "weight_decay": tm.WEIGHT_DECAY,
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_context() -> dict:
    context = h4.load_frozen_context()
    context["training_fingerprint_sha256"] = _fingerprint(context)
    return context


def dataset_summary(context: dict | None = None) -> pd.DataFrame:
    context = context or load_context()
    rows = []
    for split, frame in context["frames"].items():
        y = h4.four_targets(frame).astype(bool)
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "videos": frame["video_id"].astype(str).nunique(),
                "safe": int((~y.any(axis=1)).sum()),
                "damage": int(y.any(axis=1).sum()),
                **{
                    label: int(y[:, index].sum())
                    for index, label in enumerate(TARGET_LABELS)
                },
            }
        )
    return pd.DataFrame(rows)


def _initialize_from_04_2(model, key: str) -> dict:
    path = tm.MODEL_DIR / key / "best_checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    spec = tm.MODEL_SPECS[key]
    source_spec = checkpoint.get("model_spec", {})
    if (
        source_spec.get("model_id") != spec.model_id
        or source_spec.get("revision") != spec.revision
        or checkpoint.get("targets") != list(tm.DAMAGE_ORDER)
    ):
        raise ValueError(f"Checkpoint incompatible para {key}.")
    state = checkpoint["model_state"]
    backbone = {
        name.removeprefix("backbone."): value
        for name, value in state.items()
        if name.startswith("backbone.")
    }
    model.backbone.load_state_dict(backbone, strict=True)
    h4._copy_head(model.classifier, state, "classifier")
    return {
        "strategy": "warm_start_from_04_2_full_finetuning",
        "source": _artifact(path),
        "source_epoch": int(checkpoint["epoch"]),
        "copied": ["encoder", "mapped_four_category_head"],
        "head_merge": "mean of former ACOSO_PERSONAL and AMENAZA_DIRECTA rows",
        "thresholds_reused": False,
    }


def _train_model(key: str, context: dict, force: bool = False) -> dict:
    spec = tm.MODEL_SPECS[key]
    model_metrics = METRICS_DIR / key
    model_output = MODEL_DIR / key
    model_metrics.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    result_path = model_metrics / "resultado.json"
    if result_path.exists() and not force:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("training_fingerprint_sha256") != context["training_fingerprint_sha256"]:
            raise ValueError(f"Cambió el contrato de {key}; use force=True.")
        return cached

    tm.set_reproducibility()
    device = h4.device()
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    tokenizer.save_pretrained(model_output / "tokenizer")
    model = h4._runtime.ConditionalCategoryClassifier(spec).to(device)
    initialization = _initialize_from_04_2(model, key)
    frames = context["frames"]
    train_loader = h4._runtime._loader(
        frames["train"], tokenizer, spec.prefix, tm.TRAIN_BATCH_SIZE, True
    )
    validation_loader = h4._runtime._loader(
        frames["validation"], tokenizer, spec.prefix, tm.EVAL_BATCH_SIZE, False
    )
    y_train = h4.four_targets(frames["train"])
    y_validation = h4.four_targets(frames["validation"])
    pos_weights_np = tm.positive_weights(y_train, "sqrt_positive_weight")
    pos_weights = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)
    optimizer = AdamW(model.parameters(), lr=tm.LEARNING_RATE, weight_decay=tm.WEIGHT_DECAY)
    scheduler = tm.scheduler_for(optimizer, len(train_loader) * tm.MAX_EPOCHS)
    best_score, stale, history = -math.inf, 0, []
    best_path = model_output / "best_checkpoint.pt"
    last_path = model_output / "last_checkpoint.pt"
    start = perf_counter()
    for epoch in range(1, tm.MAX_EPOCHS + 1):
        model.train()
        cumulative, seen = 0.0, 0
        progress = tqdm(
            train_loader,
            desc=f"{spec.label} · 4 daños · época {epoch}/{tm.MAX_EPOCHS}",
            unit="lote",
        )
        for tokens, _, targets, weights, _ in progress:
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(h4._runtime._to_device(tokens, device))
            element = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weights, reduction="none"
            )
            loss = h4._runtime._weighted_mean(element.mean(dim=1), weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch = len(targets)
            cumulative += float(loss.detach()) * batch
            seen += batch
            progress.set_postfix(loss=f"{cumulative / seen:.4f}")
        validation_scores = h4._runtime._predict_categories(
            model, validation_loader, device, f"{key} · validation época {epoch}"
        )
        thresholds = tune_thresholds(y_validation.astype(np.int8), validation_scores)
        metrics, _, _ = h4.evaluate_four_scores(y_validation, validation_scores, thresholds)
        record = {
            "epoch": epoch,
            "training_loss": cumulative / seen,
            "validation_damage_pr_auc_macro": metrics["damage_pr_auc_macro"],
            "validation_damage_f1_macro": metrics["damage_f1_macro"],
            "validation_any_damage_recall": metrics["any_damage_recall"],
            "thresholds": thresholds.tolist(),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(model_metrics / "historial.csv", index=False)
        payload = {
            "model_state": model.state_dict(),
            "model_spec": asdict(spec),
            "epoch": epoch,
            "history": history,
            "thresholds": thresholds.tolist(),
            "targets": TARGET_LABELS,
            "dataset_sha256": context["dataset_sha256"],
            "training_fingerprint_sha256": context["training_fingerprint_sha256"],
            "initialization": initialization,
        }
        h4._runtime._save_checkpoint(last_path, payload)
        score = float(metrics["damage_pr_auc_macro"])
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            h4._runtime._save_checkpoint(best_path, payload)
        else:
            stale += 1
        tqdm.write(
            f"{key} época {epoch}: PR-AUC={score:.4f}; F1={metrics['damage_f1_macro']:.4f}"
        )
        if stale >= tm.EARLY_STOPPING_PATIENCE:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics, score_artifacts = {}, {}
    for split in ("validation", "test"):
        scores = h4._runtime._predict_categories(
            model,
            h4._runtime._loader(
                frames[split], tokenizer, spec.prefix, tm.EVAL_BATCH_SIZE, False
            ),
            device,
            f"{key} · {split} final",
        )
        score_path = model_metrics / f"scores_{split}.npy"
        np.save(score_path, scores)
        metrics, report, _ = h4.evaluate_four_scores(
            h4.four_targets(frames[split]),
            scores,
            np.asarray(checkpoint["thresholds"]),
        )
        split_metrics[split] = metrics
        report.to_csv(model_metrics / f"reporte_{split}.csv")
        score_artifacts[split] = _artifact(score_path)
    result = {
        "completed_at": tm.now_iso(),
        "model_key": key,
        "model": asdict(spec),
        "targets": TARGET_LABELS,
        "dataset_sha256": context["dataset_sha256"],
        "training_fingerprint_sha256": context["training_fingerprint_sha256"],
        "initialization": initialization,
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_completed": len(history),
        "best_validation_damage_pr_auc_macro": float(best_score),
        "training_seconds": perf_counter() - start,
        "positive_weights_sqrt": pos_weights_np.tolist(),
        "thresholds_selected_on_validation": checkpoint["thresholds"],
        "history": history,
        "metrics": split_metrics,
        "checkpoint": _artifact(best_path),
        "last_checkpoint": _artifact(last_path),
        "score_artifacts": score_artifacts,
    }
    tm.write_json(result_path, result)
    return result


def run_all(force: bool = False) -> dict:
    context = load_context()
    models = {key: _train_model(key, context, force=force) for key in MODEL_KEYS}
    comparison = pd.DataFrame(
        [
            {
                "model_key": key,
                "modelo": value["model"]["label"],
                "split": split,
                **{
                    metric: values[metric]
                    for metric in (
                        "damage_pr_auc_macro",
                        "damage_f1_macro",
                        "damage_recall_micro",
                        "any_damage_precision",
                        "any_damage_recall",
                        "missed_damage_as_safe",
                    )
                },
            }
            for key, value in models.items()
            for split, values in value["metrics"].items()
        ]
    )
    comparison.to_csv(METRICS_DIR / "comparacion.csv", index=False)
    validation = comparison.loc[comparison["split"].eq("validation")].sort_values(
        ["damage_pr_auc_macro", "damage_f1_macro"], ascending=False
    )
    winner_key = str(validation.iloc[0]["model_key"])
    result = {
        "completed_at": tm.now_iso(),
        "dataset": {
            "path": str(context["dataset_path"].relative_to(ROOT)),
            "sha256": context["dataset_sha256"],
            "manifest_sha256": context["manifest_sha256"],
            "split_counts": {
                split: len(frame) for split, frame in context["frames"].items()
            },
            "targets": TARGET_LABELS,
        },
        "models": models,
        "selection": {
            "partition": "validation",
            "metric": "damage_pr_auc_macro",
            "winner_key": winner_key,
            "winner_label": models[winner_key]["model"]["label"],
            "test_used_for_selection": False,
        },
        "comparison_artifact": _artifact(METRICS_DIR / "comparacion.csv"),
    }
    tm.write_json(RESULT_PATH, result)
    _write_report_and_figure(result, comparison)
    result["report_artifact"] = _artifact(REPORT_PATH)
    result["figure_artifact"] = _artifact(FIGURES_DIR / "comparacion_test.png")
    tm.write_json(RESULT_PATH, result)
    return result


def _write_report_and_figure(result: dict, comparison: pd.DataFrame) -> None:
    test = comparison.loc[comparison["split"].eq("test")]
    rows = "\n".join(
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | {row.damage_f1_macro:.4f} | "
        f"{row.any_damage_recall:.4f} | {int(row.missed_damage_as_safe)} |"
        for row in test.itertuples()
    )
    report = f"""# Transformers planos con cuatro daños

Fecha: {tm.now_iso()}

MiniLM multilingüe y E5-small se reentrenaron sobre el mismo dataset 4:1 y los mismos splits que Qwen `04_205`. Ambos parten de sus checkpoints de cinco daños de `04_2`: se copia el encoder, se conservan las filas compatibles y `ACOSO_AMENAZA` se inicializa promediando acoso y amenaza. La cabeza completa se vuelve a optimizar y los umbrales se recalibran exclusivamente en validation.

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{rows}

Ganador por validation: **{result['selection']['winner_label']}**. El test no intervino en selección. Ningún resultado autoriza autonomía sin validación humana independiente.

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    plot = test.set_index("modelo")[["damage_pr_auc_macro", "damage_f1_macro", "any_damage_recall"]]
    axis = plot.plot.bar(figsize=(11, 5))
    axis.set_ylim(0, 1)
    axis.set_title("Transformers planos · mismo test 4:1")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparacion_test.png", dpi=180, bbox_inches="tight")
    plt.close()
