"""Cascada y jerarquía multitarea sobre representaciones del Qwen de 04_205."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import hashlib
import json
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from scripts_auxiliares import entrenar_qwen_acoso_amenaza as q4
from scripts_auxiliares import entrenar_transformers_gruesos as tm
from scripts_auxiliares import experimentos_jerarquicos_4 as h4
from scripts_auxiliares.flujo_hibrido_moderador import sha256_file
from scripts_auxiliares.modelos_gruesos_moderador import tune_thresholds


ROOT = tm.ROOT
TARGET_LABELS = h4.TARGET_LABELS
METRICS_DIR = ROOT / "resultados" / "metricas" / "qwen_jerarquico_4"
FIGURES_DIR = ROOT / "resultados" / "figuras" / "qwen_jerarquico_4"
MODEL_DIR = ROOT / "modelos" / "qwen_jerarquico_4"
REPORT_PATH = ROOT / "resultados" / "INFORME_QWEN_JERARQUICO_4.md"
RESULT_PATH = METRICS_DIR / "resultado.json"
for _directory in (METRICS_DIR, FIGURES_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def _artifact(path: Path) -> dict:
    return {
        "path": tm.project_relative(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }


def _ids_sha256(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_completed_qwen() -> dict:
    operational = q4.load_operational_evaluation(load_scores=False, require_test=True)
    result = operational["training"]
    if result.get("status") != "completed":
        raise RuntimeError("El entrenamiento de 04_205 no figura como completado.")
    if int(result.get("max_epochs", 0)) < q4.MAX_EPOCHS:
        raise RuntimeError(
            f"04_205 corresponde al plan anterior de {result.get('max_epochs')} épocas. "
            f"Complete primero la extensión con máximo {q4.MAX_EPOCHS} épocas."
        )
    if int(result.get("epochs_completed", 0)) < q4.MAX_EPOCHS:
        raise RuntimeError(
            f"04_205 sólo completó {result.get('epochs_completed')} épocas. "
            f"Ejecute la época {q4.MAX_EPOCHS} forzada antes de continuar."
        )
    selection = operational["selection"]
    state_artifact = operational["artifacts"]["adapter_training_state"]
    return {
        "training": result,
        "selection": selection,
        "selected_epoch": operational["selected_epoch"],
        "selection_partition": selection["selection_partition"],
        "test_used_for_selection": False,
        "adapter_directory": tm.project_relative(operational["adapter_directory"]),
        "adapter_state": state_artifact,
        "best_adapter_state": state_artifact,
    }


def load_context(use_expanded_safe: bool = True) -> dict:
    qwen_status = _require_completed_qwen()
    frozen = h4.load_frozen_context()
    qwen_frames, qwen_audit = q4.load_frames()
    if qwen_audit["dataset_sha256"] != frozen["dataset_sha256"]:
        raise ValueError("04_205 y los experimentos jerárquicos no usan el mismo dataset.")
    if use_expanded_safe:
        train, expansion = h4.expanded_safe_gate_training_frame(frozen)
    else:
        train = frozen["frames"]["train"].copy()
        y = h4.four_targets(train).astype(bool)
        expansion = {
            "purpose": "strict_4a1_control",
            "expanded_gate_train_rows": len(train),
            "expanded_gate_safe_rows": int((~y.any(axis=1)).sum()),
            "expanded_gate_damage_rows": int(y.any(axis=1).sum()),
            "additional_safe_rows": 0,
            "validation_or_test_videos_used": False,
        }
    return {
        "frames": qwen_frames,
        "feature_train_frame": train,
        "frozen": frozen,
        "qwen": qwen_status,
        "qwen_audit": qwen_audit,
        "expansion": expansion,
        "use_expanded_safe": use_expanded_safe,
    }


class _TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame["text"].astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


class _TextCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=q4.MAX_LENGTH,
            return_tensors="pt",
        )


@torch.inference_mode()
def _extract(model, loader, description: str) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    output = []
    for tokens in tqdm(loader, desc=description, unit="lote"):
        logits = model(
            **{key: value.to(device) for key, value in tokens.items()}
        ).logits
        output.append(logits.cpu().numpy())
    return np.vstack(output).astype(np.float32)


def _feature_path(split: str) -> tuple[Path, Path]:
    return (
        METRICS_DIR / f"qwen_features_{split}.npy",
        METRICS_DIR / f"qwen_features_{split}.manifest.json",
    )


def extract_features(context: dict, force: bool = False) -> dict[str, np.ndarray]:
    adapter_sha = context["qwen"]["adapter_state"]["sha256"]
    feature_frames = {
        "train_expanded": context["feature_train_frame"],
        "validation": context["frames"]["validation"],
        "test": context["frames"]["test"],
    }
    expected = {
        split: {
            "rows": len(frame),
            "chunk_ids_sha256": _ids_sha256(frame["chunk_id"]),
            "adapter_state_sha256": adapter_sha,
            "dataset_sha256": context["frozen"]["dataset_sha256"],
            "output_labels": q4.OUTPUT_LABELS,
        }
        for split, frame in feature_frames.items()
    }
    values, pending = {}, []
    for split in feature_frames:
        path, manifest_path = _feature_path(split)
        if path.exists() and manifest_path.exists() and not force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest == expected[split]:
                array = np.load(path)
                if array.shape == (len(feature_frames[split]), len(q4.OUTPUT_LABELS)):
                    values[split] = array
                    continue
        pending.append(split)
    if pending:
        device = q4.device()
        tokenizer = q4.tokenizer()
        adapter_directory = ROOT / context["qwen"]["adapter_directory"]
        model = q4.load_adapter(adapter_directory, device)
        for split in pending:
            loader = DataLoader(
                _TextDataset(feature_frames[split]),
                batch_size=q4.EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=_TextCollator(tokenizer),
            )
            values[split] = _extract(model, loader, f"Qwen congelado · {split}")
            path, manifest_path = _feature_path(split)
            np.save(path, values[split])
            tm.write_json(manifest_path, expected[split])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return values


def _fit_logistic_head(x, y, weights, seed=tm.SEED):
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _cascade(context: dict, features: dict) -> dict:
    train = context["feature_train_frame"]
    base_train = context["frames"]["train"]
    base_rows = len(base_train)
    y_expanded = h4.four_targets(train).astype(np.int8)
    y_base = h4.four_targets(base_train).astype(np.int8)
    weights_expanded = tm.source_weights(train)
    weights_base = tm.source_weights(base_train)
    gate = _fit_logistic_head(
        features["train_expanded"], y_expanded.any(axis=1), weights_expanded
    )
    gate_base = gate.predict_proba(features["train_expanded"][:base_rows])[:, 1]
    stage_frame, selection = h4._runtime._stage2_training_frame(base_train, gate_base)
    position = {str(value): index for index, value in enumerate(base_train["chunk_id"])}
    indices = np.asarray([position[str(value)] for value in stage_frame["chunk_id"]])
    heads = [
        _fit_logistic_head(
            features["train_expanded"][:base_rows][indices],
            y_base[indices, column],
            weights_base[indices],
            seed=tm.SEED + column + 1,
        )
        for column in range(4)
    ]
    scores, gates = {}, {}
    for split in ("validation", "test"):
        x = features[split]
        gates[split] = gate.predict_proba(x)[:, 1]
        conditional = np.column_stack(
            [head.predict_proba(x)[:, 1] for head in heads]
        )
        scores[split] = gates[split][:, None] * conditional
    thresholds = tune_thresholds(
        h4.four_targets(context["frames"]["validation"]).astype(np.int8),
        scores["validation"],
    )
    path = MODEL_DIR / "cascada_qwen_features.joblib"
    joblib.dump(
        {
            "gate": gate,
            "category_heads": heads,
            "stage2_selection": selection,
            "thresholds": thresholds,
            "feature_labels": q4.OUTPUT_LABELS,
        },
        path,
    )
    return {
        "key": "qwen_frozen_cascade",
        "label": "Qwen congelado + cascada calibrada",
        "scores": scores,
        "gates": gates,
        "thresholds": thresholds,
        "training": {
            "gate_rows": len(train),
            "gate_safe_rows": int((~y_expanded.astype(bool).any(axis=1)).sum()),
            "category_rows": len(stage_frame),
            "stage2_selection": selection,
            "model": _artifact(path),
        },
    }


class _JointHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.10)
        )
        self.gate = nn.Linear(hidden_size, 1)
        self.categories = nn.Linear(hidden_size, 4)

    def forward(self, values):
        hidden = self.shared(values)
        return self.gate(hidden).squeeze(-1), self.categories(hidden)


@torch.inference_mode()
def _joint_scores(model, x: np.ndarray, mean, scale, device):
    model.eval()
    gates, categories = [], []
    for start in range(0, len(x), 1_024):
        batch = torch.tensor(
            (x[start : start + 1_024] - mean) / scale,
            dtype=torch.float32,
            device=device,
        )
        gate, category = model(batch)
        gates.append(torch.sigmoid(gate).cpu().numpy())
        categories.append(torch.sigmoid(category).cpu().numpy())
    gate = np.concatenate(gates)
    conditional = np.vstack(categories)
    return gate, gate[:, None] * conditional


def _joint(context: dict, features: dict, force: bool = False) -> dict:
    train = context["feature_train_frame"]
    base_rows = len(context["frames"]["train"])
    x = features["train_expanded"].astype(np.float32)
    y = h4.four_targets(train).astype(np.float32)
    any_damage = y.any(axis=1).astype(np.float32)
    weights = tm.source_weights(train)
    category_mask = np.zeros(len(train), dtype=np.float32)
    category_mask[:base_rows] = 1.0
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True).clip(min=1e-4)
    y_base = y[:base_rows]
    binary_weight = h4._runtime._sqrt_pos_weight(any_damage)
    category_weights = tm.positive_weights(y_base, "sqrt_positive_weight")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tm.set_reproducibility()
    model = _JointHead(x.shape[1]).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    binary_pos = torch.tensor(binary_weight, dtype=torch.float32, device=device)
    category_pos = torch.tensor(category_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(tm.SEED)
    best, stale, history = -math.inf, 0, []
    best_path = MODEL_DIR / "qwen_joint_head_best.pt"
    last_path = MODEL_DIR / "qwen_joint_head_last.pt"
    y_validation = h4.four_targets(context["frames"]["validation"])
    start_time = perf_counter()
    for epoch in range(1, 31):
        permutation = rng.permutation(len(x))
        model.train()
        cumulative, seen = 0.0, 0
        progress = tqdm(range(0, len(x), 512), desc=f"Qwen head multitarea · época {epoch}/30", unit="lote")
        for start in progress:
            indices = permutation[start : start + 512]
            batch_x = torch.tensor((x[indices] - mean) / scale, dtype=torch.float32, device=device)
            batch_any = torch.tensor(any_damage[indices], dtype=torch.float32, device=device)
            batch_y = torch.tensor(y[indices], dtype=torch.float32, device=device)
            batch_weights = torch.tensor(weights[indices], dtype=torch.float32, device=device)
            batch_mask = torch.tensor(category_mask[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            gate_logits, category_logits = model(batch_x)
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                gate_logits, batch_any, pos_weight=binary_pos, reduction="none"
            )
            category_loss = nn.functional.binary_cross_entropy_with_logits(
                category_logits, batch_y, pos_weight=category_pos, reduction="none"
            ).mean(dim=1)
            gate_probability = torch.sigmoid(gate_logits)
            maximum_category = torch.sigmoid(category_logits).max(dim=1).values
            consistency = torch.relu(maximum_category - gate_probability).square()
            per_sample = 0.5 * binary_loss + category_loss * batch_mask + 0.1 * consistency
            loss = (per_sample * batch_weights).sum() / batch_weights.sum().clamp(min=1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cumulative += float(loss.detach()) * len(indices)
            seen += len(indices)
            progress.set_postfix(loss=f"{cumulative / seen:.4f}")
        gate_validation, scores_validation = _joint_scores(
            model, features["validation"], mean, scale, device
        )
        thresholds = tune_thresholds(y_validation.astype(np.int8), scores_validation)
        metrics, _, _ = h4.evaluate_four_scores(y_validation, scores_validation, thresholds)
        score = float(metrics["damage_pr_auc_macro"])
        record = {
            "epoch": epoch,
            "training_loss": cumulative / seen,
            "validation_damage_pr_auc_macro": score,
            "validation_damage_f1_macro": metrics["damage_f1_macro"],
            "validation_binary_pr_auc": float(
                average_precision_score(y_validation.any(axis=1), gate_validation)
            ),
            "thresholds": thresholds.tolist(),
        }
        history.append(record)
        payload = {
            "model_state": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "epoch": epoch,
            "history": history,
            "thresholds": thresholds,
            "feature_labels": q4.OUTPUT_LABELS,
        }
        torch.save(payload, last_path)
        if score > best + 1e-6:
            best, stale = score, 0
            torch.save(payload, best_path)
        else:
            stale += 1
        if stale >= 4:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    scores, gates = {}, {}
    for split in ("validation", "test"):
        gates[split], scores[split] = _joint_scores(
            model, features[split], checkpoint["mean"], checkpoint["scale"], device
        )
    pd.DataFrame(history).to_csv(METRICS_DIR / "historial_qwen_joint.csv", index=False)
    return {
        "key": "qwen_frozen_joint",
        "label": "Qwen congelado + cabeza jerárquica multitarea",
        "scores": scores,
        "gates": gates,
        "thresholds": np.asarray(checkpoint["thresholds"]),
        "training": {
            "binary_rows": len(train),
            "binary_safe_rows": int((~y.astype(bool).any(axis=1)).sum()),
            "category_rows": base_rows,
            "category_loss_masked_on_extra_safe": True,
            "best_epoch": int(checkpoint["epoch"]),
            "training_seconds": perf_counter() - start_time,
            "best_checkpoint": _artifact(best_path),
            "last_checkpoint": _artifact(last_path),
            "history": history,
        },
    }


def _finalize_comparison(
    context: dict,
    models: list[dict],
    operational: dict,
    bootstrap_replicates: int,
) -> dict:
    reference_scores = operational["scores"]
    reference_thresholds = np.asarray(
        operational["thresholds_selected_on_validation"], dtype=float
    )
    rows, decisions = [], {}
    y_test = h4.four_targets(context["frames"]["test"])
    for model in models:
        for split in ("validation", "test"):
            metrics, report, _ = h4.evaluate_four_scores(
                h4.four_targets(context["frames"][split]),
                model["scores"][split],
                model["thresholds"],
            )
            rows.append(
                {
                    "model_key": model["key"],
                    "modelo": model["label"],
                    "split": split,
                    **{key: value for key, value in metrics.items() if key != "category_recall"},
                }
            )
            report.to_csv(METRICS_DIR / f"reporte_{model['key']}_{split}.csv")
            np.save(METRICS_DIR / f"scores_{model['key']}_{split}.npy", model["scores"][split])
        bootstrap = h4._runtime.paired_cluster_bootstrap(
            y_test,
            reference_scores["test"],
            reference_thresholds,
            model["scores"]["test"],
            model["thresholds"],
            context["frames"]["test"]["video_id"].astype(str).to_numpy(),
            replicates=bootstrap_replicates,
            seed=tm.SEED,
        )
        bootstrap.insert(0, "candidate_key", model["key"])
        bootstrap.to_csv(METRICS_DIR / f"bootstrap_{model['key']}.csv", index=False)
        decisions[model["key"]] = h4._runtime._decision_from_bootstrap(bootstrap)
    for split in ("validation", "test"):
        metrics, _, _ = h4.evaluate_four_scores(
            h4.four_targets(context["frames"][split]),
            reference_scores[split],
            reference_thresholds,
        )
        rows.append(
            {
                "model_key": "qwen4_flat",
                "modelo": (
                    "Qwen 04_205 plano · época operativa "
                    f"{operational['selected_epoch']}"
                ),
                "split": split,
                **{key: value for key, value in metrics.items() if key != "category_recall"},
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(METRICS_DIR / "comparacion.csv", index=False)
    validation = comparison.loc[
        comparison["split"].eq("validation") & ~comparison["model_key"].eq("qwen4_flat")
    ].sort_values(["damage_pr_auc_macro", "damage_f1_macro"], ascending=False)
    winner = str(validation.iloc[0]["model_key"])
    result = {
        "completed_at": tm.now_iso(),
        "dataset": {
            "sha256": context["frozen"]["dataset_sha256"],
            "targets": TARGET_LABELS,
            "same_splits_as_04_205": True,
            "expanded_safe": context["expansion"],
        },
        "qwen_feature_extractor": context["qwen"],
        "qwen_flat_reference": {
            "selected_epoch": operational["selected_epoch"],
            "selected_adapter": operational["selected_adapter"],
            "selection_partition": operational["selection"]["selection_partition"],
            "selection_rule": operational["selection"]["selection_rule"],
            "test_used_for_selection": False,
            "thresholds_selected_on_validation": reference_thresholds.tolist(),
            "artifacts": operational["artifacts"],
        },
        "feature_outputs": q4.OUTPUT_LABELS,
        "models": {
            model["key"]: {
                "label": model["label"],
                "thresholds": model["thresholds"].tolist(),
                "training": model["training"],
            }
            for model in models
        },
        "selection": {
            "partition": "validation",
            "metric": "damage_pr_auc_macro",
            "winner_key": winner,
            "winner_label": next(model["label"] for model in models if model["key"] == winner),
            "test_used_for_selection": False,
        },
        "paired_decisions_vs_qwen_flat": decisions,
        "bootstrap_replicates": bootstrap_replicates,
        "comparison_artifact": _artifact(METRICS_DIR / "comparacion.csv"),
    }
    tm.write_json(RESULT_PATH, result)
    _write_report_figure(result, comparison)
    result["report_artifact"] = _artifact(REPORT_PATH)
    result["figure_artifact"] = _artifact(FIGURES_DIR / "comparacion_test.png")
    tm.write_json(RESULT_PATH, result)
    return result


def run_experiment(
    force: bool = False,
    use_expanded_safe: bool = True,
    bootstrap_replicates: int = tm.BOOTSTRAP_REPLICATES,
) -> dict:
    context = load_context(use_expanded_safe=use_expanded_safe)
    features = extract_features(context, force=force)
    models = [_cascade(context, features), _joint(context, features, force=force)]
    operational = q4.load_operational_evaluation(load_scores=True, require_test=True)
    return _finalize_comparison(
        context,
        models,
        operational,
        bootstrap_replicates=bootstrap_replicates,
    )


def refresh_comparison_from_existing(
    *,
    use_expanded_safe: bool = True,
    bootstrap_replicates: int = tm.BOOTSTRAP_REPLICATES,
) -> dict:
    """Recalcula la comparación con el Qwen operativo sin reentrenar cabezas."""
    if not RESULT_PATH.is_file():
        raise FileNotFoundError(
            "No existe resultado.json de 04_206; ejecute primero el experimento."
        )
    previous = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    context = load_context(use_expanded_safe=use_expanded_safe)
    models = []
    for key, record in previous["models"].items():
        score_paths = {
            split: METRICS_DIR / f"scores_{key}_{split}.npy"
            for split in ("validation", "test")
        }
        missing = [tm.project_relative(path) for path in score_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan scores jerárquicos para refrescar la comparación:\n"
                + "\n".join(missing)
            )
        models.append(
            {
                "key": key,
                "label": record["label"],
                "thresholds": np.asarray(record["thresholds"], dtype=float),
                "training": record["training"],
                "scores": {
                    split: np.load(path) for split, path in score_paths.items()
                },
            }
        )
    operational = q4.load_operational_evaluation(load_scores=True, require_test=True)
    return _finalize_comparison(
        context,
        models,
        operational,
        bootstrap_replicates=bootstrap_replicates,
    )


def _write_report_figure(result: dict, comparison: pd.DataFrame) -> None:
    test = comparison.loc[comparison["split"].eq("test")]
    flat = test.loc[test["model_key"].eq("qwen4_flat")].iloc[0]
    winner = test.loc[
        test["model_key"].eq(result["selection"]["winner_key"])
    ].iloc[0]
    replace_flat = bool(
        result["paired_decisions_vs_qwen_flat"]
        .get(result["selection"]["winner_key"], {})
        .get("replace_flat_model", False)
    )
    rows = "\n".join(
        f"| {row.modelo} | {row.damage_pr_auc_macro:.4f} | {row.damage_f1_macro:.4f} | "
        f"{row.any_damage_recall:.4f} | {int(row.missed_damage_as_safe)} |"
        for row in test.itertuples()
    )
    report = f"""# Qwen plano, cascada y jerárquico multitarea con cuatro daños

Fecha: {tm.now_iso()}

El adaptador operativo de la época **{result['qwen_flat_reference']['selected_epoch']}**, elegido en validación antes de consultar test, permanece congelado y produce 21 logits: cuatro operativos y 17 auxiliares. Sobre esas mismas representaciones se entrenan una cascada logística y una cabeza neuronal multitarea. La supervisión binaria usa {result['dataset']['expanded_safe']['expanded_gate_safe_rows']:,} SEGURO de train; la pérdida temática de la cabeza conjunta se enmascara en los negativos adicionales. Validation/test son exactamente los de `04_205`, y tanto las cabezas jerárquicas como la referencia plana corresponden al mismo checkpoint operativo.

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
{rows}

Ganador entre los dos diseños jerárquicos por validation: **{result['selection']['winner_label']}**. Las decisiones pareadas frente a Qwen plano están en `resultado.json`. Esta variante es Qwen congelado más cabezas jerárquicas; no es un segundo fine-tuning end-to-end del LLM.

## Conclusión sobre el esquema jerárquico

El ganador jerárquico obtiene en test PR-AUC macro {winner['damage_pr_auc_macro']:.4f}, F1 macro {winner['damage_f1_macro']:.4f}, recall de daño {winner['any_damage_recall']:.4f} y deja {int(winner['missed_damage_as_safe'])} daños como seguros. La referencia Qwen plana operativa obtiene respectivamente {flat['damage_pr_auc_macro']:.4f}, {flat['damage_f1_macro']:.4f}, {flat['any_damage_recall']:.4f} y {int(flat['missed_damage_as_safe'])}. Las diferencias del ganador jerárquico frente al plano son {winner['damage_pr_auc_macro'] - flat['damage_pr_auc_macro']:+.4f} en PR-AUC, {winner['damage_f1_macro'] - flat['damage_f1_macro']:+.4f} en F1, {winner['any_damage_recall'] - flat['any_damage_recall']:+.4f} en recall y {int(winner['missed_damage_as_safe'] - flat['missed_damage_as_safe']):+d} falsos negativos de daño.

Por tanto, **{'hay evidencia para reemplazar el Qwen plano por el esquema jerárquico' if replace_flat else 'estos esquemas jerárquicos no resultan mejores que Qwen plano y no deben reemplazarlo'}** bajo el criterio pareado predefinido. Este resultado tampoco autoriza autonomía: cualquier uso operativo requiere validación humana independiente y un piloto prospectivo.

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Zhou, J., Ma, C., Long, D., Xu, G., Ding, N., Zhang, H., Xie, P., & Liu, G. (2020). Hierarchy-aware global model for hierarchical text classification. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1106–1117). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.104
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    plot = test.set_index("modelo")[["damage_pr_auc_macro", "damage_f1_macro", "any_damage_recall"]]
    axis = plot.plot.bar(figsize=(12, 5))
    axis.set_ylim(0, 1)
    axis.set_title("Qwen plano y jerárquico · mismo test 4:1")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparacion_test.png", dpi=180, bbox_inches="tight")
    plt.close()
