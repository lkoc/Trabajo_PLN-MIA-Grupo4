from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .datasets import load_split
from .device import resolve_device, torch_device_name
from .io import canonical_json_sha256, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from .manifests import artifact_reference, build_manifest, save_manifest
from .models import TRANSFORMER_SPECS, TrainingSpecification
from .taxonomy import load_taxonomy
from .training import calibrate_thresholds, classification_metrics, encode_targets


TRAINING_ENGINE_VERSION = "3.0.0"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _experiment_signature(dataset: Path, experiment: str, configuration: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "engine": TRAINING_ENGINE_VERSION,
            "experiment": experiment,
            "dataset_sha256": sha256_file(dataset),
            "configuration": configuration,
            "taxonomy": load_taxonomy().contract_id,
        }
    )


def _complete_candidate(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    candidate = json.loads(path.read_text(encoding="utf-8"))
    manifest = Path(candidate.get("checkpoint_manifest", ""))
    if not manifest.is_absolute():
        manifest = path.parent / manifest
    if candidate.get("run_signature") != signature or candidate.get("status") != "complete" or not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for record in payload.get("files", []):
        target = manifest.parent / record["path"]
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            return None
    return candidate


def _checkpoint_manifest(run_dir: Path, paths: Iterable[Path]) -> Path:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.append(path)
    records = []
    for path in sorted(set(files)):
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = run_dir / "checkpoint_manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": "2.1.0",
            "created_at": _utc_iso(),
            "files": records,
            "aggregate_sha256": canonical_json_sha256(records),
        },
    )
    return manifest


def _write_predictions(
    path: Path,
    rows: Sequence[dict[str, Any]],
    scores: np.ndarray,
) -> None:
    taxonomy = load_taxonomy()
    output = []
    for row, vector in zip(rows, scores, strict=True):
        output.append(
            {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "split": row["split"],
                "scores": {
                    label: float(vector[index])
                    for index, label in enumerate(taxonomy.target_labels)
                },
                "true_labels": row["coarse_labels"],
            }
        )
    write_jsonl_atomic(path, output)


def _evaluate(
    run_dir: Path,
    validation_rows: Sequence[dict[str, Any]],
    validation_scores: np.ndarray,
    test_rows: Sequence[dict[str, Any]],
    test_scores: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    y_validation = encode_targets(validation_rows)
    y_test = encode_targets(test_rows)
    thresholds = calibrate_thresholds(y_validation, validation_scores)
    validation_metrics = classification_metrics(y_validation, validation_scores, thresholds)
    test_metrics = classification_metrics(y_test, test_scores, thresholds)
    write_json_atomic(run_dir / "thresholds.json", thresholds)
    write_json_atomic(
        run_dir / "metrics.json",
        {
            "selection_split": "validation",
            "test_used_for_selection": False,
            "thresholds": thresholds,
            "validation": validation_metrics,
            "test": test_metrics,
        },
    )
    _write_predictions(run_dir / "predictions_validation.jsonl", validation_rows, validation_scores)
    _write_predictions(run_dir / "predictions_test.jsonl", test_rows, test_scores)
    return thresholds, validation_metrics, test_metrics


def _dataset_splits(dataset: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in read_jsonl(dataset)]
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    test = [row for row in rows if row.get("split") == "test"]
    if not train or not validation or not test:
        raise ValueError(
            "El snapshot necesita filas en train, validation y test; agregue videos, no redistribuya chunks"
        )
    return train, validation, test


def _classical_scores(model: Any, texts: Sequence[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(texts), dtype=float)
    elif hasattr(model, "decision_function"):
        values = _sigmoid(np.asarray(model.decision_function(texts), dtype=float))
    else:
        values = np.asarray(model.predict(texts), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return values


def train_classical_experiments(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    model_names: Iterable[str] | None = None,
    max_features: int = 150000,
) -> dict[str, Any]:
    """Ejecuta fit→calibración→test para los cinco baselines clásicos."""

    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc

    dataset = Path(dataset_path).resolve()
    output = Path(output_root)
    candidates_spec = {
        "dummy": DummyClassifier(strategy="prior"),
        "complement_nb": ComplementNB(alpha=1.0),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "sgd_incremental": SGDClassifier(loss="log_loss", random_state=20260805),
    }
    selected_names = list(candidates_spec) if model_names is None else list(dict.fromkeys(model_names))
    unknown = sorted(set(selected_names) - set(candidates_spec))
    if unknown:
        raise ValueError(f"Modelos clasicos desconocidos: {unknown}")
    if not selected_names:
        raise ValueError("Debe seleccionar al menos un modelo clasico")
    if max_features < 100:
        raise ValueError("max_features debe ser al menos 100")
    candidates_spec = {name: candidates_spec[name] for name in selected_names}

    configuration: dict[str, Any] = {"suite": "classical_v3", "seed": 20260805}
    if model_names is not None or max_features != 150000:
        configuration.update(
            {
                "mode": "bounded_subset",
                "models": selected_names,
                "max_features": max_features,
            }
        )
    signature = _experiment_signature(dataset, "classical_suite", configuration)
    run_dir = output / "runs" / f"classical-{signature[:16]}"
    complete = run_dir / "suite_complete.json"
    if complete.is_file() and not force:
        state = json.loads(complete.read_text(encoding="utf-8"))
        candidates = [_complete_candidate(Path(path), signature) for path in state.get("candidate_paths", [])]
        if candidates and all(candidate is not None for candidate in candidates):
            return {"status": "noop", "run_signature": signature, "candidates": candidates}

    train, validation, test = _dataset_splits(dataset)
    train_texts = [str(row["text"]) for row in train]
    targets = encode_targets(train)
    validation_texts = [str(row["text"]) for row in validation]
    test_texts = [str(row["text"]) for row in test]
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for name, estimator in candidates_spec.items():
        model_dir = run_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1 if len(train) < 100 else 2,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", OneVsRestClassifier(estimator, n_jobs=1)),
            ]
        )
        pipeline.fit(train_texts, targets)
        validation_scores = _classical_scores(pipeline, validation_texts)
        test_scores = _classical_scores(pipeline, test_texts)
        thresholds, validation_metrics, test_metrics = _evaluate(
            model_dir, validation, validation_scores, test, test_scores
        )
        checkpoint = model_dir / "model.joblib"
        joblib.dump(pipeline, checkpoint)
        bundle = model_dir / "inference.json"
        write_json_atomic(
            bundle,
            {
                "type": "sklearn_joblib",
                "model": checkpoint.name,
                "target_labels": list(load_taxonomy().target_labels),
            },
        )
        manifest = _checkpoint_manifest(model_dir, [checkpoint, bundle])
        candidate = {
            "schema_version": "2.1.0",
            "candidate_id": f"classical-{name}-{signature[:12]}",
            "experiment": name,
            "model_family": f"classical:{name}",
            "run_signature": signature,
            "dataset": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "target_labels": list(load_taxonomy().target_labels),
            "thresholds": thresholds,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "metrics_path": "metrics.json",
            "checkpoint_manifest": manifest.name,
            "inference": {"type": "sklearn_joblib", "bundle": bundle.name},
            "hardware": {"backend": "cpu", "requested": "cpu", "device_name": "CPU", "dtype": "float64"},
            "warm_start_from": None,
            "status": "complete",
            "completed_at": _utc_iso(),
        }
        candidate_path = model_dir / "candidate.json"
        write_json_atomic(candidate_path, candidate)
        save_manifest(
            model_dir / "run_manifest.json",
            build_manifest(
                run_id=candidate["candidate_id"],
                stage="03_entrenamiento",
                inputs=[artifact_reference(dataset, "model_ready_snapshot")],
                outputs=[
                    artifact_reference(candidate_path, "candidate"),
                    artifact_reference(manifest, "checkpoint_manifest"),
                    artifact_reference(model_dir / "metrics.json", "metrics"),
                ],
                configuration={"engine": TRAINING_ENGINE_VERSION, **configuration, "model": name},
                hardware=candidate["hardware"],
                counters={"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
            ),
        )
        candidate["candidate_path"] = str(candidate_path)
        candidates.append(candidate)
    write_json_atomic(
        complete,
        {
            "run_signature": signature,
            "candidate_paths": [candidate["candidate_path"] for candidate in candidates],
            "completed_at": _utc_iso(),
        },
    )
    return {"status": "trained", "run_signature": signature, "candidates": candidates}


class _TokenizedRows:
    def __init__(
        self,
        tokenizer: Any,
        rows: Sequence[dict[str, Any]],
        targets: np.ndarray,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            [str(row["text"]) for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.targets[index], dtype=torch.float32)
        return item


def _output_targets(rows: Sequence[dict[str, Any]], experiment: str) -> tuple[np.ndarray, list[str]]:
    taxonomy = load_taxonomy()
    coarse = encode_targets(rows).astype(np.float32)
    if experiment != "multitask":
        return coarse, list(taxonomy.target_labels)
    fine = np.asarray(
        [[int(label in row.get("fine_labels", [])) for label in taxonomy.fine_labels] for row in rows],
        dtype=np.float32,
    )
    flags = np.asarray(
        [[int(flag in row.get("flags_reference_only", [])) for flag in taxonomy.flags] for row in rows],
        dtype=np.float32,
    )
    return np.concatenate([coarse, fine, flags], axis=1), [
        *taxonomy.target_labels,
        *(f"fine:{label}" for label in taxonomy.fine_labels),
        *(f"flag:{flag}" for flag in taxonomy.flags),
    ]


def _build_hf_model(
    spec: TrainingSpecification,
    labels: Sequence[str],
    *,
    model_source: str | Path | None = None,
    lora: bool = False,
    adapter_source: str | Path | None = None,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc
    if adapter_source:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise RuntimeError("Instale PEFT mediante moderacion-peru[entrenamiento]") from exc
        adapter = str(adapter_source)
        peft_config = PeftConfig.from_pretrained(adapter)
        source = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(adapter)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        id2label = {index: label for index, label in enumerate(labels)}
        base = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=len(labels),
            id2label=id2label,
            label2id={label: index for index, label in id2label.items()},
            problem_type="multi_label_classification",
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, PeftModel.from_pretrained(base, adapter, is_trainable=True)
    source = str(model_source or spec.model_id)
    revision = None if model_source else spec.revision
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    id2label = {index: label for index, label in enumerate(labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        revision=revision,
        num_labels=len(labels),
        id2label=id2label,
        label2id={label: index for index, label in id2label.items()},
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=not bool(model_source),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("Instale PEFT mediante moderacion-peru[entrenamiento]") from exc
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    return tokenizer, model


def _last_candidate(output_root: Path, experiment: str, dataset_sha: str) -> dict[str, Any] | None:
    candidates = []
    for path in output_root.rglob("candidate.json") if output_root.exists() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("experiment") == experiment and row.get("status") == "complete" and row.get("dataset_sha256") != dataset_sha:
            row["candidate_path"] = str(path)
            candidates.append(row)
    return max(candidates, key=lambda row: row.get("completed_at", ""), default=None)


def _candidate_asset(candidate: dict[str, Any], value: str | Path | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(Path(candidate["candidate_path"]).parent / path)


def _fit_hf(
    model: Any,
    tokenizer: Any,
    train_rows: Sequence[dict[str, Any]],
    targets: np.ndarray,
    spec: TrainingSpecification,
    training_dir: Path,
    hardware: Any,
    *,
    structured_penalty: float = 0.0,
    output_weights: Sequence[float] | None = None,
) -> None:
    try:
        import torch
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError("Instale PyTorch, accelerate y Transformers") from exc

    dataset = _TokenizedRows(tokenizer, train_rows, targets, spec.max_length)

    class StructuredTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            elementwise = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            if output_weights is not None:
                weights = torch.as_tensor(output_weights, device=logits.device, dtype=logits.dtype)
                elementwise = elementwise * weights.unsqueeze(0)
            loss = elementwise.mean()
            if structured_penalty and logits.shape[1] == 5:
                probabilities = torch.sigmoid(logits)
                conflict = probabilities[:, 0] * probabilities[:, 1:].amax(dim=1)
                loss = loss + structured_penalty * conflict.mean()
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(training_dir),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation,
        learning_rate=spec.learning_rate,
        seed=spec.seed,
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=max(1, math.ceil(len(dataset) / max(1, spec.batch_size * 5))),
        report_to=[],
        use_cpu=hardware.backend == "cpu",
        fp16=hardware.backend in {"cuda", "rocm"} and hardware.dtype == "float16",
        bf16=hardware.backend in {"cuda", "rocm", "xpu"} and hardware.dtype == "bfloat16",
        dataloader_pin_memory=hardware.backend != "cpu",
    )
    trainer = StructuredTrainer(model=model, args=arguments, train_dataset=dataset)
    checkpoint = get_last_checkpoint(str(training_dir)) if training_dir.is_dir() else None
    trainer.train(resume_from_checkpoint=checkpoint)


def _predict_hf(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    max_length: int,
    hardware: Any,
    output_count: int,
) -> np.ndarray:
    import torch

    dummy = np.zeros((len(rows), output_count), dtype=np.float32)
    dataset = _TokenizedRows(tokenizer, rows, dummy, max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    device = torch_device_name(hardware)
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def _save_hf(model: Any, tokenizer: Any, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)


def train_neural_experiment(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    experiment: str,
    device: str = "auto",
    force: bool = False,
    spec_key: str | None = None,
) -> dict[str, Any]:
    """Cierra fit→calibración→test→candidato para una familia neuronal.

    Experimentos admitidos: ``flat_minilm``, ``flat_e5``, ``cascade``,
    ``multitask``, ``qwen_lora`` y ``qwen_structured``.
    """

    allowed = {"flat_minilm", "flat_e5", "cascade", "multitask", "qwen_lora", "qwen_structured"}
    if experiment not in allowed:
        raise ValueError(f"Experimento desconocido: {experiment}")
    key = spec_key or (
        "qwen_lora" if experiment in {"qwen_lora", "qwen_structured"} else "e5"
    )
    if experiment == "flat_minilm":
        key = "minilm"
    if experiment == "flat_e5":
        key = "e5"
    spec = TRANSFORMER_SPECS[key]
    dataset = Path(dataset_path).resolve()
    output = Path(output_root)
    hardware = resolve_device(device)
    configuration = {
        "experiment": experiment,
        "specification": asdict(spec),
        "structured_penalty": 0.2 if experiment == "qwen_structured" else 0.0,
        "hardware_backend": hardware.backend,
        "dtype": hardware.dtype,
        "multitask_weights": {"coarse": 1.0, "fine": 0.3, "flags": 0.2} if experiment == "multitask" else None,
    }
    signature = _experiment_signature(dataset, experiment, configuration)
    run_dir = output / "runs" / f"{experiment}-{signature[:16]}"
    candidate_path = run_dir / "candidate.json"
    if not force:
        complete = _complete_candidate(candidate_path, signature)
        if complete is not None:
            return {"status": "noop", "candidate": complete, "run_signature": signature}
    train, validation, test = _dataset_splits(dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = _last_candidate(output, experiment, sha256_file(dataset))
    taxonomy = load_taxonomy()

    if experiment == "cascade":
        damage_indices = [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels]
        train_full = encode_targets(train).astype(np.float32)
        gate_targets = train_full[:, damage_indices].max(axis=1, keepdims=True)
        harmful = gate_targets[:, 0] == 1
        if not harmful.any():
            raise ValueError("La cascada necesita daños explícitos en train")
        gate_spec = TrainingSpecification(**{**asdict(spec), "model_id": spec.model_id})
        previous_gate = _candidate_asset(previous, previous.get("inference", {}).get("gate_model")) if previous else None
        previous_damage = _candidate_asset(previous, previous.get("inference", {}).get("damage_model")) if previous else None
        gate_tokenizer, gate_model = _build_hf_model(gate_spec, ["ANY_DAMAGE"], model_source=previous_gate)
        damage_tokenizer, damage_model = _build_hf_model(spec, taxonomy.damage_labels, model_source=previous_damage)
        _fit_hf(gate_model, gate_tokenizer, train, gate_targets, gate_spec, run_dir / "trainer_gate", hardware)
        damage_rows = [row for row, keep in zip(train, harmful, strict=True) if keep]
        damage_targets = train_full[harmful][:, damage_indices]
        _fit_hf(damage_model, damage_tokenizer, damage_rows, damage_targets, spec, run_dir / "trainer_damage", hardware)
        gate_validation = _predict_hf(gate_model, gate_tokenizer, validation, spec.max_length, hardware, 1)
        gate_test = _predict_hf(gate_model, gate_tokenizer, test, spec.max_length, hardware, 1)
        damage_validation = _predict_hf(damage_model, damage_tokenizer, validation, spec.max_length, hardware, 4)
        damage_test = _predict_hf(damage_model, damage_tokenizer, test, spec.max_length, hardware, 4)
        validation_scores = np.concatenate([1 - gate_validation, gate_validation * damage_validation], axis=1)
        test_scores = np.concatenate([1 - gate_test, gate_test * damage_test], axis=1)
        gate_dir, damage_dir = run_dir / "gate_model", run_dir / "damage_model"
        _save_hf(gate_model, gate_tokenizer, gate_dir)
        _save_hf(damage_model, damage_tokenizer, damage_dir)
        inference = {"type": "hf_cascade", "gate_model": gate_dir.name, "damage_model": damage_dir.name}
        checkpoint_paths = [gate_dir, damage_dir]
    else:
        targets, labels = _output_targets(train, experiment)
        previous_source = None
        if previous and experiment != "qwen_lora":
            previous_source = _candidate_asset(previous, previous.get("inference", {}).get("model"))
        previous_adapter = _candidate_asset(previous, previous.get("inference", {}).get("model")) if previous and experiment == "qwen_lora" else None
        tokenizer, model = _build_hf_model(
            spec,
            labels,
            model_source=previous_source,
            lora=experiment == "qwen_lora",
            adapter_source=previous_adapter,
        )
        _fit_hf(
            model,
            tokenizer,
            train,
            targets,
            spec,
            run_dir / "trainer",
            hardware,
            structured_penalty=0.2 if experiment == "qwen_structured" else 0.0,
            output_weights=(
                [1.0] * 5 + [0.3] * len(taxonomy.fine_labels) + [0.2] * len(taxonomy.flags)
                if experiment == "multitask"
                else None
            ),
        )
        validation_all = _predict_hf(model, tokenizer, validation, spec.max_length, hardware, len(labels))
        test_all = _predict_hf(model, tokenizer, test, spec.max_length, hardware, len(labels))
        validation_scores = validation_all[:, :5]
        test_scores = test_all[:, :5]
        model_dir = run_dir / "model"
        _save_hf(model, tokenizer, model_dir)
        inference = {
            "type": "hf_peft_sequence_classifier" if experiment == "qwen_lora" else "hf_sequence_classifier",
            "model": model_dir.name,
            "primary_output_count": 5,
        }
        checkpoint_paths = [model_dir]

    thresholds, validation_metrics, test_metrics = _evaluate(
        run_dir, validation, validation_scores, test, test_scores
    )
    bundle = run_dir / "inference.json"
    write_json_atomic(bundle, {**inference, "target_labels": list(taxonomy.target_labels)})
    manifest = _checkpoint_manifest(run_dir, [*checkpoint_paths, bundle])
    candidate = {
        "schema_version": "2.1.0",
        "candidate_id": f"{experiment}-{signature[:12]}",
        "experiment": experiment,
        "model_family": experiment,
        "run_signature": signature,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "target_labels": list(taxonomy.target_labels),
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "metrics_path": "metrics.json",
        "checkpoint_manifest": manifest.name,
        "inference": {**inference, "bundle": bundle.name},
        "hardware": hardware.model_dump(),
        "warm_start_from": previous.get("candidate_id") if previous else None,
        "status": "complete",
        "completed_at": _utc_iso(),
    }
    write_json_atomic(candidate_path, candidate)
    save_manifest(
        run_dir / "run_manifest.json",
        build_manifest(
            run_id=candidate["candidate_id"],
            stage="03_entrenamiento",
            inputs=[artifact_reference(dataset, "model_ready_snapshot")],
            outputs=[
                artifact_reference(candidate_path, "candidate"),
                artifact_reference(manifest, "checkpoint_manifest"),
                artifact_reference(run_dir / "metrics.json", "metrics"),
            ],
            configuration={"engine": TRAINING_ENGINE_VERSION, **configuration},
            hardware=hardware,
            counters={"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
            warnings=["warm_start_from:" + previous["candidate_id"]] if previous else [],
        ),
    )
    return {"status": "trained", "candidate": candidate, "run_signature": signature}


def train_flat_transformers(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    results = [
        train_neural_experiment(dataset_path, output_root, experiment="flat_minilm", device=device, force=force),
        train_neural_experiment(dataset_path, output_root, experiment="flat_e5", device=device, force=force),
    ]
    return {
        "status": "noop" if all(result["status"] == "noop" for result in results) else "trained",
        "results": results,
    }
