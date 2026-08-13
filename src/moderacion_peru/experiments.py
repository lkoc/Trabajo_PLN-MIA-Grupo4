from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cascade import (
    DEFAULT_GATE_MIN_DAMAGE_RECALL,
    DEFAULT_GATE_MIN_SAFE_NPV,
    calibrate_safety_first_gate,
    combine_safety_first_cascade_scores,
)
from .datasets import deterministic_safe_downsample
from .device import (
    cuda_performance_profile,
    high_memory_bf16_cuda,
    resolve_device,
    torch_device_name,
)
from .io import (
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .manifests import artifact_reference, build_manifest, save_manifest
from .models import TRANSFORMER_SPECS, TrainingSpecification
from .persistent_checkpoints import (
    build_persistent_checkpoint_callback,
    restore_latest_trainer_checkpoint,
)
from .taxonomy import load_taxonomy
from .training import (
    calibrate_thresholds,
    classification_metrics,
    encode_targets,
    masked_multilabel_metrics,
)

TRAINING_ENGINE_VERSION = "4.2.0"
PROJECT_SAFE_TO_DAMAGE_RATIO = 4.0
DEFAULT_LOCAL_PARALLEL_WORKERS = 4
DEFAULT_LINEAR_SVM_MAX_ITER = 20000
DEFAULT_LINEAR_SVM_TOL = 1e-3
ProgressCallback = Callable[[dict[str, Any]], None]


def _notify_progress(callback: ProgressCallback | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def _require_project_safe_ratio(value: float | None) -> float:
    if value != PROJECT_SAFE_TO_DAMAGE_RATIO:
        raise ValueError(
            "La política activa exige safe_to_damage_ratio=4.0. "
            "Entrenar con todos los SEGURO queda fuera del alcance de este proyecto."
        )
    return PROJECT_SAFE_TO_DAMAGE_RATIO


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def _experiment_signature(
    dataset: Path, experiment: str, configuration: dict[str, Any]
) -> str:
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
    if (
        candidate.get("run_signature") != signature
        or candidate.get("status") != "complete"
        or not manifest.is_file()
    ):
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for record in payload.get("files", []):
        target = manifest.parent / record["path"]
        if not target.is_file() or sha256_file(target) != record["sha256"]:
            return None
    candidate["candidate_path"] = str(path)
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
    *,
    output_labels: Sequence[str] | None = None,
) -> None:
    taxonomy = load_taxonomy()
    labels = list(output_labels or taxonomy.target_labels)
    if scores.shape != (len(rows), len(labels)):
        raise ValueError("Predicciones y contrato de salidas no coinciden")
    output = []
    for row, vector in zip(rows, scores, strict=True):
        output.append(
            {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "split": row["split"],
                "scores": {
                    label: float(vector[index]) for index, label in enumerate(labels)
                },
                "true_labels": row["coarse_labels"],
            }
        )
    write_jsonl_atomic(path, output)


def _evaluate_validation(
    run_dir: Path,
    validation_rows: Sequence[dict[str, Any]],
    validation_scores_all: np.ndarray,
    output_labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    """Calibra y evalúa validation sin consultar el split test.

    Los primeros cinco scores son las salidas gruesas. Las métricas auxiliares
    respetan las máscaras de observación del snapshot.
    """

    taxonomy = load_taxonomy()
    if tuple(output_labels[:5]) != taxonomy.target_labels:
        raise ValueError("Las primeras cinco salidas deben seguir el contrato grueso")
    y_validation = encode_targets(validation_rows)
    validation_scores = validation_scores_all[:, :5]
    thresholds = calibrate_thresholds(y_validation, validation_scores)
    validation_metrics = classification_metrics(
        y_validation, validation_scores, thresholds
    )
    auxiliary: dict[str, Any] = {}
    offset = len(taxonomy.target_labels)
    if len(output_labels) >= offset + len(taxonomy.fine_labels):
        fine_truth = np.asarray(
            [
                [
                    int(label in row.get("fine_labels", ()))
                    for label in taxonomy.fine_labels
                ]
                for row in validation_rows
            ],
            dtype=np.int8,
        )
        fine_mask = np.asarray(
            [
                row.get("fine_observed_mask", [0] * len(taxonomy.fine_labels))
                for row in validation_rows
            ],
            dtype=np.int8,
        )
        fine_scores = validation_scores_all[
            :, offset : offset + len(taxonomy.fine_labels)
        ]
        fine_thresholds = calibrate_thresholds(
            fine_truth,
            fine_scores,
            taxonomy.fine_labels,
            fine_mask,
        )
        auxiliary["fine"] = masked_multilabel_metrics(
            fine_truth,
            fine_scores,
            fine_mask,
            taxonomy.fine_labels,
            fine_thresholds,
        )
        auxiliary["fine"]["thresholds"] = fine_thresholds
        offset += len(taxonomy.fine_labels)
    if len(output_labels) >= offset + len(taxonomy.flags):
        flag_truth = np.asarray(
            [
                [
                    int(flag in row.get("flags_reference_only", ()))
                    for flag in taxonomy.flags
                ]
                for row in validation_rows
            ],
            dtype=np.int8,
        )
        flag_mask = np.asarray(
            [
                row.get("flags_observed_mask", [0] * len(taxonomy.flags))
                for row in validation_rows
            ],
            dtype=np.int8,
        )
        flag_scores = validation_scores_all[:, offset : offset + len(taxonomy.flags)]
        flag_thresholds = calibrate_thresholds(
            flag_truth,
            flag_scores,
            taxonomy.flags,
            flag_mask,
        )
        auxiliary["flags"] = masked_multilabel_metrics(
            flag_truth,
            flag_scores,
            flag_mask,
            taxonomy.flags,
            flag_thresholds,
        )
        auxiliary["flags"]["thresholds"] = flag_thresholds
    write_json_atomic(run_dir / "thresholds.json", thresholds)
    write_json_atomic(
        run_dir / "metrics.json",
        {
            "selection_split": "validation",
            "test_status": "sealed_not_evaluated",
            "thresholds": thresholds,
            "validation": validation_metrics,
            "auxiliary_validation_observed_only": auxiliary,
        },
    )
    _write_predictions(
        run_dir / "predictions_validation.jsonl",
        validation_rows,
        validation_scores_all,
        output_labels=output_labels,
    )
    return thresholds, validation_metrics, auxiliary


def _dataset_splits(
    dataset: Path,
    *,
    split_scheme: str = "video",
    safe_to_damage_ratio: float | None = 4.0,
    sampling_seed: int = 20260805,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows = [row for row in read_jsonl(dataset)]
    split_field = {"video": "split", "channel": "channel_split"}.get(split_scheme)
    if split_field is None:
        raise ValueError("split_scheme debe ser 'video' o 'channel'")
    train_full = [row for row in rows if row.get(split_field) == "train"]
    validation_full = [row for row in rows if row.get(split_field) == "validation"]
    test_full = [row for row in rows if row.get(split_field) == "test"]
    train, train_sampling = deterministic_safe_downsample(
        train_full,
        safe_to_damage_ratio=safe_to_damage_ratio,
        seed=sampling_seed,
    )
    validation, validation_sampling = deterministic_safe_downsample(
        validation_full,
        safe_to_damage_ratio=safe_to_damage_ratio,
        seed=sampling_seed,
    )
    if not train or not validation or not test_full:
        raise ValueError(
            "El snapshot necesita filas en train, validation y test; agregue videos, no redistribuya chunks"
        )
    sampling = {
        "policy": "fixed_4_to_1_train_validation_full_sealed_test",
        "split_scheme": split_scheme,
        "split_field": split_field,
        "sampling_seed": sampling_seed,
        "safe_to_damage_ratio": safe_to_damage_ratio,
        "train": train_sampling,
        "validation": validation_sampling,
        "test_sealed": {
            "policy": "full_natural_prevalence",
            "rows_before": len(test_full),
            "rows_after": len(test_full),
            "rows_removed": 0,
        },
        "test_reporting": {
            "primary": "full_natural_prevalence",
            "secondary": "deterministic_4_to_1_slice_from_same_predictions",
            "single_inference_pass": True,
        },
        "prevalence_interpretation": (
            "Validation metrics describe the deterministic 4:1 benchmark. The sealed "
            "test is scored once at natural prevalence and the same predictions also "
            "produce a deterministic 4:1 secondary view."
        ),
    }
    return train, validation, test_full, sampling


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


def _estimator_iteration_diagnostic(estimator: Any) -> dict[str, Any]:
    """Resume si estimadores iterativos alcanzaron su límite configurado."""

    calibrated = getattr(estimator, "calibrated_classifiers_", ())
    fitted_estimators = (
        [item.estimator for item in calibrated] if calibrated else [estimator]
    )
    traces = []
    for fitted in fitted_estimators:
        n_iter = getattr(fitted, "n_iter_", None)
        max_iter = getattr(fitted, "max_iter", None)
        if n_iter is None or max_iter is None:
            continue
        observed = [int(value) for value in np.asarray(n_iter).reshape(-1)]
        limit = int(max_iter)
        traces.append(
            {
                "iterations": observed,
                "max_iter": limit,
                "hit_iteration_limit": any(value >= limit for value in observed),
            }
        )
    return {
        "iteration_tracking_available": bool(traces),
        "iterative_fits": len(traces),
        "fits_at_iteration_limit": sum(
            bool(trace["hit_iteration_limit"]) for trace in traces
        ),
        "converged": (
            all(not trace["hit_iteration_limit"] for trace in traces)
            if traces
            else None
        ),
        "traces": traces,
    }


class MaskedOneVsRestClassifier:
    """Un estimador binario por salida, omitiendo targets con máscara cero.

    El target usa ``-1`` para posiciones no observadas. La clase se mantiene en
    este módulo para que los artefactos joblib sean portables en inferencia.
    """

    def __init__(self, estimator: Any, *, n_jobs: int = 1) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        parameters = {"estimator": self.estimator, "n_jobs": self.n_jobs}
        if deep and hasattr(self.estimator, "get_params"):
            parameters.update(
                {
                    f"estimator__{name}": value
                    for name, value in self.estimator.get_params(deep=True).items()
                }
            )
        return parameters

    def set_params(self, **parameters: Any) -> MaskedOneVsRestClassifier:
        nested = {}
        for name, value in parameters.items():
            if name.startswith("estimator__"):
                nested[name.removeprefix("estimator__")] = value
            else:
                setattr(self, name, value)
        if nested:
            self.estimator.set_params(**nested)
        return self

    def __sklearn_tags__(self) -> Any:
        """Expone tags modernos sin hacer scikit-learn una dependencia base."""

        from sklearn.base import BaseEstimator

        return BaseEstimator.__sklearn_tags__(self)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimators_") and hasattr(self, "n_outputs_")

    def fit(self, features: Any, targets: np.ndarray) -> MaskedOneVsRestClassifier:
        from joblib import Parallel, delayed
        from sklearn.base import clone
        from sklearn.dummy import DummyClassifier

        matrix = np.asarray(targets)
        if matrix.ndim != 2:
            raise ValueError("MaskedOneVsRestClassifier requiere una matriz 2D")
        if self.n_jobs == 0:
            raise ValueError("n_jobs no puede ser cero")

        def fit_output(index: int) -> Any:
            observed = matrix[:, index] >= 0
            truth = matrix[observed, index].astype(np.int8)
            if not observed.any():
                estimator = DummyClassifier(strategy="constant", constant=0)
                estimator.fit(features[:1], np.asarray([0], dtype=np.int8))
            elif np.unique(truth).size < 2:
                estimator = DummyClassifier(strategy="constant", constant=int(truth[0]))
                estimator.fit(features[observed], truth)
            else:
                estimator = clone(self.estimator)
                estimator.fit(features[observed], truth)
            return estimator

        self.estimators_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(fit_output)(index) for index in range(matrix.shape[1])
        )
        self.n_outputs_ = matrix.shape[1]
        output_diagnostics = [
            _estimator_iteration_diagnostic(estimator) for estimator in self.estimators_
        ]
        tracked = [
            diagnostic
            for diagnostic in output_diagnostics
            if diagnostic["iteration_tracking_available"]
        ]
        self.fit_diagnostics_ = {
            "outputs": self.n_outputs_,
            "outputs_with_iteration_tracking": len(tracked),
            "outputs_at_iteration_limit": sum(
                diagnostic["fits_at_iteration_limit"] > 0 for diagnostic in tracked
            ),
            "iterative_fits": sum(
                diagnostic["iterative_fits"] for diagnostic in tracked
            ),
            "fits_at_iteration_limit": sum(
                diagnostic["fits_at_iteration_limit"] for diagnostic in tracked
            ),
            "converged": (
                all(diagnostic["converged"] is True for diagnostic in tracked)
                if tracked
                else None
            ),
            "per_output": output_diagnostics,
        }
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        from joblib import Parallel, delayed

        def score_output(estimator: Any) -> np.ndarray:
            if hasattr(estimator, "predict_proba"):
                probabilities = np.asarray(estimator.predict_proba(features))
                classes = list(getattr(estimator, "classes_", (0, 1)))
                if 1 in classes:
                    values = probabilities[:, classes.index(1)]
                else:
                    values = np.zeros(features.shape[0], dtype=float)
            elif hasattr(estimator, "decision_function"):
                values = _sigmoid(np.asarray(estimator.decision_function(features)))
            else:
                values = np.asarray(estimator.predict(features), dtype=float)
            return values

        scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(score_output)(estimator) for estimator in self.estimators_
        )
        return np.column_stack(scores)


def train_classical_experiments(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    model_names: Iterable[str] | None = None,
    max_features: int = 150000,
    variants: Iterable[str] = ("base",),
    safe_to_damage_ratio: float | None = 4.0,
    split_scheme: str = "video",
    sampling_seed: int = 20260805,
    parallel_workers: int = DEFAULT_LOCAL_PARALLEL_WORKERS,
    linear_svm_max_iter: int = DEFAULT_LINEAR_SVM_MAX_ITER,
    linear_svm_tol: float = DEFAULT_LINEAR_SVM_TOL,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Entrena baselines enmascarados y produce evidencia solo de validation.

    ``base`` usa TF-IDF de palabras y caracteres; ``policy_informed`` añade
    disparadores auditables del prompt v3.2, pero sigue siendo un clasificador
    supervisado y no se presenta como modelo condicionado por prompt.
    """

    safe_to_damage_ratio = _require_project_safe_ratio(safe_to_damage_ratio)
    available_workers = os.cpu_count() or 1
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    if linear_svm_max_iter < 1000:
        raise ValueError("linear_svm_max_iter debe ser al menos 1000")
    if not 0 < linear_svm_tol <= 0.1:
        raise ValueError("linear_svm_tol debe estar en (0, 0.1]")
    parallel_workers = min(parallel_workers, available_workers)
    try:
        import joblib
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc

    dataset = Path(dataset_path).resolve()
    output = Path(output_root)
    candidates_spec = {
        "dummy": DummyClassifier(strategy="prior"),
        "complement_nb": ComplementNB(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=linear_svm_max_iter,
                tol=linear_svm_tol,
                random_state=sampling_seed,
            ),
            method="sigmoid",
            cv=3,
        ),
        "sgd_incremental": SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=20260805,
        ),
    }
    selected_names = (
        list(candidates_spec)
        if model_names is None
        else list(dict.fromkeys(model_names))
    )
    unknown = sorted(set(selected_names) - set(candidates_spec))
    if unknown:
        raise ValueError(f"Modelos clasicos desconocidos: {unknown}")
    if not selected_names:
        raise ValueError("Debe seleccionar al menos un modelo clasico")
    if max_features < 100:
        raise ValueError("max_features debe ser al menos 100")
    candidates_spec = {name: candidates_spec[name] for name in selected_names}
    selected_variants = list(dict.fromkeys(variants))
    unknown_variants = sorted(set(selected_variants) - {"base", "policy_informed"})
    if unknown_variants or not selected_variants:
        raise ValueError(f"Variantes clásicas inválidas: {unknown_variants}")
    progress_total = 1 + len(selected_variants) * (1 + len(candidates_spec))
    _notify_progress(
        progress_callback,
        status="started",
        phase="preparando datos",
        total=progress_total,
        advance=0,
    )

    configuration: dict[str, Any] = {
        "suite": "classical_v4_masked",
        "seed": sampling_seed,
        "models": selected_names,
        "variants": selected_variants,
        "max_features": max_features,
        "safe_to_damage_ratio_train_validation": safe_to_damage_ratio,
        "test_policy": "full_natural_plus_4_to_1_secondary_same_predictions",
        "split_scheme": split_scheme,
        "parallel_workers": parallel_workers,
        "linear_svm": {
            "solver": "LinearSVC_dual_auto",
            "max_iter": linear_svm_max_iter,
            "tol": linear_svm_tol,
            "calibration": "sigmoid_cv3",
        },
        "feature_extraction": "once_per_variant_reused_by_all_models",
        "test_status": "sealed_not_evaluated",
    }
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
        candidates = [
            _complete_candidate(Path(path), signature)
            for path in state.get("candidate_paths", [])
        ]
        if candidates and all(candidate is not None for candidate in candidates):
            _notify_progress(
                progress_callback,
                status="finished",
                phase="artefactos ya existentes",
                total=progress_total,
                completed=progress_total,
            )
            return {
                "status": "noop",
                "run_signature": signature,
                "candidates": candidates,
            }

    train, validation, test, sampling = _dataset_splits(
        dataset,
        split_scheme=split_scheme,
        safe_to_damage_ratio=safe_to_damage_ratio,
        sampling_seed=sampling_seed,
    )
    train_texts = [str(row["text"]) for row in train]
    targets, masks, output_labels = _output_targets_and_masks(train)
    masked_targets = targets.astype(np.int8)
    masked_targets[masks == 0] = -1
    validation_texts = [str(row["text"]) for row in validation]
    _notify_progress(
        progress_callback,
        status="progress",
        phase="datos preparados",
        advance=1,
        details={"train": len(train), "validation": len(validation)},
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "training_sampling.json", sampling)
    candidates: list[dict[str, Any]] = []
    from .policy_features import POLICY_FEATURE_VERSION, PolicyCueTransformer

    for variant in selected_variants:
        features: list[tuple[str, Any]] = [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1 if len(train) < 100 else 2,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1 if len(train) < 100 else 2,
                    max_features=max(100, max_features // 2),
                    sublinear_tf=True,
                ),
            ),
        ]
        if variant == "policy_informed":
            features.append(("policy_v3_2", PolicyCueTransformer()))
        feature_transformer = FeatureUnion(features)
        feature_started = time.perf_counter()
        train_features = feature_transformer.fit_transform(train_texts, masked_targets)
        validation_features = feature_transformer.transform(validation_texts)
        feature_seconds = time.perf_counter() - feature_started
        feature_summary = {
            "policy": "fit_once_per_variant_reuse_across_models",
            "variant": variant,
            "train_shape": list(train_features.shape),
            "validation_shape": list(validation_features.shape),
            "elapsed_seconds": feature_seconds,
            "parallel_workers_per_multilabel_model": parallel_workers,
        }
        write_json_atomic(
            run_dir / f"feature_extraction_{variant}.json", feature_summary
        )
        _notify_progress(
            progress_callback,
            status="progress",
            phase=f"TF-IDF {variant}",
            advance=1,
            details={"columnas": int(train_features.shape[1])},
        )
        for name, estimator in candidates_spec.items():
            model_started = time.perf_counter()
            experiment_name = (
                name if selected_variants == ["base"] else f"{variant}_{name}"
            )
            model_dir = run_dir / experiment_name
            model_dir.mkdir(parents=True, exist_ok=True)
            fit_started = time.perf_counter()
            classifier = MaskedOneVsRestClassifier(
                estimator, n_jobs=parallel_workers
            ).fit(train_features, masked_targets)
            fit_seconds = time.perf_counter() - fit_started
            pipeline = Pipeline(
                [
                    ("features", feature_transformer),
                    ("classifier", classifier),
                ]
            )
            validation_started = time.perf_counter()
            validation_scores = classifier.predict_proba(validation_features)
            thresholds, validation_metrics, auxiliary_metrics = _evaluate_validation(
                model_dir, validation, validation_scores, output_labels
            )
            validation_seconds = time.perf_counter() - validation_started
            checkpoint = model_dir / "model.joblib"
            joblib.dump(pipeline, checkpoint)
            bundle = model_dir / "inference.json"
            write_json_atomic(
                bundle,
                {
                    "type": "sklearn_joblib",
                    "model": checkpoint.name,
                    "target_labels": list(load_taxonomy().target_labels),
                    "output_labels": output_labels,
                    "primary_output_count": 5,
                    "variant": variant,
                    "policy_feature_version": (
                        POLICY_FEATURE_VERSION if variant == "policy_informed" else None
                    ),
                    "parallel_workers": parallel_workers,
                    "feature_extraction": "shared_fitted_transformer_per_variant",
                },
            )
            manifest = _checkpoint_manifest(model_dir, [checkpoint, bundle])
            candidate = {
                "schema_version": "2.1.0",
                "candidate_id": f"classical-{experiment_name}-{signature[:12]}",
                "experiment": experiment_name,
                "model_family": f"classical:{variant}:{name}",
                "conditioning": (
                    "supervised_policy_features"
                    if variant == "policy_informed"
                    else "supervised_labels_only"
                ),
                "run_signature": signature,
                "dataset": str(dataset),
                "dataset_sha256": sha256_file(dataset),
                "target_labels": list(load_taxonomy().target_labels),
                "thresholds": thresholds,
                "validation_metrics": validation_metrics,
                "auxiliary_validation_metrics": auxiliary_metrics,
                "test_metrics": None,
                "test_status": "sealed_not_evaluated",
                "training_sampling": sampling,
                "metrics_path": "metrics.json",
                "checkpoint_manifest": manifest.name,
                "inference": {"type": "sklearn_joblib", "bundle": bundle.name},
                "hardware": {
                    "backend": "cpu",
                    "requested": "cpu",
                    "device_name": os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
                    "dtype": "float64",
                    "logical_processors": available_workers,
                    "parallel_workers": parallel_workers,
                },
                "runtime_optimization": {
                    "feature_extraction": feature_summary,
                    "multilabel_fit": "joblib_threads_shared_sparse_matrix",
                    "parallel_workers": parallel_workers,
                    "fit_elapsed_seconds": fit_seconds,
                },
                "fit_quality": classifier.fit_diagnostics_,
                "stage_timings_seconds": {
                    "shared_feature_extraction_variant": feature_seconds,
                    "model_fit": fit_seconds,
                    "validation_inference_and_metrics": validation_seconds,
                    "model_total_before_candidate_write": time.perf_counter()
                    - model_started,
                },
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
                    configuration={
                        "engine": TRAINING_ENGINE_VERSION,
                        **configuration,
                        "model": name,
                        "variant": variant,
                    },
                    hardware=candidate["hardware"],
                    counters={
                        "train_rows_after_safe_sampling": len(train),
                        "validation_rows_4_to_1": len(validation),
                        "test_rows_sealed": len(test),
                        "feature_columns": int(train_features.shape[1]),
                        "parallel_workers": parallel_workers,
                    },
                ),
            )
            candidate["candidate_path"] = str(candidate_path)
            candidates.append(candidate)
            _notify_progress(
                progress_callback,
                status="progress",
                phase=f"{variant} · {name}",
                advance=1,
                details={"candidatos": len(candidates)},
            )
    write_json_atomic(
        complete,
        {
            "run_signature": signature,
            "candidate_paths": [
                candidate["candidate_path"] for candidate in candidates
            ],
            "completed_at": _utc_iso(),
        },
    )
    _notify_progress(
        progress_callback,
        status="finished",
        phase="suite clásica completa",
        total=progress_total,
        completed=progress_total,
    )
    return {"status": "trained", "run_signature": signature, "candidates": candidates}


class _TokenizedRows:
    def __init__(
        self,
        tokenizer: Any,
        rows: Sequence[dict[str, Any]],
        targets: np.ndarray,
        max_length: int,
        observed_masks: np.ndarray | None = None,
    ) -> None:
        encoded = tokenizer(
            [str(row["text"]) for row in rows],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.encodings = {
            key: np.asarray(values, dtype=np.int64) for key, values in encoded.items()
        }
        self.targets = targets.astype(np.float32)
        self.observed_masks = (
            np.ones_like(self.targets, dtype=np.float32)
            if observed_masks is None
            else observed_masks.astype(np.float32)
        )
        if self.observed_masks.shape != self.targets.shape:
            raise ValueError("La máscara de entrenamiento no coincide con los targets")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = {
            key: torch.from_numpy(value[index]) for key, value in self.encodings.items()
        }
        item["labels"] = torch.from_numpy(self.targets[index])
        item["label_mask"] = torch.from_numpy(self.observed_masks[index])
        return item


def _output_targets_and_masks(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construye siempre 5+14+3 salidas y su supervisión explícita."""

    taxonomy = load_taxonomy()
    coarse = encode_targets(rows).astype(np.float32)
    fine = np.asarray(
        [
            [int(label in row.get("fine_labels", [])) for label in taxonomy.fine_labels]
            for row in rows
        ],
        dtype=np.float32,
    )
    flags = np.asarray(
        [
            [
                int(flag in row.get("flags_reference_only", []))
                for flag in taxonomy.flags
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    coarse_mask = np.asarray(
        [
            row.get("coarse_observed_mask", [1] * len(taxonomy.target_labels))
            for row in rows
        ],
        dtype=np.float32,
    )
    fine_mask = np.asarray(
        [
            row.get("fine_observed_mask", [0] * len(taxonomy.fine_labels))
            for row in rows
        ],
        dtype=np.float32,
    )
    flags_mask = np.asarray(
        [row.get("flags_observed_mask", [0] * len(taxonomy.flags)) for row in rows],
        dtype=np.float32,
    )
    labels = [
        *taxonomy.target_labels,
        *(f"fine:{label}" for label in taxonomy.fine_labels),
        *(f"flag:{flag}" for flag in taxonomy.flags),
    ]
    return (
        np.concatenate([coarse, fine, flags], axis=1),
        np.concatenate([coarse_mask, fine_mask, flags_mask], axis=1),
        labels,
    )


def _output_targets(
    rows: Sequence[dict[str, Any]], experiment: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """Compatibilidad: todos los clasificadores compatibles exponen 22 salidas."""

    targets, _masks, labels = _output_targets_and_masks(rows)
    return targets, labels


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
            raise RuntimeError(
                "Instale PEFT mediante moderacion-peru[entrenamiento]"
            ) from exc
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
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        # All remote model revisions declared by this project are public.  Being
        # explicit also prevents huggingface_hub from probing Colab's secret
        # vault when the notebook is executed through the VS Code extension.
        token=False,
    )
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
        ignore_mismatched_sizes=True,
        token=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "Instale PEFT mediante moderacion-peru[entrenamiento]"
            ) from exc
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


def _last_candidate(
    output_root: Path, experiment: str, dataset_sha: str
) -> dict[str, Any] | None:
    candidates = []
    for path in output_root.rglob("candidate.json") if output_root.exists() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            row.get("experiment") == experiment
            and row.get("status") == "complete"
            and row.get("dataset_sha256") != dataset_sha
            and row.get("output_count", 5) == 22
        ):
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


def select_qwen_lora_warm_start_candidate(
    output_root: str | Path,
    dataset_path: str | Path,
    *,
    max_length: int = 128,
) -> dict[str, Any]:
    """Selecciona un Qwen-LoRA completo y verificable como punto de partida.

    La selección exige el mismo snapshot del dataset y la longitud solicitada. De
    este modo una variante de 256 tokens no puede inicializar accidentalmente otra
    variante de 256 ni un candidato entrenado sobre datos distintos.
    """

    output = Path(output_root)
    dataset = Path(dataset_path).resolve()
    dataset_sha = sha256_file(dataset)
    candidates: list[dict[str, Any]] = []
    for candidate_path in output.rglob("candidate.json") if output.exists() else []:
        try:
            raw = json.loads(candidate_path.read_text(encoding="utf-8"))
            signature = str(raw["run_signature"])
            candidate = _complete_candidate(candidate_path, signature)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if candidate is None:
            continue
        diagnostic = candidate.get("truncation_diagnostic") or {}
        if (
            candidate.get("experiment") == "qwen_lora"
            and candidate.get("model_family") == "qwen_lora"
            and candidate.get("dataset_sha256") == dataset_sha
            and candidate.get("output_count") == 22
            and int(diagnostic.get("max_length", -1)) == int(max_length)
        ):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "No existe un candidato Qwen-LoRA completo y verificable de "
            f"{max_length} tokens para el snapshot {dataset_sha[:12]} bajo {output}. "
            "Restaure la publicación de 03_05 o ejecute primero el bloque base."
        )
    return max(
        candidates,
        key=lambda row: (str(row.get("completed_at", "")), str(row["candidate_path"])),
    )


def _load_explicit_qwen_warm_start_candidate(
    candidate_path: str | Path,
    *,
    dataset_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(candidate_path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        candidate = _complete_candidate(path, str(raw["run_signature"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Candidato warm-start inválido: {path}") from exc
    if candidate is None:
        raise ValueError(f"Candidato warm-start incompleto o con hashes inválidos: {path}")
    if (
        candidate.get("experiment") != "qwen_lora"
        or candidate.get("model_family") != "qwen_lora"
        or candidate.get("output_count") != 22
    ):
        raise ValueError("El warm-start explícito debe ser un Qwen-LoRA de 22 salidas")
    if candidate.get("dataset_sha256") != dataset_sha:
        raise ValueError("El warm-start y la continuación deben usar exactamente el mismo dataset")
    model_path = _candidate_asset(candidate, candidate.get("inference", {}).get("model"))
    if model_path is None or not Path(model_path).is_dir():
        raise ValueError("El candidato warm-start no contiene el adaptador PEFT declarado")
    manifest_path = _candidate_asset(candidate, candidate.get("checkpoint_manifest"))
    if manifest_path is None:
        raise ValueError("El candidato warm-start no declara checkpoint_manifest")
    identity = {
        "candidate_id": str(candidate["candidate_id"]),
        "run_signature": str(candidate["run_signature"]),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "max_length": int(
            (candidate.get("truncation_diagnostic") or {}).get("max_length", -1)
        ),
        "optimizer_state_reused": False,
        "policy": "trainable_peft_adapter_with_fresh_optimizer_and_scheduler",
    }
    return candidate, identity


def _hf_validation_metrics(
    logits: Any,
    label_ids: Any,
    *,
    primary_output_count: int,
) -> dict[str, float]:
    """Métrica de selección para cabezas completas o ramas de una cascada."""

    scores_array = np.asarray(logits, dtype=float)
    truth_array = np.asarray(label_ids, dtype=np.int8)
    if scores_array.ndim == 1:
        scores_array = scores_array[:, None]
    if truth_array.ndim == 1:
        truth_array = truth_array[:, None]
    if scores_array.shape != truth_array.shape:
        raise ValueError("Logits y etiquetas de validation no tienen la misma forma")
    if not 1 <= primary_output_count <= scores_array.shape[1]:
        raise ValueError("primary_output_count no coincide con la cabeza del modelo")

    primary_scores = _sigmoid(scores_array[:, :primary_output_count])
    primary_truth = truth_array[:, :primary_output_count]
    if primary_output_count == 5:
        metrics = classification_metrics(primary_truth, primary_scores)
        return {
            "macro_auprc_damage": float(metrics["average_precision_macro_damage"]),
            "macro_auprc_five": float(metrics["average_precision_macro_five"]),
        }

    from sklearn.metrics import average_precision_score

    per_output = [
        float(
            average_precision_score(primary_truth[:, index], primary_scores[:, index])
        )
        for index in range(primary_output_count)
    ]
    # Conserva el nombre histórico usado por Trainer y por los checkpoints. En la
    # compuerta representa AP(ANY_DAMAGE); en la rama de daño, la macro-AP de sus
    # cuatro categorías.
    return {"macro_auprc_damage": float(np.mean(per_output))}


def _fit_hf(
    model: Any,
    tokenizer: Any,
    train_rows: Sequence[dict[str, Any]],
    targets: np.ndarray,
    observed_masks: np.ndarray,
    validation_rows: Sequence[dict[str, Any]],
    validation_targets: np.ndarray,
    validation_masks: np.ndarray,
    spec: TrainingSpecification,
    training_dir: Path,
    hardware: Any,
    *,
    structured_penalty: float = 0.0,
    output_weights: Sequence[float] | None = None,
    primary_output_count: int = 5,
    eval_batch_size: int = 8,
    dataloader_num_workers: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_label: str = "modelo",
    persistent_checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Instale PyTorch, accelerate y Transformers") from exc

    _notify_progress(
        progress_callback,
        status="progress",
        phase=f"tokenizando {progress_label}",
        advance=0,
        details={"train": len(train_rows), "validation": len(validation_rows)},
    )
    dataset = _TokenizedRows(
        tokenizer, train_rows, targets, spec.max_length, observed_masks
    )
    validation_dataset = _TokenizedRows(
        tokenizer,
        validation_rows,
        validation_targets,
        spec.max_length,
        validation_masks,
    )
    observed_positive = (targets * observed_masks).sum(axis=0)
    observed_negative = ((1 - targets) * observed_masks).sum(axis=0)
    positive_weights = np.clip(
        observed_negative / np.maximum(1.0, observed_positive), 1.0, 20.0
    ).astype(np.float32)

    class StructuredTrainer(Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            labels = inputs.pop("labels")
            label_mask = inputs.pop("label_mask")
            outputs = model(**inputs)
            logits = outputs.logits
            positive = torch.as_tensor(
                positive_weights, device=logits.device, dtype=logits.dtype
            )
            elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
                pos_weight=positive,
            )
            if output_weights is not None:
                weights = torch.as_tensor(
                    output_weights, device=logits.device, dtype=logits.dtype
                )
                elementwise = elementwise * weights.unsqueeze(0)
            elementwise = elementwise * label_mask
            loss = elementwise.sum() / label_mask.sum().clamp_min(1.0)
            if structured_penalty and logits.shape[1] >= 5:
                probabilities = torch.sigmoid(logits)
                conflict = probabilities[:, 0] * probabilities[:, 1:5].amax(dim=1)
                loss = loss + structured_penalty * conflict.mean()
            return (loss, outputs) if return_outputs else loss

    arguments = TrainingArguments(
        output_dir=str(training_dir),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=spec.gradient_accumulation,
        learning_rate=spec.learning_rate,
        seed=spec.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_auprc_damage",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=max(1, math.ceil(len(dataset) / max(1, spec.batch_size * 5))),
        report_to=[],
        # ``label_mask`` is consumed by StructuredTrainer.compute_loss rather than
        # by model.forward.  Trainer would otherwise classify it as unused and
        # remove it from every batch before the custom loss can read it.
        remove_unused_columns=False,
        use_cpu=hardware.backend == "cpu",
        fp16=hardware.backend in {"cuda", "rocm"} and hardware.dtype == "float16",
        bf16=hardware.backend in {"cuda", "rocm", "xpu"}
        and hardware.dtype == "bfloat16",
        dataloader_pin_memory=hardware.backend != "cpu",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=dataloader_num_workers > 0,
        dataloader_prefetch_factor=2 if dataloader_num_workers > 0 else None,
        optim=("adamw_torch_fused" if hardware.backend == "cuda" else "adamw_torch"),
        tf32=high_memory_bf16_cuda(hardware),
    )

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        logits = (
            evaluation.predictions[0]
            if isinstance(evaluation.predictions, tuple)
            else evaluation.predictions
        )
        return _hf_validation_metrics(
            logits,
            evaluation.label_ids,
            primary_output_count=primary_output_count,
        )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)]
    persistent_callback = build_persistent_checkpoint_callback(
        persistent_checkpoint_dir
    )
    if persistent_callback is not None:
        callbacks.append(persistent_callback)
    trainer = StructuredTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    _notify_progress(
        progress_callback,
        status="progress",
        phase=f"entrenando {progress_label}",
        advance=0,
        details={
            "lote_GPU": spec.batch_size,
            "acumulación": spec.gradient_accumulation,
            "lote_validation": eval_batch_size,
        },
    )
    restored_checkpoint = restore_latest_trainer_checkpoint(
        persistent_checkpoint_dir, training_dir
    )
    checkpoint = str(restored_checkpoint) if restored_checkpoint is not None else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    return {
        "training_metrics": dict(result.metrics),
        "positive_weights": positive_weights.tolist(),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "resumed_from_checkpoint": checkpoint,
        "persistent_checkpoint_dir": (
            str(persistent_checkpoint_dir)
            if persistent_checkpoint_dir is not None
            else None
        ),
    }


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
    high_memory_profile = high_memory_bf16_cuda(hardware)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64 if high_memory_profile else 16,
        shuffle=False,
        num_workers=2 if high_memory_profile else 0,
        pin_memory=hardware.backend != "cpu",
        persistent_workers=high_memory_profile,
    )
    device = torch_device_name(hardware)
    model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch.pop("label_mask", None)
            batch = {
                key: value.to(device, non_blocking=hardware.backend != "cpu")
                for key, value in batch.items()
            }
            logits = model(**batch).logits
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


def _save_hf(model: Any, tokenizer: Any, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)


def _token_truncation_diagnostic(
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    max_length: int,
    *,
    batch_size: int = 512,
) -> dict[str, Any]:
    if not callable(tokenizer):
        return {
            "rows": len(rows),
            "max_length": max_length,
            "status": "unavailable_non_callable_tokenizer",
        }
    lengths: list[int] = []
    for start in range(0, len(rows), batch_size):
        encoded = tokenizer(
            [str(row["text"]) for row in rows[start : start + batch_size]],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths.extend(len(values) for values in encoded["input_ids"])
    array = np.asarray(lengths, dtype=int)
    return {
        "rows": len(rows),
        "max_length": max_length,
        "truncated_rows": int((array > max_length).sum()),
        "truncated_fraction": float((array > max_length).mean()) if len(array) else 0.0,
        "token_length_p50": float(np.quantile(array, 0.50)) if len(array) else 0.0,
        "token_length_p95": float(np.quantile(array, 0.95)) if len(array) else 0.0,
        "token_length_max": int(array.max()) if len(array) else 0,
    }


def train_neural_experiment(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    experiment: str,
    device: str = "auto",
    force: bool = False,
    spec_key: str | None = None,
    max_length: int | None = None,
    epochs: int | None = None,
    variant_id: str | None = None,
    warm_start_candidate_path: str | Path | None = None,
    safe_to_damage_ratio: float | None = 4.0,
    split_scheme: str = "video",
    sampling_seed: int = 20260805,
    cascade_gate_min_damage_recall: float = DEFAULT_GATE_MIN_DAMAGE_RECALL,
    cascade_gate_min_safe_npv: float = DEFAULT_GATE_MIN_SAFE_NPV,
    persistent_checkpoint_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Cierra fit→calibración validation→candidato sin abrir test.

    Experimentos admitidos: ``flat_minilm``, ``flat_e5``, ``cascade``,
    ``cascade_v2``, ``multitask``, ``qwen_lora`` y ``qwen_structured``.
    """

    safe_to_damage_ratio = _require_project_safe_ratio(safe_to_damage_ratio)
    allowed = {
        "flat_minilm",
        "flat_e5",
        "cascade",
        "cascade_v2",
        "multitask",
        "qwen_lora",
        "qwen_structured",
    }
    if experiment not in allowed:
        raise ValueError(f"Experimento desconocido: {experiment}")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 8
    ):
        raise ValueError("max_length debe ser un entero de al menos 8 tokens")
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
    ):
        raise ValueError("epochs debe ser un entero positivo")
    if variant_id is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", variant_id):
        raise ValueError("variant_id solo admite minúsculas, números, guion y subrayado")
    if warm_start_candidate_path is not None and experiment != "qwen_lora":
        raise ValueError("warm_start_candidate_path solo está habilitado para qwen_lora")
    if experiment == "cascade_v2":
        if not 0 < cascade_gate_min_damage_recall <= 1:
            raise ValueError("cascade_gate_min_damage_recall debe pertenecer a (0, 1]")
        if not 0 < cascade_gate_min_safe_npv <= 1:
            raise ValueError("cascade_gate_min_safe_npv debe pertenecer a (0, 1]")
    progress_total = 5 if experiment in {"cascade", "cascade_v2"} else 4
    _notify_progress(
        progress_callback,
        status="started",
        phase=f"preparando {experiment}",
        total=progress_total,
        advance=0,
    )
    run_started = time.perf_counter()
    key = spec_key or (
        "qwen_lora" if experiment in {"qwen_lora", "qwen_structured"} else "e5"
    )
    if experiment == "flat_minilm":
        key = "minilm"
    if experiment == "flat_e5":
        key = "e5"
    spec_values = {**asdict(TRANSFORMER_SPECS[key]), "seed": sampling_seed}
    if max_length is not None:
        spec_values["max_length"] = max_length
    if epochs is not None:
        spec_values["epochs"] = epochs
    spec = TrainingSpecification(**spec_values)
    dataset = Path(dataset_path).resolve()
    dataset_sha = sha256_file(dataset)
    explicit_warm_start = None
    warm_start_identity = None
    if warm_start_candidate_path is not None:
        explicit_warm_start, warm_start_identity = _load_explicit_qwen_warm_start_candidate(
            warm_start_candidate_path,
            dataset_sha=dataset_sha,
        )
        parent_max_length = int(warm_start_identity["max_length"])
        if parent_max_length < 0 or spec.max_length <= parent_max_length:
            raise ValueError(
                "La continuación debe ampliar la longitud del candidato padre: "
                f"padre={parent_max_length}, solicitada={spec.max_length}"
            )
    output = Path(output_root)
    hardware = resolve_device(device)
    performance_profile = cuda_performance_profile(hardware)
    qwen_high_memory_profile = key == "qwen_lora" and high_memory_bf16_cuda(hardware)
    if qwen_high_memory_profile:
        # Conserva el lote efectivo histórico (2 × 4 = 8), pero evita cuatro
        # micropasadas seriales cuando los 40 GB permiten un lote real de ocho.
        spec = TrainingSpecification(
            **{
                **asdict(spec),
                "batch_size": 8,
                "gradient_accumulation": 1,
            }
        )
    eval_batch_size = 32 if qwen_high_memory_profile else 8
    dataloader_num_workers = 2 if qwen_high_memory_profile else 0
    configuration = {
        "experiment": experiment,
        "specification": asdict(spec),
        "structured_penalty": (
            0.2 if experiment in {"cascade_v2", "qwen_structured"} else 0.0
        ),
        "hardware_backend": hardware.backend,
        "dtype": hardware.dtype,
        "performance_profile": performance_profile,
        "per_device_eval_batch_size": eval_batch_size,
        "dataloader_num_workers": dataloader_num_workers,
        "multitask_weights": (
            {"coarse": 1.0, "fine": 0.3, "flags": 0.2}
            if experiment == "multitask"
            else None
        ),
        "all_compatible_outputs": "5+14+3_masked",
        "safe_to_damage_ratio_train_validation": safe_to_damage_ratio,
        "test_policy": "full_natural_plus_4_to_1_secondary_same_predictions",
        "split_scheme": split_scheme,
        "sampling_seed": sampling_seed,
        "early_stopping_selection": "best_validation_macro_auprc_damage",
        "cascade_v2_gate_constraints": (
            {
                "minimum_damage_recall": cascade_gate_min_damage_recall,
                "minimum_safe_npv": cascade_gate_min_safe_npv,
                "selection_partition": "validation",
                "fallback": "route_all_to_five_output_branch",
            }
            if experiment == "cascade_v2"
            else None
        ),
        "test_status": "sealed_not_evaluated",
    }
    if variant_id is not None:
        configuration["variant_id"] = variant_id
    if warm_start_identity is not None:
        configuration["explicit_warm_start"] = warm_start_identity
    signature = _experiment_signature(dataset, experiment, configuration)
    run_prefix = f"{experiment}-{variant_id}" if variant_id else experiment
    run_dir = output / "runs" / f"{run_prefix}-{signature[:16]}"

    def persistent_trainer_dir(name: str) -> Path | None:
        if persistent_checkpoint_root is None:
            return None
        return Path(persistent_checkpoint_root) / run_dir.name / name

    candidate_path = run_dir / "candidate.json"
    if not force:
        complete = _complete_candidate(candidate_path, signature)
        if complete is not None:
            _notify_progress(
                progress_callback,
                status="finished",
                phase="candidato ya existente",
                total=progress_total,
                completed=progress_total,
            )
            return {"status": "noop", "candidate": complete, "run_signature": signature}
    train, validation, test, sampling = _dataset_splits(
        dataset,
        split_scheme=split_scheme,
        safe_to_damage_ratio=safe_to_damage_ratio,
        sampling_seed=sampling_seed,
    )
    _notify_progress(
        progress_callback,
        status="progress",
        phase="datos preparados",
        advance=1,
        details={"train": len(train), "validation": len(validation)},
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "training_sampling.json", sampling)
    previous = explicit_warm_start or _last_candidate(output, experiment, dataset_sha)
    taxonomy = load_taxonomy()
    cascade_diagnostics: dict[str, Any] | None = None
    training_elapsed = 0.0
    validation_inference_elapsed = 0.0

    if experiment in {"cascade", "cascade_v2"}:
        safety_first = experiment == "cascade_v2"
        damage_indices = [
            taxonomy.target_labels.index(label) for label in taxonomy.damage_labels
        ]
        train_full = encode_targets(train).astype(np.float32)
        gate_targets = train_full[:, damage_indices].max(axis=1, keepdims=True)
        train_all, train_masks, all_labels = _output_targets_and_masks(train)
        validation_all_targets, validation_all_masks, _ = _output_targets_and_masks(
            validation
        )
        auxiliary_count = len(taxonomy.fine_labels) + len(taxonomy.flags)
        gate_targets = np.concatenate([gate_targets, train_all[:, 5:]], axis=1)
        gate_masks = np.concatenate(
            [np.ones((len(train), 1), dtype=np.float32), train_masks[:, 5:]],
            axis=1,
        )
        validation_gate_targets = np.concatenate(
            [
                validation_all_targets[:, damage_indices].max(axis=1, keepdims=True),
                validation_all_targets[:, 5:],
            ],
            axis=1,
        )
        validation_gate_masks = np.concatenate(
            [
                np.ones((len(validation), 1), dtype=np.float32),
                validation_all_masks[:, 5:],
            ],
            axis=1,
        )
        harmful = gate_targets[:, 0] == 1
        if not harmful.any():
            raise ValueError("La cascada necesita daños explícitos en train")
        gate_spec = TrainingSpecification(**{**asdict(spec), "model_id": spec.model_id})
        previous_gate = (
            _candidate_asset(previous, previous.get("inference", {}).get("gate_model"))
            if previous
            else None
        )
        specialist_asset_key = "branch_model" if safety_first else "damage_model"
        previous_specialist = (
            _candidate_asset(
                previous, previous.get("inference", {}).get(specialist_asset_key)
            )
            if previous
            else None
        )
        gate_labels = ["ANY_DAMAGE", *all_labels[5:]]
        gate_tokenizer, gate_model = _build_hf_model(
            gate_spec, gate_labels, model_source=previous_gate
        )
        specialist_labels = (
            taxonomy.target_labels if safety_first else taxonomy.damage_labels
        )
        specialist_tokenizer, specialist_model = _build_hf_model(
            spec, specialist_labels, model_source=previous_specialist
        )
        gate_diagnostic = _token_truncation_diagnostic(
            gate_tokenizer, train, gate_spec.max_length
        )
        phase_started = time.perf_counter()
        gate_fit = _fit_hf(
            gate_model,
            gate_tokenizer,
            train,
            gate_targets,
            gate_masks,
            validation,
            validation_gate_targets,
            validation_gate_masks,
            gate_spec,
            run_dir / "trainer_gate",
            hardware,
            output_weights=[1.0]
            + [0.3] * len(taxonomy.fine_labels)
            + [0.2] * len(taxonomy.flags),
            primary_output_count=1,
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            progress_callback=progress_callback,
            progress_label="compuerta",
            persistent_checkpoint_dir=persistent_trainer_dir("trainer_gate"),
        )
        training_elapsed += time.perf_counter() - phase_started
        _notify_progress(
            progress_callback,
            status="progress",
            phase="compuerta entrenada",
            advance=1,
        )
        specialist_rows = (
            list(train)
            if safety_first
            else [row for row, keep in zip(train, harmful, strict=True) if keep]
        )
        specialist_targets = (
            train_full if safety_first else train_full[harmful][:, damage_indices]
        )
        specialist_masks = np.ones_like(specialist_targets, dtype=np.float32)
        validation_specialist_targets = (
            validation_all_targets[:, :5]
            if safety_first
            else validation_all_targets[:, damage_indices]
        )
        validation_specialist_masks = (
            validation_all_masks[:, :5]
            if safety_first
            else validation_all_masks[:, damage_indices]
        )
        specialist_diagnostic = _token_truncation_diagnostic(
            specialist_tokenizer, specialist_rows, spec.max_length
        )
        phase_started = time.perf_counter()
        specialist_fit = _fit_hf(
            specialist_model,
            specialist_tokenizer,
            specialist_rows,
            specialist_targets,
            specialist_masks,
            validation,
            validation_specialist_targets,
            validation_specialist_masks,
            spec,
            run_dir / ("trainer_branch" if safety_first else "trainer_damage"),
            hardware,
            structured_penalty=0.2 if safety_first else 0.0,
            primary_output_count=5 if safety_first else 4,
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            progress_callback=progress_callback,
            progress_label="rama especializada",
            persistent_checkpoint_dir=persistent_trainer_dir(
                "trainer_branch" if safety_first else "trainer_damage"
            ),
        )
        training_elapsed += time.perf_counter() - phase_started
        _notify_progress(
            progress_callback,
            status="progress",
            phase=(
                "rama de cinco salidas entrenada"
                if safety_first
                else "rama de daño entrenada"
            ),
            advance=1,
        )
        validation_started = time.perf_counter()
        gate_validation = _predict_hf(
            gate_model,
            gate_tokenizer,
            validation,
            spec.max_length,
            hardware,
            1 + auxiliary_count,
        )
        specialist_validation = _predict_hf(
            specialist_model,
            specialist_tokenizer,
            validation,
            spec.max_length,
            hardware,
            5 if safety_first else 4,
        )
        validation_inference_elapsed = time.perf_counter() - validation_started
        _notify_progress(
            progress_callback,
            status="progress",
            phase="validation inferida",
            advance=1,
        )
        if safety_first:
            gate_truth = validation_all_targets[:, damage_indices].max(axis=1)
            gate_calibration = calibrate_safety_first_gate(
                gate_truth,
                gate_validation[:, 0],
                min_damage_recall=cascade_gate_min_damage_recall,
                min_safe_npv=cascade_gate_min_safe_npv,
            )
            validation_primary = combine_safety_first_cascade_scores(
                gate_validation[:, 0],
                specialist_validation,
                gate_threshold=gate_calibration["threshold"],
            )
        else:
            validation_primary = np.concatenate(
                [
                    1 - gate_validation[:, :1],
                    gate_validation[:, :1] * specialist_validation,
                ],
                axis=1,
            )
        validation_scores = np.concatenate(
            [validation_primary, gate_validation[:, 1:]], axis=1
        )
        from sklearn.metrics import average_precision_score, f1_score

        gate_truth = validation_all_targets[:, damage_indices].max(axis=1)
        if safety_first:
            gate_threshold = float(gate_calibration["threshold"])
        else:
            gate_grid = np.linspace(0.05, 0.95, 91)
            gate_threshold = max(
                gate_grid,
                key=lambda value: (
                    f1_score(
                        gate_truth,
                        gate_validation[:, 0] >= value,
                        zero_division=0,
                    ),
                    value,
                ),
            )
        gate_missed = (gate_truth == 1) & (gate_validation[:, 0] < gate_threshold)
        cascade_diagnostics = {
            "gate_average_precision": float(
                average_precision_score(gate_truth, gate_validation[:, 0])
            ),
            "gate_threshold_validation": float(gate_threshold),
            "gate_f1_validation": float(
                f1_score(
                    gate_truth,
                    gate_validation[:, 0] >= gate_threshold,
                    zero_division=0,
                )
            ),
            "damage_rows_blocked_by_gate_fraction": float(
                gate_missed.sum() / max(1, gate_truth.sum())
            ),
            "comparison_against_flat": "performed in 03_07 on identical validation chunk_ids",
            "architecture": (
                "safety_first_gate_then_safe_plus_four_damage_branch"
                if safety_first
                else "soft_any_damage_gate_times_four_damage_branch"
            ),
            "safety_gate_calibration": gate_calibration if safety_first else None,
        }
        labels = all_labels
        fit_summary = {"gate": gate_fit, "specialist": specialist_fit}
        truncation = {"gate": gate_diagnostic, "specialist": specialist_diagnostic}
        gate_dir = run_dir / "gate_model"
        specialist_dir = run_dir / ("branch_model" if safety_first else "damage_model")
        _save_hf(gate_model, gate_tokenizer, gate_dir)
        _save_hf(
            specialist_model,
            specialist_tokenizer,
            specialist_dir,
        )
        inference = (
            {
                "type": "hf_cascade_v2",
                "gate_model": gate_dir.name,
                "branch_model": specialist_dir.name,
                "gate_threshold": float(gate_calibration["threshold"]),
                "gate_constraints": {
                    "minimum_damage_recall": cascade_gate_min_damage_recall,
                    "minimum_safe_npv": cascade_gate_min_safe_npv,
                },
            }
            if safety_first
            else {
                "type": "hf_cascade",
                "gate_model": gate_dir.name,
                "damage_model": specialist_dir.name,
            }
        )
        checkpoint_paths = [gate_dir, specialist_dir]
    else:
        targets, target_masks, labels = _output_targets_and_masks(train)
        validation_targets, validation_masks, _ = _output_targets_and_masks(validation)
        previous_source = None
        if previous and experiment != "qwen_lora":
            previous_source = _candidate_asset(
                previous, previous.get("inference", {}).get("model")
            )
        previous_adapter = (
            _candidate_asset(previous, previous.get("inference", {}).get("model"))
            if previous and experiment == "qwen_lora"
            else None
        )
        tokenizer, model = _build_hf_model(
            spec,
            labels,
            model_source=previous_source,
            lora=experiment == "qwen_lora",
            adapter_source=previous_adapter,
        )
        truncation = _token_truncation_diagnostic(tokenizer, train, spec.max_length)
        phase_started = time.perf_counter()
        fit_summary = _fit_hf(
            model,
            tokenizer,
            train,
            targets,
            target_masks,
            validation,
            validation_targets,
            validation_masks,
            spec,
            run_dir / "trainer",
            hardware,
            structured_penalty=0.2 if experiment == "qwen_structured" else 0.0,
            output_weights=[1.0] * 5
            + [0.3] * len(taxonomy.fine_labels)
            + [0.2] * len(taxonomy.flags),
            eval_batch_size=eval_batch_size,
            dataloader_num_workers=dataloader_num_workers,
            progress_callback=progress_callback,
            progress_label=experiment,
            persistent_checkpoint_dir=persistent_trainer_dir("trainer"),
        )
        training_elapsed = time.perf_counter() - phase_started
        _notify_progress(
            progress_callback,
            status="progress",
            phase="modelo entrenado",
            advance=1,
        )
        validation_started = time.perf_counter()
        validation_all = _predict_hf(
            model, tokenizer, validation, spec.max_length, hardware, len(labels)
        )
        validation_inference_elapsed = time.perf_counter() - validation_started
        _notify_progress(
            progress_callback,
            status="progress",
            phase="validation inferida",
            advance=1,
        )
        validation_scores = validation_all
        model_dir = run_dir / "model"
        _save_hf(model, tokenizer, model_dir)
        inference = {
            "type": (
                "hf_peft_sequence_classifier"
                if experiment == "qwen_lora"
                else "hf_sequence_classifier"
            ),
            "model": model_dir.name,
            "primary_output_count": 5,
            "output_count": len(labels),
            "output_labels": labels,
        }
        checkpoint_paths = [model_dir]

    metrics_started = time.perf_counter()
    thresholds, validation_metrics, auxiliary_metrics = _evaluate_validation(
        run_dir, validation, validation_scores, labels
    )
    validation_metrics_elapsed = time.perf_counter() - metrics_started
    write_json_atomic(run_dir / "truncation_diagnostic.json", truncation)
    write_json_atomic(run_dir / "fit_summary.json", fit_summary)
    bundle = run_dir / "inference.json"
    write_json_atomic(
        bundle, {**inference, "target_labels": list(taxonomy.target_labels)}
    )
    manifest = _checkpoint_manifest(run_dir, [*checkpoint_paths, bundle])
    candidate = {
        "schema_version": "2.1.0",
        "candidate_id": f"{run_prefix}-{signature[:12]}",
        "experiment": experiment,
        "model_family": experiment,
        "variant_id": variant_id or "default",
        "context_max_length": spec.max_length,
        "initialization": warm_start_identity,
        "run_signature": signature,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "target_labels": list(taxonomy.target_labels),
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "auxiliary_validation_metrics": auxiliary_metrics,
        "test_metrics": None,
        "test_status": "sealed_not_evaluated",
        "output_count": len(labels),
        "output_labels": labels,
        "training_sampling": sampling,
        "truncation_diagnostic": truncation,
        "cascade_diagnostics": cascade_diagnostics,
        "metrics_path": "metrics.json",
        "checkpoint_manifest": manifest.name,
        "inference": {**inference, "bundle": bundle.name},
        "hardware": hardware.model_dump(),
        "stage_timings_seconds": {
            "training_fit": training_elapsed,
            "validation_inference": validation_inference_elapsed,
            "validation_metrics_and_thresholds": validation_metrics_elapsed,
            "total_before_candidate_write": time.perf_counter() - run_started,
        },
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
            counters={
                "train_rows_after_safe_sampling": len(train),
                "validation_rows_4_to_1": len(validation),
                "test_rows_sealed": len(test),
            },
            warnings=(
                ["warm_start_from:" + previous["candidate_id"]] if previous else []
            ),
        ),
    )
    _notify_progress(
        progress_callback,
        status="finished",
        phase="candidato persistido",
        total=progress_total,
        completed=progress_total,
    )
    return {"status": "trained", "candidate": candidate, "run_signature": signature}


def train_flat_transformers(
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "auto",
    force: bool = False,
    safe_to_damage_ratio: float | None = 4.0,
    split_scheme: str = "video",
    sampling_seed: int = 20260805,
    persistent_checkpoint_root: str | Path | None = None,
    completion_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    experiments = ("flat_minilm", "flat_e5")
    _notify_progress(
        progress_callback,
        status="started",
        phase="transformers planos",
        total=len(experiments),
        advance=0,
    )
    results = []
    for experiment in experiments:
        result = train_neural_experiment(
                dataset_path,
                output_root,
                experiment=experiment,
                device=device,
                force=force,
                safe_to_damage_ratio=safe_to_damage_ratio,
                split_scheme=split_scheme,
                sampling_seed=sampling_seed,
                persistent_checkpoint_root=persistent_checkpoint_root,
            )
        results.append(result)
        if completion_callback is not None:
            completion_callback(
                {
                    "experiment": experiment,
                    "index": len(results),
                    "total": len(experiments),
                    "result": result,
                }
            )
        _notify_progress(
            progress_callback,
            status="progress",
            phase=experiment,
            advance=1,
        )
    _notify_progress(
        progress_callback,
        status="finished",
        phase="dos modelos completados",
        total=len(experiments),
        completed=len(experiments),
    )
    return {
        "status": (
            "noop"
            if all(result["status"] == "noop" for result in results)
            else "trained"
        ),
        "results": results,
    }
