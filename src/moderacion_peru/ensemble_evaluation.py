from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import deterministic_safe_downsample
from .device import resolve_device, torch_device_name
from .io import (
    canonical_json_sha256,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from .registry import discover_candidates
from .taxonomy import load_taxonomy
from .training import calibrate_thresholds, classification_metrics, encode_targets

DEFAULT_BOOTSTRAP_WORKERS = 4
BOOTSTRAP_ENGINE = "grouped-video-threaded-v2"
ProgressCallback = Callable[[dict[str, Any]], None]


def _notify_progress(callback: ProgressCallback | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def _candidate_slot(candidate: Mapping[str, Any]) -> str:
    family = str(candidate.get("model_family", "")).casefold()
    if family.startswith("classical:"):
        return "classical"
    if family.startswith("qwen") or "prompt_sft" in family:
        return "qwen"
    return "transformer"


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Ranking no degenerado: AP/F1 con control posterior de errores."""

    return (
        float(metrics.get("average_precision_macro_damage", 0.0)),
        float(metrics.get("f1_macro_damage", 0.0)),
        float(metrics.get("any_damage", {}).get("recall", 0.0)),
        -float(metrics.get("false_alarm_rate_on_safe", 1.0)),
        -float(metrics.get("review_load_rate", 1.0)),
    )


def _load_validation_predictions(
    candidate: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    taxonomy = load_taxonomy()
    path = (
        Path(str(candidate["candidate_path"])).parent / "predictions_validation.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = list(read_jsonl(path))
    scores = np.asarray(
        [
            [float(row["scores"][label]) for label in taxonomy.target_labels]
            for row in rows
        ],
        dtype=float,
    )
    return rows, scores


def _align_predictions(
    loaded: Sequence[tuple[Mapping[str, Any], list[dict[str, Any]], np.ndarray]],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    if not loaded:
        raise ValueError("No existen predicciones para alinear")
    reference_rows = loaded[0][1]
    reference_ids = [str(row["chunk_id"]) for row in reference_rows]
    aligned = []
    for candidate, rows, scores in loaded:
        index = {str(row["chunk_id"]): position for position, row in enumerate(rows)}
        if set(index) != set(reference_ids):
            raise ValueError(
                f"Validation no coincide para {candidate.get('candidate_id')}"
            )
        aligned.append(
            np.asarray([scores[index[chunk_id]] for chunk_id in reference_ids])
        )
    return reference_rows, aligned


def _pareto_front(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Frontera AP/F1/recall máximos y falsas alarmas/revisión mínimos."""

    vectors = {
        row["candidate_id"]: np.asarray(
            _selection_key(row["validation_metrics"]), dtype=float
        )
        for row in rows
    }
    frontier = []
    for identifier, vector in vectors.items():
        dominated = any(
            other != identifier
            and np.all(other_vector >= vector)
            and np.any(other_vector > vector)
            for other, other_vector in vectors.items()
        )
        if not dominated:
            frontier.append(identifier)
    return sorted(frontier)


def _grouped_bootstrap_macro_ap(
    truth: np.ndarray,
    scores: np.ndarray,
    video_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from joblib import Parallel, delayed
    from sklearn.metrics import average_precision_score

    if replicates < 1:
        raise ValueError("replicates debe ser al menos 1")
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        groups[str(video_id)].append(index)
    group_indices = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    damage_indices = np.asarray(
        [
            load_taxonomy().target_labels.index(label)
            for label in load_taxonomy().damage_labels
        ],
        dtype=int,
    )

    def macro_damage_ap(indices: np.ndarray) -> float:
        values = []
        selected_truth = truth[indices]
        selected_scores = scores[indices]
        for label_index in damage_indices:
            label_truth = selected_truth[:, label_index]
            values.append(
                float(
                    average_precision_score(
                        label_truth, selected_scores[:, label_index]
                    )
                )
                if label_truth.any()
                else 0.0
            )
        return float(np.mean(values))

    master_rng = np.random.default_rng(seed)
    replicate_seeds = master_rng.integers(
        0, np.iinfo(np.uint64).max, size=replicates, dtype=np.uint64
    )
    workers = min(parallel_workers, replicates, os.cpu_count() or 1)
    seed_chunks = np.array_split(replicate_seeds, workers)

    def evaluate_chunk(seed_chunk: np.ndarray) -> list[float]:
        values = []
        for replicate_seed in seed_chunk:
            rng = np.random.default_rng(int(replicate_seed))
            sampled = rng.integers(0, len(group_indices), size=len(group_indices))
            indices = np.concatenate([group_indices[index] for index in sampled])
            values.append(macro_damage_ap(indices))
        _notify_progress(
            progress_callback,
            status="progress",
            phase="bootstrap AP agrupado por video",
            advance=len(seed_chunk),
        )
        return values

    chunks = Parallel(n_jobs=workers, prefer="threads")(
        delayed(evaluate_chunk)(seed_chunk) for seed_chunk in seed_chunks
    )
    estimates = [value for chunk in chunks for value in chunk]
    point = macro_damage_ap(np.arange(len(truth), dtype=int))
    return {
        "metric": "average_precision_macro_damage",
        "point": point,
        "confidence_level": 0.95,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
        "grouping": "video_id",
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "parallel_workers": workers,
        "samples": estimates,
    }


def _paired_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    challenger: np.ndarray,
    video_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    from joblib import Parallel, delayed
    from sklearn.metrics import average_precision_score

    if replicates < 1:
        raise ValueError("replicates debe ser al menos 1")
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        groups[str(video_id)].append(index)
    group_indices = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    taxonomy = load_taxonomy()
    damage_indices = np.asarray(
        [taxonomy.target_labels.index(label) for label in taxonomy.damage_labels],
        dtype=int,
    )

    def macro_damage_ap(scores: np.ndarray, indices: np.ndarray) -> float:
        selected_truth = truth[indices]
        selected_scores = scores[indices]
        values = []
        for label_index in damage_indices:
            label_truth = selected_truth[:, label_index]
            values.append(
                float(
                    average_precision_score(
                        label_truth, selected_scores[:, label_index]
                    )
                )
                if label_truth.any()
                else 0.0
            )
        return float(np.mean(values))

    master_rng = np.random.default_rng(seed)
    replicate_seeds = master_rng.integers(
        0, np.iinfo(np.uint64).max, size=replicates, dtype=np.uint64
    )
    workers = min(parallel_workers, replicates, os.cpu_count() or 1)
    seed_chunks = np.array_split(replicate_seeds, workers)

    def evaluate_chunk(seed_chunk: np.ndarray) -> list[float]:
        values = []
        for replicate_seed in seed_chunk:
            rng = np.random.default_rng(int(replicate_seed))
            sampled = rng.integers(0, len(group_indices), size=len(group_indices))
            indices = np.concatenate([group_indices[index] for index in sampled])
            values.append(
                macro_damage_ap(challenger, indices)
                - macro_damage_ap(reference, indices)
            )
        _notify_progress(
            progress_callback,
            status="progress",
            phase="bootstrap pareado",
            advance=len(seed_chunk),
        )
        return values

    chunks = Parallel(n_jobs=workers, prefer="threads")(
        delayed(evaluate_chunk)(seed_chunk) for seed_chunk in seed_chunks
    )
    differences = [value for chunk in chunks for value in chunk]
    values = np.asarray(differences)
    p_value = min(
        1.0, 2 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    )
    return {
        "difference_challenger_minus_reference": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_raw": p_value,
        "replicates": replicates,
        "grouping": "video_id",
        "bootstrap_engine": BOOTSTRAP_ENGINE,
        "parallel_workers": workers,
    }


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_value_raw"])
    running = 0.0
    count = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(row["p_value_raw"]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def compare_and_freeze_validation(
    dataset_path: str | Path,
    candidate_roots: Iterable[str | Path],
    comparison_path: str | Path,
    freeze_path: str | Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 20260805,
    parallel_workers: int = DEFAULT_BOOTSTRAP_WORKERS,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compara individuos/ensembles en validation y congela sin abrir test."""

    comparison_started = time.perf_counter()
    dataset = Path(dataset_path).resolve()
    if parallel_workers < 1:
        raise ValueError("parallel_workers debe ser al menos 1")
    parallel_workers = min(parallel_workers, os.cpu_count() or 1)
    dataset_sha = sha256_file(dataset)
    taxonomy = load_taxonomy()
    eligible = []
    rejected = []
    for candidate in discover_candidates(candidate_roots):
        reasons = []
        if candidate.get("status") != "complete":
            reasons.append("incomplete")
        if candidate.get("dataset_sha256") != dataset_sha:
            reasons.append("different_snapshot")
        if tuple(candidate.get("target_labels", ())) != taxonomy.target_labels:
            reasons.append("wrong_contract")
        if candidate.get("test_metrics") not in (None, {}):
            reasons.append("test_was_opened_before_freeze")
        predictions = (
            Path(candidate["candidate_path"]).parent / "predictions_validation.jsonl"
        )
        if not predictions.is_file():
            reasons.append("validation_predictions_missing")
        if reasons:
            rejected.append(
                {"candidate_id": candidate.get("candidate_id"), "reasons": reasons}
            )
        else:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("No hay candidatos elegibles con validation y test sellado")
    signature = canonical_json_sha256(
        {
            "dataset_sha256": dataset_sha,
            "candidate_ids": sorted(row["candidate_id"] for row in eligible),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "parallel_workers": parallel_workers,
            "seed": seed,
        }
    )
    freeze = Path(freeze_path)
    comparison = Path(comparison_path)
    if freeze.is_file() and comparison.is_file() and not force:
        previous = json.loads(freeze.read_text(encoding="utf-8"))
        if previous.get("comparison_signature") == signature:
            return {
                "status": "noop",
                "freeze": str(freeze),
                "comparison": str(comparison),
            }

    best_by_slot = {
        slot: max(
            (row for row in eligible if _candidate_slot(row) == slot),
            key=lambda row: _selection_key(row["validation_metrics"]),
        )
        for slot in ("classical", "transformer", "qwen")
        if any(_candidate_slot(row) == slot for row in eligible)
    }
    members = list(best_by_slot.values())
    loaded = [
        (candidate, *_load_validation_predictions(candidate)) for candidate in members
    ]
    validation_rows, score_matrices = _align_predictions(loaded)
    truth_rows = [{"coarse_labels": row["true_labels"]} for row in validation_rows]
    truth = encode_targets(truth_rows)
    video_ids = [str(row["video_id"]) for row in validation_rows]

    evaluated: list[dict[str, Any]] = []
    score_by_id: dict[str, np.ndarray] = {}
    thresholds_by_id: dict[str, dict[str, float]] = {}
    for candidate in eligible:
        rows, scores = _load_validation_predictions(candidate)
        _, aligned = _align_predictions([loaded[0], (candidate, rows, scores)])
        candidate_scores = aligned[1]
        thresholds = calibrate_thresholds(truth, candidate_scores)
        metrics = classification_metrics(truth, candidate_scores, thresholds)
        identifier = str(candidate["candidate_id"])
        score_by_id[identifier] = candidate_scores
        thresholds_by_id[identifier] = thresholds
        evaluated.append(
            {
                "candidate_id": identifier,
                "kind": "individual",
                "model_family": candidate["model_family"],
                "members": [identifier],
                "validation_metrics": metrics,
            }
        )

    if len(score_matrices) >= 2:
        weights = np.asarray(
            [
                max(
                    1e-9,
                    float(
                        row["validation_metrics"].get(
                            "average_precision_macro_damage", 0.0
                        )
                    ),
                )
                for row in members
            ]
        )
        weights /= weights.sum()
        member_thresholds = np.asarray(
            [
                [float(row["thresholds"][label]) for label in taxonomy.target_labels]
                for row in members
            ]
        )
        hard = np.asarray(
            [
                scores >= member_thresholds[index]
                for index, scores in enumerate(score_matrices)
            ],
            dtype=float,
        ).mean(axis=0)
        ensemble_scores = {
            "ensemble_soft_mean": np.mean(score_matrices, axis=0),
            "ensemble_soft_validation_weighted": np.average(
                score_matrices, axis=0, weights=weights
            ),
            "ensemble_hard_majority": hard,
            "ensemble_union": np.max(score_matrices, axis=0),
            "ensemble_intersection": np.min(score_matrices, axis=0),
        }
        member_ids = [str(row["candidate_id"]) for row in members]
        for identifier, scores in ensemble_scores.items():
            thresholds = calibrate_thresholds(truth, scores)
            metrics = classification_metrics(truth, scores, thresholds)
            score_by_id[identifier] = scores
            thresholds_by_id[identifier] = thresholds
            evaluated.append(
                {
                    "candidate_id": identifier,
                    "kind": "ensemble",
                    "model_family": "ensemble",
                    "members": member_ids,
                    "weights": weights.tolist() if "weighted" in identifier else None,
                    "validation_metrics": metrics,
                }
            )

    bootstrap_started = time.perf_counter()
    ensemble_count = sum(row["kind"] == "ensemble" for row in evaluated)
    bootstrap_total = bootstrap_replicates * (len(evaluated) + ensemble_count)
    _notify_progress(
        progress_callback,
        status="started",
        phase="bootstrap agrupado por video",
        total=bootstrap_total,
        advance=0,
    )
    for row in evaluated:
        bootstrap = _grouped_bootstrap_macro_ap(
            truth,
            score_by_id[row["candidate_id"]],
            video_ids,
            replicates=bootstrap_replicates,
            seed=seed,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback,
        )
        bootstrap.pop("samples")
        row["bootstrap_grouped_by_video"] = bootstrap
    frontier = _pareto_front(evaluated)
    selected = max(evaluated, key=lambda row: _selection_key(row["validation_metrics"]))
    reference_individual = max(
        (row for row in evaluated if row["kind"] == "individual"),
        key=lambda row: _selection_key(row["validation_metrics"]),
    )
    tests = []
    for row in evaluated:
        if row["kind"] != "ensemble":
            continue
        test = _paired_bootstrap(
            truth,
            score_by_id[reference_individual["candidate_id"]],
            score_by_id[row["candidate_id"]],
            video_ids,
            replicates=bootstrap_replicates,
            seed=seed + len(tests) + 1,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback,
        )
        test.update(
            {
                "reference": reference_individual["candidate_id"],
                "challenger": row["candidate_id"],
            }
        )
        tests.append(test)
    _holm_adjust(tests)
    _notify_progress(
        progress_callback,
        status="finished",
        phase="bootstrap completo",
        total=bootstrap_total,
        completed=bootstrap_total,
    )
    bootstrap_elapsed = time.perf_counter() - bootstrap_started
    diversity = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            left_binary = score_matrices[left] >= member_thresholds[left]
            right_binary = score_matrices[right] >= member_thresholds[right]
            diversity.append(
                {
                    "left": members[left]["candidate_id"],
                    "right": members[right]["candidate_id"],
                    "labelwise_disagreement_rate": float(
                        (left_binary != right_binary).mean()
                    ),
                }
            )

    comparison_payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "dataset_sha256": dataset_sha,
        "selection_split": "validation",
        "test_status": "sealed_not_evaluated",
        "runtime_optimization": {
            "bootstrap_engine": BOOTSTRAP_ENGINE,
            "parallel_workers": parallel_workers,
            "shared_memory_threads": True,
        },
        "stage_timings_seconds": {
            "grouped_and_paired_bootstrap": bootstrap_elapsed,
            "comparison_total_before_write": time.perf_counter() - comparison_started,
        },
        "selection_policy": "Pareto report; AP macro de daños como desempate predeclarado, luego F1/recall/falsas alarmas/review",
        "pareto_front": frontier,
        "selected_for_freeze": selected["candidate_id"],
        "best_individual": reference_individual["candidate_id"],
        "best_by_family_slot": {
            key: row["candidate_id"] for key, row in best_by_slot.items()
        },
        "ranking": sorted(
            evaluated,
            key=lambda row: _selection_key(row["validation_metrics"]),
            reverse=True,
        ),
        "diversity": diversity,
        "paired_bootstrap_tests_holm": tests,
        "stacking_oof": {
            "status": "not_available",
            "reason": "requires out-of-fold predictions from every selected member; validation is not used to fit a meta-learner",
        },
        "rejected": rejected,
    }
    freeze_payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha,
        "selected_id": selected["candidate_id"],
        "selected_kind": selected["kind"],
        "members": selected["members"],
        "member_candidate_paths": {
            row["candidate_id"]: row["candidate_path"]
            for row in eligible
            if row["candidate_id"] in selected["members"]
        },
        "ensemble_weights": selected.get("weights"),
        "thresholds": thresholds_by_id[selected["candidate_id"]],
        "test_status": "sealed_ready_for_single_open",
        "publication_approved": False,
    }
    write_json_atomic(comparison, comparison_payload)
    write_json_atomic(freeze, freeze_payload)
    return {
        "status": "frozen",
        "selected": selected["candidate_id"],
        "pareto_front": frontier,
        "comparison": str(comparison),
        "freeze": str(freeze),
        "test_status": "sealed_ready_for_single_open",
    }


def _asset(candidate: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    return (
        path
        if path.is_absolute()
        else Path(str(candidate["candidate_path"])).parent / path
    )


def _score_candidate(
    candidate: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    device: str,
    progress_callback: ProgressCallback | None = None,
    progress_phase: str = "inferencia",
) -> np.ndarray:
    inference = candidate["inference"]
    kind = inference["type"]
    texts = [str(row["text"]) for row in rows]
    if kind == "sklearn_joblib":
        import joblib

        bundle = json.loads(
            _asset(candidate, inference["bundle"]).read_text(encoding="utf-8")
        )
        model = joblib.load(
            _asset(candidate, inference["bundle"]).parent / bundle["model"]
        )
        _notify_progress(
            progress_callback,
            status="started",
            phase=progress_phase,
            total=1,
            advance=0,
        )
        values = np.asarray(model.predict_proba(texts), dtype=float)
        _notify_progress(
            progress_callback,
            status="finished",
            phase=progress_phase,
            total=1,
            completed=1,
        )
        return values[:, :5]
    if kind == "hf_prompt_sft_json":
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from .prompt_sft import _generate_json_scores

        model_path = _asset(candidate, inference["model"])
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoPeftModelForCausalLM.from_pretrained(model_path)
        hardware = resolve_device(device)
        torch_device = torch_device_name(hardware)
        model.to(torch_device)
        provenance = json.loads(
            _asset(candidate, inference["prompt_capsule"]).read_text(encoding="utf-8")
        )

        def relay_generation_progress(event: dict[str, Any]) -> None:
            forwarded = dict(event)
            detail = str(forwarded.get("phase") or "generación")
            forwarded["phase"] = f"{progress_phase} · {detail}"
            _notify_progress(progress_callback, **forwarded)

        values, _quality = _generate_json_scores(
            model,
            tokenizer,
            rows,
            provenance["capsule"],
            device=torch_device,
            max_input_length=3840,
            progress_callback=relay_generation_progress,
        )
        return values[:, :5]
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Instale moderacion-peru[entrenamiento] para abrir test"
        ) from exc
    hardware = resolve_device(device)
    torch_device = torch_device_name(hardware)

    def sequence_scores(path: Path, count: int, phase: str) -> np.ndarray:
        tokenizer = AutoTokenizer.from_pretrained(path)
        if kind == "hf_peft_sequence_classifier":
            from peft import AutoPeftModelForSequenceClassification

            model = AutoPeftModelForSequenceClassification.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(torch_device).eval()
        batches = []
        batch_size = 16
        batch_total = max(1, math.ceil(len(texts) / batch_size))
        _notify_progress(
            progress_callback,
            status="started",
            phase=phase,
            total=batch_total,
            advance=0,
        )
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                encoded = {
                    key: value.to(torch_device) for key, value in encoded.items()
                }
                batches.append(
                    torch.sigmoid(model(**encoded).logits).float().cpu().numpy()
                )
                _notify_progress(
                    progress_callback,
                    status="progress",
                    phase=phase,
                    advance=1,
                )
        _notify_progress(
            progress_callback,
            status="finished",
            phase=phase,
            total=batch_total,
            completed=batch_total,
        )
        return np.concatenate(batches, axis=0)[:, :count]

    if kind in {"hf_sequence_classifier", "hf_peft_sequence_classifier"}:
        return sequence_scores(_asset(candidate, inference["model"]), 5, progress_phase)
    if kind == "hf_cascade":
        gate = sequence_scores(
            _asset(candidate, inference["gate_model"]),
            1,
            f"{progress_phase} · compuerta",
        )
        original_kind = inference["type"]
        inference["type"] = "hf_sequence_classifier"
        try:
            damage = sequence_scores(
                _asset(candidate, inference["damage_model"]),
                4,
                f"{progress_phase} · daño",
            )
        finally:
            inference["type"] = original_kind
        return np.concatenate([1 - gate, gate * damage], axis=1)
    raise ValueError(f"Inferencia no soportada en apertura de test: {kind}")


def evaluate_frozen_test(
    freeze_path: str | Path,
    destination_path: str | Path,
    *,
    confirm_single_test_open: bool = False,
    device: str = "auto",
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Abre test una sola vez para una decisión previamente congelada."""

    if not confirm_single_test_open:
        return {
            "status": "sealed",
            "message": "Defina confirm_single_test_open=True solo después de revisar la congelación",
        }
    evaluation_started = time.perf_counter()
    freeze_path = Path(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    destination = Path(destination_path)
    if destination.is_file() and not force:
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("comparison_signature") == freeze.get("comparison_signature"):
            return {"status": "noop", "test_report": str(destination)}
        raise ValueError("Ya existe una apertura de test para otra congelación")
    dataset = Path(freeze["dataset"])
    if sha256_file(dataset) != freeze["dataset_sha256"]:
        raise ValueError("El snapshot cambió después de congelar la selección")
    candidates = {
        identifier: json.loads(Path(path).read_text(encoding="utf-8"))
        for identifier, path in freeze["member_candidate_paths"].items()
    }
    for identifier, path in freeze["member_candidate_paths"].items():
        candidates[identifier]["candidate_path"] = path
    first = next(iter(candidates.values()))
    split_field = first.get("training_sampling", {}).get("split_field", "split")
    test_rows = [row for row in read_jsonl(dataset) if row.get(split_field) == "test"]
    sampling_policy = first.get("training_sampling", {})
    test_4_to_1, test_sampling_4_to_1 = deterministic_safe_downsample(
        test_rows,
        safe_to_damage_ratio=float(sampling_policy.get("safe_to_damage_ratio", 4.0)),
        seed=int(sampling_policy.get("sampling_seed", 20260805)),
    )
    if not test_rows:
        raise ValueError("No hay filas en el test congelado")
    inference_started = time.perf_counter()
    member_scores = {}
    for identifier, candidate in candidates.items():
        member_scores[identifier] = _score_candidate(
            candidate,
            test_rows,
            device=device,
            progress_callback=progress_callback,
            progress_phase=f"test · {identifier}",
        )
    inference_elapsed = time.perf_counter() - inference_started
    matrices = list(member_scores.values())
    selected_id = freeze["selected_id"]
    if freeze["selected_kind"] == "individual":
        scores = member_scores[selected_id]
    elif selected_id == "ensemble_soft_mean":
        scores = np.mean(matrices, axis=0)
    elif selected_id == "ensemble_soft_validation_weighted":
        scores = np.average(matrices, axis=0, weights=freeze["ensemble_weights"])
    elif selected_id == "ensemble_union":
        scores = np.max(matrices, axis=0)
    elif selected_id == "ensemble_intersection":
        scores = np.min(matrices, axis=0)
    elif selected_id == "ensemble_hard_majority":
        taxonomy = load_taxonomy()
        binary = []
        for identifier, matrix in member_scores.items():
            threshold = np.asarray(
                [
                    candidates[identifier]["thresholds"][label]
                    for label in taxonomy.target_labels
                ]
            )
            binary.append(matrix >= threshold)
        scores = np.mean(binary, axis=0)
    else:
        raise ValueError(f"Regla congelada desconocida: {selected_id}")
    metrics_started = time.perf_counter()
    truth = encode_targets(test_rows)
    metrics_natural = classification_metrics(truth, scores, freeze["thresholds"])
    controlled_ids = {row["chunk_id"] for row in test_4_to_1}
    controlled_indices = np.asarray(
        [
            index
            for index, row in enumerate(test_rows)
            if row["chunk_id"] in controlled_ids
        ],
        dtype=int,
    )
    metrics_4_to_1 = classification_metrics(
        truth[controlled_indices],
        scores[controlled_indices],
        freeze["thresholds"],
    )
    metrics_elapsed = time.perf_counter() - metrics_started
    taxonomy = load_taxonomy()
    write_jsonl_atomic(
        destination.with_name(destination.stem + "_predictions.jsonl"),
        [
            {
                "chunk_id": row["chunk_id"],
                "video_id": row["video_id"],
                "scores": {
                    label: float(scores[index, label_index])
                    for label_index, label in enumerate(taxonomy.target_labels)
                },
                "true_labels": row["coarse_labels"],
            }
            for index, row in enumerate(test_rows)
        ],
    )
    payload = {
        "schema_version": "4.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": freeze["comparison_signature"],
        "dataset_sha256": freeze["dataset_sha256"],
        "selected_id": selected_id,
        "test_open_count": 1,
        "inference_passes": 1,
        "test_rows_natural": len(test_rows),
        "test_rows_4_to_1": len(test_4_to_1),
        "test_sampling_4_to_1": test_sampling_4_to_1,
        "primary_metrics_natural_prevalence": metrics_natural,
        "secondary_metrics_4_to_1": metrics_4_to_1,
        "metrics": metrics_natural,
        "interpretation": (
            "The full natural-prevalence test is primary. The deterministic 4:1 "
            "view reuses the same predictions and is secondary; it does not open "
            "or score the test a second time."
        ),
        "stage_timings_seconds": {
            "test_inference_all_selected_members": inference_elapsed,
            "natural_and_4_to_1_metrics": metrics_elapsed,
            "test_total_before_report_write": time.perf_counter() - evaluation_started,
        },
    }
    write_json_atomic(destination, payload)
    return {
        "status": "test_evaluated_once",
        "test_report": str(destination),
        "primary_metrics_natural_prevalence": metrics_natural,
        "secondary_metrics_4_to_1": metrics_4_to_1,
    }
