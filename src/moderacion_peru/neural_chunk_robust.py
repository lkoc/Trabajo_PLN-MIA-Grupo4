from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunk_optimization import (
    CHUNK_SELECTION_VERSION,
    DEFAULT_CHUNK_SMOKE_HF_MODEL,
    DEFAULT_CHUNK_SMOKE_HF_REVISION,
    DEFAULT_CHUNK_SMOKE_OLLAMA_MODEL,
    _bounded_neural_rows,
    _frozen_hf_embeddings,
    _load_selected_transcripts,
    build_temporal_label_references,
)
from .incremental import normalize_text
from .io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from .taxonomy import load_taxonomy


NEURAL_ROBUST_VERSION = "1.0.0"
MINILM_NONINFERIORITY_VERSION = "1.0.0"
DEFAULT_NEURAL_CANDIDATE_SECONDS = (15.0, 20.0, 25.0, 30.0, 35.0)
DEFAULT_NEURAL_SEEDS = (20260805, 20260817, 20260829, 20260841, 20260853)
DEFAULT_MINILM_NONINFERIORITY_SECONDS = (20.0, 30.0)
DEFAULT_MINILM_NONINFERIORITY_REPEATS = (20260901, 20260913, 20260925)


def _require_five_candidates(candidate_seconds: Sequence[float]) -> tuple[float, ...]:
    seconds = tuple(float(value) for value in candidate_seconds)
    if seconds != DEFAULT_NEURAL_CANDIDATE_SECONDS:
        raise ValueError(
            "El perfil neuronal robusto exige exactamente "
            "CANDIDATE_SECONDS=(15,20,25,30,35) y conserva ese orden"
        )
    return seconds


def _stable_key(seed: int, *values: object) -> str:
    payload = "|".join([str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _damage_rows_by_video(
    references: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    taxonomy = load_taxonomy()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        if any(label in row["coarse_labels"] for label in taxonomy.damage_labels):
            grouped[str(row["video_id"])].append(row)
    return grouped


def select_neural_anchor_references(
    references: Sequence[dict[str, Any]],
    *,
    panel_size: int = 100,
    minimum_damage_anchors_per_label: int = 20,
    max_anchors_per_video: int = 2,
    selection_seed: int = 20260807,
) -> list[dict[str, Any]]:
    """Construye un panel enriquecido de validation sin consultar test.

    Primero conserva un ancla de cada video de validation que contiene daño,
    favoreciendo filas multietiqueta y daños con menor soporte. Después completa
    cualquier daño por debajo de la cuota y finalmente añade anclas SEGURO de
    videos distintos. El bootstrap posterior agrupa todas las anclas del mismo
    video.
    """

    if panel_size <= 0 or minimum_damage_anchors_per_label <= 0:
        raise ValueError("El tamaño del panel y las cuotas deben ser positivos")
    if max_anchors_per_video <= 0:
        raise ValueError("max_anchors_per_video debe ser positivo")
    taxonomy = load_taxonomy()
    validation = [
        dict(row) for row in references if str(row.get("split")) == "validation"
    ]
    if not validation:
        raise ValueError("No existen referencias de validation para el panel neuronal")
    damage_by_video = _damage_rows_by_video(validation)
    video_frequency = {
        label: len(
            {
                str(row["video_id"])
                for row in validation
                if label in row["coarse_labels"]
            }
        )
        for label in taxonomy.damage_labels
    }
    if any(video_frequency[label] == 0 for label in taxonomy.damage_labels):
        raise ValueError("El panel no puede cubrir los cuatro daños")

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    anchors_by_video: Counter[str] = Counter()

    def add(row: dict[str, Any], stratum: str) -> None:
        item = dict(row)
        item["anchor_id"] = str(item["chunk_id"])
        item["primary_stratum"] = stratum
        selected.append(item)
        selected_ids.add(str(item["chunk_id"]))
        anchors_by_video[str(item["video_id"])] += 1

    # Una unidad de daño por video maximiza primero el número de clusters.
    for video_id in sorted(damage_by_video):
        if len(selected) >= panel_size:
            break
        candidates = damage_by_video[video_id]
        chosen = max(
            candidates,
            key=lambda row: (
                sum(
                    1.0 / video_frequency[label]
                    for label in row["coarse_labels"]
                    if label in video_frequency
                ),
                _stable_key(selection_seed, "damage", row["chunk_id"]),
            ),
        )
        add(chosen, "DAMAGE_VIDEO")

    label_counts = Counter(label for row in selected for label in row["coarse_labels"])
    for label in sorted(
        taxonomy.damage_labels,
        key=lambda value: (label_counts[value], video_frequency[value], value),
    ):
        ordered = sorted(
            (row for row in validation if label in row["coarse_labels"]),
            key=lambda row: _stable_key(
                selection_seed, "topup", label, row["chunk_id"]
            ),
        )
        for row in ordered:
            if label_counts[label] >= minimum_damage_anchors_per_label:
                break
            video_id = str(row["video_id"])
            if (
                str(row["chunk_id"]) in selected_ids
                or anchors_by_video[video_id] >= max_anchors_per_video
                or len(selected) >= panel_size
            ):
                continue
            add(row, f"TOPUP_{label}")
            label_counts.update(row["coarse_labels"])

    missing_damage = {
        label: minimum_damage_anchors_per_label - label_counts[label]
        for label in taxonomy.damage_labels
        if label_counts[label] < minimum_damage_anchors_per_label
    }
    if missing_damage:
        raise ValueError(f"No se alcanzaron las cuotas de daño: {missing_damage}")

    safe_rows = sorted(
        (row for row in validation if row["coarse_labels"] == [taxonomy.safe_label]),
        key=lambda row: _stable_key(selection_seed, "safe", row["chunk_id"]),
    )
    for row in safe_rows:
        if len(selected) >= panel_size:
            break
        video_id = str(row["video_id"])
        if str(row["chunk_id"]) in selected_ids or anchors_by_video[video_id]:
            continue
        add(row, taxonomy.safe_label)

    if len(selected) < panel_size:
        ordered = sorted(
            validation,
            key=lambda row: _stable_key(selection_seed, "fill", row["chunk_id"]),
        )
        for row in ordered:
            if len(selected) >= panel_size:
                break
            video_id = str(row["video_id"])
            if (
                str(row["chunk_id"]) in selected_ids
                or anchors_by_video[video_id] >= max_anchors_per_video
            ):
                continue
            add(row, "FILL")
    if len(selected) != panel_size:
        raise ValueError(
            f"Solo se pudieron seleccionar {len(selected)} de {panel_size} anclas"
        )

    return sorted(
        selected,
        key=lambda row: _stable_key(selection_seed, "panel", row["anchor_id"]),
    )


def _assign_disjoint_reporting_cohorts(
    anchors: Sequence[dict[str, Any]],
    *,
    cohort_count: int,
    selection_seed: int,
) -> dict[str, int]:
    if cohort_count <= 0:
        raise ValueError("cohort_count debe ser positivo")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_video[str(anchor["video_id"])].append(anchor)
    taxonomy = load_taxonomy()
    bins: list[list[dict[str, Any]]] = [[] for _ in range(cohort_count)]
    bin_labels = [Counter() for _ in range(cohort_count)]
    groups = sorted(
        by_video.items(),
        key=lambda item: (
            -sum(
                label in row["coarse_labels"]
                for row in item[1]
                for label in taxonomy.damage_labels
            ),
            _stable_key(selection_seed, "cohort", item[0]),
        ),
    )
    for _, rows in groups:
        row_labels = Counter(label for row in rows for label in row["coarse_labels"])
        target = min(
            range(cohort_count),
            key=lambda index: (
                len(bins[index]),
                sum(bin_labels[index][label] for label in row_labels),
                index,
            ),
        )
        bins[target].extend(rows)
        bin_labels[target].update(row_labels)
    return {
        str(row["anchor_id"]): index + 1
        for index, rows in enumerate(bins)
        for row in rows
    }


def _window_from_transcript(
    transcript: dict[str, Any],
    *,
    center_seconds: float,
    duration_seconds: float,
) -> dict[str, Any]:
    start = max(0.0, center_seconds - duration_seconds / 2.0)
    end = center_seconds + duration_seconds / 2.0
    pieces = []
    for segment in transcript.get("segments", []):
        segment_start = float(segment.get("start", 0.0))
        segment_end = segment_start + max(0.0, float(segment.get("duration", 0.0)))
        if segment_end > start and segment_start < end:
            text = normalize_text(str(segment.get("text", "")))
            if text:
                pieces.append(text)
    text = normalize_text(" ".join(pieces))
    if not text:
        raise ValueError(
            f"La ventana {start:.3f}-{end:.3f} no contiene texto normalizable"
        )
    return {
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "text": text,
        "text_sha256": sha256_text(text),
    }


def build_paired_neural_panel(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = DEFAULT_NEURAL_CANDIDATE_SECONDS,
    panel_size: int = 100,
    minimum_damage_anchors_per_label: int = 20,
    max_anchors_per_video: int = 2,
    reporting_cohorts: int = 5,
    selection_seed: int = 20260807,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seconds = _require_five_candidates(candidate_seconds)
    if not seconds or any(value <= 0 for value in seconds):
        raise ValueError("candidate_seconds debe contener duraciones positivas")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "paired_validation_panel.jsonl"
    manifest_path = output / "paired_validation_panel_manifest.json"
    configuration = {
        "transcript_sha256": sha256_file(transcript_path),
        "chunks_sha256": sha256_file(chunks_path),
        "dataset_sha256": sha256_file(dataset_path),
        "candidate_seconds": list(seconds),
        "panel_size": int(panel_size),
        "minimum_damage_anchors_per_label": int(minimum_damage_anchors_per_label),
        "max_anchors_per_video": int(max_anchors_per_video),
        "reporting_cohorts": int(reporting_cohorts),
        "selection_seed": int(selection_seed),
        "split": "validation",
        "window_policy": "same_reference_midpoint_centered_window",
    }
    signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    if panel_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if cached.get("run_signature") == signature and cached.get(
            "panel_sha256"
        ) == sha256_file(panel_path):
            return list(read_jsonl(panel_path)), cached

    references, reference_match = build_temporal_label_references(
        chunks_path, dataset_path
    )
    selected = select_neural_anchor_references(
        references,
        panel_size=panel_size,
        minimum_damage_anchors_per_label=minimum_damage_anchors_per_label,
        max_anchors_per_video=max_anchors_per_video,
        selection_seed=selection_seed,
    )
    video_ids = {str(row["video_id"]) for row in selected}
    transcripts = _load_selected_transcripts(transcript_path, video_ids)
    missing = sorted(video_ids - set(transcripts))
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} transcripciones del panel; ejemplo: {missing[0]}"
        )
    cohort_by_anchor = _assign_disjoint_reporting_cohorts(
        selected,
        cohort_count=reporting_cohorts,
        selection_seed=selection_seed,
    )
    panel = []
    for anchor in selected:
        anchor_id = str(anchor["anchor_id"])
        center = (float(anchor["start_seconds"]) + float(anchor["end_seconds"])) / 2.0
        windows = {}
        for chunk_seconds in seconds:
            window = _window_from_transcript(
                transcripts[str(anchor["video_id"])],
                center_seconds=center,
                duration_seconds=chunk_seconds,
            )
            window["chunk_id"] = (
                f"{anchor_id}__neural_{chunk_seconds:g}s_"
                f"{window['text_sha256'][:10]}"
            )
            windows[f"{chunk_seconds:g}"] = window
        panel.append(
            {
                "anchor_id": anchor_id,
                "reference_chunk_id": str(anchor["chunk_id"]),
                "video_id": str(anchor["video_id"]),
                "reference_start_seconds": float(anchor["start_seconds"]),
                "reference_end_seconds": float(anchor["end_seconds"]),
                "coarse_labels": list(anchor["coarse_labels"]),
                "split": "validation",
                "primary_stratum": str(anchor["primary_stratum"]),
                "reporting_cohort": cohort_by_anchor[anchor_id],
                "windows": windows,
            }
        )
    write_jsonl_atomic(panel_path, panel)
    taxonomy = load_taxonomy()
    manifest = {
        "schema_version": "1.0",
        "neural_robust_version": NEURAL_ROBUST_VERSION,
        "run_signature": signature,
        "configuration": configuration,
        "panel_path": panel_path.name,
        "panel_sha256": sha256_file(panel_path),
        "anchors": len(panel),
        "distinct_videos": len({row["video_id"] for row in panel}),
        "anchors_per_reporting_cohort": dict(
            Counter(str(row["reporting_cohort"]) for row in panel)
        ),
        "label_counts": {
            label: sum(label in row["coarse_labels"] for row in panel)
            for label in taxonomy.target_labels
        },
        "reference_match": reference_match,
        "test_used": False,
        "limitations": [
            "El panel de validation está enriquecido y no estima prevalencia natural.",
            "Las etiquetas de referencia proceden del contrato vigente y no de una nueva anotación ciega.",
            "Las ventanas están centradas en la misma ancla para aislar longitud; no reproducen todas las fronteras secuenciales de producción.",
        ],
    }
    write_json_atomic(manifest_path, manifest)
    return panel, manifest


def _encode_truth(rows: Sequence[dict[str, Any]]):
    from .training import encode_targets

    return encode_targets([{"coarse_labels": row["coarse_labels"]} for row in rows])


def _macro_damage_ap(y_true: Any, y_score: Any) -> float:
    try:
        import numpy as np
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise RuntimeError("NumPy y scikit-learn son necesarios") from exc
    taxonomy = load_taxonomy()
    values = []
    for label in taxonomy.damage_labels:
        index = taxonomy.target_labels.index(label)
        values.append(
            float(average_precision_score(y_true[:, index], y_score[:, index]))
            if y_true[:, index].any()
            else 0.0
        )
    return float(np.mean(values))


def _hard_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.metrics import (
            hamming_loss,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise RuntimeError("NumPy y scikit-learn son necesarios") from exc
    taxonomy = load_taxonomy()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    damage_indexes = [
        taxonomy.target_labels.index(label) for label in taxonomy.damage_labels
    ]
    return {
        "f1_macro_damage": float(np.mean(f1[damage_indexes])),
        "precision_macro_damage": float(np.mean(precision[damage_indexes])),
        "recall_macro_damage": float(np.mean(recall[damage_indexes])),
        "exact_label_set_match_rate": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "hamming_loss_five": float(hamming_loss(y_true, y_pred)),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(taxonomy.target_labels)
        },
    }


def _resampled_anchor_indexes(
    panel: Sequence[dict[str, Any]],
    sampled_videos: Sequence[str],
) -> list[int]:
    indexes_by_video: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(panel):
        indexes_by_video[str(row["video_id"])].append(index)
    return [
        index for video_id in sampled_videos for index in indexes_by_video[video_id]
    ]


def _percentile_interval(values: Any, confidence_level: float) -> tuple[float, float]:
    import numpy as np

    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _minilm_interpretation(
    comparisons: Sequence[dict[str, Any]],
    *,
    reference_seconds: float,
) -> dict[str, Any]:
    reference = next(
        row
        for row in comparisons
        if float(row["chunk_seconds"]) == float(reference_seconds)
    )
    challengers = [
        row
        for row in comparisons
        if float(row["chunk_seconds"]) != float(reference_seconds)
    ]
    conflicts = [
        float(row["chunk_seconds"])
        for row in challengers
        if float(row["delta_vs_reference_ci_low"]) > 0.0
    ]
    concordant = [
        float(row["chunk_seconds"])
        for row in challengers
        if float(row["delta_vs_reference_ci_high"]) < 0.0
    ]
    best = max(comparisons, key=lambda row: row["ensemble_validation_ap_macro_damage"])
    if conflicts:
        status = "conflict_with_classical_reference"
    elif len(concordant) == len(challengers):
        status = "concordant_with_classical_reference"
    else:
        status = "inconclusive_or_noninferior_alternatives"
    return {
        "status": status,
        "best_point_estimate_seconds": float(best["chunk_seconds"]),
        "reference_seconds": float(reference_seconds),
        "reference_ap_macro_damage": float(
            reference["ensemble_validation_ap_macro_damage"]
        ),
        "alternatives_significantly_above_reference": conflicts,
        "alternatives_significantly_below_reference": concordant,
        "decision_effect": "confirmatory_only_no_automatic_selection",
    }


def _bootstrap_minilm_predictions(
    panel: Sequence[dict[str, Any]],
    ensemble_scores: dict[float, Any],
    per_seed_ap: dict[float, Sequence[float]],
    *,
    reference_seconds: float,
    bootstrap_replicates: int,
    confidence_level: float,
    noninferiority_margin: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para el bootstrap") from exc
    if bootstrap_replicates < 200:
        raise ValueError("Use al menos 200 réplicas bootstrap")
    truth = _encode_truth(panel)
    videos = sorted({str(row["video_id"]) for row in panel})
    seconds = sorted(ensemble_scores)
    reference = float(reference_seconds)
    if reference not in seconds:
        raise ValueError("La referencia no está entre las duraciones de MiniLM")
    point = {
        chunk_seconds: _macro_damage_ap(truth, ensemble_scores[chunk_seconds])
        for chunk_seconds in seconds
    }
    rng = np.random.default_rng(bootstrap_seed)
    distributions = {chunk_seconds: [] for chunk_seconds in seconds}
    for _ in range(bootstrap_replicates):
        sampled_videos = rng.choice(videos, size=len(videos), replace=True).tolist()
        indexes = _resampled_anchor_indexes(panel, sampled_videos)
        for chunk_seconds in seconds:
            distributions[chunk_seconds].append(
                _macro_damage_ap(
                    truth[indexes], ensemble_scores[chunk_seconds][indexes]
                )
            )
    reference_distribution = np.asarray(distributions[reference], dtype=float)
    comparisons = []
    for chunk_seconds in seconds:
        distribution = np.asarray(distributions[chunk_seconds], dtype=float)
        delta = distribution - reference_distribution
        ap_low, ap_high = _percentile_interval(distribution, confidence_level)
        delta_low, delta_high = _percentile_interval(delta, confidence_level)
        seed_values = list(float(value) for value in per_seed_ap[chunk_seconds])
        comparisons.append(
            {
                "chunk_seconds": chunk_seconds,
                "ensemble_validation_ap_macro_damage": point[chunk_seconds],
                "cohort_ap_mean": statistics.mean(seed_values),
                "cohort_ap_standard_deviation": (
                    statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
                ),
                "cohort_ap_values": seed_values,
                "bootstrap_ap_ci_low": ap_low,
                "bootstrap_ap_ci_high": ap_high,
                "delta_vs_reference": point[chunk_seconds] - point[reference],
                "delta_vs_reference_ci_low": delta_low,
                "delta_vs_reference_ci_high": delta_high,
                "probability_noninferior": float(
                    np.mean(delta >= -noninferiority_margin)
                ),
                "probability_better_than_reference": float(np.mean(delta > 0.0)),
                "noninferior": bool(delta_low >= -noninferiority_margin),
            }
        )
    return {
        "method": "paired_video_cluster_percentile_bootstrap",
        "unit_of_resampling": "video_id_with_all_centered_anchors",
        "aggregation": "mean_scores_across_training_cohorts_before_bootstrap",
        "split": "validation",
        "replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(bootstrap_seed),
        "reference_seconds": reference,
        "noninferiority_margin": float(noninferiority_margin),
        "distinct_video_clusters": len(videos),
        "comparisons": comparisons,
        "interpretation": _minilm_interpretation(
            comparisons, reference_seconds=reference
        ),
        "test_used": False,
    }


def run_minilm_neural_robust(
    panel: Sequence[dict[str, Any]],
    panel_manifest: dict[str, Any],
    classical_robust_root: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float],
    seeds: Sequence[int],
    model_id: str,
    revision: str,
    train_limit_per_cohort: int = 1000,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "auto",
    reference_seconds: float = 30.0,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    noninferiority_margin: float = 0.01,
    bootstrap_seed: int = 20260817,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para MiniLM") from exc
    output = Path(output_root) / "minilm"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "minilm_robust_comparison.json"
    seconds = _require_five_candidates(candidate_seconds)
    started_all = time.perf_counter()
    classical_root = Path(classical_robust_root)
    dataset_paths = {
        (int(seed), chunk_seconds): classical_root
        / "repetitions"
        / f"seed-{int(seed)}"
        / f"{chunk_seconds:g}s"
        / "toy_dataset.jsonl"
        for seed in seeds
        for chunk_seconds in seconds
    }
    missing = [str(path) for path in dataset_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan cohortes del perfil robusto clásico; ejemplo: " + missing[0]
        )
    configuration = {
        "profile": "minilm_frozen_robust",
        "panel_sha256": panel_manifest["panel_sha256"],
        "candidate_seconds": list(seconds),
        "reference_seconds": float(reference_seconds),
        "seeds": [int(seed) for seed in seeds],
        "model_id": model_id,
        "revision": revision,
        "train_limit_per_cohort": int(train_limit_per_cohort),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "device": device,
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "noninferiority_margin": float(noninferiority_margin),
        "bootstrap_seed": int(bootstrap_seed),
        "dataset_sha256": {
            f"{seed}|{chunk_seconds:g}": sha256_file(path)
            for (seed, chunk_seconds), path in dataset_paths.items()
        },
    }
    run_signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if (
            cached.get("run_signature") == run_signature
            and cached.get("reporting_status") == "complete"
        ):
            return cached

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Instale el extra entrenamiento para ejecutar MiniLM robusto"
        ) from exc
    from .device import resolve_device, torch_device_name
    from .training import classification_metrics, encode_targets

    hardware = resolve_device(device)
    torch_device = torch_device_name(hardware)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        encoder = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=True
        ).to(torch_device)
    except OSError as exc:
        raise FileNotFoundError(
            f"El checkpoint fijado de {model_id} no está completo en caché"
        ) from exc

    panel_truth = _encode_truth(panel)
    panel_embeddings = {
        chunk_seconds: _frozen_hf_embeddings(
            tokenizer,
            encoder,
            [row["windows"][f"{chunk_seconds:g}"]["text"] for row in panel],
            device=torch_device,
            batch_size=batch_size,
            max_length=max_length,
        )
        for chunk_seconds in seconds
    }
    taxonomy = load_taxonomy()
    runs = []
    scores_by_seconds: dict[float, list[Any]] = defaultdict(list)
    per_seed_ap: dict[float, list[float]] = defaultdict(list)
    for seed in seeds:
        for chunk_seconds in seconds:
            duration_root = output / f"seed-{int(seed)}" / f"{chunk_seconds:g}s"
            summary_path = duration_root / "summary.json"
            predictions_path = duration_root / "predictions_validation.jsonl"
            dataset_path = dataset_paths[(int(seed), chunk_seconds)]
            run_configuration = {
                "parent_run_signature": run_signature,
                "dataset_sha256": sha256_file(dataset_path),
                "panel_sha256": panel_manifest["panel_sha256"],
                "seed": int(seed),
                "chunk_seconds": chunk_seconds,
                "model_id": model_id,
                "revision": revision,
                "train_limit": int(train_limit_per_cohort),
                "batch_size": int(batch_size),
                "max_length": int(max_length),
            }
            child_signature = sha256_text(
                json.dumps(run_configuration, ensure_ascii=False, sort_keys=True)
            )
            if summary_path.is_file() and predictions_path.is_file():
                cached = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                if cached.get("run_signature") == child_signature:
                    prediction_rows = list(read_jsonl(predictions_path))
                    if len(prediction_rows) == len(panel):
                        scores = np.asarray(
                            [
                                [
                                    row["scores"][label]
                                    for label in taxonomy.target_labels
                                ]
                                for row in prediction_rows
                            ],
                            dtype=float,
                        )
                        scores_by_seconds[chunk_seconds].append(scores)
                        per_seed_ap[chunk_seconds].append(
                            float(cached["validation_ap_macro_damage"])
                        )
                        runs.append(cached)
                        continue

            rows = list(read_jsonl(dataset_path))
            train_rows = _bounded_neural_rows(
                rows, "train", train_limit_per_cohort, seed=int(seed)
            )
            if len(train_rows) < min(100, train_limit_per_cohort):
                raise ValueError(
                    f"La cohorte {seed}/{chunk_seconds:g}s solo tiene "
                    f"{len(train_rows)} filas de train"
                )
            started = time.perf_counter()
            train_embeddings = _frozen_hf_embeddings(
                tokenizer,
                encoder,
                [str(row["text"]) for row in train_rows],
                device=torch_device,
                batch_size=batch_size,
                max_length=max_length,
            )
            classifier = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
                n_jobs=1,
            )
            classifier.fit(train_embeddings, encode_targets(train_rows))
            scores = np.asarray(
                classifier.predict_proba(panel_embeddings[chunk_seconds]),
                dtype=float,
            )
            metrics = classification_metrics(panel_truth, scores)
            prediction_rows = [
                {
                    "anchor_id": row["anchor_id"],
                    "video_id": row["video_id"],
                    "reporting_cohort": row["reporting_cohort"],
                    "chunk_seconds": chunk_seconds,
                    "true_labels": row["coarse_labels"],
                    "scores": {
                        label: float(scores[index, label_index])
                        for label_index, label in enumerate(taxonomy.target_labels)
                    },
                }
                for index, row in enumerate(panel)
            ]
            write_jsonl_atomic(predictions_path, prediction_rows)
            summary = {
                "schema_version": "1.0",
                "run_signature": child_signature,
                "seed": int(seed),
                "chunk_seconds": chunk_seconds,
                "train_rows": len(train_rows),
                "validation_anchors": len(panel),
                "validation_video_clusters": len(
                    {str(row["video_id"]) for row in panel}
                ),
                "validation_ap_macro_damage": metrics["average_precision_macro_damage"],
                "validation_ap_macro_five": metrics["average_precision_macro_five"],
                "elapsed_seconds_observed": round(time.perf_counter() - started, 3),
                "predictions_path": predictions_path.resolve()
                .relative_to(output.resolve())
                .as_posix(),
                "hardware": hardware.model_dump(),
                "selection_effect": "confirmatory_only",
            }
            write_json_atomic(summary_path, summary)
            scores_by_seconds[chunk_seconds].append(scores)
            per_seed_ap[chunk_seconds].append(
                float(metrics["average_precision_macro_damage"])
            )
            runs.append(summary)

    ensemble_scores = {
        chunk_seconds: np.mean(np.stack(values, axis=0), axis=0)
        for chunk_seconds, values in scores_by_seconds.items()
    }
    bootstrap = _bootstrap_minilm_predictions(
        panel,
        ensemble_scores,
        per_seed_ap,
        reference_seconds=reference_seconds,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=noninferiority_margin,
        bootstrap_seed=bootstrap_seed,
    )
    payload = {
        "schema_version": "1.0",
        "neural_robust_version": NEURAL_ROBUST_VERSION,
        "profile": "minilm_frozen_paired_cohort_bootstrap",
        "run_signature": run_signature,
        "configuration": configuration,
        "design": {
            "fits": len(seconds) * len(seeds),
            "training_cohorts": len(seeds),
            "panel_anchors": len(panel),
            "panel_video_clusters": len({row["video_id"] for row in panel}),
            "primary_metric": "validation_average_precision_macro_damage",
            "test_used": False,
        },
        "runs": runs,
        "bootstrap": bootstrap,
        "interpretation": bootstrap["interpretation"],
        "runtime": {
            "wall_seconds_this_invocation": round(time.perf_counter() - started_all, 3),
            "model_elapsed_seconds_recorded": round(
                sum(float(row["elapsed_seconds_observed"]) for row in runs), 3
            ),
        },
        "reporting_status": "complete",
        "limitations": [
            "MiniLM permanece congelado; solo se ajusta la cabeza logística.",
            "Las cinco cohortes de entrenamiento pueden solaparse y se reportan como repeticiones, no como estudios independientes.",
            "El panel está enriquecido y la AP no estima prevalencia productiva.",
        ],
    }
    write_json_atomic(result_path, payload)
    try:
        import torch

        del encoder
        if hardware.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return payload


def _select_minilm_noninferiority_anchors(
    references: Sequence[dict[str, Any]],
    *,
    panel_size: int,
    minimum_damage_videos_per_label: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    """Selecciona una sola ancla por video de train para el contraste 20/30."""

    if panel_size <= 0 or minimum_damage_videos_per_label <= 0:
        raise ValueError("El tamaño del panel y las cuotas deben ser positivos")
    taxonomy = load_taxonomy()
    train = [dict(row) for row in references if str(row.get("split")) == "train"]
    videos = {str(row["video_id"]) for row in train}
    if len(videos) < panel_size:
        raise ValueError(
            f"Train solo contiene {len(videos)} videos y el panel solicita {panel_size}"
        )
    label_video_counts = {
        label: len(
            {
                str(row["video_id"])
                for row in train
                if label in row["coarse_labels"]
            }
        )
        for label in taxonomy.damage_labels
    }
    insufficient = {
        label: count
        for label, count in label_video_counts.items()
        if count < minimum_damage_videos_per_label
    }
    if insufficient:
        raise ValueError(
            "No existen suficientes videos de daño para las cuotas: "
            + json.dumps(insufficient, ensure_ascii=False, sort_keys=True)
        )

    selected: list[dict[str, Any]] = []
    selected_videos: set[str] = set()
    label_counts: Counter[str] = Counter()

    def add(row: dict[str, Any], stratum: str) -> None:
        video_id = str(row["video_id"])
        if video_id in selected_videos:
            return
        item = dict(row)
        item["anchor_id"] = str(item["chunk_id"])
        item["primary_stratum"] = stratum
        selected.append(item)
        selected_videos.add(video_id)
        label_counts.update(item["coarse_labels"])

    # Las cuotas se cubren antes de llenar el panel. Los daños con menos videos
    # elegibles se procesan primero y una fila multietiqueta aporta a cada daño.
    for label in sorted(
        taxonomy.damage_labels,
        key=lambda value: (label_video_counts[value], value),
    ):
        candidates = sorted(
            (row for row in train if label in row["coarse_labels"]),
            key=lambda row: (
                -sum(
                    1.0 / label_video_counts[value]
                    for value in row["coarse_labels"]
                    if value in label_video_counts
                ),
                _stable_key(selection_seed, "quota", label, row["chunk_id"]),
            ),
        )
        for row in candidates:
            if label_counts[label] >= minimum_damage_videos_per_label:
                break
            add(row, f"QUOTA_{label}")

    missing = {
        label: minimum_damage_videos_per_label - label_counts[label]
        for label in taxonomy.damage_labels
        if label_counts[label] < minimum_damage_videos_per_label
    }
    if missing:
        raise ValueError(f"No se alcanzaron las cuotas del panel: {missing}")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        by_video[str(row["video_id"])].append(row)
    damage_videos = sorted(
        (
            (video_id, rows)
            for video_id, rows in by_video.items()
            if any(
                label in row["coarse_labels"]
                for row in rows
                for label in taxonomy.damage_labels
            )
        ),
        key=lambda item: _stable_key(selection_seed, "damage-video", item[0]),
    )
    for video_id, rows in damage_videos:
        if len(selected) >= panel_size:
            break
        if video_id in selected_videos:
            continue
        chosen = max(
            rows,
            key=lambda row: (
                sum(
                    1.0 / label_video_counts[label]
                    for label in row["coarse_labels"]
                    if label in label_video_counts
                ),
                _stable_key(selection_seed, "damage-anchor", row["chunk_id"]),
            ),
        )
        add(chosen, "DAMAGE_VIDEO")

    remaining_videos = sorted(
        (video_id for video_id in by_video if video_id not in selected_videos),
        key=lambda video_id: _stable_key(selection_seed, "fill-video", video_id),
    )
    for video_id in remaining_videos:
        if len(selected) >= panel_size:
            break
        rows = by_video[video_id]
        safe = [row for row in rows if row["coarse_labels"] == [taxonomy.safe_label]]
        candidates = safe or rows
        chosen = min(
            candidates,
            key=lambda row: _stable_key(selection_seed, "fill-anchor", row["chunk_id"]),
        )
        add(chosen, taxonomy.safe_label if safe else "FILL")

    if len(selected) != panel_size:
        raise ValueError(
            f"Solo se seleccionaron {len(selected)} de {panel_size} videos"
        )
    return sorted(
        selected,
        key=lambda row: _stable_key(selection_seed, "panel", row["anchor_id"]),
    )


def build_minilm_noninferiority_panel(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    panel_size: int = 750,
    minimum_damage_videos_per_label: int = 80,
    folds: int = 5,
    selection_seed: int = 20260831,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construye el panel train cross-fit pareado y mantiene test cerrado."""

    if folds < 2:
        raise ValueError("El cross-fit requiere al menos dos pliegues")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "paired_train_crossfit_panel.jsonl"
    manifest_path = output / "paired_train_crossfit_panel_manifest.json"
    configuration = {
        "profile": "minilm_20_30_train_video_crossfit",
        "transcript_sha256": sha256_file(transcript_path),
        "chunks_sha256": sha256_file(chunks_path),
        "dataset_sha256": sha256_file(dataset_path),
        "candidate_seconds": list(DEFAULT_MINILM_NONINFERIORITY_SECONDS),
        "panel_size": int(panel_size),
        "minimum_damage_videos_per_label": int(
            minimum_damage_videos_per_label
        ),
        "anchors_per_video": 1,
        "folds": int(folds),
        "selection_seed": int(selection_seed),
        "source_split": "train",
        "window_policy": "same_reference_midpoint_centered_window",
        "test_used": False,
    }
    signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    if panel_path.is_file() and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if (
            cached.get("run_signature") == signature
            and cached.get("panel_sha256") == sha256_file(panel_path)
        ):
            return list(read_jsonl(panel_path)), cached

    references, reference_match = build_temporal_label_references(
        chunks_path, dataset_path
    )
    train_references = [
        row for row in references if str(row.get("split")) == "train"
    ]
    train_video_ids = {str(row["video_id"]) for row in train_references}
    transcripts = _load_selected_transcripts(transcript_path, train_video_ids)
    valid_references = []
    invalid_reference_count = 0
    for row in train_references:
        transcript = transcripts.get(str(row["video_id"]))
        if transcript is None:
            invalid_reference_count += 1
            continue
        center = (
            float(row["start_seconds"]) + float(row["end_seconds"])
        ) / 2.0
        try:
            for chunk_seconds in DEFAULT_MINILM_NONINFERIORITY_SECONDS:
                _window_from_transcript(
                    transcript,
                    center_seconds=center,
                    duration_seconds=chunk_seconds,
                )
        except ValueError:
            invalid_reference_count += 1
            continue
        valid_references.append(row)
    selected = _select_minilm_noninferiority_anchors(
        valid_references,
        panel_size=panel_size,
        minimum_damage_videos_per_label=minimum_damage_videos_per_label,
        selection_seed=selection_seed,
    )
    video_ids = {str(row["video_id"]) for row in selected}
    missing = sorted(video_ids - set(transcripts))
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} transcripciones del panel; ejemplo: {missing[0]}"
        )
    fold_by_anchor = _assign_disjoint_reporting_cohorts(
        selected,
        cohort_count=folds,
        selection_seed=selection_seed,
    )
    panel = []
    for anchor in selected:
        anchor_id = str(anchor["anchor_id"])
        center = (
            float(anchor["start_seconds"]) + float(anchor["end_seconds"])
        ) / 2.0
        windows = {}
        for chunk_seconds in DEFAULT_MINILM_NONINFERIORITY_SECONDS:
            window = _window_from_transcript(
                transcripts[str(anchor["video_id"])],
                center_seconds=center,
                duration_seconds=chunk_seconds,
            )
            window["chunk_id"] = (
                f"{anchor_id}__minilm_ni_{chunk_seconds:g}s_"
                f"{window['text_sha256'][:10]}"
            )
            windows[f"{chunk_seconds:g}"] = window
        panel.append(
            {
                "anchor_id": anchor_id,
                "reference_chunk_id": str(anchor["chunk_id"]),
                "video_id": str(anchor["video_id"]),
                "reference_start_seconds": float(anchor["start_seconds"]),
                "reference_end_seconds": float(anchor["end_seconds"]),
                "coarse_labels": list(anchor["coarse_labels"]),
                "source_split": "train",
                "primary_stratum": str(anchor["primary_stratum"]),
                "crossfit_fold": int(fold_by_anchor[anchor_id]),
                "windows": windows,
            }
        )
    write_jsonl_atomic(panel_path, panel)
    taxonomy = load_taxonomy()
    manifest = {
        "schema_version": "1.0",
        "minilm_noninferiority_version": MINILM_NONINFERIORITY_VERSION,
        "run_signature": signature,
        "configuration": configuration,
        "panel_path": panel_path.name,
        "panel_sha256": sha256_file(panel_path),
        "anchors": len(panel),
        "distinct_videos": len({row["video_id"] for row in panel}),
        "anchors_per_video_max": max(
            Counter(row["video_id"] for row in panel).values()
        ),
        "anchors_per_crossfit_fold": dict(
            Counter(str(row["crossfit_fold"]) for row in panel)
        ),
        "label_counts": {
            label: sum(label in row["coarse_labels"] for row in panel)
            for label in taxonomy.target_labels
        },
        "reference_match": reference_match,
        "valid_train_references_for_both_windows": len(valid_references),
        "invalid_or_missing_train_references_excluded": invalid_reference_count,
        "test_used": False,
        "limitations": [
            "El panel está enriquecido y no estima prevalencia natural.",
            "Las etiquetas son transferencias del contrato vigente, no una nueva anotación humana ciega.",
            "La evaluación es interna por cross-fit sobre train; no reemplaza validación externa.",
        ],
    }
    write_json_atomic(manifest_path, manifest)
    return panel, manifest


def _minilm_training_pool(
    classical_robust_root: str | Path,
    *,
    chunk_seconds: float,
    classical_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    classical_root = Path(classical_robust_root)
    paths = [
        classical_root
        / "repetitions"
        / f"seed-{int(seed)}"
        / f"{chunk_seconds:g}s"
        / "toy_dataset.jsonl"
        for seed in classical_seeds
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan cohortes clásicas para el cross-fit; ejemplo: " + missing[0]
        )
    unique: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if str(row.get("split")) != "train":
                continue
            key = (
                str(row["video_id"]),
                str(row.get("text_sha256") or sha256_text(str(row["text"]))),
                tuple(str(label) for label in row["coarse_labels"]),
            )
            if key not in unique:
                item = dict(row)
                item["_ni_pool_id"] = sha256_text(
                    json.dumps(key, ensure_ascii=False, sort_keys=False)
                )
                unique[key] = item
    rows = sorted(unique.values(), key=lambda row: row["_ni_pool_id"])
    project_root = next(
        (
            candidate
            for candidate in (classical_root.resolve(), *classical_root.resolve().parents)
            if (candidate / "pyproject.toml").is_file()
        ),
        None,
    )
    return rows, {
        (
            path.resolve().relative_to(project_root).as_posix()
            if project_root is not None
            else path.name
        ): sha256_file(path)
        for path in paths
    }


def run_minilm_20_30_noninferiority_test(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    classical_robust_root: str | Path,
    output_root: str | Path,
    *,
    panel_size: int = 750,
    minimum_damage_videos_per_label: int = 80,
    folds: int = 5,
    panel_selection_seed: int = 20260831,
    classical_seeds: Sequence[int] = DEFAULT_NEURAL_SEEDS,
    repeat_seeds: Sequence[int] = DEFAULT_MINILM_NONINFERIORITY_REPEATS,
    model_id: str = DEFAULT_CHUNK_SMOKE_HF_MODEL,
    revision: str = DEFAULT_CHUNK_SMOKE_HF_REVISION,
    train_limit_per_fit: int = 4000,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "auto",
    bootstrap_replicates: int = 5000,
    confidence_level: float = 0.95,
    noninferiority_margin: float = 0.01,
    bootstrap_seed: int = 20260907,
) -> dict[str, Any]:
    """Contraste interno confirmatorio 20 s vs. 30 s con cross-fit por video."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para MiniLM") from exc
    if len(tuple(repeat_seeds)) < 2:
        raise ValueError("Use al menos dos repeticiones de entrenamiento")
    if train_limit_per_fit < 100:
        raise ValueError("train_limit_per_fit debe ser al menos 100")
    if not 0.0 < noninferiority_margin < 1.0:
        raise ValueError("El margen de no inferioridad debe estar entre 0 y 1")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "minilm_20_30_noninferiority.json"
    started_all = time.perf_counter()
    panel, panel_manifest = build_minilm_noninferiority_panel(
        transcript_path,
        chunks_path,
        dataset_path,
        output,
        panel_size=panel_size,
        minimum_damage_videos_per_label=minimum_damage_videos_per_label,
        folds=folds,
        selection_seed=panel_selection_seed,
    )
    seconds = DEFAULT_MINILM_NONINFERIORITY_SECONDS
    training_pools: dict[float, list[dict[str, Any]]] = {}
    pool_hashes: dict[str, str] = {}
    for chunk_seconds in seconds:
        rows, hashes = _minilm_training_pool(
            classical_robust_root,
            chunk_seconds=chunk_seconds,
            classical_seeds=classical_seeds,
        )
        training_pools[chunk_seconds] = rows
        pool_hashes.update(hashes)
    configuration = {
        "profile": "minilm_20_30_train_video_crossfit_noninferiority",
        "panel_sha256": panel_manifest["panel_sha256"],
        "candidate_seconds": list(seconds),
        "reference_seconds": 30.0,
        "source_split": "train_crossfit",
        "folds": int(folds),
        "repeat_seeds": [int(seed) for seed in repeat_seeds],
        "classical_seeds": [int(seed) for seed in classical_seeds],
        "model_id": model_id,
        "revision": revision,
        "train_limit_per_fit": int(train_limit_per_fit),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "device": device,
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "noninferiority_margin": float(noninferiority_margin),
        "bootstrap_seed": int(bootstrap_seed),
        "training_pool_sha256": pool_hashes,
        "test_used": False,
    }
    run_signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    if result_path.is_file():
        cached = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if (
            cached.get("run_signature") == run_signature
            and cached.get("reporting_status") == "complete"
        ):
            return cached

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Instale el extra entrenamiento para ejecutar MiniLM"
        ) from exc
    from .device import resolve_device, torch_device_name
    from .training import encode_targets

    hardware = resolve_device(device)
    torch_device = torch_device_name(hardware)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        encoder = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=True
        ).to(torch_device)
    except OSError as exc:
        raise FileNotFoundError(
            f"El checkpoint fijado de {model_id} no está completo en caché"
        ) from exc

    panel_truth = _encode_truth(panel)
    panel_embeddings = {
        chunk_seconds: _frozen_hf_embeddings(
            tokenizer,
            encoder,
            [row["windows"][f"{chunk_seconds:g}"]["text"] for row in panel],
            device=torch_device,
            batch_size=batch_size,
            max_length=max_length,
        )
        for chunk_seconds in seconds
    }
    pool_embeddings = {
        chunk_seconds: _frozen_hf_embeddings(
            tokenizer,
            encoder,
            [str(row["text"]) for row in training_pools[chunk_seconds]],
            device=torch_device,
            batch_size=batch_size,
            max_length=max_length,
        )
        for chunk_seconds in seconds
    }
    taxonomy = load_taxonomy()
    panel_index = {str(row["anchor_id"]): index for index, row in enumerate(panel)}
    scores_by_repeat = {
        int(seed): {
            chunk_seconds: np.full(
                (len(panel), len(taxonomy.target_labels)), np.nan, dtype=float
            )
            for chunk_seconds in seconds
        }
        for seed in repeat_seeds
    }
    runs = []
    for repeat_seed in repeat_seeds:
        for fold in range(1, folds + 1):
            holdout_indexes = [
                index
                for index, row in enumerate(panel)
                if int(row["crossfit_fold"]) == fold
            ]
            holdout_videos = {str(panel[index]["video_id"]) for index in holdout_indexes}
            if not holdout_indexes:
                raise ValueError(f"El pliegue {fold} quedó vacío")
            for chunk_seconds in seconds:
                duration_root = (
                    output
                    / "crossfit"
                    / f"repeat-{int(repeat_seed)}"
                    / f"fold-{fold}"
                    / f"{chunk_seconds:g}s"
                )
                summary_path = duration_root / "summary.json"
                predictions_path = duration_root / "predictions_holdout.jsonl"
                child_configuration = {
                    "parent_run_signature": run_signature,
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold),
                    "chunk_seconds": float(chunk_seconds),
                    "holdout_video_ids_sha256": sha256_text(
                        json.dumps(sorted(holdout_videos), ensure_ascii=False)
                    ),
                }
                child_signature = sha256_text(
                    json.dumps(child_configuration, ensure_ascii=False, sort_keys=True)
                )
                if summary_path.is_file() and predictions_path.is_file():
                    cached = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                    prediction_rows = list(read_jsonl(predictions_path))
                    if (
                        cached.get("run_signature") == child_signature
                        and len(prediction_rows) == len(holdout_indexes)
                    ):
                        for row in prediction_rows:
                            index = panel_index[str(row["anchor_id"])]
                            scores_by_repeat[int(repeat_seed)][chunk_seconds][index] = [
                                float(row["scores"][label])
                                for label in taxonomy.target_labels
                            ]
                        runs.append(cached)
                        continue

                pool = training_pools[chunk_seconds]
                available = [
                    row
                    for row in pool
                    if str(row["video_id"]) not in holdout_videos
                ]
                train_rows = _bounded_neural_rows(
                    available,
                    "train",
                    train_limit_per_fit,
                    seed=int(repeat_seed) + fold * 101 + int(chunk_seconds),
                )
                pool_index = {
                    str(row["_ni_pool_id"]): index for index, row in enumerate(pool)
                }
                train_indexes = [
                    pool_index[str(row["_ni_pool_id"])] for row in train_rows
                ]
                leaked = sorted(
                    holdout_videos
                    & {str(row["video_id"]) for row in train_rows}
                )
                if leaked:
                    raise AssertionError(
                        f"Fuga de video en fold {fold}: {leaked[0]}"
                    )
                started = time.perf_counter()
                classifier = OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=int(repeat_seed),
                    ),
                    n_jobs=1,
                )
                classifier.fit(
                    pool_embeddings[chunk_seconds][train_indexes],
                    encode_targets(train_rows),
                )
                scores = np.asarray(
                    classifier.predict_proba(
                        panel_embeddings[chunk_seconds][holdout_indexes]
                    ),
                    dtype=float,
                )
                scores_by_repeat[int(repeat_seed)][chunk_seconds][holdout_indexes] = scores
                prediction_rows = [
                    {
                        "anchor_id": panel[index]["anchor_id"],
                        "video_id": panel[index]["video_id"],
                        "crossfit_fold": fold,
                        "repeat_seed": int(repeat_seed),
                        "chunk_seconds": float(chunk_seconds),
                        "true_labels": panel[index]["coarse_labels"],
                        "scores": {
                            label: float(scores[position, label_index])
                            for label_index, label in enumerate(taxonomy.target_labels)
                        },
                    }
                    for position, index in enumerate(holdout_indexes)
                ]
                write_jsonl_atomic(predictions_path, prediction_rows)
                summary = {
                    "schema_version": "1.0",
                    "run_signature": child_signature,
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold),
                    "chunk_seconds": float(chunk_seconds),
                    "train_rows": len(train_rows),
                    "train_video_clusters": len(
                        {str(row["video_id"]) for row in train_rows}
                    ),
                    "holdout_anchors": len(holdout_indexes),
                    "holdout_video_clusters": len(holdout_videos),
                    "video_leakage_count": 0,
                    "elapsed_seconds_observed": round(
                        time.perf_counter() - started, 3
                    ),
                    "predictions_path": predictions_path.resolve()
                    .relative_to(output.resolve())
                    .as_posix(),
                    "hardware": hardware.model_dump(),
                    "test_used": False,
                }
                write_json_atomic(summary_path, summary)
                runs.append(summary)

    per_repeat_ap: dict[float, list[float]] = defaultdict(list)
    for repeat_seed in repeat_seeds:
        for chunk_seconds in seconds:
            scores = scores_by_repeat[int(repeat_seed)][chunk_seconds]
            if np.isnan(scores).any():
                raise AssertionError(
                    f"Predicciones OOF incompletas para {repeat_seed}/{chunk_seconds:g}s"
                )
            per_repeat_ap[chunk_seconds].append(
                _macro_damage_ap(panel_truth, scores)
            )
    ensemble_scores = {
        chunk_seconds: np.mean(
            np.stack(
                [
                    scores_by_repeat[int(seed)][chunk_seconds]
                    for seed in repeat_seeds
                ],
                axis=0,
            ),
            axis=0,
        )
        for chunk_seconds in seconds
    }
    bootstrap = _bootstrap_minilm_predictions(
        panel,
        ensemble_scores,
        per_repeat_ap,
        reference_seconds=30.0,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=noninferiority_margin,
        bootstrap_seed=bootstrap_seed,
    )
    bootstrap["split"] = "train_crossfit_out_of_fold"
    bootstrap["aggregation"] = (
        "mean_out_of_fold_scores_across_training_repeats_before_bootstrap"
    )
    challenger = next(
        row for row in bootstrap["comparisons"] if row["chunk_seconds"] == 20.0
    )
    noninferior = bool(challenger["noninferior"])
    interpretation = {
        "status": (
            "noninferiority_established"
            if noninferior
            else "noninferiority_not_established"
        ),
        "primary_contrast": "20s_minus_30s",
        "null_hypothesis": "delta_ap_le_minus_0.01",
        "alternative_hypothesis": "delta_ap_gt_minus_0.01",
        "delta_ap": float(challenger["delta_vs_reference"]),
        "delta_ap_ci_low": float(challenger["delta_vs_reference_ci_low"]),
        "delta_ap_ci_high": float(challenger["delta_vs_reference_ci_high"]),
        "noninferiority_margin": float(noninferiority_margin),
        "noninferior": noninferior,
        "superiority_established": bool(
            float(challenger["delta_vs_reference_ci_low"]) > 0.0
        ),
        "decision_effect": "complementary_only_classical_30s_remains_primary",
    }
    payload = {
        "schema_version": "1.0",
        "minilm_noninferiority_version": MINILM_NONINFERIORITY_VERSION,
        "profile": "minilm_20_30_train_video_crossfit_noninferiority",
        "run_signature": run_signature,
        "configuration": configuration,
        "design": {
            "panel_anchors": len(panel),
            "panel_video_clusters": len({row["video_id"] for row in panel}),
            "anchors_per_video": 1,
            "folds": int(folds),
            "training_repeats": len(tuple(repeat_seeds)),
            "fits": len(seconds) * folds * len(tuple(repeat_seeds)),
            "primary_metric": "out_of_fold_average_precision_macro_damage",
            "primary_contrast": "20s_minus_30s",
            "test_used": False,
        },
        "panel": panel_manifest,
        "runs": runs,
        "bootstrap": bootstrap,
        "interpretation": interpretation,
        "runtime": {
            "wall_seconds_this_invocation": round(
                time.perf_counter() - started_all, 3
            ),
            "fit_seconds_recorded": round(
                sum(float(row["elapsed_seconds_observed"]) for row in runs), 3
            ),
        },
        "reporting_status": "complete",
        "limitations": [
            "La prueba es interna y usa predicciones fuera de pliegue sobre train; no es validación externa.",
            "El panel está enriquecido y la AP no estima prevalencia productiva.",
            "Las etiquetas son transferencias temporales y no anotaciones humanas nuevas.",
            "El resultado complementa, pero no reemplaza, la selección clásica decisoria de 30 s.",
        ],
    }
    write_json_atomic(result_path, payload)
    try:
        import torch

        del encoder
        if hardware.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return payload


def _ollama_metrics_from_records(
    panel: Sequence[dict[str, Any]],
    records: dict[tuple[str, float], dict[str, Any]],
    *,
    chunk_seconds: float,
    anchor_ids: set[str] | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para evaluar Ollama") from exc
    taxonomy = load_taxonomy()
    selected = [
        row
        for row in panel
        if anchor_ids is None or str(row["anchor_id"]) in anchor_ids
    ]
    truth = _encode_truth(selected)
    predicted = np.asarray(
        [
            [
                int(
                    label
                    in records.get(
                        (str(row["anchor_id"]), float(chunk_seconds)), {}
                    ).get("predicted_labels", [])
                )
                for label in taxonomy.target_labels
            ]
            for row in selected
        ],
        dtype=np.int8,
    )
    metrics = _hard_metrics(truth, predicted)
    successful = sum(
        (str(row["anchor_id"]), float(chunk_seconds)) in records for row in selected
    )
    return {
        "requested_rows": len(selected),
        "successful_rows": successful,
        "failed_rows": len(selected) - successful,
        "valid_schema_rate": successful / max(1, len(selected)),
        **metrics,
    }


def _bootstrap_ollama_predictions(
    panel: Sequence[dict[str, Any]],
    records: dict[tuple[str, float], dict[str, Any]],
    *,
    candidate_seconds: Sequence[float],
    reference_seconds: float,
    bootstrap_replicates: int,
    confidence_level: float,
    noninferiority_margin: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para el bootstrap Ollama") from exc
    if bootstrap_replicates < 200:
        raise ValueError("Use al menos 200 réplicas bootstrap")
    taxonomy = load_taxonomy()
    seconds = _require_five_candidates(candidate_seconds)
    reference = float(reference_seconds)
    videos = sorted({str(row["video_id"]) for row in panel})
    truth = _encode_truth(panel)
    predicted = {
        chunk_seconds: np.asarray(
            [
                [
                    int(
                        label
                        in records.get((str(row["anchor_id"]), chunk_seconds), {}).get(
                            "predicted_labels", []
                        )
                    )
                    for label in taxonomy.target_labels
                ]
                for row in panel
            ],
            dtype=np.int8,
        )
        for chunk_seconds in seconds
    }
    point = {
        chunk_seconds: _hard_metrics(truth, predicted[chunk_seconds])["f1_macro_damage"]
        for chunk_seconds in seconds
    }
    rng = np.random.default_rng(bootstrap_seed)
    distributions = {chunk_seconds: [] for chunk_seconds in seconds}
    for _ in range(bootstrap_replicates):
        sampled_videos = rng.choice(videos, size=len(videos), replace=True).tolist()
        indexes = _resampled_anchor_indexes(panel, sampled_videos)
        for chunk_seconds in seconds:
            distributions[chunk_seconds].append(
                _hard_metrics(truth[indexes], predicted[chunk_seconds][indexes])[
                    "f1_macro_damage"
                ]
            )
    reference_distribution = np.asarray(distributions[reference], dtype=float)
    comparisons = []
    for chunk_seconds in seconds:
        distribution = np.asarray(distributions[chunk_seconds], dtype=float)
        delta = distribution - reference_distribution
        metric_low, metric_high = _percentile_interval(distribution, confidence_level)
        delta_low, delta_high = _percentile_interval(delta, confidence_level)
        comparisons.append(
            {
                "chunk_seconds": chunk_seconds,
                "intention_to_evaluate_f1_macro_damage": point[chunk_seconds],
                "bootstrap_f1_ci_low": metric_low,
                "bootstrap_f1_ci_high": metric_high,
                "delta_vs_reference": point[chunk_seconds] - point[reference],
                "delta_vs_reference_ci_low": delta_low,
                "delta_vs_reference_ci_high": delta_high,
                "probability_noninferior": float(
                    np.mean(delta >= -noninferiority_margin)
                ),
                "probability_better_than_reference": float(np.mean(delta > 0.0)),
                "noninferior": bool(delta_low >= -noninferiority_margin),
            }
        )
    return {
        "method": "paired_video_cluster_percentile_bootstrap",
        "unit_of_resampling": "video_id_with_all_centered_anchors",
        "invalid_output_policy": "count_as_empty_prediction_in_primary_analysis",
        "primary_metric": "intention_to_evaluate_f1_macro_damage",
        "replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(bootstrap_seed),
        "reference_seconds": reference,
        "noninferiority_margin": float(noninferiority_margin),
        "distinct_video_clusters": len(videos),
        "comparisons": comparisons,
        "test_used": False,
    }


def _ollama_interpretation(
    duration_results: Sequence[dict[str, Any]],
    bootstrap: dict[str, Any],
    *,
    reference_seconds: float,
    minimum_schema_rate: float,
) -> dict[str, Any]:
    schema_pass = all(
        float(row["valid_schema_rate"]) >= minimum_schema_rate
        for row in duration_results
    )
    comparisons = bootstrap["comparisons"]
    conflicts = [
        float(row["chunk_seconds"])
        for row in comparisons
        if float(row["chunk_seconds"]) != float(reference_seconds)
        and float(row["delta_vs_reference_ci_low"]) > 0.0
    ]
    concordant = [
        float(row["chunk_seconds"])
        for row in comparisons
        if float(row["chunk_seconds"]) != float(reference_seconds)
        and float(row["delta_vs_reference_ci_high"]) < 0.0
    ]
    best = max(
        comparisons,
        key=lambda row: row["intention_to_evaluate_f1_macro_damage"],
    )
    if not schema_pass:
        status = "operationally_inconclusive_schema_gate_failed"
    elif conflicts:
        status = "conflict_with_classical_reference"
    elif len(concordant) == len(comparisons) - 1:
        status = "concordant_with_classical_reference"
    else:
        status = "inconclusive_or_noninferior_alternatives"
    return {
        "status": status,
        "best_point_estimate_seconds": float(best["chunk_seconds"]),
        "reference_seconds": float(reference_seconds),
        "minimum_schema_rate_predeclared": float(minimum_schema_rate),
        "schema_gate_passed": schema_pass,
        "alternatives_significantly_above_reference": conflicts,
        "alternatives_significantly_below_reference": concordant,
        "decision_effect": "confirmatory_only_no_automatic_selection",
    }


def run_ollama_neural_robust(
    panel: Sequence[dict[str, Any]],
    panel_manifest: dict[str, Any],
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float],
    model: str,
    timeout_seconds: float = 90.0,
    max_wall_seconds: float = 5400.0,
    seed: int = 20260807,
    retries: int = 1,
    reference_seconds: float = 30.0,
    bootstrap_replicates: int = 2000,
    confidence_level: float = 0.95,
    noninferiority_margin: float = 0.02,
    bootstrap_seed: int = 20260829,
    minimum_schema_rate: float = 0.95,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or max_wall_seconds <= 0:
        raise ValueError("Los límites temporales de Ollama deben ser positivos")
    if not 0 < minimum_schema_rate <= 1:
        raise ValueError("minimum_schema_rate debe estar en (0, 1]")
    from .providers import OllamaProvider

    output = Path(output_root) / "ollama"
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    errors_path = output / "errors.jsonl"
    result_path = output / "ollama_robust_comparison.json"
    # Algunos kernels abiertos antes de la incorporación de semillas conservan
    # en memoria una versión anterior del proveedor. La introspección mantiene
    # compatible esa sesión; un kernel nuevo usa siempre la semilla explícita.
    import inspect

    provider_kwargs = {
        "model": model,
        "timeout": timeout_seconds,
        "retries": retries,
        "think": False,
    }
    if "seed" in inspect.signature(OllamaProvider).parameters:
        provider_kwargs["seed"] = seed
    provider = OllamaProvider(**provider_kwargs)
    probe = provider.probe()
    if not probe.get("model_available"):
        raise FileNotFoundError(f"Ollama responde, pero {model} no está descargado")
    seconds = _require_five_candidates(candidate_seconds)
    configuration = {
        "profile": "ollama_paired_panel_robust",
        "panel_sha256": panel_manifest["panel_sha256"],
        "candidate_seconds": list(seconds),
        "reference_seconds": float(reference_seconds),
        "model": model,
        "model_digest": probe.get("model_digest"),
        "operational_prompt_path": probe.get("operational_prompt_path"),
        "operational_prompt_sha256": probe.get("operational_prompt_sha256"),
        "timeout_seconds": float(timeout_seconds),
        "max_wall_seconds": float(max_wall_seconds),
        "seed": int(seed),
        "retries": int(retries),
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "noninferiority_margin": float(noninferiority_margin),
        "bootstrap_seed": int(bootstrap_seed),
        "minimum_schema_rate": float(minimum_schema_rate),
        "invalid_output_policy": "count_as_empty_prediction",
    }
    run_signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    existing = {
        str(row["comparison_id"]): row
        for row in read_jsonl(predictions_path)
        if row.get("run_signature") == run_signature and row.get("comparison_id")
    }
    error_rows = {
        str(row["comparison_id"]): row
        for row in read_jsonl(errors_path)
        if row.get("run_signature") == run_signature and row.get("comparison_id")
    }
    error_ids = set(error_rows)
    expected: list[tuple[str, float, str]] = []
    for anchor in panel:
        for chunk_seconds in seconds:
            comparison_id = (
                f"{run_signature[:16]}|{anchor['anchor_id']}|{chunk_seconds:g}"
            )
            expected.append((str(anchor["anchor_id"]), chunk_seconds, comparison_id))
    attempted_before = len(
        {comparison_id for _, _, comparison_id in expected}
        & (set(existing) | error_ids)
    )
    wall_started = time.perf_counter()
    stopped_by_wall_clock = False
    ordered_panel = sorted(
        panel,
        key=lambda row: (
            int(row["reporting_cohort"]),
            _stable_key(seed, "ollama", row["anchor_id"]),
        ),
    )
    for anchor_index, anchor in enumerate(ordered_panel):
        rotation = anchor_index % len(seconds)
        ordered_seconds = seconds[rotation:] + seconds[:rotation]
        for chunk_seconds in ordered_seconds:
            comparison_id = (
                f"{run_signature[:16]}|{anchor['anchor_id']}|{chunk_seconds:g}"
            )
            if comparison_id in existing or comparison_id in error_ids:
                continue
            remaining = max_wall_seconds - (time.perf_counter() - wall_started)
            if remaining <= 0:
                stopped_by_wall_clock = True
                break
            # Cada anotación puede usar retries+1 intentos; esta cota preserva
            # razonablemente el presupuesto global incluso ante timeouts.
            provider.timeout = min(
                timeout_seconds,
                max(1.0, remaining / max(1, retries + 1)),
            )
            window = anchor["windows"][f"{chunk_seconds:g}"]
            chunk = {
                "chunk_id": window["chunk_id"],
                "video_id": anchor["video_id"],
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
                "text": window["text"],
                "text_sha256": window["text_sha256"],
                "cohort": f"neural_robust_{anchor['reporting_cohort']}",
            }
            call_started = time.perf_counter()
            try:
                annotation = provider.annotate(chunk)
                record = {
                    "comparison_id": comparison_id,
                    "run_signature": run_signature,
                    "anchor_id": anchor["anchor_id"],
                    "video_id": anchor["video_id"],
                    "reporting_cohort": anchor["reporting_cohort"],
                    "chunk_seconds": chunk_seconds,
                    "chunk_id": window["chunk_id"],
                    "true_labels": list(anchor["coarse_labels"]),
                    "predicted_labels": list(annotation.coarse_labels),
                    "predicted_fine_labels": list(annotation.fine_labels),
                    "score_confianza": annotation.score_confianza,
                    "needs_review": annotation.needs_review,
                    "elapsed_seconds_observed": round(
                        time.perf_counter() - call_started, 3
                    ),
                    "prompt_sha256": annotation.prompt_sha256,
                }
                append_jsonl_once(predictions_path, [record], id_field="comparison_id")
                existing[comparison_id] = record
            except Exception as exc:
                error = {
                    "comparison_id": comparison_id,
                    "run_signature": run_signature,
                    "anchor_id": anchor["anchor_id"],
                    "video_id": anchor["video_id"],
                    "reporting_cohort": anchor["reporting_cohort"],
                    "chunk_seconds": chunk_seconds,
                    "chunk_id": window["chunk_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds_observed": round(
                        time.perf_counter() - call_started, 3
                    ),
                }
                append_jsonl_once(errors_path, [error], id_field="comparison_id")
                error_rows[comparison_id] = error
                error_ids.add(comparison_id)
        if stopped_by_wall_clock:
            break

    records = {
        (str(row["anchor_id"]), float(row["chunk_seconds"])): row
        for row in existing.values()
    }
    attempted_after = len(
        {comparison_id for _, _, comparison_id in expected}
        & (set(existing) | error_ids)
    )
    reporting_complete = attempted_after == len(expected)
    duration_results = []
    for chunk_seconds in seconds:
        metrics = _ollama_metrics_from_records(
            panel, records, chunk_seconds=chunk_seconds
        )
        cohort_metrics = []
        for cohort in sorted({int(row["reporting_cohort"]) for row in panel}):
            ids = {
                str(row["anchor_id"])
                for row in panel
                if int(row["reporting_cohort"]) == cohort
            }
            cohort_metrics.append(
                {
                    "reporting_cohort": cohort,
                    **_ollama_metrics_from_records(
                        panel,
                        records,
                        chunk_seconds=chunk_seconds,
                        anchor_ids=ids,
                    ),
                }
            )
        duration_results.append(
            {
                "chunk_seconds": chunk_seconds,
                **metrics,
                "cohort_metrics": cohort_metrics,
            }
        )

    fully_valid_anchor_ids = {
        str(row["anchor_id"])
        for row in panel
        if all(
            (str(row["anchor_id"]), chunk_seconds) in records
            for chunk_seconds in seconds
        )
    }
    complete_case_results = [
        {
            "chunk_seconds": chunk_seconds,
            **_ollama_metrics_from_records(
                panel,
                records,
                chunk_seconds=chunk_seconds,
                anchor_ids=fully_valid_anchor_ids,
            ),
        }
        for chunk_seconds in seconds
    ]
    bootstrap = _bootstrap_ollama_predictions(
        panel,
        records,
        candidate_seconds=seconds,
        reference_seconds=reference_seconds,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=noninferiority_margin,
        bootstrap_seed=bootstrap_seed,
    )
    interpretation = _ollama_interpretation(
        duration_results,
        bootstrap,
        reference_seconds=reference_seconds,
        minimum_schema_rate=minimum_schema_rate,
    )
    payload = {
        "schema_version": "1.0",
        "neural_robust_version": NEURAL_ROBUST_VERSION,
        "profile": "ollama_paired_panel_cluster_bootstrap",
        "run_signature": run_signature,
        "configuration": configuration,
        "probe": probe,
        "design": {
            "requested_responses": len(expected),
            "panel_anchors": len(panel),
            "panel_video_clusters": len({row["video_id"] for row in panel}),
            "reporting_cohorts": len({int(row["reporting_cohort"]) for row in panel}),
            "primary_metric": "intention_to_evaluate_f1_macro_damage",
            "invalid_output_policy": "count_as_empty_prediction",
            "test_used": False,
        },
        "duration_results": duration_results,
        "complete_case_anchor_count": len(fully_valid_anchor_ids),
        "complete_case_results": complete_case_results,
        "bootstrap": bootstrap,
        "interpretation": interpretation,
        "runtime": {
            "attempted_before_invocation": attempted_before,
            "attempted_after_invocation": attempted_after,
            "new_attempts_this_invocation": attempted_after - attempted_before,
            "wall_seconds_this_invocation": round(
                time.perf_counter() - wall_started, 3
            ),
            "successful_response_seconds_recorded": round(
                sum(
                    float(row.get("elapsed_seconds_observed", 0.0))
                    for row in existing.values()
                ),
                3,
            ),
            "failed_response_seconds_recorded": round(
                sum(
                    float(row.get("elapsed_seconds_observed", 0.0))
                    for row in error_rows.values()
                ),
                3,
            ),
            "total_response_seconds_recorded": round(
                sum(
                    float(row.get("elapsed_seconds_observed", 0.0))
                    for row in existing.values()
                )
                + sum(
                    float(row.get("elapsed_seconds_observed", 0.0))
                    for row in error_rows.values()
                ),
                3,
            ),
            "budget_seconds": float(max_wall_seconds),
            "budget_scope": "per_invocation_not_cumulative",
            "stopped_by_wall_clock": stopped_by_wall_clock,
        },
        "reporting_status": "complete" if reporting_complete else "partial",
        "limitations": [
            "Ollama produce etiquetas duras; no se promedian con la AP continua de MiniLM.",
            "La confianza autodeclarada no se usa como probabilidad calibrada por etiqueta.",
            "El análisis primario penaliza toda salida inválida; el análisis complete-case es secundario.",
            "Los cinco bloques son disjuntos por video para reporte operativo; la inferencia principal usa bootstrap agrupado por video sobre el panel completo.",
        ],
    }
    write_json_atomic(result_path, payload)
    return payload


def build_hierarchical_neural_synthesis(
    classical: dict[str, Any],
    minilm: dict[str, Any],
    ollama: dict[str, Any],
) -> dict[str, Any]:
    """Aplica la jerarquía predeclarada sin convertir métricas heterogéneas."""

    recommendation = classical.get("recommendation", {})
    classical_seconds = float(recommendation["recommended_seconds"])
    minilm_status = str(minilm.get("interpretation", {}).get("status", "missing"))
    ollama_status = str(ollama.get("interpretation", {}).get("status", "missing"))
    evidence_complete = (
        minilm.get("reporting_status") == "complete"
        and ollama.get("reporting_status") == "complete"
    )
    if not evidence_complete:
        status = "partial_neural_evidence_hold_classical"
    elif "conflict" in minilm_status or "conflict" in ollama_status:
        status = "conflict_hold_classical_pending_independent_human_validation"
    elif minilm_status.startswith("concordant") and ollama_status.startswith(
        "concordant"
    ):
        status = "cross_family_concordance_hold_classical"
    else:
        status = "inconclusive_neural_evidence_hold_classical"
    return {
        "schema_version": "1.0",
        "neural_robust_version": NEURAL_ROBUST_VERSION,
        "hierarchy_status": status,
        "final_recommended_seconds": classical_seconds,
        "decision_changed_by_neural_tests": False,
        "metric_aggregation_across_families": "none",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "families": [
            {
                "order": 1,
                "family": "classical_robust",
                "role": "selects_or_preserves_primary_length",
                "metric": "average_precision_macro_damage",
                "recommended_seconds": classical_seconds,
                "status": "decisive_primary",
            },
            {
                "order": 2,
                "family": "minilm_frozen_robust",
                "role": "checks_representation_sensitivity",
                "metric": "continuous_average_precision_macro_damage",
                "best_point_estimate_seconds": minilm.get("interpretation", {}).get(
                    "best_point_estimate_seconds"
                ),
                "status": minilm_status,
            },
            {
                "order": 3,
                "family": "ollama_semantic_robust",
                "role": "checks_semantic_sensitivity_and_operational_feasibility",
                "metric": "hard_f1_macro_damage_with_invalid_outputs_penalized",
                "best_point_estimate_seconds": ollama.get("interpretation", {}).get(
                    "best_point_estimate_seconds"
                ),
                "status": ollama_status,
            },
        ],
        "conflict_policy": (
            "Si una familia neuronal contradice la referencia clásica, se reporta "
            "el conflicto y se conserva la decisión clásica hasta disponer de una "
            "validación humana independiente."
        ),
        "independent_human_validation_required_to_override": bool("conflict" in status),
    }


def run_neural_chunk_robust_test(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    classical_robust_root: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = DEFAULT_NEURAL_CANDIDATE_SECONDS,
    reference_seconds: float = 30.0,
    seeds: Sequence[int] = DEFAULT_NEURAL_SEEDS,
    panel_size: int = 100,
    minimum_damage_anchors_per_label: int = 20,
    max_anchors_per_video: int = 2,
    reporting_cohorts: int = 5,
    panel_selection_seed: int = 20260807,
    minilm_model_id: str = DEFAULT_CHUNK_SMOKE_HF_MODEL,
    minilm_revision: str = DEFAULT_CHUNK_SMOKE_HF_REVISION,
    minilm_train_limit_per_cohort: int = 1000,
    minilm_batch_size: int = 16,
    minilm_max_length: int = 128,
    minilm_device: str = "auto",
    minilm_bootstrap_replicates: int = 2000,
    minilm_noninferiority_margin: float = 0.01,
    minilm_bootstrap_seed: int = 20260817,
    ollama_model: str = DEFAULT_CHUNK_SMOKE_OLLAMA_MODEL,
    ollama_timeout_seconds: float = 90.0,
    ollama_max_wall_seconds: float = 5400.0,
    ollama_seed: int = 20260807,
    ollama_retries: int = 1,
    ollama_bootstrap_replicates: int = 2000,
    ollama_noninferiority_margin: float = 0.02,
    ollama_bootstrap_seed: int = 20260829,
    ollama_minimum_schema_rate: float = 0.95,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Ejecuta la triangulación neuronal robusta sobre validation.

    La función exige las cinco longitudes predeclaradas. MiniLM y Ollama son
    análisis confirmatorios: nunca modifican automáticamente la recomendación
    emitida por el perfil clásico ni consultan test.
    """

    seconds = _require_five_candidates(candidate_seconds)
    if float(reference_seconds) != 30.0:
        raise ValueError("El protocolo predeclara 30 s como referencia")
    if len(tuple(seeds)) < 5:
        raise ValueError("MiniLM robusto requiere al menos cinco cohortes clásicas")
    classical_root = Path(classical_robust_root)
    classical_path = classical_root / "robust_comparison.json"
    if not classical_path.is_file():
        raise FileNotFoundError(
            "Ejecute primero el perfil robusto clásico; falta " + str(classical_path)
        )
    classical = json.loads(classical_path.read_text(encoding="utf-8-sig"))
    if classical.get("reporting_status") != "complete":
        raise ValueError("El perfil robusto clásico no está completo")
    classical_seconds = tuple(
        float(value)
        for value in classical.get("configuration", {}).get("candidate_seconds", [])
    )
    if classical_seconds != seconds:
        raise ValueError(
            "Las duraciones del perfil clásico no coinciden con las cinco "
            "duraciones del protocolo neuronal"
        )
    selected_seconds = float(classical["recommendation"]["recommended_seconds"])
    if selected_seconds != float(reference_seconds):
        raise ValueError(
            "La referencia neuronal debe coincidir con la recomendación clásica "
            f"vigente ({selected_seconds:g} s)"
        )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    panel, panel_manifest = build_paired_neural_panel(
        transcript_path,
        chunks_path,
        dataset_path,
        output,
        candidate_seconds=seconds,
        panel_size=panel_size,
        minimum_damage_anchors_per_label=minimum_damage_anchors_per_label,
        max_anchors_per_video=max_anchors_per_video,
        reporting_cohorts=reporting_cohorts,
        selection_seed=panel_selection_seed,
    )
    minilm = run_minilm_neural_robust(
        panel,
        panel_manifest,
        classical_root,
        output,
        candidate_seconds=seconds,
        seeds=seeds,
        model_id=minilm_model_id,
        revision=minilm_revision,
        train_limit_per_cohort=minilm_train_limit_per_cohort,
        batch_size=minilm_batch_size,
        max_length=minilm_max_length,
        device=minilm_device,
        reference_seconds=reference_seconds,
        bootstrap_replicates=minilm_bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=minilm_noninferiority_margin,
        bootstrap_seed=minilm_bootstrap_seed,
    )
    ollama = run_ollama_neural_robust(
        panel,
        panel_manifest,
        output,
        candidate_seconds=seconds,
        model=ollama_model,
        timeout_seconds=ollama_timeout_seconds,
        max_wall_seconds=ollama_max_wall_seconds,
        seed=ollama_seed,
        retries=ollama_retries,
        reference_seconds=reference_seconds,
        bootstrap_replicates=ollama_bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=ollama_noninferiority_margin,
        bootstrap_seed=ollama_bootstrap_seed,
        minimum_schema_rate=ollama_minimum_schema_rate,
    )
    synthesis = build_hierarchical_neural_synthesis(classical, minilm, ollama)
    write_json_atomic(output / "hierarchical_synthesis.json", synthesis)
    reporting_status = (
        "complete"
        if minilm.get("reporting_status") == "complete"
        and ollama.get("reporting_status") == "complete"
        else "partial"
    )
    compact = {
        "schema_version": "1.0",
        "neural_robust_version": NEURAL_ROBUST_VERSION,
        "profile": "hierarchical_neural_robust",
        "configuration": {
            "candidate_seconds": list(seconds),
            "reference_seconds": float(reference_seconds),
            "panel_size": int(panel_size),
            "reporting_cohorts": int(reporting_cohorts),
            "minilm_training_cohorts": len(tuple(seeds)),
            "minilm_fits": len(seconds) * len(tuple(seeds)),
            "ollama_requested_responses": len(panel) * len(seconds),
            "confidence_level": float(confidence_level),
            "test_used": False,
        },
        "source_artifacts": {
            "classical_robust_sha256": sha256_file(classical_path),
            "paired_panel_manifest_sha256": sha256_file(
                output / "paired_validation_panel_manifest.json"
            ),
            "minilm_result_sha256": sha256_file(
                output / "minilm" / "minilm_robust_comparison.json"
            ),
            "ollama_result_sha256": sha256_file(
                output / "ollama" / "ollama_robust_comparison.json"
            ),
        },
        "panel": panel_manifest,
        "minilm": {
            "design": minilm["design"],
            "bootstrap": minilm["bootstrap"],
            "interpretation": minilm["interpretation"],
            "runtime": minilm["runtime"],
            "reporting_status": minilm["reporting_status"],
        },
        "ollama": {
            "design": ollama["design"],
            "duration_results": ollama["duration_results"],
            "complete_case_anchor_count": ollama["complete_case_anchor_count"],
            "complete_case_results": ollama["complete_case_results"],
            "bootstrap": ollama["bootstrap"],
            "interpretation": ollama["interpretation"],
            "runtime": ollama["runtime"],
            "reporting_status": ollama["reporting_status"],
        },
        "hierarchical_synthesis": synthesis,
        "runtime": {
            "wall_seconds_this_invocation": round(time.perf_counter() - started, 3)
        },
        "reporting_status": reporting_status,
    }
    write_json_atomic(output / "neural_robust_comparison.json", compact)
    return {
        **compact,
        "minilm_full": minilm,
        "ollama_full": ollama,
    }
