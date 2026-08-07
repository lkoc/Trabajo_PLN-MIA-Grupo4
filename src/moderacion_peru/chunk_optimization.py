from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .experiments import train_classical_experiments
from .incremental import (
    DEFAULT_CHUNKING_CONFIGURATION,
    DEFAULT_CHUNKING_SIGNATURE,
    TranscriptSegment,
    chunk_transcript,
    chunking_signature,
    normalize_chunking_configuration,
    normalize_text,
)
from .io import (
    append_jsonl_once,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from .taxonomy import load_taxonomy


CHUNK_SELECTION_VERSION = "1.1.0"
DEFAULT_CHUNKING_CONFIG_PATH = Path("config/chunking.json")
ACTIVE_CHUNKING_STATE_PATH = Path("datos/processed/chunking_active.json")
DEFAULT_CHUNK_SMOKE_HF_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_CHUNK_SMOKE_HF_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DEFAULT_CHUNK_SMOKE_OLLAMA_MODEL = "gemma3:4b"

# Solo artefactos derivados de la longitud. Las transcripciones y candidatos nunca se mueven.
MANAGED_ARTIFACT_PATTERNS = (
    "datos/processed/chunks_v2.jsonl",
    "datos/processed/chunking_v2_versions.jsonl",
    "datos/etiquetado/**/*.json*",
    "datos/model_ready/v2/**/*.json*",
    "modelos/**/*",
    "resultados/modelos/**/*",
    "resultados/auditorias/**/*",
    "resultados/colab_bundle/*.gz",
    "resultados/colab_bundle/*.zip",
    "resultados/colab_bundle/bundle_manifest.json",
    "resultados/colab_bundle/drive_upload.json",
)


def load_chunking_configuration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return normalize_chunking_configuration()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    return normalize_chunking_configuration(payload)


def _managed_files(root: Path) -> list[Path]:
    result: set[Path] = set()
    for pattern in MANAGED_ARTIFACT_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and path.name not in {"README.md", ".gitkeep"}:
                result.add(path.resolve())
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _assert_within(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Ruta fuera del proyecto: {path}") from exc


def _state_files(state_dir: Path) -> list[Path]:
    return sorted((path for path in state_dir.rglob("*") if path.is_file()), key=str)


def _file_manifest(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _move_preserving_relative(paths: Sequence[Path], source_root: Path, target_root: Path) -> None:
    operations: list[tuple[Path, Path]] = []
    for source in paths:
        relative = source.resolve().relative_to(source_root.resolve())
        target = target_root / relative
        _assert_within(target_root, target)
        if target.exists():
            raise FileExistsError(f"El archivo de destino ya existe: {target}")
        operations.append((source, target))
    for source, target in operations:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)


def _remove_empty_parents(paths: Iterable[Path], root: Path) -> None:
    resolved_root = root.resolve()
    parents: set[Path] = set()
    for path in paths:
        for parent in path.resolve().parents:
            if parent == resolved_root:
                break
            try:
                parent.relative_to(resolved_root)
            except ValueError:
                break
            parents.add(parent)
    ordered = sorted(parents, key=lambda path: len(path.parts), reverse=True)
    for parent in ordered:
        try:
            parent.rmdir()
        except OSError:
            pass


def activate_chunking_configuration(
    project_root: str | Path,
    configuration: dict[str, Any],
    *,
    source: str = "manual",
) -> dict[str, Any]:
    """Activa una longitud sin borrar artefactos y restaura estados previos exactos.

    El primer uso reconoce los artefactos históricos como la configuración por defecto
    de 30 s. Un cambio mueve únicamente derivados a ``archivo/chunking_configurations``;
    volver a una firma anterior restaura y verifica sus bytes.
    """

    root = Path(project_root).resolve()
    desired = normalize_chunking_configuration(configuration)
    desired_signature = chunking_signature(desired)
    config_path = root / DEFAULT_CHUNKING_CONFIG_PATH
    state_path = root / ACTIVE_CHUNKING_STATE_PATH
    archive_root = root / "archivo" / "chunking_configurations"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.is_file():
        active_state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        active = normalize_chunking_configuration(active_state.get("configuration"))
        active_signature = str(active_state.get("signature") or chunking_signature(active))
    else:
        # Los derivados preexistentes pertenecen al contrato histórico de 30 s.
        active = normalize_chunking_configuration(DEFAULT_CHUNKING_CONFIGURATION)
        active_signature = DEFAULT_CHUNKING_SIGNATURE

    if active_signature == desired_signature:
        payload = {
            "schema_version": "1.0",
            "status": "already_active_noop",
            "signature": desired_signature,
            "configuration": desired,
            "source": source,
        }
        if not config_path.is_file():
            write_json_atomic(config_path, {"schema_version": "1.0", **desired, "selection_source": source})
        write_json_atomic(state_path, payload)
        return payload

    active_files = _managed_files(root)
    active_state_dir = archive_root / active_signature / "state"
    if _state_files(active_state_dir):
        raise FileExistsError(
            f"El archivo de la configuración activa no está vacío: {active_state_dir}"
        )

    # Valida por completo el estado objetivo antes de mover el estado activo.
    restore_state_dir = archive_root / desired_signature / "state"
    restore_files = _state_files(restore_state_dir)
    restore_manifest_path = archive_root / desired_signature / "manifest.json"
    expected = {}
    if restore_files and not restore_manifest_path.is_file():
        raise ValueError(f"Estado archivado sin manifiesto: {restore_state_dir}")
    if restore_manifest_path.is_file():
        restore_manifest = json.loads(restore_manifest_path.read_text(encoding="utf-8-sig"))
        expected = {item["path"]: item for item in restore_manifest.get("files", [])}
        for archived in restore_files:
            relative = archived.relative_to(restore_state_dir).as_posix()
            entry = expected.get(relative)
            if entry is None:
                raise ValueError(f"Artefacto no declarado en el manifiesto: {relative}")
            if archived.stat().st_size != int(entry["bytes"]) or sha256_file(archived) != entry["sha256"]:
                raise ValueError(f"Artefacto archivado alterado: {relative}")

    active_manifest = _file_manifest(root, active_files)
    _move_preserving_relative(active_files, root, active_state_dir)
    _remove_empty_parents(active_files, root)
    write_json_atomic(
        archive_root / active_signature / "manifest.json",
        {
            "schema_version": "1.0",
            "signature": active_signature,
            "configuration": active,
            "files": active_manifest,
        },
    )

    _move_preserving_relative(restore_files, restore_state_dir, root)
    _remove_empty_parents(restore_files, restore_state_dir)

    payload = {
        "schema_version": "1.0",
        "status": "restored" if restore_files else "activated_empty",
        "signature": desired_signature,
        "configuration": desired,
        "source": source,
        "archived_files": len(active_files),
        "restored_files": len(restore_files),
        "archive": (archive_root / active_signature).as_posix(),
    }
    write_json_atomic(config_path, {"schema_version": "1.0", **desired, "selection_source": source})
    write_json_atomic(state_path, payload)
    return payload


def _text_key(video_id: str, text: str) -> tuple[str, str]:
    return video_id, sha256_text(normalize_text(text).casefold())


def build_temporal_label_references(
    chunks_path: str | Path,
    dataset_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Une el dataset histórico con sus tiempos sin confiar en IDs secuenciales antiguos."""

    labels_by_key: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
    conflicts: set[tuple[str, str]] = set()
    dataset_rows = 0
    for row in read_jsonl(dataset_path):
        dataset_rows += 1
        key = _text_key(str(row.get("video_id", "")), str(row.get("text", "")))
        value = (tuple(row.get("coarse_labels", [])), str(row.get("split", "")))
        previous = labels_by_key.setdefault(key, value)
        if previous != value:
            conflicts.add(key)
    for key in conflicts:
        labels_by_key.pop(key, None)

    references: list[dict[str, Any]] = []
    chunk_rows = 0
    for chunk in read_jsonl(chunks_path):
        chunk_rows += 1
        key = _text_key(str(chunk.get("video_id", "")), str(chunk.get("text", "")))
        matched = labels_by_key.get(key)
        if matched is None:
            continue
        labels, split = matched
        references.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "video_id": str(chunk["video_id"]),
                "start_seconds": float(chunk["start_seconds"]),
                "end_seconds": float(chunk["end_seconds"]),
                "coarse_labels": list(labels),
                "split": split,
            }
        )
    stats = {
        "chunk_rows": chunk_rows,
        "dataset_rows": dataset_rows,
        "matched_references": len(references),
        "conflicting_keys_excluded": len(conflicts),
        "matched_videos": len({row["video_id"] for row in references}),
    }
    return references, stats


def select_toy_video_ids(
    references: Iterable[dict[str, Any]],
    limits: dict[str, int],
    *,
    seed: int = 20260805,
) -> dict[str, list[str]]:
    """Muestra determinista con round-robin de etiquetas dentro de cada split."""

    taxonomy = load_taxonomy()
    by_split_video: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in references:
        by_split_video[str(row["split"])][str(row["video_id"])].update(row["coarse_labels"])

    result: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        limit = int(limits.get(split, 0))
        videos = by_split_video.get(split, {})
        ordered = sorted(
            videos,
            key=lambda video_id: hashlib.sha256(f"{seed}|{split}|{video_id}".encode()).hexdigest(),
        )
        queues = {
            label: [video_id for video_id in ordered if label in videos[video_id]]
            for label in taxonomy.target_labels
        }
        selected: list[str] = []
        selected_set: set[str] = set()
        positions = Counter()
        while len(selected) < limit:
            added = False
            for label in taxonomy.target_labels:
                queue = queues[label]
                while positions[label] < len(queue) and queue[positions[label]] in selected_set:
                    positions[label] += 1
                if positions[label] < len(queue):
                    video_id = queue[positions[label]]
                    positions[label] += 1
                    selected.append(video_id)
                    selected_set.add(video_id)
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
        for video_id in ordered:
            if len(selected) >= limit:
                break
            if video_id not in selected_set:
                selected.append(video_id)
                selected_set.add(video_id)
        result[split] = selected
    return result


def _load_selected_transcripts(
    transcript_path: str | Path,
    video_ids: set[str],
) -> dict[str, dict[str, Any]]:
    selected = {}
    for row in read_jsonl(transcript_path):
        video_id = str(row.get("video_id", ""))
        if video_id in video_ids:
            selected[video_id] = row
            if len(selected) == len(video_ids):
                break
    return selected


def _transfer_by_overlap(
    candidate: dict[str, Any],
    references: Sequence[dict[str, Any]],
    *,
    minimum_overlap_fraction: float,
    policy: str = "dominant",
) -> dict[str, Any] | None:
    start = float(candidate["start_seconds"])
    end = float(candidate["end_seconds"])
    duration = max(end - start, 1e-9)
    ranked = []
    for reference in references:
        overlap = max(
            0.0,
            min(end, float(reference["end_seconds"]))
            - max(start, float(reference["start_seconds"])),
        )
        if overlap:
            ranked.append((overlap, -float(reference["start_seconds"]), reference))
    if not ranked:
        return None
    if policy not in {"dominant", "agreement"}:
        raise ValueError("La política de transferencia debe ser dominant o agreement")
    if policy == "agreement":
        labels = {tuple(item[2]["coarse_labels"]) for item in ranked}
        covered_fraction = min(1.0, sum(item[0] for item in ranked) / duration)
        if len(labels) != 1 or covered_fraction < minimum_overlap_fraction:
            return None
        ordered_references = sorted(
            (item[2] for item in ranked),
            key=lambda row: float(row["start_seconds"]),
        )
        reference = ordered_references[0]
        return {
            **candidate,
            "coarse_labels": list(next(iter(labels))),
            "split": reference["split"],
            "label_source": "chunk_length_confirmatory_agreement_proxy",
            "sample_weight": 1.0,
            "training_eligible": True,
            "needs_review": False,
            "decision_status": "resolved",
            "reference_chunk_id": reference["chunk_id"],
            "reference_chunk_ids": [row["chunk_id"] for row in ordered_references],
            "overlap_fraction": round(covered_fraction, 6),
            "transfer_policy": policy,
        }
    overlap, _, reference = max(ranked, key=lambda item: (item[0], item[1]))
    if overlap / duration < minimum_overlap_fraction:
        return None
    return {
        **candidate,
        "coarse_labels": list(reference["coarse_labels"]),
        "split": reference["split"],
        "label_source": "chunk_length_smoke_temporal_proxy",
        "sample_weight": 1.0,
        "training_eligible": True,
        "needs_review": False,
        "decision_status": "resolved",
        "reference_chunk_id": reference["chunk_id"],
        "overlap_fraction": round(overlap / duration, 6),
        "transfer_policy": policy,
    }


def _mean_metric(candidates: Sequence[dict[str, Any]], split: str, metric: str) -> float:
    values = [float(candidate[f"{split}_metrics"][metric]) for candidate in candidates]
    return sum(values) / len(values)


def _metric_by_model(
    candidates: Sequence[dict[str, Any]],
    split: str,
    metric: str,
) -> dict[str, float]:
    return {
        str(candidate["experiment"]): float(candidate[f"{split}_metrics"][metric])
        for candidate in candidates
    }


def recommend_chunk_seconds(
    comparisons: Sequence[dict[str, Any]],
    *,
    max_validation_ap_drop: float = 0.02,
) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("No hay resultados para recomendar una longitud")
    best_validation = max(float(row["validation_ap_macro_damage"]) for row in comparisons)
    eligible = [
        row
        for row in comparisons
        if best_validation - float(row["validation_ap_macro_damage"]) <= max_validation_ap_drop
    ]
    chosen = min(
        eligible,
        key=lambda row: (
            int(row["compute_proxy"]),
            -float(row["validation_ap_macro_damage"]),
            -float(row["chunk_seconds"]),
        ),
    )
    return {
        "schema_version": "1.0",
        "selection_version": CHUNK_SELECTION_VERSION,
        "recommended_seconds": float(chosen["chunk_seconds"]),
        "validation_ap_macro_damage": float(chosen["validation_ap_macro_damage"]),
        "best_validation_ap_macro_damage": best_validation,
        "validation_ap_drop": best_validation - float(chosen["validation_ap_macro_damage"]),
        "max_validation_ap_drop": max_validation_ap_drop,
        "compute_proxy": int(chosen["compute_proxy"]),
        "selection_rule": "menor train_rows×modelos dentro de la tolerancia AP de validación",
        "test_used_for_selection": False,
        "warning": "Piloto local con muestra pequeña y etiquetas proxy; requiere confirmación manual.",
    }


def run_chunk_length_smoke_test(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = (15, 20, 25, 30, 35),
    model_names: Sequence[str] = ("complement_nb", "sgd_incremental"),
    video_limits: dict[str, int] | None = None,
    max_features: int = 12000,
    minimum_overlap_fraction: float = 0.5,
    transfer_policy: str = "dominant",
    max_validation_ap_drop: float = 0.02,
    selection_seed: int = 20260805,
    force: bool = False,
) -> dict[str, Any]:
    """Ejecuta el piloto CPU y recomienda por validación, nunca por test."""

    seconds = [float(value) for value in candidate_seconds]
    if not seconds or any(value <= 0 for value in seconds):
        raise ValueError("candidate_seconds debe contener longitudes positivas")
    if not 0 < minimum_overlap_fraction <= 1:
        raise ValueError("minimum_overlap_fraction debe estar en (0, 1]")
    limits = video_limits or {"train": 40, "validation": 16, "test": 16}
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    references, match_stats = build_temporal_label_references(chunks_path, dataset_path)
    selection = select_toy_video_ids(references, limits, seed=selection_seed)
    selected_ids = {video_id for values in selection.values() for video_id in values}
    transcripts = _load_selected_transcripts(transcript_path, selected_ids)
    missing = sorted(selected_ids - set(transcripts))
    if missing:
        raise ValueError(f"Faltan {len(missing)} transcripciones seleccionadas; ejemplo: {missing[0]}")
    references_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reference in references:
        if reference["video_id"] in selected_ids:
            references_by_video[reference["video_id"]].append(reference)

    comparisons: list[dict[str, Any]] = []
    for chunk_seconds in seconds:
        rows: list[dict[str, Any]] = []
        generated = 0
        started = time.perf_counter()
        for video_id in sorted(selected_ids):
            transcript = transcripts[video_id]
            segments = [
                TranscriptSegment(
                    float(segment.get("start", 0)),
                    float(segment.get("duration", 0)),
                    str(segment.get("text", "")),
                )
                for segment in transcript.get("segments", [])
            ]
            candidate_chunks = chunk_transcript(video_id, segments, max_seconds=chunk_seconds)
            generated += len(candidate_chunks)
            for chunk in candidate_chunks:
                transferred = _transfer_by_overlap(
                    chunk.to_dict(),
                    references_by_video[video_id],
                    minimum_overlap_fraction=minimum_overlap_fraction,
                    policy=transfer_policy,
                )
                if transferred is not None:
                    rows.append(transferred)
        split_counts = Counter(str(row["split"]) for row in rows)
        label_counts = {
            split: dict(
                Counter(
                    label
                    for row in rows
                    if str(row["split"]) == split
                    for label in row["coarse_labels"]
                )
            )
            for split in ("train", "validation", "test")
        }
        if any(split_counts[split] == 0 for split in ("train", "validation", "test")):
            raise ValueError(f"La duración {chunk_seconds:g} no produjo los tres splits: {split_counts}")
        duration_dir = output / f"{chunk_seconds:g}s"
        dataset = duration_dir / "toy_dataset.jsonl"
        write_jsonl_atomic(dataset, sorted(rows, key=lambda row: (row["split"], row["video_id"], row["chunk_id"])))
        result = train_classical_experiments(
            dataset,
            duration_dir / "models",
            force=force,
            model_names=model_names,
            max_features=max_features,
        )
        candidates = result["candidates"]
        elapsed = time.perf_counter() - started
        comparison = {
            "chunk_seconds": chunk_seconds,
            "generated_chunks": generated,
            "labeled_rows": len(rows),
            "train_rows": split_counts["train"],
            "validation_rows": split_counts["validation"],
            "test_rows": split_counts["test"],
            "label_counts": label_counts,
            "models": list(model_names),
            "validation_ap_macro_damage": _mean_metric(candidates, "validation", "average_precision_macro_damage"),
            "test_ap_macro_damage_descriptive": _mean_metric(candidates, "test", "average_precision_macro_damage"),
            "validation_ap_by_model": _metric_by_model(candidates, "validation", "average_precision_macro_damage"),
            "test_ap_by_model_descriptive": _metric_by_model(candidates, "test", "average_precision_macro_damage"),
            "compute_proxy": split_counts["train"] * len(model_names),
            "elapsed_seconds_observed": round(elapsed, 3),
            "run_signature": result["run_signature"],
        }
        comparisons.append(comparison)
        write_json_atomic(duration_dir / "summary.json", comparison)

    recommendation = recommend_chunk_seconds(
        comparisons,
        max_validation_ap_drop=max_validation_ap_drop,
    )
    payload = {
        "schema_version": "1.0",
        "selection_version": CHUNK_SELECTION_VERSION,
        "reference_match": match_stats,
        "selected_videos": selection,
        "selected_video_counts": {key: len(value) for key, value in selection.items()},
        "configuration": {
            "candidate_seconds": seconds,
            "model_names": list(model_names),
            "max_features": max_features,
            "minimum_overlap_fraction": minimum_overlap_fraction,
            "transfer_policy": transfer_policy,
            "max_validation_ap_drop": max_validation_ap_drop,
            "selection_seed": selection_seed,
        },
        "comparisons": comparisons,
        "recommendation": recommendation,
    }
    write_json_atomic(output / "comparison.json", payload)
    write_json_atomic(output / "recommendation.json", recommendation)
    return payload


def run_chunk_length_confirmatory_test(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = (15, 20, 25, 30, 35),
    model_names: Sequence[str] = ("complement_nb", "logistic_regression", "sgd_incremental"),
    video_limits: dict[str, int] | None = None,
    seeds: Sequence[int] = (20260805, 20260817, 20260829),
    max_features: int = 20000,
    minimum_overlap_fraction: float = 0.8,
    max_validation_ap_drop: float = 0.01,
    force: bool = False,
) -> dict[str, Any]:
    """Confirmación pareada corta con entrenamiento e inferencia por longitud.

    Cada repetición selecciona otra cohorte dentro de los splits congelados. Dentro
    de una repetición, todas las longitudes usan exactamente los mismos videos. Las
    etiquetas proxy se admiten solo si todos los chunks históricos solapados
    concuerdan. Se agrega por validación; test sigue siendo descriptivo.
    """

    if len(set(seeds)) < 2:
        raise ValueError("La confirmación requiere al menos dos semillas distintas")
    limits = video_limits or {"train": 100, "validation": 40, "test": 40}
    output = Path(output_root)
    repetitions = []
    for seed in seeds:
        repetitions.append(
            run_chunk_length_smoke_test(
                transcript_path,
                chunks_path,
                dataset_path,
                output / "repetitions" / f"seed-{seed}",
                candidate_seconds=candidate_seconds,
                model_names=model_names,
                video_limits=limits,
                max_features=max_features,
                minimum_overlap_fraction=minimum_overlap_fraction,
                transfer_policy="agreement",
                max_validation_ap_drop=max_validation_ap_drop,
                selection_seed=int(seed),
                force=force,
            )
        )

    winners = Counter(
        max(
            repetition["comparisons"],
            key=lambda row: float(row["validation_ap_macro_damage"]),
        )["chunk_seconds"]
        for repetition in repetitions
    )
    aggregated: list[dict[str, Any]] = []
    for chunk_seconds in map(float, candidate_seconds):
        rows = [
            next(
                row
                for row in repetition["comparisons"]
                if float(row["chunk_seconds"]) == chunk_seconds
            )
            for repetition in repetitions
        ]
        validation_values = [float(row["validation_ap_macro_damage"]) for row in rows]
        test_values = [float(row["test_ap_macro_damage_descriptive"]) for row in rows]
        compute_values = [int(row["compute_proxy"]) for row in rows]
        elapsed_values = [float(row["elapsed_seconds_observed"]) for row in rows]
        validation_by_model = {
            model: {
                "mean": statistics.mean(
                    float(row["validation_ap_by_model"][model]) for row in rows
                ),
                "standard_deviation": statistics.stdev(
                    float(row["validation_ap_by_model"][model]) for row in rows
                ),
            }
            for model in model_names
        }
        test_by_model = {
            model: {
                "mean": statistics.mean(
                    float(row["test_ap_by_model_descriptive"][model]) for row in rows
                ),
                "standard_deviation": statistics.stdev(
                    float(row["test_ap_by_model_descriptive"][model]) for row in rows
                ),
            }
            for model in model_names
        }
        aggregated.append(
            {
                "chunk_seconds": chunk_seconds,
                "validation_ap_macro_damage": statistics.mean(validation_values),
                "validation_ap_standard_deviation": statistics.stdev(validation_values),
                "test_ap_macro_damage_descriptive": statistics.mean(test_values),
                "test_ap_standard_deviation_descriptive": statistics.stdev(test_values),
                "compute_proxy": round(statistics.mean(compute_values)),
                "compute_proxy_standard_deviation": statistics.stdev(compute_values),
                "elapsed_seconds_observed_total": round(sum(elapsed_values), 3),
                "validation_wins": int(winners[chunk_seconds]),
                "repetitions": len(repetitions),
                "validation_ap_by_model": validation_by_model,
                "test_ap_by_model_descriptive": test_by_model,
            }
        )
    recommendation = recommend_chunk_seconds(
        aggregated,
        max_validation_ap_drop=max_validation_ap_drop,
    )
    recommendation.update(
        {
            "profile": "confirmatory_short",
            "paired_repetitions": len(repetitions),
            "seeds": [int(seed) for seed in seeds],
            "transfer_policy": "agreement",
            "warning": (
                "Confirmación corta y más estable, pero aún usa etiquetas temporales proxy; "
                "la activación final sigue siendo manual."
            ),
        }
    )
    cohort_overlap = {}
    for split in ("train", "validation", "test"):
        sets = [set(repetition["selected_videos"][split]) for repetition in repetitions]
        pairwise = []
        for left_index in range(len(sets)):
            for right_index in range(left_index + 1, len(sets)):
                intersection = sets[left_index] & sets[right_index]
                union = sets[left_index] | sets[right_index]
                pairwise.append(
                    {
                        "left_seed": int(seeds[left_index]),
                        "right_seed": int(seeds[right_index]),
                        "shared_videos": len(intersection),
                        "jaccard": len(intersection) / len(union) if union else 0.0,
                    }
                )
        cohort_overlap[split] = {
            "unique_videos_across_repetitions": len(set().union(*sets)),
            "videos_in_all_repetitions": len(set.intersection(*sets)),
            "pairwise": pairwise,
        }
    payload = {
        "schema_version": "1.0",
        "selection_version": CHUNK_SELECTION_VERSION,
        "profile": "confirmatory_short",
        "configuration": {
            "candidate_seconds": [float(value) for value in candidate_seconds],
            "model_names": list(model_names),
            "video_limits": limits,
            "seeds": [int(seed) for seed in seeds],
            "max_features": max_features,
            "minimum_overlap_fraction": minimum_overlap_fraction,
            "transfer_policy": "agreement",
            "max_validation_ap_drop": max_validation_ap_drop,
        },
        "aggregated_comparisons": aggregated,
        "cohort_overlap": cohort_overlap,
        "recommendation": recommendation,
        "repetitions": [
            {
                "seed": repetition["configuration"]["selection_seed"],
                "selected_video_counts": repetition["selected_video_counts"],
                "selected_videos": repetition["selected_videos"],
                "comparisons": repetition["comparisons"],
            }
            for repetition in repetitions
        ],
    }
    write_json_atomic(output / "confirmatory_comparison.json", payload)
    write_json_atomic(output / "confirmatory_recommendation.json", recommendation)
    return payload
