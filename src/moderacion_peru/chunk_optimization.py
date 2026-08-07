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


CHUNK_SELECTION_VERSION = "1.2.0"
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

    chunks_file = Path(chunks_path)
    dataset_file = Path(dataset_path)
    if not chunks_file.is_file():
        raise FileNotFoundError(
            f"Faltan los chunks temporales de referencia: {chunks_file}. "
            "Restaure el input 'chunks_v2' desde el bundle sincronizado."
        )
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Falta el dataset etiquetado de referencia: {dataset_file}. "
            "Restaure el input 'dataset_5_salidas' desde el bundle sincronizado."
        )

    labels_by_key: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
    conflicts: set[tuple[str, str]] = set()
    dataset_rows = 0
    for row in read_jsonl(dataset_file):
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
    for chunk in read_jsonl(chunks_file):
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
    if not references:
        raise ValueError(
            "Los chunks y el dataset existen, pero no comparten referencias "
            f"temporales por (video_id, hash_texto): {stats}"
        )
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
        validation_prediction_paths = {
            str(candidate["experiment"]): (
                Path(candidate["candidate_path"]).parent
                / "predictions_validation.jsonl"
            )
            .resolve()
            .relative_to(duration_dir.resolve())
            .as_posix()
            for candidate in candidates
        }
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
            "validation_prediction_paths": validation_prediction_paths,
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


def _bounded_neural_rows(
    rows: Sequence[dict[str, Any]],
    split: str,
    limit: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Selecciona pocas filas de forma determinista y con cobertura de etiquetas."""

    if limit <= 0:
        raise ValueError("El límite de filas neuronales debe ser positivo")
    candidates = [row for row in rows if str(row.get("split")) == split]
    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{seed}|{split}|{row.get('chunk_id', '')}".encode()
        ).hexdigest(),
    )
    taxonomy = load_taxonomy()
    queues = {
        label: [row for row in ordered if label in row.get("coarse_labels", [])]
        for label in taxonomy.target_labels
    }
    positions = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < min(limit, len(ordered)):
        added = False
        for label in taxonomy.target_labels:
            queue = queues[label]
            while (
                positions[label] < len(queue)
                and str(queue[positions[label]].get("chunk_id")) in selected_ids
            ):
                positions[label] += 1
            if positions[label] < len(queue):
                row = queue[positions[label]]
                positions[label] += 1
                identifier = str(row.get("chunk_id"))
                selected.append(row)
                selected_ids.add(identifier)
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    for row in ordered:
        if len(selected) >= limit:
            break
        identifier = str(row.get("chunk_id"))
        if identifier not in selected_ids:
            selected.append(row)
            selected_ids.add(identifier)
    return selected


def _frozen_hf_embeddings(
    tokenizer: Any,
    model: Any,
    texts: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
):
    """Calcula mean pooling normalizado sin ajustar los pesos del encoder."""

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError("Instale PyTorch y NumPy para el smoke test de MiniLM") from exc
    batches = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.float().cpu().numpy())
    return np.concatenate(batches, axis=0)


def run_bounded_neural_chunk_comparison(
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = (20, 30),
    run_hf: bool = True,
    run_ollama: bool = True,
    hf_model_id: str = DEFAULT_CHUNK_SMOKE_HF_MODEL,
    hf_revision: str = DEFAULT_CHUNK_SMOKE_HF_REVISION,
    hf_train_limit: int = 120,
    hf_validation_limit: int = 40,
    hf_batch_size: int = 16,
    hf_max_length: int = 128,
    hf_device: str = "auto",
    ollama_model: str = DEFAULT_CHUNK_SMOKE_OLLAMA_MODEL,
    ollama_validation_limit: int = 3,
    ollama_timeout_seconds: float = 90.0,
    max_ollama_wall_seconds: float = 600.0,
    seed: int = 20260806,
) -> dict[str, Any]:
    """Compara dos perfiles neuronales acotados sin alterar la recomendación de chunks.

    MiniLM se usa como encoder congelado con una cabeza logística pequeña. Ollama
    produce etiquetas duras sobre muy pocas filas de validación y sirve únicamente
    para medir latencia, cumplimiento de esquema y una señal descriptiva. Sus
    resultados no son métricas equivalentes ni participan en la selección automática.
    """

    if not run_hf and not run_ollama:
        raise ValueError("Active al menos uno de los comparadores neuronales")
    seconds = [float(value) for value in candidate_seconds]
    if not seconds or any(value <= 0 for value in seconds):
        raise ValueError("candidate_seconds debe contener longitudes positivas")
    if hf_batch_size <= 0 or hf_max_length <= 0:
        raise ValueError("El batch y la longitud máxima de MiniLM deben ser positivos")
    if ollama_timeout_seconds <= 0 or max_ollama_wall_seconds <= 0:
        raise ValueError("Los límites temporales de Ollama deben ser positivos")

    output = Path(output_root)
    datasets: dict[float, tuple[Path, list[dict[str, Any]]]] = {}
    for chunk_seconds in seconds:
        dataset = output / f"{chunk_seconds:g}s" / "toy_dataset.jsonl"
        if not dataset.is_file():
            raise FileNotFoundError(
                f"Falta {dataset}. Ejecute primero el smoke test CPU con esta longitud; "
                "esa etapa materializa las cohortes comparables."
            )
        datasets[chunk_seconds] = (dataset, list(read_jsonl(dataset)))

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "profile": "bounded_neural_chunk_comparison",
        "candidate_seconds": seconds,
        "selection_effect": "informative_only",
        "evidence_role": "bounded_confirmatory_sensitivity_not_definitive",
        "test_used_for_selection": False,
        "comparability_warning": (
            "MiniLM produce scores continuos con una cabeza supervisada; Gemma produce "
            "etiquetas duras sobre una muestra mucho menor. No se deben ordenar juntos."
        ),
    }

    if run_hf:
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Instale el extra entrenamiento para ejecutar MiniLM congelado"
            ) from exc
        from .device import resolve_device, torch_device_name
        from .training import classification_metrics, encode_targets

        hardware = resolve_device(hf_device)
        torch_device = torch_device_name(hardware)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id,
                revision=hf_revision,
                local_files_only=True,
            )
            encoder = AutoModel.from_pretrained(
                hf_model_id,
                revision=hf_revision,
                local_files_only=True,
            ).to(torch_device)
        except OSError as exc:
            raise FileNotFoundError(
                f"El checkpoint fijado de {hf_model_id} no está completo en la caché local"
            ) from exc

        hf_comparisons = []
        for chunk_seconds in seconds:
            dataset, rows = datasets[chunk_seconds]
            train_rows = _bounded_neural_rows(rows, "train", hf_train_limit, seed=seed)
            validation_rows = _bounded_neural_rows(
                rows, "validation", hf_validation_limit, seed=seed
            )
            if not train_rows or not validation_rows:
                raise ValueError(
                    f"La duración {chunk_seconds:g} no tiene train y validation para MiniLM"
                )
            configuration = {
                "dataset_sha256": sha256_file(dataset),
                "model_id": hf_model_id,
                "revision": hf_revision,
                "train_limit": hf_train_limit,
                "validation_limit": hf_validation_limit,
                "batch_size": hf_batch_size,
                "max_length": hf_max_length,
                "seed": seed,
            }
            signature = sha256_text(
                json.dumps(configuration, ensure_ascii=False, sort_keys=True)
            )
            summary_path = (
                dataset.parent / "neural_smoke" / "hf_minilm_frozen_summary.json"
            )
            if summary_path.is_file():
                cached = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                if cached.get("run_signature") == signature:
                    hf_comparisons.append(cached)
                    continue

            started = time.perf_counter()
            train_embeddings = _frozen_hf_embeddings(
                tokenizer,
                encoder,
                [str(row["text"]) for row in train_rows],
                device=torch_device,
                batch_size=hf_batch_size,
                max_length=hf_max_length,
            )
            validation_embeddings = _frozen_hf_embeddings(
                tokenizer,
                encoder,
                [str(row["text"]) for row in validation_rows],
                device=torch_device,
                batch_size=hf_batch_size,
                max_length=hf_max_length,
            )
            classifier = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=seed,
                ),
                n_jobs=1,
            )
            classifier.fit(train_embeddings, encode_targets(train_rows))
            validation_scores = np.asarray(
                classifier.predict_proba(validation_embeddings), dtype=float
            )
            metrics = classification_metrics(
                encode_targets(validation_rows), validation_scores
            )
            summary = {
                "chunk_seconds": chunk_seconds,
                "backend": "huggingface_local_cache",
                "model": hf_model_id,
                "revision": hf_revision,
                "profile": "frozen_mean_pooling_plus_logistic",
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "validation_ap_macro_damage": metrics[
                    "average_precision_macro_damage"
                ],
                "validation_ap_macro_five": metrics[
                    "average_precision_macro_five"
                ],
                "elapsed_seconds_observed": round(
                    time.perf_counter() - started, 3
                ),
                "hardware": hardware.model_dump(),
                "run_signature": signature,
                "selection_effect": "informative_only",
            }
            write_json_atomic(summary_path, summary)
            hf_comparisons.append(summary)
        payload["huggingface"] = {
            "model": hf_model_id,
            "revision": hf_revision,
            "comparisons": hf_comparisons,
            "warning": "Encoder congelado; no equivale al fine-tuning de 03_entrenamiento.",
        }

        del encoder
        try:
            import torch

            if hardware.backend in {"cuda", "rocm"}:
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    if run_ollama:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy es necesario para evaluar el smoke test Ollama") from exc
        from .providers import OllamaProvider
        from .training import classification_metrics, encode_targets

        provider = OllamaProvider(
            model=ollama_model,
            timeout=ollama_timeout_seconds,
            retries=0,
            think=False,
        )
        probe = provider.probe()
        if not probe.get("model_available"):
            raise FileNotFoundError(
                f"Ollama responde, pero {ollama_model} no está descargado"
            )
        wall_started = time.perf_counter()
        ollama_comparisons = []
        taxonomy = load_taxonomy()
        thresholds = {label: 0.5 for label in taxonomy.target_labels}
        stopped_by_wall_clock = False
        for chunk_seconds in seconds:
            dataset, rows = datasets[chunk_seconds]
            validation_rows = _bounded_neural_rows(
                rows,
                "validation",
                ollama_validation_limit,
                seed=seed,
            )
            configuration = {
                "dataset_sha256": sha256_file(dataset),
                "model": ollama_model,
                "model_digest": probe.get("model_digest"),
                "operational_prompt_sha256": probe.get(
                    "operational_prompt_sha256"
                ),
                "validation_limit": ollama_validation_limit,
                "timeout_seconds": ollama_timeout_seconds,
                "seed": seed,
            }
            signature = sha256_text(
                json.dumps(configuration, ensure_ascii=False, sort_keys=True)
            )
            smoke_dir = dataset.parent / "neural_smoke"
            predictions_path = smoke_dir / "ollama_predictions.jsonl"
            errors_path = smoke_dir / "ollama_errors.jsonl"
            existing = {
                str(row["comparison_id"]): row
                for row in read_jsonl(predictions_path)
                if row.get("comparison_id")
            }
            expected_ids = []
            for row in validation_rows:
                comparison_id = f"{signature[:16]}|{row['chunk_id']}"
                expected_ids.append(comparison_id)
                if comparison_id in existing:
                    continue
                remaining_wall_seconds = max_ollama_wall_seconds - (
                    time.perf_counter() - wall_started
                )
                if remaining_wall_seconds <= 0:
                    stopped_by_wall_clock = True
                    break
                provider.timeout = min(
                    ollama_timeout_seconds,
                    max(1.0, remaining_wall_seconds),
                )
                call_started = time.perf_counter()
                try:
                    annotation = provider.annotate(row)
                    record = {
                        "comparison_id": comparison_id,
                        "run_signature": signature,
                        "chunk_seconds": chunk_seconds,
                        "chunk_id": row["chunk_id"],
                        "gold_labels": list(row.get("coarse_labels", [])),
                        "predicted_labels": list(annotation.coarse_labels),
                        "score_confianza": annotation.score_confianza,
                        "elapsed_seconds_observed": round(
                            time.perf_counter() - call_started, 3
                        ),
                        "annotation": annotation.model_dump(mode="json"),
                    }
                    append_jsonl_once(
                        predictions_path, [record], id_field="comparison_id"
                    )
                    existing[comparison_id] = record
                except Exception as exc:
                    append_jsonl_once(
                        errors_path,
                        [
                            {
                                "comparison_id": comparison_id,
                                "run_signature": signature,
                                "chunk_seconds": chunk_seconds,
                                "chunk_id": row.get("chunk_id"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        ],
                        id_field="comparison_id",
                    )
            evaluated = [existing[item] for item in expected_ids if item in existing]
            error_ids = {
                str(row["comparison_id"])
                for row in read_jsonl(errors_path)
                if row.get("run_signature") == signature
            }
            metrics = None
            exact_match_rate = None
            if evaluated:
                truth = encode_targets(
                    [{"coarse_labels": row["gold_labels"]} for row in evaluated]
                )
                scores = np.asarray(
                    [
                        [
                            float(label in row["predicted_labels"])
                            for label in taxonomy.target_labels
                        ]
                        for row in evaluated
                    ],
                    dtype=float,
                )
                metrics = classification_metrics(truth, scores, thresholds)
                exact_match_rate = sum(
                    set(row["gold_labels"]) == set(row["predicted_labels"])
                    for row in evaluated
                ) / len(evaluated)
            attempted = len(set(expected_ids) & (set(existing) | error_ids))
            summary = {
                "chunk_seconds": chunk_seconds,
                "backend": "ollama_http",
                "model": ollama_model,
                "model_digest": probe.get("model_digest"),
                "operational_prompt_sha256": probe.get(
                    "operational_prompt_sha256"
                ),
                "profile": "bounded_structured_hard_labels",
                "requested_validation_rows": len(validation_rows),
                "successful_rows": len(evaluated),
                "failed_rows": len((set(expected_ids) & error_ids) - set(existing)),
                "valid_schema_rate": len(evaluated) / max(1, attempted),
                "exact_label_set_match_rate": exact_match_rate,
                "validation_hard_ap_macro_damage": (
                    metrics["average_precision_macro_damage"] if metrics else None
                ),
                "validation_hard_f1_macro_damage": (
                    metrics["f1_macro_damage"] if metrics else None
                ),
                "elapsed_seconds_observed_successes": round(
                    sum(float(row["elapsed_seconds_observed"]) for row in evaluated),
                    3,
                ),
                "run_signature": signature,
                "selection_effect": "informative_only",
                "completed": len(evaluated) == len(validation_rows),
            }
            write_json_atomic(smoke_dir / "ollama_summary.json", summary)
            ollama_comparisons.append(summary)
            if stopped_by_wall_clock:
                break
        payload["ollama"] = {
            "model": ollama_model,
            "probe": probe,
            "comparisons": ollama_comparisons,
            "max_wall_seconds": max_ollama_wall_seconds,
            "stopped_by_wall_clock": stopped_by_wall_clock,
            "elapsed_wall_seconds": round(time.perf_counter() - wall_started, 3),
            "warning": (
                "Muestra acotada y etiquetas duras; informa costo y cumplimiento, "
                "pero no selecciona longitud. La salida se reanuda por chunk_id."
            ),
        }

    write_json_atomic(output / "neural_smoke_comparison.json", payload)
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


def recommend_chunk_seconds_cluster_bootstrap(
    comparisons: Sequence[dict[str, Any]],
    *,
    reference_seconds: float = 30.0,
    noninferiority_margin: float = 0.01,
) -> dict[str, Any]:
    """Elige menor costo solo si el IC pareado descarta una pérdida relevante."""

    if noninferiority_margin <= 0:
        raise ValueError("El margen de no inferioridad debe ser positivo")
    by_seconds = {float(row["chunk_seconds"]): row for row in comparisons}
    reference = float(reference_seconds)
    if reference not in by_seconds:
        raise ValueError("La longitud de referencia debe estar entre las comparadas")
    eligible = [
        row
        for row in comparisons
        if float(row["delta_vs_reference_ci_low"])
        >= -float(noninferiority_margin)
    ]
    if not eligible:
        raise ValueError("Ni la referencia quedó marcada como no inferior")
    chosen = min(
        eligible,
        key=lambda row: (
            int(row["compute_proxy"]),
            -float(row["paired_validation_ap_macro_damage"]),
            -float(row["chunk_seconds"]),
        ),
    )
    return {
        "schema_version": "1.0",
        "selection_version": CHUNK_SELECTION_VERSION,
        "profile": "paired_video_cluster_bootstrap",
        "recommended_seconds": float(chosen["chunk_seconds"]),
        "reference_seconds": reference,
        "noninferiority_margin": float(noninferiority_margin),
        "paired_validation_ap_macro_damage": float(
            chosen["paired_validation_ap_macro_damage"]
        ),
        "delta_vs_reference": float(chosen["delta_vs_reference"]),
        "delta_vs_reference_ci_low": float(
            chosen["delta_vs_reference_ci_low"]
        ),
        "delta_vs_reference_ci_high": float(
            chosen["delta_vs_reference_ci_high"]
        ),
        "compute_proxy": int(chosen["compute_proxy"]),
        "eligible_seconds": sorted(float(row["chunk_seconds"]) for row in eligible),
        "selection_rule": (
            "menor filas_train×modelos entre longitudes cuyo límite inferior "
            "bootstrap de ΔAP frente a la referencia es al menos -margen"
        ),
        "bootstrap_split": "validation",
        "bootstrap_cluster": "video_id",
        "test_used_for_selection": False,
        "warning": (
            "La inferencia sigue condicionada a etiquetas temporales proxy y a "
            "cohortes parcialmente solapadas; la activación final es manual."
        ),
    }


def _prediction_blocks(path: Path) -> dict[str, tuple[Any, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy es necesario para el bootstrap agrupado") from exc
    taxonomy = load_taxonomy()
    grouped_truth: dict[str, list[list[int]]] = defaultdict(list)
    grouped_scores: dict[str, list[list[float]]] = defaultdict(list)
    for row in read_jsonl(path):
        video_id = str(row.get("video_id", ""))
        if not video_id:
            raise ValueError(f"Predicción sin video_id: {path}")
        true_labels = set(row.get("true_labels", []))
        scores = row.get("scores", {})
        if set(scores) != set(taxonomy.target_labels):
            raise ValueError(f"Predicción sin las cinco salidas: {path}")
        grouped_truth[video_id].append(
            [int(label in true_labels) for label in taxonomy.target_labels]
        )
        grouped_scores[video_id].append(
            [float(scores[label]) for label in taxonomy.target_labels]
        )
    return {
        video_id: (
            np.asarray(grouped_truth[video_id], dtype=np.int8),
            np.asarray(grouped_scores[video_id], dtype=float),
        )
        for video_id in sorted(grouped_truth)
    }


def _macro_damage_ap_from_blocks(
    blocks: dict[str, tuple[Any, Any]],
    sampled_videos: Sequence[str],
) -> float:
    try:
        import numpy as np
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise RuntimeError("Instale NumPy y scikit-learn para el bootstrap") from exc
    truth_parts = [blocks[video_id][0] for video_id in sampled_videos]
    score_parts = [blocks[video_id][1] for video_id in sampled_videos]
    if not truth_parts:
        raise ValueError("El bootstrap no recibió videos comunes")
    truth = np.concatenate(truth_parts, axis=0)
    scores = np.concatenate(score_parts, axis=0)
    taxonomy = load_taxonomy()
    values = []
    for label in taxonomy.damage_labels:
        index = taxonomy.target_labels.index(label)
        values.append(
            float(average_precision_score(truth[:, index], scores[:, index]))
            if truth[:, index].any()
            else 0.0
        )
    return float(np.mean(values))


def _paired_video_cluster_bootstrap(
    confirmatory: dict[str, Any],
    output_root: Path,
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
        raise RuntimeError("NumPy es necesario para el bootstrap agrupado") from exc
    if bootstrap_replicates < 200:
        raise ValueError("Use al menos 200 réplicas bootstrap")
    if not 0.8 <= confidence_level < 1:
        raise ValueError("confidence_level debe estar en [0.8, 1)")

    seconds = [
        float(value) for value in confirmatory["configuration"]["candidate_seconds"]
    ]
    reference = float(reference_seconds)
    if reference not in seconds:
        raise ValueError("reference_seconds debe estar entre candidate_seconds")
    models = [str(value) for value in confirmatory["configuration"]["model_names"]]
    blocks: dict[int, dict[float, dict[str, dict[str, tuple[Any, Any]]]]] = {}
    common_videos: dict[int, list[str]] = {}
    prediction_files: list[Path] = []

    for repetition in confirmatory["repetitions"]:
        seed = int(repetition["seed"])
        blocks[seed] = {}
        video_sets: list[set[str]] = []
        repetition_root = output_root / "repetitions" / f"seed-{seed}"
        for chunk_seconds in seconds:
            comparison = next(
                row
                for row in repetition["comparisons"]
                if float(row["chunk_seconds"]) == chunk_seconds
            )
            relative_paths = comparison.get("validation_prediction_paths", {})
            if set(relative_paths) != set(models):
                raise ValueError(
                    "Faltan rutas de predicción; regenere el perfil con la versión actual"
                )
            duration_root = repetition_root / f"{chunk_seconds:g}s"
            blocks[seed][chunk_seconds] = {}
            for model in models:
                prediction_path = duration_root / str(relative_paths[model])
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                prediction_files.append(prediction_path)
                model_blocks = _prediction_blocks(prediction_path)
                blocks[seed][chunk_seconds][model] = model_blocks
                video_sets.append(set(model_blocks))
        common = sorted(set.intersection(*video_sets)) if video_sets else []
        if len(common) < 20:
            raise ValueError(
                f"La cohorte {seed} solo tiene {len(common)} videos comunes; se requieren 20"
            )
        common_videos[seed] = common

    seeds = [int(repetition["seed"]) for repetition in confirmatory["repetitions"]]
    point_estimates: dict[float, float] = {}
    for chunk_seconds in seconds:
        seed_values = []
        for seed in seeds:
            model_values = [
                _macro_damage_ap_from_blocks(
                    blocks[seed][chunk_seconds][model], common_videos[seed]
                )
                for model in models
            ]
            seed_values.append(statistics.mean(model_values))
        point_estimates[chunk_seconds] = statistics.mean(seed_values)

    rng = np.random.default_rng(bootstrap_seed)
    distributions = {chunk_seconds: [] for chunk_seconds in seconds}
    for _ in range(bootstrap_replicates):
        sampled_by_seed = {
            seed: rng.choice(
                common_videos[seed],
                size=len(common_videos[seed]),
                replace=True,
            ).tolist()
            for seed in seeds
        }
        for chunk_seconds in seconds:
            seed_values = []
            for seed in seeds:
                model_values = [
                    _macro_damage_ap_from_blocks(
                        blocks[seed][chunk_seconds][model], sampled_by_seed[seed]
                    )
                    for model in models
                ]
                seed_values.append(statistics.mean(model_values))
            distributions[chunk_seconds].append(statistics.mean(seed_values))

    alpha = (1.0 - confidence_level) / 2.0
    base_by_seconds = {
        float(row["chunk_seconds"]): row
        for row in confirmatory["aggregated_comparisons"]
    }
    reference_distribution = np.asarray(distributions[reference], dtype=float)
    comparisons = []
    for chunk_seconds in seconds:
        distribution = np.asarray(distributions[chunk_seconds], dtype=float)
        delta_distribution = distribution - reference_distribution
        base = base_by_seconds[chunk_seconds]
        comparisons.append(
            {
                "chunk_seconds": chunk_seconds,
                "paired_validation_ap_macro_damage": point_estimates[chunk_seconds],
                "bootstrap_ap_mean": float(distribution.mean()),
                "bootstrap_ap_standard_error": float(distribution.std(ddof=1)),
                "bootstrap_ap_ci_low": float(np.quantile(distribution, alpha)),
                "bootstrap_ap_ci_high": float(
                    np.quantile(distribution, 1.0 - alpha)
                ),
                "delta_vs_reference": (
                    point_estimates[chunk_seconds] - point_estimates[reference]
                ),
                "delta_vs_reference_ci_low": float(
                    np.quantile(delta_distribution, alpha)
                ),
                "delta_vs_reference_ci_high": float(
                    np.quantile(delta_distribution, 1.0 - alpha)
                ),
                "probability_noninferior": float(
                    np.mean(delta_distribution >= -noninferiority_margin)
                ),
                "probability_better_than_reference": float(
                    np.mean(delta_distribution > 0)
                ),
                "noninferior": bool(
                    np.quantile(delta_distribution, alpha)
                    >= -noninferiority_margin
                ),
                "compute_proxy": int(base["compute_proxy"]),
                "validation_wins": int(base["validation_wins"]),
                "test_ap_macro_damage_descriptive": float(
                    base["test_ap_macro_damage_descriptive"]
                ),
            }
        )

    recommendation = recommend_chunk_seconds_cluster_bootstrap(
        comparisons,
        reference_seconds=reference,
        noninferiority_margin=noninferiority_margin,
    )
    return {
        "schema_version": "1.0",
        "method": "paired_video_cluster_percentile_bootstrap",
        "unit_of_resampling": "video_id_with_all_its_chunks",
        "aggregation": "mean_models_then_mean_cohorts",
        "split": "validation",
        "replicates": bootstrap_replicates,
        "confidence_level": confidence_level,
        "bootstrap_seed": bootstrap_seed,
        "reference_seconds": reference,
        "noninferiority_margin": noninferiority_margin,
        "common_validation_videos_by_seed": {
            str(seed): len(common_videos[seed]) for seed in seeds
        },
        "prediction_files": [
            {
                "path": path.resolve().relative_to(output_root.resolve()).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(set(prediction_files))
        ],
        "comparisons": comparisons,
        "recommendation": recommendation,
        "test_used_for_selection": False,
        "limitations": [
            "Las cohortes repetidas pueden solaparse y no son folds independientes.",
            "El bootstrap preserva la dependencia intravideo, pero no corrige etiquetas proxy.",
            "Se usa la intersección de videos con predicciones en todas las longitudes y modelos.",
        ],
    }


def run_chunk_length_robust_test(
    transcript_path: str | Path,
    chunks_path: str | Path,
    dataset_path: str | Path,
    output_root: str | Path,
    *,
    candidate_seconds: Sequence[float] = (15, 20, 25, 30, 35),
    reference_seconds: float = 30.0,
    model_names: Sequence[str] = (
        "complement_nb",
        "logistic_regression",
        "sgd_incremental",
    ),
    video_limits: dict[str, int] | None = None,
    seeds: Sequence[int] = (20260805, 20260817, 20260829, 20260841, 20260853),
    max_features: int = 25000,
    minimum_overlap_fraction: float = 0.8,
    bootstrap_replicates: int = 1000,
    confidence_level: float = 0.95,
    noninferiority_margin: float = 0.01,
    bootstrap_seed: int = 20260807,
    runtime_budget_seconds: float = 1800.0,
    force: bool = False,
) -> dict[str, Any]:
    """Perfil defendible de ~30 min: cinco cohortes y bootstrap por video."""

    if len(set(seeds)) < 5:
        raise ValueError("El perfil robusto requiere al menos cinco cohortes")
    limits = video_limits or {"train": 300, "validation": 100, "test": 100}
    if runtime_budget_seconds <= 0:
        raise ValueError("runtime_budget_seconds debe ser positivo")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    confirmatory_path = output / "confirmatory_comparison.json"
    robust_path = output / "robust_comparison.json"
    requested_configuration = {
        "candidate_seconds": [float(value) for value in candidate_seconds],
        "reference_seconds": float(reference_seconds),
        "model_names": list(model_names),
        "video_limits": limits,
        "seeds": [int(seed) for seed in seeds],
        "max_features": int(max_features),
        "minimum_overlap_fraction": float(minimum_overlap_fraction),
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "noninferiority_margin": float(noninferiority_margin),
        "bootstrap_seed": int(bootstrap_seed),
        "runtime_budget_seconds": float(runtime_budget_seconds),
    }
    if robust_path.is_file() and confirmatory_path.is_file() and not force:
        cached = json.loads(robust_path.read_text(encoding="utf-8-sig"))
        cached_signature_configuration = {
            "confirmatory_sha256": sha256_file(confirmatory_path),
            "reference_seconds": float(reference_seconds),
            "bootstrap_replicates": int(bootstrap_replicates),
            "confidence_level": float(confidence_level),
            "noninferiority_margin": float(noninferiority_margin),
            "bootstrap_seed": int(bootstrap_seed),
        }
        cached_signature = sha256_text(
            json.dumps(
                cached_signature_configuration,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if (
            cached.get("selection_version") == CHUNK_SELECTION_VERSION
            and cached.get("configuration") == requested_configuration
            and cached.get("run_signature") == cached_signature
            and cached.get("reporting_status") == "complete"
        ):
            return cached

    started = time.perf_counter()
    confirmatory = run_chunk_length_confirmatory_test(
        transcript_path,
        chunks_path,
        dataset_path,
        output,
        candidate_seconds=candidate_seconds,
        model_names=model_names,
        video_limits=limits,
        seeds=seeds,
        max_features=max_features,
        minimum_overlap_fraction=minimum_overlap_fraction,
        max_validation_ap_drop=noninferiority_margin,
        force=force,
    )
    configuration = {
        "confirmatory_sha256": sha256_file(confirmatory_path),
        "reference_seconds": float(reference_seconds),
        "bootstrap_replicates": int(bootstrap_replicates),
        "confidence_level": float(confidence_level),
        "noninferiority_margin": float(noninferiority_margin),
        "bootstrap_seed": int(bootstrap_seed),
    }
    run_signature = sha256_text(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True)
    )
    if robust_path.is_file() and not force:
        cached = json.loads(robust_path.read_text(encoding="utf-8-sig"))
        if cached.get("run_signature") == run_signature:
            return cached

    bootstrap = _paired_video_cluster_bootstrap(
        confirmatory,
        output,
        reference_seconds=reference_seconds,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        noninferiority_margin=noninferiority_margin,
        bootstrap_seed=bootstrap_seed,
    )
    elapsed = time.perf_counter() - started
    observed_model_seconds = sum(
        float(row["elapsed_seconds_observed_total"])
        for row in confirmatory["aggregated_comparisons"]
    )
    payload = {
        "schema_version": "1.0",
        "selection_version": CHUNK_SELECTION_VERSION,
        "profile": "robust_30min",
        "run_signature": run_signature,
        "configuration": requested_configuration,
        "design": {
            "fits": len(candidate_seconds) * len(model_names) * len(seeds),
            "paired_cohorts": len(seeds),
            "primary_metric": "validation_average_precision_macro_damage",
            "reference_predeclared": True,
            "test_used_for_selection": False,
        },
        "confirmatory_result": str(
            confirmatory_path.resolve().relative_to(output.resolve())
        ),
        "bootstrap": bootstrap,
        "recommendation": bootstrap["recommendation"],
        "runtime": {
            "wall_seconds_this_invocation": round(elapsed, 3),
            "model_elapsed_seconds_recorded": round(observed_model_seconds, 3),
            "budget_seconds": float(runtime_budget_seconds),
            "exceeded_budget": elapsed > runtime_budget_seconds,
            "note": (
                "El presupuesto es una meta de diseño, no un corte destructivo; "
                "cada ajuste y cada predicción quedan reanudables por firma."
            ),
        },
        "reporting_status": "complete",
    }
    write_json_atomic(robust_path, payload)
    write_json_atomic(output / "robust_recommendation.json", payload["recommendation"])
    return payload
