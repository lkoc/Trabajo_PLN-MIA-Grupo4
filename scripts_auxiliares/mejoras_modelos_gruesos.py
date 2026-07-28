"""Mejoras reproducibles para el clasificador de cinco daños o SEGURO.

Este módulo nunca entrena etiquetas finas ni flags. Las únicas salidas del
clasificador son las seis categorías de ``COARSE_ORDER``. El contexto, la
aumentación y la minería se aplican únicamente al conjunto de entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from scripts_auxiliares.flujo_hibrido_moderador import read_jsonl
from scripts_auxiliares.modelos_gruesos_moderador import (
    COARSE_ORDER,
    DAMAGE_ORDER,
    coarse_metrics,
    constrained_coarse_predictions,
    target_matrix,
)


def add_neighbor_context(
    frame: pd.DataFrame,
    canonical_path: Path,
    radius: int = 1,
    neighbor_max_chars: int = 700,
    include_title: bool = True,
    output_column: str = "context_text",
) -> pd.DataFrame:
    """Añade título y chunks vecinos sin usar ninguna etiqueta como feature.

    Debe llamarse antes de dividir. Como la partición posterior agrupa por
    ``video_id``, un contexto nunca cruza entrenamiento/validación/prueba.
    """
    if radius < 0:
        raise ValueError("radius debe ser no negativo.")
    canonical_rows = read_jsonl(canonical_path)
    canonical_by_id = {row["chunk_id"]: row for row in canonical_rows}
    missing = set(frame["chunk_id"]) - set(canonical_by_id)
    if missing:
        raise ValueError(f"Faltan {len(missing)} chunks del frame en el canónico.")

    ordered_by_video: dict[str, list[dict]] = {}
    for row in canonical_rows:
        ordered_by_video.setdefault(str(row["video_id"]), []).append(row)
    for video_rows in ordered_by_video.values():
        video_rows.sort(
            key=lambda row: (
                float(row.get("start_seconds") or 0.0),
                str(row["chunk_id"]),
            )
        )
    positions = {
        row["chunk_id"]: (video_id, index)
        for video_id, rows in ordered_by_video.items()
        for index, row in enumerate(rows)
    }

    context_by_id: dict[str, str] = {}
    for chunk_id in frame["chunk_id"]:
        video_id, index = positions[chunk_id]
        video_rows = ordered_by_video[video_id]
        current = canonical_by_id[chunk_id]
        parts = []
        if include_title and current.get("video_title"):
            parts.append(f"[TITULO] {current['video_title']}")
        for offset in range(-radius, radius + 1):
            position = index + offset
            if position < 0 or position >= len(video_rows):
                continue
            neighbor = video_rows[position]
            tag = "ACTUAL" if offset == 0 else ("ANTERIOR" if offset < 0 else "SIGUIENTE")
            text = str(neighbor.get("text") or "").strip()
            if offset != 0:
                text = text[:neighbor_max_chars]
            parts.append(f"[{tag}] {text}")
        context_by_id[chunk_id] = " ".join(parts)

    output = frame.copy()
    output[output_column] = output["chunk_id"].map(context_by_id)
    if output[output_column].str.strip().eq("").any():
        raise ValueError("Se generó contexto vacío.")
    return output


def _punctuation_variant(text: str, rng: random.Random, insertion_rate: float) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text
    insertions = max(1, min(12, round(len(tokens) * insertion_rate)))
    punctuation = [".", ",", "!", "?", ";", ":"]
    positions = sorted(rng.randrange(1, len(tokens)) for _ in range(insertions))
    shift = 0
    for position in positions:
        tokens.insert(position + shift, rng.choice(punctuation))
        shift += 1
    return " ".join(tokens)


def augment_damage_with_punctuation(
    training_frame: pd.DataFrame,
    text_column: str,
    seed: int = 42,
    repetitions: int = 1,
    insertion_rate: float = 0.08,
    augmented_weight: float = 0.50,
) -> pd.DataFrame:
    """Aumentación AEDA conservadora solo para daños del entrenamiento.

    Insertar puntuación no cambia las etiquetas. Las copias heredan el mismo
    ``video_id`` y nunca se generan para validación o prueba.
    """
    if repetitions < 0 or not 0 < augmented_weight <= 1:
        raise ValueError("Parámetros de aumentación inválidos.")
    rng = random.Random(seed)
    damage_mask = training_frame["coarse_labels"].map(
        lambda values: any(category in values for category in DAMAGE_ORDER)
    )
    damage_rows = training_frame.loc[damage_mask]
    augmented = []
    for row in damage_rows.to_dict("records"):
        for repetition in range(repetitions):
            copy = dict(row)
            copy["chunk_id"] = f"{row['chunk_id']}__aeda{repetition + 1}"
            copy[text_column] = _punctuation_variant(
                str(row[text_column]), rng, insertion_rate
            )
            copy["sample_weight"] = float(row["sample_weight"]) * augmented_weight
            copy["augmentation_parent"] = row["chunk_id"]
            augmented.append(copy)
    if not augmented:
        return training_frame.copy()
    original = training_frame.copy()
    original["augmentation_parent"] = None
    return pd.concat([original, pd.DataFrame(augmented)], ignore_index=True)


def tune_thresholds_for_minimum_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    minimum_recall: float = 0.80,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Maximiza precisión sujeto a recall mínimo en cada daño."""
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall debe estar en (0, 1].")
    thresholds = np.full(y_true.shape[1], 0.50, dtype=float)
    rows = []
    grid = np.linspace(0.01, 0.99, 99)
    for column, category in enumerate(COARSE_ORDER):
        if category == "SEGURO":
            continue
        candidates = []
        for threshold in grid:
            prediction = scores[:, column] >= threshold
            recall = recall_score(y_true[:, column], prediction, zero_division=0)
            precision = precision_score(y_true[:, column], prediction, zero_division=0)
            f1 = f1_score(y_true[:, column], prediction, zero_division=0)
            candidates.append((threshold, precision, recall, f1))
        eligible = [candidate for candidate in candidates if candidate[2] >= minimum_recall]
        target_met = bool(eligible)
        if eligible:
            selected = max(eligible, key=lambda item: (item[1], item[3], item[0]))
        else:
            selected = max(candidates, key=lambda item: (item[2], item[1], item[3]))
        thresholds[column] = selected[0]
        rows.append(
            {
                "categoria": category,
                "umbral": selected[0],
                "precision_validacion": selected[1],
                "recall_validacion": selected[2],
                "f1_validacion": selected[3],
                "objetivo_recall": minimum_recall,
                "objetivo_alcanzado": target_met,
            }
        )
    return thresholds, pd.DataFrame(rows)


def evaluate_threshold_policy(
    frame: pd.DataFrame,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    predictions = constrained_coarse_predictions(scores, thresholds)
    summary, report = coarse_metrics(target_matrix(frame), predictions, scores)
    return summary, report, predictions


def routing_with_damage_and_uncertainty(
    scores: np.ndarray,
    thresholds: np.ndarray,
    margin: float,
) -> np.ndarray:
    predictions = constrained_coarse_predictions(scores, thresholds)
    predicted_damage = predictions[:, 1:].any(axis=1)
    uncertainty = np.min(np.abs(scores[:, 1:] - thresholds[1:]), axis=1) <= margin
    return predicted_damage | uncertainty


def mine_flash_hard_negatives(
    model,
    frame: pd.DataFrame,
    text_column: str,
    top_n: int = 2_000,
    max_per_video: int = 3,
) -> pd.DataFrame:
    """Prioriza seguros Flash que el modelo considera cercanos a algún daño."""
    safe_flash = frame.loc[
        frame["label_source"].eq("flash_pseudo")
        & frame["coarse_labels"].map(lambda values: values == ["SEGURO"])
    ].copy()
    scores = model.predict_scores(safe_flash[text_column].tolist())
    damage_scores = scores[:, 1:]
    best_index = damage_scores.argmax(axis=1)
    safe_flash["categoria_sospechada"] = [DAMAGE_ORDER[index] for index in best_index]
    safe_flash["score_dano_maximo"] = damage_scores.max(axis=1)
    for column, category in enumerate(DAMAGE_ORDER, start=1):
        safe_flash[f"score_{category}"] = scores[:, column]
    safe_flash = safe_flash.sort_values("score_dano_maximo", ascending=False)
    safe_flash = safe_flash.groupby("video_id", sort=False).head(max_per_video).head(top_n)
    columns = [
        "chunk_id",
        "video_id",
        "categoria_sospechada",
        "score_dano_maximo",
        *[f"score_{category}" for category in DAMAGE_ORDER],
    ]
    return safe_flash[columns].reset_index(drop=True)


def write_json_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_hard_negative_manifest(path: Path, frame: pd.DataFrame, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    write_json_manifest(path.with_suffix(".manifest.json"), metadata)
