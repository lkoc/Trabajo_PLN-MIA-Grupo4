from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .device import high_memory_bf16_cuda, resolve_device, torch_device_name
from .experiments import ProgressCallback, _notify_progress
from .io import (
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from .prompt_sft import (
    PROMPT_SFT_MODEL_ID,
    PROMPT_SFT_MODEL_REVISION,
    _configure_lora_gradient_checkpointing,
    _cuda_memory_preflight,
)
from .taxonomy import load_taxonomy

TOY_EXPERIMENT_ID = "03_06b_toy_qwen_markdown_json"
TOY_SCHEMA_VERSION = "1.0.0"
TOY_SEED = 20260815
TOY_SPLIT_WEIGHTS = {"train": 80, "validation": 20, "test": 20}
TOY_LABEL_TOTALS = {
    "SEGURO": 960,
    "RACISMO_DISCRIMINACION": 60,
    "ATAQUE_POR_GENERO_IDENTIDAD": 60,
    "ACOSO_AMENAZA": 60,
    "CONTENIDO_SEXUAL": 60,
}
TOY_TOTAL_ROWS = sum(TOY_LABEL_TOTALS.values())


def toy_expected_distribution() -> dict[str, dict[str, int]]:
    """Devuelve 800/200/200 interpretando 80:20:20 como pesos 4:1:1."""

    weight_total = sum(TOY_SPLIT_WEIGHTS.values())
    result: dict[str, dict[str, int]] = {split: {} for split in TOY_SPLIT_WEIGHTS}
    for label, total in TOY_LABEL_TOTALS.items():
        for split, weight in TOY_SPLIT_WEIGHTS.items():
            numerator = total * weight
            if numerator % weight_total:
                raise ValueError(
                    f"No existe una asignación entera para {label}/{split}"
                )
            result[split][label] = numerator // weight_total
    return result


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def _selected_row(row: dict[str, Any], *, seed: int) -> dict[str, Any]:
    fields = (
        "schema_version",
        "taxonomy_version",
        "chunk_id",
        "video_id",
        "channel_id",
        "channel_title",
        "text",
        "coarse_labels",
        "fine_labels",
        "flags_reference_only",
        "coarse_observed_mask",
        "fine_observed_mask",
        "flags_observed_mask",
        "label_source",
        "split",
        "needs_review",
        "training_eligible",
    )
    selected = {key: row[key] for key in fields if key in row}
    selected.update(
        {
            "toy_experiment_id": TOY_EXPERIMENT_ID,
            "toy_source_split": row["split"],
            "toy_stratum": row["coarse_labels"][0],
            "toy_selection_seed": seed,
        }
    )
    return selected


def _distribution(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in TOY_SPLIT_WEIGHTS}
    for row in rows:
        counts[str(row["split"])][str(row["coarse_labels"][0])] += 1
    return {split: dict(counts[split]) for split in TOY_SPLIT_WEIGHTS}


def validate_toy_dataset(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    expected = toy_expected_distribution()
    if len(rows) != TOY_TOTAL_ROWS:
        raise ValueError(
            f"El toy dataset debe contener {TOY_TOTAL_ROWS} filas, no {len(rows)}"
        )
    videos: list[str] = []
    chunks: list[str] = []
    for row in rows:
        split = row.get("split")
        labels = row.get("coarse_labels")
        if split not in TOY_SPLIT_WEIGHTS:
            raise ValueError(f"Split toy desconocido: {split}")
        if not isinstance(labels, list) or len(labels) != 1:
            raise ValueError("Cada fila toy debe ser pura y tener una sola categoría")
        if labels[0] not in taxonomy.target_labels:
            raise ValueError(f"Categoría toy desconocida: {labels[0]}")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Texto vacío en {row.get('chunk_id')}")
        videos.append(str(row["video_id"]))
        chunks.append(str(row["chunk_id"]))
    if len(set(videos)) != len(videos):
        raise ValueError(
            "El toy dataset debe seleccionar como máximo un chunk por video"
        )
    if len(set(chunks)) != len(chunks):
        raise ValueError("El toy dataset contiene chunk_id duplicados")
    observed = _distribution(rows)
    if observed != expected:
        raise ValueError(
            f"Distribución toy incorrecta: {observed}; se esperaba {expected}"
        )
    split_videos = {
        split: {str(row["video_id"]) for row in rows if row["split"] == split}
        for split in TOY_SPLIT_WEIGHTS
    }
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(split_videos[left] & split_videos[right] for left, right in split_pairs):
        raise ValueError("Existe fuga de video entre train, validation y test")
    return {
        "rows": len(rows),
        "unique_videos": len(set(videos)),
        "distribution": observed,
        "split_rows": {split: sum(observed[split].values()) for split in observed},
        "pure_single_label_rows": True,
        "video_disjoint": True,
    }


def build_toy_prompt_dataset(
    source_dataset_path: str | Path,
    destination_path: str | Path,
    *,
    seed: int = TOY_SEED,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Muestrea 1.200 filas puras, aleatorias y reproducibles dentro de cada estrato.

    Se conserva el split original por video. Dentro de cada combinación
    categoría/split se asigna una prioridad pseudoaleatoria estable y se toma
    un solo chunk por video; así no se introduce fuga entre particiones.
    """

    source = Path(source_dataset_path).resolve()
    destination = Path(destination_path).resolve()
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256_file(source)
    configuration = {
        "schema_version": TOY_SCHEMA_VERSION,
        "experiment_id": TOY_EXPERIMENT_ID,
        "seed": seed,
        "split_weights": TOY_SPLIT_WEIGHTS,
        "label_totals": TOY_LABEL_TOTALS,
        "source_sha256": source_sha,
        "selection": "stable_random_priority_within_label_split_one_chunk_per_video",
    }
    signature = sha256_text(json.dumps(configuration, sort_keys=True))
    if destination.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") == signature:
            audit = validate_toy_dataset(list(read_jsonl(destination)))
            return {
                "status": "noop",
                "dataset_path": str(destination),
                "manifest_path": str(manifest_path),
                **audit,
            }

    taxonomy = load_taxonomy()
    expected = toy_expected_distribution()
    # Solo se conserva el chunk de menor prioridad aleatoria por video/estrato.
    by_stratum_video: dict[tuple[str, str], dict[str, tuple[str, dict[str, Any]]]] = {
        (split, label): {}
        for split in TOY_SPLIT_WEIGHTS
        for label in taxonomy.target_labels
    }
    scanned = eligible = 0
    _notify_progress(
        progress_callback,
        status="started",
        phase="leyendo dataset fuente",
        total=None,
        advance=0,
    )
    for row in read_jsonl(source):
        scanned += 1
        labels = row.get("coarse_labels")
        split = row.get("split")
        if (
            split not in TOY_SPLIT_WEIGHTS
            or not isinstance(labels, list)
            or len(labels) != 1
            or labels[0] not in taxonomy.target_labels
            or not str(row.get("text", "")).strip()
            or not row.get("video_id")
            or not row.get("chunk_id")
            or row.get("training_eligible") is False
        ):
            continue
        eligible += 1
        label = labels[0]
        video_id = str(row["video_id"])
        priority = _stable_digest(
            seed, "chunk", split, label, video_id, row["chunk_id"]
        )
        bucket = by_stratum_video[(split, label)]
        previous = bucket.get(video_id)
        if previous is None or priority < previous[0]:
            bucket[video_id] = (priority, row)
        if scanned % 25_000 == 0:
            _notify_progress(
                progress_callback,
                status="progress",
                phase="leyendo dataset fuente",
                advance=25_000,
                details={"filas": scanned, "puras_elegibles": eligible},
            )

    selected_rows: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    # Primero los daños más escasos; SEGURO queda al final. Esto evita que un
    # mismo video disponible en más de una categoría agote un estrato raro.
    for split in TOY_SPLIT_WEIGHTS:
        damage_order = sorted(
            taxonomy.damage_labels,
            key=lambda label: (len(by_stratum_video[(split, label)]), label),
        )
        for label in (*damage_order, taxonomy.safe_label):
            required = expected[split][label]
            candidates = [
                (video_id, priority, row)
                for video_id, (priority, row) in by_stratum_video[
                    (split, label)
                ].items()
                if video_id not in used_videos
            ]
            candidates.sort(
                key=lambda item: (
                    _stable_digest(seed, "video", split, label, item[0]),
                    item[1],
                )
            )
            if len(candidates) < required:
                raise ValueError(
                    f"Estrato insuficiente {split}/{label}: {len(candidates)} videos disponibles; se requieren {required}"
                )
            for video_id, _priority, row in candidates[:required]:
                used_videos.add(video_id)
                selected_rows.append(_selected_row(row, seed=seed))

    split_order = {split: index for index, split in enumerate(TOY_SPLIT_WEIGHTS)}
    label_order = {label: index for index, label in enumerate(taxonomy.target_labels)}
    selected_rows.sort(
        key=lambda row: (
            split_order[row["split"]],
            label_order[row["coarse_labels"][0]],
            _stable_digest(seed, "output", row["chunk_id"]),
        )
    )
    audit = validate_toy_dataset(selected_rows)
    write_jsonl_atomic(destination, selected_rows)
    manifest = {
        **configuration,
        "run_signature": signature,
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(source),
        "source_rows_scanned": scanned,
        "source_pure_rows_eligible": eligible,
        "dataset_path": str(destination),
        "dataset_sha256": sha256_file(destination),
        **audit,
        "split_interpretation": "80:20:20 son pesos normalizados 4:1:1, no porcentajes",
        "independent_experiment": True,
        "eligible_for_03_07": False,
    }
    write_json_atomic(manifest_path, manifest)
    _notify_progress(
        progress_callback,
        status="finished",
        phase="toy dataset creado",
        total=TOY_TOTAL_ROWS,
        completed=TOY_TOTAL_ROWS,
    )
    return {
        "status": "created",
        "dataset_path": str(destination),
        "manifest_path": str(manifest_path),
        **audit,
    }


def _toy_target(row: dict[str, Any], category: str | None = None) -> str:
    return json.dumps(
        {"chunk_id": row["chunk_id"], "categoria": category or row["coarse_labels"][0]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _toy_user_message(row: dict[str, Any]) -> str:
    return (
        "Clasifica el fragmento con las definiciones del contexto. "
        "Devuelve únicamente el objeto JSON permitido y copia chunk_id exactamente.\n"
        f"chunk_id: {row['chunk_id']}\nTexto:\n{row['text']}"
    )


class ToyPromptCompletionDataset:
    def __init__(
        self,
        tokenizer: Any,
        rows: Sequence[dict[str, Any]],
        system_prompt: str,
        *,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.rows = list(rows)
        self.system_prompt = system_prompt
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _toy_user_message(row)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = _toy_target(row) + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        overflow = max(0, len(prompt_ids) + len(target_ids) - self.max_length)
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        return {
            "input_ids": torch.tensor(
                input_ids + [self.tokenizer.pad_token_id] * padding
            ),
            "attention_mask": torch.tensor(attention + [0] * padding),
            "labels": torch.tensor(labels + [-100] * padding),
        }


def _completion_trie(
    tokenizer: Any, row: dict[str, Any], labels: Sequence[str]
) -> dict[tuple[int, ...], tuple[int, ...]]:
    branches: dict[tuple[int, ...], set[int]] = {}
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("El tokenizer de Qwen no define eos_token_id")
    for label in labels:
        tokens = list(
            tokenizer(_toy_target(row, label), add_special_tokens=False)["input_ids"]
        ) + [eos]
        for index, token in enumerate(tokens):
            branches.setdefault(tuple(tokens[:index]), set()).add(int(token))
    return {prefix: tuple(sorted(allowed)) for prefix, allowed in branches.items()}


def generate_constrained_toy_predictions(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    system_prompt: str,
    *,
    device: str,
    max_input_length: int,
    batch_size: int = 1,
    max_new_tokens: int = 96,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Genera sobre un trie: solo existen cinco JSON completos posibles por fila."""

    import torch

    labels = tuple(load_taxonomy().target_labels)
    records: list[dict[str, Any]] = []
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    model.eval()
    _notify_progress(
        progress_callback,
        status="started",
        phase="JSON restringido",
        total=len(rows),
        advance=0,
    )
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = list(rows[start : start + batch_size])
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": _toy_user_message(row)},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch_rows
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_input_length,
            )
            prompt_width = int(encoded["input_ids"].shape[1])
            tries = [_completion_trie(tokenizer, row, labels) for row in batch_rows]
            required_completion_tokens = max(
                max(len(prefix) for prefix in trie) + 1 for trie in tries
            )
            if required_completion_tokens > max_new_tokens:
                raise ValueError(
                    "generation_max_new_tokens no alcanza para el JSON restringido: "
                    f"se requieren {required_completion_tokens} y se configuraron "
                    f"{max_new_tokens}"
                )

            def allowed_tokens(
                batch_id: int,
                input_ids: Any,
                prompt_length: int = prompt_width,
                batch_tries: Sequence[dict[tuple[int, ...], tuple[int, ...]]] = tries,
            ) -> list[int]:
                prefix = tuple(
                    int(value) for value in input_ids[prompt_length:].tolist()
                )
                allowed = batch_tries[batch_id].get(prefix)
                return list(allowed) if allowed else [int(tokenizer.eos_token_id)]

            encoded = {
                key: value.to(device, non_blocking=device != "cpu")
                for key, value in encoded.items()
            }
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            texts = tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
            for row, text in zip(batch_rows, texts, strict=True):
                valid = False
                predicted = None
                error = None
                try:
                    payload = json.loads(text)
                    if set(payload) != {"chunk_id", "categoria"}:
                        raise ValueError("claves JSON distintas del esquema")
                    if payload["chunk_id"] != row["chunk_id"]:
                        raise ValueError("chunk_id no coincide")
                    if payload["categoria"] not in labels:
                        raise ValueError("categoría fuera del contrato")
                    predicted = payload["categoria"]
                    valid = True
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                expected = row["coarse_labels"][0]
                records.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "video_id": row["video_id"],
                        "split": row["split"],
                        "expected": expected,
                        "predicted": predicted,
                        "json_schema_valid": valid,
                        "correct": bool(valid and predicted == expected),
                        "error": error,
                        "raw_completion": text,
                    }
                )
            _notify_progress(
                progress_callback,
                status="progress",
                phase="JSON restringido",
                advance=len(batch_rows),
                details={"filas": len(records)},
            )
    finally:
        tokenizer.padding_side = previous_padding_side
    valid_count = sum(record["json_schema_valid"] for record in records)
    quality = {
        "rows": len(records),
        "valid_json_contract": valid_count,
        "schema_valid_rate": valid_count / max(1, len(records)),
        "constraint": "token_trie_exactly_five_valid_json_objects_per_row",
        "schema": {
            "type": "object",
            "required": ["chunk_id", "categoria"],
            "additionalProperties": False,
        },
    }
    _notify_progress(
        progress_callback,
        status="finished",
        phase="JSON restringido completo",
        total=len(rows),
        completed=len(rows),
    )
    return records, quality


def compute_toy_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(load_taxonomy().target_labels)
    y_true = [str(record["expected"]) for record in records]
    y_pred = [
        str(record["predicted"]) if record.get("predicted") else "__JSON_INVALIDO__"
        for record in records
    ]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    valid = [bool(record.get("json_schema_valid")) for record in records]
    valid_correct = sum(bool(record.get("correct")) for record in records)
    matrix_labels = [*labels, "__JSON_INVALIDO__"]
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    return {
        "rows": len(records),
        "primary_metric": "strict_macro_f1",
        "strict_macro_f1": float(np.mean(f1)),
        "strict_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "json_schema_valid_rate": float(np.mean(valid)) if valid else 0.0,
        "accuracy_given_valid_json": valid_correct / max(1, sum(valid)),
        "per_category": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": matrix_labels, "values": matrix.tolist()},
        "invalid_json_predictions": int(len(records) - sum(valid)),
    }


def _write_confusion_csv(path: Path, metrics: dict[str, Any]) -> None:
    labels = metrics["confusion_matrix"]["labels"]
    values = metrics["confusion_matrix"]["values"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["real\\predicha", *labels])
        for label, row in zip(labels, values, strict=True):
            writer.writerow([label, *row])


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    rows = [
        "# 03_06b · Experimento toy Qwen con contexto Markdown y JSON restringido",
        "",
        "Este ejercicio es independiente y no participa en la selección de modelos de `03_07`.",
        "",
        "## Diseño",
        "",
        "- Dataset: 1.200 fragmentos puros y 1.200 videos distintos.",
        "- Daño: 60 ejemplos por cada una de las cuatro categorías (240 en total).",
        "- `SEGURO`: 960 ejemplos.",
        "- Partición: pesos 80:20:20 normalizados a 4:1:1, equivalentes a 800/200/200.",
        "- Condición: contenido completo del archivo Markdown de definiciones, identificado por SHA-256.",
        "- Decodificación: trie de tokens que solo permite cinco objetos JSON válidos por fila.",
        "",
        "## Métrica principal",
        "",
        "`strict_macro_f1` promedia el F1 de las cinco categorías y cuenta una salida JSON inválida como predicción incorrecta.",
        "",
        "| partición | filas | strict macro-F1 | accuracy | balanced accuracy | JSON válido |",
        "|---|---:|---:|---:|---:|---:|",
        f"| validation | {validation_metrics['rows']} | {validation_metrics['strict_macro_f1']:.4f} | {validation_metrics['strict_accuracy']:.4f} | {validation_metrics['balanced_accuracy']:.4f} | {validation_metrics['json_schema_valid_rate']:.4f} |",
        f"| test | {test_metrics['rows']} | {test_metrics['strict_macro_f1']:.4f} | {test_metrics['strict_accuracy']:.4f} | {test_metrics['balanced_accuracy']:.4f} | {test_metrics['json_schema_valid_rate']:.4f} |",
        "",
        "## Test por categoría",
        "",
        "| categoría | soporte | precisión | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in test_metrics["per_category"].items():
        rows.append(
            f"| {label} | {values['support']} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |"
        )
    rows.extend(
        [
            "",
            "## Lectura crítica",
            "",
            "El conjunto está deliberadamente enriquecido y balancea la importancia analítica de los daños; por ello sus métricas no estiman la prevalencia natural de producción. Las categorías de daño tienen solo 10 casos cada una en test, así que un solo error cambia su recall en 0,10. El resultado sirve para verificar si Qwen aprende las definiciones y cumple el contrato JSON bajo este ejercicio controlado, no para declarar superioridad frente a otros modelos.",
            "",
            f"Prompt SHA-256: `{manifest['prompt_sha256']}`  ",
            f"Toy dataset SHA-256: `{manifest['dataset_sha256']}`",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train_and_evaluate_toy_qwen(
    toy_dataset_path: str | Path,
    definitions_markdown_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "auto",
    seed: int = TOY_SEED,
    epochs: int = 3,
    max_length: int = 1536,
    generation_max_new_tokens: int = 96,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Ajusta Qwen-LoRA y evalúa validation/test una sola vez en el ejercicio toy."""

    if epochs <= 0 or max_length < 512 or generation_max_new_tokens < 32:
        raise ValueError("epochs, max_length o generation_max_new_tokens inválidos")
    dataset_path = Path(toy_dataset_path).resolve()
    prompt_path = Path(definitions_markdown_path).resolve()
    output_root = Path(output_root).resolve()
    if not dataset_path.is_file() or not prompt_path.is_file():
        raise FileNotFoundError("Falta el toy dataset o el Markdown de definiciones")
    rows = list(read_jsonl(dataset_path))
    dataset_audit = validate_toy_dataset(rows)
    system_prompt = prompt_path.read_text(encoding="utf-8-sig").strip()
    taxonomy = load_taxonomy()
    missing_labels = [
        label for label in taxonomy.target_labels if label not in system_prompt
    ]
    if missing_labels:
        raise ValueError(
            f"El Markdown no define todas las categorías: {missing_labels}"
        )
    if '"chunk_id"' not in system_prompt or '"categoria"' not in system_prompt:
        raise ValueError("El Markdown no declara el esquema JSON toy")
    dataset_sha = sha256_file(dataset_path)
    prompt_sha = sha256_file(prompt_path)
    configuration = {
        "experiment_id": TOY_EXPERIMENT_ID,
        "model_id": PROMPT_SFT_MODEL_ID,
        "model_revision": PROMPT_SFT_MODEL_REVISION,
        "dataset_sha256": dataset_sha,
        "prompt_sha256": prompt_sha,
        "seed": seed,
        "epochs": epochs,
        "max_length": max_length,
        "generation_max_new_tokens": generation_max_new_tokens,
        "json_constraint": "token_trie_exactly_five_valid_json_objects_per_row",
    }
    signature = sha256_text(json.dumps(configuration, sort_keys=True))
    run_dir = output_root / "runs" / f"toy-qwen-{signature[:16]}"
    experiment_manifest_path = run_dir / "experiment_manifest.json"
    if experiment_manifest_path.is_file() and not force:
        manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("run_signature") == signature
            and manifest.get("status") == "complete"
        ):
            return {"status": "noop", "run_dir": str(run_dir), "manifest": manifest}

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc

    hardware = resolve_device(device)
    high_memory = high_memory_bf16_cuda(hardware)
    if hardware.backend == "cuda":
        cuda_preflight = _cuda_memory_preflight(torch)
    else:
        cuda_preflight = None
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    test_rows = [row for row in rows if row["split"] == "test"]
    batch_train = 8 if high_memory else 2 if hardware.backend == "cuda" else 1
    batch_eval = 16 if high_memory else 4 if hardware.backend == "cuda" else 1
    accumulation = 1 if high_memory else 4 if hardware.backend == "cuda" else 8
    torch_device = torch_device_name(hardware)
    tokenizer = AutoTokenizer.from_pretrained(
        PROMPT_SFT_MODEL_ID, revision=PROMPT_SFT_MODEL_REVISION, token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": PROMPT_SFT_MODEL_REVISION,
        "token": False,
    }
    if hardware.dtype == "bfloat16":
        model_kwargs["dtype"] = torch.bfloat16
    elif hardware.dtype == "float16":
        model_kwargs["dtype"] = torch.float16
    if hardware.backend == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(PROMPT_SFT_MODEL_ID, **model_kwargs)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.use_cache = False
    gradient_diagnostics = _configure_lora_gradient_checkpointing(model, enabled=True)
    train_dataset = ToyPromptCompletionDataset(
        tokenizer, train_rows, system_prompt, max_length=max_length
    )
    validation_dataset = ToyPromptCompletionDataset(
        tokenizer, validation_rows, system_prompt, max_length=max_length
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_train,
            per_device_eval_batch_size=batch_eval,
            gradient_accumulation_steps=accumulation,
            learning_rate=1e-4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=[],
            seed=seed,
            use_cpu=hardware.backend == "cpu",
            bf16=hardware.dtype == "bfloat16",
            fp16=hardware.dtype == "float16",
            remove_unused_columns=False,
            dataloader_pin_memory=hardware.backend != "cpu",
            optim="adamw_torch_fused" if hardware.backend == "cuda" else "adamw_torch",
            tf32=high_memory,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    _notify_progress(
        progress_callback,
        status="started",
        phase="entrenamiento Qwen toy",
        total=None,
        advance=0,
    )
    started = time.perf_counter()
    training_result = trainer.train()
    training_seconds = time.perf_counter() - started
    model.config.use_cache = True
    model.to(torch_device)
    validation_predictions, validation_quality = generate_constrained_toy_predictions(
        model,
        tokenizer,
        validation_rows,
        system_prompt,
        device=torch_device,
        max_input_length=max_length - generation_max_new_tokens,
        batch_size=batch_eval,
        max_new_tokens=generation_max_new_tokens,
        progress_callback=progress_callback,
    )
    validation_metrics = compute_toy_metrics(validation_predictions)
    test_predictions, test_quality = generate_constrained_toy_predictions(
        model,
        tokenizer,
        test_rows,
        system_prompt,
        device=torch_device,
        max_input_length=max_length - generation_max_new_tokens,
        batch_size=batch_eval,
        max_new_tokens=generation_max_new_tokens,
        progress_callback=progress_callback,
    )
    test_metrics = compute_toy_metrics(test_predictions)
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    (run_dir / "definitions_context.md").write_text(
        system_prompt + "\n", encoding="utf-8"
    )
    write_jsonl_atomic(run_dir / "predictions_validation.jsonl", validation_predictions)
    write_jsonl_atomic(run_dir / "predictions_test.jsonl", test_predictions)
    write_json_atomic(
        run_dir / "metrics_validation.json",
        {**validation_metrics, "generation_quality": validation_quality},
    )
    write_json_atomic(
        run_dir / "metrics_test.json",
        {**test_metrics, "generation_quality": test_quality},
    )
    _write_confusion_csv(run_dir / "confusion_matrix_test.csv", test_metrics)
    manifest = {
        "schema_version": TOY_SCHEMA_VERSION,
        "status": "complete",
        "experiment_id": TOY_EXPERIMENT_ID,
        "run_signature": signature,
        "completed_at": datetime.now(UTC).isoformat(),
        "independent_experiment": True,
        "eligible_for_03_07": False,
        "comparison_policy": "forbidden; no candidate.json is produced",
        "model_id": PROMPT_SFT_MODEL_ID,
        "model_revision": PROMPT_SFT_MODEL_REVISION,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "dataset_audit": dataset_audit,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_used_as_full_system_context": True,
        "structured_output": "token-level constrained JSON trie",
        "configuration": configuration,
        "hardware": hardware.model_dump(),
        "cuda_memory_preflight": cuda_preflight,
        "gradient_diagnostics": gradient_diagnostics,
        "training_metrics": dict(training_result.metrics),
        "training_seconds": training_seconds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "adapter": "adapter",
            "prompt_context": "definitions_context.md",
            "validation_predictions": "predictions_validation.jsonl",
            "test_predictions": "predictions_test.jsonl",
            "validation_metrics": "metrics_validation.json",
            "test_metrics": "metrics_test.json",
            "test_confusion_matrix": "confusion_matrix_test.csv",
            "report": "REPORTE_03_06B_TOY_QWEN.md",
        },
    }
    write_json_atomic(experiment_manifest_path, manifest)
    _write_report(
        run_dir / "REPORTE_03_06B_TOY_QWEN.md",
        manifest,
        validation_metrics,
        test_metrics,
    )
    return {
        "status": "trained_and_evaluated",
        "run_dir": str(run_dir),
        "manifest": manifest,
    }
