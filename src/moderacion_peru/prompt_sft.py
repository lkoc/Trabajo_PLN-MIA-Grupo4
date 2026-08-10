from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .device import resolve_device, torch_device_name
from .experiments import (
    TRAINING_ENGINE_VERSION,
    ProgressCallback,
    _checkpoint_manifest,
    _dataset_splits,
    _evaluate_validation,
    _experiment_signature,
    _notify_progress,
    _require_project_safe_ratio,
)
from .io import sha256_file, write_json_atomic
from .taxonomy import load_taxonomy

PROMPT_SFT_MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT_SFT_MODEL_REVISION = "6130ef31402718485ca4d80a6234f70d9a4cf362"


def compile_operational_prompt_capsule(prompt: str, *, max_chars: int = 9000) -> str:
    """Compila v3.2 conservando contrato, jerarquía, categorías y JSON.

    La procedencia se prueba con el SHA del archivo completo. La cápsula evita
    repetir unas 6k fichas léxicas en cada ejemplo SFT, algo innecesariamente
    costoso; no crea criterios nuevos ni sustituye el prompt fuente.
    """

    headings = {
        "## Tarea y salidas permitidas",
        "## Principio rector: clasificar el evento de habla, no la palabra",
        "## Jerarquía obligatoria de decisión",
        "## Reglas transversales",
        "## Formato JSON obligatorio",
    }
    lines = prompt.splitlines()
    selected: list[str] = []
    active = False
    for line in lines:
        if line.startswith("## "):
            active = line.strip() in headings or line.startswith("## Jerarquía")
        if active or line.startswith(("Versión del prompt", "Contrato y taxonomía")):
            selected.append(line)
    capsule = "\n".join(selected).strip()
    if len(capsule) > max_chars:
        capsule = capsule[:max_chars].rsplit("\n", 1)[0]
        capsule += (
            "\n\n[La cápsula termina aquí; el SHA remite al prompt v3.2 completo.]"
        )
    return capsule


def _json_target(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "coarse_labels": row["coarse_labels"],
            "fine_labels": row.get("fine_labels", []),
            "flags": row.get("flags_reference_only", []),
            "confidence": 1.0,
            "needs_review": False,
            "reasoning": "decisión supervisada del snapshot",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _mask_json_field(
    labels: list[int], offsets: Sequence[tuple[int, int]], text: str, field: str
) -> None:
    marker = f'"{field}":'
    start = text.find(marker)
    if start < 0:
        return
    value_start = start + len(marker)
    value_end = text.find("]", value_start) + 1
    for index, (left, right) in enumerate(offsets):
        if right > value_start and left < value_end:
            labels[index] = -100


class PromptCompletionDataset:
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

        taxonomy = load_taxonomy()
        row = self.rows[index]
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": "Clasifica este chunk y devuelve solo JSON:\n"
                    + str(row["text"]),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = _json_target(row) + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_encoding = self.tokenizer(
            target, add_special_tokens=False, return_offsets_mapping=True
        )
        target_ids = target_encoding["input_ids"]
        target_labels = list(target_ids)
        if not all(row.get("fine_observed_mask", [0] * len(taxonomy.fine_labels))):
            _mask_json_field(
                target_labels, target_encoding["offset_mapping"], target, "fine_labels"
            )
        if not all(row.get("flags_observed_mask", [0] * len(taxonomy.flags))):
            _mask_json_field(
                target_labels, target_encoding["offset_mapping"], target, "flags"
            )
        overflow = max(0, len(prompt_ids) + len(target_ids) - self.max_length)
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_labels
        attention = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        attention += [0] * padding
        labels += [-100] * padding
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
        }


def _generate_json_scores(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    system_prompt: str,
    *,
    device: str,
    max_input_length: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    taxonomy = load_taxonomy()
    output_labels = [*taxonomy.target_labels, *taxonomy.fine_labels, *taxonomy.flags]
    scores = np.zeros((len(rows), len(output_labels)), dtype=float)
    valid = 0
    schema_errors: dict[str, int] = {}
    model.eval()
    _notify_progress(
        progress_callback,
        status="started",
        phase="generando validation",
        total=len(rows),
        advance=0,
    )
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Clasifica este chunk y devuelve solo JSON:\n"
                    + str(row["text"]),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        try:
            payload = json.loads(match.group(0) if match else text)
            coarse = taxonomy.normalize_categories(payload.get("coarse_labels", ()))
            fine = taxonomy.normalize_fine_labels(payload.get("fine_labels", ()))
            flags = tuple(
                flag for flag in payload.get("flags", ()) if flag in taxonomy.flags
            )
            confidence = float(np.clip(payload.get("confidence", 0.5), 0.0, 1.0))
            if not coarse:
                raise ValueError("coarse_labels vacío")
            selected = set(coarse) | set(fine) | set(flags)
            scores[index] = [
                confidence if label in selected else 1 - confidence
                for label in output_labels
            ]
            valid += 1
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            key = type(exc).__name__
            schema_errors[key] = schema_errors.get(key, 0) + 1
        _notify_progress(
            progress_callback,
            status="progress",
            phase="generando validation",
            advance=1,
            details={"JSON válidos": valid, "fila": index + 1},
        )
    _notify_progress(
        progress_callback,
        status="finished",
        phase="validation generada",
        total=len(rows),
        completed=len(rows),
    )
    return scores, {
        "rows": len(rows),
        "valid_json_contract": valid,
        "schema_valid_rate": valid / max(1, len(rows)),
        "schema_errors": schema_errors,
        "score_semantics": "self_reported_confidence_for_selected_labels_and_one_minus_for_others",
    }


def train_prompt_conditioned_sft(
    dataset_path: str | Path,
    prompt_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "auto",
    safe_to_damage_ratio: float | None = 4.0,
    seed: int = 20260805,
    epochs: int = 2,
    max_length: int = 4096,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """LoRA SFT generativo condicionado por una cápsula trazable de v3.2."""

    safe_to_damage_ratio = _require_project_safe_ratio(safe_to_damage_ratio)
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
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc
    run_started = time.perf_counter()
    dataset = Path(dataset_path).resolve()
    prompt_path = Path(prompt_path).resolve()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_file(prompt_path)
    system_prompt = compile_operational_prompt_capsule(prompt_text)
    configuration = {
        "experiment": "qwen_prompt_sft",
        "model_id": PROMPT_SFT_MODEL_ID,
        "model_revision": PROMPT_SFT_MODEL_REVISION,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha,
        "prompt_capsule_sha256": __import__("hashlib")
        .sha256(system_prompt.encode())
        .hexdigest(),
        "safe_to_damage_ratio_train_validation": safe_to_damage_ratio,
        "test_policy": "full_natural_plus_4_to_1_secondary_same_predictions",
        "epochs": epochs,
        "max_length": max_length,
        "train_limit": train_limit,
        "validation_limit": validation_limit,
        "seed": seed,
        "test_status": "sealed_not_evaluated",
    }
    signature = _experiment_signature(dataset, "qwen_prompt_sft", configuration)
    run_dir = Path(output_root) / "runs" / f"qwen-prompt-sft-{signature[:16]}"
    pilot = train_limit is not None or validation_limit is not None
    candidate_path = run_dir / ("pilot_candidate.json" if pilot else "candidate.json")
    if candidate_path.is_file() and not force:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidate.get("run_signature") == signature:
            candidate["candidate_path"] = str(candidate_path)
            _notify_progress(
                progress_callback,
                status="finished",
                phase="candidato ya existente",
                total=1,
                completed=1,
            )
            return {"status": "noop", "candidate": candidate}
    train, validation, test_sealed, sampling = _dataset_splits(
        dataset,
        split_scheme="video",
        safe_to_damage_ratio=safe_to_damage_ratio,
        sampling_seed=seed,
    )
    test_count = len(test_sealed)
    if train_limit is not None:
        train = sorted(
            train,
            key=lambda row: hashlib.sha256(
                f"{seed}|prompt-sft-train|{row['chunk_id']}".encode()
            ).hexdigest(),
        )[:train_limit]
    if validation_limit is not None:
        validation = sorted(
            validation,
            key=lambda row: hashlib.sha256(
                f"{seed}|prompt-sft-validation|{row['chunk_id']}".encode()
            ).hexdigest(),
        )[:validation_limit]
    hardware = resolve_device(device)
    torch_device = torch_device_name(hardware)
    tokenizer = AutoTokenizer.from_pretrained(
        PROMPT_SFT_MODEL_ID, revision=PROMPT_SFT_MODEL_REVISION
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PROMPT_SFT_MODEL_ID,
        revision=PROMPT_SFT_MODEL_REVISION,
        torch_dtype=(torch.bfloat16 if hardware.dtype == "bfloat16" else None),
    )
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
    model.gradient_checkpointing_enable()
    train_dataset = PromptCompletionDataset(
        tokenizer, train, system_prompt, max_length=max_length
    )
    validation_dataset = PromptCompletionDataset(
        tokenizer, validation, system_prompt, max_length=max_length
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
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
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    checkpoint = (
        get_last_checkpoint(str(run_dir / "trainer"))
        if (run_dir / "trainer").is_dir()
        else None
    )
    _notify_progress(
        progress_callback,
        status="started",
        phase="entrenamiento SFT (Trainer muestra pasos y épocas)",
        total=None,
        advance=0,
    )
    training_started = time.perf_counter()
    training_result = trainer.train(resume_from_checkpoint=checkpoint)
    training_elapsed = time.perf_counter() - training_started
    model.to(torch_device)
    validation_started = time.perf_counter()
    validation_scores, generation = _generate_json_scores(
        model,
        tokenizer,
        validation,
        system_prompt,
        device=torch_device,
        max_input_length=max_length - 256,
        progress_callback=progress_callback,
    )
    validation_generation_elapsed = time.perf_counter() - validation_started
    taxonomy = load_taxonomy()
    output_labels = [
        *taxonomy.target_labels,
        *(f"fine:{label}" for label in taxonomy.fine_labels),
        *(f"flag:{flag}" for flag in taxonomy.flags),
    ]
    metrics_started = time.perf_counter()
    thresholds, validation_metrics, auxiliary = _evaluate_validation(
        run_dir, validation, validation_scores, output_labels
    )
    validation_metrics_elapsed = time.perf_counter() - metrics_started
    model_dir = run_dir / "adapter"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    write_json_atomic(
        run_dir / "prompt_provenance.json", {**configuration, "capsule": system_prompt}
    )
    write_json_atomic(run_dir / "generation_quality.json", generation)
    inference = {
        "type": "hf_prompt_sft_json",
        "model": model_dir.name,
        "base_model": PROMPT_SFT_MODEL_ID,
        "base_revision": PROMPT_SFT_MODEL_REVISION,
        "prompt_sha256": prompt_sha,
        "prompt_capsule": "prompt_provenance.json",
    }
    bundle = run_dir / "inference.json"
    write_json_atomic(bundle, inference)
    manifest = _checkpoint_manifest(
        run_dir, [model_dir, bundle, run_dir / "prompt_provenance.json"]
    )
    candidate = {
        "schema_version": "4.0.0",
        "candidate_id": f"qwen-prompt-sft-{signature[:12]}",
        "experiment": "qwen_prompt_sft",
        "model_family": "qwen_prompt_sft",
        "conditioning": "operational_prompt_v3_2_capsule_plus_chunk_to_strict_json",
        "run_signature": signature,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "target_labels": list(taxonomy.target_labels),
        "output_count": 22,
        "output_labels": output_labels,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "auxiliary_validation_metrics": auxiliary,
        "test_metrics": None,
        "test_status": "sealed_not_evaluated",
        "generation_quality": generation,
        "training_sampling": {**sampling, "split_field": "split"},
        "metrics_path": "metrics.json",
        "checkpoint_manifest": manifest.name,
        "inference": {**inference, "bundle": bundle.name},
        "hardware": hardware.model_dump(),
        "training_metrics": dict(training_result.metrics),
        "stage_timings_seconds": {
            "training_fit": training_elapsed,
            "validation_generation": validation_generation_elapsed,
            "validation_metrics_and_thresholds": validation_metrics_elapsed,
            "total_before_candidate_write": time.perf_counter() - run_started,
        },
        "status": "pilot_complete" if pilot else "complete",
        "pilot_not_eligible_for_03_07": pilot,
        "completed_at": datetime.now(UTC).isoformat(),
        "engine": TRAINING_ENGINE_VERSION,
        "test_rows_sealed": test_count,
    }
    write_json_atomic(candidate_path, candidate)
    candidate["candidate_path"] = str(candidate_path)
    return {"status": "trained", "candidate": candidate}
