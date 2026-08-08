from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .device import resolve_device, torch_device_name
from .io import sha256_file, write_json_atomic
from .manifests import artifact_reference
from .paths import find_project_root, relative_to_root
from .schemas import ModelRegistryEntry
from .taxonomy import load_taxonomy


def _classical_scores(model: Any, texts: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        values = np.asarray(model.predict_proba(texts), dtype=float)
    elif hasattr(model, "decision_function"):
        logits = np.asarray(model.decision_function(texts), dtype=float)
        values = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    else:
        values = np.asarray(model.predict(texts), dtype=float)
    return values[:, None] if values.ndim == 1 else values


def discover_candidates(roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for candidate_path in path.rglob("candidate.json"):
            try:
                candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            candidate["candidate_path"] = str(candidate_path.resolve())
            candidates.append(candidate)
    return candidates


def _selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, str]:
    """Orden de validation: falsos seguros, F1 daño, AP daño y carga de revisión."""

    metrics = candidate["validation_metrics"]
    return (
        -float(metrics.get("false_safe_rate_on_damage", 1.0)),
        float(metrics.get("f1_macro_damage", 0.0)),
        float(metrics.get("average_precision_macro_damage", 0.0)),
        -float(metrics.get("review_load_rate", 1.0)),
        str(candidate.get("candidate_id", "")),
    )


def _candidate_asset(candidate: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(candidate["candidate_path"]).parent / path


def _model_slot(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("model_family", "")).casefold()
    if family.startswith("classical:"):
        return "classical"
    if family.startswith("qwen"):
        return "qwen"
    return "transformer"


def _registry_entry(
    candidate: dict[str, Any],
    *,
    dataset_sha: str,
    comparison_registries: dict[str, Any] | None = None,
) -> ModelRegistryEntry:
    taxonomy = load_taxonomy()
    manifest_path = _candidate_asset(candidate, candidate["checkpoint_manifest"]).resolve()
    portable_inference = dict(candidate["inference"])
    for key in ("bundle", "model", "gate_model", "damage_model"):
        if portable_inference.get(key):
            portable_inference[key] = relative_to_root(
                _candidate_asset(candidate, portable_inference[key])
            )
    return ModelRegistryEntry(
        model_id=candidate["candidate_id"],
        model_family=candidate["model_family"],
        taxonomy_contract=taxonomy.contract_id,
        taxonomy_version=taxonomy.version,
        target_labels=list(taxonomy.target_labels),
        checkpoint=artifact_reference(manifest_path, "checkpoint_manifest"),
        thresholds=candidate["thresholds"],
        metrics_path=relative_to_root(_candidate_asset(candidate, candidate["metrics_path"])),
        hardware=candidate.get("hardware"),
        dataset_sha256=dataset_sha,
        run_signature=candidate["run_signature"],
        inference=portable_inference,
        selection_metrics={
            "false_safe_rate_on_damage": float(
                candidate["validation_metrics"]["false_safe_rate_on_damage"]
            ),
            "f1_macro_damage": float(candidate["validation_metrics"]["f1_macro_damage"]),
            "average_precision_macro_damage": float(
                candidate["validation_metrics"]["average_precision_macro_damage"]
            ),
            "review_load_rate": float(candidate["validation_metrics"]["review_load_rate"]),
        },
        comparison_registries=comparison_registries or {},
        status="validated",
    )


def _write_registry(path: Path, entry: ModelRegistryEntry) -> str:
    payload = entry.model_dump(mode="json")
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) == payload:
            return "noop"
        write_json_atomic(path, payload)
        return "updated"
    write_json_atomic(path, payload)
    return "created"


def compare_and_publish_registry(
    dataset_path: str | Path,
    candidate_roots: Iterable[str | Path],
    registry_path: str | Path,
    *,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Selecciona exclusivamente con validation y publica un registro verificable."""

    dataset = Path(dataset_path).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_sha = sha256_file(dataset)
    taxonomy = load_taxonomy()
    rejected = []
    eligible = []
    for candidate in discover_candidates(candidate_roots):
        reasons = []
        if candidate.get("status") != "complete":
            reasons.append("incomplete")
        if candidate.get("dataset_sha256") != dataset_sha:
            reasons.append("different_snapshot")
        if tuple(candidate.get("target_labels", [])) != taxonomy.target_labels:
            reasons.append("wrong_contract")
        manifest = _candidate_asset(candidate, candidate.get("checkpoint_manifest", ""))
        if not manifest.is_file():
            reasons.append("checkpoint_manifest_missing")
        if reasons:
            rejected.append({"candidate_id": candidate.get("candidate_id"), "reasons": reasons})
        else:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("No hay candidatos completos para el SHA-256 del snapshot activo")
    selected = max(eligible, key=_selection_key)
    destination = Path(registry_path)
    best_by_slot = {
        slot: max((row for row in eligible if _model_slot(row) == slot), key=_selection_key)
        for slot in ("classical", "transformer", "qwen")
        if any(_model_slot(row) == slot for row in eligible)
    }
    member_paths: dict[str, Path] = {}
    member_status: dict[str, str] = {}
    for slot, candidate in best_by_slot.items():
        member_path = destination.with_name(f"{destination.stem}.{slot}{destination.suffix}")
        member_paths[slot] = member_path
        member_status[slot] = _write_registry(
            member_path,
            _registry_entry(candidate, dataset_sha=dataset_sha),
        )
    comparison_registries = {
        slot: artifact_reference(path, f"production_{slot}_registry")
        for slot, path in member_paths.items()
    }
    entry = _registry_entry(
        selected,
        dataset_sha=dataset_sha,
        comparison_registries=comparison_registries,
    )
    status = _write_registry(destination, entry)
    ranking = sorted(
        (
            {
                "candidate_id": candidate["candidate_id"],
                "model_family": candidate["model_family"],
                "validation_metrics": candidate["validation_metrics"],
                "test_metrics_report_only": candidate["test_metrics"],
                "selected": candidate["candidate_id"] == selected["candidate_id"],
            }
            for candidate in eligible
        ),
        key=lambda row: _selection_key(next(item for item in eligible if item["candidate_id"] == row["candidate_id"])),
        reverse=True,
    )
    comparison = {
        "schema_version": "2.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_order": [
            "min false_safe_rate_on_damage",
            "max f1_macro_damage",
            "max average_precision_macro_damage",
            "min review_load_rate",
        ],
        "selected": selected["candidate_id"],
        "selected_by_slot": {
            slot: candidate["candidate_id"] for slot, candidate in best_by_slot.items()
        },
        "consensus_available": set(best_by_slot) == {"classical", "transformer", "qwen"},
        "ranking": ranking,
        "rejected": rejected,
    }
    comparison_destination = Path(comparison_path) if comparison_path else destination.with_name("comparacion_modelos_5_salidas.json")
    if comparison_destination.is_file():
        previous_comparison = json.loads(comparison_destination.read_text(encoding="utf-8"))
        stable_previous = {key: value for key, value in previous_comparison.items() if key != "created_at"}
        stable_current = {key: value for key, value in comparison.items() if key != "created_at"}
        if stable_previous == stable_current:
            comparison = previous_comparison
        else:
            write_json_atomic(comparison_destination, comparison)
    else:
        write_json_atomic(comparison_destination, comparison)
    return {
        "status": status,
        "selected": selected["candidate_id"],
        "selected_by_slot": {
            slot: candidate["candidate_id"] for slot, candidate in best_by_slot.items()
        },
        "consensus_available": set(best_by_slot) == {"classical", "transformer", "qwen"},
        "eligible": len(eligible),
        "rejected": len(rejected),
        "registry": str(destination),
        "member_registries": {slot: str(path) for slot, path in member_paths.items()},
        "member_status": member_status,
        "comparison": str(comparison_destination),
    }


class ProductionPredictor:
    def __init__(self, registry_path: str | Path, *, device: str = "auto") -> None:
        self.registry_path = Path(registry_path).resolve()
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.entry = ModelRegistryEntry.model_validate(payload)
        try:
            self.root = find_project_root(self.registry_path)
        except FileNotFoundError:
            self.root = find_project_root()
        checkpoint = Path(self.entry.checkpoint.path) if self.entry.checkpoint else None
        if checkpoint and not checkpoint.is_absolute():
            checkpoint = self.root / checkpoint
        if checkpoint:
            if not checkpoint.is_file() or sha256_file(checkpoint) != self.entry.checkpoint.sha256:
                raise ValueError("El manifiesto del checkpoint no coincide con el registro")
            manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
            for record in manifest.get("files", []):
                artifact = checkpoint.parent / record["path"]
                if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
                    raise ValueError(f"Checkpoint incompleto o alterado: {artifact}")
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._damage_model: Any = None
        self._damage_tokenizer: Any = None

    def _asset(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _load(self) -> None:
        if self._model is not None:
            return
        inference = self.entry.inference
        kind = inference.get("type")
        if kind == "sklearn_joblib":
            import joblib

            bundle = self._asset(inference["bundle"])
            if not bundle.is_file():
                raise FileNotFoundError(bundle)
            specification = json.loads(bundle.read_text(encoding="utf-8"))
            self._model = joblib.load(bundle.parent / specification["model"])
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        hardware = resolve_device(self.device)
        self._torch_device = torch_device_name(hardware)
        if kind == "hf_sequence_classifier":
            model_path = self._asset(inference["model"])
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self._torch_device)
        elif kind == "hf_peft_sequence_classifier":
            from peft import AutoPeftModelForSequenceClassification

            model_path = self._asset(inference["model"])
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoPeftModelForSequenceClassification.from_pretrained(model_path).to(self._torch_device)
        elif kind == "hf_cascade":
            gate_path = self._asset(inference["gate_model"])
            damage_path = self._asset(inference["damage_model"])
            self._tokenizer = AutoTokenizer.from_pretrained(gate_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(gate_path).to(self._torch_device)
            self._damage_tokenizer = AutoTokenizer.from_pretrained(damage_path)
            self._damage_model = AutoModelForSequenceClassification.from_pretrained(damage_path).to(self._torch_device)
        else:
            raise ValueError(f"Tipo de inferencia no soportado: {kind}")

    def scores(self, text: str) -> dict[str, float]:
        if not text.strip():
            raise ValueError("El texto no puede estar vacío")
        self._load()
        labels = self.entry.target_labels
        kind = self.entry.inference["type"]
        if kind == "sklearn_joblib":
            values = _classical_scores(self._model, [text])[0]
        else:
            import torch

            encoded = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
            self._model.eval()
            with torch.no_grad():
                primary = torch.sigmoid(self._model(**encoded).logits)[0].float().cpu().numpy()
            if kind == "hf_cascade":
                damage_encoded = self._damage_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                damage_encoded = {key: value.to(self._torch_device) for key, value in damage_encoded.items()}
                self._damage_model.eval()
                with torch.no_grad():
                    damage = torch.sigmoid(self._damage_model(**damage_encoded).logits)[0].float().cpu().numpy()
                gate = float(primary[0])
                values = [1 - gate, *(gate * damage)]
            else:
                values = primary[:5]
        return {label: float(values[index]) for index, label in enumerate(labels)}
