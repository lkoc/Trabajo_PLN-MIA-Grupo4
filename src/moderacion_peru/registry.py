from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cascade import combine_safety_first_cascade_scores
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


def _peft_registry_output_contract(
    inference: dict[str, Any], primary_labels: list[str]
) -> tuple[int, list[str]]:
    raw_labels = inference.get("output_labels") or ()
    labels = [str(label) for label in raw_labels]
    output_count = int(inference.get("output_count") or len(labels) or 5)
    if output_count < len(primary_labels):
        raise ValueError("El adaptador PEFT no contiene las cinco salidas primarias")
    if labels and len(labels) != output_count:
        raise ValueError("output_count y output_labels del adaptador PEFT difieren")
    if labels and labels[: len(primary_labels)] != primary_labels:
        raise ValueError("El adaptador PEFT no respeta el orden de salidas primarias")
    if not labels:
        labels = [
            *primary_labels,
            *[
                f"AUXILIAR_{index:02d}"
                for index in range(1, output_count - len(primary_labels) + 1)
            ],
        ]
    return output_count, labels


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
    manifest_path = _candidate_asset(
        candidate, candidate["checkpoint_manifest"]
    ).resolve()
    portable_inference = dict(candidate["inference"])
    for key in ("bundle", "model", "gate_model", "damage_model", "branch_model"):
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
        metrics_path=relative_to_root(
            _candidate_asset(candidate, candidate["metrics_path"])
        ),
        hardware=candidate.get("hardware"),
        dataset_sha256=dataset_sha,
        run_signature=candidate["run_signature"],
        inference=portable_inference,
        selection_metrics={
            "false_safe_rate_on_damage": float(
                candidate["validation_metrics"]["false_safe_rate_on_damage"]
            ),
            "f1_macro_damage": float(
                candidate["validation_metrics"]["f1_macro_damage"]
            ),
            "average_precision_macro_damage": float(
                candidate["validation_metrics"]["average_precision_macro_damage"]
            ),
            "review_load_rate": float(
                candidate["validation_metrics"]["review_load_rate"]
            ),
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
        family = str(candidate.get("model_family", "")).casefold()
        if (
            family.endswith(":linear_svm")
            and candidate.get("fit_quality", {}).get("converged") is not True
        ):
            reasons.append("svm_convergence_not_verified")
        manifest = _candidate_asset(candidate, candidate.get("checkpoint_manifest", ""))
        if not manifest.is_file():
            reasons.append("checkpoint_manifest_missing")
        if reasons:
            rejected.append(
                {"candidate_id": candidate.get("candidate_id"), "reasons": reasons}
            )
        else:
            eligible.append(candidate)
    if not eligible:
        raise ValueError(
            "No hay candidatos completos para el SHA-256 del snapshot activo"
        )
    selected = max(eligible, key=_selection_key)
    destination = Path(registry_path)
    best_by_slot = {
        slot: max(
            (row for row in eligible if _model_slot(row) == slot), key=_selection_key
        )
        for slot in ("classical", "transformer", "qwen")
        if any(_model_slot(row) == slot for row in eligible)
    }
    member_paths: dict[str, Path] = {}
    member_status: dict[str, str] = {}
    for slot, candidate in best_by_slot.items():
        member_path = destination.with_name(
            f"{destination.stem}.{slot}{destination.suffix}"
        )
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
        key=lambda row: _selection_key(
            next(
                item for item in eligible if item["candidate_id"] == row["candidate_id"]
            )
        ),
        reverse=True,
    )
    comparison = {
        "schema_version": "2.1.0",
        "created_at": datetime.now(UTC).isoformat(),
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
        "consensus_available": set(best_by_slot)
        == {"classical", "transformer", "qwen"},
        "ranking": ranking,
        "rejected": rejected,
    }
    comparison_destination = (
        Path(comparison_path)
        if comparison_path
        else destination.with_name("comparacion_modelos_5_salidas.json")
    )
    if comparison_destination.is_file():
        previous_comparison = json.loads(
            comparison_destination.read_text(encoding="utf-8")
        )
        stable_previous = {
            key: value
            for key, value in previous_comparison.items()
            if key != "created_at"
        }
        stable_current = {
            key: value for key, value in comparison.items() if key != "created_at"
        }
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
        "consensus_available": set(best_by_slot)
        == {"classical", "transformer", "qwen"},
        "eligible": len(eligible),
        "rejected": len(rejected),
        "registry": str(destination),
        "member_registries": {slot: str(path) for slot, path in member_paths.items()},
        "member_status": member_status,
        "comparison": str(comparison_destination),
    }


def publish_frozen_ensemble_registry(
    freeze_path: str | Path,
    candidate_roots: Iterable[str | Path],
    registry_path: str | Path,
) -> dict[str, Any]:
    """Materializa el ensemble congelado por 03_07 como registro de modo sombra."""

    freeze_file = Path(freeze_path).resolve()
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    if freeze.get("selected_id") != "ensemble_soft_mean":
        raise ValueError("El frontend vigente exige ensemble_soft_mean congelado")
    members = [str(value) for value in freeze.get("members", [])]
    if len(members) != 3:
        raise ValueError("El ensemble congelado debe contener exactamente tres miembros")

    discovered = {row["candidate_id"]: row for row in discover_candidates(candidate_roots)}
    missing = [identifier for identifier in members if identifier not in discovered]
    if missing:
        raise FileNotFoundError(
            "Faltan candidatos congelados para producción: " + ", ".join(missing)
        )
    dataset_sha = str(freeze["dataset_sha256"])
    destination = Path(registry_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    member_paths: dict[str, Path] = {}
    member_status: dict[str, str] = {}
    member_ids: dict[str, str] = {}

    for identifier in members:
        candidate = discovered[identifier]
        candidate_sha = str(candidate.get("dataset_sha256") or "")
        if candidate_sha and candidate_sha != dataset_sha:
            raise ValueError(f"Snapshot incompatible en {identifier}")
        slot = _model_slot(candidate)
        if slot in member_paths:
            raise ValueError(f"Hay más de un miembro congelado para el slot {slot}")
        entry = _registry_entry(candidate, dataset_sha=dataset_sha)
        payload = entry.model_dump(mode="json")
        payload.update(
            {
                "thresholds": freeze["member_thresholds"][identifier],
                "score_calibrators": freeze["member_score_calibrators"][identifier],
                "status": "validated",
            }
        )
        member_entry = ModelRegistryEntry.model_validate(payload)
        member_path = destination.with_name(
            f"{destination.stem}.{slot}{destination.suffix}"
        )
        member_paths[slot] = member_path
        member_ids[slot] = identifier
        member_status[slot] = _write_registry(member_path, member_entry)

    if set(member_paths) != {"classical", "transformer", "qwen"}:
        raise ValueError("El congelamiento no cubre clásico, Transformer y Qwen")
    references = {
        slot: artifact_reference(path, f"production_{slot}_registry")
        for slot, path in member_paths.items()
    }
    taxonomy = load_taxonomy()
    main = ModelRegistryEntry(
        model_id=str(freeze["selected_id"]),
        model_family="ensemble:soft_mean",
        taxonomy_contract=taxonomy.contract_id,
        taxonomy_version=taxonomy.version,
        target_labels=list(taxonomy.target_labels),
        thresholds=freeze["thresholds"],
        dataset_sha256=dataset_sha,
        run_signature=str(freeze.get("comparison_signature") or ""),
        inference={"type": "ensemble_soft_mean", "member_ids": member_ids},
        comparison_registries=references,
        score_calibrators=freeze["score_calibrators"],
        ensemble_kind="soft_mean",
        selected_members=members,
        any_damage_threshold=float(freeze["any_damage_threshold"]),
        needs_review_policy=freeze.get("needs_review_policy") or {},
        selection_artifact=artifact_reference(freeze_file, "frozen_model_selection"),
        winner_status=freeze.get("winner_status"),
        status="shadow_only",
    )
    status = _write_registry(destination, main)
    return {
        "status": status,
        "registry": str(destination),
        "selected": main.model_id,
        "deployment_status": main.status,
        "member_ids": member_ids,
        "member_registries": {slot: str(path) for slot, path in member_paths.items()},
        "member_status": member_status,
    }


def calibrate_score_mapping(
    scores: dict[str, float],
    calibrators: list[dict[str, Any]],
    labels: Iterable[str],
) -> dict[str, float]:
    ordered = list(labels)
    if not calibrators:
        return {label: float(scores[label]) for label in ordered}
    if len(calibrators) != len(ordered):
        raise ValueError("El número de calibradores no coincide con las salidas")
    calibrated: dict[str, float] = {}
    for label, record in zip(ordered, calibrators, strict=True):
        if record.get("type") != "sigmoid_platt":
            raise ValueError(f"Calibrador no soportado para {label}: {record.get('type')}")
        logit = float(record["coefficient"]) * float(scores[label]) + float(
            record["intercept"]
        )
        calibrated[label] = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))
    return calibrated


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
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != self.entry.checkpoint.sha256
            ):
                raise ValueError(
                    "El manifiesto del checkpoint no coincide con el registro"
                )
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
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, token=False, fix_mistral_regex=False
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path, token=False
            ).to(self._torch_device)
        elif kind == "hf_peft_sequence_classifier":
            from peft import AutoPeftModelForSequenceClassification

            model_path = self._asset(inference["model"])
            tokenizer_source: str | Path = model_path
            if not (model_path / "tokenizer_config.json").is_file():
                adapter_config = json.loads(
                    (model_path / "adapter_config.json").read_text(encoding="utf-8")
                )
                tokenizer_source = str(adapter_config["base_model_name_or_path"])
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, token=False, fix_mistral_regex=False
            )
            output_count, output_labels = _peft_registry_output_contract(
                inference, list(self.entry.target_labels)
            )
            id2label = dict(enumerate(output_labels))
            self._model = AutoPeftModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=output_count,
                id2label=id2label,
                label2id={label: index for index, label in id2label.items()},
                problem_type="multi_label_classification",
                token=False,
            ).to(self._torch_device)
        elif kind in {"hf_cascade", "hf_cascade_v2"}:
            gate_path = self._asset(inference["gate_model"])
            specialist_path = self._asset(
                inference["branch_model" if kind == "hf_cascade_v2" else "damage_model"]
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                gate_path, token=False, fix_mistral_regex=False
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                gate_path, token=False
            ).to(self._torch_device)
            self._damage_tokenizer = AutoTokenizer.from_pretrained(
                specialist_path, token=False, fix_mistral_regex=False
            )
            self._damage_model = AutoModelForSequenceClassification.from_pretrained(
                specialist_path, token=False
            ).to(self._torch_device)
        else:
            raise ValueError(f"Tipo de inferencia no soportado: {kind}")

    def raw_scores(self, text: str) -> dict[str, float]:
        if not text.strip():
            raise ValueError("El texto no puede estar vacío")
        self._load()
        labels = self.entry.target_labels
        kind = self.entry.inference["type"]
        if kind == "sklearn_joblib":
            values = _classical_scores(self._model, [text])[0]
        else:
            import torch

            encoded = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            )
            encoded = {
                key: value.to(self._torch_device) for key, value in encoded.items()
            }
            self._model.eval()
            with torch.no_grad():
                primary = (
                    torch.sigmoid(self._model(**encoded).logits)[0]
                    .float()
                    .cpu()
                    .numpy()
                )
            if kind in {"hf_cascade", "hf_cascade_v2"}:
                damage_encoded = self._damage_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=256
                )
                damage_encoded = {
                    key: value.to(self._torch_device)
                    for key, value in damage_encoded.items()
                }
                self._damage_model.eval()
                with torch.no_grad():
                    damage = (
                        torch.sigmoid(self._damage_model(**damage_encoded).logits)[0]
                        .float()
                        .cpu()
                        .numpy()
                    )
                gate = float(primary[0])
                if kind == "hf_cascade_v2":
                    values = combine_safety_first_cascade_scores(
                        np.asarray([gate]),
                        np.asarray([damage[:5]]),
                        gate_threshold=float(self.entry.inference["gate_threshold"]),
                    )[0]
                else:
                    values = [1 - gate, *(gate * damage)]
            else:
                values = primary[:5]
        return {label: float(values[index]) for index, label in enumerate(labels)}

    def calibrate_scores(self, scores: dict[str, float]) -> dict[str, float]:
        return calibrate_score_mapping(
            scores,
            self.entry.score_calibrators,
            self.entry.target_labels,
        )

    def scores(self, text: str) -> dict[str, float]:
        return self.calibrate_scores(self.raw_scores(text))
