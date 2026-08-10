from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .taxonomy import load_taxonomy


def utc_now() -> datetime:
    return datetime.now(UTC)


class AnnotationRecord(BaseModel):
    """Contrato canónico de una anotación del flujo v2."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.1.0"
    taxonomy_version: str = "2.1.0"
    chunk_id: str = Field(min_length=1)
    video_id: str | None = None
    channel_id: str | None = None
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    video_title: str | None = None
    channel_title: str | None = None
    source_url: str | None = None
    cohort: str | None = None
    text: str = Field(min_length=1)
    coarse_labels: list[str] = Field(default_factory=list)
    fine_labels: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    needs_review: bool = False
    training_eligible: bool = True
    decision_status: Literal["resolved", "needs_review", "excluded"] = "resolved"
    score_confianza: float | None = Field(default=None, ge=0, le=1)
    notes: str = ""
    justification: str = ""
    label_source: str
    annotator_type: Literal["human", "llm_local", "llm_remote", "migration", "rule"]
    annotator_model: str | None = None
    prompt_sha256: str | None = None
    source_record_sha256: str | None = None
    consolidated_sources: list[str] = Field(default_factory=list)
    consolidation_warning: str | None = None
    review_event_id: str | None = None
    review_action: Literal["accept", "modify", "reject", "defer"] | None = None
    reviewer_pseudonym: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_semantics(self) -> AnnotationRecord:
        taxonomy = load_taxonomy()
        self.coarse_labels = list(taxonomy.normalize_categories(self.coarse_labels))
        self.fine_labels = list(taxonomy.normalize_fine_labels(self.fine_labels))
        unknown_flags = set(self.flags) - set(taxonomy.flags)
        if unknown_flags:
            raise ValueError(f"Flags desconocidos: {sorted(unknown_flags)}")
        if not self.coarse_labels:
            if self.decision_status == "excluded" and not self.training_eligible:
                pass
            elif not self.needs_review or self.training_eligible:
                raise ValueError(
                    "Una anotación sin categoría debe requerir revisión y no ser entrenable"
                )
        if self.needs_review:
            self.decision_status = "needs_review"
        elif self.decision_status == "needs_review":
            raise ValueError("decision_status=needs_review exige needs_review=true")
        if self.decision_status == "excluded" and self.training_eligible:
            raise ValueError("Una anotación excluida no puede ser entrenable")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds < self.start_seconds
        ):
            raise ValueError("end_seconds no puede preceder a start_seconds")
        return self


class ModelReadyRecord(BaseModel):
    """Fila entrenable: conserva procedencia sin fingir que es un evento nuevo."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "2.1.0"
    taxonomy_version: str = "2.1.0"
    chunk_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    channel_id: str | None = None
    channel_title: str | None = None
    text: str = Field(min_length=1)
    coarse_labels: list[str]
    fine_labels: list[str] = Field(default_factory=list)
    flags_reference_only: list[str] = Field(default_factory=list)
    coarse_observed_mask: list[int] = Field(default_factory=list)
    fine_observed_mask: list[int] = Field(default_factory=list)
    flags_observed_mask: list[int] = Field(default_factory=list)
    label_source: str = Field(min_length=1)
    prompt_sha256: str | None = None
    sample_weight: float = Field(default=1.0, ge=0)
    campaign: str | None = None
    split: Literal["train", "validation", "test"]
    channel_split: Literal["train", "validation", "test"] | None = None
    needs_review: bool = False
    training_eligible: bool = True
    decision_status: Literal["resolved", "needs_review", "excluded"] = "resolved"
    legacy_coarse_labels: list[str] = Field(default_factory=list)
    label_source_original: str | None = None
    migration_warning: str | None = None

    @model_validator(mode="after")
    def validate_training_row(self) -> ModelReadyRecord:
        taxonomy = load_taxonomy()
        self.coarse_labels = list(taxonomy.normalize_categories(self.coarse_labels))
        self.fine_labels = list(taxonomy.normalize_fine_labels(self.fine_labels))
        unknown_flags = set(self.flags_reference_only) - set(taxonomy.flags)
        if unknown_flags:
            raise ValueError(f"Flags desconocidos: {sorted(unknown_flags)}")
        if not self.coarse_observed_mask:
            self.coarse_observed_mask = [1] * len(taxonomy.target_labels)
        if not self.fine_observed_mask:
            self.fine_observed_mask = [
                int(bool(self.fine_labels)) for _ in taxonomy.fine_labels
            ]
        if not self.flags_observed_mask:
            self.flags_observed_mask = [
                int(bool(self.flags_reference_only)) for _ in taxonomy.flags
            ]
        expected_masks = (
            (
                "coarse_observed_mask",
                self.coarse_observed_mask,
                len(taxonomy.target_labels),
            ),
            ("fine_observed_mask", self.fine_observed_mask, len(taxonomy.fine_labels)),
            ("flags_observed_mask", self.flags_observed_mask, len(taxonomy.flags)),
        )
        for name, mask, expected in expected_masks:
            if len(mask) != expected or any(value not in {0, 1} for value in mask):
                raise ValueError(
                    f"{name} debe contener {expected} posiciones binarias en orden canónico"
                )
        if self.training_eligible and (not self.coarse_labels or self.needs_review):
            raise ValueError("Una fila entrenable requiere categoría resuelta")
        if self.legacy_coarse_labels:
            migrated = taxonomy.migrate_legacy_categories(self.legacy_coarse_labels)
            if migrated != tuple(self.coarse_labels):
                raise ValueError(
                    "La salida canónica no coincide con la procedencia histórica"
                )
        return self


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    bytes: int = Field(ge=0)
    role: str
    required: bool = True


class HardwareRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: Literal["cuda", "rocm", "xpu", "cpu"]
    requested: str = "auto"
    device_name: str
    torch_version: str | None = None
    runtime_version: str | None = None
    total_memory_bytes: int | None = None
    dtype: str = "float32"
    fallback_reason: str | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.1.0"
    run_id: str
    stage: Literal["01_datos", "02_etiquetado", "03_entrenamiento", "04_produccion"]
    taxonomy_contract: str
    taxonomy_version: str
    created_at: datetime = Field(default_factory=utc_now)
    code_revision: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    hardware: HardwareRecord | None = None
    inputs: list[ArtifactReference] = Field(default_factory=list)
    outputs: list[ArtifactReference] = Field(default_factory=list)
    counters: dict[str, int | float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.1.0"
    model_id: str
    model_family: str
    taxonomy_contract: str
    taxonomy_version: str
    target_labels: list[str]
    checkpoint: ArtifactReference | None = None
    thresholds: dict[str, float]
    metrics_path: str | None = None
    parent_model_id: str | None = None
    hardware: HardwareRecord | None = None
    dataset_sha256: str | None = None
    run_signature: str | None = None
    inference: dict[str, Any] = Field(default_factory=dict)
    selection_metrics: dict[str, float] = Field(default_factory=dict)
    comparison_registries: dict[str, ArtifactReference] = Field(default_factory=dict)
    consensus_min_votes: Literal[2] = 2
    status: Literal["candidate", "validated", "shadow_only", "archived"] = "candidate"

    @model_validator(mode="after")
    def validate_targets(self) -> ModelRegistryEntry:
        taxonomy = load_taxonomy()
        if tuple(self.target_labels) != taxonomy.target_labels:
            raise ValueError("El registro no usa las cinco salidas canónicas en orden")
        if set(self.thresholds) != set(taxonomy.target_labels):
            raise ValueError("Debe existir un umbral para cada salida entrenada")
        unknown_slots = set(self.comparison_registries) - {
            "classical",
            "transformer",
            "qwen",
        }
        if unknown_slots:
            raise ValueError(f"Slots productivos desconocidos: {sorted(unknown_slots)}")
        return self


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.1.0"
    event_id: str
    chunk_id: str
    action: Literal["accept", "modify", "reject", "defer"]
    proposed_labels: list[str] = Field(default_factory=list)
    final_labels: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    reviewer: str = Field(min_length=1)
    model_id: str | None = None
    source_event_id: str | None = None
    decision_scope: Literal["chunk", "video", "channel"] = "chunk"
    decision_scope_key: str | None = None
    batch_id: str | None = None
    batch_target_count: int | None = Field(default=None, ge=1)
    taxonomy_version: str = "2.1.0"
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_review(self) -> ReviewEvent:
        taxonomy = load_taxonomy()
        self.proposed_labels = list(taxonomy.normalize_categories(self.proposed_labels))
        self.final_labels = list(taxonomy.normalize_categories(self.final_labels))
        if self.action == "defer" and self.final_labels:
            raise ValueError("Una decisión diferida no puede tener categorías finales")
        if self.action in {"accept", "modify"} and not self.final_labels:
            raise ValueError(
                "Aceptar o modificar requiere una categoría final explícita"
            )
        unknown_flags = set(self.flags) - set(taxonomy.flags)
        if unknown_flags:
            raise ValueError(f"Flags desconocidos: {sorted(unknown_flags)}")
        if self.flags and not set(self.final_labels).intersection(
            taxonomy.damage_labels
        ):
            raise ValueError("Los flags requieren al menos una categoría final de daño")
        return self


class LLMAnnotationPayload(BaseModel):
    """Esquema estricto que se entrega a los proveedores generativos."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    coarse_labels: list[str]
    fine_labels: list[str]
    flags: list[str]
    needs_review: bool
    notes: str = Field(default="", max_length=400)
    score_confianza: float = Field(ge=0, le=1)
    justificacion: str = Field(default="", max_length=1200)
