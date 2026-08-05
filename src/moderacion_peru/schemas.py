from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .taxonomy import load_taxonomy


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationRecord(BaseModel):
    """Contrato canónico de una anotación del flujo v2."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0.0"
    taxonomy_version: str = "2.0.0"
    chunk_id: str = Field(min_length=1)
    video_id: str | None = None
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
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_semantics(self) -> "AnnotationRecord":
        taxonomy = load_taxonomy()
        self.coarse_labels = list(taxonomy.normalize_categories(self.coarse_labels))
        unknown_fine = set(self.fine_labels) - set(taxonomy.fine_labels)
        unknown_flags = set(self.flags) - set(taxonomy.flags)
        if unknown_fine:
            raise ValueError(f"Etiquetas finas desconocidas: {sorted(unknown_fine)}")
        if unknown_flags:
            raise ValueError(f"Flags desconocidos: {sorted(unknown_flags)}")
        if not self.coarse_labels:
            if not self.needs_review or self.training_eligible:
                raise ValueError(
                    "Una anotación sin categoría debe requerir revisión y no ser entrenable"
                )
        if self.needs_review:
            self.decision_status = "needs_review"
        elif self.decision_status == "needs_review":
            raise ValueError("decision_status=needs_review exige needs_review=true")
        if self.decision_status == "excluded" and self.training_eligible:
            raise ValueError("Una anotación excluida no puede ser entrenable")
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

    schema_version: str = "2.0.0"
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

    schema_version: str = "2.0.0"
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
    status: Literal["candidate", "validated", "shadow_only", "archived"] = "candidate"

    @model_validator(mode="after")
    def validate_targets(self) -> "ModelRegistryEntry":
        taxonomy = load_taxonomy()
        if tuple(self.target_labels) != taxonomy.target_labels:
            raise ValueError("El registro no usa las cinco salidas canónicas en orden")
        if set(self.thresholds) != set(taxonomy.target_labels):
            raise ValueError("Debe existir un umbral para cada salida entrenada")
        return self


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0.0"
    event_id: str
    chunk_id: str
    action: Literal["accept", "modify", "reject", "defer"]
    proposed_labels: list[str] = Field(default_factory=list)
    final_labels: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    reviewer: str = Field(min_length=1)
    model_id: str | None = None
    taxonomy_version: str = "2.0.0"
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_review(self) -> "ReviewEvent":
        taxonomy = load_taxonomy()
        self.proposed_labels = list(taxonomy.normalize_categories(self.proposed_labels))
        self.final_labels = list(taxonomy.normalize_categories(self.final_labels))
        if self.action == "defer" and self.final_labels:
            raise ValueError("Una decisión diferida no puede tener categorías finales")
        if self.action in {"accept", "modify"} and not self.final_labels:
            raise ValueError("Aceptar o modificar requiere una categoría final explícita")
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

