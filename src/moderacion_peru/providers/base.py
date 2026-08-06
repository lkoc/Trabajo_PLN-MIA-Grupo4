from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import AnnotationRecord, LLMAnnotationPayload
from ..taxonomy import TaxonomyContract, load_taxonomy


class ProviderError(RuntimeError):
    pass


SYSTEM_PROMPT = """Eres un anotador de fragmentos de subtítulos peruanos. Aplica exclusivamente el contrato entregado. SEGURO es una categoría explícita y nunca puede coexistir con daño. Los cuatro daños pueden coexistir. Las etiquetas gruesas deben coincidir exactamente con las derivadas de las etiquetas finas. SEGURO exige `seguro` o `seguro_ironia_marcada` como estado fino. `score_confianza` debe ser un número decimal entre 0.0 y 1.0. Si falta contexto para decidir, devuelve coarse_labels vacío, needs_review=true y contexto_necesario; no fuerces SEGURO. Distingue discurso citado, condenado, informativo o ficticio del ataque respaldado por el hablante. Conserva exactamente el chunk_id recibido y devuelve únicamente JSON conforme al esquema, sin razonamiento ni texto adicional."""


def taxonomy_prompt(taxonomy: TaxonomyContract) -> str:
    blocks = []
    for label in taxonomy.target_labels:
        category = taxonomy.categories[label]
        blocks.append(
            f"{label}: {category.definition}\n"
            f"Incluye: {'; '.join(category.include)}.\n"
            f"Excluye: {'; '.join(category.exclude)}.\n"
            f"Etiquetas finas: {', '.join(category.fine_labels)}."
        )
    return "\n\n".join(blocks)


def normalize_payload(
    payload: dict[str, Any],
    *,
    text: str,
    source: str,
    annotator_type: str,
    model: str,
    video_id: str | None = None,
    chunk_metadata: dict[str, Any] | None = None,
    source_record_sha256: str | None = None,
    prompt_sha256: str | None = None,
    taxonomy: TaxonomyContract | None = None,
) -> AnnotationRecord:
    contract = taxonomy or load_taxonomy()
    parsed = LLMAnnotationPayload.model_validate(payload)
    fine = tuple(dict.fromkeys(parsed.fine_labels))
    coarse = contract.normalize_categories(parsed.coarse_labels)
    if fine:
        derived = contract.derive_categories(fine)
        if coarse != derived:
            raise ProviderError(
                f"Inconsistencia fina→gruesa para {parsed.chunk_id}: {fine} -> {derived}, recibió {coarse}"
            )
    if coarse == (contract.safe_label,) and not set(fine).intersection(
        contract.categories[contract.safe_label].fine_labels
    ):
        raise ProviderError("SEGURO requiere un estado fino seguro explícito")
    needs_review = parsed.needs_review or not coarse
    metadata = chunk_metadata or {}
    return AnnotationRecord(
        chunk_id=parsed.chunk_id,
        video_id=video_id,
        start_seconds=metadata.get("start_seconds"),
        end_seconds=metadata.get("end_seconds"),
        video_title=metadata.get("video_title") or metadata.get("title"),
        channel_title=metadata.get("channel_title") or metadata.get("channel"),
        source_url=metadata.get("source_url") or metadata.get("url"),
        cohort=metadata.get("cohort"),
        text=text,
        coarse_labels=list(coarse),
        fine_labels=list(fine),
        flags=list(dict.fromkeys(parsed.flags)),
        needs_review=needs_review,
        training_eligible=not needs_review,
        decision_status="needs_review" if needs_review else "resolved",
        score_confianza=parsed.score_confianza,
        notes=parsed.notes,
        justification=parsed.justificacion,
        label_source=source,
        annotator_type=annotator_type,  # type: ignore[arg-type]
        annotator_model=model,
        prompt_sha256=prompt_sha256,
        source_record_sha256=source_record_sha256,
    )


class AnnotationProvider(ABC):
    def __init__(self, model: str, taxonomy: TaxonomyContract | None = None) -> None:
        self.model = model
        self.taxonomy = taxonomy or load_taxonomy()

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def annotate(self, chunk: dict[str, Any]) -> AnnotationRecord:
        raise NotImplementedError
