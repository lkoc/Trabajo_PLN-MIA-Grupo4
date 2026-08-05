from __future__ import annotations

import pytest
from pydantic import ValidationError

from moderacion_peru.schemas import AnnotationRecord, ReviewEvent
from moderacion_peru.taxonomy import load_taxonomy


def annotation(**overrides):
    payload = {
        "chunk_id": "vid_1",
        "text": "Texto neutral",
        "coarse_labels": ["SEGURO"],
        "fine_labels": ["seguro"],
        "flags": [],
        "label_source": "fixture",
        "annotator_type": "human",
    }
    payload.update(overrides)
    return AnnotationRecord(**payload)


def test_contract_has_five_trained_outputs():
    taxonomy = load_taxonomy()
    assert taxonomy.target_labels == (
        "SEGURO",
        "RACISMO_DISCRIMINACION",
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
        "CONTENIDO_SEXUAL",
    )
    assert len(taxonomy.fine_labels) == 14
    assert len(taxonomy.flags) == 3
    assert taxonomy.categories["ATAQUE_POR_GENERO_IDENTIDAD"].display_name == (
        "Ataque por género e identidad"
    )


def test_safe_is_explicit_and_exclusive():
    assert annotation().training_eligible
    with pytest.raises(ValidationError, match="mutuamente excluyente"):
        annotation(coarse_labels=["SEGURO", "ACOSO_AMENAZA"])


def test_empty_decision_is_review_not_safe():
    record = annotation(
        coarse_labels=[],
        fine_labels=[],
        flags=["contexto_necesario"],
        needs_review=True,
        training_eligible=False,
    )
    assert record.coarse_labels == []
    assert record.decision_status == "needs_review"


def test_empty_trainable_decision_is_invalid():
    with pytest.raises(ValidationError, match="sin categoría"):
        annotation(coarse_labels=[], fine_labels=[])


def test_fine_mapping_merges_harassment_and_threat():
    taxonomy = load_taxonomy()
    assert taxonomy.derive_categories(["acoso_personal", "amenaza_directa"]) == (
        "ACOSO_AMENAZA",
    )


def test_gender_identity_attack_is_canonical_and_legacy_name_migrates():
    taxonomy = load_taxonomy()
    assert "ATAQUE_POR_GENERO_IDENTIDAD" in taxonomy.target_labels
    assert "ACOSO_GENERO_IDENTIDAD" not in taxonomy.target_labels
    assert taxonomy.derive_categories(["misoginia_acoso_genero"]) == (
        "ATAQUE_POR_GENERO_IDENTIDAD",
    )
    assert taxonomy.migrate_legacy_categories(["ACOSO_GENERO_IDENTIDAD"]) == (
        "ATAQUE_POR_GENERO_IDENTIDAD",
    )


def test_deferred_review_has_no_final_labels():
    event = ReviewEvent(
        event_id="e1",
        chunk_id="c1",
        action="defer",
        reviewer="R1",
    )
    assert event.final_labels == []
