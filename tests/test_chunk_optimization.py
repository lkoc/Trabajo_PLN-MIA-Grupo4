from __future__ import annotations

import json

import pytest

from moderacion_peru.chunk_optimization import (
    _bounded_neural_rows,
    activate_chunking_configuration,
    recommend_chunk_seconds,
    run_bounded_neural_chunk_comparison,
)
from moderacion_peru.incremental import DEFAULT_CHUNKING_CONFIGURATION


def _write(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_chunk_configuration_switch_archives_and_restores_exact_bytes(tmp_path):
    chunks = tmp_path / "datos/processed/chunks_v2.jsonl"
    dataset = tmp_path / "datos/model_ready/v2/dataset_5_salidas.jsonl"
    model = tmp_path / "modelos/v2/model.bin.fixture"
    _write(chunks, b'\n{"chunk_id":"30s"}\n')
    _write(dataset, b'\n{"dataset":"30s"}\n')
    _write(model, b"model-30s")

    initial = activate_chunking_configuration(tmp_path, DEFAULT_CHUNKING_CONFIGURATION)
    assert initial["status"] == "already_active_noop"

    config_25 = {**DEFAULT_CHUNKING_CONFIGURATION, "max_seconds": 25}
    switched = activate_chunking_configuration(tmp_path, config_25, source="test")
    assert switched["status"] == "activated_empty"
    assert not chunks.exists()
    _write(chunks, b'\n{"chunk_id":"25s"}\n')
    _write(dataset, b'\n{"dataset":"25s"}\n')

    restored = activate_chunking_configuration(tmp_path, DEFAULT_CHUNKING_CONFIGURATION, source="test")
    assert restored["status"] == "restored"
    assert chunks.read_bytes() == b'\n{"chunk_id":"30s"}\n'
    assert dataset.read_bytes() == b'\n{"dataset":"30s"}\n'
    assert model.read_bytes() == b"model-30s"

    restored_25 = activate_chunking_configuration(tmp_path, config_25, source="test")
    assert restored_25["status"] == "restored"
    assert chunks.read_bytes() == b'\n{"chunk_id":"25s"}\n'
    assert dataset.read_bytes() == b'\n{"dataset":"25s"}\n'
    state = json.loads((tmp_path / "datos/processed/chunking_active.json").read_text())
    assert state["configuration"]["max_seconds"] == 25.0


def test_recommendation_prefers_cheapest_length_inside_validation_tolerance():
    comparisons = [
        {"chunk_seconds": 15, "validation_ap_macro_damage": 0.70, "compute_proxy": 500},
        {"chunk_seconds": 25, "validation_ap_macro_damage": 0.75, "compute_proxy": 400},
        {"chunk_seconds": 35, "validation_ap_macro_damage": 0.735, "compute_proxy": 250},
    ]
    recommendation = recommend_chunk_seconds(comparisons, max_validation_ap_drop=0.02)
    assert recommendation["recommended_seconds"] == 35.0
    assert recommendation["test_used_for_selection"] is False


def test_corrupt_restore_is_rejected_before_active_state_moves(tmp_path):
    chunks = tmp_path / "datos/processed/chunks_v2.jsonl"
    _write(chunks, b"default-30")
    activate_chunking_configuration(tmp_path, DEFAULT_CHUNKING_CONFIGURATION)
    config_25 = {**DEFAULT_CHUNKING_CONFIGURATION, "max_seconds": 25}
    activate_chunking_configuration(tmp_path, config_25)
    _write(chunks, b"active-25")
    archived_default = next(
        (tmp_path / "archivo/chunking_configurations").glob(
            "*/state/datos/processed/chunks_v2.jsonl"
        )
    )
    archived_default.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="alterado"):
        activate_chunking_configuration(tmp_path, DEFAULT_CHUNKING_CONFIGURATION)

    assert chunks.read_bytes() == b"active-25"
    active = json.loads((tmp_path / "datos/processed/chunking_active.json").read_text())
    assert active["configuration"]["max_seconds"] == 25.0


def test_bounded_neural_rows_are_deterministic_and_cover_labels():
    rows = [
        {
            "chunk_id": f"chunk-{index}",
            "split": "validation",
            "coarse_labels": [label],
            "text": f"texto {index}",
        }
        for index, label in enumerate(
            [
                "SEGURO",
                "RACISMO_DISCRIMINACION",
                "ATAQUE_POR_GENERO_IDENTIDAD",
                "ACOSO_AMENAZA",
                "CONTENIDO_SEXUAL",
            ]
        )
    ]
    rows.append(
        {
            "chunk_id": "train-only",
            "split": "train",
            "coarse_labels": ["SEGURO"],
            "text": "fuera del split",
        }
    )

    selected = _bounded_neural_rows(rows, "validation", 5, seed=20260806)
    repeated = _bounded_neural_rows(rows, "validation", 5, seed=20260806)

    assert [row["chunk_id"] for row in selected] == [
        row["chunk_id"] for row in repeated
    ]
    assert {row["coarse_labels"][0] for row in selected} == {
        "SEGURO",
        "RACISMO_DISCRIMINACION",
        "ATAQUE_POR_GENERO_IDENTIDAD",
        "ACOSO_AMENAZA",
        "CONTENIDO_SEXUAL",
    }


def test_bounded_neural_comparison_requires_toggle_and_materialized_cohort(tmp_path):
    with pytest.raises(ValueError, match="al menos uno"):
        run_bounded_neural_chunk_comparison(
            tmp_path,
            run_hf=False,
            run_ollama=False,
        )

    with pytest.raises(FileNotFoundError, match="smoke test CPU"):
        run_bounded_neural_chunk_comparison(
            tmp_path,
            candidate_seconds=(20,),
            run_hf=True,
            run_ollama=False,
        )
