from __future__ import annotations

import json
from pathlib import Path

import pytest

from moderacion_peru.io import read_jsonl
from moderacion_peru.toy_prompt_sft import (
    TOY_LABEL_TOTALS,
    build_toy_prompt_dataset,
    compute_toy_metrics,
    toy_expected_distribution,
    validate_toy_dataset,
)


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = toy_expected_distribution()
    for split, distribution in expected.items():
        for label, required in distribution.items():
            for index in range(required + 5):
                video_id = f"{split}-{label}-{index}"
                for chunk in range(2):
                    rows.append(
                        {
                            "chunk_id": f"{video_id}-{chunk}",
                            "video_id": video_id,
                            "channel_id": "canal",
                            "text": f"texto {label} {index} alternativa {chunk}",
                            "coarse_labels": [label],
                            "fine_labels": [],
                            "flags_reference_only": [],
                            "split": split,
                            "training_eligible": True,
                        }
                    )
    return rows


def test_toy_distribution_normalizes_80_20_20_as_four_one_one():
    distribution = toy_expected_distribution()

    assert {split: sum(values.values()) for split, values in distribution.items()} == {
        "train": 800,
        "validation": 200,
        "test": 200,
    }
    assert distribution["train"]["SEGURO"] == 640
    assert distribution["validation"]["SEGURO"] == 160
    assert distribution["test"]["SEGURO"] == 160
    for label in set(TOY_LABEL_TOTALS) - {"SEGURO"}:
        assert [distribution[split][label] for split in distribution] == [40, 10, 10]


def test_toy_builder_is_stratified_video_disjoint_and_reproducible(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for row in _source_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    created = build_toy_prompt_dataset(source, first, seed=123)
    repeated = build_toy_prompt_dataset(source, second, seed=123)
    first_rows = list(read_jsonl(first))
    second_rows = list(read_jsonl(second))

    assert created["rows"] == 1200
    assert repeated["rows"] == 1200
    assert [row["chunk_id"] for row in first_rows] == [
        row["chunk_id"] for row in second_rows
    ]
    assert validate_toy_dataset(first_rows)["video_disjoint"] is True
    assert len({row["video_id"] for row in first_rows}) == 1200


def test_strict_metric_counts_invalid_json_as_an_error():
    labels = list(TOY_LABEL_TOTALS)
    records = [
        {
            "expected": label,
            "predicted": label,
            "json_schema_valid": True,
            "correct": True,
        }
        for label in labels
    ]
    records.append(
        {
            "expected": "SEGURO",
            "predicted": None,
            "json_schema_valid": False,
            "correct": False,
        }
    )

    metrics = compute_toy_metrics(records)

    assert metrics["invalid_json_predictions"] == 1
    assert metrics["json_schema_valid_rate"] == pytest.approx(5 / 6)
    assert metrics["strict_accuracy"] == pytest.approx(5 / 6)
    assert metrics["strict_macro_f1"] < 1.0


def test_toy_training_source_declares_hard_json_constraint_and_no_candidate():
    source = (
        Path(__file__).parents[1] / "src/moderacion_peru/toy_prompt_sft.py"
    ).read_text(encoding="utf-8")

    assert "prefix_allowed_tokens_fn=allowed_tokens" in source
    assert "token_trie_exactly_five_valid_json_objects_per_row" in source
    assert '"eligible_for_03_07": False' in source
    assert source.count("candidate.json") == 1
