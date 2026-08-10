from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from moderacion_peru import experiments
from moderacion_peru.models import TrainingSpecification


def test_hf_validation_metrics_supports_gate_and_damage_cascade_heads():
    gate_truth = np.asarray([[0], [0], [1], [1]], dtype=np.int8)
    gate_logits = np.asarray([[-5.0], [-2.0], [2.0], [5.0]])
    damage_truth = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int8,
    )
    damage_logits = np.where(damage_truth == 1, 5.0, -5.0)

    gate_metrics = experiments._hf_validation_metrics(
        gate_logits,
        gate_truth,
        primary_output_count=1,
    )
    damage_metrics = experiments._hf_validation_metrics(
        damage_logits,
        damage_truth,
        primary_output_count=4,
    )

    assert gate_metrics["macro_auprc_damage"] == 1.0
    assert damage_metrics["macro_auprc_damage"] == 1.0
    assert "macro_auprc_five" not in gate_metrics
    assert "macro_auprc_five" not in damage_metrics


def test_public_hf_model_load_does_not_probe_for_an_implicit_token(monkeypatch):
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("tokenizer", source, kwargs))
            return SimpleNamespace(pad_token_id=0)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", source, kwargs))
            return SimpleNamespace(config=SimpleNamespace(pad_token_id=None))

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    transformers.AutoModelForSequenceClassification = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = TrainingSpecification(
        family="fixture",
        model_id="org/public-model",
        revision="fixed-revision",
    )

    experiments._build_hf_model(spec, ["SEGURO", "DANO"])

    assert [kind for kind, _source, _kwargs in calls] == ["tokenizer", "model"]
    assert all(kwargs["token"] is False for _kind, _source, kwargs in calls)


def test_fit_hf_keeps_label_mask_for_structured_loss(monkeypatch, tmp_path):
    captured_arguments = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured_arguments.update(kwargs)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(
                best_model_checkpoint=None,
                best_metric=None,
            )

        def train(self, resume_from_checkpoint=None):
            return SimpleNamespace(metrics={})

    class FakeEarlyStoppingCallback:
        def __init__(self, **kwargs):
            pass

    transformers = ModuleType("transformers")
    transformers.EarlyStoppingCallback = FakeEarlyStoppingCallback
    transformers.Trainer = FakeTrainer
    transformers.TrainingArguments = FakeTrainingArguments
    trainer_utils = ModuleType("transformers.trainer_utils")
    trainer_utils.get_last_checkpoint = lambda path: None
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", trainer_utils)

    def tokenizer(texts, **kwargs):
        return {
            "input_ids": [[1, 2] for _ in texts],
            "attention_mask": [[1, 1] for _ in texts],
        }

    rows = [{"text": "ejemplo"}]
    targets = np.asarray([[1, 0, 0, 0, 0]], dtype=np.float32)
    masks = np.ones_like(targets)
    spec = TrainingSpecification(family="fixture", model_id="fixture", epochs=1)
    hardware = SimpleNamespace(backend="cpu", dtype="float32")

    experiments._fit_hf(
        object(),
        tokenizer,
        rows,
        targets,
        masks,
        rows,
        targets,
        masks,
        spec,
        tmp_path / "training",
        hardware,
    )

    assert captured_arguments["remove_unused_columns"] is False
