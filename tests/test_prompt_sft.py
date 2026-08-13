from __future__ import annotations

from dataclasses import dataclass

from moderacion_peru.prompt_sft import _configure_lora_gradient_checkpointing


@dataclass
class _Parameter:
    requires_grad: bool
    size: int

    def numel(self) -> int:
        return self.size


class _ModernLoraModel:
    def __init__(self) -> None:
        self.input_grad_enabled = False
        self.checkpointing_kwargs = None

    def named_parameters(self):
        return [
            ("base.weight", _Parameter(False, 100)),
            ("layer.lora_A.default.weight", _Parameter(True, 8)),
            ("layer.lora_B.default.weight", _Parameter(True, 8)),
        ]

    def enable_input_require_grads(self) -> None:
        self.input_grad_enabled = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.checkpointing_kwargs = kwargs


def test_prompt_sft_preserves_lora_graph_with_non_reentrant_checkpointing():
    model = _ModernLoraModel()

    diagnostics = _configure_lora_gradient_checkpointing(model)

    assert model.input_grad_enabled is True
    assert model.checkpointing_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert diagnostics["checkpointing_mode"] == "non_reentrant"
    assert diagnostics["trainable_parameter_tensors"] == 2
    assert diagnostics["trainable_parameters"] == 16
