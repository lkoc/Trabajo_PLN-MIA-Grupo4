from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from moderacion_peru.prompt_sft import (
    _configure_lora_gradient_checkpointing,
    _cuda_memory_preflight,
    _json_target,
    compile_operational_prompt_capsule,
    train_prompt_conditioned_sft,
)


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


def test_prompt_sft_can_disable_checkpointing_for_short_a100_profile():
    model = _ModernLoraModel()

    diagnostics = _configure_lora_gradient_checkpointing(model, enabled=False)

    assert model.input_grad_enabled is False
    assert model.checkpointing_kwargs is None
    assert diagnostics["checkpointing_mode"] == "disabled"
    assert diagnostics["trainable_parameters"] == 16


class _CudaMemory:
    def __init__(self, free_bytes: int) -> None:
        self.free_bytes = free_bytes
        self.empty_cache_called = False

    def empty_cache(self) -> None:
        self.empty_cache_called = True

    def mem_get_info(self):
        return self.free_bytes, 40_000_000_000

    def memory_allocated(self) -> int:
        return 12_000_000_000

    def memory_reserved(self) -> int:
        return 13_000_000_000


def test_prompt_sft_rejects_a_contaminated_cuda_runtime_before_model_load():
    cuda = _CudaMemory(5_000_000)
    torch_module = type("Torch", (), {"cuda": cuda})()

    with pytest.raises(RuntimeError, match="Reinicie por completo el runtime"):
        _cuda_memory_preflight(torch_module, minimum_free_bytes=24_000_000_000)

    assert cuda.empty_cache_called is True


def test_prompt_sft_accepts_a_clean_cuda_runtime_and_records_memory():
    cuda = _CudaMemory(31_000_000_000)
    torch_module = type("Torch", (), {"cuda": cuda})()

    diagnostics = _cuda_memory_preflight(
        torch_module, minimum_free_bytes=24_000_000_000
    )

    assert diagnostics["free_bytes"] == 31_000_000_000
    assert diagnostics["allocated_by_torch_bytes"] == 12_000_000_000


def test_prompt_capsule_keeps_contract_and_strict_response_format():
    prompt = """Versión del prompt: 3.2.0
Contrato y taxonomía: contrato
## Tarea y salidas permitidas
SEGURO o DAÑO.
## Principio rector: clasificar el evento de habla, no la palabra
Resuelva atribución.
## Jerarquía obligatoria de decisión
Regla larga.
## Consistencia obligatoria
SEGURO es excluyente.
## Formato de respuesta
Devuelve únicamente JSON.
"""

    capsule = compile_operational_prompt_capsule(prompt, max_chars=1000)

    assert len(capsule) <= 1000
    assert "## Jerarquía obligatoria de decisión" in capsule
    assert "## Consistencia obligatoria" in capsule
    assert "## Formato de respuesta" in capsule
    assert "Devuelve únicamente JSON" in capsule


def test_prompt_capsule_rejects_missing_required_sections():
    with pytest.raises(ValueError, match="secciones obligatorias"):
        compile_operational_prompt_capsule("## Tarea y salidas permitidas\nSEGURO")


def test_prompt_target_uses_operational_json_field_names():
    target = _json_target(
        {
            "chunk_id": "chunk-1",
            "coarse_labels": ["SEGURO"],
            "fine_labels": ["seguro"],
            "flags_reference_only": [],
        }
    )

    assert '"chunk_id":"chunk-1"' in target
    assert '"score_confianza":1.0' in target
    assert '"justification":' in target
    assert '"confidence":' not in target
    assert '"reasoning":' not in target


def test_prompt_sft_training_function_contains_complete_pipeline():
    source = inspect.getsource(train_prompt_conditioned_sft)

    assert "configuration =" in source
    assert "trainer = Trainer(" in source
    assert 'return {"status": "trained"' in source
