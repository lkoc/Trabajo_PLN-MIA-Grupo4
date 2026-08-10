from __future__ import annotations

import platform
import warnings
from typing import Any

from .schemas import HardwareRecord

VALID_DEVICES = {"auto", "cuda", "rocm", "xpu", "cpu"}
HIGH_MEMORY_CUDA_MIN_BYTES = 39_000_000_000


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch no está instalado; use el extra de entrenamiento adecuado"
        ) from exc
    return torch


def available_backends() -> dict[str, bool]:
    try:
        torch = _torch()
    except RuntimeError:
        return {"cuda": False, "rocm": False, "xpu": False, "cpu": True}
    hip = bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
    cuda = bool(
        torch.cuda.is_available() and getattr(torch.version, "cuda", None) and not hip
    )
    xpu_module = getattr(torch, "xpu", None)
    xpu = bool(xpu_module and xpu_module.is_available())
    return {"cuda": cuda, "rocm": hip, "xpu": xpu, "cpu": True}


def resolve_device(requested: str = "auto") -> HardwareRecord:
    requested = requested.lower().strip()
    if requested not in VALID_DEVICES:
        raise ValueError(f"Dispositivo no válido: {requested}")
    availability = available_backends()
    fallback_reason = None
    if requested == "auto":
        backend = next(
            (name for name in ("cuda", "rocm", "xpu") if availability[name]), "cpu"
        )
    elif availability[requested]:
        backend = requested
    else:
        backend = "cpu"
        fallback_reason = f"backend_requested_but_unavailable:{requested}"
        warnings.warn(f"{requested} no está disponible; se usará CPU", RuntimeWarning)

    try:
        torch = _torch()
    except RuntimeError:
        return HardwareRecord(
            backend="cpu",
            requested=requested,
            device_name=platform.processor() or "CPU",
            fallback_reason=fallback_reason or "torch_not_installed",
        )

    if backend in {"cuda", "rocm"}:
        props = torch.cuda.get_device_properties(0)
        runtime = torch.version.hip if backend == "rocm" else torch.version.cuda
        name = torch.cuda.get_device_name(0)
        total = int(props.total_memory)
        dtype = (
            "bfloat16"
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            else "float16"
        )
    elif backend == "xpu":
        props = torch.xpu.get_device_properties(0)
        runtime = "xpu"
        name = torch.xpu.get_device_name(0)
        total = int(props.total_memory)
        dtype = "bfloat16"
    else:
        runtime = None
        name = platform.processor() or "CPU"
        total = None
        dtype = "float32"
    return HardwareRecord(
        backend=backend,
        requested=requested,
        device_name=name,
        torch_version=str(torch.__version__),
        runtime_version=str(runtime) if runtime else None,
        total_memory_bytes=total,
        dtype=dtype,
        fallback_reason=fallback_reason,
    )


def torch_device_name(record: HardwareRecord) -> str:
    if record.backend in {"cuda", "rocm"}:
        return "cuda"
    return record.backend


def high_memory_bf16_cuda(record: HardwareRecord) -> bool:
    """Detecta el perfil de 40 GB o más usado para A100/H100/H200.

    La decisión se basa en capacidades observables y no en un nombre comercial,
    de modo que también funciona con aceleradores equivalentes asignados por Colab.
    """

    return bool(
        getattr(record, "backend", None) == "cuda"
        and getattr(record, "dtype", None) == "bfloat16"
        and (getattr(record, "total_memory_bytes", None) or 0)
        >= HIGH_MEMORY_CUDA_MIN_BYTES
    )


def cuda_performance_profile(record: HardwareRecord) -> str:
    if high_memory_bf16_cuda(record):
        return "cuda_bf16_40gb_plus"
    if getattr(record, "backend", None) == "cuda":
        return "cuda_standard"
    return str(getattr(record, "backend", "unknown"))
