# Compatibilidad de hardware

El parámetro común es `device=auto|cuda|rocm|xpu|cpu`.

| Hardware | Backend detectado | Instalación de PyTorch | Observación |
|---|---|---|---|
| NVIDIA | `cuda` | rueda oficial CUDA apropiada al controlador | AMP FP16/BF16 según capacidad |
| AMD | `rocm` | rueda/imagen ROCm compatible con GPU y SO | PyTorch expone el dispositivo como `cuda`, pero `torch.version.hip` identifica ROCm |
| Intel Arc/Core Ultra | `xpu` | rueda oficial XPU | disponible en esta máquina; validar memoria antes de modelos grandes |
| Sin acelerador | `cpu` | rueda CPU | ruta obligatoria para smoke tests |

El código distingue CUDA de ROCm antes de elegir kernels o registrar hardware. Si se solicita un backend ausente, se muestra una advertencia y se usa CPU; el manifiesto conserva el motivo.

Ollama administra su propia aceleración. Vulkan es experimental y no se activa automáticamente. El preflight registra versión, modelos instalados y disponibilidad, pero no descarga modelos ni inicia una corrida.

No se incluye `torch` como dependencia universal en `pyproject.toml`: cada máquina debe instalar la distribución oficial correspondiente antes del extra `entrenamiento`. `bitsandbytes` es opcional y solo se usa después de un smoke test del backend.

Fuentes técnicas: [PyTorch HIP/ROCm](https://docs.pytorch.org/docs/main/notes/hip.html), [PyTorch XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html), [Ollama hardware](https://docs.ollama.com/gpu).

