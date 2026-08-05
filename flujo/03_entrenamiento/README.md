# Etapa 03 · Entrenamiento y evaluación

Ejecute `03_01`–`03_08` en orden. Todos consumen snapshots del contrato `moderacion_peru_5_salidas_v2` y deben compartir la misma partición agrupada por `video_id`.

La selección de familia, checkpoint, época y umbrales utiliza validation. Test se consulta después de congelar la decisión. Las métricas históricas de cuatro daños se mantienen separadas en `archivo/`.

Los encoders o backbones anteriores pueden servir de inicialización, pero la cabeza principal de cinco salidas se entrena de nuevo. En incrementos posteriores, Transformers y Qwen reanudan desde el checkpoint v2; el baseline SGD admite actualización incremental.

Consulte [`docs/HARDWARE.md`](../../docs/HARDWARE.md) antes de instalar PyTorch: una rueda CUDA no sustituye una rueda ROCm o XPU.

