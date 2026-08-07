# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 03 · Entrenamiento y evaluación

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

Ejecute `03_01`–`03_08` en orden. Todos consumen snapshots del contrato `moderacion_peru_5_salidas_v2`, taxonomía `2.1.0`, y deben compartir la misma partición agrupada por `video_id`.

La selección de familia, checkpoint, época y umbrales utiliza validation. Test se consulta después de congelar la decisión. Las métricas históricas de cuatro daños se mantienen separadas en `archivo/`.

Los cuadernos ya no se limitan a construir la arquitectura: cada rama ejecuta `fit → calibración de cinco umbrales en validation → evaluación final en test → checkpoint/manifiesto → candidate.json`. `03_07` compara solo candidatos del SHA-256 activo y publica `modelos/registro_modelos_5_salidas.json`.

Los encoders o backbones anteriores pueden servir de inicialización, pero la primera cabeza principal de cinco salidas se entrena de nuevo. En incrementos posteriores, Transformers y Qwen reanudan una interrupción del mismo run o usan el checkpoint compatible anterior como *warm start*, siempre entrenando con datos anteriores+nuevos. Una firma idéntica produce `status="noop"`.

Consulte [`docs/HARDWARE.md`](../../docs/HARDWARE.md) antes de instalar PyTorch: una rueda CUDA no sustituye una rueda ROCm o XPU.

`03_02`–`03_06` incluyen un backend Colab L4 reproducible desde VS Code. El snapshot se transfiere comprimido y verificado, se copia a `/content` y los checkpoints se publican de forma atómica. `03_01`, `03_07` y `03_08` permanecen locales salvo que sea necesario regenerar inferencias. Véase [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).

Active `RUN_TRAINING=True` en cada familia que quiera comparar y `RUN_PUBLISH=True` en `03_07`. No es obligatorio entrenarlas todas: basta un candidato completo para un registro operativo, aunque una comparación académica profunda debe ejecutar las ramas declaradas.
