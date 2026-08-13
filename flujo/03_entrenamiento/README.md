# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 03 · Entrenamiento y evaluación

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

Ejecute primero `03_08`, después las ramas `03_01`–`03_06b` que quiera comparar y al final `03_07`. Todos consumen snapshots del contrato `moderacion_peru_5_salidas_v2`, taxonomía `2.1.0`, y deben compartir la misma partición agrupada por `video_id`.

La selección de familia, checkpoint, época y umbrales utiliza validation. Test se consulta después de congelar la decisión. Las métricas históricas de cuatro daños se mantienen separadas en `archivo/`.

Cada rama ejecuta `fit → calibración/evaluación en validation → checkpoint/manifiesto → candidate.json`; ninguna abre test. `03_07` compara individuos y ensembles del mismo SHA, congela la decisión y deja en interruptores separados la apertura única de test y una publicación que permanece bloqueada. Las ramas compatibles usan 5+14+3 salidas y máscaras observadas.

En ejecución local, `03_01` materializa TF–IDF una vez por variante y reutiliza la misma matriz dispersa para sus cinco estimadores; las 22 cabezas se procesan con cuatro hilos compartiendo memoria. `03_07` usa cuatro hilos para el bootstrap agrupado por video, con semillas independientes por réplica para conservar resultados idénticos entre ejecución serial y paralela. La inferencia de miembros del ensemble sigue secuencial para controlar RAM/VRAM. Cada candidato y el reporte final guardan tiempos por etapa.

Los encoders o backbones anteriores pueden servir de inicialización, pero la primera cabeza principal de cinco salidas se entrena de nuevo. En incrementos posteriores, Transformers y Qwen reanudan una interrupción del mismo run o usan el checkpoint compatible anterior como *warm start*, siempre entrenando con datos anteriores+nuevos. Una firma idéntica produce `status="noop"`.

Consulte [`docs/HARDWARE.md`](../../docs/HARDWARE.md) antes de instalar PyTorch: una rueda CUDA no sustituye una rueda ROCm o XPU.

`03_02`–`03_06b` incluyen un backend Colab L4 reproducible desde VS Code. El snapshot se transfiere comprimido y verificado y se copia a `/content`. Cada época terminada publica en Drive un checkpoint completo e inmutable de `Trainer`; al reiniciar se recupera el más nuevo verificable y se continúa con optimizador, scheduler y RNG. Los candidatos y métricas finales se publican automáticamente en dos ranuras redundantes. `03_01`, `03_07` y `03_08` permanecen locales salvo que sea necesario regenerar inferencias. Véase [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).

Active `RUN_TRAINING=True` en cada clasificador que quiera comparar. `03_05` es la excepción cuando ya existe el candidato Qwen-LoRA de 128 tokens: mantenga `RUN_TRAINING=False` y ejecute `RUN_CONTINUATION_256=True`. Ese bloque verifica el candidato padre y su manifiesto, continúa sus pesos LoRA con contexto de 256 tokens y un optimizador nuevo, y escribe un `candidate.json` independiente; no reemplaza el modelo de 128. Ambos conservan test sellado y `03_07` decide entre ellos usando las mismas filas de validation. En `03_06b`, ejecute primero `RUN_PILOT=True`; el piloto no es elegible para `03_07`. Train y validation usan una submuestra determinista `SEGURO`/daño 4:1. En `03_07`, active primero `RUN_COMPARE_AND_FREEZE`; después `RUN_TEST_ONCE` ejecuta una sola inferencia sobre todo el test natural. El informe presenta esa evaluación como principal y deriva de las mismas predicciones una vista secundaria 4:1. `RUN_PUBLISH` no publica: lanza un bloqueo hasta una aprobación posterior.
