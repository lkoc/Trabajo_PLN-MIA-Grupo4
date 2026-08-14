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

`03_02`–`03_06b` incluyen un backend Colab reproducible desde VS Code. El snapshot se transfiere comprimido y verificado y se copia a `/content`. Cada época terminada publica en Drive un checkpoint completo e inmutable de `Trainer`; al reiniciar se recupera el más nuevo verificable y se continúa con optimizador, scheduler y RNG. Los candidatos y métricas finales se publican automáticamente en dos ranuras redundantes. `03_07` también admite Colab, pero en CPU: monta Drive, restaura por manifiesto y SHA-256 los runs `03_01_working_v2_1`–`03_06b_working_v2_1` y guarda su comparación como un checkpoint propio. `03_06b` es opcional y solo entra si contiene un candidato completo, con validation común y test sellado. El cuaderno vigente de `03_01` conserva sus artefactos localmente; para restaurarlo en Colab debe existir la copia convencional `runs/03_01/03_01_working_v2_1` o una copia local bajo `modelos/v2`. Véase [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).

Active solo las campañas que quiera comparar. En `03_02`,
`RUN_MINILM_CONTEXT_SCREEN=True` crea candidatos 192/256 desde el MiniLM de 128
tokens usando una época, `1e-5` y validation fija. Tras escoger el contexto en
validation, ajuste `SELECTED_MINILM_CONTEXT` y active
`RUN_MINILM_SEED_CONFIRMATION`; junto con la semilla primaria se obtienen tres
semillas de entrenamiento sin cambiar las filas. `RUN_MINILM_FOCAL_ABLATION`
crea la comparación focal `gamma=2` separada de BCE.

`03_05` conserva el candidato Qwen-LoRA de 128 tokens con
`RUN_TRAINING=False` y permite `RUN_CONTINUATION_256=True`. En `03_06`, la ruta
recomendada es `RUN_STRUCTURED_LORA_SWEEP=True`: restaura la publicación
verificable de `03_05`, reutiliza su adaptador entrenable con optimizador nuevo
y produce tres candidatos de una época con penalizaciones `0`, `0.02` y `0.05`.
`RUN_LEGACY_FULL_TRAINING` queda apagado y preserva el experimento completo
histórico. En `03_06b`, el piloto diagnóstico no es elegible para `03_07`; el
candidato presupuestado solo entra si completó la validation común. Train y
validation usan una submuestra determinista `SEGURO`/daño 4:1. En `03_07`,
ejecute desde la primera celda con kernel Colab, confirme que
`CANDIDATE_PREFLIGHT_READY=True` y active primero `RUN_COMPARE_AND_FREEZE`; después
`RUN_TEST_ONCE` ejecuta una sola inferencia sobre todo el test natural.
`RUN_PUBLISH` continúa bloqueado hasta una aprobación posterior.
