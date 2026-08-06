# Moderación semiautomática de contenido peruano

Este proyecto reúne subtítulos públicos de YouTube peruano, los limpia y segmenta, los etiqueta con modelos locales o remotos y permite revisar las decisiones antes de entrenar clasificadores. Su finalidad es priorizar revisión humana; no elimina contenido ni sanciona usuarios.

## Categorías

El contrato `moderacion_peru_5_salidas_v2`, taxonomía `2.1.0`, entrena cinco salidas:

- `SEGURO`;
- `RACISMO_DISCRIMINACION`;
- `ATAQUE_POR_GENERO_IDENTIDAD`;
- `ACOSO_AMENAZA`;
- `CONTENIDO_SEXUAL`.

`SEGURO` es una categoría aprendida y no puede coexistir con daño. Los cuatro daños sí pueden coexistir. Los casos sin contexto suficiente se difieren y no entran al entrenamiento.

La definición, justificación bibliográfica y límites de cada daño están en [`docs/MATRIZ_EVIDENCIA_TAXONOMIA.md`](docs/MATRIZ_EVIDENCIA_TAXONOMIA.md). La evidencia peruana contextualiza el contrato, pero no se presenta como validación jurídica, experta ni de prevalencia nacional.

## Flujo

1. [`flujo/01_datos`](flujo/01_datos/README.md): reutiliza videos/subtítulos ya procesados y agrega solo material nuevo.
2. [`flujo/02_etiquetado`](flujo/02_etiquetado/README.md): Ollama local, proveedor remoto opcional y validación humana.
3. [`flujo/03_entrenamiento`](flujo/03_entrenamiento/README.md): modelos clásicos, Transformers, Qwen y comparación común.
4. [`flujo/04_produccion`](flujo/04_produccion/README.md): demostrador local en modo sombra.

Instalación mínima:

```powershell
python -m pip install -e ".[dev]"
modperu preflight
```

Los extras `datos`, `etiquetado` y `entrenamiento` se instalan solo cuando se necesitan. Las instrucciones de CUDA, ROCm, XPU y CPU están en [`docs/HARDWARE.md`](docs/HARDWARE.md). Los cuadernos con carga neuronal pueden usar una GPU L4 de Colab directamente desde VS Code mediante el flujo Drive-only descrito en [`docs/COLAB_L4.md`](docs/COLAB_L4.md).

Cada cuaderno activo cita en estilo IEEE las fuentes metodológicas, el software y los checkpoints que realmente usa, y termina con su propia celda `Referencias`. La cobertura y los controles automáticos se documentan en [`docs/AUDITORIA_CITAS_CUADERNOS.md`](docs/AUDITORIA_CITAS_CUADERNOS.md).

El orden íntegro de los 17 cuadernos, sus interruptores y los bucles incrementales está en [`docs/ORDEN_EJECUCION.md`](docs/ORDEN_EJECUCION.md). El puente `02_05` reincorpora eventos humanos y crea snapshots; `03_01`–`03_06` generan candidatos completos y `03_07` publica el registro que consume producción.

Comandos equivalentes para los puentes principales:

```powershell
modperu prepare-training
modperu train classical
modperu train flat --device auto
modperu publish-model
```

## Incrementos futuros

El flujo identifica videos por `video_id`, transcripciones por SHA-256 y chunks por un ID determinista. Una nueva corrida omite todo lo ya procesado, añade únicamente videos o subtítulos nuevos y reanuda el etiquetado por `chunk_id`. Los modelos neuronales pueden continuar una interrupción o inicializarse desde un checkpoint compatible anterior usando un snapshot que combina los datos previos y el lote nuevo. Datos y configuración idénticos producen un no-op verificable, no una reescritura silenciosa.

Los resultados ejecutados antes de esta reorganización se conservan en [`archivo`](archivo/README.md). Sus métricas corresponden a contratos anteriores y no se atribuyen al nuevo entrenamiento de cinco salidas.
