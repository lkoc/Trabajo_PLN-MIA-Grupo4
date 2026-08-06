# Contratos de datos v2.1

## Video y transcripción

Una transcripción canónica contiene `video_id`, URL, canal, fuente de subtítulo, segmentos temporales y `transcript_sha256`. La presencia del `video_id` impide una nueva descarga; un archivo de caché válido se reutiliza antes de consultar la red.

## Chunk

`chunk_id` se deriva de versión del troceador, firma de configuración, `video_id`, tiempos y texto normalizado. La firma cubre `max_seconds`, límites de caracteres y solapamiento. Cambiar una regla crea nuevos IDs y una nueva versión; no modifica los chunks anteriores. Las filas 2.1 sin firma equivalen explícitamente al contrato histórico de 30 s.

`config/chunking.json` declara la configuración deseada y
`datos/processed/chunking_active.json` registra la firma local activa. Una
transición mueve todos los derivados gestionados a
`archivo/chunking_configurations/<firma>/state/`, junto con un manifiesto de
hashes. La restauración rechaza un archivo alterado o una colisión; nunca mueve
transcripciones raw ni candidatos.

## Anotación

Campos mínimos: `chunk_id`, `video_id`, `text`, `coarse_labels`, `fine_labels`, `flags`, `needs_review`, `training_eligible`, fuente, modelo, prompt y versiones. `video_id` puede faltar en propuestas históricas, pero `02_05` debe recuperarlo del chunk fuente antes de crear datos entrenables; nunca se parte `chunk_id` para deducirlo. Invariantes:

- `SEGURO` o uno/más daños, nunca ambos;
- una lista vacía solo es válida con revisión y exclusión temporal del entrenamiento;
- los flags no son categorías principales;
- toda salida conserva procedencia.

## Evento y reconciliación humana

`ReviewEvent` es append-only y registra propuesta, decisión, acción, flags, modelo y revisor pseudonimizado. Para cada chunk se aplica el evento más reciente por `(created_at, event_id)`. `accept` y `modify` producen una decisión humana resuelta; `defer` queda fuera de entrenamiento y no crea una sexta clase; `reject` queda explícitamente excluido. El consolidado LLM y los eventos originales nunca se sobrescriben.

## Snapshot de entrenamiento

`ModelReadyRecord` conserva `chunk_id`, `video_id`, texto, categorías canónicas, señales de referencia, fuente, peso, campaña, procedencia histórica y `split`. Solo contiene decisiones resueltas y entrenables. La partición es estable por `video_id`; un video nunca cruza train, validation y test. Un incremento materializa otro snapshot y conserva el anterior.

El snapshot v2.1 usa `ATAQUE_POR_GENERO_IDENTIDAD` como objetivo. El identificador anterior solo puede aparecer en `legacy_coarse_labels`, nunca en `coarse_labels`. `modperu validate ruta.jsonl --kind model-ready` comprueba las categorías, la exclusividad de `SEGURO`, la procedencia migrada, los flags y el split. Cada snapshot vive en `datos/model_ready/v2/snapshots/<snapshot_id>/`; `dataset_5_salidas.jsonl` es una vista de conveniencia al último snapshot y solo cambia si cambia su contenido.

## Registro de modelos

Cada experimento completo escribe `candidate.json`, `metrics.json`, predicciones de validation/test, cinco umbrales, un bundle de inferencia y `checkpoint_manifest.json` con SHA-256 de todos los archivos requeridos. La firma combina dataset, configuración, taxonomía y versión del motor; repetirla es no-op.

Un modelo activo debe declarar cinco scores y cinco umbrales, el contrato exacto, checkpoint/hash, métrica, split de selección, hardware y linaje. `03_07` acepta únicamente candidatos del SHA-256 activo, ordena con métricas de validation y publica test solo como informe. La falta de cualquiera de estas piezas impide registrarlo para producción.
