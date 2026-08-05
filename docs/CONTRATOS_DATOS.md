# Contratos de datos v2.1

## Video y transcripción

Una transcripción canónica contiene `video_id`, URL, canal, fuente de subtítulo, segmentos temporales y `transcript_sha256`. La presencia del `video_id` impide una nueva descarga; un archivo de caché válido se reutiliza antes de consultar la red.

## Chunk

`chunk_id` se deriva de versión del troceador, `video_id`, tiempos y texto normalizado. Cambiar la regla de troceado crea nuevos IDs y una nueva versión; no modifica los chunks anteriores.

## Anotación

Campos mínimos: `chunk_id`, `text`, `coarse_labels`, `fine_labels`, `flags`, `needs_review`, `training_eligible`, fuente, modelo, prompt y versiones. Invariantes:

- `SEGURO` o uno/más daños, nunca ambos;
- una lista vacía solo es válida con revisión y exclusión temporal del entrenamiento;
- los flags no son categorías principales;
- toda salida conserva procedencia.

## Snapshot de entrenamiento

`ModelReadyRecord` conserva `chunk_id`, `video_id`, texto, categorías canónicas, señales de referencia, fuente, peso, campaña, procedencia histórica y `split`. Solo contiene decisiones resueltas y entrenables. La partición es estable por `video_id`; un video nunca cruza train, validation y test. Un incremento materializa otro snapshot y conserva el anterior.

El snapshot v2.1 usa `ATAQUE_POR_GENERO_IDENTIDAD` como objetivo. El identificador anterior solo puede aparecer en `legacy_coarse_labels`, nunca en `coarse_labels`. `modperu validate ruta.jsonl --kind model-ready` comprueba las categorías, la exclusividad de `SEGURO`, la procedencia migrada, los flags y el split.

## Registro de modelos

Un modelo activo debe declarar cinco scores y cinco umbrales, el contrato exacto, checkpoint/hash, métrica, split de selección, hardware y linaje. La falta de cualquiera de estas piezas impide registrarlo para producción.
