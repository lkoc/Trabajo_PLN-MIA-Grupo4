# Contratos de datos y etiquetas

El proyecto mantiene versiones independientes: `v2.1` es el contrato de etiquetas de cinco salidas y `v2.2.0` es la versión actual del troceador. Una actualización del troceador no cambia por sí sola la taxonomía, sus reglas de exclusividad ni sus umbrales; por eso no corresponde sustituir `v2.1` por `v2.2` de forma global.

El contrato de anotación y entrenamiento tiene cinco salidas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Sus umbrales y reglas de exclusividad son decisiones operativas locales.

## Video y transcripción

Una transcripción canónica contiene `video_id`, URL, canal, fuente de subtítulo, segmentos temporales y `transcript_sha256`. La presencia del `video_id` impide una nueva descarga; un archivo de caché válido se reutiliza antes de consultar la red.

Por compatibilidad histórica, `transcript_sha256` tiene alcance de capa. En la adquisición es la huella del JSON crudo de segmentos ordenado por claves; en chunks e índice de versiones es la huella de esos segmentos después de aplicar tiempos a tres decimales y la normalización textual del troceador. No se exige que ambas huellas sean iguales: la primera controla la integridad del checkpoint y la segunda la idempotencia de la materialización. La auditoría debe recalcular cada una con su algoritmo, no compararlas directamente.

`transcripts_raw.jsonl` es una vista local reconstruible. La representación sincronizable se divide bajo `transcripts_by_channel/`; el índice declara cada parte, cantidad, bytes y SHA-256. Un VTT sin JSON puede aportar una transcripción solo si supera 200 caracteres útiles. La recuperación es de solo lectura: ningún VTT se elimina ni sobrescribe.

## Chunk

`chunk_id` se deriva de versión del troceador, firma de configuración, `video_id`, tiempos y texto normalizado. La firma cubre `max_seconds`, límites de caracteres y solapamiento. Cambiar una regla crea nuevos IDs y una nueva versión; no modifica los chunks anteriores. Las filas del troceador 2.1 sin firma equivalen explícitamente a la configuración histórica de 30 s; esta versión no debe confundirse con el contrato de etiquetas v2.1.

`config/chunking.json` declara la configuración deseada y
`datos/processed/chunking_active.json` registra la firma local activa. Una
transición mueve todos los derivados gestionados a
`archivo/chunking_configurations/<firma>/state/`, junto con un manifiesto de
hashes. La restauración rechaza un archivo alterado o una colisión; nunca mueve
transcripciones raw ni candidatos.

`chunk_materialization_manifest.json` registra la firma, hashes de entrada y salida, cobertura por `video_id`, estadística descriptiva y el respaldo de una reconstrucción total. El modo incremental es idempotente por `(video_id, transcript_sha256, chunking_signature)`. Una reconstrucción bajo otra versión o firma puede cambiar `chunk_id`; las etiquetas no se transfieren por posición o texto sin una nueva adjudicación explícita.

## Anotación

Campos mínimos: `chunk_id`, `video_id`, `text`, `coarse_labels`, `fine_labels`, `flags`, `needs_review`, `training_eligible`, fuente, modelo, prompt y versiones. `video_id` puede faltar en propuestas históricas, pero `02_05` debe recuperarlo del chunk fuente antes de crear datos entrenables; nunca se parte `chunk_id` para deducirlo. Invariantes:

- `SEGURO` o una o más de `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`, nunca ambos grupos;
- una lista vacía solo es válida con revisión y exclusión temporal del entrenamiento;
- los flags no son categorías principales;
- toda salida conserva procedencia.

## Evento y reconciliación humana

`ReviewEvent` es append-only y registra propuesta, decisión, acción, flags, modelo y revisor pseudonimizado. Para cada chunk se aplica el evento más reciente por `(created_at, event_id)`. `accept` y `modify` producen una decisión humana resuelta; `defer` queda fuera de entrenamiento y no crea una sexta clase; `reject` queda explícitamente excluido. El consolidado LLM y los eventos originales nunca se sobrescriben.

## Snapshot de entrenamiento

`ModelReadyRecord` conserva `chunk_id`, `video_id`, canal, texto, categorías canónicas, señales de referencia, fuente, prompt, peso, campaña, procedencia histórica, `split` y `channel_split`. Incluye máscaras binarias separadas para gruesas, finas y flags. Una posición ausente no se interpreta como negativa: la pérdida y las métricas auxiliares solo usan posiciones observadas. La partición primaria es estable por `video_id`; la partición de robustez es estable por canal y evita compartirlo entre train, validation y test. Un incremento materializa otro snapshot y conserva el anterior.

El snapshot v2.1 usa `ATAQUE_POR_GENERO_IDENTIDAD` como objetivo. El identificador anterior solo puede aparecer en `legacy_coarse_labels`, nunca en `coarse_labels`. `modperu validate ruta.jsonl --kind model-ready` comprueba las categorías, la exclusividad de `SEGURO`, la procedencia migrada, los flags y el split. Cada snapshot vive en `datos/model_ready/v2/snapshots/<snapshot_id>/`; `dataset_5_salidas.jsonl` es una vista de conveniencia al último snapshot y solo cambia si cambia su contenido.

## Registro de modelos

Cada experimento previo a la selección escribe `candidate.json`, `metrics.json`, predicciones de validation, cinco umbrales principales, métricas auxiliares observadas, un bundle de inferencia y `checkpoint_manifest.json`. No genera predicciones ni métricas de test. La firma combina dataset, muestreo, configuración, taxonomía y versión del motor; repetirla es no-op. Train y validation conservan todos los daños y seleccionan `SEGURO` determinísticamente a 4:1, aproximadamente por canal. Test permanece completo y sellado. Tras congelar la selección, `03_07` ejecuta una sola inferencia sobre todo test: reporta como principal la prevalencia natural y como secundaria una vista determinista 4:1 obtenida de las mismas predicciones. Esta segunda vista no constituye otra apertura de test.

Un modelo activo debe declarar cinco scores y cinco umbrales principales, el contrato exacto, checkpoint/hash, métrica, split de selección, hardware y linaje. `03_07` acepta únicamente candidatos del SHA-256 activo cuyo test siga sellado; compara individuos y ensembles, informa una frontera de Pareto y congela candidatos, pesos y umbrales. Solo una acción posterior explícita abre test una vez. Publicar exige otra aprobación y no ocurre al comparar. La falta de cualquiera de estas piezas impide registrarlo para producción.
