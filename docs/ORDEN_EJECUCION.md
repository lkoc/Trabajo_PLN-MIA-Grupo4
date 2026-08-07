# Orden reproducible de ejecución

## Recorrido completo

| Paso | Cuaderno | Entrada principal | Salida principal | Reanudación/no-op |
|---:|---|---|---|---|
| 1 | `01_01_scraping_incremental` | canales/consultas + candidatos + JSON/VTT/caché históricos | VTT y JSON por canal consolidados, índices, manifiesto faltante y canónico | procesa primero el backfill VTT; cada éxito se conserva antes de continuar y se omite al reanudar |
| 2 | `01_02_optimizacion_longitud_chunks` | transcripciones + chunks etiquetados + snapshot | comparación opcional 15–35 s, perfil robusto y diagnósticos MiniLM/Gemma | reanuda modelos por firma; bootstrap y comparadores no cambian datos sin aplicación explícita |
| 3 | `01_03_limpieza_troceado_incremental` | transcripciones + configuración activa | chunks v2 | hash de transcripción + firma de configuración + `chunk_id` |
| 4 | `02_01_etiquetado_local_ollama` | chunks pendientes | anotaciones locales | `chunk_id` |
| 5 | `02_02_etiquetado_remoto` | muestra opcional | anotaciones remotas | opcional, `chunk_id` y activación comercial explícita |
| 6 | `02_03_revision_llm_dirigida` | baja confianza/duda | cola priorizada | selección determinista; opcional |
| 7 | `02_04_consolidacion_validacion_humana` | propuestas LLM + chunks | campaña consolidada + eventos humanos | precedencia, contenido estable y eventos append-only |
| 8 | `02_05_cierre_humano_snapshot` | consolidado + eventos + chunks | anotaciones revisadas + snapshot inmutable | firma de insumos y SHA-256 del contenido |
| 9 | `03_01_modelos_clasicos` | snapshot activo | cinco candidatos clásicos completos | firma dataset+configuración |
| 10 | `03_02_transformers_planos` | mismo snapshot | MiniLM y E5 completos | checkpoint de interrupción, warm start o no-op |
| 11 | `03_03_transformer_cascada` | mismo snapshot | compuerta + cuatro daños | checkpoint/warm start o no-op |
| 12 | `03_04_transformer_multitarea` | mismo snapshot | cinco salidas + auxiliares | checkpoint/warm start o no-op |
| 13 | `03_05_qwen_lora` | mismo snapshot | adaptador LoRA de cinco salidas | checkpoint/warm start o no-op |
| 14 | `03_06_qwen_estructurado` | mismo snapshot | Qwen con penalización de conflicto | checkpoint/warm start o no-op |
| 15 | `03_07_comparacion_final` | candidatos del mismo SHA-256 | comparación + registro productivo | no reescribe si la selección no cambia |
| 16 | `03_08_auditoria_finas_flags` | snapshot activo | auditoría auxiliar | SHA-256 del snapshot |
| 17 | `04_01_frontend_produccion` | registro validado | demostrador supervisado | caché de subtítulos + eventos append-only |

`03_01`–`03_06` son ramas comparables: no dependen entre sí y pueden omitirse las familias que no se quieran evaluar. `03_07` necesita al menos un candidato completo del snapshot activo. Test nunca participa en la selección; solo informa después de congelar modelo y umbrales con validation.

## Interruptores deliberados

- En `01_01`, la continuación actual usa `DISCOVER_NEW=False`, `FETCH_NEW=True` y `BACKFILL_MISSING_VTT=True`: no busca fuentes nuevas, recupera primero los VTT faltantes y después recorre candidatos materializados. `MAX_VTT_BACKFILL=None` y `MAX_NEW_VIDEOS=None` incluyen ambas colas completas. `RANDOMIZE_DOWNLOAD_QUEUE=True` ordena de forma pseudoaleatoria reproducible e intercala canales. `NETWORK_BATCH_SIZE=10` y `NETWORK_BATCH_PAUSE_SECONDS=15` regulan la ejecución; `yt-dlp` añade pausas internas de 2.5–10 segundos y un timeout de 30 segundos por operación. Cambie `FETCH_NEW=False` para una inspección sin red.
- `01_02` es opcional. `RUN_CHUNK_LENGTH_SMOKE_TEST=True` hace 10 ajustes CPU; `RUN_CHUNK_LENGTH_CONFIRMATORY_TEST=True` hace 45 ajustes en tres cohortes pareadas. El perfil mayor activa solo `RUN_CHUNK_LENGTH_ROBUST_TEST=True`: 75 ajustes en cinco cohortes de 300/100/100 videos y 1 000 réplicas bootstrap agrupadas por `video_id`, con referencia 30 s y margen 0.01 AP. `RUN_BOUNDED_HF_COMPARISON=True` usa MiniLM congelado sobre 20 s/30 s y `RUN_BOUNDED_OLLAMA_COMPARISON=True` evalúa como máximo seis respuestas estructuradas de `gemma3:4b`; son diagnósticos y no seleccionan longitud. Cada ajuste supervisado entrena, calibra e infiere con una sola longitud. Solo `APPLY_CHUNK_SELECTION=True` cambia la firma activa; por defecto no mueve nada.
- En `02_01`, active `RUN=True`; pruebe primero `LIMIT=20` y después quite el límite para cerrar el lote.
- `02_02` conserva `RUN_REMOTE=False` salvo decisión explícita de usar la API comercial.
- En `03_01`–`03_06`, active `RUN_TRAINING=True`. Una segunda ejecución con el mismo snapshot devuelve `status="noop"`.
- En `03_07`, active `RUN_PUBLISH=True` después de copiar bajo `modelos/v2` los runs neuronales devueltos por Colab.

## Reanudación después de clonar

Ejecute `python tools/restore_synced_checkpoints.py` una vez. El comando
recompone `datos/raw/transcripts_raw.jsonl` desde
`datos/raw/transcripts_by_channel/` y restaura desde el bundle los chunks y el
dataset con verificación SHA-256. Es idempotente respecto del canónico. Los
cuadernos `03_01`–`03_08` también verifican el dataset antes de usarlo y solo lo
descomprimen si falta; una copia local divergente produce un error explícito.

No es necesario sincronizar `transcripts_cache/`. Los chunks se pueden volver a
generar con `01_03`, pero la copia gzip permanece en el bundle porque `02_01` la
consume en Colab. El dataset no se considera barato de reconstruir: incorpora
anotación humana y pseudoetiquetas con procedencia, por lo que su checkpoint
comprimido sí forma parte del estado sincronizado.

Después de que `02_05` cree un snapshot distinto, ejecute
`python tools/prepare_colab_bundle.py --destination resultados/colab_bundle`
antes de iniciar `03`; así el manifiesto y el dataset local vuelven a compartir
el mismo SHA-256.

## Qué ocurre cuando crece la muestra

Agregue candidatos y ejecute otra vez `01_01→01_03`; no es necesario repetir la optimización opcional de `01_02`. Los videos, subtítulos, chunks y etiquetas ya resueltos se omiten. `01_01` adquiere los pendientes por lotes, guarda cada éxito en caché y cada fallo en JSONL antes de avanzar; un 429 excluye solo el canal afectado durante esa ejecución y el flujo continúa con los demás canales. `02_05` crea un snapshot nuevo que incluye filas anteriores y nuevas, conserva las asignaciones por `video_id` y no modifica snapshots previos. Los modelos neuronales inicializan el nuevo run desde el candidato compatible anterior y entrenan con el snapshot completo; si no cambió ninguna entrada, todo el tramo permanece en no-op.

El snapshot migrado actual permite comenzar directamente en `03_01` para conservar la línea base. Reconstruir desde adquisición requiere completar `01_01→01_03` y no infiere `SEGURO` de listas vacías.
