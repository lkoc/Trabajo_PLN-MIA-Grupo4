# Orden reproducible de ejecución

## Recorrido completo

| Paso | Cuaderno | Entrada principal | Salida principal | Reanudación/no-op |
|---:|---|---|---|---|
| 1 | `01_01_scraping_incremental` | canales/consultas + candidatos + corpus/caché existente | candidatos y transcripciones JSONL | modo semilla/dirigido/combinado; `video_id`; registra fallos y continúa |
| 2 | `01_02_limpieza_troceado_incremental` | transcripciones | chunks v2 | hash de transcripción + `chunk_id` |
| 3 | `02_01_etiquetado_local_ollama` | chunks pendientes | anotaciones locales | `chunk_id` |
| 4 | `02_02_etiquetado_remoto` | muestra opcional | anotaciones remotas | opcional, `chunk_id` y activación comercial explícita |
| 5 | `02_03_revision_llm_dirigida` | baja confianza/duda | cola priorizada | selección determinista; opcional |
| 6 | `02_04_consolidacion_validacion_humana` | propuestas LLM + chunks | campaña consolidada + eventos humanos | precedencia, contenido estable y eventos append-only |
| 7 | `02_05_cierre_humano_snapshot` | consolidado + eventos + chunks | anotaciones revisadas + snapshot inmutable | firma de insumos y SHA-256 del contenido |
| 8 | `03_01_modelos_clasicos` | snapshot activo | cinco candidatos clásicos completos | firma dataset+configuración |
| 9 | `03_02_transformers_planos` | mismo snapshot | MiniLM y E5 completos | checkpoint de interrupción, warm start o no-op |
| 10 | `03_03_transformer_cascada` | mismo snapshot | compuerta + cuatro daños | checkpoint/warm start o no-op |
| 11 | `03_04_transformer_multitarea` | mismo snapshot | cinco salidas + auxiliares | checkpoint/warm start o no-op |
| 12 | `03_05_qwen_lora` | mismo snapshot | adaptador LoRA de cinco salidas | checkpoint/warm start o no-op |
| 13 | `03_06_qwen_estructurado` | mismo snapshot | Qwen con penalización de conflicto | checkpoint/warm start o no-op |
| 14 | `03_07_comparacion_final` | candidatos del mismo SHA-256 | comparación + registro productivo | no reescribe si la selección no cambia |
| 15 | `03_08_auditoria_finas_flags` | snapshot activo | auditoría auxiliar | SHA-256 del snapshot |
| 16 | `04_01_frontend_produccion` | registro validado | demostrador supervisado | caché de subtítulos + eventos append-only |

`03_01`–`03_06` son ramas comparables: no dependen entre sí y pueden omitirse las familias que no se quieran evaluar. `03_07` necesita al menos un candidato completo del snapshot activo. Test nunca participa en la selección; solo informa después de congelar modelo y umbrales con validation.

## Interruptores deliberados

- En `01_01`, seleccione `DISCOVERY_MODE="seed"`, `"directed"` o `"both"`; `MAX_NEW_VIDEOS` limita las llamadas nuevas. Use `DISCOVER_NEW=True` y `FETCH_NEW=True` únicamente cuando quiera ampliar el corpus.
- En `02_01`, active `RUN=True`; pruebe primero `LIMIT=20` y después quite el límite para cerrar el lote.
- `02_02` conserva `RUN_REMOTE=False` salvo decisión explícita de usar la API comercial.
- En `03_01`–`03_06`, active `RUN_TRAINING=True`. Una segunda ejecución con el mismo snapshot devuelve `status="noop"`.
- En `03_07`, active `RUN_PUBLISH=True` después de copiar bajo `modelos/v2` los runs neuronales devueltos por Colab.

## Qué ocurre cuando crece la muestra

Agregue candidatos y ejecute otra vez 01→02. Los videos, subtítulos, chunks y etiquetas ya resueltos se omiten. `02_05` crea un snapshot nuevo que incluye filas anteriores y nuevas, conserva las asignaciones por `video_id` y no modifica snapshots previos. Los modelos neuronales inicializan el nuevo run desde el candidato compatible anterior y entrenan con el snapshot completo; si no cambió ninguna entrada, todo el tramo permanece en no-op.

El snapshot migrado actual permite comenzar directamente en `03_01` para conservar la línea base. Reconstruir desde adquisición requiere completar 01→02 y no infiere `SEGURO` de listas vacías.
