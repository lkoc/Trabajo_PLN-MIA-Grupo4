# Orden reproducible de ejecución

Todo el recorrido usa el contrato de etiquetas v2.1 con `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta. Los casos indeterminados se difieren y no entran al entrenamiento. El troceador se versiona por separado y actualmente corresponde a v2.2.0.

## Recorrido completo

| Paso | Cuaderno | Entrada principal | Salida principal | Reanudación/no-op |
|---:|---|---|---|---|
| 1 | `01_01_scraping_incremental` | canales/consultas + candidatos + JSON/VTT/caché históricos | VTT y JSON por canal consolidados, índices, manifiesto faltante y canónico | procesa primero el backfill VTT; cada éxito se conserva antes de continuar y se omite al reanudar |
| 2 | `01_015_ampliacion_dirigida_minorias` | decisión efectiva + déficit por daño en train | candidatos PE dirigidos, con canales/videos estimados y split estable | opcional; meta ≥2.000 chunks por daño en train, sin alterar `01_01` |
| 3 | `01_02_optimizacion_longitud_chunks` | transcripciones + chunks etiquetados + snapshot | perfil clásico decisorio y perfil neuronal robusto MiniLM/Gemma para 15/20/25/30/35 s | reanuda ajustes y respuestas por firma; la jerarquía no cambia datos sin aplicación explícita |
| 4 | `01_03_limpieza_troceado_incremental` | JSON por canal + VTT locales + configuración activa | canónico reconstruido, chunks v2 y manifiesto descriptivo | barra por video; hash de transcripción + firma; `REBUILD_CHUNKS_FROM_ZERO` solo para reconstrucción respaldada |
| 5 | `02_00_preparacion_bundle_colab` | bundle sincronizado en GitHub o carga local por navegador | `bundle_releases/<bundle_id>` y `latest.json` en Drive | se ejecuta en Colab; verifica identidad y SHA-256, reutiliza versiones válidas y se repite después de `02_05` |
| 6 | `02_01_etiquetado_local_ollama` | chunks actuales + etiquetado DeepSeek histórico | recuperación exacta 1:1, calibración Flash–Pro, primera pasada Flash y revisión Pro dirigida | firma + `chunk_id`, fsync por grupo de 5, checkpoint Drive periódico/al interrumpir, cuarentena y tope de costo |
| 7 | `02_02_etiquetado_remoto` | chunks pendientes | fallback local Qwen3-1.7B | opcional, separado por campaña y `chunk_id` |
| 8 | `02_03_revision_llm_dirigida` | artefactos de `02_01` | tablas de calibración, cobertura y enrutamiento | solo lectura; no repite API |
| 9 | `02_04_consolidacion_validacion_humana` | propuestas LLM + chunks | campaña consolidada + eventos humanos | precedencia, contenido estable y eventos append-only |
| 10 | `02_05_cierre_humano_snapshot` | consolidado + eventos + chunks | anotaciones revisadas + snapshot inmutable | firma de insumos y SHA-256 del contenido |
| 11 | `03_01_modelos_clasicos` | snapshot activo | cinco candidatos clásicos completos | firma dataset+configuración |
| 12 | `03_02_transformers_planos` | mismo snapshot | MiniLM y E5 completos | checkpoint de interrupción, warm start o no-op |
| 13 | `03_03_transformer_cascada` | mismo snapshot | compuerta + cuatro daños | checkpoint/warm start o no-op |
| 14 | `03_04_transformer_multitarea` | mismo snapshot | cinco salidas + auxiliares | checkpoint/warm start o no-op |
| 15 | `03_05_qwen_lora` | mismo snapshot | adaptador LoRA de cinco salidas | checkpoint/warm start o no-op |
| 16 | `03_06_qwen_estructurado` | mismo snapshot | Qwen con penalización de conflicto | checkpoint/warm start o no-op |
| 17 | `03_07_comparacion_final` | candidatos del mismo SHA-256 | comparación + registro productivo | no reescribe si la selección no cambia |
| 18 | `03_08_auditoria_finas_flags` | snapshot activo | auditoría auxiliar | SHA-256 del snapshot |
| 19 | `04_01_frontend_produccion` | registro validado | demostrador supervisado | caché de subtítulos + eventos append-only |

`03_01`–`03_06` son ramas comparables: no dependen entre sí y pueden omitirse las familias que no se quieran evaluar. `03_07` necesita al menos un candidato completo del snapshot activo. Test nunca participa en la selección; solo informa después de congelar modelo y umbrales con validation.

## Interruptores deliberados

- En `01_01`, la continuación actual usa `DISCOVER_NEW=False`, `FETCH_NEW=True` y `BACKFILL_MISSING_VTT=True`: no busca fuentes nuevas, recupera primero los VTT faltantes y después recorre candidatos materializados. `MAX_VTT_BACKFILL=None` y `MAX_NEW_VIDEOS=None` incluyen ambas colas completas. `RANDOMIZE_DOWNLOAD_QUEUE=True` ordena de forma pseudoaleatoria reproducible e intercala canales. `NETWORK_BATCH_SIZE=10` y `NETWORK_BATCH_PAUSE_SECONDS=15` regulan la ejecución; `yt-dlp` añade pausas internas de 2.5–10 segundos y un timeout de 30 segundos por operación. Cambie `FETCH_NEW=False` para una inspección sin red.
- `01_015` se usa solo para la ampliación minoritaria. Parte de 1.826 chunks de racismo/discriminación y 1.568 de ataque por género/identidad en `train`, pondera déficits de 174 y 432, estima el mínimo de canales PE dirigidos con su rendimiento histórico y añade un margen de canales. En el corte actual calcula 4 canales y 239/36/36 candidatos para train/validation/test mediante SHA-256 del `video_id`. Después de etiquetar, vuelva a ejecutarlo para comprobar si ya se alcanzó la meta de 2.000. Si faltan candidatos PE o la capacidad estimada del catálogo es insuficiente, se detiene antes de descargar.
- `01_02` es opcional. `RUN_CHUNK_LENGTH_SMOKE_TEST=True` hace 10 ajustes CPU y `RUN_CHUNK_LENGTH_CONFIRMATORY_TEST=True` hace 45 ajustes en tres cohortes pareadas. El perfil decisorio clásico activa `RUN_CHUNK_LENGTH_ROBUST_TEST=True`: 75 ajustes en cinco cohortes de 300/100/100 videos y 1 000 réplicas bootstrap por `video_id`, con referencia 30 s y margen 0.01 AP. Después, `RUN_NEURAL_ROBUST_TEST=True` compara las mismas cinco longitudes sobre un panel pareado de 100 anclas de `validation`: 25 cabezas MiniLM y hasta 500 respuestas estructuradas de `gemma3:4b`, cinco cohortes de reporte y 2 000 réplicas bootstrap por familia. MiniLM y Ollama son confirmatorios; no se promedian ni cambian automáticamente la longitud. Ante conflicto se conserva la selección clásica hasta validación humana independiente. Solo `APPLY_CHUNK_SELECTION=True` modifica la firma activa.
- En `02_00`, use Google Colab, mantenga `BUNDLE_SOURCE='github'` si el bundle ya está sincronizado —o use `'local_upload'` para escoger los nueve archivos locales más recientes— y active `RUN_PUBLISH_BUNDLE=True`. La autorización integrada de `drive.mount()` no requiere Google Cloud Console ni Drive Desktop.
- En `02_01`, deje `RECOVER_HISTORICAL=True`: solo las coincidencias exactas y unívocas por video/texto se marcan completas. Active luego `RUN_API_PREFLIGHT=True`, `RUN_CALIBRATION=True`, `RUN_PRIMARY=True` y finalmente `RUN_DIRECTED_REVIEW=True`. Los pilotos 300/500 ya concluyeron; la campaña completa usa `PRIMARY_LIMIT=None` y `REVIEW_LIMIT=None`. Cada grupo de cinco se sincroniza a disco y los checkpoints atómicos se publican periódicamente, al cerrar una fase o ante `Ctrl+C`.
- `02_02` conserva `RUN_FALLBACK=False`; su Qwen3-1.7B es un diagnóstico local independiente, no una segunda campaña que deba promediarse.
- En `03_01`–`03_06`, active `RUN_TRAINING=True`. Una segunda ejecución con el mismo snapshot devuelve `status="noop"`.
- En `03_07`, active `RUN_PUBLISH=True` después de copiar bajo `modelos/v2` los runs neuronales devueltos por Colab.

El corte documentado del 2026-08-08 recuperó 52 244 filas Flash y 9 912 Pro,
completó la calibración pareada sin errores y registró 14 399 pendientes nuevos
válidos a 867.279 chunks/min. El detalle de tiempo, caché, acuerdo y costo está en el
[corte cuantitativo de la campaña](../resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md).
La campaña seguía activa: al reanudar `02_01` no elimine ni renombre sus JSONL;
el cuaderno omite automáticamente los `chunk_id` ya válidos.

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

Después de que `02_05` cree un snapshot distinto, regenere el bundle local,
sincronícelo con GitHub o selecciónelo mediante `local_upload`, y vuelva a
ejecutar `02_00` en Colab antes de iniciar `03`; así se publica otro `bundle_id`
inmutable y los consumidores activan exactamente el dataset declarado.

## Qué ocurre cuando crece la muestra

Agregue candidatos y ejecute otra vez `01_01→01_03`; para corregir el desbalance use `01_01→01_015→01_03`. No es necesario repetir la optimización opcional de `01_02`. Los videos, subtítulos, chunks y etiquetas ya resueltos se omiten. `01_01` adquiere los pendientes por lotes, guarda cada éxito en caché y cada fallo en JSONL antes de avanzar; un 429 excluye solo el canal afectado durante esa ejecución y el flujo continúa con los demás canales. `01_015` limita su campaña a las categorías minoritarias y conserva una reserva de holdout. `01_03` recompone el canónico desde las partes sincronizadas, recupera VTT locales sin JSON, actualiza las partes por canal y materializa únicamente hashes nuevos o modificados. Los VTT nunca se borran. `02_05` crea un snapshot nuevo que incluye filas anteriores y nuevas cuando sus `chunk_id` permanecen compatibles; después de una reconstrucción de versión debe generarse un snapshot nuevo sin trasladar etiquetas automáticamente. Los modelos neuronales inicializan el nuevo run desde el candidato compatible anterior y entrenan con el snapshot completo; si no cambió ninguna entrada, todo el tramo permanece en no-op.

El snapshot migrado actual permite comenzar directamente en `03_01` para conservar la línea base. Reconstruir desde adquisición requiere completar `01_01→01_03` —o insertar `01_015` si se amplían minorías— y no infiere `SEGURO` de listas vacías.
