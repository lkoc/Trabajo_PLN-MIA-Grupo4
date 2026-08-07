# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 01 · Scraping, limpieza y troceado

## Orden

1. `01_01_scraping_incremental.ipynb`
2. `01_02_optimizacion_longitud_chunks.ipynb` — opcional
3. `01_03_limpieza_troceado_incremental.ipynb`

Entrada: candidatos con `video_id` y URL, snapshots históricos, transcripciones canónicas y caché local.  
Salida: transcripciones JSONL y chunks v2 con tiempos, hash de transcripción, versión y firma de configuración del troceador.
Control: nunca se descarga audio o video; primero se consolidan sin modificarlos los `transcripts_raw.jsonl` ya existentes, después se recuperan los VTT faltantes aunque el JSON ya sea canónico, se reutiliza el caché y solo al final se consulta la red para un `video_id` nuevo. La adquisición recupera la ruta histórica `yt-dlp → VTT`, conserva todas las pistas descargadas, elige la más completa, exige 200 caracteres y usa `youtube-transcript-api` solo como respaldo. La limpieza conserva la eliminación de hasta 12 palabras solapadas en subtítulos rodantes, el cierre a 30 segundos/600 caracteres y el mínimo de 90 caracteres.

`01_01` reúne el scraping inicial y la antigua ampliación dirigida. Su bloque de controles permite elegir `DISCOVERY_MODE="seed"`, `"directed"` o `"both"`; editar canales y consultas; ampliar la cobertura de búsqueda; y configurar reintentos, lote, pausa y aleatorización. La versión actual continúa la corrida parcial con `DISCOVER_NEW=False`, `FETCH_NEW=True` y `BACKFILL_MISSING_VTT=True`: primero completa VTT conocidos y luego procesa candidatos existentes, sin descubrir fuentes nuevas. `MAX_VTT_BACKFILL=None` y `MAX_NEW_VIDEOS=None` recorren ambas colas completas. `RANDOMIZE_DOWNLOAD_QUEUE=True` crea un orden reproducible con `DOWNLOAD_RANDOM_SEED` e intercala canales. La red usa lotes de 10, 15 segundos entre lotes, pausas internas de 2.5–10 segundos y timeout de 30 segundos por operación. `FETCH_NEW=False` deja la ejecución sin solicitudes de subtítulos.

La opción comentada `RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"` mueve los artefactos activos a `archivo/reinicios_dataset_videos/` y bloquea la reimportación automática de snapshots históricos. Es recuperable e idempotente; no borra código ni el archivo histórico.

Para ampliar la muestra, active el modo requerido o agregue filas a `datos/raw/video_candidates.jsonl` o `datos/raw/videos_candidatos.csv` y vuelva a ejecutar `01_01`. El corpus previo permanece intacto. `BACKFILL_MISSING_VTT=True` consume `datos/raw/vtt_by_video/missing_vtt.jsonl`; `MAX_VTT_BACKFILL=None` recorre toda esa cola. Cada éxito guarda el VTT antes de avanzar y cada fallo se anexa de forma idempotente; al reanudar no se repiten los videos consolidados. Los videos exclusivos para miembros, privados, retirados, sin subtítulos o con menos de 200 caracteres se registran sin detener el lote.

`01_02` no es necesario para cada incremento. Su modo rápido y la confirmación
corta permanecen como diagnósticos opcionales. El perfil robusto clásico es la
etapa decisoria: cinco cohortes de 300/100/100 videos, 75 ajustes y 1 000
réplicas bootstrap agrupadas por video; compara 15, 20, 25, 30 y 35 s contra la
referencia de 30 s con margen de no inferioridad de 0.01 AP y nunca selecciona
con `test`.

La etapa posterior usa `RUN_NEURAL_ROBUST_TEST=True` y conserva exactamente las
mismas cinco longitudes. Construye un panel enriquecido y pareado de 100 anclas
de `validation`, dividido por video en cinco cohortes de reporte. MiniLM queda
congelado y alimenta 25 cabezas logísticas —cinco longitudes por cinco cohortes
de entrenamiento—; `gemma3:4b` procesa hasta 500 combinaciones ancla–longitud
con [`config/prompt_operacional_ollama_v2.md`](../../config/prompt_operacional_ollama_v2.md).
Cada familia usa 2 000 réplicas bootstrap agrupadas por video. MiniLM produce AP
continua y Ollama etiquetas duras; sus métricas no se promedian. Son análisis
confirmatorios de sensibilidad y factibilidad, no un segundo selector. Si
divergen, el cuaderno informa el conflicto y conserva la decisión clásica hasta
validación humana independiente. Solo `APPLY_CHUNK_SELECTION=True` aplica una
longitud; el método citado y los resultados canónicos se documentan en
[`docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md`](../../docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md).

La ejecución completa obtuvo evidencia inconclusa en MiniLM: 20 s presentó la
mayor AP puntual, 0.59, pero su intervalo de diferencia frente a 30 s incluyó
cero. Ollama produjo 474/500 salidas válidas; 15, 20 y 25 s no alcanzaron la
compuerta estructural de 0.95. Su mayor F1 puntual fue 0.42 para 30 s. La
síntesis jerárquica mantiene 30 s y no cambia la configuración activa.

Antes de activar otra longitud, los chunks, etiquetas, snapshot, modelos,
resultados y bundle correspondientes a la firma activa se mueven a
`archivo/chunking_configurations/<firma>/state/`. No se borran. Volver a esa
firma verifica y restaura los mismos bytes; `01_03` empieza vacío únicamente
cuando la longitud nunca fue materializada.

Con `SYNC_TRANSCRIPTS_BY_CHANNEL=True`, el canónico existente se materializa sin
borrarlo bajo `datos/raw/transcripts_by_channel/`. Después, cada éxito nuevo se
anexa primero al archivo pequeño de su canal y luego al canónico. El índice de
la carpeta registra hashes y permite reconstruir `transcripts_raw.jsonl` en otra
máquina sin consultar YouTube. Git sincroniza esa carpeta y los candidatos, no
`transcripts_cache/`.

Con `SYNC_VTT_BY_VIDEO=True`, `01_01` copia sin borrar las pistas históricas a
`datos/raw/vtt_by_video/`, verifica formato y SHA-256, y publica `index.json` y
`missing_vtt.jsonl`. Una respuesta exclusiva de `youtube-transcript-api` se
serializa como `*.transcript-api.vtt` y se marca como derivada; no se presenta
como archivo original de `yt-dlp`.

Tras clonar, ejecute `python tools/restore_synced_checkpoints.py`. Los chunks son
baratos de reconstruir, aunque su gzip permanece en el bundle por ser entrada
de Colab; el dataset anotado se conserva comprimido porque no es barato
recrearlo. Los cuadernos `03_01`–`03_08` descomprimen el dataset si falta y
verifican que su SHA-256 coincida antes de usarlo.

El cuaderno presenta una barra por fuentes durante el descubrimiento y otra por videos durante la adquisición. Esta última distingue respuestas 429, canales excluidos y videos diferidos. Un 429 excluye solo el canal afectado durante esa corrida; no detiene ni difiere otros canales. Una URL de canal con HTTP 404 o sin pestaña `/videos` se clasifica como `stale_channel_or_no_videos_tab`; afecta esa fuente, no invalida toda la corrida.
