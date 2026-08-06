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
Control: nunca se descarga audio o video; primero se consolidan sin modificarlos los `transcripts_raw.jsonl` ya existentes, después se reutiliza el caché y solo al final se consulta la red para un `video_id` nuevo. La adquisición recupera la ruta histórica `yt-dlp → VTT`, elige la pista más completa, exige 200 caracteres y usa `youtube-transcript-api` solo como respaldo. La limpieza conserva la eliminación de hasta 12 palabras solapadas en subtítulos rodantes, el cierre a 30 segundos/600 caracteres y el mínimo de 90 caracteres.

`01_01` reúne el scraping inicial y la antigua ampliación dirigida. Su bloque de controles permite elegir `DISCOVERY_MODE="seed"`, `"directed"` o `"both"`; editar canales y consultas; ampliar la cobertura de búsqueda; y configurar reintentos, lote, pausa y aleatorización. `directed` usa los déficits de `train+validation`, rendimiento histórico por canal, expansión desde búsquedas y una cohorte aislada; sin etiquetas previas activa un fallback equitativo para los cuatro daños. `MAX_DIRECTED_CANDIDATES=None` conserva toda la cohorte inédita y `MAX_NEW_VIDEOS=None` recorre todos sus pendientes. `RANDOMIZE_DOWNLOAD_QUEUE=True` crea un orden reproducible con `DOWNLOAD_RANDOM_SEED` e intercala canales. La red usa lotes de 10, 60 segundos entre lotes y 5–10 segundos dentro de `yt-dlp`. `DISCOVER_NEW=False` no consulta fuentes y `FETCH_NEW=False` no obtiene subtítulos.

La opción comentada `RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"` mueve los artefactos activos a `archivo/reinicios_dataset_videos/` y bloquea la reimportación automática de snapshots históricos. Es recuperable e idempotente; no borra código ni el archivo histórico.

Para ampliar la muestra, active el modo requerido o agregue filas a `datos/raw/video_candidates.jsonl` o `datos/raw/videos_candidatos.csv` y vuelva a ejecutar `01_01`. El corpus previo permanece intacto. Cada éxito se guarda inmediatamente en caché y cada fallo se anexa de forma idempotente; al reanudar no se repiten los videos consolidados. Los videos exclusivos para miembros, privados, retirados, sin subtítulos o con menos de 200 caracteres se registran en `datos/raw/fallos_adquisicion.jsonl` y no detienen el lote.

`01_02` no es necesario para cada incremento. Su modo rápido reentrena dos
modelos para cada longitud y su confirmación corta reentrena tres modelos en
tres cohortes pareadas; en ambos casos también calibra e infiere con la misma
longitud. La confirmación vigente conserva 30 s. Una elección manual tiene
precedencia y solo se aplica con `APPLY_CHUNK_SELECTION=True`.

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

Tras clonar, ejecute `python tools/restore_synced_checkpoints.py`. Los chunks son
baratos de reconstruir, aunque su gzip permanece en el bundle por ser entrada
de Colab; el dataset anotado se conserva comprimido porque no es barato
recrearlo. Los cuadernos `03_01`–`03_08` descomprimen el dataset si falta y
verifican que su SHA-256 coincida antes de usarlo.

El cuaderno presenta una barra por fuentes durante el descubrimiento y otra por videos durante la adquisición. Esta última distingue respuestas 429, canales excluidos y videos diferidos. Un 429 excluye solo el canal afectado durante esa corrida; no detiene ni difiere otros canales. Una URL de canal con HTTP 404 o sin pestaña `/videos` se clasifica como `stale_channel_or_no_videos_tab`; afecta esa fuente, no invalida toda la corrida.
