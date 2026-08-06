# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 01 · Scraping, limpieza y troceado

## Orden

1. `01_01_scraping_incremental.ipynb`
2. `01_02_limpieza_troceado_incremental.ipynb`

Entrada: candidatos con `video_id` y URL, snapshots históricos, transcripciones canónicas y caché local.  
Salida: transcripciones JSONL y chunks v2 con tiempos, hash de transcripción y versión del troceador.  
Control: nunca se descarga audio o video; primero se consolidan sin modificarlos los `transcripts_raw.jsonl` ya existentes, después se reutiliza el caché y solo al final se consulta la red para un `video_id` nuevo. La limpieza conserva la eliminación de hasta 12 palabras solapadas en subtítulos rodantes, el cierre a 30 segundos/600 caracteres y el mínimo de 90 caracteres.

`01_01` reúne el scraping inicial y la antigua ampliación dirigida. Su bloque de controles permite elegir `DISCOVERY_MODE="seed"`, `"directed"` o `"both"`; editar canales y consultas; limitar videos por canal, resultados por búsqueda y descargas nuevas; y configurar reintentos y pausas. `directed` usa los déficits de `train+validation`, rendimiento histórico por canal, expansión desde búsquedas y una cohorte aislada; sin etiquetas previas activa un fallback equitativo para los cuatro daños. `DISCOVER_NEW=False` no consulta fuentes y `FETCH_NEW=False` no obtiene subtítulos.

La opción comentada `RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"` mueve los artefactos activos a `archivo/reinicios_dataset_videos/` y bloquea la reimportación automática de snapshots históricos. Es recuperable e idempotente; no borra código ni el archivo histórico.

Para ampliar la muestra, active el modo requerido o agregue filas a `datos/raw/video_candidates.jsonl` o `datos/raw/videos_candidatos.csv` y vuelva a ejecutar `01_01`. El corpus previo permanece intacto. Los videos exclusivos para miembros, privados, retirados o sin subtítulos se registran en `datos/raw/fallos_adquisicion.jsonl` y no detienen el lote.

El cuaderno presenta una barra por fuentes durante el descubrimiento y otra por videos durante la adquisición. Al finalizar muestra fuentes correctas/fallidas, candidatos únicos/nuevos, reutilizaciones, descargas, diferidos y fallos por motivo. Una URL de canal con HTTP 404 o sin pestaña `/videos` se clasifica como `stale_channel_or_no_videos_tab`; afecta esa fuente, no invalida toda la corrida.
