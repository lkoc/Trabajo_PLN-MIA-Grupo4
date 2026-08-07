# Auditoría de consolidación VTT · 2026-08-06

> **Adenda del 7 de agosto de 2026.** Este documento conserva el checkpoint histórico de la sincronización del día 6. La consolidación posterior incorporó todas las partes por canal, cachés y VTT locales disponibles: 5.002 transcripciones, 4.968 VTT sincronizados para 4.952 videos y 166.940 chunks. No se eliminó ni modificó ningún VTT. La metodología, los hashes y la estadística descriptiva vigentes están en [`MATERIALIZACION_TROCEADO.md`](MATERIALIZACION_TROCEADO.md).

## Versión sincronizada revisada

La rama `main` coincidía con `origin/main` en `b665434`. La serie reciente añadió
la optimización de longitud de chunks, checkpoints de descubrimiento reanudables,
inventario global de videos, adquisición tolerante a timeout/HTTP 429 y el corpus
JSONL sincronizable por canal. El último commit solo añadió instrucciones de
edición; los cambios de datos principales llegaron en `980ae33` y `92ce324`.

## Inventario previo

- 2.428 transcripciones en el checkpoint Git por canal.
- 1.858 transcripciones en el canónico local histórico.
- 1.377 transcripciones en tres ampliaciones históricas.
- 4.213 `video_id` únicos al unir las fuentes completas.
- 3.273 VTT locales en cuatro carpetas, equivalentes a 3.269 videos únicos.
- 983 transcripciones completas sin VTT válido; una correspondía a un VTT
  truncado sin cabecera `WEBVTT`.

La categoría objetivo de una ampliación se conserva solo como procedencia de
muestreo. No se usa como etiqueta verdadera. La clasificación sincronizada sigue
siendo por canal de YouTube y partes JSONL de hasta 25 MiB.

## Estado consolidado antes del backfill

- `transcripts_by_channel`: 4.213 videos, 317 canales y 320 partes JSONL.
- `vtt_by_video`: 3.274 archivos, 3.270 videos, 475,42 MiB.
- Cobertura válida respecto de las transcripciones: 3.230 videos.
- Cola exacta para redescarga: 983 videos en `missing_vtt.jsonl`.
- Prueba de red: `--0eNQ6N6lQ` se recuperó con `yt-dlp` en español, sin 429.

## Condición para recrear derivados

Ejecutar `01_01` con `FETCH_NEW=True`, `BACKFILL_MISSING_VTT=True`,
`MAX_VTT_BACKFILL=None`, `SYNC_VTT_BY_VIDEO=True` y
`SYNC_TRANSCRIPTS_BY_CHANNEL=True`. No iniciar `01_03`, etiquetado o entrenamiento
hasta que el resumen final indique `videos_sin_vtt = 0` e
`invalid_vtt_files = 0`, salvo que se documente expresamente un video imposible
de recuperar.
