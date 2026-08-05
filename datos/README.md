# Datos

> **Contrato activo:** `moderacion_peru_5_salidas_v2`. El texto histórico se mantiene debajo para reconstruir los artefactos anteriores; sus rutas pueden apuntar a `archivo/`.

Las nuevas corridas usan `raw/` para candidatos, caché por video y transcripciones; `processed/` para chunks deterministas; `etiquetado/` para salidas append-only; y `model_ready/v2/` para snapshots inmutables agrupados por `video_id`.

El scraping reutiliza primero los `video_id` ya canónicos y luego el caché local. Solo consulta subtítulos para candidatos nuevos. La migración v2 materializa `SEGURO` únicamente desde una decisión segura explícita; una lista histórica vacía se deriva a revisión.

## Documentación de la estructura anterior

Carpeta de trabajo para insumos y datasets generados por los cuadernos.

## Subcarpetas

- `raw/`: metadatos, subtitulos y transcripciones originales.
- `interim/`: texto normalizado y archivos intermedios.
- `processed/`: chunks listos para etiquetar y datasets finales. `chunks_para_etiquetar.jsonl` es la fuente canónica del texto.
- `etiquetado/`: exportaciones ligeras del frontend; contienen `chunk_id`, etiquetas y metadatos, pero no duplican el texto.

Las anotaciones se relacionan con el texto mediante `chunk_id`. El cuaderno 03 valida y consolida estas referencias; el cuaderno 04 realiza el cruce con `processed/chunks_para_etiquetar.jsonl` antes de entrenar.

La fase activa `04_201`–`04_208` usa `model_ready/transformer_grueso/dataset_balanceado_4a1_particionado.jsonl` (SHA-256 `df2ac01183271e44b6dcfb9cb4850bd6b1ef1cd11d9fc51c881be944670ef20f`) para comparaciones comunes. Sus splits congelados contienen 24.701 filas de train, 5.324 de validation y 5.290 de test, sin videos compartidos.

No subir datos sensibles, innecesarios o pesados. Conservar solo lo requerido para reproducibilidad academica.
