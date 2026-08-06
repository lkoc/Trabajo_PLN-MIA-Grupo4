# Snapshots v2.1

Cada snapshot se genera, no se edita manualmente. Debe acompañarse de un manifiesto con hashes, conteos, taxonomía, regla de split y fuente. Los archivos grandes permanecen fuera de Git.

El snapshot activo `dataset_5_salidas.jsonl` contiene 117 244 filas válidas: 83 492 de entrenamiento, 17 805 de validación y 15 947 de prueba, siempre agrupadas por `video_id`. Sus objetivos canónicos incluyen `ATAQUE_POR_GENERO_IDENTIDAD`; el nombre histórico se conserva únicamente en el campo de procedencia `legacy_coarse_labels`.

Validación:

```powershell
modperu validate datos/model_ready/v2/dataset_5_salidas.jsonl --kind model-ready
```

El hash y los conteos exactos están en `migracion_v2_1.manifest.json`.

Ese archivo es el snapshot migrado de arranque. Los incrementos creados por `02_05` se conservan en `snapshots/<snapshot_id>/dataset_5_salidas.jsonl` con `snapshot_manifest.json`; la ruta estable `dataset_5_salidas.jsonl` apunta por copia verificada al contenido activo. Las asignaciones existentes por `video_id` se heredan y no se deduce el video partiendo `chunk_id`.
