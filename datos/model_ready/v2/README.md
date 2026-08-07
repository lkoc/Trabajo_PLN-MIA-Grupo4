# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Snapshots del contrato de etiquetas v2.1

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento.

Cada snapshot se genera, no se edita manualmente. Debe acompañarse de un manifiesto con hashes, conteos, taxonomía, regla de split y fuente. Los archivos grandes permanecen fuera de Git.

El snapshot activo `dataset_5_salidas.jsonl` contiene 117 244 filas válidas: 83 492 de entrenamiento, 17 805 de validación y 15 947 de prueba, siempre agrupadas por `video_id`. Sus objetivos canónicos incluyen `ATAQUE_POR_GENERO_IDENTIDAD`; el nombre histórico se conserva únicamente en el campo de procedencia `legacy_coarse_labels`.

Validación:

```powershell
modperu validate datos/model_ready/v2/dataset_5_salidas.jsonl --kind model-ready
```

El hash y los conteos exactos están en `migracion_v2_1.manifest.json`.

Ese archivo es el snapshot migrado de arranque. Los incrementos creados por `02_05` se conservan en `snapshots/<snapshot_id>/dataset_5_salidas.jsonl` con `snapshot_manifest.json`; la ruta estable `dataset_5_salidas.jsonl` apunta por copia verificada al contenido activo. Las asignaciones existentes por `video_id` se heredan y no se deduce el video partiendo `chunk_id`.
