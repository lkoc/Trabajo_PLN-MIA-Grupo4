# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Datos

> **Contrato de etiquetas activo v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales. El texto histórico se mantiene debajo para reconstruir los artefactos anteriores; sus rutas pueden apuntar a `archivo/`.

Las nuevas corridas usan `raw/` para candidatos, caché por video y transcripciones; `processed/` para chunks deterministas; `etiquetado/` para salidas append-only; y `model_ready/v2/` para snapshots inmutables agrupados por `video_id`.

El scraping reutiliza primero los `video_id` ya canónicos y luego el caché local. Solo consulta subtítulos para candidatos nuevos: `yt-dlp` escribe VTT sin audio/video, se exige un mínimo de 200 caracteres y toda la cola se recorre en lotes de 10 con una pausa de 60 segundos. La cola pseudoaleatoria es reproducible e intercala canales; un 429 difiere solo el canal afectado. Los JSON de caché y el registro inmediato de fallos permiten reanudar. La migración v2 materializa `SEGURO` únicamente desde una decisión segura explícita; una lista histórica vacía se deriva a revisión.

La vista local `raw/transcripts_raw.jsonl` se conserva, pero Git sincroniza su
partición idempotente `raw/transcripts_by_channel/`. Cada archivo corresponde a
un canal y `index.json` registra cantidad, tamaño y SHA-256. Los candidatos,
fallos y manifiestos también se sincronizan; `raw/transcripts_cache/` no. Tras clonar,
`python tools/restore_synced_checkpoints.py` recompone el canónico y restaura
las entradas comprimidas del bundle sin repetir descargas.

El corte del 7 de agosto contiene 5.002 videos distribuidos en 339 partes y 336
claves de canal o procedencia. La parte mayor ocupa 26.061.145 bytes; ninguna
supera 50 MB. El canónico local ocupa 455.104.683 bytes y se reconstruye desde
esas partes, por lo que continúa fuera de Git.

Git también sincroniza `raw/vtt_by_video/`: contiene todas las pistas WebVTT
consolidadas, un índice con SHA-256 y `missing_vtt.jsonl`. `01_01` consume esa
cola aunque el JSON del video ya exista y guarda cada VTT antes de avanzar.

`processed/chunks_v2.jsonl` es determinista y barato de regenerar con `01_03`;
solo se conserva comprimido dentro del bundle para Colab. El dataset final no
es barato de recrear porque contiene decisiones humanas y pseudoetiquetado con
procedencia: se sincroniza como `resultados/colab_bundle/dataset_5_salidas.jsonl.gz`
y los cuadernos de entrenamiento verifican su SHA-256 antes de usarlo.

`processed/chunk_materialization_manifest.json` sí se sincroniza: no contiene
texto del corpus y registra hashes, conteos, cobertura y estadística descriptiva
de los 166.940 chunks activos. El informe legible está en
[`docs/MATERIALIZACION_TROCEADO.md`](../docs/MATERIALIZACION_TROCEADO.md).

La longitud activa está declarada en `config/chunking.json`. El identificador
del chunk incorpora la firma completa de esa configuración. Si se cambia desde
`01_02`, los derivados de la firma anterior se archivan localmente sin borrarse;
al volver a una longitud ya usada se restauran con verificación SHA-256.

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
