# Consolidación y materialización reproducible de chunks

**Fecha de corte:** 7 de agosto de 2026  
**Cuaderno:** `flujo/01_datos/01_03_limpieza_troceado_incremental.ipynb`  
**Configuración activa:** 30 s, 600 caracteres, mínimo de 90 caracteres y eliminación de hasta 12 palabras solapadas  
**Resultado:** 166.940 chunks correspondientes a 4.992 videos

En este documento, `v2.2` designa la familia actual del troceador (`chunker_version=2.2.0`). El contrato de etiquetas continúa en `v2.1`; ambos números tienen alcance y ciclo de vida independientes.

## Alcance y criterio de fuente

La unidad elegible es un video con una transcripción completa local. Un candidato descubierto, un registro de fallo o un `video_id` sin texto no se trata como transcripción disponible. La vista de trabajo `datos/raw/transcripts_raw.jsonl` es monolítica y local; no se sincroniza con Git porque pesa 455.104.683 bytes. Su fuente sincronizable son las particiones `datos/raw/transcripts_by_channel/*.jsonl` y su índice SHA-256.

La consolidación aplica esta precedencia por `video_id`:

1. particiones sincronizadas por canal;
2. vista canónica local existente;
3. snapshots completos bajo `datos/`;
4. cachés JSON por video;
5. VTT locales que todavía no tengan representación JSON.

Los VTT se leen sin modificarlos. Cuando un video tiene varias pistas, se selecciona determinísticamente la que contiene más texto útil; se conserva la lista completa como procedencia. Se exige un mínimo de 200 caracteres. Los VTT más cortos permanecen almacenados, pero no se convierten en transcripciones entrenables.

## Repetición e incremento desde los cuadernos

El uso normal es:

1. ejecutar `01_01`, que descarga únicamente candidatos pendientes y guarda inmediatamente cada VTT, caché JSON y parte por canal;
2. ejecutar `01_03`, que recompone el canónico desde los checkpoints locales, recupera VTT utilizables y actualiza las partes por canal;
3. `01_03` compara `(video_id, transcript_sha256, chunking_signature)` y procesa solo videos nuevos o modificados;
4. la celda de materialización muestra una barra `tqdm` por video y escribe `datos/processed/chunk_materialization_manifest.json`;
5. `REBUILD_CHUNKS_FROM_ZERO=True` fuerza una reconstrucción total y crea antes una copia recuperable en `archivo/chunk_rebuilds/`. El valor normal es `False`.

La reconstrucción completa ejecutada en este corte tomó 133,58 s. Una repetición inmediata con la misma entrada examinó las 5.002 transcripciones, detectó 5.002 sin cambios y añadió cero chunks o versiones.

## Reglas del troceador

El texto se normaliza mediante Unicode NFKC [1]. Se eliminan marcadores frecuentes de música, aplausos o risas y se normalizan espacios. En subtítulos rodantes se retiran hasta 12 palabras repetidas entre segmentos consecutivos. Un chunk se cierra al alcanzar 30 s o 600 caracteres y se descarta si, tras limpiar, contiene menos de 90 caracteres.

Los identificadores incorporan `video_id`, tiempos, texto normalizado, hash de la transcripción, versión del troceador y firma completa de configuración. SHA-256 se usa como función de huella [2]. La deduplicación conserva un único `chunk_id` y un único hash de texto normalizado en todo el corpus.

El campo histórico `transcript_sha256` se interpreta por capa: el canónico de adquisición hashea el JSON crudo de segmentos, mientras los chunks y `chunking_v2_versions.jsonl` hashean la secuencia temporal y textual normalizada por el troceador. Esas huellas no se comparan directamente. En la auditoría final se recalculó la segunda para las 5.002 transcripciones: los 166.940 chunks y las 5.002 versiones coincidieron, sin chunks huérfanos ni hashes de texto inválidos.

Los límites de 30 s y 600 caracteres son umbrales de cierre, no cortes dentro de una pista VTT individual. Si una sola pista supera un límite, se conserva completa para no fabricar tiempos o dividir una unidad de subtítulo sin evidencia. Esto explica máximos observados de 263,26 s y 816 caracteres; el percentil 95 permanece en 33,95 s y 613 caracteres.

## Consolidación cuantitativa

| Medida | Resultado |
|---|---:|
| Videos en el canónico anterior | 4.970 |
| VTT sin JSON detectados | 39 |
| VTT recuperados al canónico | 32 |
| VTT excluidos por menos de 200 caracteres | 7 |
| Pistas inválidas entre los VTT recuperables | 0 |
| Videos en el canónico consolidado | 5.002 |
| Partes JSONL sincronizables | 339 |
| Claves de canal o procedencia | 336 |
| Mayor parte por canal | 26.061.145 bytes |
| Partes mayores de 50 MB | 0 |
| VTT en el checkpoint sincronizado | 4.968 pistas / 4.952 videos |
| Videos canónicos con al menos un VTT | 4.945 |
| Videos canónicos aún sin VTT | 57 |

La cohorte dirigida contiene 1.043 candidatos, pero un candidato no equivale a una descarga completa: 871 ya tienen transcripción canónica, 31 tienen fallo sin éxito posterior y 141 todavía no poseen éxito ni fallo registrado. Además, 48 de los 871 éxitos habían fallado en un intento anterior y luego se recuperaron. Por ello, el incremento efectivo desde el checkpoint documentado de 4.213 videos es de 789 transcripciones, no de 1.043.

## Resultado del troceado

| Medida | Antes | Corte consolidado | Diferencia |
|---|---:|---:|---:|
| Chunks | 166.584 | 166.940 | +356 |
| Videos con chunks | 4.960 | 4.992 | +32 |
| Versiones de video registradas | 3.599 | 5.002 | +1.403 |

Los 32 videos recuperados desde VTT produjeron exactamente 356 chunks. La reconstrucción conservó 119.135 identificadores del troceador v2.2 ya existentes, sustituyó 47.449 identificadores heredados del troceador v2.1 y creó 47.805 identificadores del troceador v2.2; estos últimos incluyen las 47.449 sustituciones deterministas y los 356 chunks de videos recuperados.

La cobertura es 4.992/5.002 videos, equivalente a 99,80 %. Diez transcripciones no tienen chunks finales. Cinco no alcanzaron el mínimo de 90 caracteres después de limpiar (`0gZaIyxBANk`, `8QGI_WoaN-I`, `iDHUBY-MOL4`, `sOiJB14im-g`, `yO-RcPkIpDc`). Las otras cinco produjeron texto, pero todos sus chunks quedaron eliminados por duplicación global (`1MPPLtM6BA4`, `NGBMshQT35I`, `ObvZFla0Zq4`, `su8i8FN0V-8`, `yEgGOJ37VsY`). No existen chunks cuyo `video_id` esté ausente del canónico.

## Estadística descriptiva

| Variable | Mín. | P25 | Mediana | Media | P75 | P90 | P95 | Máx. | Desv. estándar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chunks por video | 1 | 6 | 18 | 33,44 | 35 | 86 | 158 | 767 | 46,80 |
| Duración por chunk, s | 4,01 | 30,48 | 31,16 | 31,29 | 31,96 | 32,89 | 33,95 | 263,26 | 3,89 |
| Caracteres por chunk | 90 | 408 | 483 | 466,58 | 545 | 598 | 613 | 816 | 107,82 |
| Palabras por chunk | 10 | 73 | 87 | 85,29 | 100 | 110 | 116 | 170 | 20,61 |

Los 166.940 chunks contienen 77.890.890 caracteres y 14.238.347 palabras. La suma de sus intervalos temporales es 1.451,17 horas; esta cifra representa spans de subtítulos y no duración audiovisual medida ni tiempo de habla efectivo.

## Relación con etiquetado y entrenamiento

`datos/model_ready/v2/dataset_5_salidas.jsonl` conserva 117.244 filas y 3.230 videos del snapshot etiquetado anterior. No debe crecer al ejecutar `01_03`: los chunks nuevos todavía no tienen decisión bajo el contrato de etiquetas v2.1. Además, la reconstrucción homogénea con el troceador v2.2 reemplazó los identificadores heredados, por lo que el snapshot anterior no se une al nuevo artefacto por `chunk_id`.

El siguiente ciclo correcto es ejecutar `02_00` localmente para publicar los chunks requeridos por Colab, completar `02_01`–`02_05` —generar propuestas para los chunks v2.2, resolver o diferir los casos, adjudicar cuando corresponda y crear un snapshot nuevo e inmutable— y repetir `02_00` para publicar ese snapshot antes del entrenamiento remoto. El snapshot anterior se conserva como evidencia histórica, no como etiquetas trasladables automáticamente.

## Artefactos verificables

| Artefacto | Filas / videos | SHA-256 |
|---|---:|---|
| `datos/raw/transcripts_raw.jsonl` | 5.002 videos | `a03a3800d4bdd15a7462bc8d1f43e88301396454fe6acdd55bfafda80303ee79` |
| `datos/raw/transcripts_by_channel/index.json` | 339 partes | `c56e1c6304b1c622ab79a833e2c76f285bb18c809ad0f586aee6764e47a7255b` |
| `datos/processed/chunks_v2.jsonl` | 166.940 chunks | `2506123ed7a9d78fcf466e1af8875d96a70651ae8b4c22a1e8e13ccd1c542828` |
| `datos/processed/chunking_v2_versions.jsonl` | 5.002 versiones | `8d60210f8c01fe690c2b1d20819c9de475b85a615949ee63a9aee34f22549d69` |

El troceado anterior es recuperable localmente desde `archivo/chunk_rebuilds/20260807T165807714288Z`. Los respaldos completos de esta carpeta se excluyen de Git por su tamaño; el manifiesto activo sincronizable conserva conteos, cobertura, estadística descriptiva, hashes y la referencia al respaldo local. Los VTT no forman parte de la operación destructiva: al actualizar su índice se verificaron 4.968 archivos antes y después, con cero eliminados y cero modificados.

## Referencias

[1] Unicode Consortium, “Unicode Normalization Forms,” *Unicode Standard Annex #15*, rev. 57, 2025. [En línea]. Disponible en: https://www.unicode.org/reports/tr15/tr15-57.html

[2] National Institute of Standards and Technology, “Secure Hash Standard (SHS),” *FIPS PUB 180-4*, 2015, doi: 10.6028/NIST.FIPS.180-4.
