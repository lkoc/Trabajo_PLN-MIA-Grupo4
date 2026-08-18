# Dataset etiquetado final documentado

## Identificación académica

**Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural**

Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1

Autores: Luis Enrique Koc Góngora; Alex Felipe Mancilla Antay; Herbert Antonio Meléndez García; Dennis Jack Paitán Cano.

## Descarga del dataset

El JSON completo no se almacena en GitHub por su tamaño. El paquete documentado
se descarga desde [Google Drive](https://drive.google.com/file/d/1-fAxh7Lj0RE_Imh2SL3fKPcc_TX2R_yJ/view?usp=drivesdk).
El ZIP ocupa 63 066 165 bytes y su SHA-256 es
`c67ab95b2a3bf34a2426ca2502adb79ea3176fbaa65c7061a29512c8c6cb0ffc`.
Al descomprimirlo se obtiene el JSON de 396 581 052 bytes, cuya huella SHA-256
es `f0661dc3778b5e42a1e65e844a5c5db70fcee414685a44fb9738e17657859a49`,
junto con el esquema, el manifiesto, la procedencia y el validador. El acceso
puede requerir autorización del propietario de la carpeta de Drive.

Este paquete publica el snapshot completo usado antes de separar físicamente
`train`, `validation` y `test`. Contiene **173 240 chunks únicos** de
**4 906 videos**. Se conservaron sin cambios el texto, las
etiquetas, las máscaras, la procedencia de etiquetado y las particiones del
dataset final de entrenamiento; solo se añadieron metadatos editoriales y de
localización temporal y procedencia. No se introdujo ningún chunk sintético ni
se modificó una observación para crear la huella.

## Decisión de estructura

Los tiempos y enlaces se guardan **en el mismo registro del chunk** porque son
propiedades de ese fragmento. Separarlos en otra tabla obligaría a unir por
`chunk_id` y aumentaría el riesgo de pérdida o desalineación. El único segundo
JSON es `dataset_schema.json`: no contiene observaciones, sino el contrato
formal y la descripción legible por máquina de todos los campos. No hace falta
un tercer JSON para enlaces o estampas de tiempo.

## Archivos

- `dataset_etiquetado_final_documentado_173240.json`: arreglo JSON principal, una fila compacta por línea.
- `dataset_schema.json`: JSON Schema Draft 2020-12 y diccionario formal de campos.
- `validar_dataset.py`: auditoría reproducible con la biblioteca estándar de Python.
- `PROVENANCE.md`: diseño, verificación y límites de la huella de procedencia.
- `CITATION.cff`: metadatos de citación del dataset en Citation File Format.
- `VALIDATION_REPORT.txt`: resultado de la validación ejecutada antes de empaquetar.
- `MANIFEST.sha256`: hashes SHA-256 de todos los archivos anteriores y del README.
- `README.md`: esta guía.

## Procedencia y verificación temporal

Cada transcripción fuente se volvió a trocear con el contrato del proyecto. Un
tiempo se incorporó únicamente cuando coincidieron simultáneamente el
`chunk_id` regenerado y el texto completo. El resultado fue
**173 240/173 240 coincidencias exactas**, **cero faltantes**
y **cero conflictos**. El campo `chunk_id` permite volver a verificar la unión,
pues su hash incorpora la versión/configuración del troceador, `video_id`,
inicio, fin y texto normalizado.

- SHA-256 del JSONL canónico usado por los entrenamientos:
  `013d60ba1b173d7752f453d5d05629a3439b09c71f0c343da1b5e498662c1f86`.
- SHA-256 del JSON documentado incluido:
  `f0661dc3778b5e42a1e65e844a5c5db70fcee414685a44fb9738e17657859a49`.
- SHA-256 del esquema:
  `756008cd4266794c73e66892588ece5c8e582301889bc9d35d6fe4ffb8eeeca8`.
- Particiones conservadas: 123,239 `train`,
  27,317 `validation` y 22,684 `test`.

`timestamp_url` usa el segundo entero anterior o igual a `start_seconds` para
no omitir el comienzo del fragmento. Los campos `start_seconds` y
`start_timestamp` conservan la precisión de milisegundos.

## Regla reproducible de `chunk_id`

El identificador no es un número de fila. Primero se normaliza el texto con
Unicode NFKC, se eliminan algunos marcadores no léxicos y URL, y se compactan
los espacios. Luego se forma, respetando ese orden y el separador `|`:

```text
chunker_version|chunking_signature|video_id|start_seconds:.3f|end_seconds:.3f|texto_normalizado
```

Se calcula SHA-256 sobre la cadena UTF-8, se conservan sus primeros veinte
caracteres hexadecimales y se antepone `video_id` seguido de `_`. Por ello, un
cambio en el video, los tiempos, el texto o la configuración produce otro ID.

Este bloque reproduce exactamente la regla:

```python
import hashlib
import json
import re
import unicodedata
from pathlib import Path

def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto or "").replace("\n", " ")
    texto = re.sub(
        r"\[(musica|música|aplausos|risas|music|applause|laughter)\]",
        " ",
        texto,
        flags=re.IGNORECASE,
    )
    texto = re.sub(r"https?://\S+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

def generar_chunk_id(
    video_id: str,
    start_seconds: float,
    end_seconds: float,
    text: str,
    chunker_version: str,
    chunking_signature: str,
) -> str:
    texto = normalizar_texto(text)
    material = (
        f"{chunker_version}|{chunking_signature}|{video_id}|"
        f"{start_seconds:.3f}|{end_seconds:.3f}|{texto}"
    )
    sufijo = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{video_id}_{sufijo}"

with Path("dataset_etiquetado_final_documentado_173240.json").open(encoding="utf-8") as archivo:
    dataset = json.load(archivo)

fila = dataset[0]
assert generar_chunk_id(
    fila["video_id"],
    fila["start_seconds"],
    fila["end_seconds"],
    fila["text"],
    fila["chunker_version"],
    fila["chunking_signature"],
) == fila["chunk_id"]
```

## Huella de procedencia y citación

Cada fila incorpora tres columnas que no forman parte de las variables
predictoras: `release_id`, `provenance_token` y
`provenance_key_commitment`. El token es
`HMAC-SHA256(clave_privada, release_id|chunk_id|text_sha256)`. La clave privada
no está en este ZIP ni en el repositorio; se conserva localmente y su compromiso
público es:

```text
SHA-256(clave_privada) = 76f0e0d5f59bda914abf64575284ff89341c4442a40c5638b95dc36cdeebef07
```

La huella ayuda a reconocer filas o redistribuciones copiadas sin alterar el
texto, las etiquetas, los tiempos, las particiones o la elegibilidad de
entrenamiento. No demuestra por sí sola que un modelo haya sido entrenado con
el corpus y puede eliminarse al transformar los datos. Véase
`PROVENANCE.md` para el procedimiento completo y sus límites.

Al reutilizar el dataset se solicita conservar `release_id`, citar a los cuatro
autores y mencionar esta edición. `CITATION.cff` contiene los metadatos
listos para gestores bibliográficos.

## Diccionario de campos

| Campo | Tipo | Descripción |
|---|---|---|
| `schema_version` | string | Versión del contrato de etiquetas. |
| `taxonomy_version` | string | Versión de la taxonomía. |
| `publication_metadata_version` | string | Versión del enriquecimiento editorial. |
| `release_id` | string | Identificador público de esta edición del dataset. |
| `provenance_token` | string | HMAC-SHA256 privado que vincula edición, chunk y hash del texto. |
| `provenance_key_commitment` | string | SHA-256 público de la clave privada, sin revelar la clave. |
| `chunk_id` | string | Identificador estable y verificable del chunk. |
| `video_id` | string | Identificador del video en YouTube. |
| `video_url` | string/URI | Enlace canónico al video. |
| `start_seconds`, `end_seconds` | number | Límites exactos del chunk en segundos. |
| `start_timestamp`, `end_timestamp` | string | Límites legibles `HH:MM:SS.mmm`. |
| `timestamp_url` | string/URI | Enlace que abre el video al inicio del chunk. |
| `chunker_version` | string | Versión del algoritmo de troceado. |
| `chunking_signature` | string | SHA-256 de versión y configuración del troceador. |
| `text_sha256` | string | SHA-256 del texto normalizado en minúsculas. |
| `transcript_sha256` | string | SHA-256 de la secuencia temporal/textual normalizada de la transcripción. |
| `temporal_verification` | string | Método de aceptación de los tiempos; en este paquete, coincidencia exacta de ID y texto. |
| `source_partition` | string | Partición relativa de transcripciones usada para regenerar el chunk. |
| `source_partition_sha256` | string | SHA-256 de esa partición fuente. |
| `channel_id`, `channel_title` | string o null | Identificación del canal cuando estuvo disponible. |
| `text` | string | Texto normalizado del fragmento de subtítulos. |
| `coarse_labels` | array[string] | Etiquetas gruesas multietiqueta; `SEGURO` es excluyente. |
| `fine_labels` | array[string] | Etiquetas finas efectivamente observadas. |
| `flags_reference_only` | array[string] | Señales contextuales de referencia, no sancionadoras. |
| `coarse_observed_mask` | array[5] | Máscara de observación de las cinco salidas gruesas. |
| `fine_observed_mask` | array[14] | Máscara de observación de las catorce salidas finas. |
| `flags_observed_mask` | array[3] | Máscara de observación de los tres flags. |
| `label_source` | string | Fuente efectiva de la decisión de etiquetado. |
| `prompt_sha256` | string | SHA-256 del prompt asociado al etiquetado. |
| `sample_weight` | number | Peso de la muestra en el contrato de entrenamiento. |
| `campaign` | string o null | Campaña específica de revisión; nula en este snapshot. |
| `split` | string | Partición final: `train`, `validation` o `test`. |
| `channel_split` | string | Partición auxiliar por canal para robustez. |
| `needs_review` | boolean | Indica revisión pendiente; falso en las filas publicadas. |
| `training_eligible` | boolean | Elegibilidad para entrenamiento; verdadera en este paquete. |
| `decision_status` | string | Estado consolidado de la decisión; `resolved`. |
| `legacy_coarse_labels` | array | Etiquetas heredadas; vacío tras la migración final. |
| `label_source_original` | string o null | Fuente previa a una migración, cuando corresponde. |
| `migration_warning` | string o null | Advertencia de migración; nula en este snapshot. |

El detalle normativo de tipos, valores permitidos, longitudes y obligatoriedad
está en `dataset_schema.json`.

## Taxonomía incluida

- Gruesas: `SEGURO`, `RACISMO_DISCRIMINACION`,
  `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`.
- Finas: `seguro`, `seguro_ironia_marcada`,
  `racismo_etnico_explicito`, `racismo_encubierto`, `clasismo_racial`,
  `discriminacion_regional`, `racismo_linguistico`,
  `misoginia_acoso_genero`, `homofobia_transfobia`, `acoso_personal`,
  `amenaza_directa`, `sexual_explicito`, `sexual_cosificacion` y
  `sexual_no_consensual`.
- Flags: `humor_encubridor`, `contexto_necesario` e `ironia_ambigua`.

## Ejemplos reales de chunks

Ejemplo con etiqueta segura:

```json
{
  "release_id": "grupo4-moderacion-youtube-peru-2026-08-17-v1.1.0",
  "provenance_token": "6cbd4bfd584f5df955e53e24f1f76b76d779f828e19daee2d8b05626f45c9af1",
  "provenance_key_commitment": "76f0e0d5f59bda914abf64575284ff89341c4442a40c5638b95dc36cdeebef07",
  "chunk_id": "0cAzVPQ7qnQ_312720e5d0d045781dbd",
  "video_id": "0cAzVPQ7qnQ",
  "video_url": "https://www.youtube.com/watch?v=0cAzVPQ7qnQ",
  "start_seconds": 7.07,
  "end_seconds": 12.74,
  "start_timestamp": "00:00:07.070",
  "end_timestamp": "00:00:12.740",
  "timestamp_url": "https://www.youtube.com/watch?v=0cAzVPQ7qnQ&t=7s",
  "text": "si te gusta el deporte te gusta meter goles Ven a la cancha sintética la rinconada del Cholo juaneto",
  "coarse_labels": [
    "SEGURO"
  ],
  "fine_labels": [
    "seguro"
  ],
  "flags_reference_only": [],
  "split": "train",
  "temporal_verification": "exact_chunk_id_and_text"
}
```

Ejemplo multietiqueta con contenido potencialmente sensible:

```json
{
  "release_id": "grupo4-moderacion-youtube-peru-2026-08-17-v1.1.0",
  "provenance_token": "134b2fdd197ccc6c1dd6b8cb82a87dfa260a5b04bfb58ff439531098d737c108",
  "provenance_key_commitment": "76f0e0d5f59bda914abf64575284ff89341c4442a40c5638b95dc36cdeebef07",
  "chunk_id": "fFOxvLe2BMU_449703ac8592d6ded7c5",
  "video_id": "fFOxvLe2BMU",
  "video_url": "https://www.youtube.com/watch?v=fFOxvLe2BMU",
  "start_seconds": 180.08,
  "end_seconds": 190.76,
  "start_timestamp": "00:03:00.080",
  "end_timestamp": "00:03:10.760",
  "timestamp_url": "https://www.youtube.com/watch?v=fFOxvLe2BMU&t=180s",
  "text": "es ella misma por el pánico que le genera pensar que quien la costó la golpeó y la amenazó de muerte podría salir como sin nada en libertad",
  "coarse_labels": [
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA"
  ],
  "fine_labels": [
    "misoginia_acoso_genero",
    "acoso_personal",
    "amenaza_directa"
  ],
  "flags_reference_only": [],
  "split": "train",
  "temporal_verification": "exact_chunk_id_and_text"
}
```

Los ejemplos muestran una vista reducida; los registros originales contienen
todos los campos enumerados en el esquema.

## Carga en Python o Jupyter

```python
from pathlib import Path
import json

ruta = Path("dataset_etiquetado_final_documentado_173240.json")
with ruta.open(encoding="utf-8") as archivo:
    dataset = json.load(archivo)

print(f"Chunks: {len(dataset):,}")
print(dataset[0]["timestamp_url"])
print(dataset[0]["text"])
```

Para crear una tabla:

```python
import pandas as pd

columnas = [
    "chunk_id", "video_id", "start_seconds", "end_seconds",
    "text", "coarse_labels", "fine_labels", "split"
]
df = pd.DataFrame(dataset)[columnas]
df.head()
```

## Validación e integridad

Desde la carpeta extraída:

```bash
python validar_dataset.py
```

El validador comprueba hashes del manifiesto, conteos, campos obligatorios,
tipos, máscaras, vocabularios, unicidad de `chunk_id`, coherencia de enlaces y
estampas, hashes de texto, tokens únicos y la reconstrucción de cada `chunk_id`.
El custodio puede verificar además los HMAC con la clave privada, sin copiarla
al paquete público:

```bash
python validar_dataset.py --provenance-key "/ruta/privada/publication_provenance_hmac.key"
```

En PowerShell también puede revisar un hash individual:

```powershell
Get-FileHash .\dataset_etiquetado_final_documentado_173240.json -Algorithm SHA256
```

## Alcance y precauciones

Los textos proceden de subtítulos y pueden contener errores de transcripción,
lenguaje dañino o información sensible. Un enlace puede dejar de estar
disponible si el propietario retira o restringe el video. El paquete no incluye
audio ni video y no concede una licencia nueva sobre contenido de terceros. Su
uso debe ser académico o evaluativo, respetar los términos de la plataforma y
mantener revisión humana; las etiquetas no deben emplearse por sí solas para
aplicar sanciones automáticas.
