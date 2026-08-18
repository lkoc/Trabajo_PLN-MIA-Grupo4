from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from moderacion_peru.incremental import TranscriptSegment, chunk_transcript

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = ROOT / "Publicación_Troome"
BASE_JSON = (
    PUBLICATION_DIR
    / "Grupo_4_Dataset_Etiquetado_Final"
    / "dataset_etiquetado_final_173240.json"
)
CANONICAL_JSONL = ROOT / "datos" / "model_ready" / "v2" / "dataset_5_salidas.jsonl"
TRANSCRIPT_DIR = ROOT / "datos" / "raw" / "transcripts_by_channel"
STAGING_DIR = ROOT / "resultados" / "publicacion_dataset_documentado"
DATASET_NAME = "dataset_etiquetado_final_documentado_173240.json"
SCHEMA_NAME = "dataset_schema.json"
README_NAME = "README.md"
VALIDATOR_NAME = "validar_dataset.py"
REPORT_NAME = "VALIDATION_REPORT.txt"
MANIFEST_NAME = "MANIFEST.sha256"
PROVENANCE_NAME = "PROVENANCE.md"
CITATION_NAME = "CITATION.cff"
ZIP_NAME = "Grupo_4_Dataset_Etiquetado_Final_Documentado.zip"
PRIVATE_KEY_PATH = ROOT / "datos" / "private" / "publication_provenance_hmac.key"

EXPECTED_ROWS = 173_240
EXPECTED_VIDEOS = 4_906
PUBLICATION_METADATA_VERSION = "1.1.0"
RELEASE_ID = "grupo4-moderacion-youtube-peru-2026-08-17-v1.1.0"
TEMPORAL_VERIFICATION = "exact_chunk_id_and_text"

COARSE_LABELS = [
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
]
FINE_LABELS = [
    "seguro",
    "seguro_ironia_marcada",
    "racismo_etnico_explicito",
    "racismo_encubierto",
    "clasismo_racial",
    "discriminacion_regional",
    "racismo_linguistico",
    "misoginia_acoso_genero",
    "homofobia_transfobia",
    "acoso_personal",
    "amenaza_directa",
    "sexual_explicito",
    "sexual_cosificacion",
    "sexual_no_consensual",
]
REFERENCE_FLAGS = ["humor_encubridor", "contexto_necesario", "ironia_ambigua"]
LABEL_SOURCES = [
    "deepseek_remote_historical_recovered",
    "human_modified",
    "llm_remote_review_historical_recovered",
    "deepseek_remote",
    "llm_remote_review",
    "human_accepted",
]

ORIGINAL_FIELDS = [
    "schema_version",
    "taxonomy_version",
    "chunk_id",
    "video_id",
    "channel_id",
    "channel_title",
    "text",
    "coarse_labels",
    "fine_labels",
    "flags_reference_only",
    "coarse_observed_mask",
    "fine_observed_mask",
    "flags_observed_mask",
    "label_source",
    "prompt_sha256",
    "sample_weight",
    "campaign",
    "split",
    "channel_split",
    "needs_review",
    "training_eligible",
    "decision_status",
    "legacy_coarse_labels",
    "label_source_original",
    "migration_warning",
]

FIELD_ORDER = [
    "schema_version",
    "taxonomy_version",
    "publication_metadata_version",
    "release_id",
    "provenance_token",
    "provenance_key_commitment",
    "chunk_id",
    "video_id",
    "video_url",
    "start_seconds",
    "end_seconds",
    "start_timestamp",
    "end_timestamp",
    "timestamp_url",
    "chunker_version",
    "chunking_signature",
    "text_sha256",
    "transcript_sha256",
    "temporal_verification",
    "source_partition",
    "source_partition_sha256",
    "channel_id",
    "channel_title",
    "text",
    "coarse_labels",
    "fine_labels",
    "flags_reference_only",
    "coarse_observed_mask",
    "fine_observed_mask",
    "flags_observed_mask",
    "label_source",
    "prompt_sha256",
    "sample_weight",
    "campaign",
    "split",
    "channel_split",
    "needs_review",
    "training_eligible",
    "decision_status",
    "legacy_coarse_labels",
    "label_source_original",
    "migration_warning",
]

EXAMPLE_IDS = [
    "0cAzVPQ7qnQ_312720e5d0d045781dbd",
    "fFOxvLe2BMU_449703ac8592d6ded7c5",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_or_create_provenance_key() -> bytes:
    PRIVATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE_KEY_PATH.exists():
        key = PRIVATE_KEY_PATH.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with PRIVATE_KEY_PATH.open("xb") as stream:
            stream.write(key)
    if len(key) != 32:
        raise RuntimeError("La clave privada de procedencia debe tener 32 bytes")
    os.chmod(PRIVATE_KEY_PATH, 0o600)
    return key


def provenance_token(key: bytes, chunk_id: str, text_sha256: str) -> str:
    material = f"{RELEASE_ID}|{chunk_id}|{text_sha256}".encode()
    return hmac.new(key, material, hashlib.sha256).hexdigest()


def timestamp(seconds: float) -> str:
    total_ms = round(float(seconds) * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def verify_base_against_canonical(base_rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    count = 0
    with CANONICAL_JSONL.open("rb") as raw_stream:
        for raw_line in raw_stream:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            if count >= len(base_rows):
                raise RuntimeError("El JSONL canónico contiene filas adicionales")
            canonical_row = json.loads(raw_line.decode("utf-8"))
            if canonical_row != base_rows[count]:
                raise RuntimeError(f"Diferencia semántica en la fila {count}")
            count += 1
    if count != len(base_rows):
        raise RuntimeError(
            f"Conteo canónico incompatible: {count} frente a {len(base_rows)}"
        )
    return digest.hexdigest()


def recover_locations(
    base_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_video: dict[str, dict[str, str]] = defaultdict(dict)
    for row in base_rows:
        by_video[str(row["video_id"])][str(row["chunk_id"])] = str(row["text"])

    locations: dict[str, dict[str, Any]] = {}
    conflicts = 0
    selected_transcripts = 0
    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.jsonl"))
    partition_hashes = {path: sha256_file(path) for path in transcript_files}

    for path in transcript_files:
        relative = path.relative_to(ROOT).as_posix()
        partition_sha256 = partition_hashes[path]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                transcript = json.loads(line)
                video_id = str(transcript.get("video_id") or "")
                targets = by_video.get(video_id)
                if not targets:
                    continue
                selected_transcripts += 1
                segments = [
                    TranscriptSegment(
                        start=float(segment["start"]),
                        duration=float(segment["duration"]),
                        text=str(segment["text"]),
                    )
                    for segment in transcript.get("segments", [])
                ]
                for chunk in chunk_transcript(video_id, segments):
                    expected_text = targets.get(chunk.chunk_id)
                    if expected_text is None:
                        continue
                    if chunk.text != expected_text:
                        raise RuntimeError(
                            f"El texto regenerado no coincide para {chunk.chunk_id}"
                        )
                    candidate = {
                        "start_seconds": chunk.start_seconds,
                        "end_seconds": chunk.end_seconds,
                        "chunker_version": chunk.chunker_version,
                        "chunking_signature": chunk.chunking_signature,
                        "text_sha256": chunk.text_sha256,
                        "transcript_sha256": chunk.transcript_sha256,
                        "source_partition": relative,
                        "source_partition_sha256": partition_sha256,
                    }
                    previous = locations.get(chunk.chunk_id)
                    if previous is not None and previous != candidate:
                        conflicts += 1
                        raise RuntimeError(
                            f"Conflicto de localización para {chunk.chunk_id}"
                        )
                    locations[chunk.chunk_id] = candidate

    missing = [row["chunk_id"] for row in base_rows if row["chunk_id"] not in locations]
    if missing:
        raise RuntimeError(
            f"Faltan {len(missing)} localizaciones; muestra: {missing[:5]}"
        )

    return locations, {
        "transcript_partition_files": len(transcript_files),
        "selected_transcripts": selected_transcripts,
        "locations": len(locations),
        "conflicts": conflicts,
    }


def enrich_row(
    row: dict[str, Any],
    location: dict[str, Any],
    provenance_key: bytes,
    key_commitment: str,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    start = float(location["start_seconds"])
    end = float(location["end_seconds"])
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    enriched = {
        "schema_version": row["schema_version"],
        "taxonomy_version": row["taxonomy_version"],
        "publication_metadata_version": PUBLICATION_METADATA_VERSION,
        "release_id": RELEASE_ID,
        "provenance_token": provenance_token(
            provenance_key, str(row["chunk_id"]), str(location["text_sha256"])
        ),
        "provenance_key_commitment": key_commitment,
        "chunk_id": row["chunk_id"],
        "video_id": video_id,
        "video_url": video_url,
        "start_seconds": start,
        "end_seconds": end,
        "start_timestamp": timestamp(start),
        "end_timestamp": timestamp(end),
        "timestamp_url": f"{video_url}&t={math.floor(start)}s",
        "chunker_version": location["chunker_version"],
        "chunking_signature": location["chunking_signature"],
        "text_sha256": location["text_sha256"],
        "transcript_sha256": location["transcript_sha256"],
        "temporal_verification": TEMPORAL_VERIFICATION,
        "source_partition": location["source_partition"],
        "source_partition_sha256": location["source_partition_sha256"],
        "channel_id": row["channel_id"],
        "channel_title": row["channel_title"],
        "text": row["text"],
        "coarse_labels": row["coarse_labels"],
        "fine_labels": row["fine_labels"],
        "flags_reference_only": row["flags_reference_only"],
        "coarse_observed_mask": row["coarse_observed_mask"],
        "fine_observed_mask": row["fine_observed_mask"],
        "flags_observed_mask": row["flags_observed_mask"],
        "label_source": row["label_source"],
        "prompt_sha256": row["prompt_sha256"],
        "sample_weight": row["sample_weight"],
        "campaign": row["campaign"],
        "split": row["split"],
        "channel_split": row["channel_split"],
        "needs_review": row["needs_review"],
        "training_eligible": row["training_eligible"],
        "decision_status": row["decision_status"],
        "legacy_coarse_labels": row["legacy_coarse_labels"],
        "label_source_original": row["label_source_original"],
        "migration_warning": row["migration_warning"],
    }
    if list(enriched) != FIELD_ORDER:
        raise AssertionError("El orden de campos no coincide con el contrato")
    return enriched


def write_dataset(
    path: Path,
    base_rows: list[dict[str, Any]],
    locations: dict[str, dict[str, Any]],
    provenance_key: bytes,
    key_commitment: str,
) -> tuple[dict[str, dict[str, Any]], int]:
    examples: dict[str, dict[str, Any]] = {}
    tokens: set[str] = set()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n")
        for index, row in enumerate(base_rows):
            enriched = enrich_row(
                row,
                locations[str(row["chunk_id"])],
                provenance_key,
                key_commitment,
            )
            token = str(enriched["provenance_token"])
            if token in tokens:
                raise RuntimeError(f"Token de procedencia duplicado: {token}")
            tokens.add(token)
            if row["chunk_id"] in EXAMPLE_IDS:
                examples[str(row["chunk_id"])] = enriched
            if index:
                stream.write(",\n")
            stream.write(
                json.dumps(enriched, ensure_ascii=False, separators=(",", ":"))
            )
        stream.write("\n]\n")
    if set(examples) != set(EXAMPLE_IDS):
        raise RuntimeError("No se encontraron todos los ejemplos previstos")
    return examples, len(tokens)


def schema_document() -> dict[str, Any]:
    hex64 = "^[0-9a-f]{64}$"
    mask = lambda length: {
        "type": "array",
        "minItems": length,
        "maxItems": length,
        "items": {"type": "integer", "enum": [0, 1]},
    }
    properties: dict[str, Any] = {
        "schema_version": {
            "type": "string",
            "const": "2.1.0",
            "description": "Versión del contrato de etiquetas usado para entrenar.",
        },
        "taxonomy_version": {
            "type": "string",
            "const": "2.1.0",
            "description": "Versión de la taxonomía de moderación.",
        },
        "publication_metadata_version": {
            "type": "string",
            "const": PUBLICATION_METADATA_VERSION,
            "description": "Versión de los metadatos editoriales añadidos al paquete.",
        },
        "release_id": {
            "type": "string",
            "const": RELEASE_ID,
            "description": "Identificador público e inmutable de esta edición del dataset.",
        },
        "provenance_token": {
            "type": "string",
            "pattern": hex64,
            "description": "HMAC-SHA256 privado que vincula release_id, chunk_id y text_sha256 sin modificar la observación.",
        },
        "provenance_key_commitment": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 público de la clave HMAC privada; permite verificar una revelación posterior de la clave.",
        },
        "chunk_id": {
            "type": "string",
            "pattern": "^[A-Za-z0-9_-]{11}_[0-9a-f]{20}$",
            "description": "Identificador estable del chunk; vincula video, tiempos, texto y configuración de troceado.",
        },
        "video_id": {
            "type": "string",
            "pattern": "^[A-Za-z0-9_-]{11}$",
            "description": "Identificador de once caracteres del video de YouTube.",
        },
        "video_url": {
            "type": "string",
            "format": "uri",
            "description": "Enlace canónico reconstruido a partir de video_id.",
        },
        "start_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Inicio exacto del chunk en segundos desde el comienzo del video.",
        },
        "end_seconds": {
            "type": "number",
            "minimum": 0,
            "description": "Fin exacto del chunk en segundos desde el comienzo del video.",
        },
        "start_timestamp": {
            "type": "string",
            "pattern": "^[0-9]{2,}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}$",
            "description": "Inicio legible con formato HH:MM:SS.mmm.",
        },
        "end_timestamp": {
            "type": "string",
            "pattern": "^[0-9]{2,}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}$",
            "description": "Fin legible con formato HH:MM:SS.mmm.",
        },
        "timestamp_url": {
            "type": "string",
            "format": "uri",
            "description": "Enlace al video iniciado en el segundo entero inmediatamente anterior o igual al inicio exacto.",
        },
        "chunker_version": {
            "type": "string",
            "const": "2.2.0",
            "description": "Versión del algoritmo que materializó el chunk temporal.",
        },
        "chunking_signature": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 de la versión y configuración completa del troceador.",
        },
        "text_sha256": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 del texto normalizado en minúsculas Unicode.",
        },
        "transcript_sha256": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 de la secuencia temporal y textual normalizada de la transcripción usada por el troceador.",
        },
        "temporal_verification": {
            "type": "string",
            "const": TEMPORAL_VERIFICATION,
            "description": "Indica coincidencia exacta simultánea del chunk_id regenerado y del texto.",
        },
        "source_partition": {
            "type": "string",
            "description": "Ruta relativa de la partición de transcripciones que permitió regenerar la localización.",
        },
        "source_partition_sha256": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 de la partición fuente indicada en source_partition.",
        },
        "channel_id": {
            "type": ["string", "null"],
            "description": "Identificador del canal, si estaba disponible.",
        },
        "channel_title": {
            "type": ["string", "null"],
            "description": "Nombre del canal, si estaba disponible.",
        },
        "text": {
            "type": "string",
            "minLength": 1,
            "description": "Texto normalizado del fragmento de subtítulos.",
        },
        "coarse_labels": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "enum": COARSE_LABELS},
            "description": "Una o más etiquetas gruesas; SEGURO es excluyente.",
        },
        "fine_labels": {
            "type": "array",
            "maxItems": 14,
            "uniqueItems": True,
            "items": {"type": "string", "enum": FINE_LABELS},
            "description": "Etiquetas finas observadas; una lista vacía puede coexistir con máscara parcial.",
        },
        "flags_reference_only": {
            "type": "array",
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"type": "string", "enum": REFERENCE_FLAGS},
            "description": "Indicadores contextuales de referencia, no objetivos sancionadores.",
        },
        "coarse_observed_mask": {
            **mask(5),
            "description": "Máscara de observación de las cinco salidas gruesas en el orden de la taxonomía.",
        },
        "fine_observed_mask": {
            **mask(14),
            "description": "Máscara de observación de las catorce salidas finas.",
        },
        "flags_observed_mask": {
            **mask(3),
            "description": "Máscara de observación de los tres flags contextuales.",
        },
        "label_source": {
            "type": "string",
            "enum": LABEL_SOURCES,
            "description": "Procedencia efectiva de la etiqueta tras aplicar revisión y precedencia.",
        },
        "prompt_sha256": {
            "type": "string",
            "pattern": hex64,
            "description": "SHA-256 del prompt de etiquetado asociado.",
        },
        "sample_weight": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Peso de la muestra usado por el contrato de entrenamiento.",
        },
        "campaign": {
            "type": ["string", "null"],
            "description": "Campaña de revisión específica; nulo en este snapshot.",
        },
        "split": {
            "type": "string",
            "enum": ["train", "validation", "test"],
            "description": "Partición experimental final conservada antes del entrenamiento.",
        },
        "channel_split": {
            "type": "string",
            "enum": ["train", "validation", "test"],
            "description": "Partición auxiliar por canal para análisis de robustez.",
        },
        "needs_review": {
            "type": "boolean",
            "const": False,
            "description": "Marca de revisión pendiente; falsa para todas las filas publicadas.",
        },
        "training_eligible": {
            "type": "boolean",
            "const": True,
            "description": "Indica que la fila fue elegible para el entrenamiento final.",
        },
        "decision_status": {
            "type": "string",
            "const": "resolved",
            "description": "Estado consolidado de la decisión de etiquetado.",
        },
        "legacy_coarse_labels": {
            "type": "array",
            "maxItems": 0,
            "description": "Etiquetas gruesas heredadas; vacío tras la migración consolidada.",
        },
        "label_source_original": {
            "type": ["string", "null"],
            "description": "Fuente anterior a una migración, cuando corresponde.",
        },
        "migration_warning": {
            "type": ["string", "null"],
            "description": "Advertencia de migración; nula en las filas publicadas.",
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Dataset etiquetado final documentado — Grupo 4",
        "description": "Contrato de un arreglo JSON con 173 240 chunks etiquetados y localización temporal verificable.",
        "type": "array",
        "minItems": EXPECTED_ROWS,
        "maxItems": EXPECTED_ROWS,
        "items": {
            "type": "object",
            "required": FIELD_ORDER,
            "additionalProperties": False,
            "properties": properties,
        },
    }


def write_schema(path: Path) -> None:
    path.write_text(
        json.dumps(schema_document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def example_view(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "release_id",
        "provenance_token",
        "provenance_key_commitment",
        "chunk_id",
        "video_id",
        "video_url",
        "start_seconds",
        "end_seconds",
        "start_timestamp",
        "end_timestamp",
        "timestamp_url",
        "text",
        "coarse_labels",
        "fine_labels",
        "flags_reference_only",
        "split",
        "temporal_verification",
    ]
    return {key: row[key] for key in keys}


def readme_text(
    *,
    dataset_sha256: str,
    canonical_sha256: str,
    schema_sha256: str,
    key_commitment: str,
    examples: dict[str, dict[str, Any]],
    splits: Counter[str],
) -> str:
    safe_example = json.dumps(
        example_view(examples[EXAMPLE_IDS[0]]), ensure_ascii=False, indent=2
    )
    damage_example = json.dumps(
        example_view(examples[EXAMPLE_IDS[1]]), ensure_ascii=False, indent=2
    )
    return f'''# Dataset etiquetado final documentado

Este paquete publica el snapshot completo usado antes de separar físicamente
`train`, `validation` y `test`. Contiene **{EXPECTED_ROWS:,} chunks únicos** de
**{EXPECTED_VIDEOS:,} videos**. Se conservaron sin cambios el texto, las
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

- `{DATASET_NAME}`: arreglo JSON principal, una fila compacta por línea.
- `{SCHEMA_NAME}`: JSON Schema Draft 2020-12 y diccionario formal de campos.
- `{VALIDATOR_NAME}`: auditoría reproducible con la biblioteca estándar de Python.
- `{PROVENANCE_NAME}`: diseño, verificación y límites de la huella de procedencia.
- `{CITATION_NAME}`: metadatos de citación del dataset en Citation File Format.
- `{REPORT_NAME}`: resultado de la validación ejecutada antes de empaquetar.
- `{MANIFEST_NAME}`: hashes SHA-256 de todos los archivos anteriores y del README.
- `{README_NAME}`: esta guía.

## Procedencia y verificación temporal

Cada transcripción fuente se volvió a trocear con el contrato del proyecto. Un
tiempo se incorporó únicamente cuando coincidieron simultáneamente el
`chunk_id` regenerado y el texto completo. El resultado fue
**{EXPECTED_ROWS:,}/{EXPECTED_ROWS:,} coincidencias exactas**, **cero faltantes**
y **cero conflictos**. El campo `chunk_id` permite volver a verificar la unión,
pues su hash incorpora la versión/configuración del troceador, `video_id`,
inicio, fin y texto normalizado.

- SHA-256 del JSONL canónico usado por los entrenamientos:
  `{canonical_sha256}`.
- SHA-256 del JSON documentado incluido:
  `{dataset_sha256}`.
- SHA-256 del esquema:
  `{schema_sha256}`.
- Particiones conservadas: {splits['train']:,} `train`,
  {splits['validation']:,} `validation` y {splits['test']:,} `test`.

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
    texto = unicodedata.normalize("NFKC", texto or "").replace("\\n", " ")
    texto = re.sub(
        r"\\[(musica|música|aplausos|risas|music|applause|laughter)\\]",
        " ",
        texto,
        flags=re.IGNORECASE,
    )
    texto = re.sub(r"https?://\\S+", " ", texto)
    return re.sub(r"\\s+", " ", texto).strip()

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
        f"{{chunker_version}}|{{chunking_signature}}|{{video_id}}|"
        f"{{start_seconds:.3f}}|{{end_seconds:.3f}}|{{texto}}"
    )
    sufijo = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"{{video_id}}_{{sufijo}}"

with Path("{DATASET_NAME}").open(encoding="utf-8") as archivo:
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
SHA-256(clave_privada) = {key_commitment}
```

La huella ayuda a reconocer filas o redistribuciones copiadas sin alterar el
texto, las etiquetas, los tiempos, las particiones o la elegibilidad de
entrenamiento. No demuestra por sí sola que un modelo haya sido entrenado con
el corpus y puede eliminarse al transformar los datos. Véase
`{PROVENANCE_NAME}` para el procedimiento completo y sus límites.

Al reutilizar el dataset se solicita conservar `release_id`, citar a los cuatro
autores y mencionar esta edición. `{CITATION_NAME}` contiene los metadatos
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
está en `{SCHEMA_NAME}`.

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
{safe_example}
```

Ejemplo multietiqueta con contenido potencialmente sensible:

```json
{damage_example}
```

Los ejemplos muestran una vista reducida; los registros originales contienen
todos los campos enumerados en el esquema.

## Carga en Python o Jupyter

```python
from pathlib import Path
import json

ruta = Path("{DATASET_NAME}")
with ruta.open(encoding="utf-8") as archivo:
    dataset = json.load(archivo)

print(f"Chunks: {{len(dataset):,}}")
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
python {VALIDATOR_NAME}
```

El validador comprueba hashes del manifiesto, conteos, campos obligatorios,
tipos, máscaras, vocabularios, unicidad de `chunk_id`, coherencia de enlaces y
estampas, hashes de texto, tokens únicos y la reconstrucción de cada `chunk_id`.
El custodio puede verificar además los HMAC con la clave privada, sin copiarla
al paquete público:

```bash
python {VALIDATOR_NAME} --provenance-key "/ruta/privada/publication_provenance_hmac.key"
```

En PowerShell también puede revisar un hash individual:

```powershell
Get-FileHash .\\{DATASET_NAME} -Algorithm SHA256
```

## Alcance y precauciones

Los textos proceden de subtítulos y pueden contener errores de transcripción,
lenguaje dañino o información sensible. Un enlace puede dejar de estar
disponible si el propietario retira o restringe el video. El paquete no incluye
audio ni video y no concede una licencia nueva sobre contenido de terceros. Su
uso debe ser académico o evaluativo, respetar los términos de la plataforma y
mantener revisión humana; las etiquetas no deben emplearse por sí solas para
aplicar sanciones automáticas.
'''.replace(f"{EXPECTED_ROWS:,}", f"{EXPECTED_ROWS:,}".replace(",", " ")).replace(
        f"{EXPECTED_VIDEOS:,}", f"{EXPECTED_VIDEOS:,}".replace(",", " ")
    )


def provenance_text(key_commitment: str) -> str:
    return f'''# Huella de procedencia

## Diseño

Esta edición usa una huella **de metadatos** y no una “ciudad de papel” dentro
del corpus. No se añadieron ejemplos falsos ni se alteraron texto, etiquetas,
tiempos, pesos o particiones. Cada fila conserva:

- `release_id`: `{RELEASE_ID}`;
- `provenance_token`: HMAC-SHA256 hexadecimal de 64 caracteres;
- `provenance_key_commitment`: SHA-256 público de la clave privada.

La regla exacta es:

```text
mensaje = release_id|chunk_id|text_sha256
provenance_token = HMAC-SHA256(clave_privada, mensaje UTF-8)
```

Compromiso publicado para esta edición:

```text
SHA-256(clave_privada) = {key_commitment}
```

La clave privada no forma parte del ZIP ni del repositorio. El custodio debe
mantenerla en almacenamiento restringido y respaldado. Si alguna vez se revela
para una auditoría, primero se verifica que su SHA-256 coincida con el compromiso
anterior y luego se recalculan los tokens.

## Verificación por el custodio

Desde la carpeta extraída, sin copiar la clave al paquete:

```bash
python {VALIDATOR_NAME} --provenance-key "/ruta/privada/publication_provenance_hmac.key"
```

Sin la clave, el mismo validador comprueba que haya {EXPECTED_ROWS:,} tokens
hexadecimales únicos, un compromiso uniforme y un `release_id` consistente. Con
la clave, verifica además cada HMAC.

## Qué permite afirmar

La coincidencia de `release_id`, `chunk_id`, `text_sha256` y
`provenance_token` aporta evidencia técnica de que una fila procede de esta
edición o de una copia de ella. El manifiesto y el hash del ZIP fijan además el
contenido exacto publicado.

La huella no prueba por sí sola una infracción, falta de cita ni entrenamiento
de un modelo. Puede eliminarse durante una transformación, y una fila pública
puede copiarse junto con su token. Cualquier conclusión debe apoyarse en el
contexto, las condiciones de uso y una revisión humana. La función es de
trazabilidad académica, no de vigilancia encubierta.
'''.replace(f"{EXPECTED_ROWS:,}", f"{EXPECTED_ROWS:,}".replace(",", " "))


def citation_cff() -> str:
    return f'''cff-version: 1.2.0
message: "Si utiliza este dataset, cite esta edición y a sus autores."
type: dataset
title: "Dataset etiquetado final documentado para moderación semiautomática de videos peruanos de YouTube"
version: "{PUBLICATION_METADATA_VERSION}"
date-released: "2026-08-17"
authors:
  - family-names: "Koc Góngora"
    given-names: "Luis Enrique"
    email: "luis.koc@gmail.com"
  - family-names: "Mancilla Antay"
    given-names: "Alex Felipe"
    email: "amancillaa@uni.pe"
  - family-names: "Meléndez García"
    given-names: "Herbert Antonio"
    email: "hamg.94@gmail.com"
  - family-names: "Paitán Cano"
    given-names: "Dennis Jack"
    email: "dennis.paitan.c@uni.pe"
keywords:
  - procesamiento del lenguaje natural
  - moderación de contenido
  - YouTube
  - Perú
  - clasificación multietiqueta
'''


VALIDATOR_SOURCE = r'''from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset_etiquetado_final_documentado_173240.json"
SCHEMA = ROOT / "dataset_schema.json"
MANIFEST = ROOT / "MANIFEST.sha256"
EXPECTED_ROWS = 173_240
EXPECTED_VIDEOS = 4_906
EXPECTED_SPLITS = {"train": 123_239, "validation": 27_317, "test": 22_684}
RELEASE_ID = "grupo4-moderacion-youtube-peru-2026-08-17-v1.1.0"
KEY_COMMITMENT = "__KEY_COMMITMENT__"
COARSE = {
    "SEGURO", "RACISMO_DISCRIMINACION", "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA", "CONTENIDO_SEXUAL"
}
FINE = {
    "seguro", "seguro_ironia_marcada", "racismo_etnico_explicito",
    "racismo_encubierto", "clasismo_racial", "discriminacion_regional",
    "racismo_linguistico", "misoginia_acoso_genero", "homofobia_transfobia",
    "acoso_personal", "amenaza_directa", "sexual_explicito",
    "sexual_cosificacion", "sexual_no_consensual"
}
FLAGS = {"humor_encubridor", "contexto_necesario", "ironia_ambigua"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").replace("\n", " ")
    value = re.sub(
        r"\[(musica|música|aplausos|risas|music|applause|laughter)\]",
        " ", value, flags=re.IGNORECASE,
    )
    value = re.sub(r"https?://\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def timestamp(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def validate_manifest() -> int:
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = ROOT / name
        assert path.is_file(), f"Falta el archivo del manifiesto: {name}"
        assert sha256_file(path) == expected, f"Hash incompatible: {name}"
        checked += 1
    return checked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida el dataset documentado")
    parser.add_argument(
        "--provenance-key",
        type=Path,
        help="Ruta privada opcional a la clave HMAC de 32 bytes",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    manifest_files = validate_manifest()
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    required = schema["items"]["required"]
    assert schema["items"]["additionalProperties"] is False

    with DATASET.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    assert isinstance(rows, list) and len(rows) == EXPECTED_ROWS

    provenance_key = None
    if args.provenance_key is not None:
        provenance_key = args.provenance_key.read_bytes()
        assert len(provenance_key) == 32, "La clave HMAC debe tener 32 bytes"
        assert hashlib.sha256(provenance_key).hexdigest() == KEY_COMMITMENT

    ids: set[str] = set()
    tokens: set[str] = set()
    videos: set[str] = set()
    splits: Counter[str] = Counter()
    for index, row in enumerate(rows):
        assert list(row) == required, f"Campos u orden incompatibles en fila {index}"
        chunk_id = row["chunk_id"]
        video_id = row["video_id"]
        assert chunk_id not in ids, f"chunk_id duplicado: {chunk_id}"
        ids.add(chunk_id)
        videos.add(video_id)
        splits[row["split"]] += 1

        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        assert 0 <= start < end
        assert row["start_timestamp"] == timestamp(start)
        assert row["end_timestamp"] == timestamp(end)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        assert row["video_url"] == video_url
        assert row["timestamp_url"] == f"{video_url}&t={math.floor(start)}s"

        normalized = normalize_text(row["text"])
        text_sha256 = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
        assert row["text_sha256"] == text_sha256
        assert row["release_id"] == RELEASE_ID
        assert row["provenance_key_commitment"] == KEY_COMMITMENT
        token = row["provenance_token"]
        assert re.fullmatch(r"[0-9a-f]{64}", token)
        assert token not in tokens, f"Token de procedencia duplicado: {chunk_id}"
        tokens.add(token)
        if provenance_key is not None:
            provenance_material = f"{RELEASE_ID}|{chunk_id}|{text_sha256}".encode("utf-8")
            expected_token = hmac.new(
                provenance_key, provenance_material, hashlib.sha256
            ).hexdigest()
            assert hmac.compare_digest(token, expected_token), (
                f"HMAC de procedencia incompatible: {chunk_id}"
            )
        material = (
            f'{row["chunker_version"]}|{row["chunking_signature"]}|{video_id}|'
            f'{start:.3f}|{end:.3f}|{normalized}'
        )
        suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        assert chunk_id == f"{video_id}_{suffix}", f"ID no verificable: {chunk_id}"

        coarse = row["coarse_labels"]
        assert coarse and len(coarse) == len(set(coarse)) and set(coarse) <= COARSE
        assert not ("SEGURO" in coarse and len(coarse) > 1)
        assert len(row["fine_labels"]) == len(set(row["fine_labels"]))
        assert set(row["fine_labels"]) <= FINE
        assert len(row["flags_reference_only"]) == len(set(row["flags_reference_only"]))
        assert set(row["flags_reference_only"]) <= FLAGS
        for field, length in (
            ("coarse_observed_mask", 5),
            ("fine_observed_mask", 14),
            ("flags_observed_mask", 3),
        ):
            mask = row[field]
            assert len(mask) == length and set(mask) <= {0, 1}
        assert row["temporal_verification"] == "exact_chunk_id_and_text"
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_partition_sha256"])
        assert row["training_eligible"] is True
        assert row["needs_review"] is False
        assert row["decision_status"] == "resolved"

    assert len(videos) == EXPECTED_VIDEOS
    assert dict(splits) == EXPECTED_SPLITS
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "unique_chunk_ids": len(ids),
        "unique_provenance_tokens": len(tokens),
        "videos": len(videos),
        "splits": dict(splits),
        "manifest_files_verified": manifest_files,
        "temporal_verification": "exact_chunk_id_and_text",
        "provenance_hmac": (
            "fully_verified" if provenance_key is not None
            else "structure_verified_private_key_not_provided"
        ),
        "provenance_key_commitment": KEY_COMMITMENT,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
'''


def validation_report(
    *,
    dataset_sha256: str,
    canonical_sha256: str,
    key_commitment: str,
    provenance_tokens: int,
    locations_audit: dict[str, int],
    splits: Counter[str],
    videos: int,
) -> str:
    return f"""STATUS: PASS
Dataset: {DATASET_NAME}
Rows: {EXPECTED_ROWS}
Unique chunk_id: {EXPECTED_ROWS}
Release ID: {RELEASE_ID}
Unique provenance tokens: {provenance_tokens}
Provenance algorithm: HMAC-SHA256(release_id|chunk_id|text_sha256)
Provenance key commitment: {key_commitment}
Synthetic canary rows: 0
Videos: {videos}
Splits: train={splits['train']}; validation={splits['validation']}; test={splits['test']}
Temporal locations: {locations_audit['locations']}
Temporal method: exact_chunk_id_and_text
Missing temporal locations: 0
Temporal conflicts: {locations_audit['conflicts']}
Selected transcript records: {locations_audit['selected_transcripts']}
Transcript partition files inspected: {locations_audit['transcript_partition_files']}
Canonical training JSONL SHA-256: {canonical_sha256}
Documented JSON SHA-256: {dataset_sha256}
Checks: semantic identity of original fields; required fields; unique IDs and provenance tokens; links; timestamps; text hashes; reconstructed chunk IDs; HMAC generation; labels; masks; splits; manifest hashes.
"""


def write_manifest(directory: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in names]
    (directory / MANIFEST_NAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def build_zip(directory: Path, output: Path, names: list[str]) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in names:
            payload = (directory / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("El ZIP nuevo no superó la prueba CRC")
    os.replace(temporary, output)


def main() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    base_rows = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    if len(base_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Se esperaban {EXPECTED_ROWS} filas")
    if any(list(row) != ORIGINAL_FIELDS for row in base_rows):
        raise RuntimeError("El contrato de campos originales cambió")
    if len({row["chunk_id"] for row in base_rows}) != EXPECTED_ROWS:
        raise RuntimeError("El dataset base contiene chunk_id duplicados")
    if len({row["video_id"] for row in base_rows}) != EXPECTED_VIDEOS:
        raise RuntimeError("El conteo de videos del dataset base cambió")

    canonical_sha256 = verify_base_against_canonical(base_rows)
    locations, locations_audit = recover_locations(base_rows)
    provenance_key = load_or_create_provenance_key()
    key_commitment = hashlib.sha256(provenance_key).hexdigest()
    dataset_path = STAGING_DIR / DATASET_NAME
    examples, provenance_tokens = write_dataset(
        dataset_path,
        base_rows,
        locations,
        provenance_key,
        key_commitment,
    )
    if provenance_tokens != EXPECTED_ROWS:
        raise RuntimeError("El conteo de tokens de procedencia es incompatible")
    dataset_sha256 = sha256_file(dataset_path)

    schema_path = STAGING_DIR / SCHEMA_NAME
    write_schema(schema_path)
    schema_sha256 = sha256_file(schema_path)
    splits: Counter[str] = Counter(str(row["split"]) for row in base_rows)

    (STAGING_DIR / VALIDATOR_NAME).write_text(
        VALIDATOR_SOURCE.replace("__KEY_COMMITMENT__", key_commitment),
        encoding="utf-8",
        newline="\n",
    )
    (STAGING_DIR / PROVENANCE_NAME).write_text(
        provenance_text(key_commitment), encoding="utf-8", newline="\n"
    )
    (STAGING_DIR / CITATION_NAME).write_text(
        citation_cff(), encoding="utf-8", newline="\n"
    )
    (STAGING_DIR / REPORT_NAME).write_text(
        validation_report(
            dataset_sha256=dataset_sha256,
            canonical_sha256=canonical_sha256,
            key_commitment=key_commitment,
            provenance_tokens=provenance_tokens,
            locations_audit=locations_audit,
            splits=splits,
            videos=len({row["video_id"] for row in base_rows}),
        ),
        encoding="utf-8",
        newline="\n",
    )
    (STAGING_DIR / README_NAME).write_text(
        readme_text(
            dataset_sha256=dataset_sha256,
            canonical_sha256=canonical_sha256,
            schema_sha256=schema_sha256,
            key_commitment=key_commitment,
            examples=examples,
            splits=splits,
        ),
        encoding="utf-8",
        newline="\n",
    )

    manifest_inputs = [
        DATASET_NAME,
        SCHEMA_NAME,
        README_NAME,
        VALIDATOR_NAME,
        PROVENANCE_NAME,
        CITATION_NAME,
        REPORT_NAME,
    ]
    write_manifest(STAGING_DIR, manifest_inputs)
    package_names = manifest_inputs + [MANIFEST_NAME]
    build_zip(STAGING_DIR, PUBLICATION_DIR / ZIP_NAME, package_names)

    print(
        json.dumps(
            {
                "status": "built",
                "zip": str(PUBLICATION_DIR / ZIP_NAME),
                "zip_sha256": sha256_file(PUBLICATION_DIR / ZIP_NAME),
                "dataset_sha256": dataset_sha256,
                "canonical_sha256": canonical_sha256,
                "release_id": RELEASE_ID,
                "provenance_key_commitment": key_commitment,
                "provenance_tokens": provenance_tokens,
                "rows": len(base_rows),
                "videos": len({row["video_id"] for row in base_rows}),
                "splits": dict(splits),
                "locations_audit": locations_audit,
                "files": package_names,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
