from __future__ import annotations

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
KEY_COMMITMENT = "76f0e0d5f59bda914abf64575284ff89341c4442a40c5638b95dc36cdeebef07"
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
