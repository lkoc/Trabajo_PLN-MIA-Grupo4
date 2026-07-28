"""Ampliación reproducible del corpus con muestreo dirigido a daños raros.

Este módulo reutiliza el contrato de los cuadernos 01 y 02 sin sobrescribir el
corpus canónico mientras la campaña humana de 139 casos está activa. Descarga
solo subtítulos públicos (nunca audio o video), genera chunks compatibles y
conserva manifiestos, hashes y fallos para la integración posterior.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import unicodedata

import pandas as pd
import yt_dlp


BATCH_ID = os.getenv("AMPLIACION_BATCH_ID", "ampliacion_dano_20260726").strip()
SELECTION_SEED = int(os.getenv("AMPLIACION_SEED", "26072026"))
MAX_SELECTED_VIDEOS = int(os.getenv("AMPLIACION_MAX_VIDEOS", "500"))
MIN_SELECTED_VIDEOS = int(os.getenv("AMPLIACION_MIN_VIDEOS", str(MAX_SELECTED_VIDEOS)))
MAX_CHANNEL_RESULTS = int(os.getenv("AMPLIACION_MAX_CHANNEL_RESULTS", "350"))
MAX_SEARCH_RESULTS = int(os.getenv("AMPLIACION_MAX_SEARCH_RESULTS", "55"))
MIN_TRANSCRIPT_CHARS = 200
SUBTITLE_LANGUAGES = ["es-PE", "es-419", "es"]
TRANSCRIPT_WORKERS = int(os.getenv("AMPLIACION_TRANSCRIPT_WORKERS", "1"))
DISCOVERY_MODE = os.getenv("AMPLIACION_DISCOVERY_MODE", "mixed").strip().lower()
if DISCOVERY_MODE not in {"mixed", "threat_search"}:
    raise ValueError("AMPLIACION_DISCOVERY_MODE debe ser mixed o threat_search")

CHANNEL_TARGETS = [
    {
        "name": "Hablando Huevadas",
        "url": "https://www.youtube.com/@HablandoHuevadasOficial",
        "quota": 70,
        "target": "CONTENIDO_SEXUAL|ACOSO_GENERO_IDENTIDAD|AMENAZA_DIRECTA",
        "reason": "22.60 chunks de daño y 0.86 amenazas por video en el corpus base",
    },
    {
        "name": "Goblinciano",
        "url": "https://www.youtube.com/@Goblinciano",
        "quota": 85,
        "target": "RACISMO_DISCRIMINACION|ACOSO_PERSONAL|AMENAZA_DIRECTA",
        "reason": "9.82 chunks de daño y 0.58 amenazas por video en el corpus base",
    },
    {
        "name": "Juanito y Richard",
        "url": "https://www.youtube.com/@JuanitoyRichard",
        "quota": 85,
        "target": "RACISMO_DISCRIMINACION|AMENAZA_DIRECTA",
        "reason": "2.22 chunks de daño por video; máximo observado de 11 amenazas en un episodio",
    },
    {
        "name": "Arde Troya con Juliana Oxenford",
        "url": "https://www.youtube.com/@ardetroyalr",
        "quota": 55,
        "target": "ACOSO_PERSONAL|AMENAZA_DIRECTA",
        "reason": "2.41 chunks de daño y 0.39 amenazas por video en el corpus base",
    },
    {
        "name": "Todo Good",
        "url": "https://www.youtube.com/@todogoodpe",
        "quota": 40,
        "target": "ACOSO_PERSONAL|AMENAZA_DIRECTA",
        "reason": "2.16 chunks de daño por video en el corpus base",
    },
    {
        "name": "Magaly TV La Firme",
        "url": "https://www.youtube.com/@MagalyTVLaFirmeATV",
        "quota": 35,
        "target": "ACOSO_PERSONAL|CONTENIDO_SEXUAL|AMENAZA_DIRECTA",
        "reason": "1.17 chunks de daño por video y cobertura de conflicto interpersonal",
    },
]

SEARCH_TARGETS = [
    ("amenazas de muerte Perú", "AMENAZA_DIRECTA"),
    ("amenaza juez audiencia virtual Perú", "AMENAZA_DIRECTA"),
    ("extorsionadores amenazan audio Perú", "AMENAZA_DIRECTA"),
    ("sicarios amenazan periodista Perú", "AMENAZA_DIRECTA"),
    ("te voy a matar denuncia Perú", "AMENAZA_DIRECTA"),
    ("agresor amenaza pareja Perú", "AMENAZA_DIRECTA"),
    ("denuncia amenazas farándula Perú", "AMENAZA_DIRECTA"),
    ("amenazas homofóbicas Perú", "AMENAZA_DIRECTA|ACOSO_GENERO_IDENTIDAD"),
    ("insultos racistas discriminación Perú denuncia", "RACISMO_DISCRIMINACION"),
    ("acoso sexual denuncia televisión peruana", "CONTENIDO_SEXUAL|ACOSO_GENERO_IDENTIDAD"),
]

THREAT_SEARCH_TARGETS = [
    ("amenaza de muerte denuncia Perú", "AMENAZA_DIRECTA"),
    ("extorsionador amenaza audio WhatsApp Perú", "AMENAZA_DIRECTA"),
    ("sicario amenaza empresario Perú", "AMENAZA_DIRECTA"),
    ("amenazan periodista alcalde Perú", "AMENAZA_DIRECTA"),
    ("amenaza fiscal juez audiencia Perú", "AMENAZA_DIRECTA"),
    ("te voy a matar audio denuncia Perú", "AMENAZA_DIRECTA"),
    ("amenaza con arma denuncia Perú", "AMENAZA_DIRECTA"),
    ("amenaza a pareja expareja Perú", "AMENAZA_DIRECTA"),
    ("amenazas colegio extorsión Perú", "AMENAZA_DIRECTA"),
    ("amenaza cobrador de cupos Perú", "AMENAZA_DIRECTA"),
    ("amenazan candidato político Perú", "AMENAZA_DIRECTA"),
    ("amenaza en vivo televisión peruana", "AMENAZA_DIRECTA"),
    ("amenaza redes sociales denuncia Perú", "AMENAZA_DIRECTA"),
    ("agresión amenazas mujer Perú", "AMENAZA_DIRECTA"),
    ("amenaza de secuestro denuncia Perú", "AMENAZA_DIRECTA"),
    ("amenazan transportistas extorsión Perú", "AMENAZA_DIRECTA"),
]

ACTIVE_CHANNEL_TARGETS = [] if DISCOVERY_MODE == "threat_search" else CHANNEL_TARGETS
ACTIVE_SEARCH_TARGETS = (
    THREAT_SEARCH_TARGETS if DISCOVERY_MODE == "threat_search" else SEARCH_TARGETS
)


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "datos" / "processed" / "chunks_para_etiquetar.jsonl").exists():
            return candidate
    raise FileNotFoundError("No se encontró la raíz del proyecto.")


ROOT = find_project_root()
BATCH_DIR = ROOT / "datos" / "ampliacion" / BATCH_ID
RAW_DIR = BATCH_DIR / "raw"
SUBS_DIR = RAW_DIR / "subtitulos"
PROCESSED_DIR = BATCH_DIR / "processed"
RESULTS_DIR = ROOT / "resultados"
REPORT_PATH = RESULTS_DIR / (
    "INFORME_AMPLIACION_DIRIGIDA_DANO.md"
    if BATCH_ID == "ampliacion_dano_20260726"
    else f"INFORME_{BATCH_ID.upper()}.md"
)
CANONICAL_PATH = ROOT / "datos" / "processed" / "chunks_para_etiquetar.jsonl"
BASE_TRANSCRIPTS_PATH = ROOT / "datos" / "raw" / "transcripts_raw.jsonl"
SELECTED_PATH = RAW_DIR / "videos_seleccionados.csv"
CANDIDATES_PATH = RAW_DIR / "videos_candidatos.csv"
TRANSCRIPTS_PATH = RAW_DIR / "transcripts_raw.jsonl"
FAILURES_PATH = RAW_DIR / "fallos_subtitulos.csv"
CHUNKS_PATH = PROCESSED_DIR / "chunks_para_etiquetar.jsonl"
MANIFEST_PATH = BATCH_DIR / "manifiesto_adquisicion.json"
for directory in (RAW_DIR, SUBS_DIR, PROCESSED_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    output = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en {path}, línea {line_number}: {exc}") from exc
    return output


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def existing_video_ids() -> tuple[set[str], dict[str, int]]:
    """Video IDs ya procesados o reservados en cualquier campaña anterior."""
    sources: dict[str, set[str]] = {}
    sources[str(BASE_TRANSCRIPTS_PATH.relative_to(ROOT))] = {
        str(row.get("video_id") or "")
        for row in read_jsonl(BASE_TRANSCRIPTS_PATH)
        if row.get("video_id")
    }
    canonical_chunks = read_jsonl(CANONICAL_PATH)
    sources[str(CANONICAL_PATH.relative_to(ROOT))] = {
        str(row.get("video_id") or "") for row in canonical_chunks if row.get("video_id")
    }
    expansion_root = ROOT / "datos" / "ampliacion"
    for batch_dir in sorted(expansion_root.glob("*")):
        if not batch_dir.is_dir() or batch_dir.resolve() == BATCH_DIR.resolve():
            continue
        batch_ids: set[str] = set()
        selected_path = batch_dir / "raw" / "videos_seleccionados.csv"
        if selected_path.exists() and selected_path.stat().st_size:
            selected = pd.read_csv(selected_path, dtype={"video_id": str})
            if "video_id" in selected:
                batch_ids.update(selected["video_id"].dropna().astype(str))
        for path in (
            batch_dir / "raw" / "transcripts_raw.jsonl",
            batch_dir / "processed" / "chunks_para_etiquetar.jsonl",
        ):
            batch_ids.update(
                str(row.get("video_id") or "")
                for row in read_jsonl(path)
                if row.get("video_id")
            )
        sources[str(batch_dir.relative_to(ROOT))] = batch_ids
    union = set().union(*sources.values()) if sources else set()
    return union, {name: len(values) for name, values in sources.items()}


def existing_chunk_hashes_and_ids() -> tuple[set[str], set[str], dict[str, int]]:
    """Deduplicación global contra el canónico y todas las ampliaciones previas."""
    paths = [CANONICAL_PATH]
    paths.extend(
        path
        for path in sorted((ROOT / "datos" / "ampliacion").glob(
            "*/processed/chunks_para_etiquetar.jsonl"
        ))
        if path.resolve() != CHUNKS_PATH.resolve()
    )
    hashes: set[str] = set()
    chunk_ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        rows = read_jsonl(path)
        counts[str(path.relative_to(ROOT))] = len(rows)
        hashes.update(str(row["text_hash"]) for row in rows if row.get("text_hash"))
        chunk_ids.update(str(row["chunk_id"]) for row in rows if row.get("chunk_id"))
    return hashes, chunk_ids, counts


def _flat_entries(url: str, maximum: int) -> list[dict]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "playlistend": maximum,
        "extractor_retries": 3,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    return [row for row in (info or {}).get("entries", []) or [] if row and row.get("id")]


def discover_candidates() -> pd.DataFrame:
    """Lista canales de alto rendimiento y búsquedas temáticas actuales."""
    rows: list[dict] = []
    for source in ACTIVE_CHANNEL_TARGETS:
        print(f"Listando canal: {source['name']}", flush=True)
        url = source["url"].rstrip("/") + "/videos"
        for rank, item in enumerate(_flat_entries(url, MAX_CHANNEL_RESULTS), 1):
            rows.append(
                {
                    "video_id": item["id"],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "title": item.get("title") or "",
                    "channel_title": item.get("channel") or source["name"],
                    "channel_id": item.get("channel_id"),
                    "duration": item.get("duration"),
                    "discovery_type": "high_yield_channel",
                    "discovery_source": source["name"],
                    "source_rank": rank,
                    "target_category": source["target"],
                    "selection_reason": source["reason"],
                    "quota": source["quota"],
                }
            )
    for query, target in ACTIVE_SEARCH_TARGETS:
        print(f"Buscando: {query}", flush=True)
        for rank, item in enumerate(
            _flat_entries(f"ytsearch{MAX_SEARCH_RESULTS}:{query}", MAX_SEARCH_RESULTS), 1
        ):
            rows.append(
                {
                    "video_id": item["id"],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "title": item.get("title") or "",
                    "channel_title": item.get("channel") or item.get("uploader") or "",
                    "channel_id": item.get("channel_id"),
                    "duration": item.get("duration"),
                    "discovery_type": "targeted_search",
                    "discovery_source": query,
                    "source_rank": rank,
                    "target_category": target,
                    "selection_reason": "consulta explícita de una categoría minoritaria",
                    "quota": MAX_SEARCH_RESULTS,
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("La búsqueda no devolvió videos.")
    candidates = candidates.sort_values(
        ["discovery_type", "discovery_source", "source_rank"], kind="stable"
    ).drop_duplicates("video_id", keep="first")
    candidates.to_csv(CANDIDATES_PATH, index=False)
    return candidates.reset_index(drop=True)


def select_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    prior_video_ids, prior_sources = existing_video_ids()
    available = candidates.loc[~candidates["video_id"].astype(str).isin(prior_video_ids)].copy()
    selected_parts = []
    for source in ACTIVE_CHANNEL_TARGETS:
        group = available.loc[
            (available["discovery_type"] == "high_yield_channel")
            & (available["discovery_source"] == source["name"])
        ].head(source["quota"])
        selected_parts.append(group)
    searches = available.loc[available["discovery_type"] == "targeted_search"].copy()
    # Round-robin por consulta para que una búsqueda no monopolice el lote.
    searches["round_robin"] = searches["source_rank"].astype(int)
    searches = searches.sort_values(["round_robin", "discovery_source"], kind="stable")
    selected = pd.concat([*selected_parts, searches], ignore_index=True)
    selected = selected.drop_duplicates("video_id", keep="first")
    selected = selected.head(MAX_SELECTED_VIDEOS).reset_index(drop=True)
    if len(selected) < MIN_SELECTED_VIDEOS:
        raise RuntimeError(
            f"Solo hay {len(selected)} videos inéditos; se exigieron {MIN_SELECTED_VIDEOS}. "
            "Amplíe canales/consultas antes de continuar."
        )
    overlap = set(selected["video_id"].astype(str)) & prior_video_ids
    if overlap:
        raise AssertionError(f"La selección contiene {len(overlap)} videos ya procesados.")
    selected.insert(0, "selection_order", range(1, len(selected) + 1))
    selected["batch_id"] = BATCH_ID
    selected["selected_at"] = now_iso()
    selected.to_csv(SELECTED_PATH, index=False)
    manifest = {
        "batch_id": BATCH_ID,
        "created_at": now_iso(),
        "selection_seed": SELECTION_SEED,
        "selection_method": (
            "round-robin threat-specific YouTube search"
            if DISCOVERY_MODE == "threat_search"
            else "fixed channel quotas plus round-robin targeted YouTube search"
        ),
        "discovery_mode": DISCOVERY_MODE,
        "sampling_scope": "enriched training/validation sample; not prevalence representative",
        "max_selected_videos": MAX_SELECTED_VIDEOS,
        "candidate_rows": len(candidates),
        "candidate_unique_videos": int(candidates["video_id"].nunique()),
        "excluded_existing_videos": int(candidates["video_id"].astype(str).isin(prior_video_ids).sum()),
        "prior_unique_video_ids": len(prior_video_ids),
        "prior_video_id_sources": prior_sources,
        "selected_overlap_with_prior": 0,
        "selected_videos": len(selected),
        "selected_by_type": selected["discovery_type"].value_counts().to_dict(),
        "selected_by_target": selected["target_category"].value_counts().to_dict(),
        "canonical_before_sha256": sha256_file(CANONICAL_PATH),
        "base_transcripts_sha256": sha256_file(BASE_TRANSCRIPTS_PATH),
        "selected_sha256": sha256_file(SELECTED_PATH),
        "channel_targets": ACTIVE_CHANNEL_TARGETS,
        "search_targets": ACTIVE_SEARCH_TARGETS,
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    return selected


def _subtitle_paths(video_id: str) -> list[Path]:
    priority = {language: index for index, language in enumerate(SUBTITLE_LANGUAGES)}
    paths = list(SUBS_DIR.glob(f"{video_id}*.vtt"))
    return sorted(
        paths,
        key=lambda path: (
            priority.get(path.name.split(".")[-2] if "." in path.name else "", 99),
            path.name,
        ),
    )


def _time_seconds(value: str) -> float:
    parts = [float(part) for part in value.replace(",", ".").split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, minutes, seconds = 0, parts[0], parts[1]
    return hours * 3600 + minutes * 60 + seconds


def _read_vtt(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content)
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s+-->\s+"
        r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )
    segments, seen = [], set()
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        match, index = None, 0
        for position, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                index = position
                break
        if not match:
            continue
        start, end = _time_seconds(match.group(1)), _time_seconds(match.group(2))
        phrase = " ".join(lines[index + 1 :])
        phrase = html.unescape(re.sub(r"<[^>]+>", "", phrase))
        phrase = re.sub(r"\s+", " ", phrase).strip()
        key = (round(start, 1), phrase.casefold())
        if phrase and key not in seen:
            seen.add(key)
            segments.append({"start": start, "duration": max(end - start, 0.1), "text": phrase})
    return segments


def _download_transcript(row: dict) -> tuple[dict | None, dict | None]:
    video_id = str(row["video_id"])
    try:
        paths = _subtitle_paths(video_id)
        if not paths:
            options = {
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "ignoreerrors": False,
                "skip_download": True,
                "noplaylist": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": SUBTITLE_LANGUAGES,
                "subtitlesformat": "vtt",
                "outtmpl": str(SUBS_DIR / f"{video_id}.%(ext)s"),
                "extractor_retries": 3,
                "retries": 3,
                "sleep_interval_requests": 1,
            }
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([row["url"]])
            paths = _subtitle_paths(video_id)
        candidates = [(_read_vtt(path), path) for path in paths]
        candidates.sort(key=lambda pair: len(" ".join(x["text"] for x in pair[0])), reverse=True)
        segments, path = candidates[0] if candidates else ([], None)
        text = " ".join(segment["text"] for segment in segments).strip()
        if len(text) < MIN_TRANSCRIPT_CHARS:
            return None, {"video_id": video_id, "reason": "sin subtítulo español utilizable"}
        transcript = {
            "video_id": video_id,
            "url": row.get("url"),
            "title": row.get("title"),
            "channel_id": row.get("channel_id"),
            "channel_title": row.get("channel_title"),
            "channel_url": None,
            "published_at": None,
            "categoria_fuente": "muestreo_dirigido_dano",
            "fuente_subs": f"yt-dlp-vtt:{path.name if path else ''}",
            "text_hash": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "batch_id": BATCH_ID,
            "discovery_type": row.get("discovery_type"),
            "discovery_source": row.get("discovery_source"),
            "target_category": row.get("target_category"),
            "segments": segments,
        }
        return transcript, None
    except Exception as exc:
        return None, {"video_id": video_id, "reason": f"{type(exc).__name__}: {exc}"[:500]}


def acquire_transcripts(selected: pd.DataFrame) -> list[dict]:
    selected_ids = set(selected["video_id"].astype(str))
    all_existing = {str(row["video_id"]): row for row in read_jsonl(TRANSCRIPTS_PATH)}
    existing = {
        video_id: row for video_id, row in all_existing.items() if video_id in selected_ids
    }
    orphaned = len(all_existing) - len(existing)
    if orphaned:
        # Una repetición de la búsqueda puede cambiar resultados de YouTube.
        # La cohorte oficial debe coincidir exactamente con la selección vigente.
        write_jsonl_atomic(TRANSCRIPTS_PATH, list(existing.values()))
        print(f"Transcripciones fuera de la selección vigente descartadas={orphaned}", flush=True)
    pending = [row for row in selected.to_dict("records") if row["video_id"] not in existing]
    failures: list[dict] = []
    print(f"Subtítulos previos={len(existing)}; pendientes={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=TRANSCRIPT_WORKERS) as executor:
        futures = {executor.submit(_download_transcript, row): row for row in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            transcript, failure = future.result()
            if transcript:
                existing[transcript["video_id"]] = transcript
                write_jsonl_atomic(TRANSCRIPTS_PATH, list(existing.values()))
            if failure:
                failures.append({**futures[future], **failure})
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"Procesados={completed}/{len(pending)}; con_subtítulos={len(existing)}; "
                    f"fallos={len(failures)}",
                    flush=True,
                )
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
    return list(existing.values())


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\n", " ")
    text = re.sub(r"\[(musica|aplausos|risas|music|applause|laughter)\]", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_vtt_overlap(previous: str, following: str, max_words: int = 12) -> str:
    if not previous or not following:
        return following
    previous_words = previous.casefold().split()
    following_lower = following.casefold().split()
    following_original = following.split()
    for overlap in range(min(max_words, len(previous_words), len(following_lower)), 0, -1):
        if previous_words[-overlap:] == following_lower[:overlap]:
            return " ".join(following_original[overlap:])
    return following


def _text_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def _make_chunk(record: dict, start: float, end: float, text: str, index: int) -> dict:
    video_id = record["video_id"]
    return {
        "chunk_id": f"{video_id}_{index:04d}",
        "video_id": video_id,
        "channel_id": record.get("channel_id"),
        "channel_title": record.get("channel_title"),
        "video_title": record.get("title"),
        "published_at": record.get("published_at"),
        "start_seconds": round(float(start or 0.0), 2),
        "end_seconds": round(float(end or 0.0), 2),
        "text": text,
        "text_hash": _text_hash(text),
        "labels": [],
        "flags": [],
        "needs_review": True,
        "annotator": "",
        "notes": "",
        "batch_id": BATCH_ID,
        "discovery_type": record.get("discovery_type"),
        "discovery_source": record.get("discovery_source"),
        "target_category": record.get("target_category"),
    }


def build_chunks(record: dict, target_seconds: int = 30, max_chars: int = 600, min_chars: int = 90) -> list[dict]:
    chunks, current = [], []
    start = end = None
    character_count = 0
    for segment in record.get("segments", []):
        segment_text = normalize_text(segment.get("text", ""))
        if not segment_text:
            continue
        if current:
            segment_text = remove_vtt_overlap(current[-1], segment_text)
            if not segment_text:
                continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = segment_start + float(segment.get("duration", 0.0))
        if start is None:
            start = segment_start
        end = segment_end
        current.append(segment_text)
        character_count += len(segment_text)
        if end - start >= target_seconds or character_count >= max_chars:
            text = normalize_text(" ".join(current))
            if len(text) >= min_chars:
                chunks.append(_make_chunk(record, start, end, text, len(chunks)))
            current, start, end, character_count = [], None, None, 0
    text = normalize_text(" ".join(current))
    if len(text) >= min_chars:
        chunks.append(_make_chunk(record, start or 0.0, end or 0.0, text, len(chunks)))
    return chunks


def create_chunks(transcripts: list[dict]) -> list[dict]:
    base_hashes, base_ids, prior_chunk_sources = existing_chunk_hashes_and_ids()
    prior_video_ids, _ = existing_video_ids()
    transcript_video_ids = {str(row["video_id"]) for row in transcripts}
    overlap_videos = transcript_video_ids & prior_video_ids
    if overlap_videos:
        raise AssertionError(
            f"Hay {len(overlap_videos)} videos de ampliaciones previas en el lote nuevo."
        )
    output, seen_hashes = [], set()
    duplicate_base = duplicate_batch = 0
    for transcript in transcripts:
        for chunk in build_chunks(transcript):
            if chunk["chunk_id"] in base_ids:
                raise ValueError(f"ID nuevo colisiona con el canónico: {chunk['chunk_id']}")
            if chunk["text_hash"] in base_hashes:
                duplicate_base += 1
                continue
            if chunk["text_hash"] in seen_hashes:
                duplicate_batch += 1
                continue
            seen_hashes.add(chunk["text_hash"])
            output.append(chunk)
    write_jsonl_atomic(CHUNKS_PATH, output)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.update(
        {
            "updated_at": now_iso(),
            "transcript_videos": len(transcripts),
            "subtitle_success_rate": len(transcripts) / max(manifest["selected_videos"], 1),
            "new_chunks": len(output),
            "new_videos_with_chunks": len({row["video_id"] for row in output}),
            "duplicates_against_base": duplicate_base,
            "duplicates_within_batch": duplicate_batch,
            "prior_chunk_sources": prior_chunk_sources,
            "video_overlap_with_all_prior_campaigns": 0,
            "transcripts_sha256": sha256_file(TRANSCRIPTS_PATH),
            "chunks_sha256": sha256_file(CHUNKS_PATH),
            "audio_or_video_downloaded": False,
        }
    )
    write_json_atomic(MANIFEST_PATH, manifest)
    return output


def update_report() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    selected = pd.read_csv(SELECTED_PATH) if SELECTED_PATH.exists() else pd.DataFrame()
    failures = pd.read_csv(FAILURES_PATH) if FAILURES_PATH.exists() and FAILURES_PATH.stat().st_size else pd.DataFrame()
    lines = [
        "# Informe de ampliación dirigida de categorías de daño",
        "",
        f"Fecha de actualización: {now_iso()}  ",
        f"Lote: `{BATCH_ID}`",
        "",
        "## Objetivo y alcance",
        "",
        "Aumentar los positivos de las cinco categorías gruesas, con prioridad en `AMENAZA_DIRECTA` y `CONTENIDO_SEXUAL`, sin entrenar las etiquetas finas. El lote es una muestra enriquecida para entrenamiento/validación y no debe usarse para estimar prevalencias poblacionales. El test histórico permanece congelado por video.",
        "",
        "## Balance antes de la ampliación",
        "",
        "| Categoría gruesa | Positivos provisionales | Déficit hasta 1.000 |",
        "|---|---:|---:|",
        "| RACISMO_DISCRIMINACION | 1.090 | 0 |",
        "| ACOSO_GENERO_IDENTIDAD | 1.017 | 0 |",
        "| ACOSO_PERSONAL | 1.257 | 0 |",
        "| AMENAZA_DIRECTA | 222 | 778 |",
        "| CONTENIDO_SEXUAL | 961 | 39 |",
        "",
        "Los conteos son provisionales porque los 139 casos humanos aún se están adjudicando. Aun así, identifican correctamente el cuello de botella relativo.",
        "",
        "## Estimación de videos necesarios",
        "",
        "La tasa global observada de amenaza fue 222/1.856 = 0,120 chunks por video; cubrir 778 con muestreo aleatorio requeriría aproximadamente 6.500 videos. En los cuatro canales más productivos se observaron 156 amenazas/295 videos = 0,529 por video, equivalente a unos 1.470 videos adicionales si la tasa se mantuviera. Estas son extrapolaciones de planificación, no garantías: el lote usa búsquedas explícitas para intentar elevar el rendimiento y recalcular la estimación con evidencia nueva.",
        "",
        "## Diseño de adquisición",
        "",
        "Se combinaron cuotas de seis canales con alto rendimiento histórico y diez consultas temáticas. Los videos ya presentes se excluyeron por `video_id`; la deduplicación posterior usa `text_hash`. Solo se obtienen subtítulos públicos en español mediante `yt-dlp` con `skip_download=True`; no se descarga audio ni video.",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
        f"| Candidatos únicos | {manifest.get('candidate_unique_videos', 'pendiente')} |",
        f"| Videos seleccionados | {manifest.get('selected_videos', 'pendiente')} |",
        f"| Videos con subtítulos utilizables | {manifest.get('transcript_videos', 'pendiente')} |",
        f"| Tasa de subtítulos | {manifest.get('subtitle_success_rate', 0):.1%} |" if manifest.get("subtitle_success_rate") is not None else "| Tasa de subtítulos | pendiente |",
        f"| Chunks nuevos deduplicados | {manifest.get('new_chunks', 'pendiente')} |",
        f"| Fallos de subtítulos | {len(failures) if not failures.empty else 0} |",
        "",
        "## Trazabilidad",
        "",
        f"- Selección: `{SELECTED_PATH.relative_to(ROOT)}`",
        f"- Transcripciones: `{TRANSCRIPTS_PATH.relative_to(ROOT)}`",
        f"- Chunks: `{CHUNKS_PATH.relative_to(ROOT)}`",
        f"- Manifiesto: `{MANIFEST_PATH.relative_to(ROOT)}`",
        f"- SHA-256 canónico previo: `{manifest.get('canonical_before_sha256', 'pendiente')}`",
        f"- SHA-256 selección: `{manifest.get('selected_sha256', 'pendiente')}`",
        f"- SHA-256 chunks nuevos: `{manifest.get('chunks_sha256', 'pendiente')}`",
        "",
        "## Estado de las siguientes etapas",
        "",
        "El etiquetado Flash→Pro, la integración y el reentrenamiento se completarán en las secciones posteriores de este mismo informe. La integración al corpus canónico permanece diferida mientras la campaña humana activa pueda depender de su hash y partición congelados.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(stage: str) -> None:
    candidates = None
    selected = None
    transcripts = None
    if stage in {"discover", "all"}:
        candidates = discover_candidates()
        selected = select_candidates(candidates)
        print(f"Seleccionados: {len(selected)}", flush=True)
    if stage in {"transcribe", "all"}:
        if selected is None:
            if not SELECTED_PATH.exists():
                raise FileNotFoundError("Ejecute primero --stage discover.")
            selected = pd.read_csv(SELECTED_PATH)
        transcripts = acquire_transcripts(selected)
        print(f"Transcripciones utilizables: {len(transcripts)}", flush=True)
    if stage in {"chunk", "all"}:
        transcripts = transcripts if transcripts is not None else read_jsonl(TRANSCRIPTS_PATH)
        chunks = create_chunks(transcripts)
        print(f"Chunks nuevos deduplicados: {len(chunks)}", flush=True)
    update_report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["discover", "transcribe", "chunk", "all"], default="all")
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
