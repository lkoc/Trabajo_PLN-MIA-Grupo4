"""Servidor local del modo de operación del moderador de cuatro daños.

La página servida es un único HTML con CSS/JS embebidos. La inferencia y la
persistencia permanecen en este backend local porque los checkpoints no son
ejecutables de forma segura dentro de un HTML estático.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from html import unescape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, RLock, Thread
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import uuid
import webbrowser

import joblib
import numpy as np

from scripts_auxiliares import entrenar_qwen_acoso_amenaza as q4
ROOT = q4.ROOT
HTML_PATH = ROOT / "Cuadernos" / "frontend" / "produccion_moderador.html"
REGISTRY_PATH = (
    ROOT
    / "resultados"
    / "metricas"
    / "comparacion_final_4"
    / "registro_modelos_desplegables.json"
)
_configured_operation_dir = os.getenv("PLN_OPERATION_DIR", "").strip()
OPERATION_DIR = (
    Path(_configured_operation_dir).expanduser().resolve()
    if _configured_operation_dir
    else ROOT / "resultados" / "operacion_05"
)
DATABASE_PATH = OPERATION_DIR / "estadisticas_moderacion.sqlite3"
RETRAINING_JSONL_PATH = OPERATION_DIR / "revisiones_para_reentrenamiento.jsonl"
RETRAINING_READY_PATH = OPERATION_DIR / "revisiones_adjudicadas_unicas.jsonl"
TARGET_LABELS = list(q4.TARGET_LABELS)
SAFE_LABEL = "SEGURO"
VALID_MODES = {"classical", "transformer", "qwen", "compare", "consensus"}
VALID_INPUT_TYPES = {"auto", "text", "youtube"}
DEFAULT_SUBTITLE_LANGUAGES = ("es", "es-419", "es-US", "en")
MAX_REQUEST_BYTES = 2_000_000
MAX_TEXT_CHARACTERS = 200_000
MAX_CHUNKS = 300
CONSENSUS_MIN_VOTES = 2
RETRAIN_MINIMUM_TOTAL = 500
RETRAIN_MINIMUM_PER_DAMAGE = 100
RETRAIN_MINIMUM_SAFE = 200


class OperationError(RuntimeError):
    status = HTTPStatus.BAD_REQUEST


class SubtitleUnavailableError(OperationError):
    status = HTTPStatus.UNPROCESSABLE_ENTITY


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_deployment_registry(*, verify_hashes: bool = True) -> dict:
    """Carga el registro ya publicado sin importar módulos de entrenamiento."""
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Falta el registro desplegable: {REGISTRY_PATH}")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if verify_hashes:
        for model in registry["models"].values():
            for artifact in model["artifacts"].values():
                values = artifact if isinstance(artifact, list) else [artifact]
                for item in values:
                    if item is None:
                        continue
                    path = ROOT / item["path"]
                    if not path.is_file() or _sha256_file(path) != item["sha256"]:
                        raise RuntimeError(
                            "Artefacto desplegable ausente o modificado: "
                            f"{item['path']}"
                        )
    return registry


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _path_label(path: Path) -> str:
    try:
        return q4.tm.project_relative(path)
    except ValueError:
        return str(Path(path).resolve())


def _normalize_text(text: str) -> str:
    value = unescape(str(text or "")).replace("\n", " ")
    value = re.sub(
        r"\[(musica|música|aplausos|risas|music|applause|laughter)\]",
        " ",
        value,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", value).strip()


def _remove_overlap(previous: str, following: str, max_words: int = 12) -> str:
    previous_words = previous.casefold().split()
    following_lower = following.casefold().split()
    following_original = following.split()
    for overlap in range(
        min(max_words, len(previous_words), len(following_lower)), 0, -1
    ):
        if previous_words[-overlap:] == following_lower[:overlap]:
            return " ".join(following_original[overlap:])
    return following


def _chunk_segments(
    segments: list[dict],
    video_id: str,
    *,
    target_seconds: float = 30,
    max_characters: int = 600,
    minimum_characters: int = 90,
) -> list[dict]:
    chunks, current = [], []
    start = end = None
    character_count = 0
    for segment in segments:
        text = _normalize_text(segment.get("text", ""))
        if not text:
            continue
        if current:
            text = _remove_overlap(current[-1], text)
            if not text:
                continue
        segment_start = float(segment.get("start", 0.0))
        segment_end = segment_start + max(0.0, float(segment.get("duration", 0.0)))
        if start is None:
            start = segment_start
        end = segment_end
        current.append(text)
        character_count += len(text)
        if (end - start) >= target_seconds or character_count >= max_characters:
            joined = _normalize_text(" ".join(current))
            if len(joined) >= minimum_characters:
                chunks.append(
                    {
                        "chunk_id": f"{video_id}_{len(chunks):04d}",
                        "text": joined,
                        "start_seconds": round(float(start), 2),
                        "end_seconds": round(float(end), 2),
                    }
                )
            current, start, end, character_count = [], None, None, 0
    joined = _normalize_text(" ".join(current))
    if joined and (len(joined) >= minimum_characters or not chunks):
        chunks.append(
            {
                "chunk_id": f"{video_id}_{len(chunks):04d}",
                "text": joined,
                "start_seconds": round(float(start or 0.0), 2),
                "end_seconds": round(float(end or start or 0.0), 2),
            }
        )
    return chunks


def _chunk_text(text: str, max_characters: int = 600) -> list[dict]:
    text = _normalize_text(text)
    if not text:
        raise OperationError("Escriba una frase o pegue un enlace de YouTube.")
    if len(text) > MAX_TEXT_CHARACTERS:
        raise OperationError(
            f"El texto excede el máximo local de {MAX_TEXT_CHARACTERS:,} caracteres."
        )
    pieces = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + len(piece) + 1 > max_characters:
            chunks.append(current)
            current = ""
        if len(piece) > max_characters:
            words = piece.split()
            for word in words:
                if current and len(current) + len(word) + 1 > max_characters:
                    chunks.append(current)
                    current = ""
                current = f"{current} {word}".strip()
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return [
        {
            "chunk_id": f"texto_{index:04d}",
            "text": value,
            "start_seconds": None,
            "end_seconds": None,
        }
        for index, value in enumerate(chunks)
    ]


def _youtube_video_id(value: str) -> str | None:
    try:
        parsed = urlparse(value.strip())
    except Exception:
        return None
    host = (parsed.hostname or "").casefold()
    allowed = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
        "youtu.be",
    }
    if host not in allowed:
        return None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.path == "/watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    else:
        parts = parsed.path.strip("/").split("/")
        candidate = parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed"} else ""
    return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate or "") else None


def _looks_like_youtube_url(value: str) -> bool:
    try:
        host = (urlparse(value.strip()).hostname or "").casefold()
    except Exception:
        return False
    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
        "youtu.be",
    }


def _timestamp_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        hours, (minutes, seconds) = "0", parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_vtt(text: str) -> list[dict]:
    segments = []
    blocks = re.split(r"\r?\n\s*\r?\n", text)
    timing = re.compile(
        r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d+)\s+-->\s+"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d+)"
    )
    for block in blocks:
        match = timing.search(block)
        if not match:
            continue
        start, end = _timestamp_seconds(match["start"]), _timestamp_seconds(match["end"])
        lines = block[match.end() :].strip().splitlines()
        content = _normalize_text(
            " ".join(re.sub(r"<[^>]+>", " ", line) for line in lines)
        )
        if content:
            segments.append({"start": start, "duration": max(0.0, end - start), "text": content})
    return segments


def _parse_json3(payload: dict) -> list[dict]:
    segments = []
    for event in payload.get("events", []):
        text = _normalize_text(
            "".join(item.get("utf8", "") for item in event.get("segs", []))
        )
        if text:
            segments.append(
                {
                    "start": float(event.get("tStartMs", 0)) / 1000,
                    "duration": float(event.get("dDurationMs", 0)) / 1000,
                    "text": text,
                }
            )
    return segments


def _select_subtitle_track(
    info: dict, preferred_languages: tuple[str, ...]
) -> tuple[str, dict, str]:
    pools = [
        ("manual", info.get("subtitles") or {}),
        ("automatic", info.get("automatic_captions") or {}),
    ]
    for kind, tracks in pools:
        languages = list(tracks)
        ordered = []
        for preference in preferred_languages:
            ordered.extend(
                language
                for language in languages
                if language == preference or language.startswith(preference + "-")
            )
        ordered.extend(language for language in languages if language not in ordered)
        for language in ordered:
            formats = tracks.get(language) or []
            for extension in ("json3", "vtt"):
                selected = next(
                    (item for item in formats if item.get("ext") == extension and item.get("url")),
                    None,
                )
                if selected:
                    return language, selected, kind
    raise SubtitleUnavailableError(
        "El video no ofrece subtítulos manuales ni automáticos descargables."
    )


def _download_youtube(
    value: str,
    preferred_languages: tuple[str, ...] = DEFAULT_SUBTITLE_LANGUAGES,
) -> dict:
    video_id = _youtube_video_id(value)
    if not video_id:
        raise OperationError("El enlace no es un video de YouTube válido.")
    try:
        import yt_dlp
    except ImportError as error:
        raise OperationError(
            "Falta yt-dlp. Ejecute la celda de dependencias del cuaderno 05."
        ) from error
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 20,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(canonical_url, download=False)
    except Exception as error:
        raise OperationError(f"No se pudo consultar el video: {error}") from error
    language, track, subtitle_kind = _select_subtitle_track(info, preferred_languages)
    request = Request(track["url"], headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
    except Exception as error:
        raise OperationError(f"No se pudieron descargar los subtítulos: {error}") from error
    if track.get("ext") == "json3":
        segments = _parse_json3(json.loads(raw.decode("utf-8")))
    else:
        segments = _parse_vtt(raw.decode("utf-8", errors="replace"))
    if not segments:
        raise SubtitleUnavailableError(
            "El video declara subtítulos, pero no produjo segmentos utilizables."
        )
    chunks = _chunk_segments(segments, video_id)
    return {
        "input_type": "youtube",
        "source_ref": canonical_url,
        "video_id": video_id,
        "video_title": info.get("title") or video_id,
        "channel": info.get("channel") or info.get("uploader"),
        "subtitle_language": language,
        "subtitle_kind": subtitle_kind,
        "chunks": chunks,
    }


class _ClassicalRunner:
    def __init__(self, definition: dict):
        self.definition = definition
        artifacts = definition["artifacts"]
        self.vectorizer = joblib.load(ROOT / artifacts["vectorizer"]["path"])
        self.bundle = joblib.load(ROOT / artifacts["model_bundle"]["path"])
        self.design = definition["design"]

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        full = np.column_stack(
            [head.predict_score(features) for head in self.bundle["full_heads"]]
        )
        if self.design == "flat":
            return full
        gate = self.bundle["gate"].predict_score(features)[:, None]
        if self.design == "shared_hierarchy":
            return gate * full
        conditional = np.column_stack(
            [head.predict_score(features) for head in self.bundle["cascade_heads"]]
        )
        return gate * conditional


class _TransformerRunner:
    def __init__(self, definition: dict):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = ROOT / definition["artifacts"]["checkpoint"]["path"]
        model_directory = checkpoint_path.parent
        config = AutoConfig.from_pretrained(model_directory, local_files_only=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = AutoModel.from_config(config)
                hidden = int(config.hidden_size)
                dropout = float(getattr(config, "hidden_dropout_prob", 0.1))
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden, len(TARGET_LABELS))

            def forward(self, tokens):
                hidden = self.backbone(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                return self.classifier(self.dropout(pooled))

        self.model = Model().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_directory / "tokenizer", local_files_only=True
        )
        self.prefix = definition["model_spec"].get("prefix", "")

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        values = []
        with self.torch.inference_mode():
            for offset in range(0, len(texts), 16):
                batch = [self.prefix + text for text in texts[offset : offset + 16]]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=q4.MAX_LENGTH,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                values.append(self.torch.sigmoid(self.model(tokens)).cpu().numpy())
        return np.vstack(values)


class _QwenRunner:
    def __init__(self, definition: dict):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = q4.load_adapter(ROOT / definition["selected_adapter"], self.device)
        self.model.eval()
        tokenizer_files = definition["artifacts"]["tokenizer"]
        tokenizer_directory = (ROOT / tokenizer_files[0]["path"]).parent
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_directory, local_files_only=True
        )
        self.calibrators = joblib.load(ROOT / definition["artifacts"]["calibrator"]["path"])

    def predict_scores(self, texts: list[str]) -> np.ndarray:
        logits = []
        with self.torch.inference_mode():
            for offset in range(0, len(texts), 8):
                tokens = self.tokenizer(
                    texts[offset : offset + 8],
                    padding=True,
                    truncation=True,
                    max_length=q4.MAX_LENGTH,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self.model(**tokens).logits[:, : len(TARGET_LABELS)]
                logits.append(output.float().cpu().numpy())
        return q4.apply_calibrators(self.calibrators, np.vstack(logits))


class OperationStore:
    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
        export_path: Path = RETRAINING_JSONL_PATH,
        ready_path: Path = RETRAINING_READY_PATH,
    ):
        self.database_path = Path(database_path)
        self.export_path = Path(export_path)
        self.ready_path = Path(ready_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_lock = RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inference_events (
                    event_id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    source_ref TEXT,
                    video_id TEXT,
                    chunk_id TEXT NOT NULL,
                    start_seconds REAL,
                    end_seconds REAL,
                    text TEXT NOT NULL,
                    model_slot TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    predicted_labels_json TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    requires_review INTEGER NOT NULL,
                    review_reasons_json TEXT NOT NULL,
                    review_status TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_model ON inference_events(model_slot);
                CREATE INDEX IF NOT EXISTS idx_events_analysis ON inference_events(analysis_id);
                CREATE TABLE IF NOT EXISTS human_reviews (
                    event_id TEXT PRIMARY KEY REFERENCES inference_events(event_id),
                    reviewed_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    final_labels_json TEXT NOT NULL,
                    reviewer TEXT,
                    notes TEXT
                );
                """
            )

    def record(self, event: dict) -> str:
        event_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO inference_events VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    event["analysis_id"],
                    _now_iso(),
                    event["input_type"],
                    event.get("source_ref"),
                    event.get("video_id"),
                    event["chunk_id"],
                    event.get("start_seconds"),
                    event.get("end_seconds"),
                    event["text"],
                    event["model_slot"],
                    event["model_key"],
                    event["model_label"],
                    json.dumps(event["scores"], ensure_ascii=False),
                    json.dumps(event["predicted_labels"], ensure_ascii=False),
                    event["confidence"],
                    int(event["requires_review"]),
                    json.dumps(event["review_reasons"], ensure_ascii=False),
                    "pending" if event["requires_review"] else "not_required",
                ),
            )
        return event_id

    def save_review(
        self,
        event_id: str,
        action: str,
        final_labels: list[str] | None = None,
        reviewer: str = "",
        notes: str = "",
    ) -> dict:
        action = action.casefold().strip()
        if action not in {"accept", "reject", "modify"}:
            raise OperationError("La revisión debe ser accept, reject o modify.")
        with self._connect() as connection:
            event = connection.execute(
                "SELECT * FROM inference_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if event is None:
                raise OperationError("No existe el evento que se intenta revisar.")
            if connection.execute(
                "SELECT 1 FROM human_reviews WHERE event_id = ?", (event_id,)
            ).fetchone():
                raise OperationError("Este evento ya tiene una revisión humana guardada.")
            predicted = json.loads(event["predicted_labels_json"])
            if action == "accept":
                resolved = predicted
            elif action == "reject":
                resolved = [SAFE_LABEL]
            else:
                resolved = [label for label in (final_labels or []) if label in TARGET_LABELS]
                if not resolved:
                    resolved = [SAFE_LABEL]
            reviewed_at = _now_iso()
            connection.execute(
                "INSERT INTO human_reviews VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    reviewed_at,
                    action,
                    json.dumps(resolved, ensure_ascii=False),
                    str(reviewer or "")[:200],
                    str(notes or "")[:4000],
                ),
            )
            connection.execute(
                "UPDATE inference_events SET review_status = 'completed' WHERE event_id = ?",
                (event_id,),
            )
        export = {
            "schema_version": "1.0",
            "reviewed_at": reviewed_at,
            "chunk_id": event["chunk_id"],
            "video_id": event["video_id"],
            "text": event["text"],
            "coarse_labels": resolved,
            "label_source": "human_production_review",
            "sample_weight": 1.0,
            "source_ref": event["source_ref"],
            "start_seconds": event["start_seconds"],
            "end_seconds": event["end_seconds"],
            "model_slot": event["model_slot"],
            "model_key": event["model_key"],
            "model_scores": json.loads(event["scores_json"]),
            "model_prediction": predicted,
            "human_action": action,
            "reviewer": str(reviewer or "")[:200],
            "notes": str(notes or "")[:4000],
            "exclude_from_existing_validation_test": True,
        }
        line = json.dumps(export, ensure_ascii=False) + "\n"
        with self._export_lock:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            with self.export_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self.materialize_retraining_dataset()
        return {"event_id": event_id, "action": action, "final_labels": resolved}

    def materialize_retraining_dataset(self) -> dict:
        """Deduplica revisiones concordantes y excluye conflictos para reentrenar."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.*, r.reviewed_at, r.action, r.final_labels_json, r.reviewer, r.notes
                FROM inference_events e JOIN human_reviews r USING(event_id)
                ORDER BY r.reviewed_at, e.event_id
                """
            ).fetchall()
        grouped = {}
        for row in rows:
            identity_payload = "|".join(
                [
                    str(row["video_id"] or ""),
                    str(row["start_seconds"] if row["start_seconds"] is not None else ""),
                    _normalize_text(row["text"]).casefold(),
                ]
            )
            identity = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
            grouped.setdefault(identity, []).append(row)
        records, conflicts = [], []
        for identity, values in grouped.items():
            decisions = {
                tuple(sorted(json.loads(row["final_labels_json"]))) for row in values
            }
            if len(decisions) != 1:
                conflicts.append(identity)
                continue
            row = values[-1]
            final_labels = list(next(iter(decisions)))
            records.append(
                {
                    "chunk_id": f"prod_{identity[:24]}",
                    "video_id": row["video_id"] or f"production_text_{identity[:16]}",
                    "text": row["text"],
                    "coarse_labels": final_labels,
                    "label_source": "human_production_review_adjudicated",
                    "sample_weight": 1.0,
                    "source_ref": row["source_ref"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "reviewed_at": row["reviewed_at"],
                    "reviewer": row["reviewer"],
                    "notes": row["notes"],
                    "exclude_from_existing_validation_test": True,
                }
            )
        with self._export_lock:
            self.ready_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.ready_path.with_suffix(self.ready_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ready_path)
        counts = {
            label: sum(label in record["coarse_labels"] for record in records)
            for label in [*TARGET_LABELS, SAFE_LABEL]
        }
        checks = {
            "unique_human_reviewed_at_least_500": len(records) >= RETRAIN_MINIMUM_TOTAL,
            "safe_at_least_200": counts[SAFE_LABEL] >= RETRAIN_MINIMUM_SAFE,
            **{
                f"{label}_at_least_100": counts[label] >= RETRAIN_MINIMUM_PER_DAMAGE
                for label in TARGET_LABELS
            },
        }
        return {
            "unique_adjudicated_chunks": len(records),
            "conflicting_chunks_excluded": len(conflicts),
            "category_counts": counts,
            "checks": checks,
            "ready_for_retraining_review": all(checks.values()),
            "rule_is_advisory": True,
            "output": _path_label(self.ready_path),
        }

    def statistics(self) -> dict:
        with self._connect() as connection:
            events = connection.execute("SELECT * FROM inference_events").fetchall()
            reviews = {
                row["event_id"]: row
                for row in connection.execute("SELECT * FROM human_reviews").fetchall()
            }
        by_model = {}
        for event in events:
            bucket = by_model.setdefault(
                event["model_slot"],
                {
                    "model_key": event["model_key"],
                    "model_label": event["model_label"],
                    "inference_chunks": 0,
                    "requires_review": 0,
                    "reviews_completed": 0,
                    "actions": {"accept": 0, "reject": 0, "modify": 0},
                    "categories": {
                        label: {"predicted": 0, "human_final": 0}
                        for label in [*TARGET_LABELS, SAFE_LABEL]
                    },
                },
            )
            bucket["inference_chunks"] += 1
            bucket["requires_review"] += int(event["requires_review"])
            for label in json.loads(event["predicted_labels_json"]):
                bucket["categories"][label]["predicted"] += 1
            review = reviews.get(event["event_id"])
            if review:
                bucket["reviews_completed"] += 1
                bucket["actions"][review["action"]] += 1
                for label in json.loads(review["final_labels_json"]):
                    bucket["categories"][label]["human_final"] += 1
        readiness = self.materialize_retraining_dataset()
        return {
            "generated_at": _now_iso(),
            "total_events": len(events),
            "total_human_reviews": len(reviews),
            "by_model": by_model,
            "database": _path_label(self.database_path),
            "retraining_export": _path_label(self.export_path),
            "retraining_ready_dataset": _path_label(self.ready_path),
            "retraining_readiness": readiness,
        }


class ModerationService:
    def __init__(
        self,
        registry: dict | None = None,
        *,
        database_path: Path = DATABASE_PATH,
        export_path: Path = RETRAINING_JSONL_PATH,
        ready_path: Path = RETRAINING_READY_PATH,
    ):
        self.registry = registry or load_deployment_registry(verify_hashes=True)
        self.store = OperationStore(database_path, export_path, ready_path)
        self._runners = {}
        self._runner_lock = Lock()

    def public_config(self) -> dict:
        return {
            "target_labels": TARGET_LABELS,
            "safe_is_derived": True,
            "models": {
                slot: {
                    key: value[key]
                    for key in (
                        "model_key",
                        "label",
                        "family",
                        "validation_damage_pr_auc_macro",
                        "selection_partition",
                        "test_used_for_selection",
                    )
                }
                for slot, value in self.registry["models"].items()
            },
            "modes": sorted(VALID_MODES),
            "chunking": self.registry["chunking"],
            "statistics": self.store.statistics(),
        }

    def _runner(self, slot: str):
        with self._runner_lock:
            if slot not in self._runners:
                definition = self.registry["models"][slot]
                factory = {
                    "classical": _ClassicalRunner,
                    "transformer": _TransformerRunner,
                    "qwen": _QwenRunner,
                }[slot]
                self._runners[slot] = factory(definition)
            return self._runners[slot]

    def _interpret(self, slot: str, scores: np.ndarray) -> dict:
        definition = self.registry["models"][slot]
        thresholds = np.asarray(definition["thresholds"], dtype=float)
        policy = definition["review_policy"]
        predicted = scores >= thresholds
        labels = [label for label, active in zip(TARGET_LABELS, predicted) if active]
        labels = labels or [SAFE_LABEL]
        risk_margin = float(np.max(scores - thresholds))
        uncertainty_margin = float(np.min(np.abs(scores - thresholds)))
        reasons = []
        if risk_margin >= float(policy["risk_margin_cutoff"]):
            reasons.append("ruta_alto_recall")
        if uncertainty_margin < float(policy["uncertainty_margin_cutoff"]):
            reasons.append("cerca_del_umbral")
        confidence = (
            "alta"
            if uncertainty_margin >= float(policy["uncertainty_margin_cutoff"])
            else "baja"
        )
        return {
            "model_slot": slot,
            "model_key": definition["model_key"],
            "model_label": definition["label"],
            "scores": {
                label: round(float(value), 6)
                for label, value in zip(TARGET_LABELS, scores)
            },
            "thresholds": {
                label: round(float(value), 6)
                for label, value in zip(TARGET_LABELS, thresholds)
            },
            "predicted_labels": labels,
            "confidence": confidence,
            "requires_review": bool(reasons),
            "review_reasons": reasons,
            "risk_margin": round(risk_margin, 6),
            "uncertainty_margin": round(uncertainty_margin, 6),
        }

    def _prepare_input(
        self,
        value: str,
        input_type: str,
        subtitle_languages: tuple[str, ...],
    ) -> dict:
        if input_type not in VALID_INPUT_TYPES:
            raise OperationError(f"Tipo de entrada no válido: {input_type}")
        detected_youtube = _youtube_video_id(value) is not None
        youtube_candidate = _looks_like_youtube_url(value)
        resolved = (
            "youtube"
            if input_type == "youtube"
            or (input_type == "auto" and youtube_candidate)
            else "text"
        )
        if input_type == "youtube" and not detected_youtube:
            raise OperationError("Se pidió modo YouTube, pero la entrada no es un enlace válido.")
        if resolved == "youtube":
            return _download_youtube(value, subtitle_languages)
        return {
            "input_type": "text",
            "source_ref": None,
            "video_id": None,
            "video_title": None,
            "channel": None,
            "subtitle_language": None,
            "subtitle_kind": None,
            "chunks": _chunk_text(value),
        }

    def analyze(
        self,
        value: str,
        *,
        mode: str = "consensus",
        input_type: str = "auto",
        subtitle_languages: tuple[str, ...] = DEFAULT_SUBTITLE_LANGUAGES,
        max_chunks: int = MAX_CHUNKS,
        persist: bool = True,
    ) -> dict:
        mode = mode.casefold().strip()
        if mode not in VALID_MODES:
            raise OperationError(f"Modo no válido: {mode}")
        prepared = self._prepare_input(value, input_type, subtitle_languages)
        chunks = prepared["chunks"]
        if len(chunks) > max_chunks:
            raise OperationError(
                f"La entrada produjo {len(chunks)} chunks; el límite configurado es {max_chunks}."
            )
        slots = [mode] if mode in self.registry["models"] else list(self.registry["models"])
        texts = [chunk["text"] for chunk in chunks]
        raw_scores = {slot: self._runner(slot).predict_scores(texts) for slot in slots}
        analysis_id = str(uuid.uuid4())
        output_chunks = []
        for index, chunk in enumerate(chunks):
            results = [
                self._interpret(slot, raw_scores[slot][index]) for slot in slots
            ]
            if mode == "consensus":
                votes = {
                    label: sum(
                        label in item["predicted_labels"] for item in results
                    )
                    for label in TARGET_LABELS
                }
                labels = [
                    label
                    for label, count in votes.items()
                    if count >= CONSENSUS_MIN_VOTES
                ] or [SAFE_LABEL]
                disagreement = any(count not in {0, 3} for count in votes.values())
                consensus = {
                    "model_slot": "consensus",
                    "model_key": f"consensus_{CONSENSUS_MIN_VOTES}_of_3",
                    "model_label": "Consenso mayoritario de los tres modelos",
                    "scores": {
                        label: round(
                            float(np.mean([item["scores"][label] for item in results])), 6
                        )
                        for label in TARGET_LABELS
                    },
                    "thresholds": None,
                    "predicted_labels": labels,
                    "confidence": (
                        "alta"
                        if not disagreement and all(item["confidence"] == "alta" for item in results)
                        else "baja"
                    ),
                    "requires_review": disagreement
                    or any(item["requires_review"] for item in results),
                    "review_reasons": [
                        *(["desacuerdo_entre_modelos"] if disagreement else []),
                        *(
                            ["algún_modelo_activa_revisión"]
                            if any(item["requires_review"] for item in results)
                            else []
                        ),
                    ],
                    "votes": votes,
                }
                results.append(consensus)
            for result in results:
                if persist:
                    result["event_id"] = self.store.record(
                        {
                            **result,
                            "analysis_id": analysis_id,
                            "input_type": prepared["input_type"],
                            "source_ref": prepared["source_ref"],
                            "video_id": prepared["video_id"],
                            **chunk,
                        }
                    )
            watch_url = None
            if prepared["video_id"] is not None:
                watch_url = (
                    f"https://www.youtube.com/watch?v={prepared['video_id']}"
                    f"&t={int(chunk['start_seconds'] or 0)}s"
                )
            output_chunks.append({**chunk, "watch_url": watch_url, "results": results})
        alert_chunks = sum(
            any(
                item["predicted_labels"] != [SAFE_LABEL]
                for item in chunk["results"]
                if item["model_slot"] == mode or mode in {"compare", "consensus"}
            )
            for chunk in output_chunks
        )
        return {
            "analysis_id": analysis_id,
            "created_at": _now_iso(),
            "mode": mode,
            **{key: value for key, value in prepared.items() if key != "chunks"},
            "embed_url": (
                f"https://www.youtube-nocookie.com/embed/{prepared['video_id']}"
                if prepared["video_id"]
                else None
            ),
            "summary": {
                "chunks": len(output_chunks),
                "alert_chunks": int(alert_chunks),
                "models_executed": slots,
                "persisted": persist,
            },
            "chunks": output_chunks,
        }


class _Server(ThreadingHTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    service: ModerationService
    html: bytes
    auth_user: str | None
    auth_password: str | None

    def _authorized(self) -> bool:
        if not self.auth_password:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            supplied = base64.b64decode(header[6:], validate=True).decode("utf-8")
        except Exception:
            return False
        expected = f"{self.auth_user or 'moderador'}:{self.auth_password}"
        return hmac.compare_digest(supplied, expected)

    def _require_authorization(self) -> bool:
        if self._authorized():
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Moderador 05", charset="UTF-8"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise OperationError("Tamaño de solicitud inválido.")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as error:
            raise OperationError("JSON de solicitud inválido.") from error

    def do_GET(self):
        path = urlparse(self.path).path
        if path != "/api/health" and not self._require_authorization():
            return
        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self.html)
        elif path == "/api/config":
            self._json(self.service.public_config())
        elif path == "/api/stats":
            self._json(self.service.store.statistics())
        elif path == "/api/health":
            self._json({"status": "ok", "time": _now_iso()})
        else:
            self._json({"error": "Ruta no encontrada."}, 404)

    def do_POST(self):
        try:
            if not self._require_authorization():
                return
            path = urlparse(self.path).path
            payload = self._payload()
            if path == "/api/analyze":
                languages = tuple(payload.get("subtitle_languages") or DEFAULT_SUBTITLE_LANGUAGES)
                result = self.service.analyze(
                    str(payload.get("input", "")),
                    mode=str(payload.get("mode", "consensus")),
                    input_type=str(payload.get("input_type", "auto")),
                    subtitle_languages=languages,
                    max_chunks=int(payload.get("max_chunks", MAX_CHUNKS)),
                    persist=True,
                )
                self._json(result)
            elif path == "/api/review":
                result = self.service.store.save_review(
                    str(payload.get("event_id", "")),
                    str(payload.get("action", "")),
                    list(payload.get("final_labels") or []),
                    str(payload.get("reviewer", "")),
                    str(payload.get("notes", "")),
                )
                self._json(result)
            else:
                self._json({"error": "Ruta no encontrada."}, 404)
        except OperationError as error:
            self._json({"error": str(error)}, int(error.status))
        except Exception as error:
            self._json({"error": f"Error interno: {error}"}, 500)

    def log_message(self, format, *args):
        print(f"[05] {self.address_string()} - {format % args}")


@dataclass
class ServerHandle:
    server: _Server
    thread: Thread
    url: str

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def start_server(
    service: ModerationService | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    allow_network: bool = False,
    auth_user: str | None = None,
    auth_password: str | None = None,
) -> ServerHandle:
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_network:
        raise OperationError(
            "Por seguridad el servidor sólo escucha en loopback. Use allow_network=True explícitamente para cambiarlo."
        )
    if not HTML_PATH.is_file():
        raise FileNotFoundError(f"Falta el HTML autocontenido: {HTML_PATH}")
    service = service or ModerationService()
    handler = type(
        "ModerationHandler",
        (_Handler,),
        {
            "service": service,
            "html": HTML_PATH.read_bytes(),
            "auth_user": auth_user,
            "auth_password": auth_password,
        },
    )
    server = _Server((host, int(port)), handler)
    actual_host, actual_port = server.server_address[:2]
    visible_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{visible_host}:{actual_port}/"
    thread = Thread(target=server.serve_forever, name="moderador-05", daemon=True)
    thread.start()
    if open_browser:
        webbrowser.open(url)
    return ServerHandle(server, thread, url)
