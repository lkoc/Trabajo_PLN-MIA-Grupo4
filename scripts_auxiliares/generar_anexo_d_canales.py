"""Genera el inventario reproducible de canales del Apéndice D.

La identidad se deduplica por ``channel_id``. Los metadatos históricos se
resuelven desde las URL de adquisición conservadas y los registros posteriores
aportan directamente el identificador. El resultado solo usa videos presentes
en el corpus integrado final. La tabla muestra por separado los canales con
más de dos videos y agrupa los canales con uno o dos en una fila final.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_DATASET = (
    ROOT
    / "datos/model_ready/transformer_grueso/dataset_integrado_todas_pasadas.jsonl"
)
RAW_CANDIDATES = ROOT / "datos/raw/videos_candidatos.csv"
RAW_TRANSCRIPTS = ROOT / "datos/raw/transcripts_raw.jsonl"
OUTPUT = ROOT / "Documento_final_paper/secciones/anexo_d_canales.tex"

# Resolución de las URL de canal conservadas en los metadatos históricos. Se usa
# el identificador estable para deduplicar aliases y cambios de nombre.
HISTORICAL_CHANNEL_IDS = {
    "https://www.youtube.com/@canalYAAAAA": "UCP0AJJeNkFBYzegTTVbKhPg",
    "https://www.youtube.com/@curwen": "UCHUD6A_lv3OXbzgVXwomtkw",
    "https://www.youtube.com/@singuionlr": "UCdyaE7KCDAD3NYl9r9C69-g",
    "https://www.youtube.com/@RPPNoticias": "UC5j8-2FT0ZMMBkmK72R4aeA",
    "https://www.youtube.com/@exitosape": "UCxgO_rak_BKZP8VNVmYqbWg",
    "https://www.youtube.com/@WillaxTV": "UCort3mldTtMErPqprSlSfng",
    "https://www.youtube.com/@ATVNoticias": "UCYG5uXS3xdsoaXIxum1pAEw",
    "https://www.youtube.com/@latinanoticias": "UCpSJ5fGhmAME9Kx2D3ZvN3Q",
    "https://www.youtube.com/@Panamericana-Noticias": "UCOyD-kV3zB8Cm4LI4qtd9tA",
    "https://www.youtube.com/@HablandoHuevadasOficial": "UCba1vMvOHWlMddLARS382Zw",
    "https://www.youtube.com/@todogoodpe": "UC-f4h9oSW5JreShkauu0m1A",
    "https://www.youtube.com/@Goblinciano": "UCO0cdOFdG-3hxa_VPG8imnw",
    "https://www.youtube.com/@MagalyTVLaFirmeATV": "UCF6s3gpbZEQKqZ38VvQ0gLA",
    "https://www.youtube.com/@Instarandula": "UC2EkQJU0Ani3SLAVS07MBJg",
    "https://www.youtube.com/@ElPopularPeru": "UCHTeIWzLp_m7xzm3c9GP61g",
    "https://www.youtube.com/@DiarioLiberoOficial": "UCk2OZrA0E6q6xp4bBKtf9KA",
    "https://www.youtube.com/c/Misiasperoviajeras": "UCknQM__AyaqSdxunkqpavDg",
    "https://www.youtube.com/c/BuenViajePe": "UCpTyTnL_1Hs6LyfD4L2rkpw",
    "https://www.youtube.com/@ViajayPrueba": "UCD8fpCkckYmYj_qm6hhE4jw",
    "https://www.youtube.com/@ardetroyalr": "UCgEabnv1xth8-qeWqMsTmDw",
    "https://www.youtube.com/@PanoramaPTV": "UCBiq92lt_ufNO-ktZigVXgg",
    "https://www.youtube.com/@JuanitoyRichard": "UCfiBnBtw8iNbX8vL4f6cp0Q",
    "https://www.youtube.com/@nadaespacialpodcast": "UCfwkQ4lY6UO-6K_REQ8CnNQ",
    "https://www.youtube.com/@L1MAX_": "UCGVHVLD7Nzw0zdIwbVhE3vw",
    "https://www.youtube.com/@cocinacajamarquina": "UC2lyekNE9NOGWu1R0L8046A",
    "https://www.youtube.com/@tiolenguado": "UCvPVK5xvxJRY1obR9XRrMfw",
}

# El video colaborativo preservó dos nombres en ``channel_title``; su uploader
# canónico se verificó desde la URL del video y se registra por channel_id.
VIDEO_OVERRIDES = {
    "jNMayVpAULM": (
        "MARISOL LA FARAONA DE LA CUMBIA",
        "UC3p6KiAv8Aiv0rY2-zWNW-w",
    )
}


def _latex_text(value: str) -> str:
    """Escapa texto y elimina símbolos no compatibles con pdfLaTeX."""

    value = "".join(
        char for char in unicodedata.normalize("NFC", value) if unicodedata.category(char) != "So"
    ).strip()
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _sort_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


def _load_final_video_ids() -> set[str]:
    video_ids: set[str] = set()
    with FINAL_DATASET.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            video_ids.add(str(row["video_id"]))
    return video_ids


def _load_historical_lookup() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    by_video: dict[str, dict[str, str]] = {}
    url_by_title: dict[str, str] = {}
    with RAW_CANDIDATES.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_video[str(row["video_id"])] = row
            title = str(row.get("channel_title", "")).strip()
            url = str(row.get("channel_url", "")).strip()
            if title and url:
                url_by_title.setdefault(title, url)
    return by_video, url_by_title


def build_inventory() -> list[dict[str, object]]:
    final_ids = _load_final_video_ids()
    historical_by_video, historical_url_by_title = _load_historical_lookup()
    records: dict[str, tuple[str, str]] = {}

    with RAW_TRANSCRIPTS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id not in final_ids:
                continue
            candidate = historical_by_video.get(video_id, {})
            name = str(row.get("channel_title") or candidate.get("channel_title") or "").strip()
            channel_url = str(candidate.get("channel_url") or historical_url_by_title.get(name, ""))
            channel_id = HISTORICAL_CHANNEL_IDS.get(channel_url, "")
            records[video_id] = (name, channel_id)

    selected_files = sorted(
        (ROOT / "datos/ampliacion").glob("*/raw/videos_seleccionados.csv")
    )
    for selected_file in selected_files:
        with selected_file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                video_id = str(row["video_id"])
                if video_id not in final_ids:
                    continue
                name = str(row.get("channel_title", "")).strip()
                channel_id = str(row.get("channel_id", "")).strip()
                if not channel_id:
                    historical_url = historical_url_by_title.get(name, "")
                    channel_id = HISTORICAL_CHANNEL_IDS.get(historical_url, "")
                if video_id in VIDEO_OVERRIDES:
                    name, channel_id = VIDEO_OVERRIDES[video_id]
                records[video_id] = (name, channel_id)

    missing_videos = sorted(final_ids - records.keys())
    missing_channels = sorted(
        video_id for video_id, (_, channel_id) in records.items() if not channel_id
    )
    if missing_videos or missing_channels:
        raise RuntimeError(
            f"Inventario incompleto: {len(missing_videos)} videos sin metadatos y "
            f"{len(missing_channels)} sin channel_id"
        )

    names: dict[str, Counter[str]] = defaultdict(Counter)
    videos: Counter[str] = Counter()
    for name, channel_id in records.values():
        names[channel_id][name] += 1
        videos[channel_id] += 1

    inventory: list[dict[str, object]] = []
    for channel_id, aliases in names.items():
        name = aliases.most_common(1)[0][0]
        inventory.append(
            {
                "name": name,
                "channel_id": channel_id,
                "url": f"https://www.youtube.com/channel/{channel_id}",
                "videos": videos[channel_id],
            }
        )
    return sorted(inventory, key=lambda row: _sort_key(str(row["name"])))


def render_latex(inventory: list[dict[str, object]]) -> str:
    recurrent = [row for row in inventory if int(row["videos"]) > 2]
    other_channels = [row for row in inventory if int(row["videos"]) <= 2]
    other_videos = sum(int(row["videos"]) for row in other_channels)
    lines = [
        "% Archivo generado por scripts_auxiliares/generar_anexo_d_canales.py.",
        "% No editar manualmente: regenerar después de cambiar el corpus final.",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{50mm}>{\raggedright\arraybackslash}p{103mm}r}",
        r"\caption{Canales con más de dos videos en el corpus final y agrupación de canales con uno o dos}\label{tab:canales_dataset}\\",
        r"\toprule",
        r"\textbf{Canal} & \textbf{Dirección web principal} & \textbf{Videos} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{3}{c}{\tablename\ \thetable\ (continuación)} \\",
        r"\toprule",
        r"\textbf{Canal} & \textbf{Dirección web principal} & \textbf{Videos} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{Continúa en la página siguiente} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in recurrent:
        lines.append(
            f'{_latex_text(str(row["name"]))} & '
            f'\\url{{{row["url"]}}} & {row["videos"]} \\\\'
        )
    lines.append(r"\midrule")
    lines.append(
        f"Otros ({len(other_channels)} canales con uno o dos videos) & "
        r"Direcciones canónicas conservadas en los metadatos del corpus & "
        f"{other_videos} \\\\"
    )
    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def main() -> None:
    inventory = build_inventory()
    if len(inventory) != 250:
        raise RuntimeError(f"Se esperaban 250 canales únicos y se obtuvieron {len(inventory)}")
    OUTPUT.write_text(render_latex(inventory), encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(inventory)} canales")


if __name__ == "__main__":
    main()
