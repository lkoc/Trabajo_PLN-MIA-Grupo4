"""Genera los cuadernos orquestadores activos sin resultados obsoletos incrustados."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf

from notebook_references import apply_citations


ROOT = Path(__file__).resolve().parents[1]

PROJECT_TITLE = (
    "Moderación semiautomática de videos peruanos de YouTube mediante modelos "
    "clásicos y neuronales de procesamiento del lenguaje natural"
)
PROJECT_AUTHORS = [
    "Luis Enrique Koc Góngora",
    "Alex Felipe Mancilla Antay",
    "Herbert Antonio Meléndez García",
    "Dennis Jack Paitán Cano",
]
PROJECT_COVER = (
    f"# {PROJECT_TITLE}\n\n"
    "**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en "
    "Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**\n\n"
    f"**Grupo 4:** {', '.join(PROJECT_AUTHORS[:-1])} y {PROJECT_AUTHORS[-1]}\n\n"
    "---"
)


SETUP = """from pathlib import Path
import sys

def find_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    raise FileNotFoundError('No se encontró pyproject.toml')

ROOT = find_root()
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from moderacion_peru.notebook_ui import show_callout, show_command, show_result, show_summary, show_table
show_summary('Entorno del proyecto', {'raíz': ROOT, 'backend': 'local'}, tone='success')
"""


COLAB_SETUP = """# Backend reproducible: local o Google Colab desde VS Code
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile

COLAB_NOTEBOOK_ID = "__NOTEBOOK_ID__"
COLAB_DRIVE_FOLDER = "ModeracionPeru_Colab"  # Debe coincidir con config/colab_l4.json
COLAB_RUN_ID = ""  # Vacío reanuda <notebook>_working_v2_1; use otro ID para otro experimento
COLAB_REQUIRE_L4 = True
IN_COLAB = importlib.util.find_spec("google.colab") is not None
COLAB_CONTEXT = None

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def _find_local_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("No se encontró pyproject.toml")

if IN_COLAB:
    from google.colab import drive

    # La extensión oficial de Colab para VS Code admite drive.mount desde v0.2.1.
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive") / COLAB_DRIVE_FOLDER
    BUNDLE_DIR = DRIVE_ROOT / "bundle"
    manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Falta bundle_manifest.json. Prepare y suba resultados/colab_bundle a " + str(BUNDLE_DIR)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = BUNDLE_DIR / manifest["core"]["name"]
    if _sha256(core) != manifest["core"]["sha256"]:
        raise ValueError("project_core.zip no coincide con el manifiesto SHA-256")

    RUNTIME_ROOT = Path("/content/moderacion_peru")
    ROOT = RUNTIME_ROOT / "project"
    marker = RUNTIME_ROOT / ".core_sha256"
    expected_core = manifest["core"]["sha256"]
    if not ROOT.is_dir() or not marker.is_file() or marker.read_text().strip() != expected_core:
        if ROOT.exists():
            shutil.rmtree(ROOT)
        ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(core) as archive:
            archive.extractall(ROOT)
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements/colab-l4.txt")]
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(expected_core + "\\n", encoding="utf-8")

    os.environ["MODPERU_ROOT"] = str(ROOT)
    os.environ["HF_HOME"] = "/content/huggingface"
    importlib.invalidate_caches()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.colab import colab_runtime_diagnostics, prepare_colab_context

    COLAB_CONTEXT = prepare_colab_context(
        COLAB_NOTEBOOK_ID,
        project_root=ROOT,
        drive_root=DRIVE_ROOT,
        runtime_root=RUNTIME_ROOT,
        run_id=COLAB_RUN_ID or None,
        require_l4=COLAB_REQUIRE_L4,
        resume=True,
    )
    from moderacion_peru.notebook_ui import show_callout, show_command, show_result, show_summary, show_table
    show_result('Diagnóstico de Colab', colab_runtime_diagnostics(), tone='success')
    show_result('Contexto reproducible', COLAB_CONTEXT.as_dict(), tone='success')
else:
    ROOT = _find_local_root()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.notebook_ui import show_callout, show_command, show_result, show_summary, show_table
    show_summary('Entorno del proyecto', {'raíz': ROOT, 'backend': 'local'}, tone='success')
"""


def colab_setup(notebook_id: str) -> str:
    return COLAB_SETUP.replace("__NOTEBOOK_ID__", notebook_id)


DATASET_CHECKPOINT = """from moderacion_peru.colab import prepare_local_bundle_input

if globals().get('COLAB_CONTEXT') is None:
    dataset_checkpoint = prepare_local_bundle_input('dataset_5_salidas', project_root=ROOT)
else:
    dataset_path = COLAB_CONTEXT.input('dataset_5_salidas')
    dataset_checkpoint = {
        'status': 'verified_in_colab',
        'input_key': 'dataset_5_salidas',
        'path': dataset_path,
        'bytes': dataset_path.stat().st_size,
    }
show_result('Dataset descomprimido y verificado', dataset_checkpoint, tone='success')
"""


SCRAPING_PARAMETERS = """# ══════════════════════════════════════════════════════════════════════════════
# CONTROLES DEL SCRAPING: edite únicamente este bloque
# ══════════════════════════════════════════════════════════════════════════════
DISCOVER_NEW = True           # True: consulta canales/búsquedas y guarda candidatos
FETCH_NEW = True              # True: obtiene subtítulos solo de videos aún no procesados
DISCOVERY_MODE = "seed"       # "seed", "directed" o "both"

MAX_NEW_VIDEOS = None         # None: incluye todos los pendientes; use un entero para un piloto
NETWORK_BATCH_SIZE = 10       # llamadas nuevas por lote antes de una pausa larga
NETWORK_BATCH_PAUSE_SECONDS = 60.0
EXCLUDE_CHANNEL_ON_429 = True # difiere solo el canal afectado; continúa con los demás
RANDOMIZE_DOWNLOAD_QUEUE = True
DOWNLOAD_RANDOM_SEED = 20260806 # orden reproducible e intercalado por canal
MAX_VIDEOS_PER_CHANNEL = 75   # candidatos recientes inspeccionados por canal
MAX_RESULTS_PER_QUERY = 30    # candidatos inspeccionados por consulta dirigida
MAX_DIRECTED_CANDIDATES = None # None: conserva toda la cohorte dirigida inédita
MAX_DIRECTED_SEED_CHANNELS = 16
MAX_EXPANDED_CHANNELS = 20    # canales nuevos inferidos desde búsquedas temáticas
MAX_VIDEOS_PER_EXPANDED_CHANNEL = 30

SUBTITLE_LANGUAGES = ("es-PE", "es-419", "es")
MIN_TRANSCRIPT_CHARACTERS = 200
USE_TRANSCRIPT_API_FALLBACK = True
YT_RETRIES = 3
YT_SLEEP_MIN_SECONDS = 5.0
YT_SLEEP_MAX_SECONDS = 10.0
STOP_ON_VIDEO_ERROR = False   # False: registra el fallo y continúa con el siguiente
SYNC_TRANSCRIPTS_BY_CHANNEL = True # checkpoint pequeño y sincronizable por canal

# Reinicio destructivo recuperable: deje vacío normalmente. Para archivar los
# artefactos activos y reconstruir el dataset de videos desde cero, descomente:
RESET_VIDEO_DATASET = ""
# RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"

if DISCOVERY_MODE not in {"seed", "directed", "both"}:
    raise ValueError("DISCOVERY_MODE debe ser seed, directed o both")
if ((MAX_NEW_VIDEOS is not None and MAX_NEW_VIDEOS < 0)
        or NETWORK_BATCH_SIZE < 1 or NETWORK_BATCH_PAUSE_SECONDS < 0
        or MAX_VIDEOS_PER_CHANNEL < 1 or MAX_RESULTS_PER_QUERY < 1
        or (MAX_DIRECTED_CANDIDATES is not None and MAX_DIRECTED_CANDIDATES < 1)
        or MAX_DIRECTED_SEED_CHANNELS < 1
        or MAX_EXPANDED_CHANNELS < 0 or MAX_VIDEOS_PER_EXPANDED_CHANNEL < 1):
    raise ValueError("Los límites de videos deben ser válidos")
if MIN_TRANSCRIPT_CHARACTERS < 1:
    raise ValueError("MIN_TRANSCRIPT_CHARACTERS debe ser positivo")
if RANDOMIZE_DOWNLOAD_QUEUE and not str(DOWNLOAD_RANDOM_SEED).strip():
    raise ValueError("DOWNLOAD_RANDOM_SEED no puede estar vacío")
if YT_SLEEP_MIN_SECONDS < 0 or YT_SLEEP_MAX_SECONDS < YT_SLEEP_MIN_SECONDS:
    raise ValueError("El intervalo de espera de yt-dlp no es válido")

show_summary('Configuración del scraping', {
    "discover_new": DISCOVER_NEW,
    "fetch_new": FETCH_NEW,
    "discovery_mode": DISCOVERY_MODE,
    "max_new_videos": MAX_NEW_VIDEOS,
    "network_batch_size": NETWORK_BATCH_SIZE,
    "network_batch_pause_seconds": NETWORK_BATCH_PAUSE_SECONDS,
    "exclude_channel_on_429": EXCLUDE_CHANNEL_ON_429,
    "randomize_download_queue": RANDOMIZE_DOWNLOAD_QUEUE,
    "download_random_seed": DOWNLOAD_RANDOM_SEED,
    "max_videos_per_channel": MAX_VIDEOS_PER_CHANNEL,
    "max_results_per_query": MAX_RESULTS_PER_QUERY,
    "max_directed_candidates": MAX_DIRECTED_CANDIDATES,
    "max_directed_seed_channels": MAX_DIRECTED_SEED_CHANNELS,
    "max_expanded_channels": MAX_EXPANDED_CHANNELS,
    "subtitle_languages": SUBTITLE_LANGUAGES,
    "min_transcript_characters": MIN_TRANSCRIPT_CHARACTERS,
    "transcript_api_fallback": USE_TRANSCRIPT_API_FALLBACK,
    "sync_transcripts_by_channel": SYNC_TRANSCRIPTS_BY_CHANNEL,
    "reset_video_dataset_armed": bool(RESET_VIDEO_DATASET),
}, tone='neutral')

if RESET_VIDEO_DATASET:
    from moderacion_peru.acquisition import reset_active_video_dataset
    reset_result = reset_active_video_dataset(ROOT, RESET_VIDEO_DATASET)
    show_result('Dataset activo archivado; reconstrucción desde cero habilitada', reset_result, tone='warning')
"""


SCRAPING_SOURCES = """# Canales semilla recuperados del cuaderno histórico; puede añadir, quitar o editar filas.
# `categoria_fuente` describe el dominio/registro del canal, no una etiqueta de daño.
_SEED_ROWS = [
    ("Marco Sifuentes / Ocram", "politica_analisis", "https://www.youtube.com/@canalYAAAAA"),
    ("El diario de Curwen", "politica_opinion", "https://www.youtube.com/@curwen"),
    ("Sin Guion con Rosa María Palacios", "politica_periodismo", "https://www.youtube.com/@singuionlr"),
    ("RPP Noticias", "politica_actualidad", "https://www.youtube.com/@RPPNoticias"),
    ("Exitosa Noticias", "politica_actualidad", "https://www.youtube.com/@exitosape"),
    ("Willax Television", "politica_opinion", "https://www.youtube.com/@WillaxTV"),
    ("Canal N", "politica_actualidad", "https://www.youtube.com/@canaln"),
    ("ATV Noticias", "politica_actualidad", "https://www.youtube.com/@ATVNoticias"),
    ("Latina Noticias", "politica_actualidad", "https://www.youtube.com/@latinanoticias"),
    ("Panamericana Noticias", "politica_actualidad", "https://www.youtube.com/@Panamericana-Noticias"),
    ("Hablando Huevadas", "humor_streaming", "https://www.youtube.com/@HablandoHuevadasOficial"),
    ("Todo Good", "streaming_opinion", "https://www.youtube.com/@todogoodpe"),
    ("Goblinciano", "streaming_opinion", "https://www.youtube.com/@Goblinciano"),
    ("El Cacas", "humor_streaming", "https://www.youtube.com/@ElCacas"),
    ("Negro Fuertes", "humor_comedia", "https://www.youtube.com/@NegroFuertes"),
    ("Jason Qqq", "humor_streaming", "https://www.youtube.com/@JasonQqqOficial"),
    ("La Cotorrisa Perú", "humor_podcast", "https://www.youtube.com/@LaCotorrisaPeru"),
    ("Magaly TV La Firme", "farandula", "https://www.youtube.com/@MagalyTVLaFirmeATV"),
    ("Amor y Fuego", "farandula", "https://www.youtube.com/@AmoryFuego"),
    ("América Hoy", "farandula", "https://www.youtube.com/@americahoytv"),
    ("Instarándula", "farandula_digital", "https://www.youtube.com/@Instarandula"),
    ("El Popular", "farandula_digital", "https://www.youtube.com/@ElPopularPeru"),
    ("Nico Moschella", "deportes_informal", "https://www.youtube.com/@NicoMoschella"),
    ("Líbero Deportes", "deportes", "https://www.youtube.com/@DiarioLiberoOficial"),
    ("Depor", "deportes", "https://www.youtube.com/@DeporPeru"),
    ("Misias pero viajeras", "viajes", "https://www.youtube.com/c/Misiasperoviajeras"),
    ("Buen Viaje", "viajes", "https://www.youtube.com/c/BuenViajePe"),
    ("Viaja y Prueba", "viajes_gastronomia", "https://www.youtube.com/@ViajayPrueba"),
    ("Cocinando con la Patty", "gastronomia", "https://www.youtube.com/@CocinandoConLaPatty"),
    ("Arde Troya con Juliana Oxenford", "politica_analisis", "https://www.youtube.com/@ardetroyalr"),
    ("Panorama", "politica_periodismo", "https://www.youtube.com/@PanoramaPTV"),
    ("Juanito y Richard", "humor_comedia", "https://www.youtube.com/@JuanitoyRichard"),
    ("Nada Espacial", "humor_podcast", "https://www.youtube.com/@nadaespacialpodcast"),
    ("L1MAX", "deportes_informal", "https://www.youtube.com/@L1MAX_"),
    ("Cocina Cajamarquina", "gastronomia", "https://www.youtube.com/@cocinacajamarquina"),
    ("Tío Lenguado y Descocaos", "viajes_gastronomia", "https://www.youtube.com/@tiolenguado"),
]
SEED_CHANNELS = [
    {"name": name, "categoria_fuente": category, "url": url, "sampling_mode": "seed"}
    for name, category, url in _SEED_ROWS
]

# Ampliación dirigida: las cuotas son máximos por canal, nunca prevalencias esperadas.
DIRECTED_CHANNEL_CATALOG = [
    {"name": "Hablando Huevadas", "url": "https://www.youtube.com/@HablandoHuevadasOficial", "quota": 70, "target_category": "CONTENIDO_SEXUAL|ATAQUE_POR_GENERO_IDENTIDAD|ACOSO_AMENAZA"},
    {"name": "Goblinciano", "url": "https://www.youtube.com/@Goblinciano", "quota": 85, "target_category": "RACISMO_DISCRIMINACION|ACOSO_AMENAZA"},
    {"name": "Juanito y Richard", "url": "https://www.youtube.com/@JuanitoyRichard", "quota": 85, "target_category": "RACISMO_DISCRIMINACION|ACOSO_AMENAZA"},
    {"name": "Arde Troya con Juliana Oxenford", "url": "https://www.youtube.com/@ardetroyalr", "quota": 55, "target_category": "ACOSO_AMENAZA"},
    {"name": "Todo Good", "url": "https://www.youtube.com/@todogoodpe", "quota": 40, "target_category": "ACOSO_AMENAZA"},
    {"name": "Magaly TV La Firme", "url": "https://www.youtube.com/@MagalyTVLaFirmeATV", "quota": 35, "target_category": "ACOSO_AMENAZA|CONTENIDO_SEXUAL"},
]

SEED_SEARCH_QUERIES = [
    "noticias política Perú canal YouTube",
    "periodismo opinión Perú YouTube",
    "humor streaming Perú lenguaje coloquial",
    "farándula espectáculos Perú",
    "deportes peruanos comentarios YouTube",
]
DIRECTED_QUERY_CATALOG = [
    {"query": "insultos racistas discriminación Perú denuncia", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "discriminación regional clasismo racial Perú", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "ataque machista misoginia Perú denuncia", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "ataque homofóbico transfóbico Perú denuncia", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "amenaza de muerte denuncia Perú", "target_category": "ACOSO_AMENAZA"},
    {"query": "extorsionadores amenazan audio Perú", "target_category": "ACOSO_AMENAZA"},
    {"query": "acoso sexual denuncia televisión peruana", "target_category": "CONTENIDO_SEXUAL|ACOSO_AMENAZA"},
    {"query": "cosificación contenido sexual explícito televisión Perú", "target_category": "CONTENIDO_SEXUAL"},
]
"""


SCRAPING_REUSE_AND_PLAN = """import importlib
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import (
    VIDEO_DATASET_RESET_MARKER,
    bootstrap_canonical_from_existing,
    build_directed_sampling_plan,
    discover_existing_transcript_sources,
    load_candidates,
    materialize_transcripts_by_channel,
    merge_candidates,
    select_directed_search_queries,
    select_directed_seed_channels,
)
from moderacion_peru.io import read_jsonl
from moderacion_peru.taxonomy import load_taxonomy

CANONICAL = ROOT/'datos/raw/transcripts_raw.jsonl'
CACHE = ROOT/'datos/raw/transcripts_cache'
TRANSCRIPTS_BY_CHANNEL = ROOT/'datos/raw/transcripts_by_channel'
DIRECTED_DATASET = ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
rebuild_from_zero = (ROOT/VIDEO_DATASET_RESET_MARKER).exists()
sources = [] if rebuild_from_zero else discover_existing_transcript_sources(ROOT, canonical_path=CANONICAL)
reuse_stats = bootstrap_canonical_from_existing(sources, CANONICAL)
reuse_stats['historical_bootstrap_disabled'] = rebuild_from_zero
show_result('Reutilización de snapshots', reuse_stats, tone='warning' if rebuild_from_zero else 'success')
if SYNC_TRANSCRIPTS_BY_CHANNEL:
    channel_partition_stats = materialize_transcripts_by_channel(CANONICAL, TRANSCRIPTS_BY_CHANNEL)
    show_summary('Checkpoint de transcripciones por canal', {
        'videos': channel_partition_stats['total_videos'],
        'canales': channel_partition_stats['total_channels'],
        'partes_jsonl': channel_partition_stats['total_channel_files'],
        'máximo_bytes_por_parte': channel_partition_stats['max_channel_file_bytes'],
        'carpeta': TRANSCRIPTS_BY_CHANNEL,
        'canonico_preservado': CANONICAL.exists(),
    }, tone='success')

taxonomy = load_taxonomy(ROOT/'config/taxonomia_v2.json')
directed_plan = None
DIRECTED_CHANNELS = []
DIRECTED_SEARCH_QUERIES = []
if DISCOVERY_MODE in {'directed', 'both'}:
    directed_plan = build_directed_sampling_plan(
        read_jsonl(DIRECTED_DATASET) if DIRECTED_DATASET.exists() else [],
        read_jsonl(CANONICAL) if CANONICAL.exists() else [],
        damage_labels=taxonomy.damage_labels,
    )
    DIRECTED_CHANNELS = select_directed_seed_channels(
        directed_plan,
        DIRECTED_CHANNEL_CATALOG,
        max_channels=MAX_DIRECTED_SEED_CHANNELS,
    )
    DIRECTED_SEARCH_QUERIES = select_directed_search_queries(
        directed_plan,
        DIRECTED_QUERY_CATALOG,
        max_queries=len(DIRECTED_QUERY_CATALOG),
        max_results_per_query=MAX_RESULTS_PER_QUERY,
    )
    show_summary('Plan de ampliación dirigida', {
        'estrategia': directed_plan['strategy'],
        'soporte_videos_train_validation': directed_plan['support_videos'],
        'déficit_videos': directed_plan['deficit_videos'],
        'pesos': {key: round(value, 4) for key, value in directed_plan['weights'].items()},
        'canales_semilla': len(DIRECTED_CHANNELS),
        'consultas_temáticas': len(DIRECTED_SEARCH_QUERIES),
    }, tone='warning' if directed_plan['strategy'] == 'fallback_equal' else 'success')

CHANNEL_SOURCES = (
    SEED_CHANNELS if DISCOVERY_MODE == 'seed'
    else DIRECTED_CHANNELS if DISCOVERY_MODE == 'directed'
    else SEED_CHANNELS + DIRECTED_CHANNELS
)
SEARCH_QUERIES = (
    SEED_SEARCH_QUERIES if DISCOVERY_MODE == 'seed'
    else DIRECTED_SEARCH_QUERIES if DISCOVERY_MODE == 'directed'
    else SEED_SEARCH_QUERIES + DIRECTED_SEARCH_QUERIES
)
show_summary('Fuentes seleccionadas', {
    'modo': DISCOVERY_MODE,
    'canales': len(CHANNEL_SOURCES),
    'consultas': len(SEARCH_QUERIES),
}, tone='neutral')
"""


SCRAPING_DISCOVERY = """from collections import Counter
import importlib
from tqdm.auto import tqdm
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import (
    discover_youtube_candidates,
    expand_directed_channel_sources,
    processed_video_ids,
    select_directed_candidates,
)
from moderacion_peru.io import append_jsonl_once, write_json_atomic, write_jsonl_atomic

DISCOVERED_PATH = ROOT/'datos/raw/video_candidates.jsonl'
DISCOVERY_FAILURES_PATH = ROOT/'datos/raw/fallos_descubrimiento_ultima_ejecucion.json'
DIRECTED_SELECTION_PATH = ROOT/'datos/raw/directed_candidates_latest.jsonl'
DIRECTED_PLAN_PATH = ROOT/'datos/raw/manifests/directed_plan_latest.json'
discovered = []
directed_selection = []
expanded_channels = []
if DISCOVER_NEW:
    source_outcomes = Counter()
    source_total = len(CHANNEL_SOURCES) + len(SEARCH_QUERIES)
    source_progress = tqdm(total=source_total, desc='Descubriendo fuentes', unit='fuente')

    def report_discovery(event):
        source_outcomes[event['status']] += 1
        source_progress.update(1)
        source_progress.set_postfix(
            correctas=source_outcomes['ok'],
            fallidas=source_outcomes['failed'],
            candidatos=event['candidates_unique'],
        )

    try:
        discovered, discovery_failures = discover_youtube_candidates(
            CHANNEL_SOURCES,
            SEARCH_QUERIES,
            max_videos_per_channel=MAX_VIDEOS_PER_CHANNEL,
            max_results_per_query=MAX_RESULTS_PER_QUERY,
            retries=YT_RETRIES,
            sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
            sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
            progress_callback=report_discovery,
        )
        if directed_plan is not None:
            expanded_channels = expand_directed_channel_sources(
                discovered,
                directed_plan,
                known_channel_ids=[source.get('channel_id') for source in DIRECTED_CHANNELS],
                max_channels=MAX_EXPANDED_CHANNELS,
                videos_per_channel=MAX_VIDEOS_PER_EXPANDED_CHANNEL,
            )
            if expanded_channels:
                source_total += len(expanded_channels)
                source_progress.total = source_total
                source_progress.refresh()
                expanded_candidates, expanded_failures = discover_youtube_candidates(
                    expanded_channels,
                    (),
                    max_videos_per_channel=MAX_VIDEOS_PER_EXPANDED_CHANNEL,
                    max_results_per_query=MAX_RESULTS_PER_QUERY,
                    retries=YT_RETRIES,
                    sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
                    sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
                    progress_callback=report_discovery,
                )
                discovered = merge_candidates(discovered, expanded_candidates)
                discovery_failures.extend(expanded_failures)
            directed_pool = [
                candidate for candidate in discovered
                if candidate.get('sampling_mode') == 'directed'
            ]
            directed_selection = select_directed_candidates(
                directed_pool,
                processed_video_ids(CANONICAL),
                directed_plan,
                max_candidates=(
                    len(directed_pool)
                    if MAX_DIRECTED_CANDIDATES is None
                    else MAX_DIRECTED_CANDIDATES
                ),
            )
            write_jsonl_atomic(DIRECTED_SELECTION_PATH, directed_selection)
            write_json_atomic(DIRECTED_PLAN_PATH, {
                **directed_plan,
                'planned_channels': DIRECTED_CHANNELS,
                'planned_queries': DIRECTED_SEARCH_QUERIES,
                'expanded_channels': expanded_channels,
                'directed_candidates_discovered': len(directed_pool),
                'directed_candidates_selected': len(directed_selection),
                'selection_path': DIRECTED_SELECTION_PATH,
            })
    finally:
        source_progress.close()
    added, existing = append_jsonl_once(DISCOVERED_PATH, discovered, id_field='video_id')
    write_json_atomic(DISCOVERY_FAILURES_PATH, discovery_failures)
    show_summary('Resumen del descubrimiento', {
        "sources_total": source_total,
        "sources_ok": source_outcomes['ok'],
        "sources_failed": source_outcomes['failed'],
        "candidates_unique": len(discovered),
        "candidates_added": added,
        "candidates_existing": existing,
        "expanded_channels": len(expanded_channels),
        "directed_cohort": len(directed_selection),
    }, tone='success' if not discovery_failures else 'warning')
    if discovery_failures:
        failure_counts = Counter(row['failure_kind'] for row in discovery_failures)
        show_summary('Fallos de fuentes por motivo', dict(sorted(failure_counts.items())), tone='warning')
        show_summary('Artefacto de auditoría', {'ruta': DISCOVERY_FAILURES_PATH}, tone='neutral')
else:
    show_callout('Descubrimiento desactivado', 'Se reutilizan candidatos y transcripciones locales.', tone='neutral')
"""


SCRAPING_CANDIDATES = """import importlib
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import order_candidates_for_acquisition, processed_video_ids

if DISCOVERY_MODE == 'directed':
    candidates = load_candidates(DIRECTED_SELECTION_PATH)
    candidate_origin = 'cohorte_dirigida_vigente'
elif DISCOVERY_MODE == 'both' and DISCOVER_NEW:
    seed_candidates_current = [
        candidate for candidate in discovered
        if candidate.get('sampling_mode') == 'seed'
    ]
    candidates = merge_candidates(seed_candidates_current, directed_selection)
    candidate_origin = 'descubrimiento_actual_seed_más_cohorte_dirigida'
else:
    candidate_files = [
        ROOT/'datos/raw/video_candidates.jsonl',
        ROOT/'datos/raw/videos_candidatos.csv',
    ]
    candidates = merge_candidates(*(load_candidates(source) for source in candidate_files))
    candidate_origin = 'archivo_acumulado_general'

processed_ids = processed_video_ids(CANONICAL)
existing_candidate_ids = {str(candidate['video_id']).strip() for candidate in candidates} & processed_ids
pending_candidates = [
    candidate for candidate in candidates
    if str(candidate['video_id']).strip() not in processed_ids
]
if RANDOMIZE_DOWNLOAD_QUEUE:
    pending_candidates = order_candidates_for_acquisition(
        pending_candidates,
        random_seed=DOWNLOAD_RANDOM_SEED,
    )
cached_ids = {path.stem for path in CACHE.glob('*.json')}
pending_cached = sum(
    str(candidate['video_id']).strip() in cached_ids for candidate in pending_candidates
)
show_summary('Candidatos filtrados antes de la adquisición', {
    'origen': candidate_origin,
    'únicos': len(candidates),
    'ya_canónicos_omitidos': len(existing_candidate_ids),
    'pendientes_totales': len(pending_candidates),
    'pendientes_reutilizables_desde_caché': pending_cached,
    'pendientes_que_requieren_red': len(pending_candidates) - pending_cached,
    'cola_pseudoaleatoria': RANDOMIZE_DOWNLOAD_QUEUE,
    'semilla_cola': DOWNLOAD_RANDOM_SEED if RANDOMIZE_DOWNLOAD_QUEUE else None,
}, tone='neutral')
"""


SCRAPING_EXECUTION = """from functools import partial
import importlib
from tqdm.auto import tqdm
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import fetch_youtube_subtitles, ingest_incremental

FAILURES = ROOT/'datos/raw/fallos_adquisicion.jsonl'
fetcher = partial(
    fetch_youtube_subtitles,
    languages=SUBTITLE_LANGUAGES,
    retries=YT_RETRIES,
    sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
    sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
    minimum_transcript_characters=MIN_TRANSCRIPT_CHARACTERS,
    use_transcript_api_fallback=USE_TRANSCRIPT_API_FALLBACK,
)
if pending_candidates:
    video_progress = tqdm(total=len(pending_candidates), desc='Procesando pendientes', unit='video')

    def report_acquisition(event):
        counters = event['counters']
        video_progress.update(event.get('advance', 1))
        if event['status'] == 'batch_pause':
            description = 'Pausa entre lotes'
        else:
            description = 'Procesando pendientes'
        video_progress.set_description(description)
        video_progress.set_postfix(
            existentes=counters['already_canonical'],
            cache=counters['reused_cache'],
            nuevos=counters['fetched'],
            fallidos=counters['failed'],
            diferidos=counters['deferred_by_limit'],
            pausa_429=counters['deferred_rate_limit'],
            intentos_429=counters.get('failure_rate_limited', 0),
            canales_429=counters['rate_limited_channels'],
            lotes=counters['batch_pauses'],
        )

    try:
        stats = ingest_incremental(
            pending_candidates,
            CANONICAL,
            CACHE,
            fetcher=fetcher if FETCH_NEW else None,
            failure_path=FAILURES,
            max_new_videos=MAX_NEW_VIDEOS,
            network_batch_size=NETWORK_BATCH_SIZE,
            batch_pause_seconds=NETWORK_BATCH_PAUSE_SECONDS,
            exclude_rate_limited_channels=EXCLUDE_CHANNEL_ON_429,
            stop_on_error=STOP_ON_VIDEO_ERROR,
            progress_callback=report_acquisition,
            channel_transcript_dir=TRANSCRIPTS_BY_CHANNEL if SYNC_TRANSCRIPTS_BY_CHANNEL else None,
        )
    finally:
        video_progress.close()
    stats = {
        'candidates_total': len(candidates),
        'filtered_existing_before_run': len(existing_candidate_ids),
        'pending_before_run': len(pending_candidates),
        **stats,
    }
    show_summary('Resumen de adquisición', stats, tone='success' if not stats['failed'] else 'warning')
    if stats['failed']:
        show_summary('Fallos de adquisición por motivo', {
            key.removeprefix('failure_'): value
            for key, value in stats.items()
            if key.startswith('failure_') and not key.startswith('failure_records_') and value
        }, tone='warning')
        show_summary('Videos omitidos sin detener el lote', {
            'cantidad': stats['failed'],
            'detalle': FAILURES,
        }, tone='warning')
elif candidates:
    show_callout(
        'Sin videos pendientes',
        'Todos los candidatos ya tienen una transcripción canónica; no se iniciaron descargas.',
        tone='success',
    )
else:
    show_callout(
        'No hay candidatos',
        'Active DISCOVER_NEW o añada un CSV/JSONL. El corpus existente no se vuelve a descargar.',
        tone='warning',
    )
"""


def create(
    path: str,
    title: str,
    purpose: str,
    academic_context: str,
    code_cells: list[tuple[str, str]],
    *,
    colab_notebook_id: str | None = None,
) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "authors": [{"name": name} for name in PROJECT_AUTHORS],
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "moderacion_peru": {
            "project_title": PROJECT_TITLE,
            "academic_work": "Trabajo final de Procesamiento de Lenguaje Natural (PLN)",
            "institution": "Maestría en Inteligencia Artificial, Universidad Nacional de Ingeniería (UNI)",
            "academic_term": "2026-1",
            "group": "Grupo 4",
            "contract": "moderacion_peru_5_salidas_v2",
            "taxonomy_version": "2.1.0",
            "orchestrator": True,
            "colab": {
                "eligible": colab_notebook_id is not None,
                "notebook_id": colab_notebook_id,
                "transport": "google_drive_only" if colab_notebook_id else None,
                "expected_gpu": "NVIDIA L4" if colab_notebook_id else None,
            },
        },
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"{PROJECT_COVER}\n\n## {title}\n\n{purpose}\n\n{academic_context}\n\n"
            "**Contrato v2.1:** `SEGURO` + cuatro daños entrenados, incluida "
            "`ATAQUE_POR_GENERO_IDENTIDAD`. `SEGURO` es excluyente; los daños son multietiqueta. "
            "Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, "
            "sus umbrales y sus reglas de exclusividad son decisiones operativas locales."
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibilidad\n\nEl cuaderno solo orquesta funciones versionadas de `src/moderacion_peru`. "
            "En local no instala paquetes. En Colab, únicamente la celda de bootstrap instala versiones "
            "fijadas desde el bundle SHA-256 de Drive. No usa rutas personales. Revise el README de esta etapa."
        ),
    ]
    if colab_notebook_id:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Backend opcional Google Colab L4 desde VS Code\n\n"
                "Instale la extensión oficial **Google Colab** (`google.colab`), seleccione "
                "`Select Kernel > Colab` y asigne una **NVIDIA L4**. El notebook permanece local; "
                "Drive transporta solo el bundle mínimo verificado. Edite `COLAB_RUN_ID` para separar "
                "experimentos. La compatibilidad de `drive.mount()` desde VS Code requiere la extensión "
                "v0.2.1 o posterior [@googlecolab2026vscode]. La integridad del bundle se comprueba con "
                "SHA-256 [@nist2015sha]. No sincronice cachés de modelos ni escriba checkpoints "
                "directamente en Drive."
            )
        )
        notebook.cells.append(nbf.v4.new_code_cell(colab_setup(colab_notebook_id)))
    else:
        notebook.cells.append(nbf.v4.new_code_cell(SETUP))
    for heading, source in code_cells:
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        notebook.cells.append(nbf.v4.new_code_cell(source))
    if colab_notebook_id:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Publicación o checkpoint en Drive\n\n"
                "Los archivos se generan en el SSD efímero de `/content`. Active esta celda después de "
                "un checkpoint coherente o al finalizar; publica un solo TAR.GZ y luego su manifiesto."
            )
        )
        notebook.cells.append(
            nbf.v4.new_code_cell(
                "PUBLISH_TO_DRIVE = False\n"
                "if COLAB_CONTEXT is not None and PUBLISH_TO_DRIVE:\n"
                "    from moderacion_peru.colab import publish_colab_outputs\n"
                "    show_result('Publicación en Drive', publish_colab_outputs(COLAB_CONTEXT), tone='success')\n"
                "elif COLAB_CONTEXT is not None:\n"
                "    show_callout('Publicación desactivada', 'Cambie PUBLISH_TO_DRIVE=True tras guardar un checkpoint consistente.', tone='neutral')\n"
                "else:\n"
                "    show_callout('Backend local', 'Los artefactos ya permanecen en el workspace.', tone='success')"
            )
        )
    references = apply_citations(notebook.cells, notebook.metadata["moderacion_peru"])
    notebook.cells.append(
        nbf.v4.new_markdown_cell(references, metadata={"tags": ["references"]})
    )
    cell_prefix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{cell_prefix}-{index:02d}"
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)


def main() -> None:
    create(
        "flujo/01_datos/01_01_scraping_incremental.ipynb",
        "01.01 · Adquisición incremental de subtítulos",
        "Reutiliza transcripciones canónicas y cachés por `video_id`; solo consulta YouTube para candidatos nuevos y nunca descarga audio o video.",
        "La adquisición nueva usa `yt-dlp` para escribir pistas VTT sin descargar audio ni video "
        "[@ytdlp2026], y conserva `youtube-transcript-api` únicamente como respaldo cuando la vía "
        "principal no produce una transcripción íntegra [@depoix2026transcript]. "
        "Las transcripciones automáticas se conservan como insumo imperfecto, no como verdad textual, "
        "porque se han documentado sesgos de dialecto y género en el subtitulado automático de YouTube "
        "[@tatman2017captions]. Toda ampliación debe respetar los términos de la plataforma "
        "[@youtube2023terms] y la evaluación ética contextual recomendada para investigación en "
        "Internet [@aoir2020ethics]. La reutilización de cachés y la selección de candidatos son "
        "decisiones locales registradas en manifiestos. El modo dirigido adapta principios de balance "
        "para aprendizaje activo y clasificación multietiqueta de cola larga "
        "[@fairstein2024balancing] [@huang2021balancing]: calcula soporte por videos únicos solo en "
        "`train+validation`, excluye `test`, pondera los déficits, estima rendimiento histórico por "
        "canal, expande canales desde consultas temáticas y materializa una cohorte aislada mediante "
        "*round-robin*. Si no hay etiquetas previas utilizables, asigna 25% a cada daño. La fórmula, "
        "los umbrales, las cuotas, el fallback y el aislamiento por canal ante HTTP 429 son decisiones operativas "
        "propias; `target_category` registra el motivo de selección y nunca constituye una etiqueta. "
        "Este muestreo enriquecido no permite estimar prevalencias en YouTube ni en el Perú.",
        [
            ("Preflight", "from moderacion_peru.artifacts import artifact_status\nshow_result('Disponibilidad de artefactos', artifact_status(ROOT), tone='neutral')"),
            ("Parámetros editables", SCRAPING_PARAMETERS),
            ("Canales y consultas", SCRAPING_SOURCES),
            (
                "Reutilización y plan dirigido\n\n"
                "El soporte se mide por `video_id` único para no favorecer videos largos con más chunks. "
                "La ponderación por déficit y el reparto multietiqueta se inspiran en problemas de "
                "aprendizaje activo desbalanceado y distribuciones de cola larga "
                "[@fairstein2024balancing] [@huang2021balancing]. Su implementación exacta es local: "
                "`test` permanece congelado, el fallback sin historia reparte las cuatro categorías por "
                "igual y las categorías objetivo solo orientan la adquisición.",
                SCRAPING_REUSE_AND_PLAN,
            ),
            ("Descubrimiento general o ampliación dirigida", SCRAPING_DISCOVERY),
            ("Cohorte activa y caché", SCRAPING_CANDIDATES),
            ("Ejecución controlada y tolerante a fallos", SCRAPING_EXECUTION),
        ],
    )
    create(
        "flujo/01_datos/01_02_limpieza_troceado_incremental.ipynb",
        "01.02 · Limpieza y troceado incremental",
        "Crea chunks deterministas únicamente para transcripciones nuevas o modificadas y conserva la versión del troceador.",
        "La normalización NFKC aplicada al texto sigue las formas de normalización Unicode "
        "[@unicode2025normalization], y las huellas de transcripción, texto e identificadores estables "
        "usan SHA-256 [@nist2015sha]. Las ventanas de 30 s, los límites de caracteres, el solapamiento y "
        "las reglas de deduplicación son parámetros locales versionados: las fuentes anteriores no "
        "demuestran que esos valores sean óptimos.",
        [
            ("Configuración", "from moderacion_peru.incremental import chunk_records_incrementally\nfrom moderacion_peru.io import append_jsonl_once, read_jsonl\nSOURCE=ROOT/'datos/raw/transcripts_raw.jsonl'\nOUTPUT=ROOT/'datos/processed/chunks_v2.jsonl'\nVERSION_INDEX=ROOT/'datos/processed/chunking_v2_versions.jsonl'"),
            ("Materialización", "existing=list(read_jsonl(OUTPUT)) if OUTPUT.exists() else []\nversions=list(read_jsonl(VERSION_INDEX)) if VERSION_INDEX.exists() else []\nnew_rows,new_versions,stats=chunk_records_incrementally(read_jsonl(SOURCE) if SOURCE.exists() else [],existing,versions)\nadded,skipped=append_jsonl_once(OUTPUT,new_rows,id_field='chunk_id')\nversions_added,_=append_jsonl_once(VERSION_INDEX,new_versions,id_field='version_id')\nstats.update({'added':added,'duplicate_ids':skipped,'versions_registered':versions_added})\nshow_result('Resultado de limpieza y troceado', stats, tone='success')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_01_etiquetado_local_ollama.ipynb",
        "02.01 · Etiquetado Ollama local o Hugging Face en Colab",
        "Usa Ollama HTTP en local o, opcionalmente, Hugging Face sobre Colab L4; ambos conservan el mismo contrato y reanudan por `chunk_id`.",
        "Ollama admite un JSON Schema en la salida estructurada y su validación posterior con Pydantic "
        "[@ollama2026structured]. El backend local usa exactamente `qwen3.5:4b` y debe registrar el digest "
        "publicado por Ollama [@ollama2026qwen35]; el backend Colab usa `Qwen/Qwen3-4B`, cuyo linaje se "
        "describe en el informe Qwen3 [@qwen2025qwen3] y cuya revisión exacta consta en la tarjeta del "
        "modelo [@hf2026qwen4bcard]. Las salidas LLM son propuestas de anotación y no *ground truth*: "
        "la anotación asistida en tareas subjetivas requiere controles humanos y de anclaje "
        "[@schroeder2025llmassisted]. El prompt, el reintento y la precedencia son decisiones locales.",
        [
            (
                "Selección del proveedor",
                "if COLAB_CONTEXT is not None:\n"
                "    from moderacion_peru.providers import HuggingFaceProvider\n"
                "    provider=HuggingFaceProvider(model='Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c',device='cuda',max_new_tokens=512)\n"
                "    SOURCE=COLAB_CONTEXT.input('chunks_v2')\n"
                "    OUTPUT=COLAB_CONTEXT.scratch_output_dir/'huggingface_qwen3_4b_v2.jsonl'\n"
                "    ERRORS=COLAB_CONTEXT.scratch_output_dir/'huggingface_qwen3_4b_v2.errors.jsonl'\n"
                "else:\n"
                "    from moderacion_peru.providers import OllamaProvider\n"
                "    provider=OllamaProvider(model='qwen3.5:4b')\n"
                "    SOURCE=ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "    OUTPUT=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl'\n"
                "    ERRORS=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.errors.jsonl'\n"
                "show_result('Estado del proveedor', provider.probe(), tone='success')\nshow_summary('Rutas de la campaña', {'entrada': SOURCE, 'salida': OUTPUT, 'errores': ERRORS}, tone='neutral')",
            ),
            (
                "Etiquetado incremental",
                "from moderacion_peru.io import read_jsonl\n"
                "from moderacion_peru.labeling import annotate_incremental\n"
                "RUN=False\nLIMIT=20  # Quite el límite solo después del smoke test\n"
                "if RUN:\n"
                "    show_result('Resultado del etiquetado incremental', annotate_incremental(read_jsonl(SOURCE),provider,OUTPUT,error_path=ERRORS,limit=LIMIT), tone='success')\n"
                "else:\n"
                "    show_callout('Preflight completo', 'Cambie RUN=True. La salida reanuda por chunk_id.', tone='neutral')",
            ),
        ],
        colab_notebook_id="02_01",
    )
    create(
        "flujo/02_etiquetado/02_02_etiquetado_remoto.ipynb",
        "02.02 · Etiquetado remoto opcional",
        "Mantiene un proveedor remoto compatible, pero nunca consume crédito durante preflight ni sin activación explícita.",
        "El identificador y el comportamiento del proveedor remoto se documentan con la publicación "
        "oficial de DeepSeek V4 [@deepseek2026v4]. Esta procedencia identifica el servicio, pero no valida "
        "sus etiquetas. La evidencia sobre anotación asistida por LLM aconseja mantener separadas la "
        "propuesta automática y la decisión humana [@schroeder2025llmassisted]. La activación explícita, "
        "los límites de gasto y la precedencia de fuentes son reglas locales.",
        [
            ("Preflight sin red", "from moderacion_peru.providers import DeepSeekProvider\nprovider=DeepSeekProvider()\nshow_result('Estado del proveedor remoto', provider.probe(), tone='neutral')"),
            ("Ejecución explícita", "from moderacion_peru.io import read_jsonl\nfrom moderacion_peru.labeling import annotate_incremental\nSOURCE=ROOT/'datos/processed/chunks_v2.jsonl'\nOUTPUT=ROOT/'datos/etiquetado/remoto/deepseek_v2.jsonl'\nRUN_REMOTE=False\nif RUN_REMOTE:\n    show_result('Resultado del etiquetado remoto', annotate_incremental(read_jsonl(SOURCE),provider,OUTPUT), tone='success')\nelse:\n    show_callout('API remota desactivada', 'No se realizó ninguna llamada comercial.', tone='neutral')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_03_revision_llm_dirigida.ipynb",
        "02.03 · Revisión LLM dirigida",
        "Prioriza desacuerdos, baja confianza, contexto y clases minoritarias sin tratarlos como verdad humana.",
        "La selección por incertidumbre pertenece a la familia de aprendizaje activo "
        "[@settles2009active], mientras que el balance puede aumentar la atención sobre clases raras "
        "[@fairstein2024balancing]. En lenguaje abusivo, el contexto conversacional puede cambiar la "
        "interpretación del fragmento [@bourgeade2024context]. Como una sugerencia LLM puede influir en "
        "la decisión humana [@choi2024llmeffect], el ordenamiento, el umbral 0.8 y la posibilidad de ocultar "
        "la sugerencia se tratan como decisiones locales que deben auditarse.",
        [
            ("Selección reproducible", "from moderacion_peru.io import read_jsonl\nSOURCE=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl'\nrows=list(read_jsonl(SOURCE)) if SOURCE.exists() else []\nreview=[r for r in rows if r.get('needs_review') or r.get('score_confianza',1)<0.8]\nreview.sort(key=lambda r:r['chunk_id'])\nshow_summary('Cola de revisión dirigida', {'etiquetados': len(rows), 'requieren_revisión': len(review), 'umbral_confianza': 0.8}, tone='warning' if review else 'success')"),
            ("Siguiente paso", "show_callout('Siguiente paso', 'La revisión puede usar otro modelo o pasar directamente a 02_04 para validación humana.', tone='neutral')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_04_consolidacion_validacion_humana.ipynb",
        "02.04 · Consolidación y validación humana",
        "Consolida por precedencia y sirve una interfaz sin datos masivos incrustados.",
        "El acuerdo entre codificadores requiere definir unidad, categorías y medida, no solo contar "
        "coincidencias [@artstein2008agreement]. La intervención humana tampoco garantiza por sí sola "
        "calidad en tareas subjetivas asistidas por LLM [@schroeder2025llmassisted], y mostrar primero la "
        "sugerencia puede producir influencia o anclaje [@choi2024llmeffect]. La precedencia humana, la "
        "adjudicación y el guardado *append-only* son reglas operativas locales.",
        [
            ("Consolidación", "from moderacion_peru.consolidation import consolidate_annotations\nSOURCES=[p for p in [ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl',ROOT/'datos/etiquetado/remoto/deepseek_v2.jsonl'] if p.exists()]\nCHUNKS=ROOT/'datos/processed/chunks_v2.jsonl'\nTRANSCRIPTS=ROOT/'datos/raw/transcripts_raw.jsonl'\nOUTPUT=ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\nif SOURCES:\n    show_result('Consolidación de campañas', consolidate_annotations(SOURCES,OUTPUT,chunks_source=CHUNKS,transcripts_source=TRANSCRIPTS), tone='success')\nelse:\n    show_callout('Sin campañas', 'No hay propuestas para consolidar todavía.', tone='warning')"),
            ("Frontend", "show_command('Iniciar validación humana', f'modperu serve-labeling --campaign {OUTPUT}', description='Ejecute este comando en una terminal del entorno virtual.')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_05_cierre_humano_snapshot.ipynb",
        "02.05 · Cierre humano y snapshot entrenable",
        "Reincorpora el último evento humano por chunk, recupera `video_id` desde el chunk fuente y congela un snapshot inmutable de cinco salidas.",
        "La separación agrupada por video evita que fragmentos correlacionados del mismo video crucen "
        "particiones, una decisión de diseño coherente con el control de sesgo de selección "
        "[@cawley2010selection]. Los snapshots, insumos y manifiestos usan SHA-256 [@nist2015sha]. "
        "La precedencia humana sigue siendo una regla local y no convierte automáticamente toda "
        "intervención en verdad objetiva [@schroeder2025llmassisted].",
        [
            (
                "Reconciliación append-only",
                "from moderacion_peru.consolidation import reconcile_human_reviews\n"
                "CONSOLIDATED=ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\n"
                "CHUNKS=ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "REVIEWS=[ROOT/'datos/etiquetado/humano/labeling_events_v2.jsonl']\n"
                "REVIEWED=ROOT/'datos/etiquetado/consolidado/anotaciones_revisadas_v2.jsonl'\n"
                "show_result('Reconciliación humana', reconcile_human_reviews(CONSOLIDATED,REVIEWS,REVIEWED,chunks_source=CHUNKS), tone='success')",
            ),
            (
                "Snapshot versionado",
                "from moderacion_peru.datasets import materialize_versioned_training_snapshot\n"
                "DATASET=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n"
                "snapshot=materialize_versioned_training_snapshot(REVIEWED,DATASET)\n"
                "show_result('Snapshot entrenable', snapshot, tone='success')\n"
                "show_callout('Idempotencia', 'Sin cambios de entrada, ambas operaciones devuelven status=noop y no reescriben archivos.', tone='neutral')",
            ),
        ],
    )

    training_notebooks = [
        (
            "03_01_modelos_clasicos.ipynb",
            "Modelos clásicos",
            "from moderacion_peru.experiments import train_classical_experiments\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento de modelos clásicos', train_classical_experiments(DATA,ROOT/'modelos/v2/clasicos'), tone='success')\nelse:\n    show_callout('Entrenamiento desactivado', 'Cambie RUN_TRAINING=True: ejecutará fit, calibración en validation, test y candidatos. Si el snapshot no cambió, devuelve noop.', tone='neutral')",
            None,
            "La suite representa texto mediante TF–IDF [@salton1988tfidf] y compara regresión "
            "logística [@cox1958logistic], SVM lineal [@cortes1995svm], Complement Naive Bayes "
            "[@rennie2003cnb] y descenso de gradiente estocástico [@bottou2010sgd]. La implementación "
            "usa scikit-learn [@pedregosa2011sklearn] bajo una transformación uno-contra-resto para el "
            "problema multietiqueta [@tsoumakas2007multilabel]; la elección de n-gramas, pesos de clase e "
            "hiperparámetros es local.",
        ),
        (
            "03_02_transformers_planos.ipynb",
            "Transformers planos",
            "from moderacion_peru.experiments import train_flat_transformers\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/transformers_planos'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento de Transformers planos', train_flat_transformers(DATA,OUTPUT_ROOT,device=DEVICE), tone='success')\nelse:\n    show_summary('Configuración preliminar', {'datos': DATA, 'salida': OUTPUT_ROOT, 'dispositivo': DEVICE, 'acción': 'Cambie RUN_TRAINING=True; MiniLM y E5 completarán fit→calibración→test→candidato o noop.'}, tone='neutral')",
            "03_02",
            "La arquitectura Transformer procede de [@vaswani2017attention]. MiniLM se basa en "
            "destilación de autoatención [@wang2020minilm] y su extensión multilingüe en destilación "
            "entre lenguas [@reimers2020multilingual]; E5 multilingüe se documenta en "
            "[@wang2024e5]. Los checkpoints exactos son `paraphrase-multilingual-MiniLM-L12-v2` "
            "[@hf2026minilmcard] y `multilingual-e5-small` [@hf2026e5card], cargados mediante Transformers "
            "[@wolf2020transformers]. La cabeza de cinco salidas y sus hiperparámetros son locales.",
        ),
        (
            "03_03_transformer_cascada.ipynb",
            "Transformer en cascada",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/cascada'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento de la cascada', train_neural_experiment(DATA,OUTPUT_ROOT,experiment='cascade',device=DEVICE), tone='success')\nelse:\n    show_summary('Configuración preliminar', {'datos': DATA, 'salida': OUTPUT_ROOT, 'dispositivo': DEVICE, 'compuerta': 'cualquier daño', 'salidas_daño': 4, 'acción': 'RUN_TRAINING=True completa el ciclo o devuelve noop.'}, tone='neutral')",
            "03_03",
            "La clasificación jerárquica dispone de taxonomías y estrategias generales "
            "[@silla2011hierarchical], y existen modelos de texto sensibles a jerarquía como HiAGM "
            "[@zhou2020hiagm]. Este cuaderno no implementa HiAGM: la compuerta `SEGURO` frente a cualquier "
            "daño y las cuatro salidas multietiqueta —en el sentido general de [@tsoumakas2007multilabel]— "
            "constituyen una arquitectura local cuya ventaja debe demostrarse en validación.",
        ),
        (
            "03_04_transformer_multitarea.ipynb",
            "Transformer jerárquico multitarea",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/multitarea'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento multitarea', train_neural_experiment(DATA,OUTPUT_ROOT,experiment='multitask',device=DEVICE), tone='success')\nelse:\n    show_summary('Configuración preliminar', {'datos': DATA, 'salida': OUTPUT_ROOT, 'dispositivo': DEVICE, 'salidas_principales': 5, 'auxiliares': '14 finas + 3 flags', 'acción': 'RUN_TRAINING=True completa el ciclo o devuelve noop.'}, tone='neutral')",
            "03_04",
            "El aprendizaje multitarea comparte representaciones entre objetivos relacionados "
            "[@caruana1997multitask], mientras la clasificación jerárquica [@silla2011hierarchical] y "
            "multietiqueta [@tsoumakas2007multilabel] aportan marcos generales para salidas dependientes o "
            "simultáneas. Las cinco salidas principales, las auxiliares finas, los flags y la ponderación "
            "de sus pérdidas son un diseño local y no heredan garantías de transferencia positiva.",
        ),
        (
            "03_05_qwen_lora.ipynb",
            "Qwen-LoRA",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_lora'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento Qwen-LoRA', train_neural_experiment(DATA,OUTPUT_ROOT,experiment='qwen_lora',device=DEVICE), tone='success')\nelse:\n    show_summary('Configuración preliminar', {'datos': DATA, 'salida': OUTPUT_ROOT, 'dispositivo': DEVICE, 'reanuda_colab': bool(COLAB_CONTEXT and COLAB_CONTEXT.resumed), 'acción': 'RUN_TRAINING=True completa fit→calibración→test→candidato o noop.'}, tone='neutral')",
            "03_05",
            "LoRA introduce actualizaciones entrenables de bajo rango sobre un modelo preentrenado "
            "[@hu2022lora]. El backbone pertenece a la familia Qwen3 [@qwen2025qwen3] y se fija en el "
            "checkpoint `Qwen/Qwen3-0.6B-Base` [@hf2026qwen06bcard]; la inyección de adaptadores usa PEFT "
            "0.18.0 [@hf2026peft018]. El rango 8, los módulos objetivo, la cabeza de cinco salidas y la "
            "recalibración son decisiones locales.",
        ),
        (
            "03_06_qwen_estructurado.ipynb",
            "Qwen estructurado",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_estructurado'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN_TRAINING=False\nif RUN_TRAINING:\n    show_result('Entrenamiento Qwen estructurado', train_neural_experiment(DATA,OUTPUT_ROOT,experiment='qwen_structured',device=DEVICE), tone='success')\nelse:\n    show_summary('Configuración preliminar', {'datos': DATA, 'salida': OUTPUT_ROOT, 'dispositivo': DEVICE, 'estructura': 'penaliza conflicto SEGURO+daño durante fit', 'acción': 'RUN_TRAINING=True completa el ciclo o devuelve noop.'}, tone='neutral')",
            "03_06",
            "El backbone se documenta mediante el informe Qwen3 [@qwen2025qwen3] y la tarjeta exacta de "
            "`Qwen/Qwen3-0.6B-Base` [@hf2026qwen06bcard]. La separación entre compuerta y daños toma "
            "como antecedentes la clasificación jerárquica [@silla2011hierarchical] y multietiqueta "
            "[@tsoumakas2007multilabel], pero la estructura concreta y su regla de selección son locales.",
        ),
        (
            "03_07_comparacion_final.ipynb",
            "Comparación final",
            "from moderacion_peru.registry import compare_and_publish_registry\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nCANDIDATE_ROOTS=[ROOT/'modelos/v2']\nREGISTRY=ROOT/'modelos/registro_modelos_5_salidas.json'\nRUN_PUBLISH=False\nif RUN_PUBLISH:\n    show_result('Comparación y registro productivo', compare_and_publish_registry(DATA,CANDIDATE_ROOTS,REGISTRY,comparison_path=ROOT/'resultados/modelos/comparacion_modelos_5_salidas.json'), tone='success')\nelse:\n    show_callout('Publicación desactivada', 'Cambie RUN_PUBLISH=True después de importar los runs de Colab. Solo validation selecciona; test se reporta después.', tone='neutral')",
            None,
            "Las curvas precisión–recall son especialmente informativas con clases desbalanceadas "
            "[@saito2015pr], y el AP del proyecto sigue la definición exacta de `average_precision_score` "
            "[@sklearn2026averageprecision]. Precisión, recall y F1 deben interpretarse por salida y con "
            "promedios explícitos [@sokolova2009metrics]. La calibración se audita porque las redes pueden "
            "estar descalibradas [@guo2017calibration], y test permanece fuera de toda selección para evitar "
            "sesgo [@cawley2010selection]. Falsos seguros y carga de revisión son criterios locales.",
        ),
        (
            "03_08_auditoria_finas_flags.ipynb",
            "Auditoría fina y transversal",
            "from moderacion_peru.datasets import audit_training_snapshot\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT=ROOT/'resultados/auditorias/auditoria_finas_flags_v2.json'\nshow_result('Auditoría de etiquetas finas y flags', audit_training_snapshot(DATA,OUTPUT), tone='success')",
            None,
            "La auditoría parte de tipologías generales de contenido dañino [@banko2020taxonomy] y de "
            "abuso dirigido o generalizado [@waseem2017abuse], pero separa fenómenos implícitos "
            "[@elsherief2021implicit], ironía [@ilic2018irony] y dependencia de contexto "
            "[@bourgeade2024context]. Para el Perú, `RACISMO_DISCRIMINACION` considera racialización "
            "lingüística y política documentada localmente [@almeida2022motoso]; "
            "`ATAQUE_POR_GENERO_IDENTIDAD` se apoya en estudios e informes peruanos sobre violencia de "
            "género en línea [@albornoz2018conocer], afectaciones a mujeres y personas LGBTI "
            "[@defensoria2021violenciaenlinea] y léxico lesbofóbico [@lovon2022lesbofobia]. La frontera de "
            "contenido sexual explícito también responde a política de plataforma "
            "[@youtube2026sexualpolicy]. Las fusiones, nombres y flags exactos siguen siendo locales.",
        ),
    ]
    for filename, subtitle, source, colab_id, academic_context in training_notebooks:
        create(
            f"flujo/03_entrenamiento/{filename}",
            f"03 · {subtitle}",
            "Entrena o audita el contrato v2.1 sin consultar test para seleccionar modelos, épocas o umbrales.",
            academic_context,
            [
                ("Restauración reproducible del dataset", DATASET_CHECKPOINT),
                ("Configuración y ejecución", source),
            ],
            colab_notebook_id=colab_id,
        )
    create(
        "flujo/04_produccion/04_01_frontend_produccion.ipynb",
        "04.01 · Frontend de producción supervisada",
        "Comprueba el registro v2.1 e inicia el demostrador local en modo sombra con texto, subtítulos de YouTube, cinco scores, revisión humana y estadísticas; nunca carga modelos históricos como sustitutos.",
        "La moderación algorítmica presenta retos técnicos y de gobernanza que impiden interpretar un "
        "score como decisión autosuficiente [@gorwa2020moderation]. Los sistemas semiautomáticos pueden "
        "integrar revisión humana [@andersen2021rem], y la abstención tiene antecedentes tanto en la "
        "opción de rechazo [@chow1970reject] como en clasificación selectiva [@geifman2017selective] y "
        "deferencia a una persona experta [@mozannar2020defer]. El modo sombra, los umbrales y los motivos "
        "de revisión son decisiones locales y no constituyen una garantía de seguridad.",
        [
            ("Disponibilidad", "from moderacion_peru.artifacts import artifact_status\nshow_result('Disponibilidad de producción', artifact_status(ROOT), tone='neutral')"),
            ("Inicio", "show_command('Iniciar frontend de producción', 'modperu serve-production --host 127.0.0.1 --port 8765', description='Ejecute este comando en una terminal del entorno virtual.')\nshow_callout('Modo de operación', 'La interfaz reutiliza caché de subtítulos, no descarga audio/video, registra inferencias y permite revisión append-only.', tone='info')"),
        ],
    )


if __name__ == "__main__":
    main()
