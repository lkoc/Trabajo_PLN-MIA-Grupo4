from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BODY_CITATION = re.compile(r"\[(\d+)\]")
REFERENCE_ENTRY = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)


def test_active_notebooks_are_ordered_and_clean():
    notebooks = sorted((ROOT / "flujo").rglob("*.ipynb"))
    assert len(notebooks) == 17
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.metadata["moderacion_peru"]["contract"] == "moderacion_peru_5_salidas_v2"
        assert notebook.metadata["moderacion_peru"]["taxonomy_version"] == "2.1.0"
        assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "LM Studio" not in source
        assert "D:\\" not in source
        assert "G:\\" not in source
        for cell in notebook.cells:
            if cell.cell_type == "code":
                tree = ast.parse(cell.source)
                print_calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ]
                assert not print_calls, f"Use notebook_ui en lugar de print(): {path}"


def test_each_notebook_has_consistent_ieee_references_as_final_cell():
    master_bib = (ROOT / "Documento_final_paper" / "referencias.bib").read_text(encoding="utf-8")
    master_keys = set(re.findall(r"^@[A-Za-z]+\{([^,]+),", master_bib, re.MULTILINE))
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        metadata = notebook.metadata["moderacion_peru"]
        keys = list(metadata["citation_keys"])
        assert metadata["citation_style"] == "IEEE_numeric"
        assert metadata["reference_count"] == len(keys)
        assert keys
        assert len(keys) == len(set(keys))
        assert set(keys) <= master_keys
        final_cell = notebook.cells[-1]
        assert final_cell.cell_type == "markdown"
        assert "references" in final_cell.metadata.get("tags", [])
        assert final_cell.source.startswith("## Referencias\n\n[1] ")
        expected = set(range(1, len(keys) + 1))
        body = "\n".join(
            cell.source for cell in notebook.cells[:-1] if cell.cell_type == "markdown"
        )
        assert set(map(int, BODY_CITATION.findall(body))) == expected
        assert list(map(int, REFERENCE_ENTRY.findall(final_cell.source))) == list(range(1, len(keys) + 1))
        assert "[@" not in "\n".join(cell.source for cell in notebook.cells)


def test_colab_notebooks_embed_reproducible_drive_bootstrap():
    expected = {"02_01", "03_02", "03_03", "03_04", "03_05", "03_06"}
    observed = set()
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        metadata = notebook.metadata["moderacion_peru"]["colab"]
        source = "\n".join(cell.source for cell in notebook.cells)
        if metadata["eligible"]:
            observed.add(metadata["notebook_id"])
            assert "drive.mount" in source
            assert "prepare_colab_context" in source
            assert "publish_colab_outputs" in source
            assert "COLAB_REQUIRE_L4 = True" in source
            assert "pip\", \"install\", \"-q\", \"-r\"" in source
            assert "git clone" not in source.lower()
            assert metadata["transport"] == "google_drive_only"
    assert observed == expected


def test_required_frontends_are_small_templates():
    human = ROOT / "flujo/02_etiquetado/frontend/validacion_humana.html"
    production = ROOT / "flujo/04_produccion/frontend/produccion.html"
    assert human.stat().st_size < 100_000
    assert production.stat().st_size < 100_000
    human_source = human.read_text(encoding="utf-8")
    assert "t.safe_label" in human_source
    assert "event.target.value===t.safe_label" in human_source
    assert "/api/reviews" in human_source
    assert "previous_text" in human_source
    assert "localStorage" in human_source
    assert "display_name" in human_source
    assert "display_name" in production.read_text(encoding="utf-8")
    assert "Modo sombra" in production.read_text(encoding="utf-8")
    production_source = production.read_text(encoding="utf-8")
    assert "/api/analyze" in production_source
    assert "/api/stats" in production_source
    assert "youtube.com" in production_source


def test_academic_cover_is_present_in_notebooks_frontends_and_readmes():
    title = (
        "Moderación semiautomática de videos peruanos de YouTube mediante modelos "
        "clásicos y neuronales de procesamiento del lenguaje natural"
    )
    course = (
        "Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en "
        "Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1"
    )
    authors = (
        "Luis Enrique Koc Góngora",
        "Alex Felipe Mancilla Antay",
        "Herbert Antonio Meléndez García",
        "Dennis Jack Paitán Cano",
    )
    readmes = [
        path
        for path in ROOT.rglob("README.md")
        if "archivo" not in path.parts and ".git" not in path.parts and ".pytest_cache" not in path.parts
    ]
    frontends = sorted((ROOT / "flujo").rglob("*.html"))
    for path in readmes + frontends:
        source = path.read_text(encoding="utf-8-sig")
        assert all(item in source for item in (title, course, *authors)), path
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        assert all(item in notebook.cells[0].source for item in (title, course, *authors)), path
        assert notebook.metadata["moderacion_peru"]["project_title"] == title
        assert notebook.metadata["moderacion_peru"]["academic_term"] == "2026-1"
        assert notebook.metadata["moderacion_peru"]["group"] == "Grupo 4"


def test_root_readme_summarizes_and_reproduces_the_active_workflow():
    source = (ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = (
        "## Resumen",
        "## Arquitectura del flujo",
        "## Taxonomía operativa",
        "## Qué realiza cada etapa",
        "## Reproducción local desde una clonación nueva",
        "## Incrementar la muestra sin reiniciar",
        "## Google Colab L4 desde VS Code",
        "## Alcance ético y operativo",
    )
    assert all(section in source for section in required_sections)
    assert source.count("```mermaid") >= 2
    assert "git clone https://github.com/lkoc/Trabajo_PLN-MIA-Grupo4.git" in source
    assert ".[datos,etiquetado,cuadernos,dev]" in source
    assert ".[entrenamiento]" in source
    assert "modperu.exe preflight" in source
    assert "python.exe -m pytest" in source
    for folder, expected_count in {
        "01_datos": 3,
        "02_etiquetado": 5,
        "03_entrenamiento": 8,
        "04_produccion": 1,
    }.items():
        notebooks = sorted((ROOT / "flujo" / folder).glob("*.ipynb"))
        assert len(notebooks) == expected_count
        assert all(path.stem in source for path in notebooks)


def test_scraping_notebook_exposes_historical_controls_and_safe_failure_mode():
    path = ROOT / "flujo" / "01_datos" / "01_01_scraping_incremental.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    for control in (
        "DISCOVER_NEW",
        "FETCH_NEW",
        "BACKFILL_MISSING_VTT",
        "DISCOVERY_MODE",
        "MAX_NEW_VIDEOS",
        "MAX_VTT_BACKFILL",
        "MAX_VIDEOS_PER_CHANNEL",
        "MAX_RESULTS_PER_QUERY",
        "MAX_DIRECTED_CANDIDATES",
        "MAX_EXPANDED_CHANNELS",
        "EXCLUDE_CHANNEL_ON_429",
        "RANDOMIZE_DOWNLOAD_QUEUE",
        "DOWNLOAD_RANDOM_SEED",
        "CHANNEL_SOURCES",
        "SEARCH_QUERIES",
        "STOP_ON_VIDEO_ERROR",
        "SYNC_TRANSCRIPTS_BY_CHANNEL",
        "SYNC_VTT_BY_VIDEO",
        "YT_SOCKET_TIMEOUT_SECONDS",
        "RESUME_DISCOVERY",
        "DISCOVERY_CHECKPOINT_PATH",
        "RESET_VIDEO_DATASET",
        "fallos_adquisicion.jsonl",
    ):
        assert control in source
    assert "canonical_ids = processed_video_ids(CANONICAL)" in source
    assert "consolidate_available_transcripts" in source
    assert "collect_project_video_inventory" in source
    assert "KNOWN_VIDEO_IDS" in source
    assert "videos_conocidos_globales" in source
    assert "pending_candidates = [" in source
    assert "total=len(pending_candidates)" in source
    assert "ingest_incremental(\n            pending_candidates," in source
    assert "build_directed_sampling_plan" in source
    assert "expand_directed_channel_sources" in source
    assert "select_directed_candidates" in source
    assert "if MAX_DIRECTED_CANDIDATES is None" in source
    assert "len(directed_pool)" in source
    assert "importlib.reload(acquisition_module)" in source
    assert "materialize_transcripts_by_channel" in source
    assert "materialize_vtt_checkpoint" in source
    assert "backfill_missing_vtt" in source
    assert "fuente=source_name" in source
    assert "checkpoint_path=DISCOVERY_CHECKPOINT_PATH" in source
    assert "channel_transcript_dir=TRANSCRIPTS_BY_CHANNEL" in source
    assert "cohorte_dirigida_vigente" in source
    assert '# RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"' in source
    assert not (ROOT / "flujo" / "01_datos" / "01_03_ampliacion_dirigida.ipynb").exists()
    assert (
        ROOT
        / "archivo"
        / "flujo_reorganizado_v2"
        / "01_03_ampliacion_dirigida_reemplazado.ipynb"
    ).is_file()


def test_chunk_length_pilot_is_optional_and_materialization_is_separate():
    pilot_path = ROOT / "flujo" / "01_datos" / "01_02_optimizacion_longitud_chunks.ipynb"
    materialization_path = ROOT / "flujo" / "01_datos" / "01_03_limpieza_troceado_incremental.ipynb"
    assert pilot_path.is_file()
    assert materialization_path.is_file()
    assert not (ROOT / "flujo" / "01_datos" / "01_02_limpieza_troceado_incremental.ipynb").exists()
    pilot = nbformat.read(pilot_path, as_version=4)
    source = "\n".join(cell.source for cell in pilot.cells)
    for control in (
        "RUN_CHUNK_LENGTH_SMOKE_TEST=False",
        "RUN_CHUNK_LENGTH_CONFIRMATORY_TEST=False",
        "RUN_BOUNDED_HF_COMPARISON=False",
        "RUN_BOUNDED_OLLAMA_COMPARISON=False",
        "CANDIDATE_SECONDS=(15,20,25,30,35)",
        "NEURAL_CANDIDATE_SECONDS=(20,30)",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "OLLAMA_SMOKE_MODEL='gemma3:4b'",
        "OLLAMA_SMOKE_MAX_WALL_SECONDS=600.0",
        "CONFIRMATORY_VIDEO_LIMITS={'train':200,'validation':80,'test':80}",
        "CONFIRMATORY_SEEDS=(20260805,20260817,20260829)",
        "MANUAL_CHUNK_SECONDS=30.0",
        "USE_SMOKE_RECOMMENDATION=False",
        "USE_CONFIRMATORY_RECOMMENDATION=False",
        "APPLY_CHUNK_SELECTION=False",
        "complement_nb",
        "sgd_incremental",
        "run_bounded_neural_chunk_comparison",
        "prepare_local_bundle_input('dataset_5_salidas'",
    ):
        assert control in source
    assert "test_used_for_selection" not in source
    materialization = nbformat.read(materialization_path, as_version=4)
    materialization_source = "\n".join(cell.source for cell in materialization.cells)
    assert "activate_chunking_configuration" in materialization_source
    assert "**CHUNK_CONFIG" in materialization_source


def test_dataset_consumers_restore_and_verify_the_synced_checkpoint():
    for path in sorted((ROOT / "flujo" / "03_entrenamiento").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "prepare_local_bundle_input('dataset_5_salidas'" in source, path
        assert "Dataset descomprimido y verificado" in source, path


def test_synced_checkpoint_rules_preserve_hashes_and_exclude_rebuildable_working_files():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "!datos/raw/transcripts_by_channel/**" in ignore
    assert "!datos/raw/vtt_by_video/**" in ignore
    assert "!datos/raw/video_candidates.jsonl" in ignore
    assert "!resultados/colab_bundle/dataset_5_salidas.jsonl.gz" in ignore
    assert "datos/raw/transcripts_raw.jsonl" in ignore
    assert "datos/raw/transcripts_cache/" in ignore
    assert "datos/processed/*" in ignore
    assert "datos/model_ready/v2/*" in ignore
    assert "*.jsonl text eol=lf" in attributes
    assert "*.gz binary" in attributes


def test_neural_model_revisions_are_pinned():
    from moderacion_peru.models import TRANSFORMER_SPECS

    assert all(spec.revision and len(spec.revision) == 40 for spec in TRANSFORMER_SPECS.values())


def test_lm_studio_only_exists_in_archive():
    active_roots = [ROOT / "src", ROOT / "flujo"]
    active_files = [path for root in active_roots for path in root.rglob("*") if path.is_file()]
    hits = []
    for path in active_files:
        if path.suffix.lower() not in {".md", ".py", ".ipynb", ".html", ".json"}:
            continue
        try:
            if "LM Studio" in path.read_text(encoding="utf-8"):
                hits.append(path.relative_to(ROOT).as_posix())
        except UnicodeDecodeError:
            pass
    assert hits == []
