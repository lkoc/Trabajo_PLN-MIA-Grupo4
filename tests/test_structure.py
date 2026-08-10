from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
BODY_CITATION = re.compile(r"\[(\d+)\]")
REFERENCE_ENTRY = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)
TRAINED_OUTPUTS = (
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ATAQUE_POR_GENERO_IDENTIDAD",
    "ACOSO_AMENAZA",
    "CONTENIDO_SEXUAL",
)
ABBREVIATED_CONTRACT = "`SEGURO` + cuatro daños entrenados, incluida"
CONTRACT_SUMMARY_DOCUMENTS = (
    "README.md",
    "datos/README.md",
    "datos/model_ready/v2/README.md",
    "resultados/README.md",
    "modelos/README.md",
    "bibliografia/fuentes_base.md",
    "Planning/PLAN_REORGANIZACION_REPRODUCIBLE.md",
    "config/prompt_operacional_ollama_v3_2.md",
    "docs/TAXONOMIA_V2.md",
    "docs/CONTRATOS_DATOS.md",
    "docs/MATRIZ_TRAZABILIDAD.md",
    "docs/MIGRACION_Y_COMPATIBILIDAD.md",
    "docs/ORDEN_EJECUCION.md",
    "docs/OPTIMIZACION_LONGITUD_CHUNKS.md",
    "docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md",
    "flujo/01_datos/README.md",
    "flujo/02_etiquetado/README.md",
    "flujo/03_entrenamiento/README.md",
    "flujo/04_produccion/README.md",
    "Documento_final_paper/README.md",
    "Documento_final_paper/AUDITORIA_CITAS_Y_ESTILO.md",
    "Documento_final_paper/guia_estructura_paper_ieee.md",
    "Documento_final_paper/guia_redaccion_paper_ieee.md",
    "Documento_final_paper/figuras/README.md",
    "Presentación_BEAMER/README.md",
)


def test_active_notebooks_are_ordered_and_clean():
    notebooks = sorted((ROOT / "flujo").rglob("*.ipynb"))
    assert len(notebooks) == 20
    executed_reporting_notebooks = {
        "01_03_limpieza_troceado_incremental.ipynb",
        "02_00_preparacion_bundle_colab.ipynb",
        "02_01_etiquetado_deepseek_flash_pro.ipynb",
        "02_03_revision_llm_dirigida.ipynb",
        "02_04_consolidacion_validacion_humana.ipynb",
        "02_05_cierre_humano_snapshot.ipynb",
        "03_08_auditoria_finas_flags.ipynb",
    }
    optionally_executed_notebooks = {
        "01_015_ampliacion_dirigida_minorias.ipynb",
    }
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert (
            notebook.metadata["moderacion_peru"]["contract"]
            == "moderacion_peru_5_salidas_v2"
        )
        assert notebook.metadata["moderacion_peru"]["taxonomy_version"] == "2.1.0"
        output_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.get("outputs")
        ]
        if path.name in executed_reporting_notebooks:
            # Estas salidas ejecutadas pertenecen al usuario y constituyen evidencia
            # operativa; se conservan en lugar de reescribir sus cuadernos.
            assert output_cells
        elif path.name not in optionally_executed_notebooks:
            assert not output_cells
        source = "\n".join(cell.source for cell in notebook.cells)
        assert all(output in notebook.cells[0].source for output in TRAINED_OUTPUTS)
        assert ABBREVIATED_CONTRACT not in source
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


def test_active_contract_summaries_name_every_trained_output():
    for relative in CONTRACT_SUMMARY_DOCUMENTS:
        source = (ROOT / relative).read_text(encoding="utf-8-sig")
        assert all(output in source for output in TRAINED_OUTPUTS), relative
        assert ABBREVIATED_CONTRACT not in source, relative


def test_each_notebook_has_consistent_ieee_references_as_final_cell():
    master_bib = (ROOT / "Documento_final_paper" / "referencias.bib").read_text(
        encoding="utf-8"
    )
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
        assert list(map(int, REFERENCE_ENTRY.findall(final_cell.source))) == list(
            range(1, len(keys) + 1)
        )
        assert "[@" not in "\n".join(cell.source for cell in notebook.cells)


def test_neural_chunk_report_has_consistent_numeric_references():
    report = (ROOT / "docs" / "ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md").read_text(
        encoding="utf-8"
    )
    body, references = report.split("## Referencias", maxsplit=1)
    cited = set(map(int, BODY_CITATION.findall(body)))
    entries = list(map(int, REFERENCE_ENTRY.findall(references)))

    assert cited == set(range(1, 14))
    assert entries == list(range(1, 14))
    assert len(entries) == len(set(entries))
    assert "doi.org/10.18653/v1/P18-1128" in references
    assert "proceedings.neurips.cc/paper/2020" in references


def test_chunk_optimization_notebook_loads_consolidated_results_before_models():
    notebook = nbformat.read(
        ROOT / "flujo" / "01_datos" / "01_02_optimizacion_longitud_chunks.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "FORCE_NEURAL_ROBUST_RECOMPUTE=False" in source
    assert "FORCE_MINILM_20_30_RECOMPUTE=False" in source
    assert source.index(
        "if NEURAL_RESULT_PATH.is_file() and not FORCE_NEURAL_ROBUST_RECOMPUTE:"
    ) < source.index("elif RUN_NEURAL_ROBUST_TEST:")
    assert source.index(
        "if MINILM_NI_RESULT_PATH.is_file() and not FORCE_MINILM_20_30_RECOMPUTE:"
    ) < source.index("elif RUN_MINILM_20_30_NONINFERIORITY_TEST:")
    assert "no se llamó MiniLM ni Ollama" in source
    assert "no se recalcularon embeddings, ajustes ni bootstrap" in source


def test_colab_notebooks_embed_reproducible_drive_bootstrap():
    expected_consumers = {
        "02_01",
        "02_02",
        "03_02",
        "03_03",
        "03_04",
        "03_05",
        "03_06",
        "03_06b",
    }
    expected = {"02_00", *expected_consumers}
    manifest = json.loads(
        (ROOT / "resultados" / "colab_bundle" / "bundle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_bundle_id = manifest["bundle_id"]
    expected_core_sha256 = manifest["core"]["sha256"]
    observed = set()
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        metadata = notebook.metadata["moderacion_peru"]["colab"]
        source = "\n".join(cell.source for cell in notebook.cells)
        if metadata["eligible"]:
            observed.add(metadata["notebook_id"])
            assert "drive.mount" in source
            assert f'COLAB_EXPECTED_CORE_SHA256 = "{expected_core_sha256}"' in source
            assert "bundle_releases" in source
            assert "latest.json" in source
            if metadata["notebook_id"] != "02_00":
                assert "notebook_pinned_release" in source
            assert metadata["build_bundle_id"] == expected_bundle_id
            assert metadata["expected_core_sha256"] == expected_core_sha256
            if metadata["notebook_id"] == "02_00":
                assert "raw.githubusercontent.com" in source
                assert "files.upload()" in source
                assert "RUN_PUBLISH_BUNDLE=" in source
                assert metadata["expected_gpu"] is None
                assert (
                    metadata["transport"] == "github_or_browser_upload_to_google_drive"
                )
                assert metadata["bundle_resolution"] == "publishes_drive_latest_pointer"
                continue
            assert "prepare_colab_context" in source
            assert "publish_colab_outputs" in source
            assert "COLAB_REQUIRE_L4 = True" in source
            assert "COLAB_AUTO_UPDATE_BUNDLE = True" in source
            assert "COLAB_AUTO_PUBLISH_MISSING_BUNDLE = True" in source
            assert 'COLAB_BUNDLE_SOURCE = "github"' in source
            assert 'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"' in source
            assert 'os.environ["HF_HOME"] = "/content/huggingface"' in source
            assert f'COLAB_NOTEBOOK_BUILD_BUNDLE_ID = "{expected_bundle_id}"' in source
            assert "_activate_verified_drive_release" in source
            assert "_ensure_expected_drive_release" in source
            assert "_publish_expected_bundle" in source
            assert "raw.githubusercontent.com" in source.lower()
            assert "urllib.request" in source
            assert "files.upload()" in source
            assert "GITHUB_TOKEN" not in source
            assert 'pip", "install", "-q", "-r"' in source
            assert "git clone" not in source.lower()
            assert (
                metadata["transport"]
                == "github_or_browser_upload_with_drive_versioned_releases"
            )
            assert metadata["bundle_resolution"] == "auto_publish_missing_drive_release"
            if metadata["notebook_id"] == "02_01":
                assert metadata["expected_gpu"] is None
                assert "esta campaña API funciona con runtime CPU" in source
            else:
                assert metadata["expected_gpu"] == "NVIDIA L4"
    assert observed == expected


def test_required_frontends_are_small_templates():
    human = ROOT / "flujo/02_etiquetado/frontend/validacion_humana.html"
    dashboard = ROOT / "flujo/02_etiquetado/frontend/dashboard_etiquetado.html"
    production = ROOT / "flujo/04_produccion/frontend/produccion.html"
    assert human.stat().st_size < 100_000
    assert dashboard.stat().st_size < 100_000
    assert production.stat().st_size < 100_000
    human_source = human.read_text(encoding="utf-8")
    dashboard_source = dashboard.read_text(encoding="utf-8")
    assert "taxonomy.safe_label" in human_source
    assert "event.target.value===taxonomy.safe_label" in human_source
    assert "/api/progress" in human_source
    assert "state.mode==='action'" in human_source
    assert "previous_text" in human_source
    assert "localStorage" in human_source
    assert "display_name" in human_source
    assert "display_name" in production.read_text(encoding="utf-8")
    assert "Modo sombra" in production.read_text(encoding="utf-8")
    production_source = production.read_text(encoding="utf-8")
    assert "/api/analyze" in production_source
    assert "/api/stats" in production_source
    assert "youtube.com" in production_source
    assert "Consenso 2-de-3" in production_source
    assert "Comparar familias" in production_source
    assert "Dataset para reentrenar" in production_source
    assert "event.key.toLowerCase()==='r'" in human_source
    assert "action==='defer'" in human_source
    assert "updateFlagAvailability" in human_source
    assert "Diferido · requiere acción" in human_source
    assert "Resuelto · decisión automática" in human_source
    assert "Todos los chunks" in human_source
    assert "Sugerencia automática" in human_source
    assert "revealBtn" not in human_source
    assert '<dialog id="contextDialog"' in human_source
    assert '<dialog id="proposalDialog"' in human_source
    assert '<dialog id="bulkDialog"' in human_source
    assert "/api/review-scope" in human_source
    assert "/api/review/bulk" in human_source
    assert "bulkIncludeResolved" in human_source
    assert "Acción masiva por video o canal" in human_source
    assert "Clasificar todo igual" in human_source
    assert 'id="bulkLabels"' in human_source
    assert 'id="bulkFlags"' in human_source
    assert "final_labels:finalLabels" in human_source
    assert "Guardar mi decisión" in human_source
    assert "Revisar después" in human_source
    assert "Excluir del dataset" in human_source
    assert '<dialog id="actionsDialog"' in human_source
    assert "¿Qué hace cada botón?" in human_source
    assert "Prioritarios Pro" in human_source
    assert "Urgentes" in human_source
    assert "Requieren acción" in human_source
    assert 'data-mode="action"' in human_source
    assert 'data-mode="all"' in human_source
    assert '<dialog id="filterDialog"' in human_source
    assert "Acerca de" in human_source
    assert "@media(max-width:1420px)" in human_source
    assert "@media(max-width:720px)" in human_source
    assert ".top-controls{order:2;display:grid" in human_source
    assert 'id="intermediateCount"' in human_source
    assert "needs_review Pro intermedio" in human_source
    assert "Pro pendientes finales" in human_source
    assert 'href="/dashboard"' in human_source
    assert 'fetch("/api/dashboard"' in dashboard_source
    assert 'href="/"' in dashboard_source
    assert "setInterval" in dashboard_source
    assert "Calidad de la auditoría" in dashboard_source
    assert "Confianza y calibración" in dashboard_source
    assert "Lectura cualitativa" in dashboard_source


def test_02_01_uses_explicit_operational_review_threshold_without_changing_calibration():
    notebook = nbformat.read(
        ROOT / "flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb",
        as_version=4,
    )
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    generator = (ROOT / "tools/generate_workflow_notebooks.py").read_text(
        encoding="utf-8"
    )
    for value in (source, generator):
        assert "REVIEW_CONFIDENCE_THRESHOLD=0.85" in value
        assert "SAFE_CONTROL_RATE=0.01" in value
        assert "MAX_NEEDS_REVIEW_FOR_PRO=36_000" in value
        assert "MAX_REVIEW_COST_USD=None" in value
        assert "RECOMMENDED_REVIEW_START_BALANCE_USD=15.00" in value
        assert "por encima del tope" not in value
        assert "esto no bloquea la ejecución" in value
        assert 'raise RuntimeError(f"Saldo Pro' not in value
        assert "RUN_PRIMARY=True" in value
        assert "RUN_DIRECTED_REVIEW=True" in value
        assert "confidence_threshold=REVIEW_CONFIDENCE_THRESHOLD" in value
        assert "max_needs_review=MAX_NEEDS_REVIEW_FOR_PRO" in value
        assert (
            "confidence_threshold=float(calibration['selected_threshold'])" not in value
        )


def test_latest_operational_prompt_is_used_and_covers_peruvian_edge_cases():
    prompt_path = ROOT / "config/prompt_operacional_ollama_v3_2.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    for expression in (
        "hijo de puta",
        "concha de tu madre",
        "cholo de mierda",
        "maricón de mierda",
        "maldito perro",
        "Regla semántica peruana para *poto*",
        "0.85–0.94",
        "Nunca devuelvas `fine_labels` ni `coarse_labels` vacíos",
    ):
        assert expression in prompt

    active = (
        "flujo/01_datos/01_01_scraping_incremental.ipynb",
        "flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb",
        "flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb",
        "flujo/02_etiquetado/02_02_etiquetado_hf_qwen_colab.ipynb",
    )
    for relative in active:
        notebook = nbformat.read(ROOT / relative, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "config/prompt_operacional_ollama_v3_2.md" in source

    qwen = nbformat.read(
        ROOT / "flujo/02_etiquetado/02_02_etiquetado_hf_qwen_colab.ipynb",
        as_version=4,
    )
    qwen_source = "\n".join(cell.source for cell in qwen.cells)
    assert "Qwen/Qwen3-1.7B" in qwen_source
    assert "Qwen/Qwen3-4B" in qwen_source
    assert "root_key']!='annotations'" in qwen_source


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
        if "archivo" not in path.parts
        and "modelos" not in path.parts
        and ".venv" not in path.parts
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
    ]
    frontends = sorted((ROOT / "flujo").rglob("*.html"))
    for path in readmes + frontends:
        source = path.read_text(encoding="utf-8-sig")
        assert all(item in source for item in (title, course, *authors)), path
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        assert all(
            item in notebook.cells[0].source for item in (title, course, *authors)
        ), path
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
        "01_datos": 4,
        "02_etiquetado": 6,
        "03_entrenamiento": 9,
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
    assert not (
        ROOT / "flujo" / "01_datos" / "01_03_ampliacion_dirigida.ipynb"
    ).exists()
    assert (
        ROOT
        / "archivo"
        / "flujo_reorganizado_v2"
        / "01_03_ampliacion_dirigida_reemplazado.ipynb"
    ).is_file()


def test_minority_scraping_notebook_targets_two_thousand_total_chunks_per_damage():
    path = ROOT / "flujo/01_datos/01_015_ampliacion_dirigida_minorias.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "TARGET_TOTAL_CHUNKS_PER_DAMAGE = 2_000" in source
    assert "TARGET_ELIGIBLE_SPLITS = ('train', 'validation', 'test')" in source
    assert "PERU_ONLY = True" in source
    assert "filter_peru_candidates" in source
    assert "directed_non_peru_excluded_latest.jsonl" in source
    assert '"country_scope": "PE_only_strict"' in source
    assert "estimate_directed_video_budget" in source
    assert "DIRECTED_SAFETY_FACTOR = 1.5" in source
    assert "DIRECTED_YIELD_DISCOUNT = 0.5" in source
    assert "canales_núcleo" in source
    assert "canales_margen" in source
    assert (
        "DIRECTED_SPLIT_BUDGET = DIRECTED_ACQUISITION_BUDGET['split_budget']" in source
    )
    assert "directed_split_shortfall" in source
    assert "No se encontraron suficientes videos PE verificados" not in source
    assert "DIRECTED_RESERVE_CHANNELS" in source
    assert "canales_PE_de_reserva" in source
    assert "directed_selection_incomplete" in source
    assert "refilled_on_resume" in source
    assert "Cohorte PE parcial: se continuará sin perder avances" in source
    assert "directed_candidates_carryover.jsonl" in source
    assert "write_jsonl_atomic(DIRECTED_CARRYOVER_PATH, directed_carryover)" in source
    assert "merge_candidates(active_candidates, carryover_candidates)" in source
    assert "cohorte_dirigida_vigente_más_arrastre" in source
    assert "RESUME_DISCOVERY = True" in source
    assert "directed_round_signature" in source
    assert "resumed_existing_round" in source
    assert "previous_round_selection" in source
    assert "pending_in_resumed_round" in source
    signature_block = source.split("directed_round_signature =", 1)[1].split(
        "previous_plan_payload", 1
    )[0]
    manifest_block = source.split("write_json_atomic(DIRECTED_PLAN_PATH", 1)[1]
    assert "acquisition_budget" not in signature_block
    assert "'directed_round_signature': directed_round_signature" in manifest_block
    assert "'pending_in_resumed_round': len(directed_selection)" in manifest_block
    assert "cached_transcript" not in source
    assert "reused_cache" in source
    assert "project_effective_training_rows" in source
    assert "eligible_splits=TARGET_ELIGIBLE_SPLITS" in source
    assert "target_chunks_per_label=TARGET_TOTAL_CHUNKS_PER_DAMAGE" in source
    assert "required_split=planned_split" in source
    assert "split_seed=SPLIT_SEED" in source
    assert "de Pro es un estado intermedio" in source
    assert "Hablando Huevadas" in source
    assert "Goblinciano" in source
    assert "Nunca MAS" in source
    assert "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD" in source

    carryover = [
        json.loads(line)
        for line in (ROOT / "datos/raw/directed_candidates_carryover.jsonl")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    assert len(carryover) == 5
    assert len({row["video_id"] for row in carryover}) == 5
    assert all(row["country_code"] == "PE" for row in carryover)
    assert all(row["peru_scope_verified"] is True for row in carryover)


def test_chunk_length_pilot_is_optional_and_materialization_is_separate():
    pilot_path = (
        ROOT / "flujo" / "01_datos" / "01_02_optimizacion_longitud_chunks.ipynb"
    )
    materialization_path = (
        ROOT / "flujo" / "01_datos" / "01_03_limpieza_troceado_incremental.ipynb"
    )
    assert pilot_path.is_file()
    assert materialization_path.is_file()
    assert not (
        ROOT / "flujo" / "01_datos" / "01_02_limpieza_troceado_incremental.ipynb"
    ).exists()
    pilot = nbformat.read(pilot_path, as_version=4)
    source = "\n".join(cell.source for cell in pilot.cells)
    for cell in pilot.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for call in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "show_table"
        ):
            assert all(keyword.arg != "limit" for keyword in call.keywords)
    run_controls: dict[str, bool] = {}
    for cell in pilot.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id.startswith("RUN_")
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, bool)
            ):
                run_controls[target.id] = node.value.value
    expected_run_controls = {
        "RUN_CHUNK_LENGTH_SMOKE_TEST",
        "RUN_CHUNK_LENGTH_CONFIRMATORY_TEST",
        "RUN_CHUNK_LENGTH_ROBUST_TEST",
        "RUN_NEURAL_ROBUST_TEST",
    }
    assert expected_run_controls <= run_controls.keys()
    assert not (
        run_controls["RUN_CHUNK_LENGTH_CONFIRMATORY_TEST"]
        and run_controls["RUN_CHUNK_LENGTH_ROBUST_TEST"]
    )
    assert run_controls["RUN_NEURAL_ROBUST_TEST"]
    assert "RUN_BOUNDED_HF_COMPARISON" not in source
    assert "RUN_BOUNDED_OLLAMA_COMPARISON" not in source
    assert "NEURAL_CANDIDATE_SECONDS" not in source
    for control in (
        "CANDIDATE_SECONDS=(15,20,25,30,35)",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "NEURAL_OLLAMA_MODEL='gemma3:4b'",
        "NEURAL_PANEL_SIZE=100",
        "NEURAL_REPORTING_COHORTS=5",
        "NEURAL_MINILM_TRAIN_LIMIT=1000",
        "NEURAL_MINILM_BOOTSTRAP_REPLICATES=2000",
        "NEURAL_OLLAMA_BOOTSTRAP_REPLICATES=2000",
        "NEURAL_OLLAMA_MAX_WALL_SECONDS=5400.0",
        "NEURAL_OLLAMA_MINIMUM_SCHEMA_RATE=0.95",
        "ROBUST_VIDEO_LIMITS={'train':300,'validation':100,'test':100}",
        "ROBUST_SEEDS=(20260805,20260817,20260829,20260841,20260853)",
        "ROBUST_BOOTSTRAP_REPLICATES=1000",
        "ROBUST_REFERENCE_SECONDS=30.0",
        "ROBUST_NONINFERIORITY_MARGIN=0.01",
        "USE_ROBUST_RECOMMENDATION=True",
        "CONFIRMATORY_VIDEO_LIMITS={'train':200,'validation':80,'test':80}",
        "CONFIRMATORY_SEEDS=(20260805,20260817,20260829)",
        "MANUAL_CHUNK_SECONDS=30.0",
        "APPLY_CHUNK_SELECTION=False",
        "complement_nb",
        "sgd_incremental",
        "run_neural_chunk_robust_test",
        "run_chunk_length_robust_test",
        "Perfil clásico: resultados reportables",
        "MiniLM robusto: resultados reportables",
        "Ollama robusto: resultados reportables",
        "Síntesis jerárquica",
        "Roles no intercambiables",
        "500 respuestas Ollama",
        "prepare_local_bundle_input('chunks_v2'",
        "prepare_local_bundle_input('dataset_5_salidas'",
    ):
        assert control in source
    assert "test_used_for_selection" not in source
    materialization = nbformat.read(materialization_path, as_version=4)
    materialization_source = "\n".join(cell.source for cell in materialization.cells)
    assert "activate_chunking_configuration" in materialization_source
    assert "**CHUNK_CONFIG" in materialization_source
    assert "REBUILD_CHUNKS_FROM_ZERO=False" in materialization_source
    assert "materialize_chunk_records" in materialization_source
    assert "consolidate_available_transcripts" in materialization_source
    assert "materialize_transcripts_by_channel" in materialization_source
    assert "from tqdm.auto import tqdm" in materialization_source
    assert "report_chunk_progress" in materialization_source


def test_dataset_consumers_restore_and_verify_the_synced_checkpoint():
    for path in sorted((ROOT / "flujo" / "03_entrenamiento").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "prepare_local_bundle_input('dataset_5_salidas'" in source, path
        assert "Dataset descomprimido y verificado" in source, path


def test_local_training_notebooks_expose_bounded_deterministic_parallelism():
    classical = nbformat.read(
        ROOT / "flujo/03_entrenamiento/03_01_modelos_clasicos.ipynb", as_version=4
    )
    comparison = nbformat.read(
        ROOT / "flujo/03_entrenamiento/03_07_comparacion_final.ipynb", as_version=4
    )
    classical_source = "\n".join(cell.source for cell in classical.cells)
    comparison_source = "\n".join(cell.source for cell in comparison.cells)

    assert "PARALLEL_WORKERS=4" in classical_source
    assert "parallel_workers=PARALLEL_WORKERS" in classical_source
    assert "una extracción por variante" in classical_source
    assert "LINEAR_SVM_MAX_ITER=20000" in classical_source
    assert "RUN_SVM_CONVERGENCE_REPAIR=False" in classical_source
    assert "svm_convergence_repair" in classical_source
    assert "PARALLEL_WORKERS=4" in comparison_source
    assert "parallel_workers=PARALLEL_WORKERS" in comparison_source
    assert "inferencia_miembros':'secuencial" in comparison_source


def test_active_long_running_notebooks_expose_progress_indicators():
    chunk_optimization = nbformat.read(
        ROOT / "flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb",
        as_version=4,
    )
    chunk_source = "\n".join(cell.source for cell in chunk_optimization.cells)
    assert "run_with_progress" in chunk_source

    for path in sorted((ROOT / "flujo/03_entrenamiento").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "run_with_progress" in source, path

    classical = nbformat.read(
        ROOT / "flujo/03_entrenamiento/03_01_modelos_clasicos.ipynb", as_version=4
    )
    classical_source = "\n".join(cell.source for cell in classical.cells)
    assert "progress_unit='etapa'" in classical_source
    experiments_source = (ROOT / "src/moderacion_peru/experiments.py").read_text(
        encoding="utf-8"
    )
    assert "progress_callback: ProgressCallback | None = None" in experiments_source


def test_stage_02_notebooks_show_progress_for_long_operations():
    notebook_sources = {}
    for path in sorted((ROOT / "flujo" / "02_etiquetado").glob("02_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        notebook_sources[path.name] = "\n".join(cell.source for cell in notebook.cells)
        assert "from tqdm.auto import tqdm" in notebook_sources[path.name]

    bundle = notebook_sources["02_00_preparacion_bundle_colab.ipynb"]
    assert "RUN_PUBLISH_BUNDLE=" in bundle
    generator = (ROOT / "tools" / "generate_workflow_notebooks.py").read_text(
        encoding="utf-8"
    )
    assert '"RUN_PUBLISH_BUNDLE=False' in generator
    assert "BUNDLE_SOURCE='github'" in bundle
    assert "local_upload" in bundle
    assert "raw.githubusercontent.com" in bundle
    assert "files.upload()" in bundle
    assert "drive.mount('/content/drive'" in bundle
    assert "_verify_bundle" in bundle
    assert "bundle_releases" in bundle
    assert "latest.json" in bundle
    assert "published_to_drive" in bundle
    colab_config = json.loads(
        (ROOT / "config/colab_l4.json").read_text(encoding="utf-8")
    )
    assert all(
        specification["archive"] in bundle
        for specification in colab_config["inputs"].values()
    )
    assert "Seleccione nueve archivos" in bundle
    local = notebook_sources["02_01_etiquetado_deepseek_flash_pro.ipynb"]
    assert "PRIMARY_LIMIT=None" in local
    assert "REVIEW_LIMIT=None" in local
    assert "None para TODOS y solo los pendientes" in local
    assert "Nunca lo deje en blanco" in local
    assert "progress_callback=labeling_progress" in local
    assert "from tqdm.std import tqdm" in local
    assert "Salida textual visible también desde VS Code" in local
    assert "validate_connection()" in local
    assert "expected_thinking={'type':'disabled'}" in local
    assert "expected_response_format={'type':'json_object'}" in local
    assert "02_01 exige DeepSeek V4 en modo non-thinking para Flash y Pro" in local
    assert "02_01 exige JSON object con raíz annotations para Flash y Pro" in local
    assert "automatic_prefix" in local
    assert "prompt_cache_hit_tokens" in local
    assert "balance_summary()" in local
    assert (
        "Falta DEEPSEEK_API_KEY: configúrela como variable local o secreto privado de Colab antes de etiquetar"
        in local
    )
    assert "BALANCE_REFRESH_SECONDS=60.0" in local
    assert "Saldo DeepSeek bajo" in local
    assert "'thinking':probe['thinking']" in local
    assert "'context_cache':probe['context_cache']" in local
    assert "recover_historical_annotations" in local
    assert (
        "PRESERVE_PROMPT_POLICY_SHA256='433321cf7b41f997bb277ae87bc9fee01767d225a0fe49bea2cb918239dc1f06'"
        in local
    )
    assert "RECOVER_HISTORICAL=True" in local
    assert "AUTO_PUBLISH_CHECKPOINTS=True" in local
    assert "checkpoint_callback=checkpoint_callback_for(output)" in local
    assert "interrupted_checkpoint" in local
    assert "PROCESSING_BATCH_SIZE=640" in local
    assert "max_workers=128,records_per_request=5,cache_warmup_requests=1" in local
    assert "max_workers=64,records_per_request=5,cache_warmup_requests=1" in local
    assert "estimated_cost_usd" in local
    assert "pending_current_after_recovery" in local
    assert "quarantine_invalid_progress=True" in local
    assert (
        "progress_callback=qwen_progress"
        in notebook_sources["02_02_etiquetado_hf_qwen_colab.ipynb"]
    )
    assert "Leyendo {name}" in notebook_sources["02_03_revision_llm_dirigida.ipynb"]
    assert (
        "primary_flash_v3_2.jsonl"
        in notebook_sources["02_03_revision_llm_dirigida.ipynb"]
    )
    assert (
        "review_pro_v3_2.jsonl" in notebook_sources["02_03_revision_llm_dirigida.ipynb"]
    )
    assert (
        "progress_callback=report_consolidation_progress"
        in notebook_sources["02_04_consolidacion_validacion_humana.ipynb"]
    )
    assert (
        "primary_flash_v3_2.jsonl"
        in notebook_sources["02_04_consolidacion_validacion_humana.ipynb"]
    )
    assert (
        "review_pro_v3_2.jsonl"
        in notebook_sources["02_04_consolidacion_validacion_humana.ipynb"]
    )
    closure = notebook_sources["02_05_cierre_humano_snapshot.ipynb"]
    assert "progress_callback=report_stage_progress" in closure
    assert "Deduplicando snapshot" in closure


def test_synced_checkpoint_rules_preserve_hashes_and_exclude_rebuildable_working_files():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "!datos/raw/transcripts_by_channel/**" in ignore
    assert "!datos/raw/vtt_by_video/**" in ignore
    assert "!datos/raw/video_candidates.jsonl" in ignore
    assert "!resultados/colab_bundle/dataset_5_salidas.jsonl.gz" in ignore
    config = json.loads((ROOT / "config/colab_l4.json").read_text(encoding="utf-8"))
    assert all(
        f"!resultados/colab_bundle/{specification['archive']}" in ignore
        for specification in config["inputs"].values()
    )
    assert "datos/raw/transcripts_raw.jsonl" in ignore
    assert "datos/raw/transcripts_cache/" in ignore
    assert "datos/processed/*" in ignore
    assert "datos/model_ready/v2/*" in ignore
    assert "archivo/chunk_rebuilds/" in ignore
    assert "*.jsonl text eol=lf" in attributes
    assert "*.gz binary" in attributes


def test_chunk_materialization_report_matches_the_tracked_manifest():
    manifest_path = ROOT / "datos/processed/chunk_materialization_manifest.json"
    report_path = ROOT / "docs/MATERIALIZACION_TROCEADO.md"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    report = report_path.read_text(encoding="utf-8-sig")
    chunk_rows = int(manifest["outputs"]["chunks"]["rows"])
    transcript_videos = int(manifest["coverage"]["transcript_videos"])
    formatted_rows = f"{chunk_rows:,}".replace(",", ".")
    formatted_videos = f"{transcript_videos:,}".replace(",", ".")
    assert formatted_rows in report
    assert formatted_videos in report
    assert manifest["outputs"]["chunks"]["sha256"] in report
    assert "!datos/processed/chunk_materialization_manifest.json" in (
        ROOT / ".gitignore"
    ).read_text(encoding="utf-8")


def test_neural_model_revisions_are_pinned():
    from moderacion_peru.models import TRANSFORMER_SPECS

    assert all(
        spec.revision and len(spec.revision) == 40
        for spec in TRANSFORMER_SPECS.values()
    )


def test_lm_studio_only_exists_in_archive():
    active_roots = [ROOT / "src", ROOT / "flujo"]
    active_files = [
        path for root in active_roots for path in root.rglob("*") if path.is_file()
    ]
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
