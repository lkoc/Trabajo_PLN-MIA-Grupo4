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
                ast.parse(cell.source)


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
