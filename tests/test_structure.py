from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def test_active_notebooks_are_ordered_and_clean():
    notebooks = sorted((ROOT / "flujo").rglob("*.ipynb"))
    assert len(notebooks) == 16
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        assert notebook.metadata["moderacion_peru"]["contract"] == "moderacion_peru_5_salidas_v2"
        assert notebook.metadata["moderacion_peru"]["taxonomy_version"] == "2.1.0"
        assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "LM Studio" not in source
        assert "D:\\" not in source
        assert "G:\\" not in source


def test_required_frontends_are_small_templates():
    human = ROOT / "flujo/02_etiquetado/frontend/validacion_humana.html"
    production = ROOT / "flujo/04_produccion/frontend/produccion.html"
    assert human.stat().st_size < 100_000
    assert production.stat().st_size < 100_000
    human_source = human.read_text(encoding="utf-8")
    assert "t.safe_label" in human_source
    assert "x.value!==t.safe_label" in human_source
    assert "display_name" in human_source
    assert "display_name" in production.read_text(encoding="utf-8")
    assert "Modo sombra" in production.read_text(encoding="utf-8")


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
