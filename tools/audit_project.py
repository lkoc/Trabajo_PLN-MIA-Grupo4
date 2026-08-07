"""Auditoría estática de rutas, Markdown y cuadernos activos."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
BODY_CITATION = re.compile(r"\[(\d+)\]")
REFERENCE_ENTRY = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)
MASTER_BIB_ENTRY = re.compile(r"^@[A-Za-z]+\{([^,]+),", re.MULTILINE)
DEPRECATED_GENDER_LABEL = "ACOSO_GENERO_IDENTIDAD"
PROJECT_TITLE = (
    "Moderación semiautomática de videos peruanos de YouTube mediante modelos "
    "clásicos y neuronales de procesamiento del lenguaje natural"
)
COURSE_LINE = (
    "Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en "
    "Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1"
)
PROJECT_AUTHORS = (
    "Luis Enrique Koc Góngora",
    "Alex Felipe Mancilla Antay",
    "Herbert Antonio Meléndez García",
    "Dennis Jack Paitán Cano",
)


def markdown_issues() -> list[str]:
    issues = []
    for path in ROOT.rglob("*.md"):
        # El archivo es evidencia inmutable: sus enlaces relativos describen la
        # topología de origen y no se interpretan como enlaces activos.
        if ".git" in path.parts or "archivo" in path.parts:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for raw in MARKDOWN_LINK.findall(text):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                issues.append(f"broken_link:{path.relative_to(ROOT)}->{target}")
    return issues


def notebook_issues() -> list[str]:
    issues = []
    notebooks = sorted((ROOT / "flujo").rglob("*.ipynb"))
    master_bib = (ROOT / "Documento_final_paper" / "referencias.bib").read_text(
        encoding="utf-8"
    )
    master_keys = set(MASTER_BIB_ENTRY.findall(master_bib))
    expected_prefixes = {
        "01_datos": ("01_01", "01_02", "01_03"),
        "02_etiquetado": ("02_01", "02_02", "02_03", "02_04", "02_05"),
        "03_entrenamiento": tuple(f"03_{i:02d}" for i in range(1, 9)),
        "04_produccion": ("04_01",),
    }
    for folder, prefixes in expected_prefixes.items():
        names = [path.name for path in notebooks if folder in path.parts]
        if len(names) != len(prefixes):
            issues.append(f"notebook_count:{folder}:{len(names)}")
        for prefix in prefixes:
            if not any(name.startswith(prefix) for name in names):
                issues.append(f"missing_notebook:{folder}:{prefix}")
    for path in notebooks:
        try:
            notebook = nbformat.read(path, as_version=4)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"invalid_notebook:{path.relative_to(ROOT)}:{exc}")
            continue
        if (
            notebook.metadata.get("moderacion_peru", {}).get("contract")
            != "moderacion_peru_5_salidas_v2"
        ):
            issues.append(f"missing_contract_metadata:{path.relative_to(ROOT)}")
        if (
            notebook.metadata.get("moderacion_peru", {}).get("taxonomy_version")
            != "2.1.0"
        ):
            issues.append(f"wrong_taxonomy_version:{path.relative_to(ROOT)}")
        citation_metadata = notebook.metadata.get("moderacion_peru", {})
        citation_keys = list(citation_metadata.get("citation_keys", []))
        if citation_metadata.get("citation_style") != "IEEE_numeric":
            issues.append(f"wrong_citation_style:{path.relative_to(ROOT)}")
        if not citation_keys or len(citation_keys) != len(set(citation_keys)):
            issues.append(f"invalid_citation_keys:{path.relative_to(ROOT)}")
        if citation_metadata.get("reference_count") != len(citation_keys):
            issues.append(f"wrong_reference_count:{path.relative_to(ROOT)}")
        for missing_key in sorted(set(citation_keys) - master_keys):
            issues.append(
                f"citation_missing_master_bib:{path.relative_to(ROOT)}:{missing_key}"
            )
        if not notebook.cells or notebook.cells[-1].cell_type != "markdown":
            issues.append(f"missing_final_references:{path.relative_to(ROOT)}")
        else:
            final = notebook.cells[-1]
            if not final.source.startswith("## Referencias\n\n[1] "):
                issues.append(f"invalid_final_references:{path.relative_to(ROOT)}")
            if "references" not in final.metadata.get("tags", []):
                issues.append(f"missing_references_tag:{path.relative_to(ROOT)}")
            body = "\n".join(
                cell.source
                for cell in notebook.cells[:-1]
                if cell.cell_type == "markdown"
            )
            expected = set(range(1, len(citation_keys) + 1))
            observed_citations = set(map(int, BODY_CITATION.findall(body)))
            observed_entries = list(map(int, REFERENCE_ENTRY.findall(final.source)))
            if observed_citations != expected:
                issues.append(f"citation_number_mismatch:{path.relative_to(ROOT)}")
            if observed_entries != list(range(1, len(citation_keys) + 1)):
                issues.append(f"reference_number_mismatch:{path.relative_to(ROOT)}")
        source = "\n".join(cell.source for cell in notebook.cells)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                try:
                    ast.parse(cell.source)
                except SyntaxError as exc:
                    issues.append(
                        f"invalid_code_cell:{path.relative_to(ROOT)}:{index}:{exc}"
                    )
        colab = notebook.metadata.get("moderacion_peru", {}).get("colab", {})
        if colab.get("eligible"):
            for required in (
                "drive.mount",
                "prepare_colab_context",
                "publish_colab_outputs",
                "COLAB_REQUIRE_L4",
            ):
                if required not in source:
                    issues.append(
                        f"missing_colab_bootstrap:{path.relative_to(ROOT)}:{required}"
                    )
            if "git clone" in source.lower():
                issues.append(f"colab_uses_git:{path.relative_to(ROOT)}")
        for forbidden in ("LM Studio", "D:\\", "G:\\"):
            if forbidden in source:
                issues.append(f"forbidden:{path.relative_to(ROOT)}:{forbidden}")
    return issues


def taxonomy_name_issues() -> list[str]:
    """Impide que el alias histórico vuelva a convertirse en salida activa."""
    issues = []
    active_roots = (
        ROOT / "README.md",
        ROOT / "Planning",
        ROOT / "docs",
        ROOT / "flujo",
        ROOT / "Documento_final_paper",
        ROOT / "Presentación_BEAMER",
    )
    for root in active_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".jpg"}:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            if (
                DEPRECATED_GENDER_LABEL in text
                and path != ROOT / "docs" / "MIGRACION_Y_COMPATIBILIDAD.md"
            ):
                issues.append(f"deprecated_gender_label:{path.relative_to(ROOT)}")
    return issues


def branding_issues() -> list[str]:
    """Comprueba la carátula académica en todos los puntos de entrada activos."""
    issues = []
    readmes = [
        path
        for path in ROOT.rglob("README.md")
        if "archivo" not in path.parts
        and "modelos" not in path.parts
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
    ]
    frontends = sorted((ROOT / "flujo").rglob("*.html"))
    for path in readmes + frontends:
        text = path.read_text(encoding="utf-8-sig")
        for required in (PROJECT_TITLE, COURSE_LINE, *PROJECT_AUTHORS):
            if required not in text:
                issues.append(
                    f"missing_academic_cover:{path.relative_to(ROOT)}:{required}"
                )
    for path in sorted((ROOT / "flujo").rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        first = (
            notebook.cells[0].source
            if notebook.cells and notebook.cells[0].cell_type == "markdown"
            else ""
        )
        for required in (PROJECT_TITLE, COURSE_LINE, *PROJECT_AUTHORS):
            if required not in first:
                issues.append(
                    f"missing_notebook_cover:{path.relative_to(ROOT)}:{required}"
                )
        metadata = notebook.metadata.get("moderacion_peru", {})
        if metadata.get("project_title") != PROJECT_TITLE:
            issues.append(f"missing_project_title_metadata:{path.relative_to(ROOT)}")
        if (
            metadata.get("academic_term") != "2026-1"
            or metadata.get("group") != "Grupo 4"
        ):
            issues.append(f"wrong_academic_metadata:{path.relative_to(ROOT)}")
    return issues


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    issues = (
        markdown_issues()
        + notebook_issues()
        + taxonomy_name_issues()
        + branding_issues()
    )
    notebooks = [
        nbformat.read(path, as_version=4)
        for path in sorted((ROOT / "flujo").rglob("*.ipynb"))
    ]
    result = {
        "taxonomy": taxonomy.contract_id,
        "markdown_files": len(list(ROOT.rglob("*.md"))),
        "active_notebooks": len(list((ROOT / "flujo").rglob("*.ipynb"))),
        "notebook_citations": sum(
            int(notebook.metadata.get("moderacion_peru", {}).get("reference_count", 0))
            for notebook in notebooks
        ),
        "notebooks_with_final_references": sum(
            bool(notebook.cells)
            and notebook.cells[-1].cell_type == "markdown"
            and notebook.cells[-1].source.startswith("## Referencias")
            for notebook in notebooks
        ),
        "issues": issues,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
