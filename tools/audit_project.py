"""Auditoría estática de rutas, Markdown y cuadernos activos."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
DEPRECATED_GENDER_LABEL = "ACOSO_GENERO_IDENTIDAD"


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
    expected_prefixes = {
        "01_datos": ("01_01", "01_02", "01_03"),
        "02_etiquetado": ("02_01", "02_02", "02_03", "02_04"),
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
        if notebook.metadata.get("moderacion_peru", {}).get("contract") != "moderacion_peru_5_salidas_v2":
            issues.append(f"missing_contract_metadata:{path.relative_to(ROOT)}")
        if notebook.metadata.get("moderacion_peru", {}).get("taxonomy_version") != "2.1.0":
            issues.append(f"wrong_taxonomy_version:{path.relative_to(ROOT)}")
        source = "\n".join(cell.source for cell in notebook.cells)
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


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.taxonomy import load_taxonomy

    taxonomy = load_taxonomy()
    issues = markdown_issues() + notebook_issues() + taxonomy_name_issues()
    result = {
        "taxonomy": taxonomy.contract_id,
        "markdown_files": len(list(ROOT.rglob("*.md"))),
        "active_notebooks": len(list((ROOT / "flujo").rglob("*.ipynb"))),
        "issues": issues,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
