"""Añade una advertencia uniforme sin modificar el significado histórico."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNER = (
    "> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a "
    "`moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no "
    "definen el flujo activo. Consulte el README raíz y `archivo/README.md`.\n\n"
)


def main() -> None:
    changed = 0
    for path in sorted((ROOT / "archivo").rglob("*.md")):
        if path == ROOT / "archivo" / "README.md":
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "Documento histórico preservado" in text[:500]:
            continue
        lines = text.splitlines(keepends=True)
        if lines and lines[0].lstrip().startswith("#"):
            updated = lines[0] + "\n" + BANNER + "".join(lines[1:])
        else:
            updated = BANNER + text
        path.write_text(updated, encoding="utf-8", newline="\n")
        changed += 1
    print(f"Markdown históricos marcados: {changed}")


if __name__ == "__main__":
    main()

