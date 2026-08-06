from __future__ import annotations

import dataclasses
import html
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_PALETTE = {
    "info": ("#075985", "#e0f2fe", "#7dd3fc"),
    "success": ("#166534", "#dcfce7", "#86efac"),
    "warning": ("#92400e", "#fef3c7", "#fcd34d"),
    "error": ("#991b1b", "#fee2e2", "#fca5a5"),
    "neutral": ("#334155", "#f8fafc", "#cbd5e1"),
}


def _normalise(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise(item) for item in value]
    return value


def _text(value: Any) -> str:
    value = _normalise(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Sí" if value else "No"
    return str(value)


def _display(markup: str, fallback: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if get_ipython() is None:
            raise RuntimeError("No hay una sesión IPython activa")
        display(HTML(markup))
    except (ImportError, RuntimeError):
        sys.stdout.write(fallback.rstrip() + "\n")


def _card(title: str, body: str, *, tone: str) -> str:
    foreground, background, border = _PALETTE.get(tone, _PALETTE["info"])
    return (
        f'<section style="border:1px solid {border};border-left:5px solid {foreground};'
        f'background:{background};border-radius:8px;padding:12px 14px;margin:8px 0 14px 0;'
        'font-family:Inter,Segoe UI,Arial,sans-serif;color:#0f172a">'
        f'<div style="font-weight:700;color:{foreground};margin-bottom:8px">'
        f'{html.escape(str(title))}</div>{body}</section>'
    )


def show_callout(title: str, message: Any, *, tone: str = "info") -> None:
    """Muestra un estado o instrucción breve como tarjeta accesible."""

    rendered = html.escape(_text(message)).replace("\n", "<br>")
    _display(_card(title, f'<div style="line-height:1.45">{rendered}</div>', tone=tone), f"{title}: {_text(message)}")


def show_summary(title: str, values: Mapping[str, Any], *, tone: str = "info") -> None:
    """Presenta un mapeo como tabla clave–valor compacta."""

    normalised = _normalise(values)
    rows = []
    for key, value in normalised.items():
        rendered = html.escape(_text(value))
        if "\n" in _text(value) or len(_text(value)) > 120:
            rendered = (
                '<details><summary style="cursor:pointer">Ver detalle</summary>'
                f'<pre style="white-space:pre-wrap;margin:6px 0 0">{rendered}</pre></details>'
            )
        rows.append(
            '<tr>'
            f'<th style="text-align:left;padding:5px 12px 5px 0;vertical-align:top;color:#475569">{html.escape(str(key))}</th>'
            f'<td style="padding:5px 0;vertical-align:top;font-family:ui-monospace,Consolas,monospace">{rendered}</td>'
            '</tr>'
        )
    body = '<table style="border-collapse:collapse;width:100%">' + "".join(rows) + "</table>"
    fallback = f"{title}\n" + "\n".join(f"- {key}: {_text(value)}" for key, value in normalised.items())
    _display(_card(title, body, tone=tone), fallback)


def show_table(
    title: str,
    records: Sequence[Mapping[str, Any]],
    *,
    max_rows: int = 20,
    tone: str = "neutral",
) -> None:
    """Muestra registros homogéneos y limita la vista sin truncar el artefacto."""

    rows = [_normalise(row) for row in records]
    if not rows:
        show_callout(title, "Sin filas para mostrar.", tone="neutral")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    head = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:1px solid #cbd5e1">{html.escape(str(column))}</th>'
        for column in columns
    )
    body_rows = []
    for row in rows[:max_rows]:
        body_rows.append(
            "<tr>"
            + "".join(
                f'<td style="padding:6px 10px;border-bottom:1px solid #e2e8f0;vertical-align:top">'
                f'{html.escape(_text(row.get(column)))}</td>'
                for column in columns
            )
            + "</tr>"
        )
    note = (
        f'<div style="margin-top:8px;color:#64748b">Se muestran {min(len(rows), max_rows)} de {len(rows)} filas.</div>'
        if len(rows) > max_rows
        else ""
    )
    markup = '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">' + f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>{note}"
    _display(_card(title, markup, tone=tone), f"{title}: {len(rows)} filas")


def show_result(title: str, value: Any, *, tone: str = "info") -> None:
    """Elige tarjeta, resumen o tabla según la forma del resultado."""

    normalised = _normalise(value)
    if isinstance(normalised, Mapping):
        show_summary(title, normalised, tone=tone)
    elif (
        isinstance(normalised, Sequence)
        and not isinstance(normalised, (str, bytes, bytearray))
        and normalised
        and all(isinstance(item, Mapping) for item in normalised)
    ):
        show_table(title, normalised, tone=tone)
    else:
        show_callout(title, normalised, tone=tone)


def show_command(title: str, command: str, *, description: str | None = None) -> None:
    """Presenta un comando copiable sin confundirlo con una ejecución realizada."""

    explanation = (
        f'<div style="margin-bottom:8px;line-height:1.4">{html.escape(description)}</div>'
        if description
        else ""
    )
    code = html.escape(command)
    body = explanation + (
        '<code style="display:block;background:#0f172a;color:#e2e8f0;padding:10px 12px;'
        f'border-radius:6px;overflow-x:auto;white-space:pre">{code}</code>'
    )
    _display(_card(title, body, tone="neutral"), f"{title}: {command}")
