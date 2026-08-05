"""Benchmark reproducible del etiquetador local servido por LM Studio.

Compara lotes secuenciales con parallel=1 y dos trabajadores concurrentes
con parallel=2. Usa una muestra fija del manifiesto piloto, excluye el
calentamiento de las mediciones y crea un informe Markdown mínimo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def ejecutar_comando(comando: list[str], timeout: int = 600) -> str:
    resultado = subprocess.run(
        comando,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    salida = resultado.stdout or resultado.stderr or ""
    return ANSI_RE.sub("", salida).strip()


def cargar_modelo(
    model_key: str,
    model_id: str,
    context_length: int,
    parallel: int,
) -> None:
    print(f"Cargando {model_key} con parallel={parallel}...", flush=True)
    ejecutar_comando(["lms", "unload", "--all"])
    salida = ejecutar_comando(
        [
            "lms",
            "load",
            model_key,
            "--context-length",
            str(context_length),
            "--gpu",
            "off",
            "--parallel",
            str(parallel),
            "--identifier",
            model_id,
            "-y",
        ]
    )
    if salida:
        print(salida, flush=True)


def asegurar_servidor(namespace: dict) -> None:
    try:
        response = namespace["requests"].get(f"{namespace['API_BASE']}/models", timeout=10)
        response.raise_for_status()
        return
    except namespace["requests"].RequestException:
        print("LM Studio no responde; iniciando daemon y servidor...", flush=True)
    ejecutar_comando(["lms", "daemon", "up"])
    ejecutar_comando(["lms", "server", "start", "--port", "1234"])
    for _ in range(15):
        try:
            response = namespace["requests"].get(
                f"{namespace['API_BASE']}/models", timeout=10
            )
            response.raise_for_status()
            return
        except namespace["requests"].RequestException:
            time.sleep(2)
    raise RuntimeError(f"LM Studio no respondió en {namespace['API_BASE']}.")


def buscar_celda(notebook: dict, marcador: str) -> str:
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and marcador in source:
            return source
    raise RuntimeError(f"No se encontró en el cuaderno la celda con {marcador!r}.")


def preparar_entorno(repo: Path, max_batch_size: int) -> dict:
    notebook_path = repo / "03_1_etiquetado_llm" / "03_1_etiquetado_llm.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict = {}
    fuentes = [
        buscar_celda(notebook, "from __future__ import annotations"),
        buscar_celda(notebook, "def leer_jsonl"),
    ]
    for source in fuentes:
        exec(compile(source, str(notebook_path), "exec"), namespace)
    namespace["BATCH_SIZE"] = max_batch_size
    for marcador in ("skill_text =", "def validar_semantica"):
        source = buscar_celda(notebook, marcador)
        exec(compile(source, str(notebook_path), "exec"), namespace)
    return namespace


def cargar_muestra(namespace: dict, sample_size: int) -> tuple[list[dict], dict[str, dict]]:
    manifest_path = namespace["OUTPUT_DIR"] / "piloto_ids_n300_seed42.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Ejecuta primero la celda de muestra piloto."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = sample_size + 2
    ids = manifest["chunk_ids"][:required]
    if len(ids) < required:
        raise ValueError(f"El manifiesto necesita al menos {required} IDs.")
    records = [namespace["CHUNK_BY_ID"][chunk_id] for chunk_id in ids]
    references: dict[str, dict] = {}
    reference_dir = namespace["ROOT"] / "para_equiquetado_LLM"
    for path in sorted(reference_dir.glob("cgt_labeled_chunks_parte_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                references[row["chunk_id"]] = row
    return records, references


def resumir_resultado(
    escenario: str,
    server_parallel: int,
    workers: int,
    group_size: int,
    predictions: list[dict],
    usages: list[dict],
    references: dict[str, dict],
    elapsed: float,
    call_seconds: list[float],
) -> dict:
    comparable = [row for row in predictions if row["chunk_id"] in references]
    exact = sum(
        set(row["labels"]) == set(references[row["chunk_id"]]["labels"])
        for row in comparable
    )
    return {
        "escenario": escenario,
        "server_parallel": server_parallel,
        "workers": workers,
        "group_size": group_size,
        "chunks": len(predictions),
        "seconds": round(elapsed, 2),
        "chunks_per_minute": round(len(predictions) / elapsed * 60, 3),
        "call_seconds": [round(value, 2) for value in call_seconds],
        "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usages),
        "reference_records": len(comparable),
        "exact_label_matches": exact,
        "exact_label_pct": round(100 * exact / len(comparable), 1) if comparable else None,
    }


def clasificar_grupo(namespace: dict, records: list[dict]) -> tuple[list[dict], dict, float]:
    started = time.perf_counter()
    rows, usage = namespace["clasificar_lote"](
        records,
        namespace["PRIMARY_MODEL_ID"],
        namespace["PRIMARY_ANNOTATOR_ID"],
    )
    return rows, usage, time.perf_counter() - started


def medir_secuencial(
    namespace: dict,
    sample: list[dict],
    references: dict[str, dict],
    group_size: int,
) -> dict:
    groups = [sample[i : i + group_size] for i in range(0, len(sample), group_size)]
    predictions: list[dict] = []
    usages: list[dict] = []
    durations: list[float] = []
    started = time.perf_counter()
    for group in groups:
        rows, usage, duration = clasificar_grupo(namespace, group)
        predictions.extend(rows)
        usages.append(usage)
        durations.append(duration)
    elapsed = time.perf_counter() - started
    return resumir_resultado(
        f"secuencial_lote_{group_size}",
        1,
        1,
        group_size,
        predictions,
        usages,
        references,
        elapsed,
        durations,
    )


def medir_concurrente(
    namespace: dict,
    sample: list[dict],
    references: dict[str, dict],
    workers: int,
    group_size: int,
) -> dict:
    groups = [sample[i : i + group_size] for i in range(0, len(sample), group_size)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        calls = list(pool.map(lambda group: clasificar_grupo(namespace, group), groups))
    elapsed = time.perf_counter() - started
    predictions = [row for rows, _usage, _duration in calls for row in rows]
    usages = [usage for _rows, usage, _duration in calls]
    durations = [duration for _rows, _usage, duration in calls]
    return resumir_resultado(
        f"concurrente_{workers}h_lote_{group_size}",
        workers,
        workers,
        group_size,
        predictions,
        usages,
        references,
        elapsed,
        durations,
    )


def crear_reporte(
    output_path: Path,
    args: argparse.Namespace,
    results: list[dict],
    warmups: dict[str, float],
) -> None:
    max_exact = max((r["exact_label_pct"] or 0) for r in results)
    candidates = [r for r in results if (r["exact_label_pct"] or 0) == max_exact]
    winner = max(candidates, key=lambda row: row["chunks_per_minute"])
    baseline = next(r for r in results if r["escenario"] == "secuencial_lote_1")
    improvement = 100 * (
        winner["chunks_per_minute"] / baseline["chunks_per_minute"] - 1
    )
    lines = [
        "# Benchmark de LM Studio",
        "",
        f"Fecha: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Entorno",
        "",
        f"- Plataforma: `{platform.platform()}`",
        f"- Procesador: `{os.getenv('PROCESSOR_IDENTIFIER', platform.processor())}`",
        f"- CPU lógicas: {os.cpu_count()}",
        f"- Modelo: `{args.model_key}` con identificador `{args.model_id}`",
        f"- Contexto: {args.context_length} tokens; GPU desactivada",
        f"- Muestra fija: {args.sample_size} chunks del manifiesto piloto",
        "",
        "## Criterios",
        "",
        "1. Rendimiento principal: chunks por minuto; un valor mayor es mejor.",
        "2. Tiempo total para la misma muestra; un valor menor es mejor.",
        "3. Estabilidad: todas las respuestas deben cerrar y validar el JSON.",
        "4. Control de calidad: coincidencia exacta del conjunto de etiquetas con la referencia CGT.",
        "5. El calentamiento se mide por separado y no se incluye en el rendimiento.",
        "",
        "## Resultados cuantitativos",
        "",
        "| Escenario | Parallel | Hilos | Lote | Segundos | Chunks/min | Prompt tokens | Completion tokens | Coincidencia exacta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['escenario']} | {row['server_parallel']} | {row['workers']} | "
            f"{row['group_size']} | {row['seconds']:.2f} | {row['chunks_per_minute']:.3f} | "
            f"{row['prompt_tokens']} | {row['completion_tokens']} | "
            f"{row['exact_label_matches']}/{row['reference_records']} ({row['exact_label_pct']}%) |"
        )
    lines.extend(
        [
            "",
            "## Calentamiento",
            "",
            f"- `parallel=1`: {warmups['parallel_1']:.2f} s.",
            f"- `parallel=2` con dos solicitudes: {warmups['parallel_2']:.2f} s.",
            "",
            "## Recomendación",
            "",
            f"La mejor configuración de esta ejecución fue **{winner['escenario']}**, "
            f"con **{winner['chunks_per_minute']:.3f} chunks/min**, "
            f"**{winner['exact_label_matches']}/{winner['reference_records']}** coincidencias "
            f"y una mejora de **{improvement:.1f}%** frente al lote unitario secuencial.",
            "",
            "La muestra es pequeña; repite con 20 o más chunks antes de una producción extensa. "
            "La proyección lineal no incluye calentamiento, reintentos ni variación por longitud.",
            "",
            "## Resultados JSON",
            "",
            "```json",
            json.dumps(results, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--model-key", default="qwen/qwen3.5-9b")
    parser.add_argument("--model-id", default="qwen-local-primary")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--restore-parallel", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).with_name("REPORTE_DESEMPENO_LMSTUDIO.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size < 4:
        raise ValueError("Usa --sample-size 4 o mayor para comparar todos los escenarios.")
    repo = Path(__file__).resolve().parent.parent
    namespace = preparar_entorno(repo, max_batch_size=4)
    namespace["PRIMARY_MODEL_KEY"] = args.model_key
    namespace["PRIMARY_MODEL_ID"] = args.model_id
    asegurar_servidor(namespace)
    records, references = cargar_muestra(namespace, args.sample_size)
    sample = records[: args.sample_size]
    warmups: dict[str, float] = {}
    results: list[dict] = []
    try:
        cargar_modelo(args.model_key, args.model_id, args.context_length, parallel=1)
        started = time.perf_counter()
        clasificar_grupo(namespace, [records[-2]])
        warmups["parallel_1"] = time.perf_counter() - started
        for group_size in (1, 2, 4):
            print(f"Midiendo secuencial, lote={group_size}...", flush=True)
            result = medir_secuencial(namespace, sample, references, group_size)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

        cargar_modelo(args.model_key, args.model_id, args.context_length, parallel=2)
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda row: clasificar_grupo(namespace, [row]), records[-2:]))
        warmups["parallel_2"] = time.perf_counter() - started
        for group_size in (1, 2):
            print(f"Midiendo concurrente, trabajadores=2, lote={group_size}...", flush=True)
            result = medir_concurrente(namespace, sample, references, 2, group_size)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        print(f"Restaurando el modelo con parallel={args.restore_parallel}...", flush=True)
        cargar_modelo(
            args.model_key,
            args.model_id,
            args.context_length,
            parallel=args.restore_parallel,
        )
    crear_reporte(args.report.resolve(), args, results, warmups)
    print(f"Reporte: {args.report.resolve()}")


if __name__ == "__main__":
    main()
