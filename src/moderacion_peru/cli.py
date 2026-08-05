from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .artifacts import artifact_status
from .device import resolve_device
from .io import read_jsonl
from .migration import migrate_jsonl
from .paths import find_project_root
from .pilot import build_human_pilot, run_ollama_pilot
from .providers import OllamaProvider, ProviderError
from .schemas import AnnotationRecord
from .servers import serve
from .taxonomy import load_taxonomy


def print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def command_preflight(args: argparse.Namespace) -> int:
    root = find_project_root()
    taxonomy = load_taxonomy()
    result = {
        "project_root": str(root),
        "taxonomy": taxonomy.contract_id,
        "target_labels": taxonomy.target_labels,
        "device": resolve_device(args.device).model_dump(),
        "artifacts": artifact_status(root),
    }
    try:
        result["ollama"] = OllamaProvider(model=args.ollama_model).probe()
    except ProviderError as exc:
        result["ollama"] = {"available": False, "error": str(exc)}
    print_json(result)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    taxonomy = load_taxonomy()
    counters = {"valid": 0, "invalid": 0, "errors": []}
    if args.path:
        for row in read_jsonl(args.path):
            try:
                AnnotationRecord.model_validate(row)
                counters["valid"] += 1
            except ValidationError as exc:
                counters["invalid"] += 1
                if len(counters["errors"]) < 20:
                    counters["errors"].append(str(exc))
    print_json({"taxonomy": taxonomy.model_dump(), "records": counters})
    return 1 if counters["invalid"] else 0


def command_artifacts(_: argparse.Namespace) -> int:
    status = artifact_status()
    print_json(status)
    return 0 if status["missing"] == 0 else 2


def command_migrate(args: argparse.Namespace) -> int:
    print_json(migrate_jsonl(args.source, args.destination, args.manifest))
    return 0


def command_pilot(args: argparse.Namespace) -> int:
    sample = build_human_pilot(args.source, size=args.size, seed=args.seed)
    report = run_ollama_pilot(sample, args.models, args.output)
    print_json({key: value for key, value in report.items() if key != "models"})
    return 0


def command_run_stage(args: argparse.Namespace) -> int:
    status = artifact_status()
    print_json({
        "stage": args.stage,
        "mode": "incremental",
        "message": "El cuaderno de la etapa orquesta el módulo correspondiente y omite IDs ya procesados.",
        "artifacts": status,
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modperu")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Verifica taxonomía, hardware, Ollama y artefactos")
    preflight.add_argument("--device", default="auto")
    preflight.add_argument("--ollama-model", default="qwen3.5:4b")
    preflight.set_defaults(func=command_preflight)

    validate = subparsers.add_parser("validate", help="Valida el contrato y un JSONL opcional")
    validate.add_argument("path", nargs="?")
    validate.set_defaults(func=command_validate)

    artifacts = subparsers.add_parser("artifacts", help="Informa artefactos disponibles y faltantes")
    artifacts.set_defaults(func=command_artifacts)

    migrate = subparsers.add_parser("migrate", help="Materializa el contrato v2 sin sobrescribir la fuente")
    migrate.add_argument("source")
    migrate.add_argument("destination")
    migrate.add_argument("manifest")
    migrate.set_defaults(func=command_migrate)

    pilot = subparsers.add_parser("pilot", help="Compara modelos Ollama en referencia humana asistida")
    pilot.add_argument("source")
    pilot.add_argument("output")
    pilot.add_argument("--models", nargs="+", default=["qwen3.5:4b", "qwen3.5:9b", "gemma3:4b"])
    pilot.add_argument("--size", type=int, default=200)
    pilot.add_argument("--seed", type=int, default=20260805)
    pilot.set_defaults(func=command_pilot)

    run_stage = subparsers.add_parser("run-stage", help="Preflight de una etapa incremental")
    run_stage.add_argument("stage", choices=["01_datos", "02_etiquetado", "03_entrenamiento", "04_produccion"])
    run_stage.set_defaults(func=command_run_stage)

    labeling = subparsers.add_parser("serve-labeling", help="Inicia la validación humana local")
    labeling.add_argument("--host", default="127.0.0.1")
    labeling.add_argument("--port", type=int, default=8765)
    labeling.add_argument("--campaign")
    labeling.add_argument("--reviews")
    labeling.set_defaults(func=lambda args: serve(mode="labeling", host=args.host, port=args.port, campaign=args.campaign, reviews=args.reviews) or 0)

    production = subparsers.add_parser("serve-production", help="Inicia el demostrador local")
    production.add_argument("--host", default="127.0.0.1")
    production.add_argument("--port", type=int, default=8765)
    production.add_argument("--reviews")
    production.set_defaults(func=lambda args: serve(mode="production", host=args.host, port=args.port, reviews=args.reviews) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

