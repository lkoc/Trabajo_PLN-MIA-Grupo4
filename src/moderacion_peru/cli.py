from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .artifacts import artifact_status
from .consolidation import reconcile_human_reviews
from .datasets import materialize_versioned_training_snapshot
from .device import resolve_device
from .experiments import train_classical_experiments, train_flat_transformers, train_neural_experiment
from .io import read_jsonl
from .migration import migrate_jsonl
from .paths import find_project_root
from .pilot import build_human_pilot, run_ollama_pilot
from .providers import OllamaProvider, ProviderError
from .registry import compare_and_publish_registry
from .schemas import AnnotationRecord, ModelReadyRecord
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
    counters = {"valid": 0, "invalid": 0, "schema_counts": {}, "errors": []}
    if args.path:
        for row in read_jsonl(args.path):
            try:
                kind = args.kind
                if kind == "auto":
                    kind = (
                        "model-ready"
                        if "split" in row or "sample_weight" in row or "legacy_coarse_labels" in row
                        else "annotation"
                    )
                schema = ModelReadyRecord if kind == "model-ready" else AnnotationRecord
                schema.model_validate(row)
                counters["valid"] += 1
                counters["schema_counts"][kind] = counters["schema_counts"].get(kind, 0) + 1
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


def command_prepare_training(args: argparse.Namespace) -> int:
    root = find_project_root()
    consolidated = Path(args.consolidated or root / "datos/etiquetado/consolidado/anotaciones_v2.jsonl")
    chunks = Path(args.chunks or root / "datos/processed/chunks_v2.jsonl")
    reviews = [Path(path) for path in (args.reviews or [root / "datos/etiquetado/humano/labeling_events_v2.jsonl"])]
    reviewed = Path(args.reviewed or root / "datos/etiquetado/consolidado/anotaciones_revisadas_v2.jsonl")
    dataset = Path(args.dataset or root / "datos/model_ready/v2/dataset_5_salidas.jsonl")
    reconciliation = reconcile_human_reviews(
        consolidated,
        reviews,
        reviewed,
        chunks_source=chunks,
    )
    snapshot = materialize_versioned_training_snapshot(reviewed, dataset)
    print_json({"reconciliation": reconciliation, "snapshot": snapshot})
    return 0


def command_train(args: argparse.Namespace) -> int:
    root = find_project_root()
    dataset = Path(args.dataset or root / "datos/model_ready/v2/dataset_5_salidas.jsonl")
    output = Path(args.output or root / "modelos/v2" / args.family)
    if args.family == "classical":
        result = train_classical_experiments(dataset, output, force=args.force)
    elif args.family == "flat":
        result = train_flat_transformers(dataset, output, device=args.device, force=args.force)
    else:
        result = train_neural_experiment(
            dataset,
            output,
            experiment=args.family,
            device=args.device,
            force=args.force,
        )
    print_json(result)
    return 0


def command_publish(args: argparse.Namespace) -> int:
    root = find_project_root()
    result = compare_and_publish_registry(
        args.dataset or root / "datos/model_ready/v2/dataset_5_salidas.jsonl",
        args.candidate_roots or [root / "modelos/v2"],
        args.registry or root / "modelos/registro_modelos_5_salidas.json",
        comparison_path=args.comparison,
    )
    print_json(result)
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
    validate.add_argument(
        "--kind",
        choices=["auto", "annotation", "model-ready"],
        default="auto",
        help="Esquema esperado; auto distingue eventos de filas de entrenamiento",
    )
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

    prepare = subparsers.add_parser("prepare-training", help="Reincorpora revisión humana y congela el snapshot entrenable")
    prepare.add_argument("--consolidated")
    prepare.add_argument("--chunks")
    prepare.add_argument("--reviews", nargs="+")
    prepare.add_argument("--reviewed")
    prepare.add_argument("--dataset")
    prepare.set_defaults(func=command_prepare_training)

    train = subparsers.add_parser("train", help="Ejecuta fit, calibración, test y candidato idempotente")
    train.add_argument("family", choices=["classical", "flat", "cascade", "multitask", "qwen_lora", "qwen_structured"])
    train.add_argument("--dataset")
    train.add_argument("--output")
    train.add_argument("--device", default="auto")
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=command_train)

    publish = subparsers.add_parser("publish-model", help="Compara en validation y publica el registro productivo")
    publish.add_argument("--dataset")
    publish.add_argument("--candidate-roots", nargs="+")
    publish.add_argument("--registry")
    publish.add_argument("--comparison")
    publish.set_defaults(func=command_publish)

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
    production.add_argument("--registry")
    production.set_defaults(func=lambda args: serve(mode="production", host=args.host, port=args.port, reviews=args.reviews, registry=args.registry) or 0)
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
