from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import moderacion_peru.comparison_reporting as reporting
from moderacion_peru.comparison_reporting import (
    COMPARISON_FILENAME,
    REPORT_FILENAME,
    discover_comparison_bundles,
    generate_comparison_report,
    synchronize_latest_local_results,
)
from moderacion_peru.taxonomy import load_taxonomy


def _comparison_payload(*, created_at: str, signature: str) -> dict:
    labels = load_taxonomy().target_labels
    per_label = {
        label: {
            "precision": 0.70,
            "recall": 0.60,
            "f1": 0.6462,
            "support": 20,
        }
        for label in labels
    }
    calibration = {label: {"ece": 0.03, "brier": 0.08} for label in labels}
    metrics = {
        "binary_any_damage_oof": {
            "balanced_accuracy": 0.81,
            "false_negative_rate": 0.16,
            "false_positive_rate": 0.22,
            "f1": 0.73,
            "average_precision": 0.79,
            "risk_lambda": {"0.67": 0.18},
        },
        "average_precision_macro_damage_oof": 0.58,
        "average_precision_macro_damage": 0.57,
        "average_precision_by_label": {label: 0.62 for label in labels},
        "per_label": per_label,
        "calibration_by_label": calibration,
        "f1_macro_damage": 0.61,
        "f1_macro_five": 0.67,
        "f1_micro": 0.78,
        "false_safe_rate_on_damage": 0.20,
        "false_alarm_rate_on_safe": 0.10,
        "expected_calibration_error": 0.03,
        "brier_macro": 0.08,
        "needs_review": {
            "operating_policy": {
                "status": "feasible",
                "validation_operating_point": {"review_load_rate": 0.12},
            }
        },
    }
    candidate = {
        "candidate_id": "model-a",
        "kind": "individual",
        "model_family": "classical:base:logistic_regression",
        "training_regime": "full",
        "validation_metrics": metrics,
        "macro_auprc_safeguard": {
            "status": "pass",
            "difference_candidate_minus_reference": 0.0,
        },
    }
    return {
        "schema_version": "1.2.0",
        "created_at": created_at,
        "comparison_signature": signature,
        "dataset_sha256": "d" * 64,
        "selection_split": "validation",
        "test_status": "sealed_pending_predeclared_operating_policy",
        "selection_policy": {
            "primary": "balanced_accuracy_binary_any_damage_oof",
            "aggregation": "lexicographic",
            "safeguard": "macro_auprc_damage",
            "max_review_rate": 0.20,
            "macro_auprc_noninferiority_margin": 0.02,
        },
        "pareto_front": ["model-a"],
        "selected_for_freeze": "model-a",
        "winner_status": "statistical_tie_or_inconclusive",
        "best_individual": "model-a",
        "best_by_family_slot": {"classical": "model-a"},
        "ranking": [candidate],
        "paired_bootstrap_tests_holm": [
            {
                "metric": "balanced_accuracy_binary_any_damage_oof",
                "reference": "model-a",
                "challenger": "model-b",
                "difference_challenger_minus_reference": -0.01,
                "ci_low": -0.03,
                "ci_high": 0.01,
                "p_value_raw": 0.18,
                "p_value_holm": 0.36,
                "replicates": 2000,
                "grouping": "video_id",
                "parallel_workers": 4,
            }
        ],
    }


def _write_comparison(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / COMPARISON_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_materializes_complete_tables_and_critical_markdown(tmp_path: Path):
    comparison = _write_comparison(
        tmp_path / "input",
        _comparison_payload(created_at="2026-08-15T01:00:00+00:00", signature="sig-a"),
    )

    result = generate_comparison_report(
        comparison,
        output_dir=tmp_path / "output",
        generate_figures=False,
    )

    report = Path(result["report_path"])
    assert report.name == REPORT_FILENAME
    report_text = report.read_text(encoding="utf-8")
    assert "Comparación global de todos los modelos" in report_text
    assert "Mejor modelo por tipo" in report_text
    assert "Desempeño del seleccionado por categoría" in report_text
    assert "Análisis crítico" in report_text
    assert "statistical_tie_or_inconclusive" in report_text
    assert "Test todavía no aparece" in report_text
    assert len(result["global_rows"]) == 1
    assert [row["model_type"] for row in result["best_by_model_type_rows"]] == [
        "classical"
    ]
    assert len(result["selected_category_rows"]) == 5
    assert all(Path(path).is_file() for path in result["table_paths"].values())
    best_by_type_csv = Path(result["table_paths"]["best_by_model_type"]).read_text(
        encoding="utf-8-sig"
    )
    assert "model_type" in best_by_type_csv
    assert "classical" in best_by_type_csv
    bootstrap_csv = Path(result["table_paths"]["paired_bootstrap"]).read_text(
        encoding="utf-8-sig"
    )
    assert "difference_challenger_minus_reference" in bootstrap_csv
    assert "-0.01" in bootstrap_csv


def test_sync_promotes_the_newest_valid_bundle_without_model_files(tmp_path: Path):
    staging = tmp_path / "resultados" / "sincronizados" / "03_07"
    older = _write_comparison(
        staging / "older",
        _comparison_payload(created_at="2026-08-14T23:00:00+00:00", signature="old"),
    )
    newest = _write_comparison(
        staging / "newest",
        _comparison_payload(created_at="2026-08-15T02:00:00+00:00", signature="new"),
    )
    (older.parent / "weights.safetensors").write_bytes(b"not-a-real-model")

    discovered = discover_comparison_bundles([staging])
    assert discovered[0]["comparison_path"] == newest.resolve()

    result = synchronize_latest_local_results(tmp_path, search_roots=[staging])
    copied = Path(result["comparison_path"])
    assert copied.parent == (tmp_path / "resultados" / "modelos").resolve()
    assert (
        json.loads(copied.read_text(encoding="utf-8"))["comparison_signature"] == "new"
    )
    assert not (copied.parent / "weights.safetensors").exists()
    assert result["ranking_count"] == 1
    assert Path(result["manifest_path"]).is_file()


def test_google_drive_sync_downloads_only_new_verified_result_jsons(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "remote_payload"
    comparison_path = _write_comparison(
        source,
        _comparison_payload(
            created_at="2026-08-15T01:00:00+00:00",
            signature="remote-new",
        ),
    )
    (source / "weights.safetensors").write_bytes(b"model-must-not-be-extracted")
    archive_path = tmp_path / "remote.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(
            comparison_path,
            arcname=f"resultados_modelos/{COMPARISON_FILENAME}",
        )
        archive.add(
            source / "weights.safetensors",
            arcname="trainer/weights.safetensors",
        )
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.2.0",
        "published_at": "2026-08-15T01:01:00+00:00",
        "notebook_id": "03_07",
        "run_id": "03_07_working_v2_1",
        "publication_slot": "a",
        "archive": {
            "name": "publications/run_outputs-a.tar",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
        },
    }
    remote = {
        "manifest": manifest,
        "manifest_metadata": {"id": "manifest-id", "name": "run_manifest.json"},
        "remote_files": [],
    }
    monkeypatch.setattr(
        reporting,
        "_remote_publication_candidates",
        lambda *_args, **_kwargs: [remote],
    )
    materializations = []

    def materialize(_session, _remote, destination):
        target = destination / "run_outputs-a.tar"
        destination.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            materializations.append("reused")
            return target, False
        shutil.copyfile(archive_path, target)
        materializations.append("downloaded")
        return target, True

    monkeypatch.setattr(reporting, "_materialize_remote_archive", materialize)

    first = reporting.synchronize_google_drive_results(
        tmp_path,
        authorized_session=object(),
    )
    assert first["status"] == "downloaded_and_synchronized"
    assert first["comparison_signature"] == "remote-new"
    assert first["downloaded"] is True
    assert not any(
        path.name == "weights.safetensors"
        for path in (tmp_path / "resultados").rglob("*")
    )
    state = json.loads(
        (
            tmp_path / "resultados" / "modelos" / reporting.GOOGLE_DRIVE_SYNC_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert state["archive_sha256"] == archive_sha256
    assert state["remote_published_at"] == manifest["published_at"]

    second = reporting.synchronize_google_drive_results(
        tmp_path,
        authorized_session=object(),
    )
    assert second["status"] == "remote_already_current"
    assert second["downloaded"] is False
    assert materializations == ["downloaded", "reused"]
