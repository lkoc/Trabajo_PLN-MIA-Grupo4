from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .io import sha256_file, write_json_atomic
from .taxonomy import load_taxonomy

COMPARISON_FILENAME = "comparacion_individual_ensemble_validation.json"
FREEZE_FILENAME = "seleccion_congelada.json"
TEST_FILENAME = "test_final_abierto_una_vez.json"
TEST_PREDICTIONS_FILENAME = "test_final_abierto_una_vez_predictions.jsonl"
REPORT_FILENAME = "REPORTE_COMPARACION_MODELOS_03_07.md"
GOOGLE_DRIVE_SYNC_FILENAME = "sincronizacion_google_drive_03_07.json"
GOOGLE_DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
DEFAULT_DRIVE_RUN_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1V-oHSklbHQ4sisVPdwayF-Tv66b6av0-"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}")
    return payload


def _parse_timestamp(value: Any, fallback: float) -> float:
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return fallback


def _valid_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _matching_optional_artifact(
    path: Path, comparison_signature: str | None
) -> Path | None:
    if not path.is_file():
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        comparison_signature
        and payload.get("comparison_signature") != comparison_signature
    ):
        return None
    return path


def _relative_or_absolute(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _drive_folder_id(value: str) -> str:
    text = str(value).strip()
    marker = "/folders/"
    folder_id = (
        text.split(marker, 1)[1].split("/", 1)[0].split("?", 1)[0]
        if marker in text
        else text
    )
    if not folder_id or any(
        character
        not in "-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for character in folder_id
    ):
        raise ValueError(f"Identificador o URL de carpeta Drive inválido: {value!r}")
    return folder_id


def _load_google_drive_sync_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "google_drive_03_07.json"
    if not path.is_file():
        return {
            "run_folder_url": DEFAULT_DRIVE_RUN_FOLDER_URL,
            "notebook_id": "03_07",
            "run_id": "03_07_working_v2_1",
            "oauth_client_path": "config/google_drive_oauth_client.json",
            "token_path": ".secrets/google_drive_token.json",
        }
    payload = _load_json(path)
    required = {"run_folder_url", "notebook_id", "run_id"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{path.name} no declara: {', '.join(missing)}")
    return payload


def _google_drive_authorized_session(
    credentials_path: Path,
    token_path: Path,
    *,
    interactive_auth: bool,
) -> Any:
    """Autoriza Drive en modo lectura sin depender de Drive Desktop."""

    try:
        import truststore

        truststore.inject_into_ssl()
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import AuthorizedSession, Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError(
            "Faltan dependencias de sincronización. Instale el proyecto con "
            '`python -m pip install -e ".[cuadernos]"`.'
        ) from exc

    scopes = [GOOGLE_DRIVE_READONLY_SCOPE]
    credentials = None
    if token_path.is_file():
        credentials = Credentials.from_authorized_user_file(token_path, scopes)
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except RefreshError:
            credentials = None
    if not credentials or not credentials.valid:
        if not interactive_auth:
            raise PermissionError(
                "Google Drive requiere autorización. Ejecute una vez con "
                "interactive_auth=True para abrir el consentimiento OAuth."
            )
        if not credentials_path.is_file():
            raise FileNotFoundError(
                "Falta el cliente OAuth de Google Drive: "
                f"{credentials_path}. Descargue un cliente de tipo aplicación de escritorio, "
                "guárdelo en esa ruta y vuelva a ejecutar la celda."
            )
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
        credentials = flow.run_local_server(
            port=0, access_type="offline", prompt="consent"
        )
    token_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(token_path, credentials.to_json())
    try:
        token_path.chmod(0o600)
    except OSError:
        pass
    return AuthorizedSession(credentials)


def _drive_api_error(response: Any, action: str) -> None:
    if 200 <= int(response.status_code) < 300:
        return
    detail = str(getattr(response, "text", ""))[:500]
    raise RuntimeError(
        f"Google Drive API rechazó {action}: HTTP {response.status_code}. {detail}"
    )


def _drive_folder_children(session: Any, folder_id: str) -> list[dict[str, Any]]:
    endpoint = "https://www.googleapis.com/drive/v3/files"
    page_token = None
    files: list[dict[str, Any]] = []
    while True:
        parameters = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "spaces": "drive",
            "pageSize": 1000,
            "orderBy": "modifiedTime desc,name",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,md5Checksum)",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token:
            parameters["pageToken"] = page_token
        response = session.get(endpoint, params=parameters, timeout=60)
        _drive_api_error(response, f"listar la carpeta {folder_id}")
        payload = response.json()
        files.extend(payload.get("files", ()))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return files


def _drive_download_bytes(
    session: Any,
    file_id: str,
    *,
    maximum_bytes: int,
) -> bytes:
    endpoint = f"https://www.googleapis.com/drive/v3/files/{quote(file_id, safe='')}"
    response = session.get(endpoint, params={"alt": "media"}, timeout=120)
    _drive_api_error(response, f"descargar {file_id}")
    content = bytes(response.content)
    if len(content) > maximum_bytes:
        raise ValueError(f"El archivo {file_id} excede {maximum_bytes} bytes")
    return content


def _drive_download_json(session: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    content = _drive_download_bytes(
        session, str(metadata["id"]), maximum_bytes=2 * 1024 * 1024
    )
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata.get('name')} no contiene un objeto JSON")
    return payload


def _drive_download_file(
    session: Any,
    metadata: Mapping[str, Any],
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == expected_bytes
        and sha256_file(destination) == expected_sha256
    ):
        return destination
    endpoint = f"https://www.googleapis.com/drive/v3/files/{quote(str(metadata['id']), safe='')}"
    response = session.get(endpoint, params={"alt": "media"}, timeout=300, stream=True)
    _drive_api_error(response, f"descargar {metadata.get('name')}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    digest = hashlib.sha256()
    written = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                digest.update(block)
                written += len(block)
            handle.flush()
            os.fsync(handle.fileno())
        if written != expected_bytes:
            raise ValueError(
                f"{metadata.get('name')}: {written} de {expected_bytes} bytes"
            )
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"{metadata.get('name')}: SHA-256 no coincide")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        close = getattr(response, "close", None)
        if callable(close):
            close()
    return destination


def _remote_publication_candidates(
    session: Any,
    run_folder_id: str,
    *,
    notebook_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    run_children = _drive_folder_children(session, run_folder_id)
    publications_folder = next(
        (
            item
            for item in run_children
            if item.get("name") == "publications"
            and item.get("mimeType") == "application/vnd.google-apps.folder"
        ),
        None,
    )
    publication_children = (
        _drive_folder_children(session, str(publications_folder["id"]))
        if publications_folder
        else []
    )
    by_relative_name = {
        str(item.get("name")): item for item in run_children if item.get("name")
    }
    by_relative_name.update(
        {
            f"publications/{item.get('name')}": item
            for item in publication_children
            if item.get("name")
        }
    )
    manifest_items = [
        item for item in run_children if item.get("name") == "run_manifest.json"
    ]
    manifest_items.extend(
        item
        for item in publication_children
        if str(item.get("name", "")).startswith("run_manifest-")
        and str(item.get("name", "")).endswith(".json")
    )
    candidates = []
    seen: set[tuple[str, str]] = set()
    for item in manifest_items:
        try:
            manifest = _drive_download_json(session, item)
            archive = manifest["archive"]
            identity = (str(archive["name"]), str(archive["sha256"]).lower())
            if (
                manifest.get("notebook_id") != notebook_id
                or manifest.get("run_id") != run_id
            ):
                continue
            if identity in seen:
                continue
            expected_bytes = int(archive["bytes"])
            expected_sha = str(archive["sha256"]).lower()
            if expected_bytes <= 0 or not _valid_sha256(expected_sha):
                continue
            remote_files = []
            parts = archive.get("parts")
            declared_files = parts if parts else [archive]
            for declared in declared_files:
                name = str(declared["name"])
                metadata = by_relative_name.get(name)
                if metadata is None:
                    raise FileNotFoundError(f"Drive no contiene {name}")
                remote_files.append({"declaration": declared, "metadata": metadata})
            candidates.append(
                {
                    "manifest": manifest,
                    "manifest_metadata": item,
                    "remote_files": remote_files,
                    "archive_identity": identity,
                }
            )
            seen.add(identity)
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            FileNotFoundError,
        ):
            continue
    candidates.sort(
        key=lambda entry: _parse_timestamp(entry["manifest"].get("published_at"), 0.0),
        reverse=True,
    )
    return candidates


def discover_comparison_bundles(
    search_roots: Iterable[str | Path],
) -> list[dict[str, Any]]:
    """Descubre comparaciones locales válidas y sus artefactos compañeros."""

    discovered: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw_root in search_roots:
        root = Path(raw_root).resolve()
        if not root.exists():
            continue
        candidates = (
            [root]
            if root.name == COMPARISON_FILENAME
            else root.rglob(COMPARISON_FILENAME)
        )
        for comparison_path in candidates:
            comparison_path = comparison_path.resolve()
            if comparison_path in seen or not comparison_path.is_file():
                continue
            seen.add(comparison_path)
            try:
                comparison = _load_json(comparison_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            signature = comparison.get("comparison_signature")
            ranking = comparison.get("ranking")
            if not signature or not isinstance(ranking, list) or not ranking:
                continue
            freeze_path = _matching_optional_artifact(
                comparison_path.with_name(FREEZE_FILENAME), str(signature)
            )
            test_path = _matching_optional_artifact(
                comparison_path.with_name(TEST_FILENAME), str(signature)
            )
            created_timestamp = _parse_timestamp(
                comparison.get("created_at"), comparison_path.stat().st_mtime
            )
            if test_path is not None:
                test_payload = _load_json(test_path)
                created_timestamp = max(
                    created_timestamp,
                    _parse_timestamp(
                        test_payload.get("created_at"), test_path.stat().st_mtime
                    ),
                )
            discovered.append(
                {
                    "comparison_path": comparison_path,
                    "freeze_path": freeze_path,
                    "test_path": test_path,
                    "test_predictions_path": (
                        comparison_path.with_name(TEST_PREDICTIONS_FILENAME)
                        if test_path is not None
                        and comparison_path.with_name(
                            TEST_PREDICTIONS_FILENAME
                        ).is_file()
                        else None
                    ),
                    "comparison_signature": str(signature),
                    "dataset_sha256": comparison.get("dataset_sha256"),
                    "created_at": comparison.get("created_at"),
                    "timestamp": created_timestamp,
                    "ranking_count": len(ranking),
                    "selected_id": comparison.get("selected_for_freeze"),
                    "has_test": test_path is not None,
                }
            )
    return sorted(
        discovered,
        key=lambda row: (float(row["timestamp"]), bool(row["has_test"])),
        reverse=True,
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    try:
        shutil.copyfile(source, temporary_name)
        if sha256_file(source) != sha256_file(temporary_name):
            raise ValueError(f"La copia local de {source.name} no conserva SHA-256")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def synchronize_latest_local_results(
    project_root: str | Path,
    *,
    search_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Promueve la comparación local más reciente a ``resultados/modelos``.

    Esta función consume tanto una copia obtenida mediante la Google Drive API como
    resultados que ya existan en el proyecto, valida la firma compartida y conserva
    los bytes originales.
    """

    root = Path(project_root).resolve()
    roots = list(search_roots or ())
    if not roots:
        roots = [
            root / "resultados" / "sincronizados" / "03_07",
            root / "resultados" / "modelos",
        ]
    bundles = discover_comparison_bundles(roots)
    if not bundles:
        searched = ", ".join(str(Path(path)) for path in roots)
        raise FileNotFoundError(
            "No se encontró una comparación válida de 03_07. "
            f"Rutas inspeccionadas: {searched}"
        )
    selected = bundles[0]
    destination = root / "resultados" / "modelos"
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in (
        "comparison_path",
        "freeze_path",
        "test_path",
        "test_predictions_path",
    ):
        source = selected.get(key)
        if source is None:
            continue
        source_path = Path(source)
        target = destination / source_path.name
        _copy_atomic(source_path, target)
        copied[key] = str(target)
    manifest = {
        "schema_version": "1.0.0",
        "operation": "synchronize_03_07_results",
        "synchronized_at": datetime.now(UTC).isoformat(),
        "source_directory": _relative_or_absolute(
            Path(selected["comparison_path"]).parent, root
        ),
        "comparison_signature": selected["comparison_signature"],
        "dataset_sha256": selected.get("dataset_sha256"),
        "source_created_at": selected.get("created_at"),
        "ranking_count": selected["ranking_count"],
        "selected_id": selected.get("selected_id"),
        "has_test": selected["has_test"],
        "artifacts": {
            key: {
                "path": _relative_or_absolute(value, root),
                "bytes": Path(value).stat().st_size,
                "sha256": sha256_file(value),
            }
            for key, value in copied.items()
        },
        "candidates_considered": [
            {
                "comparison_path": _relative_or_absolute(row["comparison_path"], root),
                "created_at": row.get("created_at"),
                "has_test": row["has_test"],
                "comparison_signature": row["comparison_signature"],
            }
            for row in bundles
        ],
    }
    manifest_path = destination / "sincronizacion_03_07.json"
    write_json_atomic(manifest_path, manifest)
    return {
        "status": "synchronized",
        **copied,
        "manifest_path": str(manifest_path),
        "comparison_signature": selected["comparison_signature"],
        "source_created_at": selected.get("created_at"),
        "ranking_count": selected["ranking_count"],
        "selected_id": selected.get("selected_id"),
        "has_test": selected["has_test"],
    }


def _materialize_remote_archive(
    session: Any,
    remote: Mapping[str, Any],
    destination_root: Path,
) -> tuple[Path, bool]:
    manifest = remote["manifest"]
    archive = manifest["archive"]
    expected_bytes = int(archive["bytes"])
    expected_sha256 = str(archive["sha256"]).lower()
    archive_name = Path(str(archive["name"])).name
    archive_path = destination_root / archive_name
    if (
        archive_path.is_file()
        and archive_path.stat().st_size == expected_bytes
        and sha256_file(archive_path) == expected_sha256
    ):
        return archive_path, False

    remote_files = list(remote["remote_files"])
    if not archive.get("parts"):
        entry = remote_files[0]
        return (
            _drive_download_file(
                session,
                entry["metadata"],
                archive_path,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            ),
            True,
        )

    part_paths = []
    for entry in remote_files:
        declaration = entry["declaration"]
        part_path = destination_root / "parts" / Path(str(declaration["name"])).name
        part_paths.append(
            _drive_download_file(
                session,
                entry["metadata"],
                part_path,
                expected_bytes=int(declaration["bytes"]),
                expected_sha256=str(declaration["sha256"]).lower(),
            )
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.", dir=archive_path.parent
    )
    digest = hashlib.sha256()
    written = 0
    try:
        with os.fdopen(fd, "wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        written += len(block)
            output.flush()
            os.fsync(output.fileno())
        if written != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError(
                "Las partes descargadas no reconstruyen el archivo declarado"
            )
        os.replace(temporary_name, archive_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return archive_path, True


def _extract_result_jsons(archive_path: Path, destination: Path) -> list[Path]:
    allowed = {
        COMPARISON_FILENAME,
        FREEZE_FILENAME,
        TEST_FILENAME,
    }
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    seen: set[str] = set()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            filename = Path(member.name).name
            if filename not in allowed:
                continue
            if not member.isfile() or filename in seen:
                raise ValueError(
                    f"Entrada de resultados insegura o duplicada: {member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"No se pudo leer {member.name}")
            target = destination / filename
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", dir=destination
            )
            try:
                with source, os.fdopen(fd, "wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            seen.add(filename)
            extracted.append(target)
    if COMPARISON_FILENAME not in seen:
        raise ValueError(
            f"La publicación {archive_path.name} no contiene {COMPARISON_FILENAME}"
        )
    bundles = discover_comparison_bundles([destination])
    if not bundles:
        raise ValueError("Los JSON extraídos no forman un bundle 03_07 verificable")
    return extracted


def synchronize_google_drive_results(
    project_root: str | Path,
    *,
    run_folder_url: str | None = None,
    credentials_path: str | Path | None = None,
    token_path: str | Path | None = None,
    interactive_auth: bool = True,
    authorized_session: Any | None = None,
) -> dict[str, Any]:
    """Sincroniza automáticamente la publicación 03_07 más reciente desde Drive.

    Usa OAuth de solo lectura y la API oficial; no depende de Drive Desktop. Solo
    extrae los JSON de comparación, selección congelada y test. El TAR se valida
    por tamaño y SHA-256 antes de leerlo, y nunca se materializan pesos.
    """

    root = Path(project_root).resolve()
    config = _load_google_drive_sync_config(root)
    active_run_url = str(run_folder_url or config["run_folder_url"])
    client_path = (
        Path(credentials_path).resolve()
        if credentials_path
        else root
        / str(config.get("oauth_client_path", "config/google_drive_oauth_client.json"))
    )
    local_token_path = (
        Path(token_path).resolve()
        if token_path
        else root / str(config.get("token_path", ".secrets/google_drive_token.json"))
    )
    session = authorized_session or _google_drive_authorized_session(
        client_path,
        local_token_path,
        interactive_auth=interactive_auth,
    )
    candidates = _remote_publication_candidates(
        session,
        _drive_folder_id(active_run_url),
        notebook_id=str(config["notebook_id"]),
        run_id=str(config["run_id"]),
    )
    if not candidates:
        raise FileNotFoundError(
            "Google Drive no contiene una publicación 03_07 válida en la carpeta configurada"
        )
    remote = candidates[0]
    manifest = remote["manifest"]
    archive = manifest["archive"]
    remote_published_at = str(manifest.get("published_at", ""))
    remote_timestamp = _parse_timestamp(remote_published_at, 0.0)
    staging_root = root / "resultados" / "sincronizados" / "03_07"
    canonical_root = root / "resultados" / "modelos"
    local_bundles = discover_comparison_bundles([canonical_root, staging_root])
    latest_local_timestamp = (
        float(local_bundles[0]["timestamp"]) if local_bundles else 0.0
    )
    state_path = canonical_root / GOOGLE_DRIVE_SYNC_FILENAME
    previous_state = _load_json(state_path) if state_path.is_file() else {}
    expected_sha256 = str(archive["sha256"]).lower()
    same_remote_publication = (
        previous_state.get("archive_sha256") == expected_sha256
        and previous_state.get("remote_published_at") == remote_published_at
    )
    if (
        not same_remote_publication
        and latest_local_timestamp > remote_timestamp
        and local_bundles
    ):
        local_sync = synchronize_latest_local_results(root)
        return {
            "status": "local_newer_than_drive",
            "remote_published_at": remote_published_at,
            "remote_archive_sha256": expected_sha256,
            **local_sync,
        }

    publication_root = staging_root / str(config["run_id"]) / expected_sha256[:12]
    archive_path, downloaded = _materialize_remote_archive(
        session, remote, publication_root
    )
    extracted_root = publication_root / "extraido" / "resultados_modelos"
    existing_bundles = discover_comparison_bundles([extracted_root])
    if not existing_bundles:
        extracted_paths = _extract_result_jsons(archive_path, extracted_root)
    else:
        extracted_paths = [
            path
            for path in (
                extracted_root / COMPARISON_FILENAME,
                extracted_root / FREEZE_FILENAME,
                extracted_root / TEST_FILENAME,
            )
            if path.is_file()
        ]
    local_sync = synchronize_latest_local_results(
        root,
        search_roots=(staging_root, canonical_root),
    )
    state = {
        "schema_version": "1.0.0",
        "status": (
            "downloaded_and_synchronized" if downloaded else "remote_already_current"
        ),
        "checked_at": datetime.now(UTC).isoformat(),
        "remote_published_at": remote_published_at,
        "remote_manifest_file_id": remote["manifest_metadata"].get("id"),
        "remote_manifest_name": remote["manifest_metadata"].get("name"),
        "remote_run_folder_url": active_run_url,
        "notebook_id": manifest.get("notebook_id"),
        "run_id": manifest.get("run_id"),
        "publication_slot": manifest.get("publication_slot"),
        "archive_name": archive.get("name"),
        "archive_sha256": expected_sha256,
        "archive_bytes": int(archive["bytes"]),
        "archive_path": _relative_or_absolute(archive_path, root),
        "extracted_result_files": [
            {
                "path": _relative_or_absolute(path, root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in extracted_paths
        ],
        "comparison_signature": local_sync["comparison_signature"],
        "selected_id": local_sync.get("selected_id"),
        "has_test": local_sync["has_test"],
    }
    write_json_atomic(state_path, state)
    return {
        **local_sync,
        "status": state["status"],
        "drive_state_path": str(state_path),
        "remote_published_at": remote_published_at,
        "remote_archive_sha256": expected_sha256,
        "downloaded": downloaded,
    }


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _metric_sort_value(row: Mapping[str, Any], metric: str) -> float:
    value = _float(row.get(metric))
    return value if value is not None else -1.0


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _global_validation_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(comparison.get("ranking", ()), start=1):
        metrics = candidate.get("validation_metrics", {})
        binary = metrics.get("binary_any_damage_oof", metrics.get("any_damage", {}))
        review_policy = _nested(metrics, "needs_review", "operating_policy") or {}
        operating_point = review_policy.get("validation_operating_point") or {}
        safeguard = candidate.get("macro_auprc_safeguard", {})
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "kind": candidate.get("kind"),
                "model_family": candidate.get("model_family"),
                "training_regime": candidate.get("training_regime"),
                "members": "; ".join(
                    str(item) for item in candidate.get("members", ())
                ),
                "balanced_accuracy_any_damage_oof": _float(
                    binary.get("balanced_accuracy")
                ),
                "fnr_any_damage_oof": _float(binary.get("false_negative_rate")),
                "fpr_any_damage_oof": _float(binary.get("false_positive_rate")),
                "risk_0_67_oof": _float(_nested(binary, "risk_lambda", "0.67")),
                "f1_any_damage_oof": _float(binary.get("f1")),
                "auprc_any_damage_oof": _float(binary.get("average_precision")),
                "macro_auprc_damage_oof": _float(
                    metrics.get("average_precision_macro_damage_oof")
                ),
                "macro_auprc_damage_deployment": _float(
                    metrics.get("average_precision_macro_damage")
                ),
                "macro_f1_damage": _float(metrics.get("f1_macro_damage")),
                "macro_f1_five": _float(metrics.get("f1_macro_five")),
                "micro_f1": _float(metrics.get("f1_micro")),
                "false_safe_rate_on_damage": _float(
                    metrics.get("false_safe_rate_on_damage")
                ),
                "false_alarm_rate_on_safe": _float(
                    metrics.get("false_alarm_rate_on_safe")
                ),
                "ece": _float(metrics.get("expected_calibration_error")),
                "brier_macro": _float(metrics.get("brier_macro")),
                "review_policy_status": review_policy.get("status"),
                "review_load_rate": _float(operating_point.get("review_load_rate")),
                "macro_auprc_safeguard": safeguard.get("status"),
                "macro_auprc_difference_vs_reference": _float(
                    safeguard.get("difference_candidate_minus_reference")
                ),
                "selected": candidate.get("candidate_id")
                == comparison.get("selected_for_freeze"),
            }
        )
    return rows


def _best_by_model_type_rows(
    comparison: Mapping[str, Any],
    global_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Resume el mejor clásico, transformer, Qwen y ensemble.

    "Mejor" conserva el orden lexicográfico global ya congelado en 03_07. Para
    las tres familias individuales se prioriza ``best_by_family_slot``; si un
    artefacto anterior no incluye ese campo, se usa el primer candidato del
    tipo en el ranking. El ensemble se obtiene siempre del mismo ranking.
    """

    rows_by_id = {str(row.get("candidate_id")): row for row in global_rows}
    declared = comparison.get("best_by_family_slot") or {}

    def belongs_to_type(row: Mapping[str, Any], model_type: str) -> bool:
        kind = str(row.get("kind") or "").lower()
        family = str(row.get("model_family") or "").lower()
        if model_type == "ensemble":
            return kind == "ensemble" or family == "ensemble"
        if kind == "ensemble":
            return False
        if model_type == "classical":
            return family.startswith("classical")
        if model_type == "qwen":
            return "qwen" in family
        if model_type == "transformer":
            return not family.startswith("classical") and "qwen" not in family
        return False

    result: list[dict[str, Any]] = []
    for model_type in ("classical", "transformer", "qwen", "ensemble"):
        candidate = rows_by_id.get(str(declared.get(model_type)))
        if candidate is None or not belongs_to_type(candidate, model_type):
            candidate = next(
                (row for row in global_rows if belongs_to_type(row, model_type)),
                None,
            )
        if candidate is None:
            continue
        result.append(
            {
                "model_type": model_type,
                "candidate_id": candidate.get("candidate_id"),
                "kind": candidate.get("kind"),
                "model_family": candidate.get("model_family"),
                "rank": candidate.get("rank"),
                "balanced_accuracy_any_damage_oof": candidate.get(
                    "balanced_accuracy_any_damage_oof"
                ),
                "macro_auprc_damage_oof": candidate.get(
                    "macro_auprc_damage_oof"
                ),
                "fnr_any_damage_oof": candidate.get("fnr_any_damage_oof"),
                "fpr_any_damage_oof": candidate.get("fpr_any_damage_oof"),
                "macro_f1_damage": candidate.get("macro_f1_damage"),
                "macro_auprc_safeguard": candidate.get(
                    "macro_auprc_safeguard"
                ),
                "selected": candidate.get("selected"),
                "selection_basis": "global_lexicographic_ranking_validation",
            }
        )
    return result


def _category_validation_rows(
    comparison: Mapping[str, Any], labels: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(comparison.get("ranking", ()), start=1):
        metrics = candidate.get("validation_metrics", {})
        safeguard_status = _nested(candidate, "macro_auprc_safeguard", "status")
        for label in labels:
            discrete = _nested(metrics, "per_label", label) or {}
            calibration = _nested(metrics, "calibration_by_label", label) or {}
            rows.append(
                {
                    "rank_global": rank,
                    "candidate_id": candidate.get("candidate_id"),
                    "kind": candidate.get("kind"),
                    "model_family": candidate.get("model_family"),
                    "macro_auprc_safeguard": safeguard_status,
                    "label": label,
                    "average_precision": _float(
                        _nested(metrics, "average_precision_by_label", label)
                    ),
                    "precision": _float(discrete.get("precision")),
                    "recall": _float(discrete.get("recall")),
                    "f1": _float(discrete.get("f1")),
                    "support": discrete.get("support"),
                    "ece": _float(calibration.get("ece")),
                    "brier": _float(calibration.get("brier")),
                    "selected": candidate.get("candidate_id")
                    == comparison.get("selected_for_freeze"),
                }
            )
    return rows


def _category_winners(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> list[dict[str, Any]]:
    winners = []
    for label in labels:
        all_label_rows = [row for row in rows if row.get("label") == label]
        # Una cabeza degenerada puede maximizar recall prediciendo siempre positivo.
        # El resumen de ganadores solo considera candidatos que superaron la
        # salvaguarda global; el CSV detallado conserva todos los candidatos.
        label_rows = [
            row for row in all_label_rows if row.get("macro_auprc_safeguard") != "fail"
        ] or all_label_rows
        if not label_rows:
            continue

        ap = max(
            label_rows, key=lambda item: _metric_sort_value(item, "average_precision")
        )
        f1 = max(label_rows, key=lambda item: _metric_sort_value(item, "f1"))
        recall = max(label_rows, key=lambda item: _metric_sort_value(item, "recall"))
        winners.append(
            {
                "label": label,
                "best_average_precision_model": ap.get("candidate_id"),
                "best_average_precision": ap.get("average_precision"),
                "best_f1_model": f1.get("candidate_id"),
                "best_f1": f1.get("f1"),
                "best_recall_model": recall.get("candidate_id"),
                "best_recall": recall.get("recall"),
                "precision_at_best_recall": recall.get("precision"),
                "support": f1.get("support"),
            }
        )
    return winners


def _bootstrap_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for test in comparison.get("paired_bootstrap_tests_holm", ()):
        rows.append(
            {
                "metric": test.get("metric"),
                "reference": test.get("reference"),
                "challenger": test.get("challenger"),
                "difference_challenger_minus_reference": _float(
                    test.get("difference_challenger_minus_reference")
                ),
                "ci_low": _float(test.get("ci_low")),
                "ci_high": _float(test.get("ci_high")),
                "p_value_raw": _float(test.get("p_value_raw")),
                "p_value_holm": _float(test.get("p_value_holm")),
                "replicates": test.get("replicates"),
                "grouping": test.get("grouping"),
                "parallel_workers": test.get("parallel_workers"),
            }
        )
    return rows


def _test_tables(
    test: Mapping[str, Any] | None, labels: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not test:
        return [], []
    views = (
        (
            "natural",
            test.get("primary_metrics_natural_prevalence") or test.get("metrics"),
        ),
        ("4_to_1", test.get("secondary_metrics_4_to_1")),
    )
    global_rows = []
    category_rows = []
    for view_name, metrics in views:
        if not isinstance(metrics, Mapping):
            continue
        binary = metrics.get(
            "binary_any_damage_frozen_gate", metrics.get("any_damage", {})
        )
        global_rows.append(
            {
                "view": view_name,
                "selected_id": test.get("selected_id"),
                "rows": test.get(
                    "test_rows_natural"
                    if view_name == "natural"
                    else "test_rows_4_to_1"
                ),
                "balanced_accuracy_any_damage": _float(binary.get("balanced_accuracy")),
                "fnr_any_damage": _float(binary.get("false_negative_rate")),
                "fpr_any_damage": _float(binary.get("false_positive_rate")),
                "f1_any_damage": _float(binary.get("f1")),
                "macro_auprc_damage": _float(
                    metrics.get("average_precision_macro_damage")
                ),
                "macro_f1_damage": _float(metrics.get("f1_macro_damage")),
                "macro_f1_five": _float(metrics.get("f1_macro_five")),
                "micro_f1": _float(metrics.get("f1_micro")),
                "false_safe_rate_on_damage": _float(
                    metrics.get("false_safe_rate_on_damage")
                ),
                "false_alarm_rate_on_safe": _float(
                    metrics.get("false_alarm_rate_on_safe")
                ),
                "ece": _float(metrics.get("expected_calibration_error")),
                "brier_macro": _float(metrics.get("brier_macro")),
            }
        )
        for label in labels:
            discrete = _nested(metrics, "per_label", label) or {}
            category_rows.append(
                {
                    "view": view_name,
                    "label": label,
                    "average_precision": _float(
                        _nested(metrics, "average_precision_by_label", label)
                    ),
                    "precision": _float(discrete.get("precision")),
                    "recall": _float(discrete.get("recall")),
                    "f1": _float(discrete.get("f1")),
                    "support": discrete.get("support"),
                    "ece": _float(
                        _nested(metrics, "calibration_by_label", label, "ece")
                    ),
                    "brier": _float(
                        _nested(metrics, "calibration_by_label", label, "brier")
                    ),
                }
            )
    return global_rows, category_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _short_identifier(value: Any, limit: int = 42) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_metric(value: Any) -> str:
    number = _float(value)
    return "—" if number is None else f"{number:.4f}"


def _markdown_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]
) -> str:
    if not rows:
        return "_No disponible._"
    header = "| " + " | ".join(title for _, title in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rendered = [header, separator]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cell = _format_metric(value)
            else:
                cell = "—" if value is None else str(value)
            values.append(cell.replace("|", "\\|").replace("\n", " "))
        rendered.append("| " + " | ".join(values) + " |")
    return "\n".join(rendered)


def _generate_figures(
    output_dir: Path,
    global_rows: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
    selected_id: str,
    *,
    max_models: int,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib y numpy son necesarios para generar las figuras de 03_07a"
        ) from exc

    figure_dir = output_dir / "figuras_03_07a"
    figure_dir.mkdir(parents=True, exist_ok=True)
    top = list(global_rows[:max_models])
    figures: list[Path] = []

    labels = [_short_identifier(row["candidate_id"], 34) for row in reversed(top)]
    ba = [row.get("balanced_accuracy_any_damage_oof") or 0.0 for row in reversed(top)]
    ap = [row.get("macro_auprc_damage_oof") or 0.0 for row in reversed(top)]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(labels))))
    ax.barh(positions - 0.18, ba, height=0.35, label="BA ANY_DAMAGE OOF")
    ax.barh(positions + 0.18, ap, height=0.35, label="Macro-AUPRC daño OOF")
    ax.set_yticks(positions, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Métrica (0–1)")
    ax.set_title(f"Comparación global · primeros {len(top)}")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = figure_dir / "ranking_global_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)

    selected_categories = [
        row for row in category_rows if row.get("candidate_id") == selected_id
    ]
    if selected_categories:
        category_labels = [str(row["label"]) for row in selected_categories]
        f1 = [row.get("f1") or 0.0 for row in selected_categories]
        category_ap = [
            row.get("average_precision") or 0.0 for row in selected_categories
        ]
        x = np.arange(len(category_labels))
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(x - 0.2, category_ap, width=0.4, label="AUPRC")
        ax.bar(x + 0.2, f1, width=0.4, label="F1")
        ax.set_xticks(x, category_labels, rotation=22, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Métrica (0–1)")
        ax.set_title(f"Desempeño por categoría · {_short_identifier(selected_id, 70)}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = figure_dir / "seleccion_por_categoria_validation.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append(path)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for kind, marker, color in (
        ("individual", "o", "#2563eb"),
        ("ensemble", "s", "#ea580c"),
    ):
        subset = [row for row in global_rows if row.get("kind") == kind]
        ax.scatter(
            [row.get("macro_auprc_damage_oof") or 0.0 for row in subset],
            [row.get("balanced_accuracy_any_damage_oof") or 0.0 for row in subset],
            label=kind,
            marker=marker,
            color=color,
            alpha=0.75,
        )
    selected = next((row for row in global_rows if row.get("selected")), None)
    if selected:
        ax.scatter(
            [selected.get("macro_auprc_damage_oof") or 0.0],
            [selected.get("balanced_accuracy_any_damage_oof") or 0.0],
            marker="*",
            s=260,
            color="#16a34a",
            edgecolor="black",
            label="seleccionado",
            zorder=5,
        )
    ax.set_xlabel("Macro-AUPRC de daño OOF")
    ax.set_ylabel("Balanced accuracy ANY_DAMAGE OOF")
    ax.set_title("Frontera de desempeño en validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = figure_dir / "frontera_ba_macro_auprc_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return figures


def _critical_analysis(
    comparison: Mapping[str, Any],
    freeze: Mapping[str, Any] | None,
    global_rows: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    selected_id = str(comparison.get("selected_for_freeze"))
    selected = next(
        row for row in global_rows if row.get("candidate_id") == selected_id
    )
    best_individual_id = comparison.get("best_individual")
    best_individual = next(
        (row for row in global_rows if row.get("candidate_id") == best_individual_id),
        None,
    )
    paragraphs = []
    if best_individual:
        delta_ba = (selected.get("balanced_accuracy_any_damage_oof") or 0.0) - (
            best_individual.get("balanced_accuracy_any_damage_oof") or 0.0
        )
        delta_ap = (selected.get("macro_auprc_damage_oof") or 0.0) - (
            best_individual.get("macro_auprc_damage_oof") or 0.0
        )
        paragraphs.append(
            f"La regla predeclarada seleccionó `{selected_id}`. Frente al mejor individuo "
            f"(`{best_individual_id}`), la diferencia puntual es {delta_ba:+.4f} en BA "
            f"ANY_DAMAGE OOF y {delta_ap:+.4f} en macro-AUPRC de daño OOF. Estas diferencias "
            "describen validation y no deben interpretarse aisladamente como ganancia de producción."
        )
    winner_status = comparison.get("winner_status")
    if winner_status == "statistical_tie_or_inconclusive":
        closest = comparison.get("closest_eligible_challenger_test") or {}
        paragraphs.append(
            "Las pruebas pareadas no confirmaron un ganador estadísticamente superior. El elemento "
            "congelado es el primero según la política lexicográfica y sus salvaguardas, no un "
            "ganador universal; alternativas cercanas siguen siendo compatibles con la evidencia. "
            f"El retador más próximo fue `{closest.get('challenger')}`: diferencia retador−referencia "
            f"{_format_metric(closest.get('difference_challenger_minus_reference'))}, IC 95 % "
            f"[{_format_metric(closest.get('ci_low'))}, {_format_metric(closest.get('ci_high'))}] y "
            f"p de Holm {_format_metric(closest.get('p_value_holm'))}."
        )
    else:
        paragraphs.append(
            "La selección quedó confirmada frente al retador elegible más cercano bajo las pruebas "
            "pareadas y la corrección múltiple registradas por 03_07."
        )
    selected_categories = [
        row for row in category_rows if row.get("candidate_id") == selected_id
    ]
    damage_labels = set(load_taxonomy().damage_labels)
    damage_rows = [
        row for row in selected_categories if row.get("label") in damage_labels
    ]
    if damage_rows:
        strongest = max(damage_rows, key=lambda row: row.get("f1") or -1.0)
        weakest = min(damage_rows, key=lambda row: row.get("f1") or 2.0)
        paragraphs.append(
            f"Por categoría de daño, el mayor F1 del seleccionado aparece en "
            f"`{strongest['label']}` ({_format_metric(strongest.get('f1'))}) y el menor en "
            f"`{weakest['label']}` ({_format_metric(weakest.get('f1'))}). La lectura debe acompañarse "
            "del soporte y AUPRC: una categoría minoritaria puede tener mayor incertidumbre aunque el "
            "promedio global sea competitivo."
        )
    policy = comparison.get("selection_policy", {})
    max_review = _float(policy.get("max_review_rate"))
    margin = _float(policy.get("macro_auprc_noninferiority_margin"))
    operating_load = _float(
        _nested(
            freeze or {},
            "needs_review_policy",
            "validation_operating_point",
            "review_load_rate",
        )
    )
    if max_review is not None:
        detail = (
            f"; el punto elegido usa {_format_metric(operating_load)}"
            if operating_load is not None
            else ""
        )
        paragraphs.append(
            f"La capacidad máxima de revisión declarada fue {max_review:.1%}{detail}. "
            + (
                "Es una carga humana alta y requiere justificar volumen, tiempo y error residual del revisor."
                if max_review >= 0.25
                else "Debe contrastarse con la capacidad humana realmente disponible."
            )
        )
    if margin is not None:
        paragraphs.append(
            f"El margen de no inferioridad macro-AUPRC fue {margin:.3f}. "
            + (
                "Es permisivo para una métrica en escala 0–1; el informe conserva la frontera completa "
                "para que esta decisión operacional no se confunda con equivalencia estadística."
                if margin >= 0.05
                else "La decisión exige que el límite inferior pareado no caiga más allá de ese margen."
            )
        )
    runtime = comparison.get("runtime_optimization") or {}
    if runtime.get("parallel_workers") is not None:
        paragraphs.append(
            f"El artefacto registra {runtime.get('parallel_workers')} hilos efectivos para el "
            f"bootstrap agrupado, con motor `{runtime.get('bootstrap_engine')}`. Este valor observado "
            "prevalece sobre cualquier configuración ejemplificada en documentación anterior."
        )
    if not test_rows:
        paragraphs.append(
            "Test todavía no aparece en los artefactos sincronizados. Por tanto, este informe solo "
            "sustenta selección en validation; aún no estima desempeño final con prevalencia natural."
        )
    else:
        natural = next((row for row in test_rows if row.get("view") == "natural"), None)
        if natural:
            delta_test_ba = (natural.get("balanced_accuracy_any_damage") or 0.0) - (
                selected.get("balanced_accuracy_any_damage_oof") or 0.0
            )
            paragraphs.append(
                f"La apertura única de test natural registró BA ANY_DAMAGE "
                f"{_format_metric(natural.get('balanced_accuracy_any_damage'))}, una diferencia de "
                f"{delta_test_ba:+.4f} frente a validation OOF. Test informa generalización; no debe "
                "usarse para cambiar el modelo o reajustar umbrales."
            )
    return paragraphs


def generate_comparison_report(
    comparison_path: str | Path,
    *,
    freeze_path: str | Path | None = None,
    test_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    generate_figures: bool = True,
    max_models_in_figures: int = 15,
) -> dict[str, Any]:
    """Genera tablas, figuras y un informe crítico reproducible de 03_07."""

    comparison_source = Path(comparison_path).resolve()
    comparison = _load_json(comparison_source)
    signature = comparison.get("comparison_signature")
    if not signature or not comparison.get("ranking"):
        raise ValueError("El artefacto no contiene una comparación 03_07 completa")
    freeze_source = (
        Path(freeze_path).resolve()
        if freeze_path
        else comparison_source.with_name(FREEZE_FILENAME)
    )
    freeze = _load_json(freeze_source) if freeze_source.is_file() else None
    if freeze and freeze.get("comparison_signature") != signature:
        raise ValueError("La selección congelada pertenece a otra comparación")
    test_source = (
        Path(test_path).resolve()
        if test_path
        else comparison_source.with_name(TEST_FILENAME)
    )
    test = _load_json(test_source) if test_source.is_file() else None
    if test and test.get("comparison_signature") != signature:
        raise ValueError("El test pertenece a otra comparación")
    destination = Path(output_dir).resolve() if output_dir else comparison_source.parent
    destination.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy()
    labels = taxonomy.target_labels

    global_rows = _global_validation_rows(comparison)
    category_rows = _category_validation_rows(comparison, labels)
    winners = _category_winners(category_rows, labels)
    bootstrap_rows = _bootstrap_rows(comparison)
    test_global_rows, test_category_rows = _test_tables(test, labels)
    selected_id = str(comparison.get("selected_for_freeze"))
    selected_row = next(
        row for row in global_rows if row.get("candidate_id") == selected_id
    )
    selected_categories = [
        row for row in category_rows if row.get("candidate_id") == selected_id
    ]
    best_by_type_rows = _best_by_model_type_rows(comparison, global_rows)
    rows_by_id = {str(row.get("candidate_id")): row for row in global_rows}
    family_rows = []
    for family, candidate_id in (comparison.get("best_by_family_slot") or {}).items():
        candidate_row = rows_by_id.get(str(candidate_id), {})
        family_rows.append(
            {
                "family_slot": family,
                "candidate_id": candidate_id,
                "balanced_accuracy_any_damage_oof": candidate_row.get(
                    "balanced_accuracy_any_damage_oof"
                ),
                "macro_auprc_damage_oof": candidate_row.get("macro_auprc_damage_oof"),
                "rank": candidate_row.get("rank"),
            }
        )

    table_dir = destination / "tablas_03_07a"
    table_paths = {
        "ranking_validation": table_dir / "ranking_validation.csv",
        "categories_validation": table_dir / "metricas_por_categoria_validation.csv",
        "category_winners": table_dir / "ganadores_por_categoria_validation.csv",
        "best_by_model_type": table_dir / "mejores_por_tipo_validation.csv",
        "paired_bootstrap": table_dir / "comparaciones_bootstrap_pareadas.csv",
    }
    for name, rows in (
        ("ranking_validation", global_rows),
        ("categories_validation", category_rows),
        ("category_winners", winners),
        ("best_by_model_type", best_by_type_rows),
        ("paired_bootstrap", bootstrap_rows),
    ):
        _write_csv(table_paths[name], rows)
    if test_global_rows:
        table_paths["test_global"] = table_dir / "metricas_globales_test.csv"
        table_paths["test_categories"] = table_dir / "metricas_por_categoria_test.csv"
        _write_csv(table_paths["test_global"], test_global_rows)
        _write_csv(table_paths["test_categories"], test_category_rows)

    figures = (
        _generate_figures(
            destination,
            global_rows,
            category_rows,
            selected_id,
            max_models=max_models_in_figures,
        )
        if generate_figures
        else []
    )
    analysis = _critical_analysis(
        comparison, freeze, global_rows, category_rows, test_global_rows
    )
    individual_count = sum(row.get("kind") == "individual" for row in global_rows)
    ensemble_count = sum(row.get("kind") == "ensemble" for row in global_rows)
    policy = comparison.get("selection_policy", {})
    closest_test = comparison.get("closest_eligible_challenger_test") or {}
    lines = [
        "# Reporte de comparación de modelos — 03_07a",
        "",
        f"**Generado:** {datetime.now(UTC).isoformat()}",
        f"**Comparación:** `{comparison_source.name}`",
        f"**SHA-256 del artefacto:** `{sha256_file(comparison_source)}`",
        f"**Firma de comparación:** `{signature}`",
        f"**SHA-256 del dataset:** `{comparison.get('dataset_sha256')}`",
        f"**Split de selección:** `{comparison.get('selection_split', 'validation')}`",
        f"**Estado de test:** `{'evaluado_una_vez' if test else comparison.get('test_status')}`",
        "",
        "## Resumen ejecutivo",
        "",
        f"Se compararon **{individual_count} modelos individuales** y **{ensemble_count} ensembles**. "
        f"La política de selección congeló **`{selected_id}`**; el mejor individuo según la regla fue "
        f"**`{comparison.get('best_individual')}`**. El estado inferencial del líder es "
        f"**`{comparison.get('winner_status')}`**.",
        "",
        _markdown_table(
            [selected_row],
            (
                ("candidate_id", "Seleccionado"),
                ("balanced_accuracy_any_damage_oof", "BA ANY_DAMAGE OOF"),
                ("macro_auprc_damage_oof", "Macro-AUPRC daño OOF"),
                ("fnr_any_damage_oof", "FNR"),
                ("fpr_any_damage_oof", "FPR"),
                ("macro_f1_damage", "Macro-F1 daño"),
                ("ece", "ECE"),
                ("review_load_rate", "Carga revisión"),
            ),
        ),
        "",
        "## Criterio de comparación",
        "",
        f"- Ranking primario: `{policy.get('primary')}`.",
        f"- Agregación: `{policy.get('aggregation')}`.",
        f"- Salvaguarda: `{policy.get('safeguard')}`.",
        f"- Capacidad máxima de revisión: `{policy.get('max_review_rate')}`.",
        f"- Margen de no inferioridad: `{policy.get('macro_auprc_noninferiority_margin')}`.",
        "- Test no interviene en esta selección.",
        "",
        "## Contraste inferencial con el retador elegible más cercano",
        "",
        _markdown_table(
            [closest_test] if closest_test else [],
            (
                ("reference", "Referencia"),
                ("challenger", "Retador"),
                ("difference_challenger_minus_reference", "Δ retador−referencia"),
                ("ci_low", "IC 95 % inferior"),
                ("ci_high", "IC 95 % superior"),
                ("p_value_holm", "p Holm"),
                ("replicates", "Réplicas"),
                ("parallel_workers", "Hilos efectivos"),
            ),
        ),
        "",
        "## Comparación global de todos los modelos",
        "",
        _markdown_table(
            global_rows,
            (
                ("rank", "#"),
                ("candidate_id", "Modelo/ensemble"),
                ("kind", "Tipo"),
                ("model_family", "Familia"),
                ("balanced_accuracy_any_damage_oof", "BA OOF"),
                ("macro_auprc_damage_oof", "Macro-AUPRC OOF"),
                ("fnr_any_damage_oof", "FNR"),
                ("fpr_any_damage_oof", "FPR"),
                ("macro_f1_damage", "Macro-F1 daño"),
                ("micro_f1", "Micro-F1"),
            ),
        ),
        "",
        "La tabla completa, con calibración, riesgo, carga de revisión y salvaguardas, está en "
        "[`tablas_03_07a/ranking_validation.csv`](tablas_03_07a/ranking_validation.csv).",
        "",
        "## Mejor modelo por tipo",
        "",
        "El mejor de cada tipo se toma del mismo ranking lexicográfico de validation usado para "
        "la congelación; no se recalculó un ranking alternativo por una sola métrica.",
        "",
        _markdown_table(
            best_by_type_rows,
            (
                ("model_type", "Tipo"),
                ("candidate_id", "Mejor modelo"),
                ("rank", "Ranking global"),
                ("balanced_accuracy_any_damage_oof", "BA OOF"),
                ("macro_auprc_damage_oof", "Macro-AUPRC OOF"),
                ("fnr_any_damage_oof", "FNR"),
                ("fpr_any_damage_oof", "FPR"),
                ("macro_f1_damage", "Macro-F1 daño"),
                ("selected", "Seleccionado final"),
            ),
        ),
        "",
        "El detalle reproducible está en "
        "[`tablas_03_07a/mejores_por_tipo_validation.csv`](tablas_03_07a/mejores_por_tipo_validation.csv).",
        "",
        "## Mejor representante individual por familia",
        "",
        _markdown_table(
            family_rows,
            (
                ("family_slot", "Familia"),
                ("candidate_id", "Representante"),
                ("rank", "Ranking global"),
                ("balanced_accuracy_any_damage_oof", "BA OOF"),
                ("macro_auprc_damage_oof", "Macro-AUPRC OOF"),
            ),
        ),
        "",
        "## Desempeño del seleccionado por categoría",
        "",
        _markdown_table(
            selected_categories,
            (
                ("label", "Categoría"),
                ("support", "Soporte"),
                ("average_precision", "AUPRC"),
                ("precision", "Precisión"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("ece", "ECE"),
                ("brier", "Brier"),
            ),
        ),
        "",
        "## Mejores resultados por categoría",
        "",
        _markdown_table(
            winners,
            (
                ("label", "Categoría"),
                ("best_average_precision_model", "Mejor AUPRC"),
                ("best_average_precision", "AUPRC"),
                ("best_f1_model", "Mejor F1"),
                ("best_f1", "F1"),
                ("best_recall_model", "Mejor recall"),
                ("best_recall", "Recall"),
                ("precision_at_best_recall", "Precisión asociada"),
            ),
        ),
        "",
        "Los ganadores por categoría se restringen a candidatos que no fallaron la salvaguarda "
        "macro-AUPRC global. Esto evita declarar ganador a un clasificador degenerado que maximiza "
        "recall prediciendo siempre positivo. El CSV detallado conserva los candidatos fallidos.",
        "",
        "El detalle de cada modelo × categoría está en "
        "[`tablas_03_07a/metricas_por_categoria_validation.csv`](tablas_03_07a/metricas_por_categoria_validation.csv).",
        "",
        "## Figuras",
        "",
    ]
    for figure in figures:
        lines.extend(
            [
                f"### {figure.stem.replace('_', ' ').title()}",
                "",
                f"![{figure.stem}](figuras_03_07a/{figure.name})",
                "",
            ]
        )
    if test_global_rows:
        lines.extend(
            [
                "## Apertura única de test",
                "",
                _markdown_table(
                    test_global_rows,
                    (
                        ("view", "Vista"),
                        ("rows", "Filas"),
                        ("balanced_accuracy_any_damage", "BA ANY_DAMAGE"),
                        ("macro_auprc_damage", "Macro-AUPRC daño"),
                        ("fnr_any_damage", "FNR"),
                        ("fpr_any_damage", "FPR"),
                        ("macro_f1_damage", "Macro-F1 daño"),
                        ("ece", "ECE"),
                    ),
                ),
                "",
                "La vista natural es primaria. La vista 4:1 reutiliza las mismas predicciones y es "
                "secundaria; no constituye otra apertura de test.",
                "",
            ]
        )
    lines.extend(["## Análisis crítico", ""])
    lines.extend(f"{index}. {paragraph}" for index, paragraph in enumerate(analysis, 1))
    lines.extend(
        [
            "",
            "## Límites de interpretación",
            "",
            "- La comparación usa chunks agrupados por video; un chunk no representa un video independiente.",
            "- Las métricas por categoría reflejan el corpus y la supervisión disponible, no prevalencia nacional.",
            "- Un score alto no sustituye revisión contextual ni convierte la etiqueta de referencia en verdad humana absoluta.",
            "- Si el líder es estadísticamente inconcluso, debe informarse la selección operacional y la frontera, no una superioridad universal.",
            "- Test, cuando exista, se reporta una sola vez y no autoriza reajustar la selección.",
        ]
    )
    report_path = destination / REPORT_FILENAME
    _write_text_atomic(report_path, "\n".join(lines))
    summary_path = destination / "resumen_03_07a.json"
    summary = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison_signature": signature,
        "dataset_sha256": comparison.get("dataset_sha256"),
        "selected_id": selected_id,
        "best_individual": comparison.get("best_individual"),
        "winner_status": comparison.get("winner_status"),
        "individual_models": individual_count,
        "ensembles": ensemble_count,
        "test_available": bool(test),
        "selected_validation_metrics": selected_row,
        "best_by_family": family_rows,
        "best_by_model_type": best_by_type_rows,
        "selected_category_metrics": selected_categories,
        "critical_analysis": analysis,
        "report_path": _relative_or_absolute(report_path, destination),
        "tables": {
            key: _relative_or_absolute(value, destination)
            for key, value in table_paths.items()
        },
        "figures": [_relative_or_absolute(path, destination) for path in figures],
    }
    write_json_atomic(summary_path, summary)
    return {
        "status": "reported",
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "selected_id": selected_id,
        "winner_status": comparison.get("winner_status"),
        "test_available": bool(test),
        "global_rows": global_rows,
        "selected_category_rows": selected_categories,
        "best_by_model_type_rows": best_by_type_rows,
        "category_winners": winners,
        "table_paths": {key: str(value) for key, value in table_paths.items()},
        "figure_paths": [str(path) for path in figures],
        "critical_analysis": analysis,
    }
