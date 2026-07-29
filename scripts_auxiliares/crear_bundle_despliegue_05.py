"""Construye una carpeta autocontenida para desplegar el moderador 05.

El bundle sólo incorpora los modelos seleccionados con validation. Crea una
base operativa vacía: las inferencias y revisiones locales existentes no se
copian para evitar publicar datos de operación accidentalmente.
"""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys

from scripts_auxiliares import registro_modelos_produccion_4 as registry4
from scripts_auxiliares import servidor_moderacion_05 as moderation05


ROOT = registry4.ROOT
DEFAULT_OUTPUT_DIR = ROOT / "05_frontend_despliegue"
CONSENSUS_MIN_VOTES = 2
RUNTIME_SCRIPTS = (
    "servidor_moderacion_05.py",
    "_experimentos_jerarquicos_clasicos_4_runtime.py",
    "entrenar_qwen_acoso_amenaza.py",
    "entrenar_transformers_gruesos.py",
    "flujo_hibrido_moderador.py",
    "modelos_gruesos_moderador.py",
    "preparar_entrenamiento_ampliado.py",
    "mejoras_modelos_gruesos.py",
)
LOCK_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "tokenizers",
    "yt-dlp",
    "huggingface-hub",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "tqdm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_output(path: Path) -> Path:
    value = Path(path).resolve()
    if value.parent != ROOT.resolve() or value.name != "05_frontend_despliegue":
        raise ValueError(
            "Por seguridad el bundle sólo puede generarse como "
            f"{DEFAULT_OUTPUT_DIR}."
        )
    return value


def _copy_file(source: Path, destination: Path) -> None:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Falta el asset requerido: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def _registry_artifacts(registry: dict):
    seen = set()
    for model in registry["models"].values():
        for artifact in model["artifacts"].values():
            values = artifact if isinstance(artifact, list) else [artifact]
            for item in values:
                if item is None or item["path"] in seen:
                    continue
                seen.add(item["path"])
                yield item


def _validate_final_audit(registry: dict) -> tuple[Path, dict]:
    audit_relative = Path(registry.get("audit_result", ""))
    audit_path = ROOT / audit_relative
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Falta la auditoría final de 04_208. Ejecute 04_207, luego 04_208 "
            "y finalmente vuelva a crear el bundle con 05."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("dataset_sha256") != registry["dataset_sha256"]:
        raise ValueError("04_208 auditó un dataset distinto al registro de 05.")
    if audit.get("missing_results"):
        raise ValueError(
            "04_208 todavía declara resultados pendientes: "
            + ", ".join(audit["missing_results"])
        )
    analyzed = {item["key"]: item for item in audit.get("models_analyzed", [])}
    expected_keys = {
        f"classic__{registry['models']['classical']['model_key'].replace('__', '__')}",
        f"transformer_flat__{registry['models']['transformer']['model_key']}",
        registry["models"]["qwen"]["model_key"],
    }
    missing_models = sorted(expected_keys - set(analyzed))
    if missing_models:
        raise ValueError(f"04_208 no auditó los modelos desplegados: {missing_models}")
    qwen_provenance = analyzed[registry["models"]["qwen"]["model_key"]].get(
        "checkpoint_provenance"
    ) or {}
    selection = registry["models"]["qwen"]["artifacts"]["selection"]
    if (qwen_provenance.get("selection_artifact") or {}).get("sha256") != selection["sha256"]:
        raise ValueError(
            "04_208 corresponde a otra selección Qwen. Reejecútelo después de 04_207."
        )
    return audit_path, audit


def _qwen_snapshot(registry: dict, *, download_if_needed: bool) -> Path:
    from huggingface_hub import snapshot_download

    spec = registry["models"]["qwen"]["model_spec"]
    arguments = {
        "repo_id": spec["model_id"],
        "revision": spec["revision"],
    }
    try:
        return Path(snapshot_download(**arguments, local_files_only=True))
    except Exception as local_error:
        if not download_if_needed:
            raise FileNotFoundError(
                "Qwen base no está en el caché local. Active "
                "DESCARGAR_QWEN_BASE_SI_FALTA o descárguelo antes."
            ) from local_error
        return Path(snapshot_download(**arguments))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def _dependency_lock() -> str:
    rows = ["--extra-index-url https://download.pytorch.org/whl/cpu"]
    for package in LOCK_PACKAGES:
        try:
            rows.append(f"{package}=={version(package)}")
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"No se puede fijar el bundle: falta la dependencia {package}."
            ) from error
    return "\n".join(rows)


def _generated_files() -> dict[str, str]:
    return {
        "app.py": r'''
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PLN_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("PLN_OPERATION_DIR", str(ROOT / "data"))
os.environ.setdefault("QWEN_BASE_MODEL_PATH", str(ROOT / "modelos" / "qwen3_06b_lora_acoso_amenaza_4" / "base_model"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("WEB_CONCURRENCY", "2"))

from scripts_auxiliares import servidor_moderacion_05 as moderation05

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
ACCESS_USER = os.getenv("MODERATOR_ACCESS_USER", "").strip()
ACCESS_PASSWORD = os.getenv("MODERATOR_ACCESS_PASSWORD", "").strip()

service = moderation05.ModerationService()
server = moderation05.start_server(
    service,
    host=HOST,
    port=PORT,
    open_browser=False,
    allow_network=True,
    auth_user=ACCESS_USER or None,
    auth_password=ACCESS_PASSWORD or None,
)
print(f"Moderador 05 disponible en {server.url}", flush=True)
try:
    server.thread.join()
except KeyboardInterrupt:
    server.stop()
''',
        "requirements.txt": r'''
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.5,<3
transformers>=4.51,<6
peft>=0.15,<1
safetensors>=0.5
tokenizers>=0.21
yt-dlp>=2025.6
huggingface-hub>=0.30
numpy>=1.26,<3
pandas>=2.2,<3
scipy>=1.12,<2
scikit-learn>=1.4,<2
joblib>=1.3,<2
matplotlib>=3.8,<4
tqdm>=4.66,<5
''',
        "Dockerfile": r'''
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLN_PROJECT_ROOT=/app \
    PLN_OPERATION_DIR=/app/data \
    QWEN_BASE_MODEL_PATH=/app/modelos/qwen3_06b_lora_acoso_amenaza_4/base_model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=7860

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-lock.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements-lock.txt
COPY . .
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1
CMD ["python", "app.py"]
''',
        "docker-compose.yml": r'''
services:
  moderador:
    build: .
    ports:
      - "8765:7860"
    environment:
      MODERATOR_ACCESS_USER: ${MODERATOR_ACCESS_USER:-}
      MODERATOR_ACCESS_PASSWORD: ${MODERATOR_ACCESS_PASSWORD:-}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
''',
        ".env.example": r'''
MODERATOR_ACCESS_USER=moderador
MODERATOR_ACCESS_PASSWORD=cambie-esta-clave-antes-de-publicar
''',
        ".gitattributes": r'''
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
''',
        ".dockerignore": r'''
__pycache__/
*.py[cod]
.git/
.env
data/*.sqlite3-wal
data/*.sqlite3-shm
data/*.jsonl
''',
        "README.md": r'''
---
title: Moderador local de cuatro daños
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# Moderador 05 · bundle desplegable

Aplicación web autocontenida con los tres modelos elegidos usando exclusivamente
validation: ML clásico, E5-small y Qwen3-0.6B LoRA. El consenso es mayoritario:
una categoría se activa con **al menos 2 votos de 3**, sin exigir unanimidad.
`requirements-lock.txt` fija las versiones exactas usadas al construirlo.

Consulte [`GUIA_DESPLIEGUE.md`](GUIA_DESPLIEGUE.md) para el procedimiento
completo en Hugging Face Spaces, Python local y Docker local.

## Inicio local

Con Python 3.11:

```bash
python -m pip install -r requirements.txt
python app.py
```

Con Docker:

```bash
docker compose up --build
```

Abra `http://localhost:8765`. Para publicación externa, copie `.env.example`
como `.env`, cambie la contraseña y coloque el servicio detrás de HTTPS.

## Persistencia

`data/estadisticas_moderacion.sqlite3` empieza vacío deliberadamente. Monte
`data/` como volumen persistente para conservar inferencias, revisiones y los
JSONL de reentrenamiento. El bundle nunca contiene estadísticas tomadas del
entorno donde fue construido.

## Hugging Face Spaces

Esta carpeta ya contiene el encabezado y el `Dockerfile` de un Docker Space.
Los archivos grandes deben subirse con Git LFS/Xet. El almacenamiento normal
del contenedor puede ser efímero; conecte un bucket/volumen o una base externa
si las revisiones deben sobrevivir reinicios.

## Límites

Qwen y E5 funcionan en CPU, pero el primer análisis puede tardar mientras se
cargan los pesos. El sistema es apoyo a revisión humana y no debe tomar medidas
autónomas de sanción o bloqueo.
''',
        "GUIA_DESPLIEGUE.md": r'''
# Cómo levantar el moderador 05

## Opción A · Local con Docker (recomendada para comprobar el bundle)

Requisitos: Docker Desktop o Docker Engine con Compose y al menos 6 GB de RAM
disponible.

1. Abra una terminal dentro de `05_frontend_despliegue`.
2. Copie `.env.example` como `.env` y cambie usuario y contraseña.
3. Ejecute:

   ```bash
   docker compose up --build
   ```

4. Abra `http://localhost:8765` y use las credenciales de `.env`.
5. Detenga el servicio con `Ctrl+C` o `docker compose down`.

La carpeta `data/` está montada como volumen y conserva SQLite y los JSONL si
se vuelve a crear el contenedor.

## Opción B · Local sin Docker

Requisitos: Python 3.11 de 64 bits y aproximadamente 6 GB de RAM disponible.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:MODERATOR_ACCESS_USER='moderador'
$env:MODERATOR_ACCESS_PASSWORD='una-clave-larga'
python app.py
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MODERATOR_ACCESS_USER=moderador
export MODERATOR_ACCESS_PASSWORD='una-clave-larga'
python app.py
```

Abra `http://127.0.0.1:7860`. Aunque `app.py` escucha en todas las interfaces
para ser portable a la nube, no abra el puerto del equipo a Internet sin HTTPS.

## Opción C · Hugging Face Docker Space

El bundle ya contiene `README.md` con `sdk: docker`, `app_port: 7860`, el
`Dockerfile` y los pesos. La aplicación no descarga modelos durante inferencia.

1. Cree una cuenta y un token de escritura en Hugging Face.
2. Instale/autentique la CLI:

   ```bash
   python -m pip install --upgrade huggingface_hub
   hf auth login
   ```

3. Cree un Space Docker; reemplace `USUARIO`:

   ```bash
   hf repos create USUARIO/moderador-05 --repo-type space --space-sdk docker
   ```

4. Desde el directorio padre del bundle, suba la carpeta completa:

   ```bash
   hf upload USUARIO/moderador-05 ./05_frontend_despliegue . --repo-type space
   ```

5. En `Settings → Variables and secrets` del Space, agregue:

   - `MODERATOR_ACCESS_USER` como variable;
   - `MODERATOR_ACCESS_PASSWORD` como secreto.

6. Espere el build y abra la URL `https://USUARIO-moderador-05.hf.space`.

Los archivos del bundle suman alrededor de 1.7 GiB; la primera subida y el
primer build pueden tardar. El filesystem normal de Spaces es efímero: para no
perder las revisiones, conecte `data/` a un bucket/volumen escribible o adapte
la persistencia a una base externa. No publique una app de moderación con la
contraseña vacía.

Actualmente un Space Docker con cómputo puede requerir un plan de cuenta aunque
CPU Basic no tenga costo horario. Confirme la condición vigente antes de crear
el Space: <https://huggingface.co/docs/hub/spaces-overview>.

## Opción D · VM persistente, incluida OCI Always Free

Para conservar SQLite sin adaptar el código a otra base, una VM con disco
persistente es más apropiada. OCI ofrece recursos Always Free en la región
principal, sujetos a capacidad disponible. Cree una VM Ampere A1 con Ubuntu,
instale Docker, copie esta carpeta y ejecute `docker compose up -d --build`.
Abra únicamente 80/443, coloque Caddy o Nginx delante de `8765` para HTTPS y no
publique el puerto de la aplicación sin protección. Al ser ARM64, confirme que
las versiones fijadas en `requirements-lock.txt` tengan wheel para la imagen
elegida antes de usarla como servicio estable.

## Comprobaciones después del despliegue

1. `GET /api/health` debe responder `status: ok`.
2. Pruebe una frase neutral y una frase de alerta con cada modelo.
3. Compruebe que consenso activa una categoría con dos votos, sin exigir tres.
4. Registre una revisión humana y verifique que aumenta la estadística.
5. Reinicie el contenedor y confirme que la revisión persiste.

El análisis de YouTube requiere salida HTTPS hacia YouTube y puede fallar si la
plataforma bloquea la IP del proveedor. Los videos sin subtítulos se rechazan.
''',
        "MODELOS_Y_LICENCIAS.md": r'''
# Modelos incorporados

- `intfloat/multilingual-e5-small`, revisión fijada en el registro desplegable.
- `Qwen/Qwen3-0.6B-Base`, revisión fijada en el registro, más el adaptador LoRA
  entrenado por el proyecto.
- SVM lineal calibrado y vectorizador entrenados por el proyecto.

Antes de redistribuir públicamente el bundle, verifique en las páginas de los
modelos las licencias y condiciones vigentes. El manifiesto registra hashes,
rutas y revisiones para mantener trazabilidad.
''',
    }


def _write_manifest(staging: Path, registry: dict, qwen_source: Path) -> dict:
    files = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative == "bundle_manifest.json":
            continue
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": str(ROOT),
        "dataset_sha256": registry["dataset_sha256"],
        "selection_partition": registry["global_selection_partition"],
        "test_used_for_model_selection": registry["test_used_for_model_selection"],
        "consensus": {
            "minimum_votes": CONSENSUS_MIN_VOTES,
            "total_models": 3,
            "unanimity_required": False,
        },
        "qwen_base_source": str(qwen_source),
        "privacy": "fresh empty operational database; no prior usage copied",
        "reproducibility": {
            "functional_rebuild": True,
            "artifact_hashes_recorded": True,
            "dependency_versions_locked": True,
            "bit_for_bit_identical": False,
            "variable_fields": [
                "generated_at",
                "source_root",
                "empty SQLite internal metadata",
            ],
        },
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    (staging / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_deployment_bundle(
    registry: dict | None = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    recreate: bool = True,
    download_qwen_base_if_needed: bool = True,
) -> dict:
    """Crea el bundle completo y devuelve su manifiesto."""
    registry = registry or registry4.load_registry(verify_hashes=True)
    if registry["comparison_mode"]["consensus_rule"] != (
        "category accepted when at least two of three models activate it"
    ):
        raise ValueError("El registro no declara el consenso 2 de 3 esperado.")
    audit_path, audit = _validate_final_audit(registry)
    target = _safe_output(output_dir)
    staging = ROOT / ".05_frontend_despliegue.staging"
    if staging.exists():
        shutil.rmtree(staging)
    if target.exists():
        if not recreate:
            raise FileExistsError(f"El bundle ya existe: {target}")
        if not (target / "bundle_manifest.json").is_file():
            raise RuntimeError(
                f"Se rehúsa borrar {target}: no parece un bundle generado por 05."
            )
        shutil.rmtree(target)
    staging.mkdir(parents=True)

    try:
        _copy_file(
            ROOT / "Cuadernos" / "frontend" / "produccion_moderador.html",
            staging / "Cuadernos" / "frontend" / "produccion_moderador.html",
        )
        script_target = staging / "scripts_auxiliares"
        script_target.mkdir(parents=True)
        _write_text(script_target / "__init__.py", "")
        for name in RUNTIME_SCRIPTS:
            _copy_file(ROOT / "scripts_auxiliares" / name, script_target / name)

        registry_path = registry4.REGISTRY_PATH
        _copy_file(registry_path, staging / registry_path.relative_to(ROOT))
        for artifact in _registry_artifacts(registry):
            relative = Path(artifact["path"])
            source = ROOT / relative
            if _sha256(source) != artifact["sha256"]:
                raise RuntimeError(f"Hash inesperado antes de copiar {relative}.")
            _copy_file(source, staging / relative)
        audit_relative = audit_path.relative_to(ROOT)
        _copy_file(audit_path, staging / audit_relative)

        qwen_source = _qwen_snapshot(
            registry, download_if_needed=download_qwen_base_if_needed
        )
        qwen_target = (
            staging
            / "modelos"
            / "qwen3_06b_lora_acoso_amenaza_4"
            / "base_model"
        )
        for source in sorted(item for item in qwen_source.rglob("*") if item.is_file()):
            _copy_file(source, qwen_target / source.relative_to(qwen_source))
        for required in ("config.json", "model.safetensors"):
            if not (qwen_target / required).is_file():
                raise FileNotFoundError(f"Qwen base incompleto: falta {required}.")

        generated = _generated_files()
        generated["requirements-lock.txt"] = _dependency_lock()
        generated["build_environment.json"] = json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    package: version(package) for package in LOCK_PACKAGES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        for relative, value in generated.items():
            _write_text(staging / relative, value)

        data_dir = staging / "data"
        moderation05.OperationStore(
            data_dir / "estadisticas_moderacion.sqlite3",
            data_dir / "revisiones_para_reentrenamiento.jsonl",
            data_dir / "revisiones_adjudicadas_unicas.jsonl",
        )
        _write_text(data_dir / "revisiones_para_reentrenamiento.jsonl", "")
        _write_text(data_dir / "revisiones_adjudicadas_unicas.jsonl", "")
        manifest = _write_manifest(staging, registry, qwen_source)
        manifest["final_audit"] = {
            "path": audit_relative.as_posix(),
            "completed_at": audit.get("completed_at"),
            "sha256": _sha256(audit_path),
            "missing_results": audit.get("missing_results", []),
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.rename(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        **manifest,
        "output_dir": str(target),
        "total_gib": manifest["total_bytes"] / (1024**3),
        "models": {
            slot: value["label"] for slot, value in registry["models"].items()
        },
    }


if __name__ == "__main__":
    result = build_deployment_bundle()
    print(json.dumps(result, ensure_ascii=False, indent=2))
