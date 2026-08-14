"""Genera los cuadernos orquestadores activos sin resultados obsoletos incrustados."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nbformat as nbf
from notebook_references import apply_citations

ROOT = Path(__file__).resolve().parents[1]
ONLY_NOTEBOOKS: set[str] | None = None
_COLAB_BUNDLE_IDENTITY: dict[str, str] | None = None
OPERATIONAL_PROMPT_RELATIVE = "config/prompt_operacional_ollama_v3_2.md"
QWEN_A100_NOTEBOOKS = {"02_02", "03_05", "03_06", "03_06b"}
CPU_COLAB_NOTEBOOKS = {"03_07"}

PROJECT_TITLE = (
    "Moderación semiautomática de videos peruanos de YouTube mediante modelos "
    "clásicos y neuronales de procesamiento del lenguaje natural"
)
PROJECT_AUTHORS = [
    "Luis Enrique Koc Góngora",
    "Alex Felipe Mancilla Antay",
    "Herbert Antonio Meléndez García",
    "Dennis Jack Paitán Cano",
]
PROJECT_COVER = (
    f"# {PROJECT_TITLE}\n\n"
    "**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en "
    "Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**\n\n"
    f"**Grupo 4:** {', '.join(PROJECT_AUTHORS[:-1])} y {PROJECT_AUTHORS[-1]}\n\n"
    "---"
)
CONTRACT_SUMMARY = (
    "**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, "
    "`RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y "
    "`CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son "
    "multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran "
    "al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son "
    "decisiones operativas locales."
)

LABELING_FRONTEND_GUIDE = r"""Ejecute los comandos siguientes en una terminal **PowerShell** abierta en la raíz del repositorio, no dentro de una celda Python. Cada bloque incluye un botón **Copiar**; la celda Python que aparece después solo presenta los comandos y no los ejecuta.

### Preparación inicial

Este bloque solo es necesario la primera vez (o si se eliminó `.venv`):

<div style="margin:8px 0 18px">
<button type="button" onclick="const code=this.nextElementSibling.innerText.trim(); navigator.clipboard.writeText(code); this.textContent='Copiado'; setTimeout(() => { this.textContent='Copiar'; }, 1200);" style="float:right;margin:6px;border:1px solid #94a3b8;background:#fff;border-radius:6px;padding:5px 10px;cursor:pointer;font-weight:600">Copiar</button>
<pre style="clear:both;overflow-x:auto"><code>
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[datos,etiquetado,cuadernos,dev]"
</code></pre>
</div>

### Inicio del frontend

<div style="margin:8px 0 18px">
<button type="button" onclick="const code=this.nextElementSibling.innerText.trim(); navigator.clipboard.writeText(code); this.textContent='Copiado'; setTimeout(() => { this.textContent='Copiar'; }, 1200);" style="float:right;margin:6px;border:1px solid #94a3b8;background:#fff;border-radius:6px;padding:5px 10px;cursor:pointer;font-weight:600">Copiar</button>
<pre style="clear:both;overflow-x:auto"><code>
.\.venv\Scripts\modperu.exe serve-labeling `
  --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
</code></pre>
</div>

Mantenga esa terminal abierta y visite <http://127.0.0.1:8765>. La interfaz empieza en **Requieren acción**, que contiene únicamente chunks pendientes o diferidos. **Todos los chunks** recorre realmente la campaña completa, incluidos casos resueltos y excluidos. También puede usar **Urgentes**, **Prioritarios Pro** y **Excluidos**; esta última vista permite reclasificar chunks fuera del dataset entrenable.

Para detener el servidor, vuelva a la terminal y presione `Ctrl+C`. En ejecuciones posteriores basta con repetir el bloque de inicio. Si se actualizó el código del servidor, deténgalo y vuelva a iniciar: recargar el navegador por sí solo no activa esos cambios."""


SETUP = """from pathlib import Path
import sys

def find_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    raise FileNotFoundError('No se encontró pyproject.toml')

ROOT = find_root()
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from moderacion_peru.notebook_ui import notebook_progress, run_with_progress, show_callout, show_command, show_result, show_summary, show_table
OPERATIONAL_PROMPT=ROOT/'config/prompt_operacional_ollama_v3_2.md'
if not OPERATIONAL_PROMPT.is_file():
    raise FileNotFoundError(f'Falta el prompt operacional vigente: {OPERATIONAL_PROMPT}')
show_summary('Entorno del proyecto', {'raíz': ROOT, 'backend': 'local', 'prompt_operacional': OPERATIONAL_PROMPT}, tone='success')
"""


COLAB_PUBLISHER_SETUP = """# Este cuaderno se ejecuta en Google Colab y no requiere Google Cloud Console.
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
import urllib.parse
import urllib.request
import uuid

from IPython.display import Markdown, display
from tqdm.auto import tqdm

if importlib.util.find_spec("google.colab") is None:
    raise RuntimeError(
        "02_00 debe abrirse y ejecutarse en Google Colab. La carga local usa el selector del navegador."
    )

COLAB_EXPECTED_CORE_SHA256 = "__EXPECTED_CORE_SHA256__"
COLAB_NOTEBOOK_BUILD_BUNDLE_ID = "__EXPECTED_BUNDLE_ID__"

def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

def show_result(title, value, tone="success"):
    display(Markdown(f"### {title}\\n\\n```json\\n{_json_text(value)}\\n```"))

def show_summary(title, value, tone="neutral"):
    show_result(title, value, tone=tone)

def show_callout(title, message, tone="neutral"):
    display(Markdown(f"> **{title}.** {message}"))

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _bundle_id_for_manifest(manifest):
    core = manifest["core"]
    inputs = manifest["inputs"]
    identity = {
        "schema_version": manifest["schema_version"],
        "taxonomy_contract": manifest["taxonomy_contract"],
        "taxonomy_version": manifest["taxonomy_version"],
        "core": {"name": core["name"], "sha256": core["sha256"]},
        "inputs": {
            key: {
                "archive": value["archive"],
                "archive_sha256": value["archive_sha256"],
                "source_sha256": value["source_sha256"],
            }
            for key, value in sorted(inputs.items())
        },
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _bundle_specs(manifest):
    specs = [(manifest["core"]["name"], manifest["core"]["sha256"])]
    specs.extend(
        (entry["archive"], entry["archive_sha256"])
        for entry in manifest.get("inputs", {}).values()
    )
    for name, expected_sha256 in specs:
        if Path(name).name != name or len(str(expected_sha256)) != 64:
            raise ValueError(f"Entrada insegura o incompleta en bundle_manifest.json: {name!r}")
    return [(str(name), str(expected_sha256)) for name, expected_sha256 in specs]

def _verify_bundle(directory, expected_bundle_id=None):
    bundle_dir = Path(directory)
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta {manifest_path}")
    manifest = _read_json(manifest_path)
    computed = _bundle_id_for_manifest(manifest)
    if manifest.get("bundle_id") != computed:
        raise ValueError("bundle_manifest.json no contiene un bundle_id válido")
    if expected_bundle_id is not None and computed != expected_bundle_id:
        raise ValueError(f"Bundle inesperado: esperado={expected_bundle_id}, obtenido={computed}")
    if manifest["core"]["sha256"] != COLAB_EXPECTED_CORE_SHA256:
        raise RuntimeError(
            "El core del bundle no coincide con este 02_00. Descargue el cuaderno y el bundle de la misma revisión."
        )
    for name, expected_sha256 in _bundle_specs(manifest):
        artifact = bundle_dir / name
        if not artifact.is_file():
            raise FileNotFoundError(f"Falta el artefacto declarado {artifact}")
        actual = _sha256(artifact)
        if actual != expected_sha256:
            raise ValueError(f"SHA-256 inválido para {name}: esperado={expected_sha256}, obtenido={actual}")
    return manifest

show_summary(
    "Entorno publicador",
    {
        "backend": "Google Colab",
        "Google Cloud Console": "no requerido",
        "Drive Desktop": "no requerido",
        "core esperado": COLAB_EXPECTED_CORE_SHA256,
        "bundle al generar el cuaderno": COLAB_NOTEBOOK_BUILD_BUNDLE_ID,
    },
    tone="success",
)
"""


COLAB_SETUP = """# Backend reproducible: local o Google Colab desde VS Code
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
import zipfile

COLAB_NOTEBOOK_ID = "__NOTEBOOK_ID__"
COLAB_DRIVE_FOLDER = "ModeracionPeru_Colab"  # Debe coincidir con config/colab_l4.json
COLAB_RUN_ID = ""  # Vacío reanuda <notebook>_working_v2_1; use otro ID para otro experimento
COLAB_REQUIRE_L4 = __REQUIRE_L4__
COLAB_AUTO_UPDATE_BUNDLE = True
COLAB_AUTO_PUBLISH_MISSING_BUNDLE = True
COLAB_BUNDLE_SOURCE = "github"  # "github" o "local_upload"
COLAB_GITHUB_REPOSITORY = "lkoc/Trabajo_PLN-MIA-Grupo4"
COLAB_GITHUB_REF = "main"
COLAB_GITHUB_BUNDLE_PATH = "resultados/colab_bundle"
COLAB_NOTEBOOK_BUILD_BUNDLE_ID = "__EXPECTED_BUNDLE_ID__"  # Trazabilidad al generar el notebook
COLAB_EXPECTED_CORE_SHA256 = "__EXPECTED_CORE_SHA256__"
IN_COLAB = importlib.util.find_spec("google.colab") is not None
COLAB_CONTEXT = None

# Los modelos configurados son públicos. Evita que huggingface_hub intente
# consultar el vault de secretos, que solo funciona desde la interfaz web de Colab.
if IN_COLAB:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HOME"] = "/content/huggingface"

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def _find_local_root(start=Path.cwd()):
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("No se encontró pyproject.toml")

def _read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _bundle_id_for_manifest(manifest):
    core = manifest["core"]
    inputs = manifest["inputs"]
    identity = {
        "schema_version": manifest["schema_version"],
        "taxonomy_contract": manifest["taxonomy_contract"],
        "taxonomy_version": manifest["taxonomy_version"],
        "core": {"name": core["name"], "sha256": core["sha256"]},
        "inputs": {
            key: {
                "archive": value["archive"],
                "archive_sha256": value["archive_sha256"],
                "source_sha256": value["source_sha256"],
            }
            for key, value in sorted(inputs.items())
        },
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _bundle_specs(manifest):
    specs = [(manifest["core"]["name"], manifest["core"]["sha256"])]
    specs.extend(
        (entry["archive"], entry["archive_sha256"])
        for entry in manifest.get("inputs", {}).values()
    )
    for name, expected_sha256 in specs:
        if Path(name).name != name or not expected_sha256:
            raise ValueError(f"Entrada insegura o incompleta en bundle_manifest.json: {name!r}")
    return specs

def _verify_expected_bundle(bundle_dir, expected_bundle_id=COLAB_NOTEBOOK_BUILD_BUNDLE_ID):
    manifest_path = Path(bundle_dir) / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta {manifest_path}")
    manifest = _read_manifest(manifest_path)
    computed_bundle_id = _bundle_id_for_manifest(manifest)
    if manifest.get("bundle_id") != computed_bundle_id:
        raise ValueError("bundle_manifest.json no contiene una identidad válida")
    if computed_bundle_id != expected_bundle_id:
        raise ValueError(
            f"Bundle inesperado: esperado={expected_bundle_id}, obtenido={computed_bundle_id}"
        )
    if manifest["core"]["sha256"] != COLAB_EXPECTED_CORE_SHA256:
        raise ValueError("El core del bundle no coincide con el fijado por este cuaderno")
    for name, expected_sha256 in _bundle_specs(manifest):
        artifact = Path(bundle_dir) / name
        if not artifact.is_file() or _sha256(artifact) != expected_sha256:
            raise ValueError(f"Artefacto ausente o inválido: {artifact}")
    return manifest

def _bundle_is_current(bundle_dir, manifest_path, expected_bundle_id):
    if Path(manifest_path) != Path(bundle_dir) / "bundle_manifest.json":
        return False
    try:
        _verify_expected_bundle(bundle_dir, expected_bundle_id)
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

def _download_bundle_file(url, destination):
    destination = Path(destination)
    partial = destination.with_name(f".{destination.name}.partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ModeracionPeru-Colab-Bundle/2.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()

def _prepare_bundle_staging():
    staging = Path("/content/moderacion_peru_bundle_source")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging

def _uploaded_bundle_member(uploaded, expected_name):
    # Resuelve el nombre exacto o el sufijo (N) que agrega Colab al repetir una carga.
    if expected_name in uploaded:
        return expected_name
    suffix = Path(expected_name).suffix
    base = expected_name[:-len(suffix)] if suffix else expected_name
    prefix = f"{base} ("
    ending = f"){suffix}"
    candidates = []
    for actual_name in uploaded:
        if not actual_name.startswith(prefix) or not actual_name.endswith(ending):
            continue
        duplicate_number = actual_name[len(prefix):-len(ending)]
        if duplicate_number.isdigit():
            candidates.append(actual_name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"La selección contiene varias copias de {expected_name}: {sorted(candidates)}"
        )
    return None

def _acquire_expected_bundle():
    staging = _prepare_bundle_staging()
    if COLAB_BUNDLE_SOURCE == "github":
        encoded_ref = urllib.parse.quote(COLAB_GITHUB_REF, safe="")
        base = (
            f"https://raw.githubusercontent.com/{COLAB_GITHUB_REPOSITORY}/"
            f"{encoded_ref}/{COLAB_GITHUB_BUNDLE_PATH}"
        )
        cache_key = urllib.parse.quote(COLAB_NOTEBOOK_BUILD_BUNDLE_ID, safe="")
        manifest_path = staging / "bundle_manifest.json"
        _download_bundle_file(
            f"{base}/bundle_manifest.json?bundle_id={cache_key}", manifest_path
        )
        manifest = _read_manifest(manifest_path)
        if manifest.get("bundle_id") != _bundle_id_for_manifest(manifest):
            raise ValueError("El manifiesto descargado desde GitHub no es válido")
        if manifest["bundle_id"] != COLAB_NOTEBOOK_BUILD_BUNDLE_ID:
            raise RuntimeError(
                "GitHub todavía no contiene el bundle fijado por este cuaderno. "
                "Sincronice resultados/colab_bundle o use COLAB_BUNDLE_SOURCE='local_upload'."
            )
        if manifest["core"]["sha256"] != COLAB_EXPECTED_CORE_SHA256:
            raise RuntimeError("GitHub contiene un project_core.zip distinto al esperado")
        for name, _ in _bundle_specs(manifest):
            encoded_name = urllib.parse.quote(name, safe="")
            _download_bundle_file(
                f"{base}/{encoded_name}?bundle_id={cache_key}", staging / name
            )
    elif COLAB_BUNDLE_SOURCE == "local_upload":
        from google.colab import files

        uploaded = files.upload()
        manifest_upload = _uploaded_bundle_member(uploaded, "bundle_manifest.json")
        if manifest_upload is None:
            raise FileNotFoundError("La selección no incluyó bundle_manifest.json")
        (staging / "bundle_manifest.json").write_bytes(uploaded[manifest_upload])
        manifest = _read_manifest(staging / "bundle_manifest.json")
        if manifest.get("bundle_id") != COLAB_NOTEBOOK_BUILD_BUNDLE_ID:
            raise RuntimeError("Los archivos seleccionados no pertenecen al bundle esperado")
        required = {"bundle_manifest.json", *(name for name, _ in _bundle_specs(manifest))}
        resolved = {
            name: _uploaded_bundle_member(uploaded, name)
            for name in required - {"bundle_manifest.json"}
        }
        missing = sorted(name for name, actual_name in resolved.items() if actual_name is None)
        if missing:
            raise FileNotFoundError(f"Faltaron archivos del bundle: {missing}")
        for name, actual_name in resolved.items():
            (staging / name).write_bytes(uploaded[actual_name])
    else:
        raise ValueError("COLAB_BUNDLE_SOURCE debe ser 'github' o 'local_upload'")
    return staging, _verify_expected_bundle(staging)

def _write_latest_pointer(releases_dir, release_dir, manifest):
    pointer = {
        "schema_version": "1.0.0",
        "bundle_id": manifest["bundle_id"],
        "core_sha256": manifest["core"]["sha256"],
        "manifest_sha256": _sha256(Path(release_dir) / "bundle_manifest.json"),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_path = Path(releases_dir) / "latest.json"
    partial = Path(releases_dir) / f".latest-{uuid.uuid4().hex}.json"
    partial.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    os.replace(partial, latest_path)
    return pointer

def _publish_expected_bundle(staging, releases_dir):
    manifest = _verify_expected_bundle(staging)
    releases_dir = Path(releases_dir)
    releases_dir.mkdir(parents=True, exist_ok=True)
    release_dir = releases_dir / COLAB_NOTEBOOK_BUILD_BUNDLE_ID
    if release_dir.exists():
        _verify_expected_bundle(release_dir)
        release_status = "already_present_and_verified"
    else:
        partial = releases_dir / f".{COLAB_NOTEBOOK_BUILD_BUNDLE_ID}.partial-{uuid.uuid4().hex}"
        partial.mkdir()
        try:
            for name, _ in _bundle_specs(manifest):
                shutil.copyfile(Path(staging) / name, partial / name)
            shutil.copyfile(
                Path(staging) / "bundle_manifest.json",
                partial / "bundle_manifest.json",
            )
            _verify_expected_bundle(partial)
            os.replace(partial, release_dir)
        finally:
            if partial.exists():
                shutil.rmtree(partial)
        release_status = "auto_published_and_verified"
    pointer = _write_latest_pointer(releases_dir, release_dir, manifest)
    return {
        "status": release_status,
        "release_dir": release_dir,
        "latest_pointer": pointer,
    }

def _ensure_expected_drive_release(releases_dir):
    release_dir = Path(releases_dir) / COLAB_NOTEBOOK_BUILD_BUNDLE_ID
    if _bundle_is_current(
        release_dir,
        release_dir / "bundle_manifest.json",
        COLAB_NOTEBOOK_BUILD_BUNDLE_ID,
    ):
        return {"status": "already_present_and_verified", "release_dir": release_dir}
    if not COLAB_AUTO_PUBLISH_MISSING_BUNDLE:
        raise RuntimeError(
            "Drive no contiene el release esperado y COLAB_AUTO_PUBLISH_MISSING_BUNDLE=False"
        )
    staging, _ = _acquire_expected_bundle()
    return _publish_expected_bundle(staging, releases_dir)

def _activate_verified_drive_release(release_dir, bundle_dir, expected_bundle_id):
    release_manifest_path = release_dir / "bundle_manifest.json"
    if not _bundle_is_current(release_dir, release_manifest_path, expected_bundle_id):
        raise RuntimeError(
            "La versión esperada no está completa o no coincide con sus SHA-256: " + str(release_dir)
        )
    manifest = _read_manifest(release_manifest_path)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Todos los artefactos se validaron antes; el manifiesto activo se reemplaza al final.
    for name, _ in _bundle_specs(manifest):
        partial = bundle_dir / f".{name}.partial"
        shutil.copyfile(release_dir / name, partial)
        os.replace(partial, bundle_dir / name)
    partial_manifest = bundle_dir / ".bundle_manifest.json.partial"
    shutil.copyfile(release_manifest_path, partial_manifest)
    os.replace(partial_manifest, bundle_dir / "bundle_manifest.json")
    if not _bundle_is_current(bundle_dir, bundle_dir / "bundle_manifest.json", expected_bundle_id):
        raise RuntimeError("La activación desde bundle_releases no superó la verificación final")
    return manifest

if IN_COLAB:
    from google.colab import drive

    # La extensión oficial de Colab para VS Code admite drive.mount desde v0.2.1.
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive") / COLAB_DRIVE_FOLDER
    BUNDLE_DIR = DRIVE_ROOT / "bundle"
    RELEASES_DIR = DRIVE_ROOT / "bundle_releases"
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    release_check = _ensure_expected_drive_release(RELEASES_DIR)
    latest_bundle_id = COLAB_NOTEBOOK_BUILD_BUNDLE_ID
    RELEASE_DIR = RELEASES_DIR / latest_bundle_id
    manifest = _verify_expected_bundle(RELEASE_DIR)
    release_manifest_path = RELEASE_DIR / "bundle_manifest.json"
    latest_pointer_path = RELEASES_DIR / "latest.json"
    latest_pointer = _read_manifest(latest_pointer_path) if latest_pointer_path.is_file() else {}
    latest_matches_notebook = (
        latest_pointer.get("bundle_id") == COLAB_NOTEBOOK_BUILD_BUNDLE_ID
        and latest_pointer.get("core_sha256") == COLAB_EXPECTED_CORE_SHA256
        and latest_pointer.get("manifest_sha256") == _sha256(release_manifest_path)
    )
    if latest_matches_notebook:
        release_source = (
            "auto_published_from_" + COLAB_BUNDLE_SOURCE
            if release_check["status"] == "auto_published_and_verified"
            else "latest_pointer"
        )
    else:
        # Un cuaderno reproducible puede activar su release inmutable exacto aunque
        # latest todavía apunte a otra versión; jamás mezcla código e inputs.
        release_source = "notebook_pinned_release"
    manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    bundle_activated = False
    modules_loaded_before_update = any(
        name == "moderacion_peru" or name.startswith("moderacion_peru.") for name in sys.modules
    )
    if not _bundle_is_current(BUNDLE_DIR, manifest_path, latest_bundle_id):
        if not COLAB_AUTO_UPDATE_BUNDLE:
            raise RuntimeError("El bundle de Drive está desactualizado y COLAB_AUTO_UPDATE_BUNDLE=False")
        try:
            manifest = _activate_verified_drive_release(RELEASE_DIR, BUNDLE_DIR, latest_bundle_id)
            bundle_activated = True
        except Exception as exc:
            raise RuntimeError(
                "No fue posible activar la versión esperada desde Google Drive después de "
                f"verificar o autopublicar {RELEASE_DIR}."
            ) from exc
    else:
        manifest = _read_manifest(manifest_path)

    core = BUNDLE_DIR / manifest["core"]["name"]
    if _sha256(core) != manifest["core"]["sha256"]:
        raise ValueError("project_core.zip no coincide con el manifiesto SHA-256")

    RUNTIME_ROOT = Path("/content/moderacion_peru")
    ROOT = RUNTIME_ROOT / "project"
    marker = RUNTIME_ROOT / ".core_sha256"
    expected_core = manifest["core"]["sha256"]
    if not ROOT.is_dir() or not marker.is_file() or marker.read_text().strip() != expected_core:
        if ROOT.exists():
            shutil.rmtree(ROOT)
        ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(core) as archive:
            archive.extractall(ROOT)
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements/colab-l4.txt")]
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(ROOT)])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(expected_core + "\\n", encoding="utf-8")

    if bundle_activated and modules_loaded_before_update:
        raise RuntimeError(
            "El bundle se actualizó y verificó en Drive, pero este kernel ya había importado una "
            "versión anterior de moderacion_peru. Reinicie completamente el kernel de Colab y vuelva "
            "a ejecutar el cuaderno desde la primera celda."
        )

    os.environ["MODPERU_ROOT"] = str(ROOT)
    importlib.invalidate_caches()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.colab import colab_runtime_diagnostics, prepare_colab_context

    COLAB_CONTEXT = prepare_colab_context(
        COLAB_NOTEBOOK_ID,
        project_root=ROOT,
        drive_root=DRIVE_ROOT,
        runtime_root=RUNTIME_ROOT,
        run_id=COLAB_RUN_ID or None,
        require_l4=COLAB_REQUIRE_L4,
        resume=True,
    )
    from moderacion_peru.notebook_ui import notebook_progress, run_with_progress, show_callout, show_command, show_result, show_summary, show_table
    show_result('Bundle de Colab verificado', {
        'estado': 'activado_desde_drive' if bundle_activated else 'ya_estaba_actualizado',
        'bundle_id': manifest['bundle_id'],
        'bundle_del_notebook_al_generarse': COLAB_NOTEBOOK_BUILD_BUNDLE_ID,
        'origen_del_release': release_source,
        'estado_del_release': release_check['status'],
        'core_sha256': expected_core,
        'generado': manifest.get('generated_at'),
        'versión_inmutable_drive': RELEASE_DIR,
    }, tone='success')
    show_result('Diagnóstico de Colab', colab_runtime_diagnostics(), tone='success')
    show_result('Contexto reproducible', COLAB_CONTEXT.as_dict(), tone='success')
else:
    ROOT = _find_local_root()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from moderacion_peru.notebook_ui import notebook_progress, run_with_progress, show_callout, show_command, show_result, show_summary, show_table
    show_summary('Entorno del proyecto', {'raíz': ROOT, 'backend': 'local'}, tone='success')
OPERATIONAL_PROMPT=ROOT/'config/prompt_operacional_ollama_v3_2.md'
if not OPERATIONAL_PROMPT.is_file():
    raise FileNotFoundError(f'Falta el prompt operacional vigente: {OPERATIONAL_PROMPT}')
show_summary('Prompt operacional vigente', {'ruta': OPERATIONAL_PROMPT, 'versión': '3.2.0'}, tone='success')
"""


def colab_bundle_identity() -> dict[str, str]:
    """Fija el bundle local y lo reconstruye automáticamente cuando quedó obsoleto."""

    global _COLAB_BUNDLE_IDENTITY
    if _COLAB_BUNDLE_IDENTITY is not None:
        return _COLAB_BUNDLE_IDENTITY
    from prepare_colab_bundle import ensure_prepared_bundle

    bundle_dir = ROOT / "resultados" / "colab_bundle"
    # Regenerar un cuaderno para corregir el core no debe cambiar silenciosamente
    # el snapshot experimental. Las entradas ya empaquetadas se conservan solo
    # después de verificar sus SHA-256; la preparación explícita del bundle sigue
    # siendo la operación autorizada para incorporar datos nuevos.
    manifest = ensure_prepared_bundle(bundle_dir, preserve_verified_inputs=True)
    _COLAB_BUNDLE_IDENTITY = {
        "bundle_id": str(manifest["bundle_id"]),
        "core_sha256": str(manifest["core"]["sha256"]),
    }
    return _COLAB_BUNDLE_IDENTITY


def colab_setup(notebook_id: str) -> str:
    identity = colab_bundle_identity()
    return (
        COLAB_SETUP.replace("__NOTEBOOK_ID__", notebook_id)
        .replace(
            "__REQUIRE_L4__",
            str(
                notebook_id not in QWEN_A100_NOTEBOOKS
                and notebook_id not in CPU_COLAB_NOTEBOOKS
            ),
        )
        .replace("__EXPECTED_BUNDLE_ID__", identity["bundle_id"])
        .replace("__EXPECTED_CORE_SHA256__", identity["core_sha256"])
    )


def colab_expected_gpu(notebook_id: str | None, requires_gpu: bool) -> str | None:
    if not notebook_id or not requires_gpu:
        return None
    if notebook_id in QWEN_A100_NOTEBOOKS:
        return "NVIDIA A100 40GB or equivalent CUDA BF16 GPU"
    return "NVIDIA L4"


def colab_publisher_setup() -> str:
    identity = colab_bundle_identity()
    return COLAB_PUBLISHER_SETUP.replace(
        "__EXPECTED_BUNDLE_ID__", identity["bundle_id"]
    ).replace("__EXPECTED_CORE_SHA256__", identity["core_sha256"])


DATASET_CHECKPOINT = """from moderacion_peru.colab import prepare_local_bundle_input

if globals().get('COLAB_CONTEXT') is None:
    dataset_checkpoint = prepare_local_bundle_input('dataset_5_salidas', project_root=ROOT)
else:
    dataset_path = COLAB_CONTEXT.input('dataset_5_salidas')
    dataset_checkpoint = {
        'status': 'verified_in_colab',
        'input_key': 'dataset_5_salidas',
        'path': dataset_path,
        'bytes': dataset_path.stat().st_size,
    }
show_result('Dataset descomprimido y verificado', dataset_checkpoint, tone='success')
"""


SCRAPING_PARAMETERS = """# ══════════════════════════════════════════════════════════════════════════════
# CONTROLES DEL SCRAPING: edite únicamente este bloque
# ══════════════════════════════════════════════════════════════════════════════
DISCOVER_NEW = False          # True: consulta canales/búsquedas y guarda candidatos
FETCH_NEW = True              # True: obtiene subtítulos solo de videos aún no procesados
BACKFILL_MISSING_VTT = True   # True: recupera VTT aunque el JSON del video ya exista
DISCOVERY_MODE = "directed"   # "seed", "directed" o "both"

MAX_NEW_VIDEOS = None         # None: incluye todos los pendientes; use un entero para un piloto
MAX_VTT_BACKFILL = None       # None: intenta todos los VTT faltantes; reanudable por archivo
NETWORK_BATCH_SIZE = 10       # llamadas nuevas por lote antes de una pausa larga
NETWORK_BATCH_PAUSE_SECONDS = 15.0
EXCLUDE_CHANNEL_ON_429 = True # difiere solo el canal afectado; continúa con los demás
RANDOMIZE_DOWNLOAD_QUEUE = True
DOWNLOAD_RANDOM_SEED = 20260806 # orden reproducible e intercalado por canal
MAX_VIDEOS_PER_CHANNEL = 75   # candidatos recientes inspeccionados por canal
MAX_RESULTS_PER_QUERY = 30    # candidatos inspeccionados por consulta dirigida
MAX_DIRECTED_CANDIDATES = None # None: conserva toda la cohorte dirigida inédita
MAX_DIRECTED_SEED_CHANNELS = 16
MAX_EXPANDED_CHANNELS = 20    # canales nuevos inferidos desde búsquedas temáticas
MAX_VIDEOS_PER_EXPANDED_CHANNEL = 30

SUBTITLE_LANGUAGES = ("es-PE", "es-419", "es")
MIN_TRANSCRIPT_CHARACTERS = 200
USE_TRANSCRIPT_API_FALLBACK = True
YT_RETRIES = 3
YT_SLEEP_MIN_SECONDS = 2.5
YT_SLEEP_MAX_SECONDS = 10.0
YT_SOCKET_TIMEOUT_SECONDS = 30.0 # máximo por operación HTTP antes de reintentar/omitir
RESUME_DISCOVERY = True       # checkpoint atómico después de cada canal o consulta
STOP_ON_VIDEO_ERROR = False   # False: registra el fallo y continúa con el siguiente
SYNC_TRANSCRIPTS_BY_CHANNEL = True # checkpoint pequeño y sincronizable por canal
SYNC_VTT_BY_VIDEO = True      # checkpoint VTT crudo, deduplicado y sincronizable

# Reinicio destructivo recuperable: deje vacío normalmente. Para archivar los
# artefactos activos y reconstruir el dataset de videos desde cero, descomente:
RESET_VIDEO_DATASET = ""
# RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"

if DISCOVERY_MODE not in {"seed", "directed", "both"}:
    raise ValueError("DISCOVERY_MODE debe ser seed, directed o both")
if ((MAX_NEW_VIDEOS is not None and MAX_NEW_VIDEOS < 0)
        or (MAX_VTT_BACKFILL is not None and MAX_VTT_BACKFILL < 0)
        or NETWORK_BATCH_SIZE < 1 or NETWORK_BATCH_PAUSE_SECONDS < 0
        or MAX_VIDEOS_PER_CHANNEL < 1 or MAX_RESULTS_PER_QUERY < 1
        or (MAX_DIRECTED_CANDIDATES is not None and MAX_DIRECTED_CANDIDATES < 1)
        or MAX_DIRECTED_SEED_CHANNELS < 1
        or MAX_EXPANDED_CHANNELS < 0 or MAX_VIDEOS_PER_EXPANDED_CHANNEL < 1):
    raise ValueError("Los límites de videos deben ser válidos")
if MIN_TRANSCRIPT_CHARACTERS < 1:
    raise ValueError("MIN_TRANSCRIPT_CHARACTERS debe ser positivo")
if RANDOMIZE_DOWNLOAD_QUEUE and not str(DOWNLOAD_RANDOM_SEED).strip():
    raise ValueError("DOWNLOAD_RANDOM_SEED no puede estar vacío")
if YT_SLEEP_MIN_SECONDS < 0 or YT_SLEEP_MAX_SECONDS < YT_SLEEP_MIN_SECONDS:
    raise ValueError("El intervalo de espera de yt-dlp no es válido")
if YT_SOCKET_TIMEOUT_SECONDS <= 0:
    raise ValueError("YT_SOCKET_TIMEOUT_SECONDS debe ser positivo")

show_summary('Configuración del scraping', {
    "discover_new": DISCOVER_NEW,
    "fetch_new": FETCH_NEW,
    "backfill_missing_vtt": BACKFILL_MISSING_VTT,
    "discovery_mode": DISCOVERY_MODE,
    "max_new_videos": MAX_NEW_VIDEOS,
    "max_vtt_backfill": MAX_VTT_BACKFILL,
    "network_batch_size": NETWORK_BATCH_SIZE,
    "network_batch_pause_seconds": NETWORK_BATCH_PAUSE_SECONDS,
    "exclude_channel_on_429": EXCLUDE_CHANNEL_ON_429,
    "randomize_download_queue": RANDOMIZE_DOWNLOAD_QUEUE,
    "download_random_seed": DOWNLOAD_RANDOM_SEED,
    "max_videos_per_channel": MAX_VIDEOS_PER_CHANNEL,
    "max_results_per_query": MAX_RESULTS_PER_QUERY,
    "max_directed_candidates": MAX_DIRECTED_CANDIDATES,
    "max_directed_seed_channels": MAX_DIRECTED_SEED_CHANNELS,
    "max_expanded_channels": MAX_EXPANDED_CHANNELS,
    "subtitle_languages": SUBTITLE_LANGUAGES,
    "min_transcript_characters": MIN_TRANSCRIPT_CHARACTERS,
    "transcript_api_fallback": USE_TRANSCRIPT_API_FALLBACK,
    "yt_socket_timeout_seconds": YT_SOCKET_TIMEOUT_SECONDS,
    "resume_discovery": RESUME_DISCOVERY,
    "sync_transcripts_by_channel": SYNC_TRANSCRIPTS_BY_CHANNEL,
    "sync_vtt_by_video": SYNC_VTT_BY_VIDEO,
    "reset_video_dataset_armed": bool(RESET_VIDEO_DATASET),
}, tone='neutral')

if RESET_VIDEO_DATASET:
    from moderacion_peru.acquisition import reset_active_video_dataset
    reset_result = reset_active_video_dataset(ROOT, RESET_VIDEO_DATASET)
    show_result('Dataset activo archivado; reconstrucción desde cero habilitada', reset_result, tone='warning')
"""


SCRAPING_SOURCES = """# Canales semilla recuperados del cuaderno histórico; puede añadir, quitar o editar filas.
# `categoria_fuente` describe el dominio/registro del canal, no una etiqueta de daño.
_SEED_ROWS = [
    ("Marco Sifuentes / Ocram", "politica_analisis", "https://www.youtube.com/@canalYAAAAA"),
    ("El diario de Curwen", "politica_opinion", "https://www.youtube.com/@curwen"),
    ("Sin Guion con Rosa María Palacios", "politica_periodismo", "https://www.youtube.com/@singuionlr"),
    ("RPP Noticias", "politica_actualidad", "https://www.youtube.com/@RPPNoticias"),
    ("Exitosa Noticias", "politica_actualidad", "https://www.youtube.com/@exitosape"),
    ("Willax Television", "politica_opinion", "https://www.youtube.com/@WillaxTV"),
    ("Canal N", "politica_actualidad", "https://www.youtube.com/@canaln"),
    ("ATV Noticias", "politica_actualidad", "https://www.youtube.com/@ATVNoticias"),
    ("Latina Noticias", "politica_actualidad", "https://www.youtube.com/@latinanoticias"),
    ("Panamericana Noticias", "politica_actualidad", "https://www.youtube.com/@Panamericana-Noticias"),
    ("Hablando Huevadas", "humor_streaming", "https://www.youtube.com/@HablandoHuevadasOficial"),
    ("Todo Good", "streaming_opinion", "https://www.youtube.com/@todogoodpe"),
    ("Goblinciano", "streaming_opinion", "https://www.youtube.com/@Goblinciano"),
    ("El Cacas", "humor_streaming", "https://www.youtube.com/@ElCacas"),
    ("Negro Fuertes", "humor_comedia", "https://www.youtube.com/@NegroFuertes"),
    ("Jason Qqq", "humor_streaming", "https://www.youtube.com/@JasonQqqOficial"),
    ("La Cotorrisa Perú", "humor_podcast", "https://www.youtube.com/@LaCotorrisaPeru"),
    ("Magaly TV La Firme", "farandula", "https://www.youtube.com/@MagalyTVLaFirmeATV"),
    ("Amor y Fuego", "farandula", "https://www.youtube.com/@AmoryFuego"),
    ("América Hoy", "farandula", "https://www.youtube.com/@americahoytv"),
    ("Instarándula", "farandula_digital", "https://www.youtube.com/@Instarandula"),
    ("El Popular", "farandula_digital", "https://www.youtube.com/@ElPopularPeru"),
    ("Nico Moschella", "deportes_informal", "https://www.youtube.com/@NicoMoschella"),
    ("Líbero Deportes", "deportes", "https://www.youtube.com/@DiarioLiberoOficial"),
    ("Depor", "deportes", "https://www.youtube.com/@DeporPeru"),
    ("Misias pero viajeras", "viajes", "https://www.youtube.com/c/Misiasperoviajeras"),
    ("Buen Viaje", "viajes", "https://www.youtube.com/c/BuenViajePe"),
    ("Viaja y Prueba", "viajes_gastronomia", "https://www.youtube.com/@ViajayPrueba"),
    ("Cocinando con la Patty", "gastronomia", "https://www.youtube.com/@CocinandoConLaPatty"),
    ("Arde Troya con Juliana Oxenford", "politica_analisis", "https://www.youtube.com/@ardetroyalr"),
    ("Panorama", "politica_periodismo", "https://www.youtube.com/@PanoramaPTV"),
    ("Juanito y Richard", "humor_comedia", "https://www.youtube.com/@JuanitoyRichard"),
    ("Nada Espacial", "humor_podcast", "https://www.youtube.com/@nadaespacialpodcast"),
    ("L1MAX", "deportes_informal", "https://www.youtube.com/@L1MAX_"),
    ("Cocina Cajamarquina", "gastronomia", "https://www.youtube.com/@cocinacajamarquina"),
    ("Tío Lenguado y Descocaos", "viajes_gastronomia", "https://www.youtube.com/@tiolenguado"),
]
SEED_CHANNELS = [
    {"name": name, "categoria_fuente": category, "url": url, "sampling_mode": "seed"}
    for name, category, url in _SEED_ROWS
]

# Ampliación dirigida: las cuotas son máximos por canal, nunca prevalencias esperadas.
DIRECTED_CHANNEL_CATALOG = [
    {"name": "Hablando Huevadas", "url": "https://www.youtube.com/@HablandoHuevadasOficial", "quota": 70, "target_category": "CONTENIDO_SEXUAL|ATAQUE_POR_GENERO_IDENTIDAD|ACOSO_AMENAZA"},
    {"name": "Goblinciano", "url": "https://www.youtube.com/@Goblinciano", "quota": 85, "target_category": "RACISMO_DISCRIMINACION|ACOSO_AMENAZA"},
    {"name": "Juanito y Richard", "url": "https://www.youtube.com/@JuanitoyRichard", "quota": 85, "target_category": "RACISMO_DISCRIMINACION|ACOSO_AMENAZA"},
    {"name": "Arde Troya con Juliana Oxenford", "url": "https://www.youtube.com/@ardetroyalr", "quota": 55, "target_category": "ACOSO_AMENAZA"},
    {"name": "Todo Good", "url": "https://www.youtube.com/@todogoodpe", "quota": 40, "target_category": "ACOSO_AMENAZA"},
    {"name": "Magaly TV La Firme", "url": "https://www.youtube.com/@MagalyTVLaFirmeATV", "quota": 35, "target_category": "ACOSO_AMENAZA|CONTENIDO_SEXUAL"},
]

SEED_SEARCH_QUERIES = [
    "noticias política Perú canal YouTube",
    "periodismo opinión Perú YouTube",
    "humor streaming Perú lenguaje coloquial",
    "farándula espectáculos Perú",
    "deportes peruanos comentarios YouTube",
]
DIRECTED_QUERY_CATALOG = [
    {"query": "insultos racistas discriminación Perú denuncia", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "discriminación regional clasismo racial Perú", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "ataque machista misoginia Perú denuncia", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "ataque homofóbico transfóbico Perú denuncia", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "amenaza de muerte denuncia Perú", "target_category": "ACOSO_AMENAZA"},
    {"query": "extorsionadores amenazan audio Perú", "target_category": "ACOSO_AMENAZA"},
    {"query": "acoso sexual denuncia televisión peruana", "target_category": "CONTENIDO_SEXUAL|ACOSO_AMENAZA"},
    {"query": "cosificación contenido sexual explícito televisión Perú", "target_category": "CONTENIDO_SEXUAL"},
]
"""


SCRAPING_REUSE_AND_PLAN = """import importlib
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import (
    VIDEO_DATASET_RESET_MARKER,
    build_directed_sampling_plan,
    collect_project_video_inventory,
    consolidate_available_transcripts,
    load_candidates,
    load_vtt_backfill_candidates,
    materialize_vtt_checkpoint,
    materialize_transcripts_by_channel,
    merge_candidates,
    select_directed_search_queries,
    select_directed_seed_channels,
)
from moderacion_peru.io import read_jsonl
from moderacion_peru.taxonomy import load_taxonomy

CANONICAL = ROOT/'datos/raw/transcripts_raw.jsonl'
CACHE = ROOT/'datos/raw/transcripts_cache'
TRANSCRIPTS_BY_CHANNEL = ROOT/'datos/raw/transcripts_by_channel'
VTT_BY_VIDEO = ROOT/'datos/raw/vtt_by_video'
DIRECTED_DATASET = ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
rebuild_from_zero = (ROOT/VIDEO_DATASET_RESET_MARKER).exists()
consolidation_stats = consolidate_available_transcripts(
    ROOT,
    CANONICAL,
    cache_dir=CACHE,
    channel_dir=TRANSCRIPTS_BY_CHANNEL,
    vtt_dir=VTT_BY_VIDEO if VTT_BY_VIDEO.is_dir() else None,
    include_historical_snapshots=not rebuild_from_zero,
)
consolidation_stats['historical_bootstrap_disabled'] = rebuild_from_zero
show_result(
    'Consolidación de todas las transcripciones disponibles',
    consolidation_stats,
    tone='warning' if rebuild_from_zero else 'success',
)
if SYNC_TRANSCRIPTS_BY_CHANNEL:
    channel_partition_stats = materialize_transcripts_by_channel(CANONICAL, TRANSCRIPTS_BY_CHANNEL)
    show_summary('Checkpoint de transcripciones por canal', {
        'videos': channel_partition_stats['total_videos'],
        'canales': channel_partition_stats['total_channels'],
        'partes_jsonl': channel_partition_stats['total_channel_files'],
        'máximo_bytes_por_parte': channel_partition_stats['max_channel_file_bytes'],
        'carpeta': TRANSCRIPTS_BY_CHANNEL,
        'canonico_preservado': CANONICAL.exists(),
    }, tone='success')

VTT_BACKFILL_CANDIDATES = []
if SYNC_VTT_BY_VIDEO:
    vtt_checkpoint_stats = materialize_vtt_checkpoint(
        ROOT,
        VTT_BY_VIDEO,
        read_jsonl(CANONICAL) if CANONICAL.exists() else [],
    )
    VTT_BACKFILL_CANDIDATES = load_vtt_backfill_candidates(VTT_BY_VIDEO)
    show_summary('Checkpoint consolidado de VTT por video', {
        'archivos_vtt': vtt_checkpoint_stats['total_files'],
        'videos_con_vtt': vtt_checkpoint_stats['total_videos'],
        'videos_con_transcripcion': vtt_checkpoint_stats['transcript_videos'],
        'videos_sin_vtt': vtt_checkpoint_stats['missing_vtt_videos'],
        'vtt_invalidos': vtt_checkpoint_stats['invalid_vtt_files'],
        'cola_backfill': VTT_BY_VIDEO/'missing_vtt.jsonl',
        'carpeta': VTT_BY_VIDEO,
    }, tone='warning' if VTT_BACKFILL_CANDIDATES else 'success')

KNOWN_VIDEO_IDS, VIDEO_INVENTORY = collect_project_video_inventory(
    ROOT,
    canonical_path=CANONICAL,
    cache_dir=CACHE,
    include_historical_sources=not rebuild_from_zero,
    include_derived_sources=not rebuild_from_zero,
)
show_summary('Inventario global para evitar duplicaciones', {
    'transcripciones_canónicas_completas': VIDEO_INVENTORY['canonical_transcripts'],
    'transcripciones_completas_disponibles': VIDEO_INVENTORY['full_transcripts_union'],
    'videos_con_texto_derivado': VIDEO_INVENTORY['derived_text_videos'],
    'videos_solo_en_derivados': VIDEO_INVENTORY['derived_only_videos'],
    'videos_conocidos_globales': VIDEO_INVENTORY['known_videos_union'],
    'fuentes_raw_históricas': VIDEO_INVENTORY['historical_source_files'],
    'fuentes_derivadas': VIDEO_INVENTORY['derived_source_files'],
}, tone='warning' if VIDEO_INVENTORY['derived_only_videos'] else 'success')

taxonomy = load_taxonomy(ROOT/'config/taxonomia_v2.json')
directed_plan = None
DIRECTED_CHANNELS = []
DIRECTED_SEARCH_QUERIES = []
if DISCOVERY_MODE in {'directed', 'both'}:
    directed_plan = build_directed_sampling_plan(
        read_jsonl(DIRECTED_DATASET) if DIRECTED_DATASET.exists() else [],
        read_jsonl(CANONICAL) if CANONICAL.exists() else [],
        damage_labels=taxonomy.damage_labels,
    )
    DIRECTED_CHANNELS = select_directed_seed_channels(
        directed_plan,
        DIRECTED_CHANNEL_CATALOG,
        max_channels=MAX_DIRECTED_SEED_CHANNELS,
    )
    DIRECTED_SEARCH_QUERIES = select_directed_search_queries(
        directed_plan,
        DIRECTED_QUERY_CATALOG,
        max_queries=len(DIRECTED_QUERY_CATALOG),
        max_results_per_query=MAX_RESULTS_PER_QUERY,
    )
    show_summary('Plan de ampliación dirigida', {
        'estrategia': directed_plan['strategy'],
        'soporte_videos_train_validation': directed_plan['support_videos'],
        'déficit_videos': directed_plan['deficit_videos'],
        'pesos': {key: round(value, 4) for key, value in directed_plan['weights'].items()},
        'canales_semilla': len(DIRECTED_CHANNELS),
        'consultas_temáticas': len(DIRECTED_SEARCH_QUERIES),
    }, tone='warning' if directed_plan['strategy'] == 'fallback_equal' else 'success')

CHANNEL_SOURCES = (
    SEED_CHANNELS if DISCOVERY_MODE == 'seed'
    else DIRECTED_CHANNELS if DISCOVERY_MODE == 'directed'
    else SEED_CHANNELS + DIRECTED_CHANNELS
)
SEARCH_QUERIES = (
    SEED_SEARCH_QUERIES if DISCOVERY_MODE == 'seed'
    else DIRECTED_SEARCH_QUERIES if DISCOVERY_MODE == 'directed'
    else SEED_SEARCH_QUERIES + DIRECTED_SEARCH_QUERIES
)
show_summary('Fuentes seleccionadas', {
    'modo': DISCOVERY_MODE,
    'canales': len(CHANNEL_SOURCES),
    'consultas': len(SEARCH_QUERIES),
}, tone='neutral')
"""


SCRAPING_DISCOVERY = """from collections import Counter
import importlib
from tqdm.auto import tqdm
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import (
    discover_youtube_candidates,
    expand_directed_channel_sources,
    processed_video_ids,
    select_directed_candidates,
)
from moderacion_peru.io import append_jsonl_once, write_json_atomic, write_jsonl_atomic

DISCOVERED_PATH = ROOT/'datos/raw/video_candidates.jsonl'
DISCOVERY_FAILURES_PATH = ROOT/'datos/raw/fallos_descubrimiento_ultima_ejecucion.json'
DIRECTED_SELECTION_PATH = ROOT/'datos/raw/directed_candidates_latest.jsonl'
DIRECTED_PLAN_PATH = ROOT/'datos/raw/manifests/directed_plan_latest.json'
DISCOVERY_CHECKPOINT_PATH = ROOT/f'datos/raw/manifests/discovery_{DISCOVERY_MODE}_checkpoint.json'
discovered = []
directed_selection = []
expanded_channels = []
if DISCOVER_NEW:
    source_outcomes = Counter()
    source_total = len(CHANNEL_SOURCES) + len(SEARCH_QUERIES)
    source_progress = tqdm(total=source_total, desc='Descubriendo fuentes', unit='fuente')

    def report_discovery(event):
        source_name = str(event.get('source') or '(fuente sin nombre)')
        if event['status'] == 'started':
            source_progress.set_description(f'Descubriendo · {source_name[:42]}')
            source_progress.set_postfix(
                fuente=source_name[:42],
                correctas=source_outcomes['ok'],
                fallidas=source_outcomes['failed'],
                reanudadas=source_outcomes['resumed'],
                candidatos=event['candidates_unique'],
            )
            return
        source_outcomes[event['status']] += 1
        if event.get('resumed'):
            source_outcomes['resumed'] += 1
        source_progress.update(1)
        source_progress.set_postfix(
            fuente=source_name[:42],
            correctas=source_outcomes['ok'],
            fallidas=source_outcomes['failed'],
            reanudadas=source_outcomes['resumed'],
            candidatos=event['candidates_unique'],
        )

    try:
        discovered, discovery_failures = discover_youtube_candidates(
            CHANNEL_SOURCES,
            SEARCH_QUERIES,
            max_videos_per_channel=MAX_VIDEOS_PER_CHANNEL,
            max_results_per_query=MAX_RESULTS_PER_QUERY,
            retries=YT_RETRIES,
            sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
            sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
            socket_timeout_seconds=YT_SOCKET_TIMEOUT_SECONDS,
            checkpoint_path=DISCOVERY_CHECKPOINT_PATH if RESUME_DISCOVERY else None,
            progress_callback=report_discovery,
        )
        if directed_plan is not None:
            expanded_channels = expand_directed_channel_sources(
                discovered,
                directed_plan,
                known_channel_ids=[source.get('channel_id') for source in DIRECTED_CHANNELS],
                max_channels=MAX_EXPANDED_CHANNELS,
                videos_per_channel=MAX_VIDEOS_PER_EXPANDED_CHANNEL,
            )
            if expanded_channels:
                source_total += len(expanded_channels)
                source_progress.total = source_total
                source_progress.refresh()
                expanded_candidates, expanded_failures = discover_youtube_candidates(
                    expanded_channels,
                    (),
                    max_videos_per_channel=MAX_VIDEOS_PER_EXPANDED_CHANNEL,
                    max_results_per_query=MAX_RESULTS_PER_QUERY,
                    retries=YT_RETRIES,
                    sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
                    sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
                    socket_timeout_seconds=YT_SOCKET_TIMEOUT_SECONDS,
                    checkpoint_path=DISCOVERY_CHECKPOINT_PATH if RESUME_DISCOVERY else None,
                    progress_callback=report_discovery,
                )
                discovered = merge_candidates(discovered, expanded_candidates)
                discovery_failures.extend(expanded_failures)
            directed_pool = [
                candidate for candidate in discovered
                if candidate.get('sampling_mode') == 'directed'
            ]
            directed_known_excluded = len({
                str(candidate.get('video_id') or '').strip()
                for candidate in directed_pool
            } & KNOWN_VIDEO_IDS)
            directed_selection = select_directed_candidates(
                directed_pool,
                KNOWN_VIDEO_IDS,
                directed_plan,
                max_candidates=(
                    len(directed_pool)
                    if MAX_DIRECTED_CANDIDATES is None
                    else MAX_DIRECTED_CANDIDATES
                ),
            )
            write_jsonl_atomic(DIRECTED_SELECTION_PATH, directed_selection)
            write_json_atomic(DIRECTED_PLAN_PATH, {
                **directed_plan,
                'planned_channels': DIRECTED_CHANNELS,
                'planned_queries': DIRECTED_SEARCH_QUERIES,
                'expanded_channels': expanded_channels,
                'directed_candidates_discovered': len(directed_pool),
                'directed_known_videos_excluded': directed_known_excluded,
                'directed_candidates_selected': len(directed_selection),
                'selection_path': DIRECTED_SELECTION_PATH,
            })
    finally:
        source_progress.close()
    added, existing = append_jsonl_once(DISCOVERED_PATH, discovered, id_field='video_id')
    write_json_atomic(DISCOVERY_FAILURES_PATH, discovery_failures)
    show_summary('Resumen del descubrimiento', {
        "sources_total": source_total,
        "sources_ok": source_outcomes['ok'],
        "sources_failed": source_outcomes['failed'],
        "sources_resumed": source_outcomes['resumed'],
        "candidates_unique": len(discovered),
        "candidates_added": added,
        "candidates_existing": existing,
        "expanded_channels": len(expanded_channels),
        "directed_cohort": len(directed_selection),
        "directed_known_videos_excluded": (
            directed_known_excluded if directed_plan is not None else 0
        ),
        "checkpoint": DISCOVERY_CHECKPOINT_PATH if RESUME_DISCOVERY else None,
    }, tone='success' if not discovery_failures else 'warning')
    if discovery_failures:
        failure_counts = Counter(row['failure_kind'] for row in discovery_failures)
        show_summary('Fallos de fuentes por motivo', dict(sorted(failure_counts.items())), tone='warning')
        show_summary('Artefacto de auditoría', {'ruta': DISCOVERY_FAILURES_PATH}, tone='neutral')
else:
    show_callout('Descubrimiento desactivado', 'Se reutilizan candidatos y transcripciones locales.', tone='neutral')
"""


SCRAPING_CANDIDATES = """import importlib
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import order_candidates_for_acquisition, processed_video_ids

if DISCOVERY_MODE == 'directed':
    candidates = load_candidates(DIRECTED_SELECTION_PATH)
    candidate_origin = 'cohorte_dirigida_vigente'
elif DISCOVERY_MODE == 'both' and DISCOVER_NEW:
    seed_candidates_current = [
        candidate for candidate in discovered
        if candidate.get('sampling_mode') == 'seed'
    ]
    candidates = merge_candidates(seed_candidates_current, directed_selection)
    candidate_origin = 'descubrimiento_actual_seed_más_cohorte_dirigida'
else:
    candidate_files = [
        ROOT/'datos/raw/video_candidates.jsonl',
        ROOT/'datos/raw/videos_candidatos.csv',
    ]
    candidates = merge_candidates(*(load_candidates(source) for source in candidate_files))
    candidate_origin = 'archivo_acumulado_general'

canonical_ids = processed_video_ids(CANONICAL)
candidate_ids = {str(candidate['video_id']).strip() for candidate in candidates}
existing_canonical_ids = candidate_ids & canonical_ids
existing_derived_ids = candidate_ids & (KNOWN_VIDEO_IDS - canonical_ids)
existing_known_ids = candidate_ids & KNOWN_VIDEO_IDS
pending_candidates = [
    candidate for candidate in candidates
    if str(candidate['video_id']).strip() not in KNOWN_VIDEO_IDS
]
if RANDOMIZE_DOWNLOAD_QUEUE:
    pending_candidates = order_candidates_for_acquisition(
        pending_candidates,
        random_seed=DOWNLOAD_RANDOM_SEED,
    )
cached_ids = {path.stem for path in CACHE.glob('*.json')}
pending_cached = sum(
    str(candidate['video_id']).strip() in cached_ids for candidate in pending_candidates
)
show_summary('Candidatos filtrados antes de la adquisición', {
    'origen': candidate_origin,
    'únicos': len(candidates),
    'transcripciones_canónicas_totales': len(canonical_ids),
    'videos_conocidos_globales': len(KNOWN_VIDEO_IDS),
    'ya_canónicos_omitidos': len(existing_canonical_ids),
    'ya_derivados_históricos_omitidos': len(existing_derived_ids),
    'ya_conocidos_omitidos_total': len(existing_known_ids),
    'pendientes_totales': len(pending_candidates),
    'pendientes_reutilizables_desde_caché': pending_cached,
    'pendientes_que_requieren_red': len(pending_candidates) - pending_cached,
    'cola_pseudoaleatoria': RANDOMIZE_DOWNLOAD_QUEUE,
    'semilla_cola': DOWNLOAD_RANDOM_SEED if RANDOMIZE_DOWNLOAD_QUEUE else None,
}, tone='neutral')
"""


SCRAPING_EXECUTION = """from functools import partial
import importlib
from tqdm.auto import tqdm
import moderacion_peru.acquisition as acquisition_module
importlib.reload(acquisition_module)

from moderacion_peru.acquisition import (
    backfill_missing_vtt,
    fetch_youtube_subtitles,
    ingest_incremental,
    materialize_transcripts_by_channel,
    materialize_vtt_checkpoint,
    order_candidates_for_acquisition,
)

FAILURES = ROOT/'datos/raw/fallos_adquisicion.jsonl'
VTT_FAILURES = ROOT/'datos/raw/fallos_vtt_backfill.jsonl'
fetcher = partial(
    fetch_youtube_subtitles,
    languages=SUBTITLE_LANGUAGES,
    retries=YT_RETRIES,
    sleep_min_seconds=YT_SLEEP_MIN_SECONDS,
    sleep_max_seconds=YT_SLEEP_MAX_SECONDS,
    socket_timeout_seconds=YT_SOCKET_TIMEOUT_SECONDS,
    minimum_transcript_characters=MIN_TRANSCRIPT_CHARACTERS,
    use_transcript_api_fallback=USE_TRANSCRIPT_API_FALLBACK,
    vtt_output_dir=VTT_BY_VIDEO if SYNC_VTT_BY_VIDEO else None,
)

vtt_backfill_queue = list(VTT_BACKFILL_CANDIDATES)
if RANDOMIZE_DOWNLOAD_QUEUE:
    vtt_backfill_queue = order_candidates_for_acquisition(
        vtt_backfill_queue,
        random_seed=DOWNLOAD_RANDOM_SEED,
    )
if BACKFILL_MISSING_VTT and vtt_backfill_queue:
    vtt_progress = tqdm(total=len(vtt_backfill_queue), desc='Recuperando VTT faltantes', unit='video')

    def report_vtt_backfill(event):
        counters = event['counters']
        vtt_progress.update(event.get('advance', 1))
        vtt_progress.set_description(
            'Pausa entre lotes VTT' if event['status'] == 'batch_pause' else 'Recuperando VTT faltantes'
        )
        vtt_progress.set_postfix(
            recuperados=counters['fetched'],
            fallidos=counters['failed'],
            diferidos=counters['deferred_by_limit'],
            pausa_429=counters['deferred_rate_limit'],
            canales_429=counters['rate_limited_channels'],
            lotes=counters['batch_pauses'],
        )

    try:
        vtt_backfill_stats = backfill_missing_vtt(
            vtt_backfill_queue,
            VTT_BY_VIDEO,
            fetcher=fetcher if FETCH_NEW else None,
            failure_path=VTT_FAILURES,
            max_new_videos=MAX_VTT_BACKFILL,
            network_batch_size=NETWORK_BATCH_SIZE,
            batch_pause_seconds=NETWORK_BATCH_PAUSE_SECONDS,
            exclude_rate_limited_channels=EXCLUDE_CHANNEL_ON_429,
            stop_on_error=STOP_ON_VIDEO_ERROR,
            progress_callback=report_vtt_backfill,
        )
    finally:
        vtt_progress.close()
    show_summary(
        'Resumen de recuperación VTT',
        {'pendientes_antes': len(vtt_backfill_queue), **vtt_backfill_stats},
        tone='success' if not vtt_backfill_stats['failed'] else 'warning',
    )
elif vtt_backfill_queue:
    show_callout(
        'Backfill VTT pendiente',
        f'Hay {len(vtt_backfill_queue)} videos sin VTT; active BACKFILL_MISSING_VTT y FETCH_NEW.',
        tone='warning',
    )
else:
    show_callout('VTT completos', 'No hay transcripciones canónicas sin VTT.', tone='success')

if pending_candidates:
    video_progress = tqdm(total=len(pending_candidates), desc='Procesando pendientes', unit='video')

    def report_acquisition(event):
        counters = event['counters']
        video_progress.update(event.get('advance', 1))
        if event['status'] == 'batch_pause':
            description = 'Pausa entre lotes'
        else:
            description = 'Procesando pendientes'
        video_progress.set_description(description)
        video_progress.set_postfix(
            existentes=counters['already_canonical'],
            cache=counters['reused_cache'],
            nuevos=counters['fetched'],
            fallidos=counters['failed'],
            diferidos=counters['deferred_by_limit'],
            pausa_429=counters['deferred_rate_limit'],
            intentos_429=counters.get('failure_rate_limited', 0),
            canales_429=counters['rate_limited_channels'],
            lotes=counters['batch_pauses'],
        )

    try:
        stats = ingest_incremental(
            pending_candidates,
            CANONICAL,
            CACHE,
            fetcher=fetcher if FETCH_NEW else None,
            failure_path=FAILURES,
            max_new_videos=MAX_NEW_VIDEOS,
            network_batch_size=NETWORK_BATCH_SIZE,
            batch_pause_seconds=NETWORK_BATCH_PAUSE_SECONDS,
            exclude_rate_limited_channels=EXCLUDE_CHANNEL_ON_429,
            stop_on_error=STOP_ON_VIDEO_ERROR,
            progress_callback=report_acquisition,
            channel_transcript_dir=TRANSCRIPTS_BY_CHANNEL if SYNC_TRANSCRIPTS_BY_CHANNEL else None,
        )
    finally:
        video_progress.close()
    stats = {
        'candidates_total': len(candidates),
        'filtered_existing_before_run': len(existing_known_ids),
        'pending_before_run': len(pending_candidates),
        **stats,
    }
    show_summary('Resumen de adquisición', stats, tone='success' if not stats['failed'] else 'warning')
    if stats['failed']:
        show_summary('Fallos de adquisición por motivo', {
            key.removeprefix('failure_'): value
            for key, value in stats.items()
            if key.startswith('failure_') and not key.startswith('failure_records_') and value
        }, tone='warning')
        show_summary('Videos omitidos sin detener el lote', {
            'cantidad': stats['failed'],
            'detalle': FAILURES,
        }, tone='warning')
elif candidates:
    show_callout(
        'Sin videos pendientes',
        'Todos los candidatos ya tienen una transcripción canónica; no se iniciaron descargas.',
        tone='success',
    )
else:
    show_callout(
        'No hay candidatos',
        'Active DISCOVER_NEW o añada un CSV/JSONL. El corpus existente no se vuelve a descargar.',
        tone='warning',
    )

if SYNC_TRANSCRIPTS_BY_CHANNEL:
    final_channel_stats = materialize_transcripts_by_channel(CANONICAL, TRANSCRIPTS_BY_CHANNEL)
    show_summary('JSONL por canal consolidados al cierre', {
        'videos': final_channel_stats['total_videos'],
        'canales': final_channel_stats['total_channels'],
        'partes_jsonl': final_channel_stats['total_channel_files'],
    }, tone='success')
if SYNC_VTT_BY_VIDEO:
    final_vtt_stats = materialize_vtt_checkpoint(
        ROOT,
        VTT_BY_VIDEO,
        read_jsonl(CANONICAL) if CANONICAL.exists() else [],
    )
    show_summary('VTT consolidados al cierre', {
        'archivos_vtt': final_vtt_stats['total_files'],
        'videos_con_vtt': final_vtt_stats['total_videos'],
        'videos_sin_vtt': final_vtt_stats['missing_vtt_videos'],
        'manifiesto_faltantes': VTT_BY_VIDEO/'missing_vtt.jsonl',
    }, tone='success' if not final_vtt_stats['missing_vtt_videos'] else 'warning')
"""


def _replace_required(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"No se encontró el bloque esperado: {old[:80]!r}")
    return source.replace(old, new, 1)


SCRAPING_MINORITY_PARAMETERS = SCRAPING_PARAMETERS
for _old, _new in (
    (
        "DISCOVER_NEW = False          # True: consulta canales/búsquedas y guarda candidatos",
        "DISCOVER_NEW = True           # Esta campaña descubre fuentes nuevas dirigidas",
    ),
    (
        "BACKFILL_MISSING_VTT = True   # True: recupera VTT aunque el JSON del video ya exista",
        "BACKFILL_MISSING_VTT = False  # El backfill general permanece en 01_01",
    ),
    (
        'DISCOVERY_MODE = "directed"   # "seed", "directed" o "both"',
        'DISCOVERY_MODE = "directed"   # Fijo: ampliación de daños minoritarios',
    ),
    (
        "MAX_VIDEOS_PER_CHANNEL = 75   # candidatos recientes inspeccionados por canal",
        "MAX_VIDEOS_PER_CHANNEL = 400  # inspección amplia; no implica descargar todos",
    ),
    (
        "MAX_RESULTS_PER_QUERY = 30    # candidatos inspeccionados por consulta dirigida",
        "MAX_RESULTS_PER_QUERY = 80    # candidatos inspeccionados por consulta dirigida",
    ),
    (
        "MAX_DIRECTED_CANDIDATES = None # None: conserva toda la cohorte dirigida inédita",
        "MAX_DIRECTED_CANDIDATES = None # se calcula por déficit y rendimiento histórico",
    ),
    (
        "MAX_EXPANDED_CHANNELS = 20    # canales nuevos inferidos desde búsquedas temáticas",
        "MAX_EXPANDED_CHANNELS = 16    # canales nuevos inferidos desde búsquedas temáticas",
    ),
    (
        "MAX_VIDEOS_PER_EXPANDED_CHANNEL = 30",
        "MAX_VIDEOS_PER_EXPANDED_CHANNEL = 150",
    ),
):
    SCRAPING_MINORITY_PARAMETERS = _replace_required(
        SCRAPING_MINORITY_PARAMETERS, _old, _new
    )
SCRAPING_MINORITY_PARAMETERS = _replace_required(
    SCRAPING_MINORITY_PARAMETERS,
    'DISCOVERY_MODE = "directed"   # Fijo: ampliación de daños minoritarios\n\n',
    'DISCOVERY_MODE = "directed"   # Fijo: ampliación de daños minoritarios\n'
    "PERU_ONLY = True              # Invariante: bloquea canales extranjeros o no verificados\n"
    "TARGET_TOTAL_CHUNKS_PER_DAMAGE = 2_000\n"
    "TARGET_ELIGIBLE_SPLITS = ('train', 'validation', 'test')\n"
    "SPLIT_SEED = 20260805\n"
    "DIRECTED_SAFETY_FACTOR = 1.5  # margen ante menor rendimiento de videos nuevos\n"
    "DIRECTED_YIELD_DISCOUNT = 0.5 # usa solo 50% del mejor rendimiento histórico\n"
    "MIN_DIRECTED_TRAIN_VIDEOS = 50\n"
    "DIRECTED_HOLDOUT_FRACTION = 0.15\n"
    "DIRECTED_SPLIT_BUDGET = None  # se materializa después de calcular el plan\n\n",
)
SCRAPING_MINORITY_PARAMETERS = _replace_required(
    SCRAPING_MINORITY_PARAMETERS,
    'if DISCOVERY_MODE not in {"seed", "directed", "both"}:\n',
    "if TARGET_TOTAL_CHUNKS_PER_DAMAGE < 1:\n"
    '    raise ValueError("TARGET_TOTAL_CHUNKS_PER_DAMAGE debe ser positivo")\n'
    "if PERU_ONLY is not True:\n"
    '    raise ValueError("01_015 exige PERU_ONLY=True")\n'
    "if DIRECTED_SAFETY_FACTOR < 1 or not 0 < DIRECTED_YIELD_DISCOUNT <= 1:\n"
    '    raise ValueError("Los supuestos conservadores del presupuesto no son válidos")\n'
    "if MIN_DIRECTED_TRAIN_VIDEOS < 1 or DIRECTED_HOLDOUT_FRACTION < 0:\n"
    '    raise ValueError("Los mínimos del presupuesto dirigido no son válidos")\n'
    'if DISCOVERY_MODE not in {"seed", "directed", "both"}:\n',
)
SCRAPING_MINORITY_PARAMETERS = _replace_required(
    SCRAPING_MINORITY_PARAMETERS,
    '    "discovery_mode": DISCOVERY_MODE,\n',
    '    "discovery_mode": DISCOVERY_MODE,\n'
    '    "peru_only": PERU_ONLY,\n'
    '    "target_total_chunks_per_damage": TARGET_TOTAL_CHUNKS_PER_DAMAGE,\n'
    '    "target_eligible_splits": TARGET_ELIGIBLE_SPLITS,\n'
    '    "split_seed": SPLIT_SEED,\n'
    '    "directed_split_budget": DIRECTED_SPLIT_BUDGET,\n'
    '    "directed_safety_factor": DIRECTED_SAFETY_FACTOR,\n'
    '    "directed_yield_discount": DIRECTED_YIELD_DISCOUNT,\n',
)


SCRAPING_MINORITY_SOURCES = """# Fuentes priorizadas por rendimiento histórico en train.
# Las cuotas son máximos de candidatos inspeccionados y nunca etiquetas automáticas.
SEED_CHANNELS = []
SEED_SEARCH_QUERIES = []
DIRECTED_CHANNEL_CATALOG = [
    {"name": "Hablando Huevadas", "url": "https://www.youtube.com/@HablandoHuevadasOficial", "quota": 350, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Goblinciano", "url": "https://www.youtube.com/@Goblinciano", "quota": 350, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Juanito y Richard", "url": "https://www.youtube.com/@JuanitoyRichard", "quota": 250, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Nunca MAS", "url": "https://www.youtube.com/channel/UCFqwxsa2Wp6Y5FkUAcGShGA", "quota": 200, "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "PBO", "url": "https://www.youtube.com/channel/UCgR0st4ZLABi-LQcWNu3wnQ", "quota": 150, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Magaly TV La Firme", "url": "https://www.youtube.com/@MagalyTVLaFirmeATV", "quota": 150, "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Arde Troya con Juliana Oxenford", "url": "https://www.youtube.com/@ardetroyalr", "quota": 150, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
    {"name": "Todo Good", "url": "https://www.youtube.com/@todogoodpe", "quota": 150, "target_category": "RACISMO_DISCRIMINACION|ATAQUE_POR_GENERO_IDENTIDAD"},
]
DIRECTED_QUERY_CATALOG = [
    {"query": "podcast peruano insulto cholo serrano indio clasismo", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "comediante peruano racismo clasismo broma", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "streaming peruano burla regional conero provinciano", "target_category": "RACISMO_DISCRIMINACION"},
    {"query": "podcast Perú ataque por género identidad burla", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "streaming peruano burla machista misógina homofóbica", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
    {"query": "comediante peruano ataque machista homofóbico", "target_category": "ATAQUE_POR_GENERO_IDENTIDAD"},
]
for source in DIRECTED_CHANNEL_CATALOG:
    source.update({
        "country_code": "PE",
        "peru_scope_verified": True,
        "scope_evidence": "curated_peruvian_channel",
    })
"""


SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_REUSE_AND_PLAN,
    "from moderacion_peru.io import read_jsonl\n",
    "from moderacion_peru.datasets import project_effective_training_rows\n"
    "from moderacion_peru.io import read_jsonl\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "    build_directed_sampling_plan,\n",
    "    build_directed_sampling_plan,\n    estimate_directed_video_budget,\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "DIRECTED_DATASET = ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n",
    "DIRECTED_DATASET = ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n"
    "DIRECTED_CAMPAIGN = ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\n"
    "DIRECTED_REVIEWS = ROOT/'datos/etiquetado/humano/labeling_events_v2.jsonl'\n"
    "previous_snapshot_rows = list(read_jsonl(DIRECTED_DATASET)) if DIRECTED_DATASET.exists() else []\n"
    "if DIRECTED_CAMPAIGN.exists():\n"
    "    DIRECTED_TRAINING_ROWS = project_effective_training_rows(\n"
    "        read_jsonl(DIRECTED_CAMPAIGN),\n"
    "        read_jsonl(DIRECTED_REVIEWS) if DIRECTED_REVIEWS.exists() else [],\n"
    "        previous_snapshot_rows,\n"
    "        seed=SPLIT_SEED,\n"
    "    )\n"
    "else:\n"
    "    DIRECTED_TRAINING_ROWS = previous_snapshot_rows\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "        read_jsonl(DIRECTED_DATASET) if DIRECTED_DATASET.exists() else [],\n"
    "        read_jsonl(CANONICAL) if CANONICAL.exists() else [],\n"
    "        damage_labels=taxonomy.damage_labels,\n",
    "        DIRECTED_TRAINING_ROWS,\n"
    "        read_jsonl(CANONICAL) if CANONICAL.exists() else [],\n"
    "        damage_labels=taxonomy.damage_labels,\n"
    "        eligible_splits=TARGET_ELIGIBLE_SPLITS,\n"
    "        target_chunks_per_label=TARGET_TOTAL_CHUNKS_PER_DAMAGE,\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "        'soporte_videos_train_validation': directed_plan['support_videos'],\n"
    "        'déficit_videos': directed_plan['deficit_videos'],\n",
    "        'objetivo_chunks_por_daño_total': directed_plan['target_chunks_per_label'],\n"
    "        'splits_que_cuentan': TARGET_ELIGIBLE_SPLITS,\n"
    "        'soporte_chunks_total': directed_plan['support_chunks'],\n"
    "        'déficit_chunks_total': directed_plan['deficit_chunks'],\n"
    "        'adquisición_necesaria': directed_plan['acquisition_needed'],\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "    DIRECTED_SEARCH_QUERIES = select_directed_search_queries(\n",
    "    for source in DIRECTED_CHANNELS:\n"
    "        source.setdefault('quota', MAX_VIDEOS_PER_EXPANDED_CHANNEL)\n"
    "        source.update({\n"
    "            'country_code': 'PE',\n"
    "            'peru_scope_verified': True,\n"
    "            'scope_evidence': 'effective_dataset_after_foreign_exclusions',\n"
    "        })\n"
    "    DIRECTED_SEARCH_QUERIES = select_directed_search_queries(\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "        max_channels=MAX_DIRECTED_SEED_CHANNELS,\n",
    "        max_channels=MAX_DIRECTED_SEED_CHANNELS,\n"
    "        include_historical_channels=False,\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "    show_summary('Plan de ampliación dirigida', {\n",
    "    DIRECTED_ACQUISITION_BUDGET = estimate_directed_video_budget(\n"
    "        directed_plan,\n"
    "        DIRECTED_CHANNELS,\n"
    "        safety_factor=DIRECTED_SAFETY_FACTOR,\n"
    "        yield_discount=DIRECTED_YIELD_DISCOUNT,\n"
    "        minimum_train_videos=MIN_DIRECTED_TRAIN_VIDEOS,\n"
    "        validation_fraction=DIRECTED_HOLDOUT_FRACTION,\n"
    "        test_fraction=DIRECTED_HOLDOUT_FRACTION,\n"
    "    )\n"
    "    if (directed_plan['acquisition_needed']\n"
    "            and not DIRECTED_ACQUISITION_BUDGET['selected_channel_pool_capacity_sufficient']):\n"
    "        raise RuntimeError(\n"
    "            'El catálogo PE verificado no tiene capacidad histórica suficiente; '\n"
    "            'agregue canales peruanos verificados antes de descubrir videos.'\n"
    "        )\n"
    "    recommended_channel_names = set(\n"
    "        DIRECTED_ACQUISITION_BUDGET['recommended_channel_names']\n"
    "    )\n"
    "    DIRECTED_PRIMARY_CHANNELS = [\n"
    "        source for source in DIRECTED_CHANNELS\n"
    "        if str(source.get('name')) in recommended_channel_names\n"
    "    ]\n"
    "    DIRECTED_RESERVE_CHANNELS = [\n"
    "        source for source in DIRECTED_CHANNELS\n"
    "        if str(source.get('name')) not in recommended_channel_names\n"
    "    ]\n"
    "    # Se descubren también las reservas PE verificadas. La selección sigue\n"
    "    # limitada al presupuesto y solo usa reservas si faltan inéditos del núcleo.\n"
    "    DIRECTED_CHANNELS = DIRECTED_PRIMARY_CHANNELS + DIRECTED_RESERVE_CHANNELS\n"
    "    DIRECTED_SPLIT_BUDGET = DIRECTED_ACQUISITION_BUDGET['split_budget']\n"
    "    MAX_DIRECTED_CANDIDATES = DIRECTED_ACQUISITION_BUDGET['total_candidate_videos']\n"
    "    show_summary('Plan de ampliación dirigida', {\n",
)
SCRAPING_MINORITY_REUSE_AND_PLAN = _replace_required(
    SCRAPING_MINORITY_REUSE_AND_PLAN,
    "        'consultas_temáticas': len(DIRECTED_SEARCH_QUERIES),\n",
    "        'consultas_temáticas': len(DIRECTED_SEARCH_QUERIES),\n"
    "        'canales_núcleo': DIRECTED_ACQUISITION_BUDGET['core_channel_count'],\n"
    "        'nombres_canales_núcleo': DIRECTED_ACQUISITION_BUDGET['core_channel_names'],\n"
    "        'canales_margen': DIRECTED_ACQUISITION_BUDGET['margin_channels'],\n"
    "        'nombres_canales_margen': DIRECTED_ACQUISITION_BUDGET['margin_channel_names'],\n"
    "        'canales_recomendados': DIRECTED_ACQUISITION_BUDGET['recommended_channel_count'],\n"
    "        'canales_PE_de_reserva': len(DIRECTED_RESERVE_CHANNELS),\n"
    "        'nombres_canales_PE_de_reserva': [\n"
    "            source.get('name') for source in DIRECTED_RESERVE_CHANNELS\n"
    "        ],\n"
    "        'presupuesto_videos_por_split': DIRECTED_SPLIT_BUDGET,\n"
    "        'estimación_por_daño': DIRECTED_ACQUISITION_BUDGET['per_label'],\n",
)


_DEFAULT_SELECTION = """            directed_selection = select_directed_candidates(
                directed_pool,
                KNOWN_VIDEO_IDS,
                directed_plan,
                max_candidates=(
                    len(directed_pool)
                    if MAX_DIRECTED_CANDIDATES is None
                    else MAX_DIRECTED_CANDIDATES
                ),
            )
"""
_MINORITY_SELECTION = """            directed_round_signature = sha256_text(json.dumps({
                'deficit_chunks': directed_plan['deficit_chunks'],
                'support_chunks': directed_plan['support_chunks'],
                'recommended_channels': DIRECTED_ACQUISITION_BUDGET['recommended_channel_names'],
                'discovery_channels': [source.get('name') for source in DIRECTED_CHANNELS],
                'split_budget': DIRECTED_SPLIT_BUDGET,
                'target_total_chunks_per_damage': TARGET_TOTAL_CHUNKS_PER_DAMAGE,
                'target_eligible_splits': TARGET_ELIGIBLE_SPLITS,
                'split_seed': SPLIT_SEED,
                'country_scope': 'PE_only_strict',
            }, ensure_ascii=False, sort_keys=True))
            previous_plan_payload = (
                json.loads(DIRECTED_PLAN_PATH.read_text(encoding='utf-8-sig'))
                if DIRECTED_PLAN_PATH.exists() else {}
            )
            previous_round_selection = (
                load_candidates(DIRECTED_SELECTION_PATH)
                if DIRECTED_SELECTION_PATH.exists() else []
            )
            previous_carryover = (
                load_candidates(DIRECTED_CARRYOVER_PATH)
                if DIRECTED_CARRYOVER_PATH.exists() else []
            )
            resumed_existing_round = (
                DIRECTED_SELECTION_PATH.exists()
                and previous_plan_payload.get('directed_round_signature')
                == directed_round_signature
            )
            completed_before_resume = 0
            refilled_on_resume = 0
            if resumed_existing_round:
                directed_selection = [
                    candidate for candidate in previous_round_selection
                    if str(candidate.get('video_id') or '').strip() not in KNOWN_VIDEO_IDS
                ]
                completed_before_resume = (
                    int(previous_plan_payload.get('completed_before_resume') or 0)
                    + len(previous_round_selection) - len(directed_selection)
                )
                directed_split_shortfall = {
                    split: int(
                        (previous_plan_payload.get('directed_split_shortfall') or {}).get(
                            split, 0
                        )
                    )
                    for split in DIRECTED_SPLIT_BUDGET
                }
                pending_ids = {
                    str(candidate.get('video_id') or '').strip()
                    for candidate in directed_selection
                }
                for planned_split, requested in tuple(directed_split_shortfall.items()):
                    if requested <= 0:
                        continue
                    refill = select_directed_candidates(
                        directed_pool,
                        KNOWN_VIDEO_IDS | pending_ids,
                        directed_plan,
                        max_candidates=requested,
                        required_split=planned_split,
                        split_seed=SPLIT_SEED,
                    )
                    directed_selection = merge_candidates(directed_selection, refill)
                    pending_ids.update(
                        str(candidate.get('video_id') or '').strip()
                        for candidate in refill
                    )
                    refilled_on_resume += len(refill)
                    directed_split_shortfall[planned_split] = max(
                        0, requested - len(refill)
                    )
            else:
                selection_limit = (
                    len(directed_pool)
                    if MAX_DIRECTED_CANDIDATES is None
                    else MAX_DIRECTED_CANDIDATES
                )
                directed_selection = []
                remaining_selection = selection_limit
                for planned_split, requested in DIRECTED_SPLIT_BUDGET.items():
                    split_selection = select_directed_candidates(
                        directed_pool,
                        KNOWN_VIDEO_IDS,
                        directed_plan,
                        max_candidates=min(requested, remaining_selection),
                        required_split=planned_split,
                        split_seed=SPLIT_SEED,
                    )
                    directed_selection = merge_candidates(directed_selection, split_selection)
                    remaining_selection -= len(split_selection)
                directed_selection = [
                    {**candidate, 'directed_selection_rank': index}
                    for index, candidate in enumerate(directed_selection, 1)
                ]
                fresh_split_counts = Counter(
                    candidate.get('planned_split') for candidate in directed_selection
                )
                directed_split_shortfall = {
                    split: max(0, requested - fresh_split_counts.get(split, 0))
                    for split, requested in DIRECTED_SPLIT_BUDGET.items()
                }
            directed_split_counts = dict(Counter(
                candidate.get('planned_split') for candidate in directed_selection
            ))
            active_selection_ids = {
                str(candidate.get('video_id') or '').strip()
                for candidate in directed_selection
            }
            carryover_sources = (
                previous_carryover
                if resumed_existing_round
                else merge_candidates(previous_carryover, previous_round_selection)
            )
            directed_carryover = [
                candidate for candidate in carryover_sources
                if (
                    str(candidate.get('video_id') or '').strip()
                    and str(candidate.get('video_id') or '').strip() not in KNOWN_VIDEO_IDS
                    and str(candidate.get('video_id') or '').strip()
                    not in active_selection_ids
                )
            ]
            directed_selection_incomplete = any(directed_split_shortfall.values())
            write_jsonl_atomic(PERU_SCOPE_EXCLUSIONS_PATH, peru_scope_excluded)
            write_jsonl_atomic(DIRECTED_CARRYOVER_PATH, directed_carryover)
"""
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_DISCOVERY, _DEFAULT_SELECTION, _MINORITY_SELECTION
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "from collections import Counter\n",
    "from collections import Counter\nimport json\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "from moderacion_peru.io import append_jsonl_once, write_json_atomic, write_jsonl_atomic\n",
    "from moderacion_peru.io import (\n"
    "    append_jsonl_once, sha256_text, write_json_atomic, write_jsonl_atomic,\n"
    ")\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "    discover_youtube_candidates,\n",
    "    discover_youtube_candidates,\n    filter_peru_candidates,\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "DIRECTED_PLAN_PATH = ROOT/'datos/raw/manifests/directed_plan_latest.json'\n",
    "DIRECTED_PLAN_PATH = ROOT/'datos/raw/manifests/directed_plan_latest.json'\n"
    "DIRECTED_CARRYOVER_PATH = ROOT/'datos/raw/directed_candidates_carryover.jsonl'\n"
    "PERU_SCOPE_EXCLUSIONS_PATH = ROOT/'datos/raw/manifests/directed_non_peru_excluded_latest.jsonl'\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "expanded_channels = []\n",
    "expanded_channels = []\nperu_scope_excluded = []\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "        if directed_plan is not None:\n",
    "        allowed_peru_channel_ids = [\n"
    "            source.get('channel_id') for source in DIRECTED_CHANNELS\n"
    "            if source.get('channel_id')\n"
    "        ]\n"
    "        allowed_peru_channel_titles = [\n"
    "            source.get('name') for source in DIRECTED_CHANNELS\n"
    "            if source.get('name')\n"
    "        ]\n"
    "        discovered, excluded_now = filter_peru_candidates(\n"
    "            discovered,\n"
    "            allowed_channel_ids=allowed_peru_channel_ids,\n"
    "            allowed_channel_titles=allowed_peru_channel_titles,\n"
    "        )\n"
    "        peru_scope_excluded.extend(excluded_now)\n"
    "        if directed_plan is not None:\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "            if expanded_channels:\n",
    "            for source in expanded_channels:\n"
    "                source.update({\n"
    "                    'country_code': 'PE',\n"
    "                    'peru_scope_verified': True,\n"
    "                    'scope_evidence': 'verified_peru_search_candidate',\n"
    "                })\n"
    "            if expanded_channels:\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "                discovered = merge_candidates(discovered, expanded_candidates)\n",
    "                discovered = merge_candidates(discovered, expanded_candidates)\n"
    "                discovered, excluded_now = filter_peru_candidates(\n"
    "                    discovered,\n"
    "                    allowed_channel_ids=[\n"
    "                        *allowed_peru_channel_ids,\n"
    "                        *(source.get('channel_id') for source in expanded_channels),\n"
    "                    ],\n"
    "                    allowed_channel_titles=[\n"
    "                        *allowed_peru_channel_titles,\n"
    "                        *(source.get('name') for source in expanded_channels),\n"
    "                    ],\n"
    "                )\n"
    "                peru_scope_excluded.extend(excluded_now)\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "                'directed_candidates_selected': len(directed_selection),\n",
    "                'directed_candidates_selected': len(directed_selection),\n"
    "                'directed_split_counts': directed_split_counts,\n"
    "                'directed_split_shortfall': directed_split_shortfall,\n"
    "                'target_total_chunks_per_damage': TARGET_TOTAL_CHUNKS_PER_DAMAGE,\n"
    "                'target_eligible_splits': TARGET_ELIGIBLE_SPLITS,\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "                'target_total_chunks_per_damage': TARGET_TOTAL_CHUNKS_PER_DAMAGE,\n"
    "                'target_eligible_splits': TARGET_ELIGIBLE_SPLITS,\n"
    "                'selection_path': DIRECTED_SELECTION_PATH,\n",
    "                'target_total_chunks_per_damage': TARGET_TOTAL_CHUNKS_PER_DAMAGE,\n"
    "                'target_eligible_splits': TARGET_ELIGIBLE_SPLITS,\n"
    "                'acquisition_budget': DIRECTED_ACQUISITION_BUDGET,\n"
    "                'directed_round_signature': directed_round_signature,\n"
    "                'resumed_existing_round': resumed_existing_round,\n"
    "                'completed_before_resume': completed_before_resume,\n"
    "                'refilled_on_resume': refilled_on_resume,\n"
    "                'pending_in_resumed_round': len(directed_selection),\n"
    "                'selection_incomplete': directed_selection_incomplete,\n"
    "                'carryover_pending': len(directed_carryover),\n"
    "                'carryover_path': DIRECTED_CARRYOVER_PATH,\n"
    "                'selection_path': DIRECTED_SELECTION_PATH,\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "                'selection_path': DIRECTED_SELECTION_PATH,\n",
    "                'selection_path': DIRECTED_SELECTION_PATH,\n"
    "                'country_scope': 'PE_only_strict',\n"
    "                'non_peru_or_unverified_excluded': len(peru_scope_excluded),\n"
    "                'scope_exclusions_path': PERU_SCOPE_EXCLUSIONS_PATH,\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "    added, existing = append_jsonl_once(DISCOVERED_PATH, discovered, id_field='video_id')\n",
    "    write_jsonl_atomic(PERU_SCOPE_EXCLUSIONS_PATH, peru_scope_excluded)\n"
    "    added, existing = append_jsonl_once(DISCOVERED_PATH, discovered, id_field='video_id')\n",
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    '        "directed_cohort": len(directed_selection),\n',
    '        "directed_cohort": len(directed_selection),\n'
    '        "country_scope": "PE_only_strict",\n'
    '        "non_peru_or_unverified_excluded": len(peru_scope_excluded),\n'
    '        "scope_exclusions_path": PERU_SCOPE_EXCLUSIONS_PATH,\n'
    '        "directed_split_counts": (directed_split_counts if directed_plan is not None else {}),\n'
    '        "directed_split_shortfall": (directed_split_shortfall if directed_plan is not None else {}),\n'
    '        "selection_incomplete": (directed_selection_incomplete if directed_plan is not None else False),\n'
    '        "carryover_pending": (len(directed_carryover) if directed_plan is not None else 0),\n'
    '        "carryover_path": (DIRECTED_CARRYOVER_PATH if directed_plan is not None else None),\n'
    '        "resumed_existing_round": (resumed_existing_round if directed_plan is not None else False),\n'
    '        "completed_before_resume": (completed_before_resume if directed_plan is not None else 0),\n',
)
SCRAPING_MINORITY_DISCOVERY = _replace_required(
    SCRAPING_MINORITY_DISCOVERY,
    "    }, tone='success' if not discovery_failures else 'warning')\n"
    "    if discovery_failures:\n",
    "    }, tone=(\n"
    "        'warning' if (\n"
    "            discovery_failures\n"
    "            or (directed_plan is not None and directed_selection_incomplete)\n"
    "        ) else 'success'\n"
    "    ))\n"
    "    if directed_plan is not None and directed_selection_incomplete:\n"
    "        show_callout(\n"
    "            'Cohorte PE parcial: se continuará sin perder avances',\n"
    "            'Se procesarán ahora los videos PE disponibles. El faltante por split '\n"
    "            f'{directed_split_shortfall} quedó guardado y podrá rellenarse al '\n"
    "            'reanudar o en la siguiente ronda después del etiquetado.',\n"
    "            tone='warning',\n"
    "        )\n"
    "    if discovery_failures:\n",
)

SCRAPING_MINORITY_CANDIDATES = _replace_required(
    SCRAPING_CANDIDATES,
    "if DISCOVERY_MODE == 'directed':\n"
    "    candidates = load_candidates(DIRECTED_SELECTION_PATH)\n"
    "    candidate_origin = 'cohorte_dirigida_vigente'\n",
    "if DISCOVERY_MODE == 'directed':\n"
    "    active_candidates = load_candidates(DIRECTED_SELECTION_PATH)\n"
    "    carryover_candidates = load_candidates(DIRECTED_CARRYOVER_PATH)\n"
    "    candidates = merge_candidates(active_candidates, carryover_candidates)\n"
    "    candidate_origin = 'cohorte_dirigida_vigente_más_arrastre'\n",
)
SCRAPING_MINORITY_CANDIDATES = _replace_required(
    SCRAPING_MINORITY_CANDIDATES,
    "    'únicos': len(candidates),\n",
    "    'únicos': len(candidates),\n"
    "    'cohorte_vigente': len(active_candidates),\n"
    "    'arrastre_histórico': len(carryover_candidates),\n"
    "    'duplicados_cohorte_arrastre_eliminados': (\n"
    "        len(active_candidates) + len(carryover_candidates) - len(candidates)\n"
    "    ),\n",
)


def preserve_matching_execution_outputs(
    notebook: nbf.NotebookNode, target: Path
) -> None:
    """Keep executed evidence when regenerating a notebook with unchanged code cells."""

    if not target.is_file():
        return
    previous = nbf.read(target, as_version=4)
    previous_by_source: dict[str, list[nbf.NotebookNode]] = {}
    for cell in previous.cells:
        if cell.cell_type != "code" or not cell.get("outputs"):
            continue
        previous_by_source.setdefault(cell.source, []).append(cell)
    for cell in notebook.cells:
        candidates = previous_by_source.get(cell.source, [])
        if cell.cell_type != "code" or len(candidates) != 1:
            continue
        previous_cell = candidates[0]
        cell["execution_count"] = previous_cell.get("execution_count")
        cell["outputs"] = previous_cell.get("outputs", [])


PERSISTENT_COLAB_TRAINING_ACTIVITY = {
    "03_02": "RUN_TRAINING or RUN_CHANNEL_ROBUSTNESS",
    "03_03": "RUN_TRAINING",
    "03_03b": "RUN_TRAINING",
    "03_04": "RUN_TRAINING",
    "03_05": "RUN_TRAINING",
    "03_06": "RUN_LEGACY_FULL_TRAINING or RUN_STRUCTURED_LORA_SWEEP",
    "03_06b": "RUN_BUDGETED_COMPARABLE or RUN_DIAGNOSTIC_PILOT or RUN_FULL_TRAINING",
}


QWEN_LORA_BASE_SOURCE = (
    "from moderacion_peru.experiments import train_neural_experiment\n"
    "DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n"
    "OUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_lora'\n"
    "DEVICE='cuda' if COLAB_CONTEXT else 'auto'\n"
    "PERSISTENT_CHECKPOINT_ROOT=COLAB_CONTEXT.drive_run_dir/'trainer_checkpoints' if COLAB_CONTEXT else None\n"
    "RUN_TRAINING=False  # Solo actívelo si falta el candidato base de 128 tokens\n"
    "if RUN_TRAINING:\n"
    "    qwen_lora_result=run_with_progress('Qwen-LoRA 128 tokens',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='qwen_lora',device=DEVICE,safe_to_damage_ratio=4.0,persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='etapa')\n"
    "    show_result('Qwen-LoRA base de 128 tokens',qwen_lora_result,tone='success')\n"
    "else:\n"
    "    show_summary('Candidato base conservado',{'acción':'no se reentrena ni reemplaza el candidato de 128 tokens','método':'clasificador supervisado, no condicionado por prompt','longitud':128,'salidas':'5+14+3 enmascaradas','perfil_GPU':'A100: BF16, lote 8×1, validation 32 y 2 workers; fallback 2×4','checkpoints_drive':PERSISTENT_CHECKPOINT_ROOT,'SEGURO_train_validation':'4:1 fijo','test':'natural completo, sellado'},tone='neutral')"
)


QWEN_LORA_256_SOURCE = (
    "from moderacion_peru.experiments import select_qwen_lora_warm_start_candidate, train_neural_experiment\n"
    "CONTINUATION_MAX_LENGTH=256\n"
    "CONTINUATION_EPOCHS=4  # máximo adicional; early stopping conserva la mejor validation\n"
    "CONTINUATION_VARIANT_ID='context256_from128'\n"
    "RUN_CONTINUATION_256=True\n"
    "if RUN_CONTINUATION_256:\n"
    "    base_128=select_qwen_lora_warm_start_candidate(OUTPUT_ROOT,DATA,max_length=128)\n"
    "    show_result('Warm-start Qwen-LoRA verificado',{'candidate_id':base_128['candidate_id'],'candidate_path':base_128['candidate_path'],'dataset_sha256':base_128['dataset_sha256'],'longitud_origen':base_128['truncation_diagnostic']['max_length'],'longitud_destino':CONTINUATION_MAX_LENGTH,'optimizador':'nuevo; no reutiliza estado del Trainer'},tone='success')\n"
    "    qwen_lora_256_result=run_with_progress('Qwen-LoRA 256 desde 128',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='qwen_lora',device=DEVICE,max_length=CONTINUATION_MAX_LENGTH,epochs=CONTINUATION_EPOCHS,variant_id=CONTINUATION_VARIANT_ID,warm_start_candidate_path=base_128['candidate_path'],safe_to_damage_ratio=4.0,persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='etapa')\n"
    "    show_result('Candidato adicional Qwen-LoRA de 256 tokens',qwen_lora_256_result,tone='success')\n"
    "    if COLAB_CONTEXT is not None:\n"
    "        from moderacion_peru.colab import publish_colab_outputs\n"
    "        show_result('Publicación del candidato 256',publish_colab_outputs(COLAB_CONTEXT),tone='success')\n"
    "else:\n"
    "    show_summary('Continuación 256 desactivada',{'padre_requerido':'Qwen-LoRA completo de 128 tokens y mismo dataset','variante':CONTINUATION_VARIANT_ID,'longitud':CONTINUATION_MAX_LENGTH,'épocas_adicionales_máximas':CONTINUATION_EPOCHS,'selección':'validation; 03_07 comparará ambos candidatos','test':'permanece sellado'},tone='neutral')"
)


MINILM_IMPROVEMENTS_SOURCE = """from moderacion_peru.experiments import select_neural_warm_start_candidate,train_flat_transformers,train_neural_experiment
DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
OUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/transformers_planos'
DEVICE='cuda' if COLAB_CONTEXT else 'auto'
SAFE_TO_DAMAGE_RATIO=4.0
SAMPLING_SEED=20260805  # fija exactamente las mismas filas train/validation en todas las comparaciones
PRIMARY_TRAINING_SEED=20260805
ADDITIONAL_TRAINING_SEEDS=(20260817,20260829)
CONTEXT_SCREEN_LENGTHS=(192,256)
SELECTED_MINILM_CONTEXT=192  # cámbielo solo después de comparar la pantalla en validation
CONTINUATION_EPOCHS=1
CONTINUATION_LEARNING_RATE=1e-5
FOCAL_GAMMA=2.0
RUN_TRAINING=False  # baselines MiniLM y E5 de 128 tokens
RUN_CHANNEL_ROBUSTNESS=False
RUN_MINILM_CONTEXT_SCREEN=False
RUN_MINILM_SEED_CONFIRMATION=False
RUN_MINILM_FOCAL_ABLATION=False

def publish_minilm_variant(label):
    if COLAB_CONTEXT is None: return
    from moderacion_peru.colab import publish_colab_outputs
    show_result(label,publish_colab_outputs(COLAB_CONTEXT),tone='success')

if RUN_TRAINING:
    flat_result=run_with_progress('Transformers planos',train_flat_transformers,DATA,OUTPUT_ROOT,device=DEVICE,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,sampling_seed=SAMPLING_SEED,progress_unit='modelo')
    show_result('Transformers planos 22 salidas',flat_result,tone='success')
if RUN_CHANNEL_ROBUSTNESS:
    robustness_result=run_with_progress('MiniLM por canal',train_neural_experiment,DATA,OUTPUT_ROOT/'channel_heldout',experiment='flat_minilm',device=DEVICE,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,split_scheme='channel',sampling_seed=SAMPLING_SEED,training_seed=PRIMARY_TRAINING_SEED,progress_unit='etapa')
    show_result('MiniLM con canales retenidos',robustness_result,tone='success')

RUN_MINILM_IMPROVEMENTS=RUN_MINILM_CONTEXT_SCREEN or RUN_MINILM_SEED_CONFIRMATION or RUN_MINILM_FOCAL_ABLATION
if RUN_MINILM_IMPROVEMENTS:
    minilm_parent=select_neural_warm_start_candidate(OUTPUT_ROOT,DATA,experiment='flat_minilm',max_length=128)
    show_result('MiniLM padre verificado',{'candidate_id':minilm_parent['candidate_id'],'dataset_sha256':minilm_parent['dataset_sha256'],'contexto_padre':128,'checkpoint':minilm_parent['candidate_path'],'optimizador':'nuevo en cada variante'},tone='success')

if RUN_MINILM_CONTEXT_SCREEN:
    for context_length in CONTEXT_SCREEN_LENGTHS:
        variant=f'context{context_length}_bce_seed{PRIMARY_TRAINING_SEED}'
        result=run_with_progress(f'MiniLM {context_length} tokens · BCE',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='flat_minilm',device=DEVICE,max_length=context_length,epochs=CONTINUATION_EPOCHS,learning_rate=CONTINUATION_LEARNING_RATE,training_seed=PRIMARY_TRAINING_SEED,sampling_seed=SAMPLING_SEED,variant_id=variant,warm_start_candidate_path=minilm_parent['candidate_path'],loss_mode='weighted_bce',safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,progress_unit='etapa')
        show_result(f'Candidato {variant}',result,tone='success')
        publish_minilm_variant(f'Publicación {variant}')

if RUN_MINILM_SEED_CONFIRMATION:
    for training_seed in ADDITIONAL_TRAINING_SEEDS:
        variant=f'context{SELECTED_MINILM_CONTEXT}_bce_seed{training_seed}'
        result=run_with_progress(f'MiniLM {SELECTED_MINILM_CONTEXT} tokens · semilla {training_seed}',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='flat_minilm',device=DEVICE,max_length=SELECTED_MINILM_CONTEXT,epochs=CONTINUATION_EPOCHS,learning_rate=CONTINUATION_LEARNING_RATE,training_seed=training_seed,sampling_seed=SAMPLING_SEED,variant_id=variant,warm_start_candidate_path=minilm_parent['candidate_path'],loss_mode='weighted_bce',safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,progress_unit='etapa')
        show_result(f'Candidato {variant}',result,tone='success')
        publish_minilm_variant(f'Publicación {variant}')

if RUN_MINILM_FOCAL_ABLATION:
    variant=f'context{SELECTED_MINILM_CONTEXT}_focal_g2_seed{PRIMARY_TRAINING_SEED}'
    focal_result=run_with_progress(f'MiniLM {SELECTED_MINILM_CONTEXT} tokens · focal',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='flat_minilm',device=DEVICE,max_length=SELECTED_MINILM_CONTEXT,epochs=CONTINUATION_EPOCHS,learning_rate=CONTINUATION_LEARNING_RATE,training_seed=PRIMARY_TRAINING_SEED,sampling_seed=SAMPLING_SEED,variant_id=variant,warm_start_candidate_path=minilm_parent['candidate_path'],loss_mode='focal',focal_gamma=FOCAL_GAMMA,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,progress_unit='etapa')
    show_result(f'Ablación {variant}',focal_result,tone='success')
    publish_minilm_variant(f'Publicación {variant}')

if not (RUN_TRAINING or RUN_CHANNEL_ROBUSTNESS or RUN_MINILM_IMPROVEMENTS):
    show_summary('Campaña MiniLM preparada',{'baseline':'128 tokens, intacto','fase_1':'192 frente a 256; BCE, una época y la misma validation','fase_2':'contexto elegido con 3 semillas de entrenamiento en total','ablación':'focal gamma=2 frente a BCE con la semilla primaria','separación_semillas':f'sampling={SAMPLING_SEED} fijo; solo cambia inicialización/orden','warm_start':'mejor checkpoint completo del candidato 128; optimizador nuevo','learning_rate':CONTINUATION_LEARNING_RATE,'test':'natural completo, sellado'},tone='neutral')"""


QWEN_STRUCTURED_IMPROVEMENTS_SOURCE = """from moderacion_peru.experiments import select_qwen_lora_warm_start_candidate,train_neural_experiment
DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
OUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_estructurado'
DEVICE='cuda' if COLAB_CONTEXT else 'auto'
PERSISTENT_CHECKPOINT_ROOT=COLAB_CONTEXT.drive_run_dir/'trainer_checkpoints' if COLAB_CONTEXT else None
QWEN_LORA_PARENT_RUN_ID='03_05_working_v2_1'
QWEN_LORA_PARENT_MAX_LENGTH=128
STRUCTURED_PENALTIES=(0.0,0.02,0.05)
STRUCTURED_EPOCHS=1
STRUCTURED_LEARNING_RATE=2e-5
SAMPLING_SEED=20260805
TRAINING_SEED=20260805
RUN_LEGACY_FULL_TRAINING=False  # conserva el experimento histórico; no es la opción recomendada
RUN_STRUCTURED_LORA_SWEEP=False  # recomendado: tres adaptadores pequeños desde 03_05

def publish_structured_variant(label):
    if COLAB_CONTEXT is None: return
    from moderacion_peru.colab import publish_colab_outputs
    show_result(label,publish_colab_outputs(COLAB_CONTEXT),tone='success')

if RUN_LEGACY_FULL_TRAINING:
    legacy_result=run_with_progress('Qwen estructurado · full fine-tuning histórico',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='qwen_structured',device=DEVICE,safe_to_damage_ratio=4.0,persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='etapa')
    show_result('Candidato histórico independiente',legacy_result,tone='warning')

if RUN_STRUCTURED_LORA_SWEEP:
    if COLAB_CONTEXT is not None:
        from moderacion_peru.colab import restore_colab_run_outputs
        QWEN_LORA_PARENT_ROOT=COLAB_CONTEXT.runtime_root/'warm_starts'/'03_05'/QWEN_LORA_PARENT_RUN_ID
        restored_parent=restore_colab_run_outputs(COLAB_CONTEXT.drive_root,notebook_id='03_05',run_id=QWEN_LORA_PARENT_RUN_ID,destination=QWEN_LORA_PARENT_ROOT)
        show_result('Publicación 03_05 restaurada',restored_parent,tone='success')
    else:
        QWEN_LORA_PARENT_ROOT=ROOT/'modelos/v2/qwen_lora'
    qwen_lora_parent=select_qwen_lora_warm_start_candidate(QWEN_LORA_PARENT_ROOT,DATA,max_length=QWEN_LORA_PARENT_MAX_LENGTH)
    show_result('Adaptador padre verificado',{'candidate_id':qwen_lora_parent['candidate_id'],'candidate_path':qwen_lora_parent['candidate_path'],'dataset_sha256':qwen_lora_parent['dataset_sha256'],'contexto':QWEN_LORA_PARENT_MAX_LENGTH,'optimizador':'nuevo por variante'},tone='success')
    for penalty in STRUCTURED_PENALTIES:
        penalty_code=f'{int(round(penalty*100)):03d}'
        variant=f'lora03_05_structured_p{penalty_code}'
        result=run_with_progress(f'Qwen estructurado LoRA · penalización {penalty:.2f}',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='qwen_structured',device=DEVICE,max_length=QWEN_LORA_PARENT_MAX_LENGTH,epochs=STRUCTURED_EPOCHS,learning_rate=STRUCTURED_LEARNING_RATE,training_seed=TRAINING_SEED,sampling_seed=SAMPLING_SEED,variant_id=variant,warm_start_candidate_path=qwen_lora_parent['candidate_path'],structured_penalty=penalty,loss_mode='weighted_bce',safe_to_damage_ratio=4.0,persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='etapa')
        show_result(f'Candidato {variant}',result,tone='success')
        publish_structured_variant(f'Publicación {variant}')

if not (RUN_LEGACY_FULL_TRAINING or RUN_STRUCTURED_LORA_SWEEP):
    show_summary('Qwen estructurado de bajo costo preparado',{'recomendado':'LoRA entrenable restaurado desde el candidato 03_05','comparación':'penalización 0, 0.02 y 0.05 sobre las mismas filas','épocas':STRUCTURED_EPOCHS,'learning_rate':STRUCTURED_LEARNING_RATE,'semillas':f'sampling={SAMPLING_SEED}; training={TRAINING_SEED}','estado_optimizador':'nuevo en cada candidato','candidato_full_histórico':'se conserva; no se sobrescribe','test':'natural completo, sellado'},tone='neutral')"""


COMPARISON_DRIVE_RESTORE_SOURCE = """from pathlib import Path
import shutil
import time
from moderacion_peru.colab import restore_colab_run_outputs
from moderacion_peru.ensemble_evaluation import audit_validation_candidate_eligibility

DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
LOCAL_CANDIDATE_ROOT=ROOT/'modelos/v2'
# Si usa Google Drive para escritorio, indique aquí la carpeta ModeracionPeru_Colab.
# En un kernel Colab se monta y detecta automáticamente; no requiere Drive Desktop.
LOCAL_GOOGLE_DRIVE_ROOT=None  # p. ej. Path('G:/My Drive/ModeracionPeru_Colab')
RESTORE_CANDIDATES_FROM_DRIVE=True
DRIVE_RUN_IDS={
    '03_01':'03_01_working_v2_1',
    '03_02':'03_02_working_v2_1',
    '03_03':'03_03_working_v2_1',
    '03_03b':'03_03b_working_v2_1',
    '03_04':'03_04_working_v2_1',
    '03_05':'03_05_working_v2_1',
    '03_06':'03_06_working_v2_1',
    '03_06b':'03_06b_working_v2_1',
}
OPTIONAL_DRIVE_RUNS={'03_06b'}
REQUIRED_FAMILIES={
    '03_01':('classical:',),
    '03_02':('flat_minilm','flat_e5'),
    '03_03':('cascade',),
    '03_03b':('cascade_v2',),
    '03_04':('multitask',),
    '03_05':('qwen_lora',),
    '03_06':('qwen_structured',),
}

drive_candidate_root=(
    COLAB_CONTEXT.runtime_root/'comparison_inputs'/'03_07'
    if COLAB_CONTEXT is not None
    else ROOT/'modelos/v2/restored_from_drive'
)
drive_root=(
    COLAB_CONTEXT.drive_root
    if COLAB_CONTEXT is not None
    else Path(LOCAL_GOOGLE_DRIVE_ROOT).expanduser()
    if LOCAL_GOOGLE_DRIVE_ROOT is not None
    else None
)
CANDIDATE_ROOTS=[LOCAL_CANDIDATE_ROOT]
restoration=[]
if RESTORE_CANDIDATES_FROM_DRIVE and drive_root is not None:
    for restore_index,(notebook_id,run_id) in enumerate(DRIVE_RUN_IDS.items(),start=1):
        destination=drive_candidate_root/notebook_id/run_id
        show_callout(
            f'Restauración {restore_index}/{len(DRIVE_RUN_IDS)} · {notebook_id}',
            f'Verificando SHA-256 y extrayendo {run_id}. Esta etapa puede tardar varios minutos.',
            tone='info',
        )
        restore_started=time.perf_counter()
        try:
            restored=restore_colab_run_outputs(drive_root,notebook_id=notebook_id,run_id=run_id,destination=destination)
            restore_entry={'notebook_id':notebook_id,'run_id':run_id,**restored}
        except (FileNotFoundError,ValueError) as exc:
            corrupt=isinstance(exc,ValueError)
            restore_entry={
                'notebook_id':notebook_id,
                'run_id':run_id,
                'status':(
                    'corrupt_optional' if corrupt and notebook_id in OPTIONAL_DRIVE_RUNS else
                    'corrupt_required' if corrupt else
                    'missing_optional' if notebook_id in OPTIONAL_DRIVE_RUNS else
                    'missing_required'
                ),
                'source':str(Path(drive_root)/'runs'/notebook_id/run_id),
                'detail':str(exc),
            }
        restoration.append(restore_entry)
        disk=shutil.disk_usage(drive_candidate_root)
        show_result(f'Estado {restore_index}/{len(DRIVE_RUN_IDS)} · {notebook_id}',{
            'status':restore_entry['status'],
            'minutos':round((time.perf_counter()-restore_started)/60,2),
            'disco_usado_GiB':round((disk.total-disk.free)/(1024**3),2),
            'disco_libre_GiB':round(disk.free/(1024**3),2),
            'detalle':restore_entry.get('detail'),
        },tone='success' if restore_entry['status'].startswith(('restored','existing')) else 'warning')
    CANDIDATE_ROOTS.append(drive_candidate_root)
elif RESTORE_CANDIDATES_FROM_DRIVE:
    show_callout(
        'Drive no está montado en el kernel local',
        'Seleccione un kernel Google Colab y ejecute desde la primera celda, o configure LOCAL_GOOGLE_DRIVE_ROOT si usa Drive para escritorio.',
        tone='warning',
    )

candidate_audit=audit_validation_candidate_eligibility(DATA,CANDIDATE_ROOTS)
eligible_candidates=[
    {
        'candidate_id':row.get('candidate_id'),
        'model_family':row.get('model_family'),
        'training_regime':row.get('training_regime','full'),
        'candidate_path':row.get('candidate_path'),
    }
    for row in candidate_audit['eligible']
]
eligible_families={str(row.get('model_family','')).casefold() for row in candidate_audit['eligible']}

def family_is_present(expected_family):
    expected=str(expected_family).casefold()
    return any(
        family.startswith(expected) if expected.endswith(':') else family==expected
        for family in eligible_families
    )

missing_required_families={
    notebook_id:[family for family in families if not family_is_present(family)]
    for notebook_id,families in REQUIRED_FAMILIES.items()
}
missing_required_families={key:value for key,value in missing_required_families.items() if value}
prompt_sft_active=family_is_present('qwen_prompt_sft')
required_restore_failures=[
    row for row in restoration
    if row.get('status') in {'missing_required','corrupt_required'}
]
CANDIDATE_PREFLIGHT_READY=(
    candidate_audit['eligible_count']>0
    and not missing_required_families
    and not required_restore_failures
)
show_result('Restauración de publicaciones 03_01–03_06b',{
    'drive_root':drive_root,
    'runs':restoration,
    '03_06b':'incluido' if prompt_sft_active else 'no encontrado o no elegible; omitido',
},tone='success' if CANDIDATE_PREFLIGHT_READY else 'warning')
show_result('Elegibilidad previa a 03_07',{
    'dataset_sha256':candidate_audit['dataset_sha256'],
    'descubiertos':candidate_audit['discovered_count'],
    'elegibles':eligible_candidates,
    'rechazados':candidate_audit['rejected'],
    'publicaciones_requeridas_fallidas':required_restore_failures,
    'familias_requeridas_ausentes':missing_required_families,
    'listo_para_comparar':CANDIDATE_PREFLIGHT_READY,
},tone='success' if CANDIDATE_PREFLIGHT_READY else 'warning')
if prompt_sft_active:
    show_callout('03_06b activado','Existe un candidato completo, con validation común y test sellado; entrará en la comparación.',tone='success')
else:
    show_callout('03_06b omitido','Su ausencia o inelegibilidad no bloquea la comparación de 03_01–03_06.',tone='neutral')"""


COMPARISON_RUN_SOURCE = """from moderacion_peru.ensemble_evaluation import compare_and_freeze_validation,evaluate_frozen_test
RESULT_ROOT=(COLAB_CONTEXT.scratch_output_dir/'resultados_modelos' if COLAB_CONTEXT else ROOT/'resultados/modelos')
COMPARISON=RESULT_ROOT/'comparacion_individual_ensemble_validation.json'
FREEZE=RESULT_ROOT/'seleccion_congelada.json'
TEST_REPORT=RESULT_ROOT/'test_final_abierto_una_vez.json'
PARALLEL_WORKERS=4  # bootstrap pareado por video
BOOTSTRAP_REPLICATES=2000
SELECTION_FOLDS=5
# Deben acordarse ANTES de comparar. None permite el informe/Pareto, pero mantiene test sellado.
MAX_REVIEW_RATE=None  # p. ej. 0.10 solo si capacidad humana <=10% fue aprobada
MACRO_AUPRC_NONINFERIORITY_MARGIN=None  # p. ej. 0.02 solo si fue predeclarado
RUN_COMPARE_AND_FREEZE=False
RUN_TEST_ONCE=False
RUN_PUBLISH=False

def checkpoint_03_07_to_drive(label):
    if COLAB_CONTEXT is None:
        return None
    from moderacion_peru.colab import publish_colab_outputs
    publication=publish_colab_outputs(COLAB_CONTEXT)
    show_result(label,publication,tone='success')
    return publication

if RUN_COMPARE_AND_FREEZE:
    if not CANDIDATE_PREFLIGHT_READY:
        raise RuntimeError(
            'Preflight incompleto: faltan familias requeridas de 03_01–03_06. '
            f'Revise familias_requeridas_ausentes={missing_required_families} y los run_id de Drive.'
        )
    comparison_result=run_with_progress('BA OOF, riesgo-cobertura y bootstrap',compare_and_freeze_validation,DATA,CANDIDATE_ROOTS,COMPARISON,FREEZE,bootstrap_replicates=BOOTSTRAP_REPLICATES,selection_folds=SELECTION_FOLDS,max_review_rate=MAX_REVIEW_RATE,macro_auprc_noninferiority_margin=MACRO_AUPRC_NONINFERIORITY_MARGIN,parallel_workers=PARALLEL_WORKERS,progress_unit='réplica')
    show_result('Comparación y congelación en validation',comparison_result,tone='success')
    checkpoint_03_07_to_drive('Checkpoint verificable de la comparación en Drive')
if RUN_TEST_ONCE:
    if not FREEZE.is_file():
        raise FileNotFoundError('Falta la selección congelada; ejecute y revise primero RUN_COMPARE_AND_FREEZE.')
    test_result=run_with_progress('Inferencia de test',evaluate_frozen_test,FREEZE,TEST_REPORT,confirm_single_test_open=True,progress_unit='lote')
    show_result('Apertura única de test natural + vista 4:1',test_result,tone='warning')
    checkpoint_03_07_to_drive('Checkpoint verificable del test abierto una vez')
if RUN_PUBLISH:
    raise RuntimeError('Publicación productiva bloqueada por diseño: habilítela solo tras aprobación posterior y revisión de FREEZE/TEST_REPORT.')
if not (RUN_COMPARE_AND_FREEZE or RUN_TEST_ONCE or RUN_PUBLISH):
    show_summary('Criterio vigente',{'candidatos_elegibles':candidate_audit['eligible_count'],'03_06b':'incluido' if prompt_sft_active else 'omitido','preflight_listo':CANDIDATE_PREFLIGHT_READY,'ranking':'BA binaria ANY_DAMAGE OOF a cobertura completa','agregación':'lexicográfica; no suma métricas redundantes','salvaguarda':'macro-AUPRC daños + frontera Pareto','desempate':'menor R_0.67; luego macro-AUPRC','NEEDS_REVIEW':'política posterior bajo capacidad humana declarada','bootstrap':f'{BOOTSTRAP_REPLICATES} réplicas pareadas por video en {PARALLEL_WORKERS} hilos','persistencia':'validation se publica como checkpoint verificable del run 03_07 en Drive','test':'bloqueado hasta fijar capacidad y margen antes de comparar'},tone='neutral')"""


MINILM_EXECUTION_GUIDE = """Guía reproducible de la campaña MiniLM

Esta campaña se ejecuta **por etapas y siempre con el mismo `COLAB_RUN_ID`**. El valor vacío reanuda `03_02_working_v2_1`. No borre `runs/`, `trainer_checkpoints` ni las publicaciones de Drive: cada configuración tiene `variant_id` y firma propios; al repetir una ya terminada devuelve `status=noop`, y una interrumpida continúa desde el checkpoint verificable más reciente.

Tampoco limpie las salidas del cuaderno: constituyen el registro de la ejecución. Antes de cada etapa ejecute en orden el bootstrap y la restauración del dataset. Mantenga sin cambios el `COLAB_RUN_ID`, el SHA-256 del dataset, `SAMPLING_SEED` y los hiperparámetros indicados. Conserve el resumen que muestra identificador del bundle, firma del candidato, semillas y métricas de `validation`; eso permite auditar y repetir la comparación en otra sesión o máquina.

La suite principal (`RUN_TRAINING`) y la robustez por canal (`RUN_CHANNEL_ROBUSTNESS`) se ejecutan por separado. La segunda usa otra partición, es diagnóstica y no entra en la validation común de `03_07`. Para la campaña MiniLM descrita abajo, mantenga ambas en `False` y active solo la subetapa correspondiente de `RUN_MINILM_IMPROVEMENTS`.

### Etapa 0 · Baseline, solo si falta el padre

El flujo normal ya dispone del candidato MiniLM de 128 tokens. Si la selección informa que falta, ejecute una sola vez:

```python
RUN_TRAINING=True
RUN_CHANNEL_ROBUSTNESS=False
RUN_MINILM_CONTEXT_SCREEN=False
RUN_MINILM_SEED_CONFIRMATION=False
RUN_MINILM_FOCAL_ABLATION=False
```

Esto vuelve a materializar MiniLM y E5 base. Después restaure `RUN_TRAINING=False`. No active esta etapa si el padre de 128 tokens ya fue verificado.

### Etapa 1 · Pantalla pareada de contexto

```python
RUN_TRAINING=False
RUN_CHANNEL_ROBUSTNESS=False
RUN_MINILM_CONTEXT_SCREEN=True
RUN_MINILM_SEED_CONFIRMATION=False
RUN_MINILM_FOCAL_ABLATION=False
```

Produce 192 y 256 tokens desde **el mismo** padre de 128, con una época, `learning_rate=1e-5`, `sampling_seed=20260805` y `training_seed=20260805`. Cada candidato se publica antes de iniciar el siguiente. Para escoger contexto use exclusivamente `validation`: gana la mayor `average_precision_macro_damage`; ante empate exacto, elija 192 por menor costo. Registre además macro-F1 y falso `SEGURO` como salvaguardas; test continúa sellado.

### Etapa 2 · Confirmación con tres semillas

Asigne a `SELECTED_MINILM_CONTEXT` el ganador de la etapa 1 y ejecute:

```python
RUN_MINILM_CONTEXT_SCREEN=False
RUN_MINILM_SEED_CONFIRMATION=True
RUN_MINILM_FOCAL_ABLATION=False
```

El candidato de la semilla primaria ya existe desde la etapa 1. Esta etapa añade `20260817` y `20260829`; no cambia `sampling_seed`, de modo que train/validation contienen las mismas filas. Informe media, desviación y resultados individuales de las tres semillas.

### Etapa 3 · Ablación focal

Conserve el mismo `SELECTED_MINILM_CONTEXT` y ejecute por separado:

```python
RUN_MINILM_CONTEXT_SCREEN=False
RUN_MINILM_SEED_CONFIRMATION=False
RUN_MINILM_FOCAL_ABLATION=True
```

Compara focal `gamma=2` contra BCE ponderada usando la semilla primaria y las mismas filas. No mezcle esta corrida con la confirmación de semillas.

### Cierre, interrupciones y recuperación

Al terminar deje los tres interruptores en `False`. Si Colab se desconecta, vuelva a ejecutar desde la primera celda con el mismo `COLAB_RUN_ID` y la misma etapa activa; no use `force`, no cambie semillas y no borre artefactos ni salidas anteriores. La publicación es automática por variante; `PUBLISH_TO_DRIVE=True` solo repite manualmente la última publicación. Los candidatos históricos de 128, 192, 256, BCE y focal permanecen coexistiendo para `03_07` y para comparaciones futuras.

El aviso de Transformers 4.57.6 sobre `fix_mistral_regex` al leer el MiniLM local es un falso positivo: MiniLM usa BERT/WordPiece, no Mistral. No establezca ese parámetro en `True`. El cargador vigente lo fija en `False`; una corrida que ya mostró el aviso puede continuar porque Transformers tampoco aplicó el parche."""


MULTI_RUN_GUIDES = {
    "flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb": """Este cuaderno materializa resultados por etapa. En una sesión nueva ejecute siempre desde la primera celda y conserve las semillas, los paneles y las rutas de salida. `FORCE_*=False` reutiliza artefactos cuya firma coincida; no elimine los JSON consolidados entre corridas.

1. **Inspección o reutilización.** Mantenga `APPLY_CHUNK_SELECTION=False`. Los interruptores activos pueden leer resultados existentes sin recalcular; revise primero los paneles que indican si cada etapa está completa.
2. **Diagnósticos clásicos opcionales.** Ejecute el *smoke* o la confirmación corta por separado. Nunca active `RUN_CHUNK_LENGTH_CONFIRMATORY_TEST` y `RUN_CHUNK_LENGTH_ROBUST_TEST` a la vez.
3. **Perfil clásico decisorio.** Para reconstruirlo active solo `RUN_CHUNK_LENGTH_ROBUST_TEST=True`. Debe terminar y escribir `robust_comparison.json` antes del perfil neuronal.
4. **Perfil neuronal pareado.** Active `RUN_NEURAL_ROBUST_TEST=True` con `FORCE_NEURAL_ROBUST_RECOMPUTE=False`. Si se interrumpe, repita la misma corrida: MiniLM y Ollama reutilizan checkpoints con firma compatible.
5. **No inferioridad MiniLM 20 s–30 s.** Active `RUN_MINILM_20_30_NONINFERIORITY_TEST=True` después del perfil clásico. Conserve `FORCE_MINILM_20_30_RECOMPUTE=False` para reanudar por pliegue.
6. **Activación.** Solo después de revisar los informes cambie `APPLY_CHUNK_SELECTION=True`. Esta corrida modifica la configuración activa; manténgala separada de las corridas experimentales.

`test` no participa en ninguna decisión. Registre qué interruptor estuvo activo y el SHA-256 de los artefactos usados en cada corrida.""",
    "flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb": """Use siempre el mismo `COLAB_RUN_ID` para continuar una campaña y ejecute el bootstrap desde la primera celda al cambiar de runtime. Mantenga `RECOVER_HISTORICAL=True` y `AUTO_PUBLISH_CHECKPOINTS=True`; no borre el run, los JSONL ni sus publicaciones de Drive.

1. **Preflight sin etiquetado.** Active `RUN_API_PREFLIGHT=True` y deje las tres fases de etiquetado en `False`. Verifique modelos, modo *non-thinking*, contrato JSON y saldo visible.
2. **Calibración pareada.** Active solo `RUN_CALIBRATION=True`. No inicie la campaña completa hasta revisar acuerdo, enrutamiento y costo del panel Flash–Pro.
3. **Primera pasada Flash.** Active `RUN_PRIMARY=True`, con calibración y revisión en `False`. `PRIMARY_LIMIT=20` sirve para un *smoke* reanudable; `None` procesa toda y solo la cola pendiente.
4. **Revisión dirigida Pro.** Cuando exista la cola enrutada, active solo `RUN_DIRECTED_REVIEW=True`. Use `REVIEW_LIMIT=20` para el *smoke* o `None` para completar pendientes.
5. **Cierre.** Confirme que las colas previstas quedaron en cero y que la publicación verificable está en Drive. Después deje todos los `RUN_*` en `False`.

Una desconexión no obliga a empezar de nuevo: vuelva a ejecutar desde arriba con el mismo run. Los límites nunca se dejan en blanco y no se cambia el prompt, proveedor o semilla dentro de una campaña iniciada.""",
    "flujo/02_etiquetado/02_02_etiquetado_hf_qwen_colab.ipynb": """Este flujo es un respaldo local Hugging Face y se mantiene separado de la campaña DeepSeek. Use el mismo `COLAB_RUN_ID`, el mismo snapshot y los mismos límites al reanudar; ejecute desde la primera celda después de una desconexión.

1. **Primera pasada.** Active solo `RUN_PRIMARY=True`. Empiece con `PRIMARY_LIMIT=20`; si el contrato JSON y el checkpoint son correctos, use `None` para todos los pendientes.
2. **Construcción de la cola.** Ejecute la celda de enrutamiento con los artefactos de la primera pasada ya persistidos. No cambie los umbrales durante la misma campaña.
3. **Revisión dirigida.** Desactive `RUN_PRIMARY`, active `RUN_REVIEW=True` y use primero `REVIEW_LIMIT=20`; luego `None` para completar la cola.
4. **Cierre.** Verifique la publicación del run y deje ambos interruptores en `False`.

No active las dos fases a la vez en una corrida de diagnóstico y no mezcle ni promedie estos resultados con `02_01`: son campañas alternativas con procedencia distinta.""",
    "flujo/03_entrenamiento/03_01_modelos_clasicos.ipynb": """Todas las corridas deben usar el mismo dataset verificado. Ejecute primero la restauración del dataset y active una sola fase por vez; cada fase escribe en un subdirectorio distinto y una firma idéntica produce `status=noop`.

1. **Suite principal.** Active solo `RUN_TRAINING=True`. Esta es la corrida comparable que crea los candidatos base e informados por política.
2. **Reparación SVM.** Ejecútela solo si el candidato SVM informa convergencia no verificada: desactive la suite y active `RUN_SVM_CONVERGENCE_REPAIR=True`. No sustituye silenciosamente los otros estimadores.
3. **Robustez por canal.** Active solo `RUN_CHANNEL_ROBUSTNESS=True`. Es un diagnóstico con otra partición y no entra en la comparación común de `03_07`.
4. **Cierre.** Revise que los candidatos principales tengan `status=complete`, predicciones de validation y test sellado; después deje todos los interruptores en `False`.

No mueva candidatos entre snapshots ni copie métricas del diagnóstico por canal al ranking principal.""",
    "flujo/03_entrenamiento/03_02_transformers_planos.ipynb": MINILM_EXECUTION_GUIDE,
    "flujo/03_entrenamiento/03_05_qwen_lora.ipynb": """Conserve el mismo `COLAB_RUN_ID` (`03_05_working_v2_1` por defecto), dataset, semillas e hiperparámetros. En una sesión nueva ejecute desde el bootstrap: el run se restaura desde Drive y una variante completa devuelve `status=noop`.

1. **Candidato base de 128 tokens.** Active `RUN_TRAINING=True` solo si falta. Espere `status=complete` y la publicación verificable antes de continuar; luego vuelva a `False`.
2. **Continuación a 256 tokens.** Mantenga el entrenamiento base en `False` y active `RUN_CONTINUATION_256=True`. La celda verifica el padre de 128, conserva sus pesos, reinicia optimizador y crea otro candidato sin sobrescribirlo.
3. **Reanudación.** Si Colab se interrumpe, repita desde la primera celda con el mismo run y la misma fase activa. No use `force` ni elimine `trainer_checkpoints`.
4. **Cierre.** Compruebe que ambos candidatos completos coexisten en la publicación y deje los interruptores en `False`.

Las longitudes 128 y 256 son candidatos independientes para validation; nunca se decide entre ellas con test.""",
    "flujo/03_entrenamiento/03_06_qwen_estructurado.ipynb": """Use el mismo `COLAB_RUN_ID`, snapshot y semillas en todas las reanudaciones. El run padre debe permanecer en `QWEN_LORA_PARENT_RUN_ID='03_05_working_v2_1'`; su publicación se restaura y verifica antes de entrenar.

1. **Ruta recomendada.** Deje `RUN_LEGACY_FULL_TRAINING=False` y active solo `RUN_STRUCTURED_LORA_SWEEP=True`. Se crean tres candidatos separados con penalizaciones `0`, `0.02` y `0.05`, cada uno con optimizador nuevo y publicación propia.
2. **Ruta histórica.** `RUN_LEGACY_FULL_TRAINING=True` existe solo para reproducir el full fine-tuning antiguo. Ejecútelo en una corrida separada y nunca junto al barrido LoRA.
3. **Reanudación.** Ante una desconexión, vuelva al bootstrap con el mismo run y repita la fase: variantes y checkpoints completos se reutilizan por firma.
4. **Cierre.** Verifique los `candidate.json`, las predicciones de validation y test sellado; deje ambos interruptores en `False`.

La penalización se selecciona exclusivamente con validation y el candidato histórico se informa por separado.""",
    "flujo/03_entrenamiento/03_06b_qwen_prompt_sft.ipynb": """Conserve el mismo `COLAB_RUN_ID`, snapshot, cápsula de prompt y semillas al reanudar. Active un solo perfil por corrida; piloto, presupuesto corto y entrenamiento completo responden preguntas distintas.

1. **Piloto diagnóstico opcional.** Active solo `RUN_DIAGNOSTIC_PILOT=True`. Valida memoria y contrato JSON, pero usa validation parcial, no se publica como candidato final y nunca entra en `03_07`.
2. **Corrida corta comparable.** Desactive el piloto y active solo `RUN_BUDGETED_COMPARABLE=True`. Entrena con el presupuesto declarado y debe completar la validation común para producir un `candidate.json` elegible con su *disclaimer*.
3. **Entrenamiento completo opcional.** Active solo `RUN_FULL_TRAINING=True` si se aprueba el costo. No lo ejecute junto con el perfil presupuestado ni lo presente como la misma intervención.
4. **Interrupciones.** Si se corta durante entrenamiento, vuelva a ejecutar desde arriba con el mismo run. Si vence el tiempo durante validation y no existe candidato completo, la corrida no es elegible y debe reanudarse o repetirse.
5. **Cierre.** Confirme publicación, validation completa y test sellado; deje los tres interruptores en `False`.

`03_07` incluirá automáticamente solo los candidatos completos y elegibles; la ausencia de `03_06b` no bloquea las demás familias.""",
    "flujo/03_entrenamiento/03_07_comparacion_final.ipynb": """**Entorno recomendado.** Abra una copia nueva de este cuaderno desde GitHub en **Google Colab web**, seleccione un runtime **CPU** y ejecute desde la primera celda. No use para esta corrida el kernel local ni Colab desde VS Code: los candidatos están publicados únicamente en Google Drive y la autorización integrada de `drive.mount()` es más confiable en la interfaz web. Mantenga `COLAB_BUNDLE_SOURCE='github'`; el bundle se descarga y verifica automáticamente, sin selector manual de archivos ni Google Drive para escritorio.

Este cuaderno se ejecuta por compuertas deliberadas. Use el mismo `COLAB_RUN_ID='03_07_working_v2_1'` para que la selección congelada se restaure desde Drive. Nunca active comparación y test en la misma corrida.

1. **Inicio sin carga manual.** Abra <https://colab.research.google.com/github/lkoc/Trabajo_PLN-MIA-Grupo4/blob/main/flujo/03_entrenamiento/03_07_comparacion_final.ipynb>, confirme runtime CPU y conserve `COLAB_BUNDLE_SOURCE='github'`. `local_upload` queda solo como recuperación excepcional; la corrida normal no solicita nueve archivos.
2. **Preflight de candidatos.** Mantenga `RUN_COMPARE_AND_FREEZE=False`, `RUN_TEST_ONCE=False` y `RUN_PUBLISH=False`. La restauración verifica los manifiestos y SHA-256 de `03_01`–`03_06`; `03_06b` solo se agrega si existe y es elegible. No avance hasta ver `listo_para_comparar=True`.
3. **Predeclaración.** Fije `MAX_REVIEW_RATE` y `MACRO_AUPRC_NONINFERIORITY_MARGIN` antes de comparar. Si permanecen en `None`, se genera el informe/Pareto, pero test sigue bloqueado.
4. **Comparación en validation.** Active solo `RUN_COMPARE_AND_FREEZE=True`. Mantenga test y publicación en `False`. El resultado y la selección congelada se guardan como checkpoint verificable del run `03_07` en Drive.
5. **Revisión humana del congelado.** Vuelva a dejar la comparación en `False` y revise candidatos, umbrales, capacidad, margen y advertencias. No cambie estos valores después de mirar test.
6. **Apertura única de test.** Solo con la selección aprobada active `RUN_TEST_ONCE=True`. Si el modelo elegido necesita GPU, cambie el runtime pero conserve el mismo run y ejecute de nuevo desde arriba. Test se infiere una vez y se vuelve a guardar en Drive.
7. **Publicación.** `RUN_PUBLISH` permanece en `False`; la publicación productiva requiere una aprobación posterior separada.

Si una sesión se desconecta antes de terminar el bootstrap o la comparación, reanude desde la primera celda. No borre el run ni copie manualmente candidatos entre snapshots.""",
}


def persistent_colab_training_source(source: str, notebook_id: str | None) -> str:
    """Conecta checkpoints por época y publicación final en todos los 03_x GPU."""

    activity = PERSISTENT_COLAB_TRAINING_ACTIVITY.get(str(notebook_id))
    if activity is None:
        return source
    activity_names = {
        token for token in activity.replace("(", " ").replace(")", " ").split()
        if token.startswith("RUN_")
    }
    if not any(name in source for name in activity_names):
        return source
    if "PERSISTENT_CHECKPOINT_ROOT=" not in source:
        device_line = "DEVICE='cuda' if COLAB_CONTEXT else 'auto'\n"
        if device_line not in source:
            raise ValueError(f"{notebook_id} no declara DEVICE de Colab")
        source = source.replace(
            device_line,
            device_line
            + "PERSISTENT_CHECKPOINT_ROOT=COLAB_CONTEXT.drive_run_dir/'trainer_checkpoints' if COLAB_CONTEXT else None\n",
            1,
        )
        source = source.replace(
            ",progress_unit=",
            ",persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit=",
        )
    if notebook_id == "03_02" and "completion_callback=publish_completed_flat_model" not in source:
        source = source.replace(
            "RUN_CHANNEL_ROBUSTNESS=False\n",
            "RUN_CHANNEL_ROBUSTNESS=False\n"
            "def publish_completed_flat_model(event):\n"
            "    if COLAB_CONTEXT is None: return\n"
            "    from moderacion_peru.colab import publish_colab_outputs\n"
            "    publication=publish_colab_outputs(COLAB_CONTEXT)\n"
            "    show_result(f\"Checkpoint final {event['experiment']} ({event['index']}/{event['total']})\",publication,tone='success')\n",
            1,
        )
        source = source.replace(
            ",persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='modelo'",
            ",persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,completion_callback=publish_completed_flat_model,progress_unit='modelo'",
            1,
        )
    if notebook_id == "03_06b" and "Publicación final budgeted" not in source:
        source = source.replace(
            "    show_result('Candidato SFT budgeted elegible para 03_07',budgeted_result,tone='warning')\n",
            "    show_result('Candidato SFT budgeted elegible para 03_07',budgeted_result,tone='warning')\n"
            "    if COLAB_CONTEXT is not None:\n"
            "        from moderacion_peru.colab import publish_colab_outputs\n"
            "        show_result('Publicación final budgeted',publish_colab_outputs(COLAB_CONTEXT),tone='success')\n",
            1,
        )
        source = source.replace(
            "    show_result('SFT generativo completo condicionado por prompt v3.2',full_result,tone='success')\n",
            "    show_result('SFT generativo completo condicionado por prompt v3.2',full_result,tone='success')\n"
            "    if COLAB_CONTEXT is not None:\n"
            "        from moderacion_peru.colab import publish_colab_outputs\n"
            "        show_result('Publicación final del SFT completo',publish_colab_outputs(COLAB_CONTEXT),tone='success')\n",
            1,
        )
    if notebook_id != "03_06b" and "Publicación final automática" not in source:
        final_activity = "RUN_CHANNEL_ROBUSTNESS" if notebook_id == "03_02" else activity
        source += (
            "\nif COLAB_CONTEXT is not None and ("
            + final_activity
            + "):\n"
            "    from moderacion_peru.colab import publish_colab_outputs\n"
            "    final_publication=publish_colab_outputs(COLAB_CONTEXT)\n"
            "    show_result('Publicación final automática',final_publication,tone='success')"
        )
    return source


def create(
    path: str,
    title: str,
    purpose: str,
    academic_context: str,
    code_cells: list[tuple[str, str]],
    *,
    colab_notebook_id: str | None = None,
    colab_publisher: bool = False,
    colab_requires_gpu: bool = True,
) -> None:
    if ONLY_NOTEBOOKS is not None and path not in ONLY_NOTEBOOKS:
        return
    if colab_notebook_id is not None and colab_publisher:
        raise ValueError(
            "Un cuaderno no puede ser consumidor y publicador Colab a la vez"
        )
    colab_eligible = colab_notebook_id is not None or colab_publisher
    bundle_identity = colab_bundle_identity() if colab_eligible else None
    metadata_notebook_id = colab_notebook_id or ("02_00" if colab_publisher else None)
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "authors": [{"name": name} for name in PROJECT_AUTHORS],
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "moderacion_peru": {
            "project_title": PROJECT_TITLE,
            "academic_work": "Trabajo final de Procesamiento de Lenguaje Natural (PLN)",
            "institution": "Maestría en Inteligencia Artificial, Universidad Nacional de Ingeniería (UNI)",
            "academic_term": "2026-1",
            "group": "Grupo 4",
            "contract": "moderacion_peru_5_salidas_v2",
            "taxonomy_version": "2.1.0",
            "orchestrator": True,
            "colab": {
                "eligible": colab_eligible,
                "notebook_id": metadata_notebook_id,
                "transport": (
                    "github_or_browser_upload_to_google_drive"
                    if colab_publisher
                    else (
                        "github_or_browser_upload_with_drive_versioned_releases"
                        if colab_notebook_id
                        else None
                    )
                ),
                "expected_gpu": colab_expected_gpu(
                    colab_notebook_id, colab_requires_gpu
                ),
                "build_bundle_id": (
                    bundle_identity["bundle_id"] if bundle_identity else None
                ),
                "bundle_resolution": (
                    "publishes_drive_latest_pointer"
                    if colab_publisher
                    else "auto_publish_missing_drive_release"
                    if bundle_identity
                    else None
                ),
                "expected_core_sha256": (
                    bundle_identity["core_sha256"] if bundle_identity else None
                ),
            },
        },
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"{PROJECT_COVER}\n\n## {title}\n\n{purpose}\n\n{academic_context}\n\n"
            f"{CONTRACT_SUMMARY}"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibilidad\n\n"
            + (
                "El cuaderno obtiene un bundle ya construido desde GitHub o desde el selector de archivos "
                "del navegador, verifica su identidad y todos sus SHA-256, y solo entonces publica una "
                "versión inmutable en Drive. No requiere Google Cloud Console ni Drive Desktop."
                if colab_publisher
                else "El cuaderno solo orquesta funciones versionadas de `src/moderacion_peru`. En local no "
                "instala paquetes. En Colab, únicamente la celda de bootstrap instala versiones fijadas "
                "desde el bundle SHA-256 de Drive. No usa rutas personales. Revise el README de esta etapa."
            )
        ),
    ]
    if colab_publisher:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Backend Google Colab\n\n"
                "Abra este archivo en **Google Colab** y ejecute sus celdas en orden. El modo `github` "
                "descarga el bundle sincronizado de la revisión configurada; `local_upload` permite escoger "
                "todos los archivos del bundle local más reciente. La autorización integrada de "
                "`drive.mount()` publica en `Mi unidad/ModeracionPeru_Colab` [@googlecolab2026faq]."
            )
        )
        notebook.cells.append(nbf.v4.new_code_cell(colab_publisher_setup()))
    elif colab_notebook_id:
        if colab_notebook_id == "03_07":
            notebook.cells.append(
                nbf.v4.new_markdown_cell(
                    "## Entorno recomendado: Google Colab web\n\n"
                    "Abra una copia nueva desde [GitHub en Google Colab web]"
                    "(https://colab.research.google.com/github/lkoc/Trabajo_PLN-MIA-Grupo4/"
                    "blob/main/flujo/03_entrenamiento/03_07_comparacion_final.ipynb), "
                    "seleccione un runtime **CPU** y ejecute desde la primera celda; esta comparación "
                    "funciona con runtime CPU. Para esta "
                    "comparación no use el kernel local ni Colab desde VS Code: los candidatos "
                    "están publicados únicamente en Google Drive y la autorización integrada de "
                    "`drive.mount()` es más confiable en la interfaz web.\n\n"
                    "Conserve `COLAB_BUNDLE_SOURCE='github'`. El bootstrap descarga el bundle fijado, "
                    "verifica todos sus SHA-256, monta Drive y restaura las publicaciones; no solicita "
                    "seleccionar nueve archivos y no requiere Google Drive para escritorio. "
                    "`local_upload` queda únicamente como recuperación excepcional si GitHub no "
                    "contuviera el bundle esperado."
                )
            )
        else:
            notebook.cells.append(
                nbf.v4.new_markdown_cell(
                    "## Backend opcional Google Colab desde VS Code\n\n"
                    "Instale la extensión oficial **Google Colab** (`google.colab`), seleccione "
                    + (
                        "`Select Kernel > Colab` y asigne una **NVIDIA A100 de 40 GB** "
                        "para Qwen. "
                        if colab_notebook_id in QWEN_A100_NOTEBOOKS
                        else "`Select Kernel > Colab` y asigne una **NVIDIA L4**. "
                        if colab_requires_gpu
                        else "`Select Kernel > Colab`; esta campaña API funciona con runtime CPU. "
                    )
                    + "El notebook permanece local; "
                    "Drive transporta solo versiones inmutables del bundle. La celda detecta si falta el "
                    "release exacto: en ese único caso lo obtiene desde GitHub —o mediante `local_upload`—, "
                    "verifica todos sus SHA-256 y lo publica de forma atómica. Después promueve la copia "
                    "activa cuando sea necesario; ya no requiere ejecutar `02_00` previamente. Edite "
                    "`COLAB_RUN_ID` para separar "
                    "experimentos. La compatibilidad de `drive.mount()` desde VS Code requiere la extensión "
                    "v0.2.1 o posterior [@googlecolab2026vscode]. La integridad del bundle se comprueba con "
                    "SHA-256 [@nist2015sha]. No sincronice cachés de modelos ni entrene directamente "
                    "sobre Drive; los cuadernos de entrenamiento Qwen copian cada checkpoint terminado "
                    "mediante un TAR atómico y reanudable."
                )
            )
        notebook.cells.append(nbf.v4.new_code_cell(colab_setup(colab_notebook_id)))
    else:
        notebook.cells.append(nbf.v4.new_code_cell(SETUP))
    multi_run_guide = MULTI_RUN_GUIDES.get(path)
    if multi_run_guide:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Procedimiento reproducible por corridas\n\n" + multi_run_guide
            )
        )
    for heading, source in code_cells:
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        notebook.cells.append(
            nbf.v4.new_code_cell(
                persistent_colab_training_source(source, colab_notebook_id)
            )
        )
    if colab_notebook_id and colab_notebook_id != "03_07":
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Publicación o checkpoint en Drive\n\n"
                "Cada época completa se guarda y verifica por separado en `trainer_checkpoints`. "
                + (
                    "El piloto diagnóstico no se publica automáticamente porque no es elegible para `03_07`. "
                    if colab_notebook_id == "03_06b"
                    else ""
                )
                + "Al terminar una corrida 03_x se publica automáticamente el candidato final en una "
                "de dos ranuras redundantes. Esta celda permite repetir manualmente esa publicación; "
                "no vuelve a incluir los directorios transitorios de `Trainer`."
            )
        )
        notebook.cells.append(
            nbf.v4.new_code_cell(
                "PUBLISH_TO_DRIVE = False\n"
                "if COLAB_CONTEXT is not None and PUBLISH_TO_DRIVE:\n"
                "    from moderacion_peru.colab import publish_colab_outputs\n"
                "    show_result('Publicación en Drive', publish_colab_outputs(COLAB_CONTEXT), tone='success')\n"
                "elif COLAB_CONTEXT is not None and globals().get('AUTO_PUBLISH_CHECKPOINTS'):\n"
                "    show_callout('Checkpoint automático activo', 'La recuperación, los checkpoints periódicos, Ctrl+C y cada cierre de campaña ya publican una copia verificable en Drive.', tone='success')\n"
                "elif COLAB_CONTEXT is not None:\n"
                "    show_callout('Publicación manual desactivada', 'Los entrenamientos 03_x ya publican automáticamente al completar; active esta celda solo para repetir la publicación final.', tone='neutral')\n"
                "else:\n"
                "    show_callout('Backend local', 'Los artefactos ya permanecen en el workspace.', tone='success')"
            )
        )
    references = apply_citations(notebook.cells, notebook.metadata["moderacion_peru"])
    notebook.cells.append(
        nbf.v4.new_markdown_cell(references, metadata={"tags": ["references"]})
    )
    cell_prefix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{cell_prefix}-{index:02d}"
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if path != "flujo/03_entrenamiento/03_07_comparacion_final.ipynb":
        preserve_matching_execution_outputs(notebook, target)
    nbf.write(notebook, target)


def main(*, only_notebooks: set[str] | None = None) -> None:
    global ONLY_NOTEBOOKS
    ONLY_NOTEBOOKS = only_notebooks
    create(
        "flujo/01_datos/01_01_scraping_incremental.ipynb",
        "01.01 · Adquisición incremental de subtítulos",
        "Reutiliza transcripciones canónicas y cachés por `video_id`; solo consulta YouTube para candidatos nuevos y nunca descarga audio o video.",
        "La adquisición nueva usa `yt-dlp` para escribir pistas VTT sin descargar audio ni video "
        "[@ytdlp2026], y conserva `youtube-transcript-api` únicamente como respaldo cuando la vía "
        "principal no produce una transcripción íntegra [@depoix2026transcript]. "
        "Las transcripciones automáticas se conservan como insumo imperfecto, no como verdad textual, "
        "porque se han documentado sesgos de dialecto y género en el subtitulado automático de YouTube "
        "[@tatman2017captions]. Toda ampliación debe respetar los términos de la plataforma "
        "[@youtube2023terms] y la evaluación ética contextual recomendada para investigación en "
        "Internet [@aoir2020ethics]. La reutilización de cachés y la selección de candidatos son "
        "decisiones locales registradas en manifiestos. El modo dirigido adapta principios de balance "
        "para aprendizaje activo y clasificación multietiqueta de cola larga "
        "[@fairstein2024balancing] [@huang2021balancing]: calcula soporte por videos únicos solo en "
        "`train+validation`, excluye `test`, pondera los déficits, estima rendimiento histórico por "
        "canal, expande canales desde consultas temáticas y materializa una cohorte aislada mediante "
        "*round-robin*. Si no hay etiquetas previas utilizables, asigna 25% a cada daño. La fórmula, "
        "los umbrales, las cuotas, el fallback y el aislamiento por canal ante HTTP 429 son decisiones operativas "
        "propias; `target_category` registra el motivo de selección y nunca constituye una etiqueta. "
        "Este muestreo enriquecido no permite estimar prevalencias en YouTube ni en el Perú.",
        [
            (
                "Preflight",
                "from moderacion_peru.artifacts import artifact_status\nshow_result('Disponibilidad de artefactos', artifact_status(ROOT), tone='neutral')",
            ),
            ("Parámetros editables", SCRAPING_PARAMETERS),
            ("Canales y consultas", SCRAPING_SOURCES),
            (
                "Reutilización y plan dirigido\n\n"
                "El soporte se mide por `video_id` único para no favorecer videos largos con más chunks. "
                "La ponderación por déficit y el reparto multietiqueta se inspiran en problemas de "
                "aprendizaje activo desbalanceado y distribuciones de cola larga "
                "[@fairstein2024balancing] [@huang2021balancing]. Su implementación exacta es local: "
                "`test` permanece congelado, el fallback sin historia reparte las cuatro categorías por "
                "igual y las categorías objetivo solo orientan la adquisición.",
                SCRAPING_REUSE_AND_PLAN,
            ),
            ("Descubrimiento general o ampliación dirigida", SCRAPING_DISCOVERY),
            ("Cohorte activa y caché", SCRAPING_CANDIDATES),
            ("Ejecución controlada y tolerante a fallos", SCRAPING_EXECUTION),
        ],
    )
    create(
        "flujo/01_datos/01_015_ampliacion_dirigida_minorias.ipynb",
        "01.015 · Ampliación dirigida de categorías minoritarias",
        "Descubre y adquiere videos peruanos nuevos para llevar cada categoría de daño a por lo menos 2.000 chunks en el total train + validation + test, sin modificar el cuaderno 01_01 de scraping inicial. Una compuerta estricta PE excluye antes de la descarga todo canal extranjero o cuyo origen peruano no pueda verificarse. La ejecución es reanudable: conserva fuentes terminadas, caché por video y checkpoints canónicos, y en cada nueva ronda recalcula únicamente el déficit pendiente.",
        "La selección usa el snapshot efectivo después de las decisiones CODEX–Sol-EH y mide el "
        "déficit en el total de `train`, `validation` y `test`. El muestreo dirigido adapta principios de aprendizaje "
        "activo ante desbalance y clasificación multietiqueta de cola larga "
        "[@fairstein2024balancing] [@huang2021balancing]. Los candidatos reciben un split estable por "
        "`video_id` antes de descargar y etiquetar. Los tres splits cuentan para la condición de parada "
        "de 2.000; el split permanece agrupado por video y se informa por separado para detectar desbalance. "
        "La adquisición escribe VTT mediante `yt-dlp` sin descargar audio o video [@ytdlp2026] y "
        "mantiene `youtube-transcript-api` como respaldo [@depoix2026transcript]. Las consultas y "
        "`target_category` son mecanismos de recuperación, nunca etiquetas: todo chunk nuevo debe pasar "
        "por el prompt operativo y la jerarquía de revisión. Las transcripciones automáticas pueden "
        "contener sesgos dialectales [@tatman2017captions], y el uso de la plataforma debe respetar sus "
        "términos y el análisis ético contextual [@youtube2023terms] [@aoir2020ethics]. La expresión "
        "Perú en una consulta no se considera evidencia de origen: solo se admiten canales del catálogo "
        "PE curado, canales ya validados en el dataset efectivo o metadatos explícitos de país PE; los "
        "demás quedan en un manifiesto de exclusiones auditable.",
        [
            (
                "Preflight",
                "from moderacion_peru.artifacts import artifact_status\nshow_result('Disponibilidad de artefactos', artifact_status(ROOT), tone='neutral')",
            ),
            ("Parámetros de la meta total ≥ 2.000", SCRAPING_MINORITY_PARAMETERS),
            ("Canales y consultas para daños minoritarios", SCRAPING_MINORITY_SOURCES),
            (
                "Snapshot efectivo y déficit por chunks totales\n\n"
                "`needs_review` de Pro es un estado intermedio. Una decisión posterior CODEX–Sol-EH "
                "o humana lo sustituye; si CODEX no cambió una propuesta Pro no vacía, prevalece Pro. "
                "El plan usa la última decisión efectiva, conserva el split histórico por video y "
                "calcula cuánto falta para 2.000 asignaciones en cada daño sumando train, validation y test. Los canales se "
                "ordenan por rendimiento histórico, pero se combinan varias fuentes para evitar que "
                "una clase aprenda únicamente el estilo de un canal.",
                SCRAPING_MINORITY_REUSE_AND_PLAN,
            ),
            (
                "Descubrimiento y cohorte separada por split",
                SCRAPING_MINORITY_DISCOVERY,
            ),
            ("Cohorte activa, arrastre y caché", SCRAPING_MINORITY_CANDIDATES),
            ("Ejecución controlada y tolerante a fallos", SCRAPING_EXECUTION),
        ],
    )
    create(
        "flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb",
        "01.02 · Selección robusta y triangulación neuronal de longitud",
        "Compara 15, 20, 25, 30 y 35 segundos mediante un perfil clásico decisorio y dos análisis neuronales confirmatorios.",
        "La selección usa exclusivamente `validation`; consultar `test` para elegir longitud produciría "
        "sesgo de selección [@cawley2010selection]. La métrica clásica y la de MiniLM es *average "
        "precision* macro de los cuatro daños, apropiada para clases desbalanceadas [@saito2015pr]. "
        "Los baselines clásicos reutilizan TF-IDF y estimadores de scikit-learn [@salton1988tfidf] "
        "[@pedregosa2011sklearn]. MiniLM funciona como encoder congelado con una cabeza logística; "
        "su familia se fundamenta en destilación y representaciones multilingües [@wang2020minilm] "
        "[@reimers2020multilingual], y el checkpoint queda fijado por su tarjeta [@hf2026minilmcard]. "
        "Ollama usa `gemma3:4b`, salida estructurada y el prompt operativo vigente en "
        "`config/prompt_operacional_ollama_v3_2.md` "
        "[@ollama2026gemma34b] [@ollama2026structured]. El bootstrap remuestrea videos completos para "
        "preservar la dependencia entre ventanas pareadas [@efron1979bootstrap] "
        "[@field2007clusterbootstrap]; la comparación pareada y las hipótesis predeclaradas siguen "
        "recomendaciones de evaluación estadística en PLN [@dror2018significance]. El contraste "
        "complementario formula explícitamente la hipótesis direccional de no inferioridad "
        "[@blackwelder1982null]. La composición del "
        "panel enriquecido, las cuotas por daño, los márgenes de no inferioridad, la penalización de "
        "salidas inválidas y la jerarquía de decisión son elecciones locales. Las métricas heterogéneas "
        "nunca se promedian: el perfil clásico selecciona o conserva la longitud; MiniLM examina "
        "sensibilidad a representaciones neuronales y Ollama examina sensibilidad semántica y "
        "viabilidad operativa. Si una familia diverge, se reporta el conflicto y se mantiene la "
        "decisión clásica hasta una validación humana independiente. `test` permanece cerrado.",
        [
            (
                "Controles y protocolo predeclarado",
                "RUN_CHUNK_LENGTH_SMOKE_TEST=False\n"
                "RUN_CHUNK_LENGTH_CONFIRMATORY_TEST=False\n"
                "RUN_CHUNK_LENGTH_ROBUST_TEST=False  # Active solo para reconstruir la etapa clásica\n"
                "RUN_NEURAL_ROBUST_TEST=True\n"
                "RUN_MINILM_20_30_NONINFERIORITY_TEST=True\n"
                "FORCE_NEURAL_ROBUST_RECOMPUTE=False\n"
                "FORCE_MINILM_20_30_RECOMPUTE=False\n"
                "CANDIDATE_SECONDS=(15,20,25,30,35)\n"
                "TOY_MODELS=('complement_nb','sgd_incremental')\n"
                "TOY_VIDEO_LIMITS={'train':40,'validation':16,'test':16}\n"
                "TOY_MAX_FEATURES=12000\n"
                "CONFIRMATORY_MODELS=('complement_nb','logistic_regression','sgd_incremental')\n"
                "CONFIRMATORY_VIDEO_LIMITS={'train':200,'validation':80,'test':80}\n"
                "CONFIRMATORY_SEEDS=(20260805,20260817,20260829)\n"
                "CONFIRMATORY_MAX_FEATURES=20000\n"
                "ROBUST_VIDEO_LIMITS={'train':300,'validation':100,'test':100}\n"
                "ROBUST_SEEDS=(20260805,20260817,20260829,20260841,20260853)\n"
                "ROBUST_MAX_FEATURES=25000\n"
                "ROBUST_REFERENCE_SECONDS=30.0\n"
                "ROBUST_NONINFERIORITY_MARGIN=0.01\n"
                "ROBUST_BOOTSTRAP_REPLICATES=1000\n"
                "ROBUST_CONFIDENCE_LEVEL=0.95\n"
                "ROBUST_BOOTSTRAP_SEED=20260807\n"
                "ROBUST_RUNTIME_BUDGET_SECONDS=1800.0\n"
                "MAX_VALIDATION_AP_DROP=0.02\n"
                "NEURAL_PANEL_SIZE=100\n"
                "NEURAL_MIN_DAMAGE_PER_LABEL=20\n"
                "NEURAL_MAX_ANCHORS_PER_VIDEO=2\n"
                "NEURAL_REPORTING_COHORTS=5\n"
                "NEURAL_PANEL_SELECTION_SEED=20260807\n"
                "NEURAL_MINILM_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'\n"
                "NEURAL_MINILM_REVISION='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'\n"
                "NEURAL_MINILM_TRAIN_LIMIT=1000\n"
                "NEURAL_MINILM_BATCH_SIZE=16\n"
                "NEURAL_MINILM_MAX_LENGTH=128\n"
                "NEURAL_MINILM_BOOTSTRAP_REPLICATES=2000\n"
                "NEURAL_MINILM_NONINFERIORITY_MARGIN=0.01\n"
                "NEURAL_OLLAMA_MODEL='gemma3:4b'\n"
                "NEURAL_OLLAMA_TIMEOUT_SECONDS=90.0\n"
                "NEURAL_OLLAMA_MAX_WALL_SECONDS=5400.0\n"
                "NEURAL_OLLAMA_RETRIES=1\n"
                "NEURAL_OLLAMA_BOOTSTRAP_REPLICATES=2000\n"
                "NEURAL_OLLAMA_NONINFERIORITY_MARGIN=0.02\n"
                "NEURAL_OLLAMA_MINIMUM_SCHEMA_RATE=0.95\n"
                "MINILM_NI_PANEL_SIZE=750\n"
                "MINILM_NI_MIN_DAMAGE_VIDEOS=80\n"
                "MINILM_NI_FOLDS=5\n"
                "MINILM_NI_PANEL_SELECTION_SEED=20260831\n"
                "MINILM_NI_REPEAT_SEEDS=(20260901,20260913,20260925)\n"
                "MINILM_NI_TRAIN_LIMIT=4000\n"
                "MINILM_NI_BOOTSTRAP_REPLICATES=5000\n"
                "MINILM_NI_BOOTSTRAP_SEED=20260907\n"
                "MANUAL_CHUNK_SECONDS=30.0\n"
                "USE_ROBUST_RECOMMENDATION=True\n"
                "APPLY_CHUNK_SELECTION=False\n"
                "from moderacion_peru.colab import prepare_local_bundle_input\n"
                "from moderacion_peru.chunk_optimization import activate_chunking_configuration, run_chunk_length_confirmatory_test, run_chunk_length_robust_test, run_chunk_length_smoke_test\n"
                "from moderacion_peru.neural_chunk_robust import run_minilm_20_30_noninferiority_test, run_neural_chunk_robust_test\n"
                "from moderacion_peru.incremental import DEFAULT_CHUNKING_CONFIGURATION\n"
                "import json\n"
                "def sig2(value):\n"
                "    return None if value is None else float(f'{float(value):.2g}')\n"
                "TRANSCRIPTS=ROOT/'datos/raw/transcripts_raw.jsonl'\n"
                "CHUNKS_CHECKPOINT=prepare_local_bundle_input('chunks_v2',project_root=ROOT)\n"
                "CHUNKS=Path(CHUNKS_CHECKPOINT['path'])\n"
                "DATASET_CHECKPOINT=prepare_local_bundle_input('dataset_5_salidas',project_root=ROOT)\n"
                "DATASET=Path(DATASET_CHECKPOINT['path'])\n"
                "PILOT_ROOT=ROOT/'resultados/pilotos/chunk_length'\n"
                "ROBUST_ROOT=PILOT_ROOT/'robust_30min'\n"
                "NEURAL_ROOT=PILOT_ROOT/'neural_robust'\n"
                "MINILM_NI_ROOT=NEURAL_ROOT/'minilm_20_30_noninferiority'\n"
                "ROBUST_RESULT_PATH=ROBUST_ROOT/'robust_comparison.json'\n"
                "NEURAL_RESULT_PATH=NEURAL_ROOT/'neural_robust_comparison.json'\n"
                "MINILM_NI_RESULT_PATH=MINILM_NI_ROOT/'minilm_20_30_noninferiority.json'\n"
                "ROBUST_RECOMMENDATION=ROBUST_ROOT/'robust_recommendation.json'\n"
                "if tuple(CANDIDATE_SECONDS)!=(15,20,25,30,35):\n"
                "    raise ValueError('Este protocolo exige CANDIDATE_SECONDS=(15,20,25,30,35)')\n"
                "if RUN_CHUNK_LENGTH_CONFIRMATORY_TEST and RUN_CHUNK_LENGTH_ROBUST_TEST:\n"
                "    raise ValueError('El perfil robusto ya incluye la confirmación; active solo uno')\n"
                "show_summary('Secuencia configurada',{'1_clasico_decisorio':RUN_CHUNK_LENGTH_ROBUST_TEST or ROBUST_RESULT_PATH.is_file(),'2_minilm_confirmatorio':RUN_NEURAL_ROBUST_TEST,'3_ollama_confirmatorio':RUN_NEURAL_ROBUST_TEST,'4_minilm_no_inferioridad_20_30':RUN_MINILM_20_30_NONINFERIORITY_TEST,'longitudes_perfil':CANDIDATE_SECONDS,'contraste_complementario':(20,30),'panel_validation':NEURAL_PANEL_SIZE,'panel_crossfit_train':MINILM_NI_PANEL_SIZE,'cohortes_reporte':NEURAL_REPORTING_COHORTS,'respuestas_ollama_previstas':NEURAL_PANEL_SIZE*len(CANDIDATE_SECONDS),'test_usado_para_seleccion':False},tone='neutral')",
            ),
            (
                "Diagnósticos clásicos opcionales",
                "if RUN_CHUNK_LENGTH_SMOKE_TEST:\n"
                "    smoke_result=run_with_progress('Smoke de longitudes',run_chunk_length_smoke_test,TRANSCRIPTS,CHUNKS,DATASET,PILOT_ROOT,candidate_seconds=CANDIDATE_SECONDS,model_names=TOY_MODELS,video_limits=TOY_VIDEO_LIMITS,max_features=TOY_MAX_FEATURES,max_validation_ap_drop=MAX_VALIDATION_AP_DROP,progress_unit='etapa')\n"
                "    show_result('Recomendación exploratoria',smoke_result['recommendation'],tone='success')\n"
                "    show_table('Smoke clásico',smoke_result['comparisons'],max_rows=len(CANDIDATE_SECONDS))\n"
                "else:\n"
                "    show_callout('Smoke clásico omitido','Es opcional y no sustituye el perfil robusto.',tone='neutral')\n"
                "if RUN_CHUNK_LENGTH_CONFIRMATORY_TEST:\n"
                "    confirmatory_result=run_with_progress('Confirmación de longitudes',run_chunk_length_confirmatory_test,TRANSCRIPTS,CHUNKS,DATASET,PILOT_ROOT,candidate_seconds=CANDIDATE_SECONDS,model_names=CONFIRMATORY_MODELS,video_limits=CONFIRMATORY_VIDEO_LIMITS,seeds=CONFIRMATORY_SEEDS,max_features=CONFIRMATORY_MAX_FEATURES,progress_unit='etapa')\n"
                "    show_result('Recomendación confirmatoria corta',confirmatory_result['recommendation'],tone='success')\n"
                "    show_table('Confirmación clásica corta',confirmatory_result['aggregated_comparisons'],max_rows=len(CANDIDATE_SECONDS))\n"
                "else:\n"
                "    show_callout('Confirmación corta omitida','Es opcional porque el perfil robusto ya incorpora cinco cohortes.',tone='neutral')",
            ),
            (
                "Etapa 1 — perfil robusto clásico decisorio",
                "if RUN_CHUNK_LENGTH_ROBUST_TEST:\n"
                "    robust_result=run_with_progress('Perfil clásico robusto',run_chunk_length_robust_test,TRANSCRIPTS,CHUNKS,DATASET,ROBUST_ROOT,candidate_seconds=CANDIDATE_SECONDS,reference_seconds=ROBUST_REFERENCE_SECONDS,model_names=CONFIRMATORY_MODELS,video_limits=ROBUST_VIDEO_LIMITS,seeds=ROBUST_SEEDS,max_features=ROBUST_MAX_FEATURES,bootstrap_replicates=ROBUST_BOOTSTRAP_REPLICATES,confidence_level=ROBUST_CONFIDENCE_LEVEL,noninferiority_margin=ROBUST_NONINFERIORITY_MARGIN,bootstrap_seed=ROBUST_BOOTSTRAP_SEED,runtime_budget_seconds=ROBUST_RUNTIME_BUDGET_SECONDS,progress_unit='etapa')\n"
                "elif ROBUST_RESULT_PATH.is_file():\n"
                "    robust_result=json.loads(ROBUST_RESULT_PATH.read_text(encoding='utf-8-sig'))\n"
                "else:\n"
                "    robust_result=None\n"
                "if robust_result:\n"
                "    classical_rows=[{'longitud_s':row['chunk_seconds'],'AP_validation':sig2(row['paired_validation_ap_macro_damage']),'IC95_AP':[sig2(row['bootstrap_ap_ci_low']),sig2(row['bootstrap_ap_ci_high'])],'delta_vs_30s':sig2(row['delta_vs_reference']),'IC95_delta':[sig2(row['delta_vs_reference_ci_low']),sig2(row['delta_vs_reference_ci_high'])],'no_inferior':'Sí' if row['noninferior'] else 'No','proxy_costo':row['compute_proxy']} for row in robust_result['bootstrap']['comparisons']]\n"
                "    show_table('Perfil clásico: resultados reportables',classical_rows,max_rows=len(CANDIDATE_SECONDS))\n"
                "    show_summary('Decisión primaria',{'longitud_s':robust_result['recommendation']['recommended_seconds'],'partición':'validation','métrica':'AP macro de cuatro daños','cohortes':robust_result['design']['paired_cohorts'],'ajustes':robust_result['design']['fits'],'réplicas_bootstrap':robust_result['bootstrap']['replicates'],'test_usado_para_seleccion':False,'artefacto':ROBUST_RESULT_PATH},tone='success')\n"
                "else:\n"
                "    show_callout('Falta el perfil clásico','Active RUN_CHUNK_LENGTH_ROBUST_TEST=True antes del perfil neuronal.',tone='danger')",
            ),
            (
                "Etapas 2 y 3 — perfil neuronal robusto pareado",
                "if NEURAL_RESULT_PATH.is_file() and not FORCE_NEURAL_ROBUST_RECOMPUTE:\n"
                "    neural_result=json.loads(NEURAL_RESULT_PATH.read_text(encoding='utf-8-sig'))\n"
                "    show_callout('Perfil neuronal cargado','Se leyó el JSON consolidado; no se llamó MiniLM ni Ollama.',tone='success')\n"
                "elif RUN_NEURAL_ROBUST_TEST:\n"
                "    if not ROBUST_RESULT_PATH.is_file():\n"
                "        raise FileNotFoundError('Ejecute primero la etapa clásica robusta')\n"
                "    show_callout('Perfil neuronal en ejecución','La celda escribe checkpoints durante MiniLM y Ollama. Las tablas aparecen al final; una nueva ejecución reutiliza todo resultado con firma compatible.',tone='neutral')\n"
                "    neural_result=run_with_progress('Perfil neuronal robusto',run_neural_chunk_robust_test,TRANSCRIPTS,CHUNKS,DATASET,ROBUST_ROOT,NEURAL_ROOT,candidate_seconds=CANDIDATE_SECONDS,reference_seconds=ROBUST_REFERENCE_SECONDS,seeds=ROBUST_SEEDS,panel_size=NEURAL_PANEL_SIZE,minimum_damage_anchors_per_label=NEURAL_MIN_DAMAGE_PER_LABEL,max_anchors_per_video=NEURAL_MAX_ANCHORS_PER_VIDEO,reporting_cohorts=NEURAL_REPORTING_COHORTS,panel_selection_seed=NEURAL_PANEL_SELECTION_SEED,minilm_model_id=NEURAL_MINILM_MODEL,minilm_revision=NEURAL_MINILM_REVISION,minilm_train_limit_per_cohort=NEURAL_MINILM_TRAIN_LIMIT,minilm_batch_size=NEURAL_MINILM_BATCH_SIZE,minilm_max_length=NEURAL_MINILM_MAX_LENGTH,minilm_bootstrap_replicates=NEURAL_MINILM_BOOTSTRAP_REPLICATES,minilm_noninferiority_margin=NEURAL_MINILM_NONINFERIORITY_MARGIN,ollama_model=NEURAL_OLLAMA_MODEL,ollama_timeout_seconds=NEURAL_OLLAMA_TIMEOUT_SECONDS,ollama_max_wall_seconds=NEURAL_OLLAMA_MAX_WALL_SECONDS,ollama_retries=NEURAL_OLLAMA_RETRIES,ollama_bootstrap_replicates=NEURAL_OLLAMA_BOOTSTRAP_REPLICATES,ollama_noninferiority_margin=NEURAL_OLLAMA_NONINFERIORITY_MARGIN,ollama_minimum_schema_rate=NEURAL_OLLAMA_MINIMUM_SCHEMA_RATE,confidence_level=ROBUST_CONFIDENCE_LEVEL,progress_unit='etapa')\n"
                "else:\n"
                "    neural_result=None\n"
                "if neural_result:\n"
                "    panel=neural_result['panel']\n"
                "    show_summary('Panel pareado de validation',{'anclas':panel['anchors'],'videos':panel['distinct_videos'],'conteos_etiqueta':panel['label_counts'],'cohortes_disjuntas':panel['anchors_per_reporting_cohort'],'muestra':'enriquecida; no estima prevalencia','test_usado':False},tone='neutral')\n"
                "    minilm_rows=[{'longitud_s':row['chunk_seconds'],'AP_ensemble':sig2(row['ensemble_validation_ap_macro_damage']),'IC95_AP':[sig2(row['bootstrap_ap_ci_low']),sig2(row['bootstrap_ap_ci_high'])],'delta_vs_30s':sig2(row['delta_vs_reference']),'IC95_delta':[sig2(row['delta_vs_reference_ci_low']),sig2(row['delta_vs_reference_ci_high'])],'no_inferior':'Sí' if row['noninferior'] else 'No'} for row in neural_result['minilm']['bootstrap']['comparisons']]\n"
                "    show_table('MiniLM robusto: resultados reportables',minilm_rows,max_rows=len(CANDIDATE_SECONDS))\n"
                "    ollama_boot={float(row['chunk_seconds']):row for row in neural_result['ollama']['bootstrap']['comparisons']}\n"
                "    ollama_rows=[]\n"
                "    for row in neural_result['ollama']['duration_results']:\n"
                "        boot=ollama_boot[float(row['chunk_seconds'])]\n"
                "        ollama_rows.append({'longitud_s':row['chunk_seconds'],'válidas':f\"{row['successful_rows']}/{row['requested_rows']}\",'tasa_esquema':sig2(row['valid_schema_rate']),'F1_macro_daños':sig2(row['f1_macro_damage']),'IC95_F1':[sig2(boot['bootstrap_f1_ci_low']),sig2(boot['bootstrap_f1_ci_high'])],'delta_vs_30s':sig2(boot['delta_vs_reference']),'IC95_delta':[sig2(boot['delta_vs_reference_ci_low']),sig2(boot['delta_vs_reference_ci_high'])],'exact_match':sig2(row['exact_label_set_match_rate']),'hamming_loss':sig2(row['hamming_loss_five'])})\n"
                "    show_table('Ollama robusto: resultados reportables',ollama_rows,max_rows=len(CANDIDATE_SECONDS))\n"
                "    hierarchy=neural_result['hierarchical_synthesis']\n"
                "    hierarchy_warning='conflict' in hierarchy['hierarchy_status'] or neural_result['reporting_status']!='complete'\n"
                "    show_summary('Síntesis jerárquica',hierarchy,tone='warning' if hierarchy_warning else 'success')\n"
                "    if neural_result['reporting_status']!='complete':\n"
                "        show_callout('Ejecución parcial y reanudable','Vuelva a ejecutar esta celda. Se conservarán respuestas válidas y ajustes MiniLM cuya firma coincida.',tone='warning')\n"
                "else:\n"
                "    show_callout('Perfil neuronal pendiente','Active RUN_NEURAL_ROBUST_TEST=True después del perfil clásico. El diseño solicita 25 cabezas MiniLM y 500 respuestas Ollama.',tone='neutral')",
            ),
            (
                "Etapa 4 — no inferioridad MiniLM 20 s frente a 30 s",
                "# Esta etapa resuelve el resultado MiniLM inconcluso del panel piloto.\n"
                "# No repite las cinco longitudes: usa predicción fuera de pliegue por video\n"
                "# sobre train, mantiene test cerrado y conserva 30 s como decisión clásica.\n"
                "if MINILM_NI_RESULT_PATH.is_file() and not FORCE_MINILM_20_30_RECOMPUTE:\n"
                "    minilm_ni_result=json.loads(MINILM_NI_RESULT_PATH.read_text(encoding='utf-8-sig'))\n"
                "    show_callout('Contraste 20 s–30 s cargado','Se leyó el JSON consolidado; no se recalcularon embeddings, ajustes ni bootstrap.',tone='success')\n"
                "elif RUN_MINILM_20_30_NONINFERIORITY_TEST:\n"
                "    show_callout('Contraste 20 s–30 s en ejecución','Primera corrida estimada: 12–20 min en CPU. El resultado final y cada pliegue son reanudables por firma.',tone='neutral')\n"
                "    minilm_ni_result=run_with_progress('No inferioridad MiniLM 20/30',run_minilm_20_30_noninferiority_test,TRANSCRIPTS,CHUNKS,DATASET,ROBUST_ROOT,MINILM_NI_ROOT,panel_size=MINILM_NI_PANEL_SIZE,minimum_damage_videos_per_label=MINILM_NI_MIN_DAMAGE_VIDEOS,folds=MINILM_NI_FOLDS,panel_selection_seed=MINILM_NI_PANEL_SELECTION_SEED,classical_seeds=ROBUST_SEEDS,repeat_seeds=MINILM_NI_REPEAT_SEEDS,model_id=NEURAL_MINILM_MODEL,revision=NEURAL_MINILM_REVISION,train_limit_per_fit=MINILM_NI_TRAIN_LIMIT,batch_size=NEURAL_MINILM_BATCH_SIZE,max_length=NEURAL_MINILM_MAX_LENGTH,bootstrap_replicates=MINILM_NI_BOOTSTRAP_REPLICATES,confidence_level=ROBUST_CONFIDENCE_LEVEL,noninferiority_margin=NEURAL_MINILM_NONINFERIORITY_MARGIN,bootstrap_seed=MINILM_NI_BOOTSTRAP_SEED,progress_unit='etapa')\n"
                "else:\n"
                "    minilm_ni_result=None\n"
                "if minilm_ni_result:\n"
                "    ni_rows=[{'longitud_s':row['chunk_seconds'],'AP_OOF':sig2(row['ensemble_validation_ap_macro_damage']),'IC95_AP':[sig2(row['bootstrap_ap_ci_low']),sig2(row['bootstrap_ap_ci_high'])],'delta_vs_30s':sig2(row['delta_vs_reference']),'IC95_delta':[sig2(row['delta_vs_reference_ci_low']),sig2(row['delta_vs_reference_ci_high'])],'no_inferior':'Sí' if row['noninferior'] else 'No'} for row in minilm_ni_result['bootstrap']['comparisons']]\n"
                "    show_table('MiniLM 20 s–30 s: contraste reportable',ni_rows,max_rows=2)\n"
                "    ni=minilm_ni_result['interpretation']\n"
                "    show_summary('Conclusión de no inferioridad',{'estado':ni['status'],'hipótesis_nula':ni['null_hypothesis'],'margen':ni['noninferiority_margin'],'delta_AP_20_menos_30':sig2(ni['delta_ap']),'IC95_delta':[sig2(ni['delta_ap_ci_low']),sig2(ni['delta_ap_ci_high'])],'no_inferior':'Sí' if ni['noninferior'] else 'No','superioridad_demostrada':'Sí' if ni['superiority_established'] else 'No','videos':minilm_ni_result['design']['panel_video_clusters'],'pliegues':minilm_ni_result['design']['folds'],'repeticiones':minilm_ni_result['design']['training_repeats'],'test_usado':False,'efecto_decisorio':'Ninguno; 30 s continúa como selección clásica'},tone='success' if ni['noninferior'] else 'warning')\n"
                "else:\n"
                "    show_callout('Contraste complementario omitido','El panel neuronal original permanece inconcluso; active RUN_MINILM_20_30_NONINFERIORITY_TEST=True.',tone='warning')",
            ),
            (
                "Lectura académica y límites de aplicación",
                "if neural_result:\n"
                "    show_summary('Roles no intercambiables',{'clásico':'decisorio; selecciona o conserva longitud','MiniLM robusto':'confirmatorio; sensibilidad a representación neuronal continua','MiniLM 20/30 OOF':'complementario; contrasta no inferioridad interna sin cambiar la selección','Ollama':'confirmatorio; sensibilidad semántica y factibilidad de salida estructurada','agregación_entre_familias':'ninguna','política_de_conflicto':'conservar la selección clásica hasta validación humana independiente','artefacto_perfil':NEURAL_RESULT_PATH,'artefacto_no_inferioridad':MINILM_NI_RESULT_PATH},tone='neutral')\n"
                "    ni_followup=globals().get('minilm_ni_result')\n"
                "    if ni_followup:\n"
                "        ni=ni_followup['interpretation']\n"
                "        show_summary('Conclusión conjunta actualizada',{'otra_longitud_demostró_ser_mejor_que_30s':False,'20s_no_inferior_a_30s_en_MiniLM':'Sí' if ni['noninferior'] else 'No','20s_superior_a_30s_en_MiniLM':'Sí' if ni['superiority_established'] else 'No','selección_principal_s':30,'justificación':'La evidencia complementaria no reemplaza el perfil clásico decisorio.'},tone='success')\n"
                "    show_callout('Alcance','Los intervalos describen este panel enriquecido de validation. No estiman prevalencia ni desempeño productivo. La confianza declarada por Ollama no se interpreta como probabilidad calibrada.',tone='warning')\n"
                "else:\n"
                "    show_callout('Resultados aún no ejecutados','No reporte expectativas como resultados. Ejecute la etapa neuronal y use su artefacto canónico.',tone='neutral')",
            ),
            (
                "Activación manual y reversible",
                "if USE_ROBUST_RECOMMENDATION:\n"
                "    if not ROBUST_RECOMMENDATION.is_file():\n"
                "        raise FileNotFoundError('Ejecute primero el perfil robusto clásico')\n"
                "    selected_seconds=float(json.loads(ROBUST_RECOMMENDATION.read_text(encoding='utf-8-sig'))['recommended_seconds'])\n"
                "    selection_source='01_02_robust_bootstrap_recommendation'\n"
                "else:\n"
                "    selected_seconds=float(MANUAL_CHUNK_SECONDS)\n"
                "    selection_source='01_02_manual'\n"
                "selected_config={**DEFAULT_CHUNKING_CONFIGURATION,'max_seconds':selected_seconds}\n"
                "if APPLY_CHUNK_SELECTION:\n"
                "    activation=activate_chunking_configuration(ROOT,selected_config,source=selection_source)\n"
                "    show_result('Configuración activada sin borrar derivados',activation,tone='success')\n"
                "else:\n"
                "    show_summary('Selección previsualizada',{'segundos':selected_seconds,'origen':selection_source,'regla':'Las pruebas neuronales no modifican automáticamente este valor.','acción':'Active APPLY_CHUNK_SELECTION=True; 01_03 materializará o restaurará la firma.'},tone='neutral')",
            ),
        ],
    )
    create(
        "flujo/01_datos/01_03_limpieza_troceado_incremental.ipynb",
        "01.03 · Limpieza y troceado incremental",
        "Recompone la vista canónica desde los checkpoints sincronizados, recupera VTT locales utilizables y crea chunks deterministas únicamente para transcripciones nuevas o modificadas.",
        "La normalización NFKC aplicada al texto sigue las formas de normalización Unicode "
        "[@unicode2025normalization], y las huellas de transcripción, texto e identificadores estables "
        "usan SHA-256 [@nist2015sha]. La longitud, los límites de caracteres, el solapamiento y las reglas "
        "de deduplicación son parámetros locales versionados. Cada firma tiene un archivo recuperable: "
        "cambiarla mueve los derivados vigentes y volver a una firma restaura sus bytes verificados.",
        [
            (
                "Preparación reproducible y consolidación local",
                "from moderacion_peru.acquisition import consolidate_available_transcripts, materialize_transcripts_by_channel\nfrom moderacion_peru.chunk_optimization import activate_chunking_configuration, load_chunking_configuration\nfrom moderacion_peru.incremental import materialize_chunk_records\nfrom moderacion_peru.io import read_jsonl\nSOURCE=ROOT/'datos/raw/transcripts_raw.jsonl'\nTRANSCRIPTS_BY_CHANNEL=ROOT/'datos/raw/transcripts_by_channel'\nTRANSCRIPTS_CACHE=ROOT/'datos/raw/transcripts_cache'\nVTT_BY_VIDEO=ROOT/'datos/raw/vtt_by_video'\nOUTPUT=ROOT/'datos/processed/chunks_v2.jsonl'\nVERSION_INDEX=ROOT/'datos/processed/chunking_v2_versions.jsonl'\nMATERIALIZATION_MANIFEST=ROOT/'datos/processed/chunk_materialization_manifest.json'\nCHUNK_CONFIG_PATH=ROOT/'config/chunking.json'\nREBUILD_CHUNKS_FROM_ZERO=False  # True: copia recuperable y reconstrucción total; después vuelva a False\nCHUNK_CONFIG=load_chunking_configuration(CHUNK_CONFIG_PATH)\nactivation=activate_chunking_configuration(ROOT,CHUNK_CONFIG,source='01_03_materialization')\nconsolidation=consolidate_available_transcripts(ROOT,SOURCE,cache_dir=TRANSCRIPTS_CACHE,channel_dir=TRANSCRIPTS_BY_CHANNEL,vtt_dir=VTT_BY_VIDEO if VTT_BY_VIDEO.is_dir() else None)\nchannel_checkpoint=materialize_transcripts_by_channel(SOURCE,TRANSCRIPTS_BY_CHANNEL)\nshow_result('Estado de la configuración de chunks',activation,tone='success')\nshow_summary('Cobertura consolidada antes del troceado',{'videos_canónicos':consolidation['canonical_videos'],'VTT_recuperados':consolidation['vtt_added'],'VTT_demasiado_cortos':consolidation['vtt_recovery']['too_short'],'partes_por_canal':channel_checkpoint['total_channel_files'],'canales':channel_checkpoint['total_channels'],'canónico_local':SOURCE,'checkpoint_Git':TRANSCRIPTS_BY_CHANNEL},tone='success')\nif consolidation['vtt_recovery']['too_short_records']:\n    show_table('VTT excluidos por menos de 200 caracteres',consolidation['vtt_recovery']['too_short_records'],max_rows=len(consolidation['vtt_recovery']['too_short_records']))",
            ),
            (
                "Materialización",
                "from tqdm.auto import tqdm\ntranscript_total=consolidation['canonical_videos']\nchunk_progress=tqdm(total=transcript_total,desc='Materializando transcripciones',unit='video')\ndef report_chunk_progress(event):\n    chunk_progress.update(event.get('advance',1))\n    chunk_progress.set_postfix(nuevos=event.get('new_or_changed_videos',0),sin_cambios=event.get('unchanged_videos',0),chunks_video=event.get('generated_chunks_for_video',0))\ntry:\n    materialization=materialize_chunk_records(ROOT,read_jsonl(SOURCE),source_path=SOURCE,output_path=OUTPUT,version_index_path=VERSION_INDEX,manifest_path=MATERIALIZATION_MANIFEST,rebuild=REBUILD_CHUNKS_FROM_ZERO,progress_callback=report_chunk_progress,**CHUNK_CONFIG)\nfinally:\n    chunk_progress.close()\nshow_result('Resultado de limpieza y troceado',materialization['stats'],tone='success')\nshow_summary('Cobertura final reportable',{'transcripciones':materialization['coverage']['transcript_videos'],'videos_con_chunks':materialization['coverage']['videos_with_chunks'],'videos_sin_chunks':materialization['coverage']['videos_without_chunks'],'chunks_totales':materialization['outputs']['chunks']['rows'],'reconstrucción_total':REBUILD_CHUNKS_FROM_ZERO,'respaldo':materialization['backup'],'manifiesto':MATERIALIZATION_MANIFEST},tone='warning' if materialization['coverage']['videos_without_chunks'] else 'success')\nif materialization['coverage']['video_ids_without_chunks']:\n    show_table('Videos evaluados sin chunks materializables',[{'video_id':video_id} for video_id in materialization['coverage']['video_ids_without_chunks']],max_rows=len(materialization['coverage']['video_ids_without_chunks']))",
            ),
        ],
    )
    create(
        "flujo/02_etiquetado/02_00_preparacion_bundle_colab.ipynb",
        "02.00 · Publicador manual opcional del bundle de Colab",
        "Alternativa manual para descargar —o cargar—, verificar y publicar el bundle. Los demás cuadernos Colab ya incorporan esta operación y la ejecutan automáticamente solo cuando falta su release exacto.",
        "La identidad estable combina los SHA-256 del código y de las entradas comprimidas. El cuaderno "
        "verifica el `bundle_id` y cada artefacto antes de escribir en Drive; publica los artefactos y "
        "deja el manifiesto y `latest.json` para el final, de modo que una interrupción no se presente "
        "como una versión completa [@nist2015sha]. El almacenamiento de la VM de Colab es efímero y el "
        "montaje de Drive solicita autorización integrada durante la sesión [@googlecolab2026faq]. "
        "La revisión Git, la carpeta de Drive y el momento de publicación son decisiones locales.",
        [
            (
                "Configuración de la fuente y el destino",
                "RUN_PUBLISH_BUNDLE=False  # Cambie a True después de revisar esta celda.\n"
                "BUNDLE_SOURCE='github'     # 'github' o 'local_upload'.\n"
                "# github: descarga la versión ya sincronizada; local_upload: selector del navegador.\n"
                "GITHUB_REPOSITORY='lkoc/Trabajo_PLN-MIA-Grupo4'\n"
                "GITHUB_REF='main'          # Rama, etiqueta o commit reproducible.\n"
                "GITHUB_BUNDLE_PATH='resultados/colab_bundle'\n"
                "COLAB_DRIVE_FOLDER='ModeracionPeru_Colab'\n"
                "STAGING=Path('/content/moderacion_peru_bundle_source')\n"
                "REQUIRED_BUNDLE_FILES=('project_core.zip','chunks_v2.jsonl.gz','chunks_deepseek_historicos.jsonl.gz','deepseek_flash_historico.jsonl.gz','deepseek_pro_historico_principal.jsonl.gz','deepseek_pro_historico_umbral.jsonl.gz','deepseek_pro_historico_sospechosos.jsonl.gz','dataset_5_salidas.jsonl.gz','bundle_manifest.json')\n\n"
                "if BUNDLE_SOURCE not in {'github','local_upload'}:\n"
                "    raise ValueError(\"BUNDLE_SOURCE debe ser 'github' o 'local_upload'.\")\n"
                "show_summary('Publicación preparada',{'ejecutar':RUN_PUBLISH_BUNDLE,'fuente':BUNDLE_SOURCE,'repositorio':GITHUB_REPOSITORY if BUNDLE_SOURCE=='github' else None,'revisión':GITHUB_REF if BUNDLE_SOURCE=='github' else None,'archivos_locales_requeridos':REQUIRED_BUNDLE_FILES if BUNDLE_SOURCE=='local_upload' else None,'destino':f'Mi unidad/{COLAB_DRIVE_FOLDER}/bundle_releases'},tone='neutral')\n"
                "show_callout('Uso opcional','No necesita ejecutar 02_00 antes de cada consumidor. Úselo solo si desea publicar o auditar manualmente un bundle; 02_01 y 03_02–03_06b detectan y publican por sí mismos el release faltante.',tone='info')",
            ),
            (
                "Adquisición y verificación integral",
                "def _download_file(url, destination):\n"
                "    request=urllib.request.Request(url,headers={'User-Agent':'ModeracionPeru-Colab-Bundle/1.0'})\n"
                "    with urllib.request.urlopen(request,timeout=120) as response, destination.open('wb') as target:\n"
                "        total=int(response.headers.get('Content-Length') or 0)\n"
                "        with tqdm(total=total or None,desc=f'Descargando {destination.name}',unit='B',unit_scale=True) as bar:\n"
                "            while block:=response.read(1024*1024):\n"
                "                target.write(block); bar.update(len(block))\n\n"
                "def _prepare_staging():\n"
                "    if STAGING != Path('/content/moderacion_peru_bundle_source'):\n"
                "        raise RuntimeError('STAGING debe permanecer dentro del espacio efímero controlado de Colab.')\n"
                "    if STAGING.exists():\n"
                "        shutil.rmtree(STAGING)\n"
                "    STAGING.mkdir(parents=True)\n\n"
                "if RUN_PUBLISH_BUNDLE:\n"
                "    _prepare_staging()\n"
                "    if BUNDLE_SOURCE=='github':\n"
                "        encoded_ref=urllib.parse.quote(GITHUB_REF,safe='')\n"
                "        base=f'https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{encoded_ref}/{GITHUB_BUNDLE_PATH}'\n"
                "        cache_key=urllib.parse.quote(COLAB_NOTEBOOK_BUILD_BUNDLE_ID,safe='')\n"
                "        for name in ('bundle_manifest.json',*REQUIRED_BUNDLE_FILES[:-1]):\n"
                "            _download_file(f'{base}/{name}?bundle_id={cache_key}',STAGING/name)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        show_callout('Seleccione nueve archivos','Abra resultados/colab_bundle en su PC y seleccione simultáneamente los nueve archivos indicados. El navegador es el puente; Colab no puede leer D: directamente.',tone='info')\n"
                "        uploaded=files.upload()\n"
                "        missing=sorted(set(REQUIRED_BUNDLE_FILES)-set(uploaded))\n"
                "        if missing:\n"
                "            raise FileNotFoundError(f'Faltaron archivos en la carga local: {missing}')\n"
                "        unexpected=sorted(set(uploaded)-set(REQUIRED_BUNDLE_FILES))\n"
                "        if unexpected:\n"
                "            show_callout('Archivos adicionales ignorados',str(unexpected),tone='warning')\n"
                "        for name in REQUIRED_BUNDLE_FILES:\n"
                "            (STAGING/name).write_bytes(uploaded[name])\n"
                "    bundle_manifest=_verify_bundle(STAGING)\n"
                "    show_result('Bundle adquirido y verificado',{'bundle_id':bundle_manifest['bundle_id'],'core_sha256':bundle_manifest['core']['sha256'],'fuente':BUNDLE_SOURCE,'archivos':{name:{'bytes':(STAGING/name).stat().st_size,'sha256':_sha256(STAGING/name)} for name in REQUIRED_BUNDLE_FILES}},tone='success')\n"
                "else:\n"
                "    bundle_manifest=None\n"
                "    show_callout('Adquisición desactivada','Revise la configuración y cambie RUN_PUBLISH_BUNDLE=True. Aún no se descargó ni cargó nada.',tone='neutral')",
            ),
            (
                "Publicación inmutable y actualización de latest.json",
                "def _copy_with_progress(source,destination):\n"
                "    with source.open('rb') as raw, destination.open('wb') as target, tqdm(total=source.stat().st_size,desc=f'Publicando {source.name}',unit='B',unit_scale=True) as bar:\n"
                "        while block:=raw.read(1024*1024):\n"
                "            target.write(block); bar.update(len(block))\n\n"
                "if RUN_PUBLISH_BUNDLE:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive',force_remount=False)\n"
                "    DRIVE_ROOT=Path('/content/drive/MyDrive')/COLAB_DRIVE_FOLDER\n"
                "    RELEASES_DIR=DRIVE_ROOT/'bundle_releases'\n"
                "    RELEASES_DIR.mkdir(parents=True,exist_ok=True)\n"
                "    bundle_id=str(bundle_manifest['bundle_id'])\n"
                "    release_dir=RELEASES_DIR/bundle_id\n"
                "    specs=_bundle_specs(bundle_manifest)\n"
                "    if release_dir.exists():\n"
                "        _verify_bundle(release_dir,expected_bundle_id=bundle_id)\n"
                "        release_status='already_present_and_verified'\n"
                "    else:\n"
                "        partial=RELEASES_DIR/f'.{bundle_id}.partial-{uuid.uuid4().hex}'\n"
                "        partial.mkdir()\n"
                "        try:\n"
                "            for name,_ in specs:\n"
                "                _copy_with_progress(STAGING/name,partial/name)\n"
                "            _copy_with_progress(STAGING/'bundle_manifest.json',partial/'bundle_manifest.json')\n"
                "            _verify_bundle(partial,expected_bundle_id=bundle_id)\n"
                "            os.replace(partial,release_dir)\n"
                "        finally:\n"
                "            if partial.exists():\n"
                "                shutil.rmtree(partial)\n"
                "        _verify_bundle(release_dir,expected_bundle_id=bundle_id)\n"
                "        release_status='published_and_verified'\n"
                "    pointer={'schema_version':'1.0.0','bundle_id':bundle_id,'core_sha256':bundle_manifest['core']['sha256'],'manifest_sha256':_sha256(release_dir/'bundle_manifest.json'),'published_at':datetime.now(timezone.utc).isoformat()}\n"
                "    latest_path=RELEASES_DIR/'latest.json'\n"
                "    partial_latest=RELEASES_DIR/f'.latest-{uuid.uuid4().hex}.json'\n"
                "    partial_latest.write_text(json.dumps(pointer,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')\n"
                "    os.replace(partial_latest,latest_path)\n"
                "    persisted_pointer=_read_json(latest_path)\n"
                "    if persisted_pointer!=pointer:\n"
                "        raise RuntimeError('latest.json no conservó exactamente el puntero publicado')\n"
                "    bundle_result={'status':'published_to_drive','release_status':release_status,'bundle_id':bundle_id,'release_dir':str(release_dir),'latest_pointer':str(latest_path),'manifest_sha256':pointer['manifest_sha256']}\n"
                "    show_result('Versión de Colab publicada en Drive',bundle_result,tone='success')\n"
                "    show_callout('Siguiente paso','Puede abrir directamente 02_01 o 03_02–03_06b; cada consumidor volverá a verificar esta versión antes de importar el proyecto.',tone='success')\n"
                "else:\n"
                "    show_callout('Publicación desactivada','No se montó Drive ni se escribió ningún archivo.',tone='neutral')",
            ),
        ],
        colab_publisher=True,
    )
    create(
        "flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb",
        "02.01 · Cascada de etiquetado calibrada",
        "Reproduce el patrón histórico Flash→Pro: calibra una primera pasada económica, etiqueta el corpus por lotes y dirige los casos riesgosos a un revisor más capaz.",
        "La procedencia de `deepseek-v4-flash` y `deepseek-v4-pro` está documentada por el proveedor "
        "[@deepseek2026v4], al igual que sus precios por tokens y caché [@deepseek2026pricing]. La selección "
        "dirigida pertenece a la familia de aprendizaje activo [@settles2009active]. El acuerdo Flash–Pro "
        "calibra una regla operativa, pero no constituye *ground truth*; las tareas subjetivas conservan una "
        "instancia humana final independiente [@schroeder2025llmassisted]. Los umbrales, el presupuesto, "
        "el control seguro y la precedencia son decisiones locales auditables.",
        [
            (
                "Configuración explícita y credencial",
                "import os\n"
                "from pathlib import Path\n"
                "from moderacion_peru.providers import DeepSeekProvider\n\n"
                "if globals().get('IN_COLAB') and not os.getenv('DEEPSEEK_API_KEY'):\n"
                "    from google.colab import userdata\n\n"
                "    try:\n"
                "        os.environ['DEEPSEEK_API_KEY']=userdata.get('DEEPSEEK_API_KEY') or ''\n"
                "    except Exception:\n"
                "        pass\n\n"
                "SOURCE=COLAB_CONTEXT.input('chunks_v2') if COLAB_CONTEXT else ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "CAMPAIGN_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'datos/etiquetado/cascada_deepseek_v4'\n"
                "CAMPAIGN_ROOT.mkdir(parents=True,exist_ok=True)\n"
                "HISTORICAL_CHUNKS=COLAB_CONTEXT.input('chunks_deepseek_historicos') if COLAB_CONTEXT else ROOT/'datos/processed/chunks_para_etiquetar.jsonl'\n"
                "HISTORICAL_FLASH_SOURCES=(COLAB_CONTEXT.input('deepseek_flash_historico'),) if COLAB_CONTEXT else (ROOT/'datos/etiquetado/llm_api/deepseek-v4-flash_labeled_chunks_seed42.jsonl',)\n"
                "HISTORICAL_PRO_SOURCES=(COLAB_CONTEXT.input('deepseek_pro_historico_principal'),COLAB_CONTEXT.input('deepseek_pro_historico_umbral'),COLAB_CONTEXT.input('deepseek_pro_historico_sospechosos')) if COLAB_CONTEXT else (ROOT/'datos/etiquetado/llm_api/deepseek-v4-pro_revision_de_deepseek-v4-flash_seed42.jsonl',ROOT/'datos/etiquetado/llm_api/deepseek-v4-pro_revision_umbral_recalibrado_t090_seed42.jsonl',ROOT/'datos/etiquetado/llm_api/deepseek-v4-pro_revision_sospechosos_gruesos_seed42.jsonl')\n"
                "HISTORICAL_PROMPT_SHA256='52d4fec14ad433d35ec20de5f51a6954aad69dcedd1422059419dcecc2f9e778'\n"
                "PRESERVE_PROMPT_POLICY_SHA256='433321cf7b41f997bb277ae87bc9fee01767d225a0fe49bea2cb918239dc1f06'\n"
                "PREVIOUS_PRIMARY_PATH=CAMPAIGN_ROOT/'primary_flash.jsonl'\n"
                "PREVIOUS_REVIEW_PATH=CAMPAIGN_ROOT/'review_pro.jsonl'\n"
                "PRIMARY_PATH=CAMPAIGN_ROOT/'primary_flash_v3_2.jsonl'\n"
                "REVIEW_PATH=CAMPAIGN_ROOT/'review_pro_v3_2.jsonl'\n"
                "RECOVER_HISTORICAL=True  # Recupera solo coincidencias exactas 1:1; nunca transfiere segmentos distintos.\n"
                "AUTO_PUBLISH_CHECKPOINTS=True  # En Colab publica TAR.GZ atómico al recuperar, periódicamente y al interrumpir.\n"
                "DRIVE_CHECKPOINT_EVERY_BATCHES=3  # 3 ventanas × 640 chunks; cada grupo de 5 ya queda fsync local.\n"
                "RUN_API_PREFLIGHT=True  # Consulta /models y /user/balance sin enviar textos ni consumir tokens de etiquetado.\n"
                "RUN_CALIBRATION=False  # Primero: panel pareado Flash–Pro. Costo esperado muy bajo.\n"
                "RUN_PRIMARY=True  # Procesa únicamente los chunk_id nuevos o todavía pendientes en Flash.\n"
                "RUN_DIRECTED_REVIEW=True  # Reanuda Pro únicamente sobre la cola presupuestada pendiente.\n"
                "CALIBRATION_PANEL_SIZE=1000  # Aún breve; permite evaluar LI95%≈0.95 con potencia útil.\n"
                "PRIMARY_LIMIT=None  # None para TODOS y solo los pendientes; use 20 únicamente para un smoke test. Nunca lo deje en blanco.\n"
                "REVIEW_LIMIT=None   # Campaña: TODA y solo la cola dirigida pendiente.\n"
                "PROCESSING_BATCH_SIZE=640  # Ventana persistible: 128 solicitudes de 5; Pro limita el fan-out a 64.\n"
                "MAX_PRIMARY_COST_USD=60.0\n"
                "MAX_REVIEW_COST_USD=None  # Sin bloqueo artificial: el saldo real y los checkpoints gobiernan la reanudación.\n"
                "BALANCE_REFRESH_SECONDS=60.0  # Consulta de saldo sin corpus durante la ejecución.\n"
                "LOW_BALANCE_WARNING_USD=2.0\n"
                "CACHE_ALERT_AFTER_REQUESTS=50\n"
                "MIN_CACHE_HIT_RATE=0.50\n"
                "SAFE_CONTROL_RATE=0.01  # Control seguro aleatorio reproducible del 1%.\n"
                "REVIEW_CONFIDENCE_THRESHOLD=0.85  # Pro revisa seguros Flash < 0.85; el 0.95 sigue siendo solo diagnóstico.\n"
                "MAX_NEEDS_REVIEW_FOR_PRO=36_000  # Abstenciones Flash de menor confianza; desempate SHA-256 reproducible.\n\n"
                "OBSERVED_REVIEW_COST_PER_1000_USD=0.335785  # Tramo Pro medido: US$4.727523 / 14 079 respuestas.\n"
                "RECOMMENDED_REVIEW_START_BALANCE_USD=15.00  # Solo advertencia; no bloquea una corrida con saldo menor.\n\n"
                "primary_provider=DeepSeekProvider(model='deepseek-v4-flash',max_workers=128,records_per_request=5,cache_warmup_requests=1,max_cost_usd=MAX_PRIMARY_COST_USD,label_source='deepseek_remote',operational_prompt_path=OPERATIONAL_PROMPT)\n"
                "reviewer_provider=DeepSeekProvider(model='deepseek-v4-pro',max_workers=64,records_per_request=5,cache_warmup_requests=1,max_cost_usd=MAX_REVIEW_COST_USD,label_source='llm_remote_review',operational_prompt_path=OPERATIONAL_PROMPT)\n"
                "primary_probe=primary_provider.probe(); reviewer_probe=reviewer_provider.probe()\n"
                "expected_thinking={'type':'disabled'}\n"
                "expected_response_format={'type':'json_object'}\n"
                "expected_cache_usage_fields=['prompt_cache_hit_tokens','prompt_cache_miss_tokens']\n"
                "if primary_probe['thinking'] != expected_thinking or reviewer_probe['thinking'] != expected_thinking:\n"
                "    raise RuntimeError('02_01 exige DeepSeek V4 en modo non-thinking para Flash y Pro')\n"
                "if primary_probe['response_format'] != expected_response_format or reviewer_probe['response_format'] != expected_response_format or primary_probe['output_contract']['root_key'] != 'annotations' or reviewer_probe['output_contract']['root_key'] != 'annotations':\n"
                "    raise RuntimeError('02_01 exige JSON object con raíz annotations para Flash y Pro')\n"
                "if primary_probe['context_cache']['mode'] != 'automatic_prefix' or reviewer_probe['context_cache']['mode'] != 'automatic_prefix':\n"
                "    raise RuntimeError('02_01 exige caché de contexto automática con prefijo estable')\n"
                "if primary_probe['context_cache']['verified_from_usage_fields'] != expected_cache_usage_fields or reviewer_probe['context_cache']['verified_from_usage_fields'] != expected_cache_usage_fields:\n"
                "    raise RuntimeError('02_01 debe medir aciertos y fallos reales de caché en la respuesta de DeepSeek')\n"
                "if not primary_probe['credential_configured']:\n"
                "    show_callout('Falta credencial','Defina DEEPSEEK_API_KEY en el entorno o como secreto de Colab. El preflight no consume crédito.',tone='warning')\n"
                "show_result('Primera pasada',primary_probe,tone='success')\n"
                "show_result('Revisor dirigido',reviewer_probe,tone='success')\n"
                "show_summary('Modo DeepSeek verificado',{'Flash':primary_probe['thinking'],'Pro':reviewer_probe['thinking'],'JSON_Flash':primary_probe['response_format'],'JSON_Pro':reviewer_probe['response_format'],'contrato_salida':primary_probe['output_contract'],'caché_Flash':primary_probe['context_cache'],'caché_Pro':reviewer_probe['context_cache'],'lote':5,'concurrencia_Flash':primary_probe['max_workers'],'concurrencia_Pro':reviewer_probe['max_workers']},tone='success')\n"
                "if RUN_API_PREFLIGHT and primary_probe['credential_configured']:\n"
                "    flash_connection=primary_provider.validate_connection(); pro_connection=reviewer_provider.validate_connection()\n"
                "    if not flash_connection['model_available'] or not pro_connection['model_available']:\n"
                "        raise RuntimeError('Flash o Pro no aparece disponible en el catálogo de DeepSeek')\n"
                "    initial_balance=primary_provider.balance_summary()\n"
                "    show_result('Credencial, modelos y saldo verificados; no se enviaron textos',{'Flash':flash_connection,'Pro':pro_connection,'saldo':initial_balance},tone='success' if initial_balance['is_available'] else 'warning')\n"
                "show_summary('Rutas y activación',{'entrada':SOURCE,'campaña':CAMPAIGN_ROOT,'recuperación_histórica':RECOVER_HISTORICAL,'checkpoint_drive_automático':bool(COLAB_CONTEXT and AUTO_PUBLISH_CHECKPOINTS),'calibración':RUN_CALIBRATION,'primaria':RUN_PRIMARY,'revisión':RUN_DIRECTED_REVIEW,'umbral_seguro_Flash_a_Pro':REVIEW_CONFIDENCE_THRESHOLD,'máximo_abstenciones_Pro':MAX_NEEDS_REVIEW_FOR_PRO,'control_seguro':SAFE_CONTROL_RATE,'tope_reanudación_USD':MAX_REVIEW_COST_USD},tone='neutral')",
            ),
            (
                "Carga visible del corpus",
                "if globals().get('IN_COLAB'):\n"
                "    from tqdm.std import tqdm  # Salida textual visible también desde VS Code.\n"
                "else:\n"
                "    from tqdm.auto import tqdm\n"
                "from moderacion_peru.io import read_jsonl\n"
                "CHUNKS=list(tqdm(read_jsonl(SOURCE),desc='Cargando chunks',unit='chunk'))\n"
                "show_summary('Corpus disponible',{'chunks_totales':len(CHUNKS),'nota':'La recuperación histórica se ejecuta antes de calcular lo pendiente y su costo.'},tone='neutral')",
            ),
            (
                "Funciones de ejecución, avance y checkpoint",
                "from moderacion_peru.io import read_jsonl,write_json_atomic,write_jsonl_atomic\n"
                "from moderacion_peru.labeling import annotate_batched_incremental,historical_recovery_signature,recover_historical_annotations\n"
                "import time\n"
                "if COLAB_CONTEXT is not None:\n"
                "    from moderacion_peru.colab import publish_colab_outputs\n\n"
                "def labeling_progress(description,provider):\n"
                "    state={'bar':None,'last_balance_check':0.0,'balance':None,'balance_error':None,'cache_alerted':False,'low_balance_alerted':False}\n"
                "    def refresh_balance(*,force=False):\n"
                "        now=time.monotonic()\n"
                "        if not force and now-state['last_balance_check'] < BALANCE_REFRESH_SECONDS: return\n"
                "        state['last_balance_check']=now\n"
                "        try:\n"
                "            state['balance']=provider.balance_summary(); state['balance_error']=None\n"
                "            total=state['balance']['total_balance_usd']\n"
                "            if total <= LOW_BALANCE_WARNING_USD and not state['low_balance_alerted']:\n"
                "                tqdm.write(f'⚠ Saldo DeepSeek bajo: US${total:.2f}. Considere recargar antes de continuar.')\n"
                "                state['low_balance_alerted']=True\n"
                "            elif total > LOW_BALANCE_WARNING_USD:\n"
                "                state['low_balance_alerted']=False\n"
                "        except Exception as exc:\n"
                "            message=f'{type(exc).__name__}: {exc}'\n"
                "            if message != state['balance_error']: tqdm.write(f'⚠ No se pudo actualizar el saldo DeepSeek: {message}')\n"
                "            state['balance_error']=message\n"
                "    def update_postfix(event):\n"
                "        bar=state.get('bar'); usage=event.get('provider_usage') or {}; cache=usage.get('cache_hit_rate')\n"
                "        balance=state.get('balance') or {}; total=balance.get('total_balance_usd')\n"
                "        if bar is not None:\n"
                "            bar.set_postfix(ok=event.get('labeled',0),errores=event.get('errors',0),gastado_USD=f\"{usage.get('estimated_cost_usd',0):.4f}\",saldo_USD='—' if total is None else f'{total:.2f}',caché='—' if cache is None else f'{100*cache:.1f}%')\n"
                "        if usage.get('requests',0) >= CACHE_ALERT_AFTER_REQUESTS and cache is not None and cache < MIN_CACHE_HIT_RATE and not state['cache_alerted']:\n"
                "            tqdm.write(f'⚠ Caché DeepSeek baja ({100*cache:.1f}%). Revise antes de ampliar la campaña; el progreso ya guardado no se pierde.')\n"
                "            state['cache_alerted']=True\n"
                "    def callback(event):\n"
                "        if event['status']=='phase_started':\n"
                "            if state.get('bar') is not None: state['bar'].close()\n"
                "            label='Verificando progreso guardado' if event['phase']=='existing_progress' else 'Buscando chunks pendientes'\n"
                "            state['bar']=tqdm(total=event.get('total'),desc=label,unit='chunk'); return\n"
                "        if event['status']=='phase_progress':\n"
                "            if state.get('bar') is not None: state['bar'].update(event.get('phase_advance',0))\n"
                "            return\n"
                "        if event['status']=='phase_finished':\n"
                "            if state.get('bar') is not None: state['bar'].close(); state['bar']=None\n"
                "            return\n"
                "        if event['status']=='started':\n"
                "            state['bar']=tqdm(total=event['selected'],desc=description,unit='chunk')\n"
                "            refresh_balance(force=True); update_postfix(event)\n"
                "            if state.get('balance') is not None and not state['balance']['is_available']:\n"
                "                state['bar'].close(); state['bar']=None\n"
                "                raise RuntimeError(f\"DeepSeek no tiene saldo disponible (US${state['balance']['total_balance_usd']:.2f}); recargue y vuelva a ejecutar. No se envió ningún chunk pendiente.\")\n"
                "            return\n"
                "        bar=state.get('bar')\n"
                "        if bar is not None and event.get('advance'):\n"
                "            bar.update(event['advance'])\n"
                "            refresh_balance(); update_postfix(event)\n"
                "        if event['status'] in {'finished','interrupted_checkpoint'} and bar is not None:\n"
                "            refresh_balance(force=True); update_postfix(event)\n"
                "            bar.close(); state['bar']=None\n"
                "    return callback\n\n"
                "def provider_run_metadata(provider,historical_recovery=None):\n"
                "    probe=provider.probe()\n"
                "    signature={'model':probe['model'],'thinking':probe['thinking'],'response_format':probe['response_format'],'output_contract':probe['output_contract'],'context_cache':probe['context_cache'],'prompt_sha256':probe['prompt_sha256'],'operational_prompt_sha256':probe['operational_prompt_sha256'],'records_per_request':probe['records_per_request'],'label_source':probe['label_source']}\n"
                "    if historical_recovery is not None: signature['historical_recovery']=historical_recovery\n"
                "    return {'provider':signature,'taxonomy':'moderacion_peru_5_salidas_v2','taxonomy_version':'2.1.0'}\n\n"
                "FLASH_RECOVERY_CHUNKS=SOURCE if PREVIOUS_PRIMARY_PATH.is_file() else HISTORICAL_CHUNKS\n"
                "FLASH_RECOVERY_SOURCES=(PREVIOUS_PRIMARY_PATH,) if PREVIOUS_PRIMARY_PATH.is_file() else HISTORICAL_FLASH_SOURCES\n"
                "FLASH_RECOVERY_PROMPT=PRESERVE_PROMPT_POLICY_SHA256 if PREVIOUS_PRIMARY_PATH.is_file() else HISTORICAL_PROMPT_SHA256\n"
                "PRO_RECOVERY_CHUNKS=SOURCE if PREVIOUS_REVIEW_PATH.is_file() else HISTORICAL_CHUNKS\n"
                "PRO_RECOVERY_SOURCES=(PREVIOUS_REVIEW_PATH,) if PREVIOUS_REVIEW_PATH.is_file() else HISTORICAL_PRO_SOURCES\n"
                "PRO_RECOVERY_PROMPT=PRESERVE_PROMPT_POLICY_SHA256 if PREVIOUS_REVIEW_PATH.is_file() else HISTORICAL_PROMPT_SHA256\n"
                "FLASH_HISTORY_SIGNATURE=historical_recovery_signature(FLASH_RECOVERY_CHUNKS,FLASH_RECOVERY_SOURCES,expected_model='deepseek-v4-flash',historical_prompt_sha256=FLASH_RECOVERY_PROMPT) if RECOVER_HISTORICAL else None\n"
                "PRO_HISTORY_SIGNATURE=historical_recovery_signature(PRO_RECOVERY_CHUNKS,PRO_RECOVERY_SOURCES,expected_model='deepseek-v4-pro',historical_prompt_sha256=PRO_RECOVERY_PROMPT) if RECOVER_HISTORICAL else None\n"
                "PRIMARY_RUN_METADATA=provider_run_metadata(primary_provider,FLASH_HISTORY_SIGNATURE)\n"
                "REVIEW_RUN_METADATA=provider_run_metadata(reviewer_provider,PRO_HISTORY_SIGNATURE)\n"
                "if RECOVER_HISTORICAL:\n"
                "    primary_recovery=recover_historical_annotations(CHUNKS,FLASH_RECOVERY_CHUNKS,FLASH_RECOVERY_SOURCES,PRIMARY_PATH,expected_model='deepseek-v4-flash',historical_prompt_sha256=FLASH_RECOVERY_PROMPT,run_metadata=PRIMARY_RUN_METADATA)\n"
                "    review_recovery=recover_historical_annotations(CHUNKS,PRO_RECOVERY_CHUNKS,PRO_RECOVERY_SOURCES,REVIEW_PATH,expected_model='deepseek-v4-pro',historical_prompt_sha256=PRO_RECOVERY_PROMPT,run_metadata=REVIEW_RUN_METADATA,label_source='llm_remote_review_historical_recovered')\n"
                "    show_result('Recuperación exacta de opiniones previas',{'Flash':primary_recovery,'Pro':review_recovery,'salida_Flash_vigente':PRIMARY_PATH,'salida_Pro_vigente':REVIEW_PATH,'regla_prompt':'se conserva el prompt_sha256 original de cada opinión; solo los pendientes usan 3.2.0'},tone='success')\n"
                "    if COLAB_CONTEXT is not None and AUTO_PUBLISH_CHECKPOINTS and (primary_recovery['recovered_new'] or review_recovery['recovered_new']):\n"
                "        show_result('Checkpoint histórico publicado en Drive',publish_colab_outputs(COLAB_CONTEXT),tone='success')\n\n"
                "primary_pending=primary_recovery['pending_current_after_recovery'] if RECOVER_HISTORICAL else len(CHUNKS)\n"
                "# Consumo Flash medido en el histórico por cada 5 000 chunks y tasa de caché observada de 78.56%.\n"
                "scale=primary_pending/5000; input_m=8.28*scale; output_m=0.724*scale; observed_cache_rate=0.7856\n"
                "cost_no_cache=input_m*0.14+output_m*0.28\n"
                "cost_observed_cache=input_m*((1-observed_cache_rate)*0.14+observed_cache_rate*0.0028)+output_m*0.28\n"
                "show_summary('Costo Flash de lo realmente pendiente',{'total_actual':len(CHUNKS),'recuperado_o_ya_guardado':len(CHUNKS)-primary_pending,'pendiente_Flash':primary_pending,'entrada_proyectada_M':round(input_m,2),'salida_proyectada_M':round(output_m,2),'sin_caché_USD':round(cost_no_cache,2),'con_caché_histórica_78.56%_USD':round(cost_observed_cache,2),'tope_configurado_USD':MAX_PRIMARY_COST_USD},tone='success')\n\n"
                "def checkpoint_callback_for(output):\n"
                "    checkpoint_path=output.with_suffix(output.suffix+'.checkpoint.json')\n"
                "    def callback(event):\n"
                "        write_json_atomic(checkpoint_path,event)\n"
                "        if COLAB_CONTEXT is not None and AUTO_PUBLISH_CHECKPOINTS and event['status'] in {'periodic_checkpoint','interrupted_checkpoint'}:\n"
                "            publish_colab_outputs(COLAB_CONTEXT)\n"
                "    return callback\n\n"
                "def run_campaign(rows,provider,output_name,*,limit,description):\n"
                "    if not provider.probe()['credential_configured']:\n"
                "        raise RuntimeError('Falta DEEPSEEK_API_KEY: configúrela como variable local o secreto privado de Colab antes de etiquetar')\n"
                "    output=CAMPAIGN_ROOT/output_name\n"
                "    run_metadata=PRIMARY_RUN_METADATA if output_name=='primary_flash.jsonl' else REVIEW_RUN_METADATA if output_name=='review_pro.jsonl' else provider_run_metadata(provider)\n"
                "    result=annotate_batched_incremental(rows,provider,output,error_path=output.with_suffix('.errors.jsonl'),limit=limit,processing_batch_size=PROCESSING_BATCH_SIZE,progress_callback=labeling_progress(description,provider),checkpoint_callback=checkpoint_callback_for(output),checkpoint_every_batches=DRIVE_CHECKPOINT_EVERY_BATCHES,run_metadata=run_metadata,quarantine_invalid_progress=True)\n"
                "    try: result['account_balance']=provider.balance_summary()\n"
                "    except Exception as exc: result['account_balance_error']=f'{type(exc).__name__}: {exc}'\n"
                "    write_json_atomic(output.with_suffix('.result.json'),result)\n"
                "    if COLAB_CONTEXT is not None and AUTO_PUBLISH_CHECKPOINTS: publish_colab_outputs(COLAB_CONTEXT)\n"
                "    return output,result",
            ),
            (
                "Calibración corta Flash frente a Pro",
                "from moderacion_peru.labeling_calibration import select_calibration_panel,calibrate_primary_against_reviewer\n\n"
                "PANEL_PATH=CAMPAIGN_ROOT/'calibration_panel.jsonl'\n"
                "CALIBRATION_PATH=CAMPAIGN_ROOT/'calibration_flash_vs_pro.json'\n"
                "if RUN_CALIBRATION:\n"
                "    if PANEL_PATH.is_file():\n"
                "        panel=list(tqdm(read_jsonl(PANEL_PATH),desc='Recuperando panel congelado',unit='chunk'))\n"
                "        if len(panel)!=CALIBRATION_PANEL_SIZE: raise ValueError('El panel guardado no coincide con CALIBRATION_PANEL_SIZE; use otra carpeta de campaña')\n"
                "    else:\n"
                "        panel_progress={'bar':tqdm(total=len(CHUNKS),desc='Seleccionando panel',unit='chunk')}\n"
                "        def report_panel(event):\n"
                "            if event.get('advance'): panel_progress['bar'].update(event['advance'])\n"
                "        panel=select_calibration_panel(CHUNKS,panel_size=CALIBRATION_PANEL_SIZE,seed=42,max_per_video=1,progress_callback=report_panel)\n"
                "        panel_progress['bar'].close()\n"
                "        write_jsonl_atomic(PANEL_PATH,panel)\n"
                "    flash_path,flash_panel_result=run_campaign(panel,primary_provider,'calibration_flash.jsonl',limit=None,description='Calibración Flash')\n"
                "    pro_path,pro_panel_result=run_campaign(panel,reviewer_provider,'calibration_pro.jsonl',limit=None,description='Calibración Pro')\n"
                "    calibration=calibrate_primary_against_reviewer(read_jsonl(flash_path),read_jsonl(pro_path),minimum_auto_count=200,bootstrap_replicates=1000)\n"
                "    write_json_atomic(CALIBRATION_PATH,calibration)\n"
                "    show_table('Riesgo–cobertura por umbral',calibration['comparisons'],max_rows=len(calibration['comparisons']))\n"
                "    show_result('Umbral operativo calibrado',calibration,tone='success' if calibration['threshold_status']=='calibrated' else 'warning')\n"
                "elif CALIBRATION_PATH.is_file():\n"
                "    calibration=__import__('json').loads(CALIBRATION_PATH.read_text(encoding='utf-8-sig'))\n"
                "    show_table('Calibración guardada (sin repetir API)',calibration['comparisons'],max_rows=len(calibration['comparisons']))\n"
                "else:\n"
                "    calibration=None\n"
                "    show_callout('Calibración pendiente','Active RUN_CALIBRATION=True. El panel de 1 000 es pareado por chunk y el bootstrap agrupa por video.',tone='neutral')",
            ),
            (
                "Primera pasada completa con Flash",
                "if RUN_PRIMARY:\n"
                "    PRIMARY_PATH,primary_result=run_campaign(CHUNKS,primary_provider,'primary_flash_v3_2.jsonl',limit=PRIMARY_LIMIT,description='Primera pasada Flash')\n"
                "    show_result('Resultado Flash',primary_result,tone='success')\n"
                "else:\n"
                "    show_callout('Primera pasada desactivada','PRIMARY_LIMIT=None procesa todos y solo los pendientes; use 20 únicamente para un smoke mínimo. La salida reanuda por chunk_id y muestra costo, caché y saldo reales.',tone='neutral')",
            ),
            (
                "Enrutamiento y revisión dirigida con Pro",
                "from moderacion_peru.labeling_calibration import build_directed_review_queue\n"
                "REVIEW_QUEUE_PATH=CAMPAIGN_ROOT/'directed_review_queue.jsonl'\n"
                "if RUN_DIRECTED_REVIEW:\n"
                "    if calibration is None or not PRIMARY_PATH.is_file():\n"
                "        raise FileNotFoundError('Complete la calibración y la primera pasada antes de revisar')\n"
                "    primary_rows=list(tqdm(read_jsonl(PRIMARY_PATH),desc='Cargando propuestas Flash',unit='anotación'))\n"
                "    primary_ids={row['chunk_id'] for row in primary_rows}\n"
                "    paired_chunks=[row for row in tqdm(CHUNKS,desc='Uniendo chunks con Flash',unit='chunk') if row['chunk_id'] in primary_ids]\n"
                "    queue_progress={'bar':tqdm(total=len(paired_chunks),desc='Construyendo cola Pro',unit='chunk')}\n"
                "    def report_queue(event):\n"
                "        if event.get('advance'): queue_progress['bar'].update(event['advance'])\n"
                "    review_queue,routing=build_directed_review_queue(paired_chunks,primary_rows,confidence_threshold=REVIEW_CONFIDENCE_THRESHOLD,safe_control_rate=SAFE_CONTROL_RATE,max_needs_review=MAX_NEEDS_REVIEW_FOR_PRO,seed=42,progress_callback=report_queue)\n"
                "    queue_progress['bar'].close()\n"
                "    write_jsonl_atomic(REVIEW_QUEUE_PATH,review_queue); write_json_atomic(CAMPAIGN_ROOT/'routing_summary.json',routing)\n"
                "    reviewed_ids={row['chunk_id'] for row in read_jsonl(REVIEW_PATH)} if REVIEW_PATH.is_file() else set()\n"
                "    pending_review_count=sum(row['chunk_id'] not in reviewed_ids for row in review_queue)\n"
                "    projected_review_cost=pending_review_count*OBSERVED_REVIEW_COST_PER_1000_USD/1000\n"
                "    budget_balance=reviewer_provider.balance_summary()\n"
                "    show_summary('Previsión antes de enviar corpus a Pro',{'revisiones_Pro_preservadas':len(reviewed_ids),'cola_nueva_pendiente':pending_review_count,'controles_seguros_seleccionados':routing['routing_reasons'].get('safe_control',0),'costo_puntual_proyectado_USD':round(projected_review_cost,2),'tope_artificial_USD':MAX_REVIEW_COST_USD,'saldo_actual_USD':budget_balance['total_balance_usd'],'saldo_recomendado_no_bloqueante_USD':RECOMMENDED_REVIEW_START_BALANCE_USD},tone='success' if budget_balance['total_balance_usd']>=projected_review_cost else 'warning')\n"
                "    if budget_balance['total_balance_usd'] < RECOMMENDED_REVIEW_START_BALANCE_USD:\n"
                "        show_callout('Saldo menor que la recomendación',f\"Saldo Pro US${budget_balance['total_balance_usd']:.2f}; la referencia conservadora es US${RECOMMENDED_REVIEW_START_BALANCE_USD:.2f}, pero esto no bloquea la ejecución. El proveedor se detendrá si se agota el saldo y la reanudación continuará por chunk_id.\",tone='warning')\n"
                "    REVIEW_PATH,review_result=run_campaign(review_queue,reviewer_provider,'review_pro_v3_2.jsonl',limit=REVIEW_LIMIT,description='Revisión dirigida Pro')\n"
                "    show_summary('Enrutamiento operativo actualizado',routing,tone='success')\n"
                "    show_result('Resultado Pro',review_result,tone='success')\n"
                "else:\n"
                "    show_callout('Revisión dirigida desactivada','Active solo después de completar Flash. La regla presupuestada revisa todo daño, las 36 000 abstenciones de menor confianza, seguros con confianza menor que 0.85 y un control seguro aleatorio reproducible del 1%.',tone='neutral')",
            ),
            (
                "Resultados persistidos y reportables",
                "import json\n"
                "saved={}\n"
                "for path in sorted(CAMPAIGN_ROOT.glob('*.result.json')):\n"
                "    saved[path.stem.replace('.result','')]=json.loads(path.read_text(encoding='utf-8-sig'))\n"
                "if CALIBRATION_PATH.is_file():\n"
                "    current=json.loads(CALIBRATION_PATH.read_text(encoding='utf-8-sig'))\n"
                "    show_table('Tabla reportable de calibración',current['comparisons'],max_rows=len(current['comparisons']))\n"
                "    show_summary('Conclusión de calibración',{'estado':current['threshold_status'],'umbral':current['selected_threshold'],'pares':current['paired_chunks'],'referencia':current['reference_kind'],'bootstrap_agrupado_por_video':current['selected_threshold_cluster_bootstrap_95']},tone='success' if current['threshold_status']=='calibrated' else 'warning')\n"
                "show_result('Resultados recuperados sin repetir cálculos',saved,tone='success' if saved else 'neutral')\n"
                "show_callout('Límite inferencial','Flash–Pro es una calibración operativa y no reemplaza validación humana independiente. El score declarado no se interpreta como probabilidad estadística.',tone='warning')",
            ),
        ],
        colab_notebook_id="02_01",
        colab_requires_gpu=False,
    )
    create(
        "flujo/02_etiquetado/02_02_etiquetado_hf_qwen_colab.ipynb",
        "02.02 · Cascada HF–Qwen en Colab",
        "Ejecuta sin costo de API una primera pasada Qwen3-1.7B y dirige daño, dudas y controles a Qwen3-4B; es una alternativa experimental independiente de DeepSeek.",
        "Qwen3 es una familia multilingüe de pesos abiertos [@qwen2025qwen3]. Las revisiones exactas "
        "de `Qwen/Qwen3-1.7B` y `Qwen/Qwen3-4B` se fijan mediante sus tarjetas oficiales "
        "[@hf2026qwen17bcard] [@hf2026qwen4bcard]. La cascada carga los modelos secuencialmente en una "
        "GPU CUDA BF16; el perfil A100 de 40 GB duplica prudentemente el lote de inferencia. El modelo "
        "1.7B aporta cobertura y 4B revisa daño, abstenciones, baja confianza y un control "
        "seguro. Esta política de enrutamiento es una decisión local que debe compararse con una pasada "
        "4B y con referencias humanas; las propuestas no son *ground truth* [@schroeder2025llmassisted].",
        [
            (
                "Configuración Colab y contrato JSON",
                "from moderacion_peru.providers import HuggingFaceProvider\n"
                "from moderacion_peru.device import high_memory_bf16_cuda,resolve_device\n"
                "from moderacion_peru.io import read_jsonl,write_json_atomic,write_jsonl_atomic\n\n"
                "SOURCE=COLAB_CONTEXT.input('chunks_v2') if COLAB_CONTEXT else ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "CAMPAIGN_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'datos/etiquetado/cascada_qwen_hf'\n"
                "CAMPAIGN_ROOT.mkdir(parents=True,exist_ok=True)\n"
                "PRIMARY=CAMPAIGN_ROOT/'qwen3_1_7b_primary_v3_2.jsonl'\n"
                "REVIEW=CAMPAIGN_ROOT/'qwen3_4b_review_v3_2.jsonl'\n"
                "QUEUE=CAMPAIGN_ROOT/'qwen3_4b_review_queue_v3_2.jsonl'\n"
                "RUN_PRIMARY=False\n"
                "RUN_REVIEW=False\n"
                "PRIMARY_LIMIT=20  # Smoke reanudable; use None para todos los pendientes.\n"
                "REVIEW_LIMIT=20   # Smoke reanudable; use None para toda la cola dirigida.\n"
                "REVIEW_CONFIDENCE_THRESHOLD=0.85\n"
                "SAFE_CONTROL_RATE=0.05\n"
                "MAX_NEEDS_REVIEW_FOR_QWEN4B=None\n"
                "QWEN_HARDWARE=resolve_device('auto')\n"
                "QWEN_A100_PROFILE=high_memory_bf16_cuda(QWEN_HARDWARE)\n"
                "PRIMARY_INFERENCE_BATCH_SIZE=8 if QWEN_A100_PROFILE else 4\n"
                "REVIEW_INFERENCE_BATCH_SIZE=4 if QWEN_A100_PROFILE else 2\n\n"
                "primary_provider=HuggingFaceProvider(model='Qwen/Qwen3-1.7B',revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',device='auto',records_per_request=5,inference_batch_size=PRIMARY_INFERENCE_BATCH_SIZE,max_new_tokens=256,operational_prompt_path=OPERATIONAL_PROMPT,label_source='qwen_hf_colab_primary')\n"
                "reviewer_provider=HuggingFaceProvider(model='Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c',device='auto',records_per_request=5,inference_batch_size=REVIEW_INFERENCE_BATCH_SIZE,max_new_tokens=256,operational_prompt_path=OPERATIONAL_PROMPT,label_source='qwen_hf_colab_review')\n"
                "primary_probe=primary_provider.probe(); reviewer_probe=reviewer_provider.probe()\n"
                "for role,probe in [('primario',primary_probe),('revisor',reviewer_probe)]:\n"
                "    if probe['response_format']!={'type':'json_object'} or probe['output_contract']['root_key']!='annotations':\n"
                "        raise RuntimeError(f'El proveedor {role} no garantiza el wrapper JSON annotations')\n"
                "    if Path(probe['operational_prompt_path']).resolve()!=OPERATIONAL_PROMPT.resolve():\n"
                "        raise RuntimeError(f'El proveedor {role} no usa el prompt operacional vigente')\n"
                "show_result('Qwen3-1.7B primario',primary_probe,tone='success')\n"
                "show_result('Qwen3-4B revisor',reviewer_probe,tone='success')\n"
                "show_callout('Perfil de aceleración',f'02_02 no llama a DeepSeek. Perfil A100={QWEN_A100_PROFILE}; lotes GPU: {PRIMARY_INFERENCE_BATCH_SIZE} prompts para 1.7B y {REVIEW_INFERENCE_BATCH_SIZE} para 4B. Los modelos se cargan secuencialmente.',tone='neutral')\n"
                "show_callout('Contrato verificado','Ambos modelos reciben el prompt operacional 3.2.0 y solo se persisten objetos validados con raíz annotations, orden y chunk_id exactos.',tone='success')",
            ),
            (
                "Primera pasada Qwen3-1.7B",
                "from tqdm.auto import tqdm\n"
                "from moderacion_peru.labeling import annotate_batched_incremental\n\n"
                "progress={'bar':None,'description':''}\n"
                "def qwen_progress(description):\n"
                "    progress['description']=description\n"
                "    def callback(event):\n"
                "        if event['status']=='started':\n"
                "            progress['bar']=tqdm(total=event['selected'],desc=description,unit='chunk'); return\n"
                "        bar=progress.get('bar')\n"
                "        if bar is not None and event.get('advance'):\n"
                "            bar.update(event['advance']); bar.set_postfix(ok=event['labeled'],errores=event['errors'])\n"
                "        if event['status'] in {'finished','interrupted_checkpoint'} and bar is not None:\n"
                "            bar.close(); progress['bar']=None\n"
                "    return callback\n\n"
                "if RUN_PRIMARY:\n"
                "    try:\n"
                "        primary_result=annotate_batched_incremental(read_jsonl(SOURCE),primary_provider,PRIMARY,error_path=PRIMARY.with_suffix('.errors.jsonl'),limit=PRIMARY_LIMIT,processing_batch_size=20,progress_callback=qwen_progress('Qwen3-1.7B'),run_metadata={'provider':primary_probe,'role':'qwen_primary','prompt_version':'3.2.0'})\n"
                "    finally:\n"
                "        primary_provider.unload()\n"
                "    show_result('Primera pasada persistida',primary_result,tone='success')\n"
                "else:\n"
                "    primary_provider.unload()\n"
                "    show_callout('Primera pasada desactivada','Use 20 para el smoke y después None; la salida reanuda por chunk_id.',tone='neutral')",
            ),
            (
                "Enrutamiento reproducible hacia Qwen3-4B",
                "from moderacion_peru.labeling_calibration import build_directed_review_queue\n\n"
                "if PRIMARY.is_file():\n"
                "    primary_rows=list(read_jsonl(PRIMARY))\n"
                "    primary_ids={row['chunk_id'] for row in primary_rows}\n"
                "    paired_chunks=[row for row in read_jsonl(SOURCE) if row['chunk_id'] in primary_ids]\n"
                "    review_queue,routing=build_directed_review_queue(paired_chunks,primary_rows,confidence_threshold=REVIEW_CONFIDENCE_THRESHOLD,safe_control_rate=SAFE_CONTROL_RATE,max_needs_review=MAX_NEEDS_REVIEW_FOR_QWEN4B,seed=42)\n"
                "    write_jsonl_atomic(QUEUE,review_queue)\n"
                "    write_json_atomic(CAMPAIGN_ROOT/'qwen_routing_summary_v3_2.json',routing)\n"
                "    show_result('Cola Qwen3-4B',routing,tone='success')\n"
                "else:\n"
                "    review_queue=[]\n"
                "    show_callout('Falta primera pasada','Ejecute Qwen3-1.7B antes de construir la cola 4B.',tone='warning')",
            ),
            (
                "Revisión dirigida Qwen3-4B y checkpoint",
                "if RUN_REVIEW:\n"
                "    if not QUEUE.is_file(): raise FileNotFoundError('Falta la cola dirigida Qwen3-4B')\n"
                "    try:\n"
                "        review_result=annotate_batched_incremental(read_jsonl(QUEUE),reviewer_provider,REVIEW,error_path=REVIEW.with_suffix('.errors.jsonl'),limit=REVIEW_LIMIT,processing_batch_size=10,progress_callback=qwen_progress('Qwen3-4B'),run_metadata={'provider':reviewer_probe,'role':'qwen_directed_review','prompt_version':'3.2.0'})\n"
                "    finally:\n"
                "        reviewer_provider.unload()\n"
                "    show_result('Revisión 4B persistida',review_result,tone='success')\n"
                "    if COLAB_CONTEXT is not None:\n"
                "        from moderacion_peru.colab import publish_colab_outputs\n"
                "        show_result('Checkpoint publicado en Drive',publish_colab_outputs(COLAB_CONTEXT),tone='success')\n"
                "else:\n"
                "    reviewer_provider.unload()\n"
                "    show_callout('Revisión 4B desactivada','La cola incluye daño, abstenciones, confianza menor que 0.85 y un control seguro reproducible del 5%.',tone='neutral')",
            ),
        ],
        colab_notebook_id="02_02",
        colab_requires_gpu=True,
    )
    create(
        "flujo/02_etiquetado/02_03_revision_llm_dirigida.ipynb",
        "02.03 · Auditoría de la revisión dirigida",
        "Recupera sin repetir API la calibración, el enrutamiento y las dos capas de etiquetas generadas en 02_01.",
        "La selección por incertidumbre pertenece a la familia de aprendizaje activo "
        "[@settles2009active], mientras que el balance puede aumentar la atención sobre clases raras "
        "[@fairstein2024balancing]. En lenguaje abusivo, el contexto conversacional puede cambiar la "
        "interpretación del fragmento [@bourgeade2024context]. Como una sugerencia LLM puede influir en "
        "la decisión humana [@choi2024llmeffect], el ordenamiento, el umbral 0.8 y la visualización inicial "
        "de la sugerencia se tratan como decisiones locales que deben auditarse.",
        [
            (
                "Resultados persistidos de la cascada",
                "import json\n"
                "from tqdm.auto import tqdm\n"
                "from moderacion_peru.io import read_jsonl\n"
                "CAMPAIGN_ROOT=ROOT/'datos/etiquetado/cascada_deepseek_v4'\n"
                "PRIMARY=CAMPAIGN_ROOT/'primary_flash_v3_2.jsonl'\n"
                "REVIEW=CAMPAIGN_ROOT/'review_pro_v3_2.jsonl'\n"
                "QUEUE=CAMPAIGN_ROOT/'directed_review_queue.jsonl'\n"
                "CALIBRATION=CAMPAIGN_ROOT/'calibration_flash_vs_pro.json'\n"
                "ROUTING=CAMPAIGN_ROOT/'routing_summary.json'\n"
                "counts={}\n"
                "for name,path in [('Flash',PRIMARY),('cola_Pro',QUEUE),('Pro',REVIEW)]:\n"
                "    counts[name]=sum(1 for _ in tqdm(read_jsonl(path),desc=f'Leyendo {name}',unit='fila')) if path.is_file() else 0\n"
                "show_summary('Cobertura persistida',counts,tone='success' if counts['Flash'] else 'warning')\n"
                "if CALIBRATION.is_file():\n"
                "    calibration=json.loads(CALIBRATION.read_text(encoding='utf-8-sig'))\n"
                "    show_table('Riesgo–cobertura Flash frente a Pro',calibration['comparisons'],max_rows=len(calibration['comparisons']))\n"
                "    show_summary('Regla calibrada',{'estado':calibration['threshold_status'],'umbral':calibration['selected_threshold'],'pares':calibration['paired_chunks'],'bootstrap_por_video':calibration['selected_threshold_cluster_bootstrap_95']},tone='success' if calibration['threshold_status']=='calibrated' else 'warning')\n"
                "if ROUTING.is_file(): show_result('Composición de la cola dirigida',json.loads(ROUTING.read_text(encoding='utf-8-sig')),tone='success')",
            ),
            (
                "Siguiente paso",
                "show_callout('Interpretación','Pro tiene precedencia sobre Flash solo en los chunks revisados. El desacuerdo o la confianza baja permanece visible y la decisión final corresponde a 02_04–02_05 con revisión humana.',tone='warning')",
            ),
        ],
    )
    create(
        "flujo/02_etiquetado/02_04_consolidacion_validacion_humana.ipynb",
        "02.04 · Consolidación y validación humana",
        "Consolida por precedencia y sirve una interfaz sin datos masivos incrustados.",
        "El acuerdo entre codificadores requiere definir unidad, categorías y medida, no solo contar "
        "coincidencias [@artstein2008agreement]. La intervención humana tampoco garantiza por sí sola "
        "calidad en tareas subjetivas asistidas por LLM [@schroeder2025llmassisted], y mostrar primero la "
        "sugerencia puede producir influencia o anclaje [@choi2024llmeffect]. Mostrarla desde el inicio, "
        "usar diálogos compactos y permitir lotes confirmados por video/canal son decisiones operativas; "
        "la precedencia humana, la adjudicación y el guardado *append-only* permanecen obligatorios.",
        [
            (
                "Consolidación",
                "from tqdm.auto import tqdm\n"
                "from moderacion_peru.consolidation import consolidate_annotations\n"
                "SOURCES=[p for p in [ROOT/'datos/etiquetado/cascada_deepseek_v4/primary_flash_v3_2.jsonl',ROOT/'datos/etiquetado/cascada_deepseek_v4/review_pro_v3_2.jsonl',ROOT/'datos/etiquetado/cascada_qwen_hf/qwen3_4b_review_v3_2.jsonl',ROOT/'datos/etiquetado/cascada_qwen_hf/qwen3_1_7b_primary_v3_2.jsonl'] if p.exists()]\n"
                "CHUNKS=ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "TRANSCRIPTS=ROOT/'datos/raw/transcripts_raw.jsonl'\n"
                "OUTPUT=ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\n"
                "consolidation_progress={'bar':None}\n"
                "CONSOLIDATION_PHASES={'loading_annotations':'Leyendo campañas','loading_chunks':'Cargando chunks','loading_transcripts':'Cargando transcripciones','consolidating':'Consolidando propuestas','checking_existing':'Verificando salida existente'}\n"
                "def report_consolidation_progress(event):\n"
                "    if event['status']=='phase_started':\n"
                "        if consolidation_progress.get('bar') is not None:\n"
                "            consolidation_progress['bar'].close()\n"
                "        consolidation_progress['bar']=tqdm(total=event.get('total'),desc=CONSOLIDATION_PHASES.get(event['phase'],event['phase']),unit='registro')\n"
                "        return\n"
                "    bar=consolidation_progress.get('bar')\n"
                "    if bar is not None and event.get('advance'):\n"
                "        bar.update(event['advance'])\n"
                "        if 'conflicts' in event:\n"
                "            bar.set_postfix(conflictos=event['conflicts'])\n"
                "    if event['status']=='finished' and bar is not None:\n"
                "        bar.close()\n"
                "        consolidation_progress['bar']=None\n\n"
                "if SOURCES:\n"
                "    try:\n"
                "        consolidation_result=consolidate_annotations(SOURCES,OUTPUT,chunks_source=CHUNKS,transcripts_source=TRANSCRIPTS,progress_callback=report_consolidation_progress)\n"
                "    finally:\n"
                "        if consolidation_progress.get('bar') is not None:\n"
                "            consolidation_progress['bar'].close()\n"
                "    show_result('Consolidación de campañas',consolidation_result,tone='success')\n"
                "else:\n"
                "    show_callout('Sin campañas','No hay propuestas para consolidar todavía.',tone='warning')",
            ),
            (
                f"Frontend\n\n{LABELING_FRONTEND_GUIDE}",
                "setup_command=(\n"
                "    f'Set-Location \"{ROOT}\"\\n'\n"
                "    'py -3.12 -m venv .venv\\n'\n"
                "    '.\\\\.venv\\\\Scripts\\\\python.exe -m pip install --upgrade pip\\n'\n"
                "    '.\\\\.venv\\\\Scripts\\\\python.exe -m pip install -e \".[datos,etiquetado,cuadernos,dev]\"'\n"
                ")\n"
                "start_command=(\n"
                "    f'Set-Location \"{ROOT}\"\\n'\n"
                "    f'.\\\\.venv\\\\Scripts\\\\modperu.exe serve-labeling `\\n  --campaign \"{OUTPUT}\"'\n"
                ")\n"
                "show_command('Preparación inicial (solo la primera vez)',setup_command,description='Ejecute este bloque en PowerShell si todavía no existe .venv.')\n"
                "show_command('Iniciar validación humana',start_command,description='Ejecute este bloque en PowerShell y mantenga la terminal abierta.')\n"
                "show_callout('Abrir o reiniciar el frontend','Visite http://127.0.0.1:8765. Si el servidor ya estaba abierto antes de una actualización, presione Ctrl+C y ejecute otra vez el bloque de inicio; recargar el navegador no actualiza el código Python del servidor.',tone='warning')",
            ),
        ],
    )
    create(
        "flujo/02_etiquetado/02_05_cierre_humano_snapshot.ipynb",
        "02.05 · Cierre humano y snapshot entrenable",
        "Reincorpora el último evento humano por chunk, recupera `video_id` desde el chunk fuente y congela un snapshot inmutable de cinco salidas.",
        "La separación agrupada por video evita que fragmentos correlacionados del mismo video crucen "
        "particiones, una decisión de diseño coherente con el control de sesgo de selección "
        "[@cawley2010selection]. Los snapshots, insumos y manifiestos usan SHA-256 [@nist2015sha]. "
        "La precedencia humana sigue siendo una regla local y no convierte automáticamente toda "
        "intervención en verdad objetiva [@schroeder2025llmassisted].",
        [
            (
                "Reconciliación append-only",
                "from tqdm.auto import tqdm\n"
                "from moderacion_peru.consolidation import reconcile_human_reviews\n"
                "CONSOLIDATED=ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\n"
                "CHUNKS=ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "REVIEWS=[ROOT/'datos/etiquetado/humano/labeling_events_v2.jsonl']\n"
                "REVIEWED=ROOT/'datos/etiquetado/consolidado/anotaciones_revisadas_v2.jsonl'\n"
                "stage_progress={'bar':None}\n"
                "STAGE_PHASES={'loading_chunks':'Cargando chunks','loading_review_events':'Leyendo eventos humanos','reconciling':'Reconciliando decisiones','loading_previous_snapshot':'Leyendo snapshot anterior','preparing_snapshot':'Preparando snapshot','deduplicating_snapshot':'Deduplicando snapshot','validating_video_splits':'Validando splits por video'}\n"
                "def report_stage_progress(event):\n"
                "    if event['status']=='phase_started':\n"
                "        if stage_progress.get('bar') is not None:\n"
                "            stage_progress['bar'].close()\n"
                "        stage_progress['bar']=tqdm(total=event.get('total'),desc=STAGE_PHASES.get(event['phase'],event['phase']),unit='registro')\n"
                "        return\n"
                "    bar=stage_progress.get('bar')\n"
                "    if bar is not None and event.get('advance'):\n"
                "        bar.update(event['advance'])\n"
                "        details={}\n"
                "        if 'reviewed' in event:\n"
                "            details['revisados']=event['reviewed']\n"
                "        if 'eligible' in event:\n"
                "            details['elegibles']=event['eligible']\n"
                "        if details:\n"
                "            bar.set_postfix(**details)\n"
                "    if event['status']=='finished' and bar is not None:\n"
                "        bar.close()\n"
                "        stage_progress['bar']=None\n"
                "try:\n"
                "    reconciliation_result=reconcile_human_reviews(CONSOLIDATED,REVIEWS,REVIEWED,chunks_source=CHUNKS,progress_callback=report_stage_progress)\n"
                "finally:\n"
                "    if stage_progress.get('bar') is not None:\n"
                "        stage_progress['bar'].close()\n"
                "show_result('Reconciliación humana',reconciliation_result,tone='success')",
            ),
            (
                "Snapshot versionado",
                "from moderacion_peru.datasets import materialize_versioned_training_snapshot\n"
                "DATASET=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n"
                "try:\n"
                "    snapshot=materialize_versioned_training_snapshot(REVIEWED,DATASET,progress_callback=report_stage_progress)\n"
                "finally:\n"
                "    if stage_progress.get('bar') is not None:\n"
                "        stage_progress['bar'].close()\n"
                "show_result('Snapshot entrenable', snapshot, tone='success')\n"
                "show_callout('Idempotencia', 'Sin cambios de entrada, ambas operaciones devuelven status=noop y no reescriben archivos.', tone='neutral')",
            ),
        ],
    )

    training_notebooks = [
        (
            "03_01_modelos_clasicos.ipynb",
            "Modelos clásicos",
            "from moderacion_peru.experiments import train_classical_experiments\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT=ROOT/'modelos/v2/clasicos'\nSAFE_TO_DAMAGE_RATIO=4.0  # política fija en train y validation\nPARALLEL_WORKERS=4  # 4/16 hilos: comparte la matriz dispersa sin saturar RAM\nLINEAR_SVM_MAX_ITER=20000  # 1.000 produjo ConvergenceWarning en 22 salidas × 3 pliegues\nRUN_TRAINING=False\nRUN_SVM_CONVERGENCE_REPAIR=False\nRUN_CHANNEL_ROBUSTNESS=False\nif RUN_TRAINING:\n    classical_result=run_with_progress('Suite clásica',train_classical_experiments,DATA,OUTPUT,variants=('base','policy_informed'),safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,parallel_workers=PARALLEL_WORKERS,linear_svm_max_iter=LINEAR_SVM_MAX_ITER,progress_unit='etapa')\n    show_result('Clásicos base e informados por política',classical_result,tone='success')\nif RUN_SVM_CONVERGENCE_REPAIR:\n    svm_result=run_with_progress('Reparación SVM',train_classical_experiments,DATA,OUTPUT/'svm_convergence_repair',model_names=('linear_svm',),variants=('base','policy_informed'),safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,parallel_workers=PARALLEL_WORKERS,linear_svm_max_iter=LINEAR_SVM_MAX_ITER,progress_unit='etapa')\n    show_result('SVM recalibrado con convergencia verificable',svm_result,tone='warning' if any(candidate['fit_quality']['converged'] is not True for candidate in svm_result['candidates']) else 'success')\nif RUN_CHANNEL_ROBUSTNESS:\n    robustness_result=run_with_progress('Robustez por canal',train_classical_experiments,DATA,OUTPUT/'channel_heldout',model_names=('logistic_regression',),variants=('base','policy_informed'),safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,split_scheme='channel',parallel_workers=PARALLEL_WORKERS,progress_unit='etapa')\n    show_result('Robustez con canales retenidos',robustness_result,tone='success')\nif not (RUN_TRAINING or RUN_SVM_CONVERGENCE_REPAIR or RUN_CHANNEL_ROBUSTNESS):\n    show_summary('Entrenamiento desactivado',{'salidas':'22 enmascaradas','SEGURO_train_validation':'4:1','TF-IDF':'una extracción por variante, reutilizada por cinco modelos','SVM':f'max_iter={LINEAR_SVM_MAX_ITER}; convergencia registrada y exigida en 03_07','paralelismo':f'{PARALLEL_WORKERS} hilos compartiendo matriz dispersa','progreso':'barra por preparación, TF-IDF, candidato y validation','test':'natural completo, sellado'},tone='neutral')",
            None,
            "La suite representa texto mediante TF–IDF [@salton1988tfidf] y compara regresión "
            "logística [@cox1958logistic], SVM lineal [@cortes1995svm], Complement Naive Bayes "
            "[@rennie2003cnb] y descenso de gradiente estocástico [@bottou2010sgd]. La implementación "
            "usa scikit-learn [@pedregosa2011sklearn] bajo una transformación uno-contra-resto para el "
            "problema multietiqueta [@tsoumakas2007multilabel]; la elección de n-gramas, pesos de clase e "
            "hiperparámetros es local.",
        ),
        (
            "03_02_transformers_planos.ipynb",
            "Transformers planos",
            MINILM_IMPROVEMENTS_SOURCE,
            "03_02",
            "La arquitectura Transformer procede de [@vaswani2017attention]. MiniLM se basa en "
            "destilación de autoatención [@wang2020minilm] y su extensión multilingüe en destilación "
            "entre lenguas [@reimers2020multilingual]; E5 multilingüe se documenta en "
            "[@wang2024e5]. Los checkpoints exactos son `paraphrase-multilingual-MiniLM-L12-v2` "
            "[@hf2026minilmcard] y `multilingual-e5-small` [@hf2026e5card], cargados mediante Transformers "
            "[@wolf2020transformers]. La continuación con baja tasa conserva el mejor checkpoint de "
            "validation y reinicia el optimizador; las tres semillas separan el muestreo fijo de "
            "`SEGURO` de la aleatoriedad del entrenamiento. La focal loss se mantiene como ablación "
            "frente a BCE ponderada, siguiendo el principio de concentrarse en ejemplos difíciles "
            "descrito por [@lin2017focal]. Estas elecciones y los contextos 192/256 son locales y deben "
            "decidirse solo con validation.",
        ),
        (
            "03_03_transformer_cascada.ipynb",
            "Transformer en cascada",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/cascada'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nSAFE_TO_DAMAGE_RATIO=4.0\nRUN_TRAINING=False\nif RUN_TRAINING:\n    cascade_result=run_with_progress('Transformer en cascada',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='cascade',device=DEVICE,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,progress_unit='etapa')\n    show_result('Cascada con auxiliares',cascade_result,tone='success')\nelse:\n    show_summary('Configuración preliminar',{'compuerta':'cualquier daño + 14 finas + 3 flags','rama_daño':4,'diagnóstico':'propagación de error; comparación plana en 03_07','progreso':'Trainer por lote/época + barra por etapa','SEGURO_train_validation':'4:1 fijo','test':'natural completo, sellado'},tone='neutral')",
            "03_03",
            "La clasificación jerárquica dispone de taxonomías y estrategias generales "
            "[@silla2011hierarchical], y existen modelos de texto sensibles a jerarquía como HiAGM "
            "[@zhou2020hiagm]. Este cuaderno no implementa HiAGM: la compuerta `SEGURO` frente a cualquier "
            "daño y las cuatro salidas multietiqueta —en el sentido general de [@tsoumakas2007multilabel]— "
            "constituyen una arquitectura local cuya ventaja debe demostrarse en validación.",
        ),
        (
            "03_03b_transformer_cascada_segura.ipynb",
            "Transformer en cascada v2 orientado a seguridad",
            "from moderacion_peru.experiments import train_neural_experiment\n"
            "from moderacion_peru.models import TRANSFORMER_SPECS\n"
            "DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\n"
            "OUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/cascada_v2'\n"
            "DEVICE='cuda' if COLAB_CONTEXT else 'auto'\n"
            "SAFE_TO_DAMAGE_RATIO=4.0\n"
            "GATE_MIN_DAMAGE_RECALL=0.99  # en validation: al menos 99 % de los daños pasan a la rama\n"
            "GATE_MIN_SAFE_NPV=0.99  # en validation: al menos 99 % de los bloqueados por la compuerta son SEGURO\n"
            "EPOCHS=TRANSFORMER_SPECS['e5'].epochs  # máximo 3; se restaura la mejor época de validation\n"
            "RUN_TRAINING=False\n"
            "if RUN_TRAINING:\n"
            "    cascade_v2_result=run_with_progress('Cascada v2 orientada a seguridad',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='cascade_v2',device=DEVICE,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,cascade_gate_min_damage_recall=GATE_MIN_DAMAGE_RECALL,cascade_gate_min_safe_npv=GATE_MIN_SAFE_NPV,progress_unit='etapa')\n"
            "    show_result('Cascada v2: compuerta y rama de cinco salidas',cascade_v2_result,tone='success')\n"
            "else:\n"
            "    show_summary('Configuración preliminar',{'compuerta':'ANY_DAMAGE + 14 finas + 3 flags','restricción_recall_daño':GATE_MIN_DAMAGE_RECALL,'restricción_NPV_seguro':GATE_MIN_SAFE_NPV,'fallback':'si no hay umbral no trivial factible, deriva todo a la segunda rama','rama_especializada':'SEGURO + cuatro daños; penaliza conflicto SEGURO+daño','épocas_máximas':EPOCHS,'selección':'mejor macro-AUPRC de daño en validation','advertencia':'las restricciones son evidencia empírica de validation, no garantía poblacional','test':'natural completo, sellado'},tone='warning')",
            "03_03b",
            "La propuesta es una clasificación jerárquica local, relacionada con los marcos generales "
            "de jerarquías [@silla2011hierarchical] y clasificación multietiqueta "
            "[@tsoumakas2007multilabel]. La compuerta se selecciona únicamente en `validation` bajo "
            "restricciones empíricas de sensibilidad al daño y valor predictivo de la decisión "
            "`SEGURO`; esta precaución es pertinente porque las redes modernas pueden estar "
            "descalibradas [@guo2017calibration]. La segunda rama vuelve a incluir `SEGURO` más los "
            "cuatro daños para recuperar falsos positivos de la compuerta. El umbral, las metas de "
            "0,99 y la penalización de conflicto son decisiones del proyecto, no garantías "
            "poblacionales.",
        ),
        (
            "03_04_transformer_multitarea.ipynb",
            "Transformer jerárquico multitarea",
            "from moderacion_peru.experiments import train_neural_experiment\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/multitarea'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nSAFE_TO_DAMAGE_RATIO=4.0  # política fija en train y validation\nRUN_TRAINING=False\nif RUN_TRAINING:\n    multitask_result=run_with_progress('Transformer multitarea',train_neural_experiment,DATA,OUTPUT_ROOT,experiment='multitask',device=DEVICE,safe_to_damage_ratio=SAFE_TO_DAMAGE_RATIO,progress_unit='etapa')\n    show_result('Multitarea enmascarada',multitask_result,tone='success')\nelse:\n    show_summary('Configuración preliminar',{'salidas':'5 gruesas + 14 finas + 3 flags','máscaras':'ausente no equivale a negativo','progreso':'Trainer por lote/época + barra por etapa','SEGURO_train_validation':'4:1 fijo','test':'natural completo, sellado'},tone='neutral')",
            "03_04",
            "El aprendizaje multitarea comparte representaciones entre objetivos relacionados "
            "[@caruana1997multitask], mientras la clasificación jerárquica [@silla2011hierarchical] y "
            "multietiqueta [@tsoumakas2007multilabel] aportan marcos generales para salidas dependientes o "
            "simultáneas. Las cinco salidas principales, las auxiliares finas, los flags y la ponderación "
            "de sus pérdidas son un diseño local y no heredan garantías de transferencia positiva.",
        ),
        (
            "03_05_qwen_lora.ipynb",
            "Qwen-LoRA",
            QWEN_LORA_BASE_SOURCE,
            "03_05",
            "LoRA introduce actualizaciones entrenables de bajo rango sobre un modelo preentrenado "
            "[@hu2022lora]. El backbone pertenece a la familia Qwen3 [@qwen2025qwen3] y se fija en el "
            "checkpoint `Qwen/Qwen3-0.6B-Base` [@hf2026qwen06bcard]; la inyección de adaptadores usa PEFT "
            "0.18.0 [@hf2026peft018]. El rango 8, los módulos objetivo, la cabeza de cinco salidas y la "
            "recalibración son decisiones locales. Además del candidato base de 128 tokens, el "
            "cuaderno puede continuar sus pesos LoRA con 256 tokens y un optimizador nuevo. Esa "
            "continuación conserva el mismo snapshot y mantiene test sellado; produce un candidato "
            "independiente cuya posible ventaja debe decidirse exclusivamente en validation.",
        ),
        (
            "03_06_qwen_estructurado.ipynb",
            "Qwen estructurado",
            QWEN_STRUCTURED_IMPROVEMENTS_SOURCE,
            "03_06",
            "El backbone se documenta mediante el informe Qwen3 [@qwen2025qwen3] y la tarjeta exacta de "
            "`Qwen/Qwen3-0.6B-Base` [@hf2026qwen06bcard]. La separación entre compuerta y daños toma "
            "como antecedentes la clasificación jerárquica [@silla2011hierarchical] y multietiqueta "
            "[@tsoumakas2007multilabel]. La mejora reutiliza como inicialización entrenable el "
            "adaptador LoRA verificable de 03_05 [@hu2022lora], reinicia optimizador y compara una sola "
            "época con penalizaciones 0, 0,02 y 0,05. La penalización de conflicto y su regla de "
            "selección son locales y se deciden exclusivamente en validation.",
        ),
        (
            "03_06b_qwen_prompt_sft.ipynb",
            "Qwen SFT condicionado por prompt",
            """from moderacion_peru.prompt_sft import train_prompt_conditioned_sft
DATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'
PROMPT=ROOT/'config/prompt_operacional_ollama_v3_2.md'
OUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_prompt_sft'
DEVICE='cuda' if COLAB_CONTEXT else 'auto'
PERSISTENT_CHECKPOINT_ROOT=COLAB_CONTEXT.drive_run_dir/'trainer_checkpoints' if COLAB_CONTEXT else None

# Perfil recomendado: candidato comparable bajo un presupuesto aproximado de una hora A100.
BUDGET_TRAIN_ROWS=3000
BUDGET_EPOCHS=1
BUDGET_MAX_LENGTH=2048
BUDGET_TRAINING_SECONDS=1500  # 25 min; deja el resto a validation generativa completa
BUDGET_TOTAL_SECONDS=4500  # corte duro total de 75 min; si vence, no crea candidate.json
ESTIMATED_A100_HOURS=(0.75,1.25)
REFERENCE_A100_CU_PER_HOUR=5.4  # solo referencia observada; use la tasa que muestre Colab
ESTIMATED_COMPUTE_UNITS=tuple(round(hours*REFERENCE_A100_CU_PER_HOUR,1) for hours in ESTIMATED_A100_HOURS)

RUN_BUDGETED_COMPARABLE=False
RUN_DIAGNOSTIC_PILOT=False
RUN_FULL_TRAINING=False
show_summary('Preflight de costo y comparabilidad',{
    'perfil_recomendado':'budgeted_comparable: 3.000 train, 1 época y validation común completa',
    'tiempo_A100_estimado':f'{ESTIMATED_A100_HOURS[0]*60:.0f}–{ESTIMATED_A100_HOURS[1]*60:.0f} min; objetivo ≈60 min',
    'corte_entrenamiento':f'{BUDGET_TRAINING_SECONDS/60:.0f} min',
    'corte_total':f'{BUDGET_TOTAL_SECONDS/60:.0f} min; sin candidate.json si validation no termina',
    'unidades_estimadas_a_5.4_CU_h':f'{ESTIMATED_COMPUTE_UNITS[0]}–{ESTIMATED_COMPUTE_UNITS[1]} CU',
    'tarifa_real':'Colab es dinámico; multiplique las GPU-horas por la tasa CU/h visible en la sesión',
    'elegibilidad_03_07':'sí, con disclaimer de presupuesto limitado y validation completa',
    'test':'sellado; no se abre en esta corrida',
},tone='warning')
if RUN_BUDGETED_COMPARABLE:
    budgeted_result=run_with_progress('Qwen SFT budgeted comparable',train_prompt_conditioned_sft,DATA,PROMPT,OUTPUT_ROOT/'budgeted_comparable',device=DEVICE,safe_to_damage_ratio=4.0,training_regime='budgeted_comparable',eligible_for_03_07=True,train_limit=BUDGET_TRAIN_ROWS,validation_limit=None,epochs=BUDGET_EPOCHS,max_length=BUDGET_MAX_LENGTH,max_training_seconds=BUDGET_TRAINING_SECONDS,max_total_seconds=BUDGET_TOTAL_SECONDS,generation_max_new_tokens=160,prompt_capsule_max_chars=4800,run_label='a100_about_one_hour_v2_memory_safe',persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='fila')
    show_result('Candidato SFT budgeted elegible para 03_07',budgeted_result,tone='warning')
if RUN_DIAGNOSTIC_PILOT:
    pilot_result=run_with_progress('Piloto diagnóstico Qwen SFT',train_prompt_conditioned_sft,DATA,PROMPT,OUTPUT_ROOT/'diagnostic_pilot',device=DEVICE,safe_to_damage_ratio=4.0,training_regime='diagnostic_pilot',eligible_for_03_07=False,train_limit=500,validation_limit=200,epochs=1,max_length=2560,generation_max_new_tokens=192,run_label='diagnostic_only',persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='fila')
    show_result('Piloto SFT no elegible para 03_07',pilot_result,tone='warning')
if RUN_FULL_TRAINING:
    full_result=run_with_progress('Qwen SFT completo',train_prompt_conditioned_sft,DATA,PROMPT,OUTPUT_ROOT/'full',device=DEVICE,safe_to_damage_ratio=4.0,training_regime='full',eligible_for_03_07=True,validation_limit=None,persistent_checkpoint_root=PERSISTENT_CHECKPOINT_ROOT,progress_unit='fila')
    show_result('SFT generativo completo condicionado por prompt v3.2',full_result,tone='success')
if not (RUN_BUDGETED_COMPARABLE or RUN_DIAGNOSTIC_PILOT or RUN_FULL_TRAINING):
    show_callout('Listo pero desactivado','Para la corrida solicitada active únicamente RUN_BUDGETED_COMPARABLE=True. No active piloto y full a la vez.',tone='success')""",
            "03_06b",
            "La rama usa el modelo conversacional oficial `Qwen/Qwen3-0.6B` "
            "[@hf2026qwen06binstructcard] y LoRA [@hu2022lora]. A diferencia de los clasificadores "
            "03_05 y 03_06, recibe una cápsula reproducible del prompt operacional v3.2 y genera JSON. "
            "La compilación de la cápsula, la pérdida solo sobre la respuesta y el contrato JSON son "
            "decisiones locales; sus resultados se comparan por separado para no confundir prompting "
            "con supervisión por etiquetas.",
        ),
        (
            "03_07_comparacion_final.ipynb",
            "Comparación final",
            COMPARISON_RUN_SOURCE,
            "03_07",
            "Balanced accuracy pondera por igual el reconocimiento de daño y de seguro bajo desbalance "
            "[@brodersen2010balanced]. Las curvas precisión–recall siguen siendo salvaguardas informativas "
            "[@saito2015pr], pero no se suman a BA con pesos arbitrarios: el aprendizaje multiobjetivo "
            "recomienda explicitar preferencias y soluciones Pareto [@jin2008pareto]. La calibración se "
            "audita [@guo2017calibration], la revisión se informa con riesgo–cobertura "
            "[@geifman2017selective], y test permanece fuera de toda selección para evitar sesgo "
            "[@cawley2010selection]. La capacidad humana, el margen de no inferioridad y los costos son "
            "decisiones locales que deben predeclararse.",
        ),
        (
            "03_08_auditoria_finas_flags.ipynb",
            "Auditoría fina y transversal",
            "from moderacion_peru.datasets import audit_auxiliary_candidate_metrics,audit_training_snapshot\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT=ROOT/'resultados/auditorias/auditoria_finas_flags_v2.json'\nPREDICTIVE=ROOT/'resultados/auditorias/calidad_predictiva_auxiliar_observada.json'\naudit_result=run_with_progress('Auditoría del snapshot',audit_training_snapshot,DATA,OUTPUT,progress_unit='etapa')\nshow_result('Auditoría de máscaras, cobertura y consistencia',audit_result,tone='success')\npredictive_result=run_with_progress('Auditoría de candidatos',audit_auxiliary_candidate_metrics,[ROOT/'modelos/v2'],PREDICTIVE,progress_unit='etapa')\nshow_result('Calidad auxiliar disponible',predictive_result,tone='success')",
            None,
            "La auditoría parte de tipologías generales de contenido dañino [@banko2020taxonomy] y de "
            "abuso dirigido o generalizado [@waseem2017abuse], pero separa fenómenos implícitos "
            "[@elsherief2021implicit], ironía [@ilic2018irony] y dependencia de contexto "
            "[@bourgeade2024context]. Para el Perú, `RACISMO_DISCRIMINACION` considera racialización "
            "lingüística y política documentada localmente [@almeida2022motoso]; "
            "`ATAQUE_POR_GENERO_IDENTIDAD` se apoya en estudios e informes peruanos sobre violencia de "
            "género en línea [@albornoz2018conocer], afectaciones a mujeres y personas LGBTI "
            "[@defensoria2021violenciaenlinea] y léxico lesbofóbico [@lovon2022lesbofobia]. La frontera de "
            "contenido sexual explícito también responde a política de plataforma "
            "[@youtube2026sexualpolicy]. Las fusiones, nombres y flags exactos siguen siendo locales.",
        ),
    ]
    for filename, subtitle, source, colab_id, academic_context in training_notebooks:
        execution_heading = (
            "Configuración y candidato base de 128 tokens"
            if filename == "03_05_qwen_lora.ipynb"
            else "Configuración y ejecución"
        )
        code_cells = [("Restauración reproducible del dataset", DATASET_CHECKPOINT)]
        if filename == "03_07_comparacion_final.ipynb":
            code_cells.append(
                (
                    "Restauración verificable de candidatos desde Drive",
                    COMPARISON_DRIVE_RESTORE_SOURCE,
                )
            )
        code_cells.append((execution_heading, source))
        if filename == "03_05_qwen_lora.ipynb":
            code_cells.append(
                (
                    "Continuación opcional a 256 tokens\n\n"
                    "Este bloque no modifica el candidato base. Verifica el manifiesto y el "
                    "snapshot del modelo de 128 tokens, carga su adaptador como entrenable y "
                    "crea una corrida distinta. Reinicia optimizador y scheduler para no mezclar "
                    "un estado construido con secuencias de otra longitud. `03_07` descubrirá "
                    "ambos candidatos y los comparará sobre las mismas filas de validation.",
                    QWEN_LORA_256_SOURCE,
                )
            )
        create(
            f"flujo/03_entrenamiento/{filename}",
            f"03 · {subtitle}",
            "Entrena o audita el contrato de etiquetas v2.1 sin consultar test para seleccionar modelos, épocas o umbrales.",
            academic_context,
            code_cells,
            colab_notebook_id=colab_id,
            colab_requires_gpu=filename != "03_07_comparacion_final.ipynb",
        )
    create(
        "flujo/04_produccion/04_01_frontend_produccion.ipynb",
        "04.01 · Frontend de producción supervisada",
        "Comprueba el registro del contrato de etiquetas v2.1 e inicia el demostrador local en modo sombra con texto, subtítulos de YouTube, cinco scores, revisión humana y estadísticas; nunca carga modelos históricos como sustitutos.",
        "La moderación algorítmica presenta retos técnicos y de gobernanza que impiden interpretar un "
        "score como decisión autosuficiente [@gorwa2020moderation]. Los sistemas semiautomáticos pueden "
        "integrar revisión humana [@andersen2021rem], y la abstención tiene antecedentes tanto en la "
        "opción de rechazo [@chow1970reject] como en clasificación selectiva [@geifman2017selective] y "
        "deferencia a una persona experta [@mozannar2020defer]. El modo sombra, los umbrales y los motivos "
        "de revisión son decisiones locales y no constituyen una garantía de seguridad.",
        [
            (
                "Disponibilidad",
                "from moderacion_peru.artifacts import artifact_status\nshow_result('Disponibilidad de producción', artifact_status(ROOT), tone='neutral')",
            ),
            (
                "Inicio",
                "show_command('Iniciar frontend de producción', 'modperu serve-production --host 127.0.0.1 --port 8765', description='Ejecute este comando en una terminal del entorno virtual.')\nshow_callout('Modo de operación', 'La interfaz reutiliza caché de subtítulos, no descarga audio/video, registra inferencias y permite revisión append-only.', tone='info')",
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="RUTA",
        help="Regenera únicamente las rutas de cuaderno indicadas, relativas al repositorio.",
    )
    arguments = parser.parse_args()
    main(only_notebooks=set(arguments.only) if arguments.only else None)
