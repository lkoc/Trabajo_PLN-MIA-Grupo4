"""Genera los cuadernos orquestadores activos sin resultados obsoletos incrustados."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


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
print('Proyecto:', ROOT)
"""


COLAB_SETUP = """# Backend reproducible: local o Google Colab desde VS Code
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile

COLAB_NOTEBOOK_ID = "__NOTEBOOK_ID__"
COLAB_DRIVE_FOLDER = "ModeracionPeru_Colab"  # Debe coincidir con config/colab_l4.json
COLAB_RUN_ID = ""  # Vacío reanuda <notebook>_working_v2_1; use otro ID para otro experimento
COLAB_REQUIRE_L4 = True
IN_COLAB = importlib.util.find_spec("google.colab") is not None
COLAB_CONTEXT = None

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

if IN_COLAB:
    from google.colab import drive

    # La extensión oficial de Colab para VS Code admite drive.mount desde v0.2.1.
    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = Path("/content/drive/MyDrive") / COLAB_DRIVE_FOLDER
    BUNDLE_DIR = DRIVE_ROOT / "bundle"
    manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Falta bundle_manifest.json. Prepare y suba resultados/colab_bundle a " + str(BUNDLE_DIR)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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

    os.environ["MODPERU_ROOT"] = str(ROOT)
    os.environ["HF_HOME"] = "/content/huggingface"
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
    print(colab_runtime_diagnostics())
    print(COLAB_CONTEXT.as_dict())
else:
    ROOT = _find_local_root()
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    print("Backend local:", ROOT)
"""


def colab_setup(notebook_id: str) -> str:
    return COLAB_SETUP.replace("__NOTEBOOK_ID__", notebook_id)


def create(
    path: str,
    title: str,
    purpose: str,
    code_cells: list[tuple[str, str]],
    *,
    colab_notebook_id: str | None = None,
) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "moderacion_peru": {
            "contract": "moderacion_peru_5_salidas_v2",
            "taxonomy_version": "2.1.0",
            "orchestrator": True,
            "colab": {
                "eligible": colab_notebook_id is not None,
                "notebook_id": colab_notebook_id,
                "transport": "google_drive_only" if colab_notebook_id else None,
                "expected_gpu": "NVIDIA L4" if colab_notebook_id else None,
            },
        },
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{purpose}\n\n"
            "**Contrato v2.1:** `SEGURO` + cuatro daños entrenados, incluida "
            "`ATAQUE_POR_GENERO_IDENTIDAD`. `SEGURO` es excluyente; los daños son multietiqueta. "
            "Los casos indeterminados se difieren y no entran al entrenamiento."
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibilidad\n\nEl cuaderno solo orquesta funciones versionadas de `src/moderacion_peru`. "
            "En local no instala paquetes. En Colab, únicamente la celda de bootstrap instala versiones "
            "fijadas desde el bundle SHA-256 de Drive. No usa rutas personales. Revise el README de esta etapa."
        ),
    ]
    if colab_notebook_id:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Backend opcional Google Colab L4 desde VS Code\n\n"
                "Instale la extensión oficial **Google Colab** (`google.colab`), seleccione "
                "`Select Kernel > Colab` y asigne una **NVIDIA L4**. El notebook permanece local; "
                "Drive transporta solo el bundle mínimo verificado. Edite `COLAB_RUN_ID` para separar "
                "experimentos. No sincronice cachés de modelos ni escriba checkpoints directamente en Drive."
            )
        )
        notebook.cells.append(nbf.v4.new_code_cell(colab_setup(colab_notebook_id)))
    else:
        notebook.cells.append(nbf.v4.new_code_cell(SETUP))
    for heading, source in code_cells:
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        notebook.cells.append(nbf.v4.new_code_cell(source))
    if colab_notebook_id:
        notebook.cells.append(
            nbf.v4.new_markdown_cell(
                "## Publicación o checkpoint en Drive\n\n"
                "Los archivos se generan en el SSD efímero de `/content`. Active esta celda después de "
                "un checkpoint coherente o al finalizar; publica un solo TAR.GZ y luego su manifiesto."
            )
        )
        notebook.cells.append(
            nbf.v4.new_code_cell(
                "PUBLISH_TO_DRIVE = False\n"
                "if COLAB_CONTEXT is not None and PUBLISH_TO_DRIVE:\n"
                "    from moderacion_peru.colab import publish_colab_outputs\n"
                "    print(publish_colab_outputs(COLAB_CONTEXT))\n"
                "elif COLAB_CONTEXT is not None:\n"
                "    print('Publicación desactivada; cambie PUBLISH_TO_DRIVE=True tras guardar un checkpoint consistente.')\n"
                "else:\n"
                "    print('Backend local: los artefactos ya permanecen en el workspace.')"
            )
        )
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)


def main() -> None:
    create(
        "flujo/01_datos/01_01_scraping_incremental.ipynb",
        "01.01 · Adquisición incremental de subtítulos",
        "Reutiliza transcripciones canónicas y cachés por `video_id`; solo consulta YouTube para candidatos nuevos y nunca descarga audio o video.",
        [
            ("Preflight", "from moderacion_peru.artifacts import artifact_status\nartifact_status(ROOT)"),
            ("Reutilización de snapshots existentes", "from moderacion_peru.acquisition import (bootstrap_canonical_from_existing, discover_existing_transcript_sources, fetch_youtube_subtitles, ingest_incremental, load_candidates)\nCANONICAL = ROOT/'datos/raw/transcripts_raw.jsonl'\nCACHE = ROOT/'datos/raw/transcripts_cache'\nsources = discover_existing_transcript_sources(ROOT, canonical_path=CANONICAL)\nreuse_stats = bootstrap_canonical_from_existing(sources, CANONICAL)\nprint(reuse_stats)"),
            ("Candidatos y caché", "CANDIDATE_FILES = [ROOT/'datos/raw/video_candidates.jsonl', ROOT/'datos/raw/videos_candidatos.csv']\ncandidates_by_id = {}\nfor source in CANDIDATE_FILES:\n    for row in load_candidates(source):\n        candidates_by_id.setdefault(str(row['video_id']), row)\ncandidates = list(candidates_by_id.values())\nprint('Candidatos:', len(candidates), '· los ya existentes se omitirán')"),
            ("Ejecución controlada", "FETCH_NEW = False  # Cambie a True solo para consultar candidatos no vistos\nif candidates:\n    stats = ingest_incremental(candidates, CANONICAL, CACHE, fetcher=fetch_youtube_subtitles if FETCH_NEW else None)\n    print(stats)\nelse:\n    print('Agregue candidatos con video_id y url; el corpus existente no se vuelve a descargar.')"),
        ],
    )
    create(
        "flujo/01_datos/01_02_limpieza_troceado_incremental.ipynb",
        "01.02 · Limpieza y troceado incremental",
        "Crea chunks deterministas únicamente para transcripciones nuevas o modificadas y conserva la versión del troceador.",
        [
            ("Configuración", "from moderacion_peru.incremental import chunk_records_incrementally\nfrom moderacion_peru.io import append_jsonl_once, read_jsonl\nSOURCE=ROOT/'datos/raw/transcripts_raw.jsonl'\nOUTPUT=ROOT/'datos/processed/chunks_v2.jsonl'\nVERSION_INDEX=ROOT/'datos/processed/chunking_v2_versions.jsonl'"),
            ("Materialización", "existing=list(read_jsonl(OUTPUT)) if OUTPUT.exists() else []\nversions=list(read_jsonl(VERSION_INDEX)) if VERSION_INDEX.exists() else []\nnew_rows,new_versions,stats=chunk_records_incrementally(read_jsonl(SOURCE) if SOURCE.exists() else [],existing,versions)\nadded,skipped=append_jsonl_once(OUTPUT,new_rows,id_field='chunk_id')\nversions_added,_=append_jsonl_once(VERSION_INDEX,new_versions,id_field='version_id')\nstats.update({'added':added,'duplicate_ids':skipped,'versions_registered':versions_added})\nprint(stats)"),
        ],
    )
    create(
        "flujo/01_datos/01_03_ampliacion_dirigida.ipynb",
        "01.03 · Ampliación dirigida",
        "Añade candidatos para daños minoritarios sin mezclar adquisición, etiquetado y entrenamiento en una sola celda.",
        [
            ("Estado incremental", "from moderacion_peru.acquisition import processed_video_ids\nCANONICAL=ROOT/'datos/raw/transcripts_raw.jsonl'\nprocessed=processed_video_ids(CANONICAL)\nprint('Videos ya aprovechados:',len(processed))"),
            ("Handoff", "print('Guarde nuevos candidatos en datos/raw/video_candidates.jsonl y vuelva a 01_01. Los IDs existentes serán omitidos.')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_01_etiquetado_local_ollama.ipynb",
        "02.01 · Etiquetado Ollama local o Hugging Face en Colab",
        "Usa Ollama HTTP en local o, opcionalmente, Hugging Face sobre Colab L4; ambos conservan el mismo contrato y reanudan por `chunk_id`.",
        [
            (
                "Selección del proveedor",
                "if COLAB_CONTEXT is not None:\n"
                "    from moderacion_peru.providers import HuggingFaceProvider\n"
                "    provider=HuggingFaceProvider(model='Qwen/Qwen3-4B',device='cuda',max_new_tokens=512)\n"
                "    SOURCE=COLAB_CONTEXT.input('chunks_v2')\n"
                "    OUTPUT=COLAB_CONTEXT.scratch_output_dir/'huggingface_qwen3_4b_v2.jsonl'\n"
                "    ERRORS=COLAB_CONTEXT.scratch_output_dir/'huggingface_qwen3_4b_v2.errors.jsonl'\n"
                "else:\n"
                "    from moderacion_peru.providers import OllamaProvider\n"
                "    provider=OllamaProvider(model='qwen3.5:4b')\n"
                "    SOURCE=ROOT/'datos/processed/chunks_v2.jsonl'\n"
                "    OUTPUT=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl'\n"
                "    ERRORS=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.errors.jsonl'\n"
                "print(provider.probe())\nprint({'source':str(SOURCE),'output':str(OUTPUT)})",
            ),
            (
                "Etiquetado incremental",
                "from moderacion_peru.io import read_jsonl\n"
                "from moderacion_peru.labeling import annotate_incremental\n"
                "RUN=False\nLIMIT=20  # Quite el límite solo después del smoke test\n"
                "if RUN:\n"
                "    print(annotate_incremental(read_jsonl(SOURCE),provider,OUTPUT,error_path=ERRORS,limit=LIMIT))\n"
                "else:\n"
                "    print('Preflight completo; cambie RUN=True. La salida reanuda por chunk_id.')",
            ),
        ],
        colab_notebook_id="02_01",
    )
    create(
        "flujo/02_etiquetado/02_02_etiquetado_remoto.ipynb",
        "02.02 · Etiquetado remoto opcional",
        "Mantiene un proveedor remoto compatible, pero nunca consume crédito durante preflight ni sin activación explícita.",
        [
            ("Preflight sin red", "from moderacion_peru.providers import DeepSeekProvider\nprovider=DeepSeekProvider()\nprovider.probe()"),
            ("Ejecución explícita", "from moderacion_peru.io import read_jsonl\nfrom moderacion_peru.labeling import annotate_incremental\nSOURCE=ROOT/'datos/processed/chunks_v2.jsonl'\nOUTPUT=ROOT/'datos/etiquetado/remoto/deepseek_v2.jsonl'\nRUN_REMOTE=False\nif RUN_REMOTE:\n    print(annotate_incremental(read_jsonl(SOURCE),provider,OUTPUT))\nelse:\n    print('No se realizó ninguna llamada comercial.')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_03_revision_llm_dirigida.ipynb",
        "02.03 · Revisión LLM dirigida",
        "Prioriza desacuerdos, baja confianza, contexto y clases minoritarias sin tratarlos como verdad humana.",
        [
            ("Selección reproducible", "from moderacion_peru.io import read_jsonl\nSOURCE=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl'\nrows=list(read_jsonl(SOURCE)) if SOURCE.exists() else []\nreview=[r for r in rows if r.get('needs_review') or r.get('score_confianza',1)<0.8]\nreview.sort(key=lambda r:r['chunk_id'])\nprint({'labeled':len(rows),'directed_review':len(review)})"),
            ("Siguiente paso", "print('La revisión puede usar otro modelo/familia o pasar directamente a 02_04 para validación humana.')"),
        ],
    )
    create(
        "flujo/02_etiquetado/02_04_consolidacion_validacion_humana.ipynb",
        "02.04 · Consolidación y validación humana",
        "Consolida por precedencia y sirve una interfaz sin datos masivos incrustados.",
        [
            ("Consolidación", "from moderacion_peru.consolidation import consolidate_annotations\nSOURCES=[p for p in [ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl',ROOT/'datos/etiquetado/remoto/deepseek_v2.jsonl'] if p.exists()]\nOUTPUT=ROOT/'datos/etiquetado/consolidado/anotaciones_v2.jsonl'\nprint(consolidate_annotations(SOURCES,OUTPUT) if SOURCES else 'No hay campañas para consolidar')"),
            ("Frontend", "print(f'modperu serve-labeling --campaign {OUTPUT}')"),
        ],
    )

    training_notebooks = [
        ("03_01_modelos_clasicos.ipynb", "Modelos clásicos", "from moderacion_peru.datasets import load_split\nfrom moderacion_peru.models import train_classical_suite\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nRUN=False\nif RUN:\n    print(train_classical_suite(load_split(DATA,'train'),ROOT/'modelos/v2/clasicos'))\nelse: print('Listo para Dummy, NB, logística, SVM y SGD incremental. La L4 no acelera TF-IDF/sklearn.')", None),
        ("03_02_transformers_planos.ipynb", "Transformers planos", "from moderacion_peru.models import TRANSFORMER_SPECS, build_transformer_classifier\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/transformers_planos'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN=False\nif RUN:\n    tokenizer,model,hardware=build_transformer_classifier(TRANSFORMER_SPECS['e5'],DEVICE); print(hardware)\nelse: print({'data':str(DATA),'output':str(OUTPUT_ROOT),'specs':[TRANSFORMER_SPECS['minilm'],TRANSFORMER_SPECS['e5']]})", "03_02"),
        ("03_03_transformer_cascada.ipynb", "Transformer en cascada", "from moderacion_peru.taxonomy import load_taxonomy\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/cascada'\nt=load_taxonomy(); print({'data':str(DATA),'output':str(OUTPUT_ROOT),'gate':'SEGURO frente a cualquier daño','children':t.damage_labels,'conflicts':'review'})", "03_03"),
        ("03_04_transformer_multitarea.ipynb", "Transformer jerárquico multitarea", "from moderacion_peru.taxonomy import load_taxonomy\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/multitarea'\nt=load_taxonomy(); print({'data':str(DATA),'output':str(OUTPUT_ROOT),'primary_outputs':t.target_labels,'auxiliary_fine':len(t.fine_labels),'flags':t.flags})", "03_04"),
        ("03_05_qwen_lora.ipynb", "Qwen-LoRA", "from moderacion_peru.models import build_qwen_lora_classifier\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_lora'\nDEVICE='cuda' if COLAB_CONTEXT else 'auto'\nRUN=False\nif RUN:\n    tokenizer,model,hardware=build_qwen_lora_classifier(DEVICE); print(hardware); model.print_trainable_parameters()\nelse: print({'data':str(DATA),'output':str(OUTPUT_ROOT),'resume':bool(COLAB_CONTEXT and COLAB_CONTEXT.resumed),'note':'Cabeza nueva de cinco salidas; recalibrar umbrales.'})", "03_05"),
        ("03_06_qwen_estructurado.ipynb", "Qwen estructurado", "from moderacion_peru.taxonomy import load_taxonomy\nDATA=COLAB_CONTEXT.input('dataset_5_salidas') if COLAB_CONTEXT else ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nOUTPUT_ROOT=COLAB_CONTEXT.scratch_output_dir if COLAB_CONTEXT else ROOT/'modelos/v2/qwen_estructurado'\nt=load_taxonomy(); print({'data':str(DATA),'output':str(OUTPUT_ROOT),'flat_reference':'03_05','gate_output':'SEGURO','damage_outputs':t.damage_labels,'selection':'validation only'})", "03_06"),
        ("03_07_comparacion_final.ipynb", "Comparación final", "from moderacion_peru.artifacts import artifact_status\nprint('Comparar localmente artefactos ya publicados; la L4 solo se necesita si faltan predicciones.'); artifact_status(ROOT)", None),
        ("03_08_auditoria_finas_flags.ipynb", "Auditoría fina y transversal", "from moderacion_peru.taxonomy import load_taxonomy\nt=load_taxonomy(); print({'fine_labels':t.fine_labels,'flags':t.flags,'gold_inputs':False,'gpu':'no requerida para métricas ya materializadas'})", None),
    ]
    for filename, subtitle, source, colab_id in training_notebooks:
        create(
            f"flujo/03_entrenamiento/{filename}",
            f"03 · {subtitle}",
            "Entrena o audita el contrato v2.1 sin consultar test para seleccionar modelos, épocas o umbrales.",
            [("Configuración y ejecución", source)],
            colab_notebook_id=colab_id,
        )
    create(
        "flujo/04_produccion/04_01_frontend_produccion.ipynb",
        "04.01 · Frontend de producción supervisada",
        "Comprueba el registro v2.1 e inicia el demostrador local en modo sombra; nunca carga modelos históricos como sustitutos.",
        [
            ("Disponibilidad", "from moderacion_peru.artifacts import artifact_status\nartifact_status(ROOT)"),
            ("Inicio", "print('modperu serve-production --host 127.0.0.1 --port 8765')"),
        ],
    )


if __name__ == "__main__":
    main()
