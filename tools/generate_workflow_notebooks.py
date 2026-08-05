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


def create(path: str, title: str, purpose: str, code_cells: list[tuple[str, str]]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "moderacion_peru": {
            "contract": "moderacion_peru_5_salidas_v2",
            "taxonomy_version": "2.1.0",
            "orchestrator": True,
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
            "No instala paquetes ni usa rutas personales. Revise el README de esta etapa antes de ejecutar."
        ),
        nbf.v4.new_code_cell(SETUP),
    ]
    for heading, source in code_cells:
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        notebook.cells.append(nbf.v4.new_code_cell(source))
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
        "02.01 · Etiquetado local con Ollama",
        "Usa la API HTTP local de Ollama, con JSON Schema, reintento y reanudación por `chunk_id`.",
        [
            ("Preflight Ollama", "from moderacion_peru.providers import OllamaProvider\nprovider=OllamaProvider(model='qwen3.5:4b')\nprovider.probe()"),
            ("Etiquetado incremental", "from moderacion_peru.io import read_jsonl\nfrom moderacion_peru.labeling import annotate_incremental\nSOURCE=ROOT/'datos/processed/chunks_v2.jsonl'\nOUTPUT=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.jsonl'\nERRORS=ROOT/'datos/etiquetado/local/ollama_qwen35_4b_v2.errors.jsonl'\nRUN=False\nif RUN:\n    print(annotate_incremental(read_jsonl(SOURCE),provider,OUTPUT,error_path=ERRORS))\nelse:\n    print('Preflight completo; cambie RUN=True para etiquetar solo chunks pendientes.')"),
        ],
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
        ("03_01_modelos_clasicos.ipynb", "Modelos clásicos", "from moderacion_peru.datasets import load_split\nfrom moderacion_peru.models import train_classical_suite\nDATA=ROOT/'datos/model_ready/v2/dataset_5_salidas.jsonl'\nRUN=False\nif RUN:\n    print(train_classical_suite(load_split(DATA,'train'),ROOT/'modelos/v2/clasicos'))\nelse: print('Listo para Dummy, NB, logística, SVM y SGD incremental.')"),
        ("03_02_transformers_planos.ipynb", "Transformers planos", "from moderacion_peru.models import TRANSFORMER_SPECS, build_transformer_classifier\nRUN=False\nif RUN:\n    tokenizer,model,hardware=build_transformer_classifier(TRANSFORMER_SPECS['e5'],'auto'); print(hardware)\nelse: print(TRANSFORMER_SPECS['minilm'],TRANSFORMER_SPECS['e5'])"),
        ("03_03_transformer_cascada.ipynb", "Transformer en cascada", "from moderacion_peru.taxonomy import load_taxonomy\nt=load_taxonomy(); print({'gate':'SEGURO frente a cualquier daño','children':t.damage_labels,'conflicts':'review'})"),
        ("03_04_transformer_multitarea.ipynb", "Transformer jerárquico multitarea", "from moderacion_peru.taxonomy import load_taxonomy\nt=load_taxonomy(); print({'primary_outputs':t.target_labels,'auxiliary_fine':len(t.fine_labels),'flags':t.flags})"),
        ("03_05_qwen_lora.ipynb", "Qwen-LoRA", "from moderacion_peru.models import build_qwen_lora_classifier\nRUN=False\nif RUN:\n    tokenizer,model,hardware=build_qwen_lora_classifier('auto'); print(hardware); model.print_trainable_parameters()\nelse: print('La nueva cabeza de cinco salidas se inicializa y todos los umbrales se recalibran.')"),
        ("03_06_qwen_estructurado.ipynb", "Qwen estructurado", "from moderacion_peru.taxonomy import load_taxonomy\nt=load_taxonomy(); print({'flat_reference':'03_05','gate_output':'SEGURO','damage_outputs':t.damage_labels,'selection':'validation only'})"),
        ("03_07_comparacion_final.ipynb", "Comparación final", "from moderacion_peru.artifacts import artifact_status\nprint('Comparar solo modelos con contrato moderacion_peru_5_salidas_v2 y el mismo test.'); artifact_status(ROOT)"),
        ("03_08_auditoria_finas_flags.ipynb", "Auditoría fina y transversal", "from moderacion_peru.taxonomy import load_taxonomy\nt=load_taxonomy(); print({'fine_labels':t.fine_labels,'flags':t.flags,'gold_inputs':False})"),
    ]
    for filename, subtitle, source in training_notebooks:
        create(
            f"flujo/03_entrenamiento/{filename}",
            f"03 · {subtitle}",
            "Entrena o audita el contrato v2.1 sin consultar test para seleccionar modelos, épocas o umbrales.",
            [("Configuración y ejecución", source)],
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
