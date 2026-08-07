# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Bundle mínimo para Colab

Esta carpeta contiene el staging local y versionable del bundle reproducible. `project_core.zip`, `chunks_v2.jsonl.gz` y `dataset_5_salidas.jsonl.gz` se regeneran con `02_00_preparacion_bundle_colab.ipynb` o `tools/prepare_colab_bundle.py`; `bundle_manifest.json` registra tamaños, SHA-256 y un `bundle_id` estable derivado del contenido.

El bundle local del 7 de agosto de 2026 contiene los 166.940 chunks v2.2 (`source_sha256=2506123ed7a9d78fcf466e1af8875d96a70651ae8b4c22a1e8e13ccd1c542828`). El dataset de 117.244 filas sigue siendo el snapshot etiquetado anterior y no hereda automáticamente los nuevos `chunk_id`; debe reemplazarse después de ejecutar `02_01`–`02_05` y publicarse repitiendo `02_00`. `drive_upload.json` describe la carga remota anterior y no sustituye este paso.

El dataset se comprime con gzip reproducible nivel 9. Su tamaño actual permite conservar un solo archivo; si se aproxima a 45–50 MiB comprimidos deberá partirse por `split` y partes numeradas, actualizando primero el contrato del manifiesto. Los chunks son reconstruibles, pero su gzip permanece aquí porque `02_01` lo consume en Colab. El dataset anotado no se considera barato de recrear.

Después de clonar, `python tools/restore_synced_checkpoints.py` verifica el manifiesto y descomprime atómicamente las entradas a sus rutas de trabajo. Los cuadernos `03_01`–`03_08` vuelven a verificar el dataset y no reemplazan silenciosamente una copia local divergente.

`02_00` publica exclusivamente en `ModeracionPeru_Colab/bundle_releases/<bundle_id>` mediante Google Drive Desktop. Si la versión ya existe y supera nuevamente la verificación devuelve `already_present`; nunca modifica una versión inmutable. Después actualiza atómicamente `bundle_releases/latest.json`. Los cuadernos Colab leen ese puntero y, si hace falta, promueven la versión a la carpeta activa `bundle/` antes de instalar. `drive_upload.json` conserva evidencia histórica de la carga anterior y no se usa como puntero vigente del nuevo protocolo.

No añada modelos, cachés de Hugging Face, videos ni checkpoints sueltos. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
