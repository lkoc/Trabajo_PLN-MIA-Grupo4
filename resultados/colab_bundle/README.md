# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Bundle mínimo para Colab

Esta carpeta es el bundle reproducible sincronizado tanto por Git como por Google Drive. `project_core.zip`, `chunks_v2.jsonl.gz` y `dataset_5_salidas.jsonl.gz` se regeneran con `tools/prepare_colab_bundle.py`; `bundle_manifest.json` registra sus tamaños y SHA-256, y `drive_upload.json` registra los IDs y carpetas comprobados mediante lectura posterior en Google Drive.

El dataset se comprime con gzip reproducible nivel 9. Su tamaño actual permite conservar un solo archivo; si se aproxima a 45–50 MiB comprimidos deberá partirse por `split` y partes numeradas, actualizando primero el contrato del manifiesto. Los chunks son reconstruibles, pero su gzip permanece aquí porque `02_01` lo consume en Colab. El dataset anotado no se considera barato de recrear.

Después de clonar, `python tools/restore_synced_checkpoints.py` verifica el manifiesto y descomprime atómicamente las entradas a sus rutas de trabajo. Los cuadernos `03_01`–`03_08` vuelven a verificar el dataset y no reemplazan silenciosamente una copia local divergente.

`drive_upload.json` solo puede declararse vigente después de reemplazar los archivos remotos y comprobarlos mediante lectura posterior. El estado `remote_bundle_outdated` significa que el bundle local/Git es más reciente que la carga de Drive registrada.

No añada modelos, cachés de Hugging Face, videos ni checkpoints sueltos. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
