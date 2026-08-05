# Bundle mínimo para Colab

Esta carpeta conserva la trazabilidad local del bundle Drive-only. `project_core.zip`, `chunks_v2.jsonl.gz` y `dataset_5_salidas.jsonl.gz` se regeneran con `tools/prepare_colab_bundle.py` y están excluidos de Git. `bundle_manifest.json` registra sus tamaños y SHA-256; `drive_upload.json` registra los IDs y carpetas comprobados mediante lectura posterior en Google Drive.

No añada modelos, cachés de Hugging Face, videos ni checkpoints sueltos. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
