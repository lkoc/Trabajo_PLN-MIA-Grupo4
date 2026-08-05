# Matriz de trazabilidad de implementación

| Afirmación/decisión | Fuente externa | Artefacto interno | Ubicación de uso | Estado |
|---|---|---|---|---|
| cinco salidas entrenadas y `SEGURO` excluyente | decisión del proyecto | `config/taxonomia_v2.json` | etiquetado, entrenamiento y frontend | implementado |
| cuatro daños pueden coexistir | clasificación multietiqueta | esquema Pydantic y taxonomía | `AnnotationRecord` | implementado |
| acoso personal y amenaza se fusionan por soporte | informes ejecutados y auditoría taxonómica | migrador v2 | snapshot v2 | implementado; no equivalencia jurídica |
| videos previos no se descargan | requisito incremental | `acquisition.ingest_incremental` | etapa 01 | implementado |
| partición sin fuga por video | protocolo experimental | `datasets.stable_video_split` | etapa 03 | implementado |
| Ollama es la ruta local oficial | documentación de JSON Schema de Ollama | `providers/OllamaProvider` | etapa 02 | implementado |
| CUDA/ROCm/XPU/CPU | documentación PyTorch | `device.resolve_device` | entrenamiento/manifiestos | implementado |
| métricas actuales no pertenecen a v2 | resultados históricos | `archivo/contrato_4_danos_seguro_derivado` | paper/resultados | preservado |

Las cifras editoriales deben añadir artefacto, campo, split y fecha/hash antes de incorporarse al paper.

