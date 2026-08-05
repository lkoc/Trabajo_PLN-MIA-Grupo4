# Matriz de trazabilidad de implementación

| Afirmación/decisión | Fuente externa | Artefacto interno | Ubicación de uso | Estado |
|---|---|---|---|---|
| cinco salidas entrenadas y `SEGURO` excluyente | decisión del proyecto | `config/taxonomia_v2.json` | etiquetado, entrenamiento y frontend | implementado |
| cuatro daños pueden coexistir | clasificación multietiqueta | esquema Pydantic y taxonomía | `AnnotationRecord` | implementado |
| `RACISMO_DISCRIMINACION` incluye racialización lingüística, clasismo y formas encubiertas pertinentes al Perú | Waseem y Banko; Callirgos, Portocarrero/Vich, Zavala/Almeida, Brañez y Salem | definición, cinco etiquetas finas y contraejemplos | contrato, prompt, paper y auditoría fina | implementado; requiere evaluación funcional peruana |
| `ATAQUE_POR_GENERO_IDENTIDAD` expresa daño por género, orientación o identidad sin exigir acoso ni intención de odio | Banko, Zeinert, EXIST y Chakravarthi; Albornoz/Flores, Defensoría, Lovón-Cueva y Rottenbacher | definición, dos etiquetas finas y alias de migración | contrato, prompt, frontend, paper y migrador | implementado en v2.1; no es categoría jurídica |
| `ACOSO_AMENAZA` conserva ataque personal y amenaza como fenómenos distintos | Waseem, Wulczyn y Banko; Albornoz/Flores y Defensoría | dos etiquetas finas; unión gruesa | contrato, migrador, métricas finas | implementado; fusión estadística, no equivalencia semántica |
| `CONTENIDO_SEXUAL` separa explícito, cosificación y no consentimiento | Banko, Zeinert y política de plataforma; Albornoz/Flores y Defensoría | tres etiquetas finas y exclusiones informativas | contrato, prompt, paper y auditoría fina | implementado; el modelo textual no infiere imagen, consentimiento ni delito |
| acoso personal y amenaza se fusionan por soporte | informes ejecutados y auditoría taxonómica | migrador v2 | snapshot v2 | implementado; no equivalencia jurídica |
| videos previos no se descargan | requisito incremental | `acquisition.ingest_incremental` | etapa 01 | implementado |
| partición sin fuga por video | protocolo experimental | `datasets.stable_video_split` | etapa 03 | implementado |
| Ollama es la ruta local oficial | documentación de JSON Schema de Ollama | `providers/OllamaProvider` | etapa 02 | implementado |
| CUDA/ROCm/XPU/CPU | documentación PyTorch | `device.resolve_device` | entrenamiento/manifiestos | implementado |
| métricas actuales no pertenecen a v2 | resultados históricos | `archivo/contrato_4_danos_seguro_derivado` | paper/resultados | preservado |

Las cifras editoriales deben añadir artefacto, campo, split y fecha/hash antes de incorporarse al paper.
