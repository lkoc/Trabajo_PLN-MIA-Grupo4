# Matriz de trazabilidad de implementación

| Afirmación/decisión | Fuente externa | Artefacto interno | Ubicación de uso | Estado |
|---|---|---|---|---|
| cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`; `SEGURO` es excluyente | decisión del proyecto | `config/taxonomia_v2.json` | etiquetado, entrenamiento y frontend | implementado |
| `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL` pueden coexistir | clasificación multietiqueta | esquema Pydantic y taxonomía | `AnnotationRecord` | implementado |
| `RACISMO_DISCRIMINACION` incluye racialización lingüística, clasismo y formas encubiertas pertinentes al Perú | Waseem y Banko; Callirgos, Portocarrero/Vich, Zavala/Almeida, Brañez y Salem | definición, cinco etiquetas finas y contraejemplos | contrato, prompt, paper y auditoría fina | implementado; requiere evaluación funcional peruana |
| `ATAQUE_POR_GENERO_IDENTIDAD` expresa daño por género, orientación o identidad sin exigir acoso ni intención de odio | Banko, Zeinert, EXIST y Chakravarthi; Albornoz/Flores, Defensoría, Lovón-Cueva y Rottenbacher | definición, dos etiquetas finas y alias de migración | contrato, prompt, frontend, paper y migrador | implementado en el contrato de etiquetas v2.1; no es categoría jurídica |
| `ACOSO_AMENAZA` conserva ataque personal y amenaza como fenómenos distintos | Waseem, Wulczyn y Banko; Albornoz/Flores y Defensoría | dos etiquetas finas; unión gruesa | contrato, migrador, métricas finas | implementado; fusión estadística, no equivalencia semántica |
| `CONTENIDO_SEXUAL` separa explícito, cosificación y no consentimiento | Banko, Zeinert y política de plataforma; Albornoz/Flores y Defensoría | tres etiquetas finas y exclusiones informativas | contrato, prompt, paper y auditoría fina | implementado; el modelo textual no infiere imagen, consentimiento ni delito |
| acoso personal y amenaza se fusionan por soporte | informes ejecutados y auditoría taxonómica | migrador v2 | snapshot v2 | implementado; no equivalencia jurídica |
| videos previos no se descargan | requisito incremental | `acquisition.ingest_incremental` | etapa 01 | implementado |
| el canónico local se reconstruye desde partes pequeñas por canal y VTT recuperables sin borrar pistas | requisito de continuidad entre máquinas | `consolidate_available_transcripts`, `recover_transcripts_from_vtt`, índice por canal | `01_01`, `01_03` | implementado; 5.002 videos, 339 partes, parte máxima de 26.061.145 bytes |
| el troceado incremental es observable y la reconstrucción total es recuperable | requisito de repetibilidad | barra por video, firma, manifiesto y `archivo/chunk_rebuilds/` | `01_03` y `docs/MATERIALIZACION_TROCEADO.md` | implementado; 166.940 chunks, 99,80 % de videos con salida |
| partición sin fuga por video | protocolo experimental | `datasets.stable_video_split` | etapa 03 | implementado |
| eventos humanos vuelven al consolidado sin sobrescribir propuestas | requisito de precedencia y trazabilidad | `consolidation.reconcile_human_reviews`, `ReviewEvent` | `02_05` | implementado; último evento por fecha+ID |
| `video_id` no se infiere desde un `chunk_id` ambiguo | requisito de integridad de grupos | `materialize_versioned_training_snapshot` | `02_05` y split | implementado; ausencia explícita detiene la etapa |
| aumentar muestra crea snapshot nuevo y ejecución sin cambios es no-op | requisito incremental | firma de insumos, snapshot por contenido y run signature | `02_05`, `03_01`–`03_08` | implementado y probado |
| todas las ramas completan fit, calibración, test y candidato | protocolo experimental | `experiments.py` | `03_01`–`03_06` | implementado; smoke real clásico y neuronal simulado |
| selección productiva no consulta test | control de sesgo de selección | `registry.compare_and_publish_registry` | `03_07` | implementado; ranking solo validation |
| frontend histórico recuperado bajo el contrato de etiquetas v2.1 | requisito funcional | `servers.py` y ambos HTML activos | etapas 02 y 04 | implementado: contexto, YouTube, revisión, estadísticas y exportación |
| Ollama es la ruta local oficial | documentación de JSON Schema de Ollama | `providers/OllamaProvider` | etapa 02 | implementado |
| CUDA/ROCm/XPU/CPU | documentación PyTorch | `device.resolve_device` | entrenamiento/manifiestos | implementado |
| Colab L4 desde VS Code sin duplicar el repositorio | extensión oficial Google Colab y FAQ de Drive/Colab | `config/colab_l4.json`, bundle SHA-256 y `colab.prepare_colab_context` | `02_01`, `03_02`–`03_06` | implementado; Drive-only, sin GitHub |
| métricas actuales no pertenecen a v2 | resultados históricos | `archivo/contrato_4_danos_seguro_derivado` | paper/resultados | preservado |

Las cifras editoriales deben añadir artefacto, campo, split y fecha/hash antes de incorporarse al paper.
