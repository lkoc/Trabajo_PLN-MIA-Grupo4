# Informe de la revisión Pro de 2.000 seguros sospechosos

Fecha de ejecución: 26 de julio de 2026 (America/Lima)  
Estado: completado y auditado  
Cuaderno reproducible: `03_2_etiquetado_llm_api/03_2_1_revision_sospechosos.ipynb`  
Módulo reanudable: `scripts_auxiliares/revision_sospechosos_pro.py`

## Resumen ejecutivo

Se revisaron con `deepseek-v4-pro` los 2.000 chunks que Flash había clasificado como seguros, sin `needs_review` y con confianza mayor o igual que 0,90, pero que el moderador grueso situó más cerca de alguna categoría de daño. La selección comprendió 905 videos, con máximo tres chunks por video.

Pro confirmó 1.755 chunks como `SEGURO` (87,75%) y corrigió 245 a una o más categorías gruesas de daño (12,25%). Por tanto, **sí corresponde volver a entrenar** si se desea que el moderador aprenda estas correcciones. No basta con producir las nuevas etiquetas: deben sustituir a Flash con precedencia Pro dentro de la partición original.

Existe una salvedad importante: 139 de las 245 correcciones a daño conservan `needs_review=True` en Pro. Para una versión académicamente más defendible se recomienda incorporar de inmediato las 106 correcciones Pro sin duda residual y someter los otros 139 casos a adjudicación humana antes del ajuste definitivo. Una alternativa provisional es excluir o reducir el peso de esos 139 casos y documentar un análisis de sensibilidad.

## Pregunta y propósito

La revisión respondió a la pregunta operativa siguiente: entre los casos que Flash consideró seguros y confiables, ¿puede un modelo textual entrenado con las categorías gruesas detectar ejemplos sospechosos que merecen una segunda lectura por Pro?

El propósito no fue estimar la tasa global de error de Flash. Los 2.000 registros constituyen una selección dirigida de casos difíciles (*hard negatives*), no una muestra probabilística del corpus. Su tasa de corrección es válida como descripción de este estrato completo y como evidencia para depurar la frontera de entrenamiento, pero no debe extrapolarse a los aproximadamente 69 mil chunks.

La priorización de ejemplos informativos o potencialmente mal etiquetados es coherente con la literatura de aprendizaje activo y depuración de etiquetas (Brodley & Friedl, 1999; Settles, 2009). La limitación de inferencia poblacional se deriva del diseño no probabilístico de la selección (Lohr, 2021).

## Diseño y controles contra fuga de información

La selección de candidatos se generó con estas reglas:

- origen exclusivo: particiones de entrenamiento y validación;
- Flash había producido únicamente una etiqueta segura;
- `needs_review=False` en Flash;
- `score_confianza >= 0.90` en Flash;
- orden descendente según el máximo score de daño del moderador grueso;
- máximo tres chunks por video;
- 2.000 chunks únicos y 905 videos.

Antes de cualquier llamada facturable, el flujo reprodujo la partición agrupada por `video_id` con semilla 131. Se reconstruyeron las 10.293 filas de test y se verificó:

- solapamiento de `chunk_id` candidato–test: 0;
- solapamiento de `video_id` candidato–test: 0;
- solapamiento con las 10.000 revisiones Pro originales: 0;
- solapamiento con las 1.421 revisiones Pro del umbral recalibrado: 0;
- candidatos no seguros o con duda Flash: 0.

La distribución original de los 2.000 casos fue 1.635 en train y 365 en validation. Se conservó esta pertenencia; ningún caso fue trasladado entre particiones.

## Protocolo de etiquetado

| Elemento | Configuración |
|---|---|
| Proveedor | DeepSeek API compatible con `/chat/completions` |
| Modelo | `deepseek-v4-pro` |
| Identificador de anotador | `DSP` |
| Prompt | compacto, bundle 1.1, el mismo contrato normativo de `03_2` |
| Temperatura | 0,0 |
| Razonamiento | desactivado |
| Contexto | título del canal, título del video, chunk anterior y posterior del mismo video cuando existían |
| Tamaño de lote | 5 chunks |
| Paralelismo | 16 workers |
| Reintentos máximos | 5, con *backoff* exponencial |
| Formato | objeto JSON con arreglo `annotations`; validación local de esquema y semántica |
| Persistencia | JSONL incremental, `flush` + `fsync`, reanudación por prefijo ordenado |

La salida Pro conserva las 14 etiquetas finas y los tres flags solo para trazabilidad y mejora de los criterios de anotación. Para el entrenamiento, el mapeo normativo produce exclusivamente `SEGURO` o las cinco categorías gruesas. Ni las etiquetas finas ni los flags transversales se usan como predictores o clases de entrenamiento.

## Ejecución y costo

Se completaron 400 lotes exitosos de cinco chunks. El piloto de cinco registros tardó 9,45 segundos y la corrida principal de 1.990 registros tardó 171,96 segundos. El tiempo registrado de esas dos corridas fue 181,41 segundos, equivalente a 661,5 chunks por minuto.

Uso registrado por las corridas que devolvieron sus métricas:

| Métrica | Valor |
|---|---:|
| Prompt tokens | 2.318.154 |
| Completion tokens | 207.900 |
| Total tokens | 2.526.054 |
| Prompt cache hit tokens | 1.400.832 |
| Prompt cache miss tokens | 917.322 |
| Costo estimado registrado | USD 0,584986 |

Durante una primera tentativa de la corrida completa el proceso local agotó su tiempo de espera después de escribir cinco filas. La ejecución posterior se reanudó correctamente desde esas diez filas acumuladas. Las solicitudes que pudieron quedar en vuelo durante esa interrupción no devolvieron su objeto `usage` al cliente; por eso USD 0,584986 es el costo estimado **registrado**, mientras que el cargo real del proveedor puede ser ligeramente mayor. Esta incidencia no produjo IDs duplicados, desorden ni filas inválidas.

## Resultados

### Resultado binario y duda residual de Pro

| Resultado Pro | n | % de 2.000 |
|---|---:|---:|
| Confirma `SEGURO` | 1.755 | 87,75% |
| Corrige a una o más categorías gruesas de daño | 245 | 12,25% |
| Conserva `needs_review=True` | 139 | 6,95% |

Los 139 casos con `needs_review=True` pertenecen al grupo corregido a daño. Por lo tanto, hay 106 correcciones a daño sin duda residual de Pro y 139 que requieren una validación adicional.

Por partición:

| Partición original | Revisados | Confirma seguro | Corrige a daño | Daño con `needs_review=True` |
|---|---:|---:|---:|---:|
| Train | 1.635 | 1.432 | 203 | 114 |
| Validation | 365 | 323 | 42 | 25 |
| Total | 2.000 | 1.755 | 245 | 139 |

### Categorías gruesas asignadas por Pro

| Categoría gruesa | Chunks positivos |
|---|---:|
| `RACISMO_DISCRIMINACION` | 60 |
| `ACOSO_GENERO_IDENTIDAD` | 54 |
| `ACOSO_PERSONAL` | 101 |
| `AMENAZA_DIRECTA` | 12 |
| `CONTENIDO_SEXUAL` | 60 |

Las cuentas por categoría suman más de 245 porque la salida permite multilabel.

### Rendimiento de la selección de casos difíciles

| Categoría sospechada por el moderador | Revisados | Pro confirma daño | Tasa de confirmación |
|---|---:|---:|---:|
| `AMENAZA_DIRECTA` | 21 | 7 | 33,33% |
| `CONTENIDO_SEXUAL` | 266 | 48 | 18,05% |
| `ACOSO_GENERO_IDENTIDAD` | 305 | 42 | 13,77% |
| `ACOSO_PERSONAL` | 930 | 103 | 11,08% |
| `RACISMO_DISCRIMINACION` | 478 | 45 | 9,41% |

La tasa de confirmación Pro aumentó monotónicamente con el score de daño del moderador:

| Quintil del score | Rango observado | Pro confirma daño |
|---|---:|---:|
| 1 | 0,342–0,358 | 2,25% |
| 2 | 0,358–0,381 | 7,50% |
| 3 | 0,381–0,410 | 8,50% |
| 4 | 0,410–0,462 | 12,75% |
| 5 | 0,462–0,758 | 30,25% |

Esta relación es evidencia descriptiva de que el moderador fue útil para priorizar revisiones. No es una estimación de sensibilidad, especificidad ni prevalencia en el corpus, porque no hubo muestreo aleatorio y Flash había etiquetado todos estos casos como seguros.

El gráfico reproducible se guardó en `resultados/figuras/revision_pro_sospechosos_2000.png` y también quedó renderizado dentro del cuaderno `03_2_1`.

## Decisión de reentrenamiento

Sí corresponde reentrenar, con estas reglas:

1. Mantener intacta la partición agrupada por video y conservar los 10.293 chunks de test fuera de cualquier selección, ajuste o calibración.
2. Aplicar precedencia `Pro > Flash` a los IDs revisados dentro de su partición original.
3. Usar únicamente los objetivos gruesos: `SEGURO`, `RACISMO_DISCRIMINACION`, `ACOSO_GENERO_IDENTIDAD`, `ACOSO_PERSONAL`, `AMENAZA_DIRECTA` y `CONTENIDO_SEXUAL`.
4. Para el ajuste académicamente preferible, incorporar las 106 correcciones Pro sin duda y adjudicar humanamente los 139 casos con `needs_review=True` antes del entrenamiento definitivo. Mientras se revisan, excluirlos o reducir su peso es más prudente que tratarlos como verdad indiscutible.
5. Ajustar modelos con train; seleccionar hiperparámetros y umbrales solo con validation; reajustar el ganador con train + validation; evaluar una única vez el cambio sobre el test intacto.
6. Comparar contra el modelo congelado anterior y reportar métricas por categoría, macro-F1, sensibilidad de daño, precisión, tasa de falsos negativos y volumen enviado a revisión. El reentrenamiento se considera útil solo si mejora el objetivo operativo sin deterioros relevantes en el test.

Los 1.755 seguros confirmados no cambian su objetivo grueso, aunque constituyen evidencia adicional de que la selección no debe recodificarse automáticamente como daño. Las 245 correcciones sí cambian el conjunto; sin reentrenamiento, el modelo desplegable no puede aprenderlas.

## Limitaciones

- Pro es una referencia de mayor capacidad, no verdad humana definitiva.
- La selección dirigida maximiza utilidad para depuración, pero impide inferencia poblacional directa.
- La selección se calculó con scores de un moderador previamente ajustado; las tasas aquí descritas evalúan el mecanismo de priorización, no un conjunto independiente para medir desempeño final.
- Ciento treinta y nueve correcciones Pro aún manifiestan duda y deben priorizarse para revisión humana.
- La comparación final de beneficio exige reentrenar y evaluar sobre el test intacto; este informe documenta la corrección de etiquetas, no el rendimiento del modelo posterior.

## Archivos y trazabilidad

| Artefacto | Ruta / SHA-256 |
|---|---|
| Selección candidata | `datos/processed/flash_seguros_dificiles_para_revision.csv` |
| SHA-256 selección | `3c0a292562196fb63141cb6bad6c7b119f6c128b9e86124e7f0430c76ca4505d` |
| SHA-256 canónico | `eb90debf66d5e16af72c41c17c3701197e42bdcc78b81e0f914c6a49c56f8ab4` |
| Salida Pro | `datos/etiquetado/llm_api/deepseek-v4-pro_revision_sospechosos_gruesos_seed42.jsonl` |
| SHA-256 salida Pro | `745a9dc191482fd0a8609f5ffec3266abcf7a0bd40860d7c7b541f7dac045e34` |
| Manifiesto Pro | `datos/etiquetado/llm_api/deepseek-v4-pro_revision_sospechosos_gruesos_seed42.manifest.json` |
| SHA-256 manifiesto | `d033d92c32fd5ddf08ed5892e565055b453c2e377d07cc4652a09c4db82786af` |
| Métricas | `datos/etiquetado/llm_api/deepseek-v4-pro_revision_sospechosos_gruesos_seed42.metrics.json` |
| Decisión de reentrenamiento | `datos/etiquetado/llm_api/deepseek-v4-pro_revision_sospechosos_gruesos_seed42.retraining_decision.json` |
| SHA-256 skill | `45f9d3231a92453835ee6dfcbb8cfff0b682718caa4111f4bca3e841573a0efb` |
| SHA-256 prompt operativo completo | `a42004317f115b52b429771214afe3491224c0a19f14c9a07ea153ed74a82a57` |
| SHA-256 prompt compacto | `52d4fec14ad433d35ec20de5f51a6954aad69dcedd1422059419dcecc2f9e778` |

## Referencias

Brodley, C. E., & Friedl, M. A. (1999). Identifying mislabeled training data. *Journal of Artificial Intelligence Research, 11*, 131–167. https://doi.org/10.1613/jair.606

Lohr, S. L. (2021). *Sampling: Design and analysis* (3rd ed.). Chapman and Hall/CRC.

Settles, B. (2009). *Active learning literature survey* (Computer Sciences Technical Report 1648). University of Wisconsin–Madison. https://research.cs.wisc.edu/techreports/2009/TR1648.pdf
