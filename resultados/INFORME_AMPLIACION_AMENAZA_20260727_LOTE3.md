# Informe reproducible de ampliación dirigida de categorías de daño

Actualización: 2026-07-27T10:28:25-05:00  
Lote: `ampliacion_amenaza_20260727_lote3`  
Estado del entrenamiento: **DATOS PREPARADOS; ENTRENAMIENTO AÚN NO EJECUTADO**

## 1. Objetivo y salvaguardas

Se amplió el corpus desde el flujo del cuaderno 01 con muestreo dirigido a categorías minoritarias. Esta selección es adecuada para enriquecer entrenamiento, pero no para estimar prevalencia poblacional. El test histórico se mantiene congelado y ningún video nuevo entra en test. Las etiquetas finas solo fundamentan/proyectan objetivos gruesos; no se entrenan. Los flags transversales se conservan para enrutamiento y tampoco son categorías base.

## 2. Adquisición y segmentación

Se encontraron 1,450 candidatos únicos, se seleccionaron 360 y 353 tuvieron subtítulos utilizables (98.1%). Se descargaron exclusivamente subtítulos VTT públicos; no audio ni video. La segmentación compatible con el cuaderno 02 produjo 5,451 chunks nuevos, eliminando 129 duplicados por hash.

Los corpus segmentados previos sumaban 111,815 chunks; el nuevo lote eleva ese inventario auditable a 117,266. Se añadieron 351 videos con chunks y el manifiesto de adquisición confirma intersección cero con los videos ya procesados. El conjunto utilizable para integración pasa de 110,719 a 116,170 chunks; quedan 0 dudas humanas sin cerrar. De las 108 dudas nuevas originales, las aceptadas o modificadas se incorporan y las rechazadas o todavía abiertas se excluyen.

| Estrategia de adquisición | Videos seleccionados |
|---|---:|
| Canales de alto rendimiento histórico | 0 |
| Búsquedas temáticas dirigidas | 360 |

Fuentes de canal priorizadas y criterio previo:

| Fuente | Cuota | Criterio observado en el corpus base |
|---|---:|---|

Rendimiento limpio del lote por estrategia (después de excluir dudas Pro):

| Estrategia | Videos utilizables | Chunks utilizables | Chunks de daño | Amenazas directas |
|---|---:|---:|---:|---:|
| Búsquedas temáticas dirigidas | 351 | 5,451 | 239 | 149 |

Las búsquedas dirigidas se usan para extrapolar el tamaño restante porque fueron diseñadas específicamente para enriquecer amenaza directa; los conteos se recalculan al integrar la adjudicación humana.

## 3. Etiquetado Flash → Pro

Flash etiquetó 5,451 chunks, con 383 daños y costo estimado nuevo de USD 0.31. La regla Pro fue: todo daño Flash, toda alerta, `score_confianza < 0.90` y control aleatorio del 10% de seguros confiables. Pro revisó 921 chunks por aproximadamente USD 0.27.

Pro resolvió 813 seleccionados y derivó 108 a humano. La adjudicación humana está completa: 108 incluidos y 0 rechazados. Persisten 0 sin resolver (0.00% del lote nuevo). El costo API nuevo registrado fue aproximadamente USD 0.58.

## 4. Balance antes/después

| Categoría gruesa | Antes | Agregados Pro resueltos | Después utilizable | Déficit a 1.000 |
|---|---:|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 1,601 | 14 | 1,615 | 0 |
| `ACOSO_GENERO_IDENTIDAD` | 1,654 | 35 | 1,689 | 0 |
| `ACOSO_PERSONAL` | 2,342 | 188 | 2,530 | 0 |
| `AMENAZA_DIRECTA` | 486 | 149 | 635 | 365 |
| `CONTENIDO_SEXUAL` | 1,849 | 12 | 1,861 | 0 |

![Balance antes y después](figuras/ampliacion_amenaza_20260727_lote3/balance_antes_despues.png)

4 de cinco categorías de daño quedan por encima de 1.000. La razón máxima/mínima entre daños cambia de 4.82:1 a 3.98:1. Amenaza directa aumenta de 486 a 635 ejemplos limpios.

## 5. Estimación restante para amenaza directa

Entre 351 videos de búsqueda dirigida se obtuvieron 149 amenazas Pro resueltas: 0.425 por video. El bootstrap por video (10.000 remuestras) dio IC 95% [0.319, 0.541] para el rendimiento medio. Para cubrir los 365 faltantes se estiman 860 videos dirigidos; usando el límite inferior como plan conservador, 1,144. Esta extrapolación presupone que se mantiene la mezcla de consultas y la disponibilidad de subtítulos.

## 6. Particiones y control de fuga

Los 5,451 casos nuevos utilizables se agruparon por `video_id`: 4,332 train y 1,119 validation, cero test. La búsqueda de semilla evaluó 500 particiones y seleccionó `26072183` sin celdas gruesas vacías. La validación enriquecida es diagnóstica; la selección del modelo usa la validación histórica y la evaluación final usa el test histórico congelado por video.

## 7. Reentrenamiento incremental autorizado

El flujo se detuvo antes de `04_2`, como fue solicitado. El progreso humano disponible para este lote es 108/108 decisiones completas. Las decisiones cerradas elegibles se aplican; rechazos y 0 casos todavía abiertos se excluyen del dataset utilizable. `04_2` descubrirá este archivo mediante su manifiesto, sin requerir una ruta codificada.

## 8. Artefactos y hashes

- Dataset nuevo utilizable: `datos\ampliacion\ampliacion_amenaza_20260727_lote3\processed\dataset_etiquetado_utilizable.jsonl`; SHA-256 `599b562ac41748e26f195f8cf22c08b5a7bf58718070c5e8f994841c8a67a915`.
- Cola humana nueva: `datos\ampliacion\ampliacion_amenaza_20260727_lote3\processed\pendientes_revision_humana.jsonl`; SHA-256 `c8edd437c206a2f5eb985a8f8c7bc51e927437f6aa777b8578ab27a9a5f134cb`.
- Manifiesto: `datos\ampliacion\ampliacion_amenaza_20260727_lote3\processed\dataset_etiquetado_utilizable.manifest.json`.
- Cuaderno orquestador: `Cuadernos\01_1_ampliacion_dirigida_dano.ipynb`.
- Rendimiento detallado por fuente: `resultados\metricas\ampliacion_amenaza_20260727_lote3\rendimiento_por_fuente.csv`.
- Chunks nuevos: SHA-256 `025b9f7e80e35bc95058983cbb5096f4f4c2c0df91c83d0b63734b708264ea3c`.
- Etiquetas Flash: SHA-256 `ac9cdac4681309da9051f98997850f358ad739dfa7adb793f37fb06de7dc7fdf`.
- Revisión Pro: SHA-256 `a8e4aefbdd5c18743d5d4b690b6d6a1a7312b8c4fdb6d12677fd4606c2756b79`.

Comandos reproducibles desde la raíz del repositorio:

```powershell
$env:AMPLIACION_BATCH_ID='ampliacion_amenaza_20260727_lote3'
$env:AMPLIACION_SEED='26072026'
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage discover
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage transcribe
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage chunk
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage flash
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage pro
python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage prepare
python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage train
```

## 9. Limitaciones metodológicas

- La adquisición dirigida altera deliberadamente la prevalencia de entrenamiento; no estima la prevalencia natural de YouTube.
- Los conteos antes de cerrar la adjudicación humana combinada son provisionales.
- Excluir dudas Pro mejora pureza aparente, pero puede retirar ejemplos fronterizos; por eso se conserva la cola humana completa.
- El test histórico ya fue observado en experimentos anteriores; sirve para comparación de ingeniería, no sustituye un holdout humano ciego nuevo.
- Llegar a 1.000 no garantiza suficiencia: importa diversidad de videos/canales y calidad de etiqueta, además del conteo.

## 10. Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Fairstein, Y., Kalinsky, O., Karnin, Z., Kushilevitz, G., Libov, A., & Tolmach, S. (2024). Class balancing for efficient active learning in imbalanced datasets. In *Proceedings of the 18th Linguistic Annotation Workshop* (pp. 77–86). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.law-1.8

Fithian, W., & Hastie, T. (2014). Local case-control sampling: Efficient subsampling in imbalanced data sets. *The Annals of Statistics, 42*(5), 1693–1724. https://doi.org/10.1214/14-AOS1220

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Scikit-learn developers. (2026). *GroupShuffleSplit*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
