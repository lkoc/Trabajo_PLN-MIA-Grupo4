# Informe reproducible de ampliación dirigida de categorías de daño

Actualización: 2026-07-27T10:28:34-05:00  
Lote: `ampliacion_dano_20260727_lote2`  
Estado del entrenamiento: **DATOS PREPARADOS; ENTRENAMIENTO AÚN NO EJECUTADO**

## 1. Objetivo y salvaguardas

Se amplió el corpus desde el flujo del cuaderno 01 con muestreo dirigido a categorías minoritarias. Esta selección es adecuada para enriquecer entrenamiento, pero no para estimar prevalencia poblacional. El test histórico se mantiene congelado y ningún video nuevo entra en test. Las etiquetas finas solo fundamentan/proyectan objetivos gruesos; no se entrenan. Los flags transversales se conservan para enrutamiento y tampoco son categorías base.

## 2. Adquisición y segmentación

Se encontraron 2,458 candidatos únicos, se seleccionaron 600 y 548 tuvieron subtítulos utilizables (91.3%). Se descargaron exclusivamente subtítulos VTT públicos; no audio ni video. La segmentación compatible con el cuaderno 02 produjo 19,971 chunks nuevos, eliminando 64 duplicados por hash.

Los corpus segmentados previos sumaban 91,844 chunks; el nuevo lote eleva ese inventario auditable a 111,815. Se añadieron 547 videos con chunks y el manifiesto de adquisición confirma intersección cero con los videos ya procesados. El conjunto utilizable para integración pasa de 97,287 a 117,244 chunks; quedan 0 dudas humanas sin cerrar. De las 1,088 dudas nuevas originales, las aceptadas o modificadas se incorporan y las rechazadas o todavía abiertas se excluyen.

| Estrategia de adquisición | Videos seleccionados |
|---|---:|
| Canales de alto rendimiento histórico | 312 |
| Búsquedas temáticas dirigidas | 288 |

Fuentes de canal priorizadas y criterio previo:

| Fuente | Cuota | Criterio observado en el corpus base |
|---|---:|---|
| Hablando Huevadas | 70 | 22.60 chunks de daño y 0.86 amenazas por video en el corpus base |
| Goblinciano | 85 | 9.82 chunks de daño y 0.58 amenazas por video en el corpus base |
| Juanito y Richard | 85 | 2.22 chunks de daño por video; máximo observado de 11 amenazas en un episodio |
| Arde Troya con Juliana Oxenford | 55 | 2.41 chunks de daño y 0.39 amenazas por video en el corpus base |
| Todo Good | 40 | 2.16 chunks de daño por video en el corpus base |
| Magaly TV La Firme | 35 | 1.17 chunks de daño por video y cobertura de conflicto interpersonal |

Rendimiento limpio del lote por estrategia (después de excluir dudas Pro):

| Estrategia | Videos utilizables | Chunks utilizables | Chunks de daño | Amenazas directas |
|---|---:|---:|---:|---:|
| Canales de alto rendimiento | 279 | 15,013 | 1,032 | 46 |
| Búsquedas temáticas dirigidas | 268 | 4,944 | 210 | 107 |

Las búsquedas dirigidas se usan para extrapolar el tamaño restante porque fueron diseñadas específicamente para enriquecer amenaza directa; los conteos se recalculan al integrar la adjudicación humana.

## 3. Etiquetado Flash → Pro

Flash etiquetó 19,971 chunks, con 1,826 daños y costo estimado nuevo de USD 0.94. La regla Pro fue: todo daño Flash, toda alerta, `score_confianza < 0.90` y control aleatorio del 10% de seguros confiables. Pro revisó 4,105 chunks por aproximadamente USD 1.27.

Pro resolvió 3,017 seleccionados y derivó 1,088 a humano. La adjudicación humana está completa: 1,074 incluidos y 14 rechazados. Persisten 0 sin resolver (0.00% del lote nuevo). El costo API nuevo registrado fue aproximadamente USD 2.22.

## 4. Balance antes/después

| Categoría gruesa | Antes | Agregados Pro resueltos | Después utilizable | Déficit a 1.000 |
|---|---:|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 1,591 | 265 | 1,856 | 0 |
| `ACOSO_GENERO_IDENTIDAD` | 1,599 | 362 | 1,961 | 0 |
| `ACOSO_PERSONAL` | 2,303 | 551 | 2,854 | 0 |
| `AMENAZA_DIRECTA` | 535 | 153 | 688 | 312 |
| `CONTENIDO_SEXUAL` | 1,734 | 548 | 2,282 | 0 |

![Balance antes y después](figuras/ampliacion_dano_20260727_lote2/balance_antes_despues.png)

4 de cinco categorías de daño quedan por encima de 1.000. La razón máxima/mínima entre daños cambia de 4.30:1 a 4.15:1. Amenaza directa aumenta de 535 a 688 ejemplos limpios.

## 5. Estimación restante para amenaza directa

Entre 268 videos de búsqueda dirigida se obtuvieron 107 amenazas Pro resueltas: 0.399 por video. El bootstrap por video (10.000 remuestras) dio IC 95% [0.246, 0.582] para el rendimiento medio. Para cubrir los 312 faltantes se estiman 782 videos dirigidos; usando el límite inferior como plan conservador, 1,267. Esta extrapolación presupone que se mantiene la mezcla de consultas y la disponibilidad de subtítulos.

## 6. Particiones y control de fuga

Los 19,957 casos nuevos utilizables se agruparon por `video_id`: 15,936 train y 4,021 validation, cero test. La búsqueda de semilla evaluó 500 particiones y seleccionó `26072495` sin celdas gruesas vacías. La validación enriquecida es diagnóstica; la selección del modelo usa la validación histórica y la evaluación final usa el test histórico congelado por video.

## 7. Reentrenamiento incremental autorizado

El flujo se detuvo antes de `04_2`, como fue solicitado. El progreso humano disponible para este lote es 1088/1088 decisiones completas. Las decisiones cerradas elegibles se aplican; rechazos y 0 casos todavía abiertos se excluyen del dataset utilizable. `04_2` descubrirá este archivo mediante su manifiesto, sin requerir una ruta codificada.

## 8. Artefactos y hashes

- Dataset nuevo utilizable: `datos\ampliacion\ampliacion_dano_20260727_lote2\processed\dataset_etiquetado_utilizable.jsonl`; SHA-256 `3bfc1fc938570d65abb182f7e1c3e8f230397d69a5be8e315e927256245b0682`.
- Cola humana nueva: `datos\ampliacion\ampliacion_dano_20260727_lote2\processed\pendientes_revision_humana.jsonl`; SHA-256 `56430b83aa953ddabfc050eac581e29dcadb03d26e9fc7642bf53f646bb63c1d`.
- Manifiesto: `datos\ampliacion\ampliacion_dano_20260727_lote2\processed\dataset_etiquetado_utilizable.manifest.json`.
- Cuaderno orquestador: `Cuadernos\01_1_ampliacion_dirigida_dano.ipynb`.
- Rendimiento detallado por fuente: `resultados\metricas\ampliacion_dano_20260727_lote2\rendimiento_por_fuente.csv`.
- Chunks nuevos: SHA-256 `36df3309bd2794de17c434db9f18345aba4d6a581ed9529f8cb331534c9f71bc`.
- Etiquetas Flash: SHA-256 `2cdde58a025d1273efe27aab4cd2a9b3846b3a4eba50b607560288bbc8299d9e`.
- Revisión Pro: SHA-256 `5d46e5599b754002a8f60ea987d54cc4de620fc9cd4e4f9a3f18dda5784261f2`.

Comandos reproducibles desde la raíz del repositorio:

```powershell
$env:AMPLIACION_BATCH_ID='ampliacion_dano_20260727_lote2'
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
