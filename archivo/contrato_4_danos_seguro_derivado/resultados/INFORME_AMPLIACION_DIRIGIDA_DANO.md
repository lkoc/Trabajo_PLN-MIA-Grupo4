# Informe reproducible de ampliación dirigida de categorías de daño

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Actualización: 2026-07-27T05:56:52-05:00  
Lote: `ampliacion_dano_20260726`  
Estado del entrenamiento: **COMPLETADO**

## 1. Objetivo y salvaguardas

Se amplió el corpus desde el flujo del cuaderno 01 con muestreo dirigido a categorías minoritarias. Esta selección es adecuada para enriquecer entrenamiento, pero no para estimar prevalencia poblacional. El test histórico se mantiene congelado y ningún video nuevo entra en test. Las etiquetas finas solo fundamentan/proyectan objetivos gruesos; no se entrenan. Los flags transversales se conservan para enrutamiento y tampoco son categorías base.

## 2. Adquisición y segmentación

Se encontraron 2,458 candidatos únicos, se seleccionaron 500 y 476 tuvieron subtítulos utilizables (95.2%). Se descargaron exclusivamente subtítulos VTT públicos; no audio ni video. La segmentación compatible con el cuaderno 02 produjo 21,991 chunks nuevos, eliminando 147 duplicados por hash.

El corpus segmentado pasa de 69,853 a 91,844 chunks y de 1,856 a 2,332 videos. El conjunto híbrido utilizable contiene 90,065 chunks; quedan 1,779 dudas humanas sin cerrar. De las 1,779 dudas nuevas originales, las aceptadas o modificadas se incorporan y las rechazadas se excluyen.

| Estrategia de adquisición | Videos seleccionados |
|---|---:|
| Canales de alto rendimiento histórico | 370 |
| Búsquedas temáticas dirigidas | 130 |

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
| Canales de alto rendimiento | 352 | 18,841 | 375 | 22 |
| Búsquedas temáticas dirigidas | 122 | 1,371 | 82 | 57 |

Las búsquedas dirigidas se usan para extrapolar el tamaño restante porque fueron diseñadas específicamente para enriquecer amenaza directa; los conteos se recalculan al integrar la adjudicación humana.

## 3. Etiquetado Flash → Pro

Flash etiquetó 21,991 chunks, con 2,694 daños y costo estimado nuevo de USD 1.06. La regla Pro fue: todo daño Flash, toda alerta, `score_confianza < 0.90` y control aleatorio del 10% de seguros confiables. Pro revisó 5,183 chunks por aproximadamente USD 1.73.

Pro resolvió 3,404 seleccionados y derivó 1,779 a humano. La adjudicación humana está pendiente: 0 incluidos y 0 rechazados. Persisten 1,779 sin resolver (8.09% del lote nuevo). El costo API nuevo registrado fue aproximadamente USD 2.79.

## 4. Balance antes/después

| Categoría gruesa | Antes | Agregados Pro resueltos | Después utilizable | Déficit a 1.000 |
|---|---:|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 1,054 | 44 | 1,098 | 0 |
| `ACOSO_GENERO_IDENTIDAD` | 985 | 42 | 1,027 | 0 |
| `ACOSO_PERSONAL` | 1,238 | 277 | 1,515 | 0 |
| `AMENAZA_DIRECTA` | 221 | 79 | 300 | 700 |
| `CONTENIDO_SEXUAL` | 928 | 126 | 1,054 | 0 |

![Balance antes y después](figuras/ampliacion_dano/balance_antes_despues.png)

Cuatro de cinco categorías de daño quedan por encima de 1.000. La razón máxima/mínima entre daños baja de 5.60:1 a 5.05:1. Amenaza directa aumenta de 222 a 300 ejemplos limpios.

## 5. Estimación restante para amenaza directa

Entre 124 videos de búsqueda dirigida se obtuvieron 57 amenazas Pro resueltas: 0.460 por video. El bootstrap por video (10.000 remuestras) dio IC 95% [0.274, 0.677] para el rendimiento medio. Para cubrir los 700 faltantes se estiman 1,523 videos dirigidos; usando el límite inferior como plan conservador, 2,553. Esta extrapolación presupone que se mantiene la mezcla de consultas y la disponibilidad de subtítulos.

## 6. Particiones y control de fuga

Los 20,212 casos nuevos utilizables se agruparon por `video_id`: 16,168 train y 4,044 validation, cero test. La búsqueda de semilla evaluó 500 particiones y seleccionó `26072033` sin celdas gruesas vacías. La validación enriquecida es diagnóstica; la selección del modelo usa la validación histórica y la evaluación final usa el test histórico congelado por video.

## 7. Reentrenamiento y resultados

| Experimento | PR-AUC daño validación | F1 daño test | PR-AUC daño test | Recall micro daño test |
|---|---:|---:|---:|---:|
| `ampliado_con_aeda` | 0.2319 | 0.2487 | 0.2202 | 0.2457 |
| `baseline_reproducido` | 0.2264 | 0.2564 | 0.2265 | 0.2693 |
| `ampliado_sin_aeda` | 0.2261 | 0.2444 | 0.2211 | 0.2409 |

![Comparación del reentrenamiento](figuras/ampliacion_dano/comparacion_entrenamiento.png)

Ganador y configuración exportada por el criterio predefinido de validación: `ampliado_con_aeda`. La evidencia del test NO respalda una mejora general en detección de daño. La selección no se cambia retrospectivamente usando test.

## 8. Artefactos y hashes

- Dataset nuevo utilizable: `datos\ampliacion\ampliacion_dano_20260726\processed\dataset_etiquetado_utilizable.jsonl`; SHA-256 `6f05735c93c16b165707240946156651affa90d21a5a72578c66ce6a578e4b41`.
- Cola humana nueva: `datos\ampliacion\ampliacion_dano_20260726\processed\pendientes_revision_humana.jsonl`; SHA-256 `d65622af30683f2821ac89629055222189b4d0597a9346689d6c31429d35d305`.
- Manifiesto: `datos\ampliacion\ampliacion_dano_20260726\processed\dataset_etiquetado_utilizable.manifest.json`.
- Cuaderno orquestador: `Cuadernos\01_1_ampliacion_dirigida_dano.ipynb`.
- Rendimiento detallado por fuente: `resultados\metricas\rendimiento_ampliacion_por_fuente.csv`.
- Chunks nuevos: SHA-256 `a1529ef1060442e70b14b7d19a4d4b7c280b851bf91b544714f45d04d4d897fc`.
- Etiquetas Flash: SHA-256 `4703295f2b70ea9aa563a5ce6e482f9144566547b0b0c9c7e5afacfbe1d1e73e`.
- Revisión Pro: SHA-256 `e45f28dd25d9f5761bf6e54db83fe7f73df4765c5b26448c95d047a320e1e065`.

Comandos reproducibles desde la raíz del repositorio:

```powershell
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
