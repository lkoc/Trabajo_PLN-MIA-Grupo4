# Informe consolidado de ampliación, etiquetado jerárquico e integración del dataset

Fecha de cierre operativo: 2026-07-27  
Alcance: lotes `ampliacion_dano_20260727_lote2` y `ampliacion_amenaza_20260727_lote3`  
Estado: **datos preparados e integrados; entrenamiento de modelos de 04_2 no ejecutado**

## 1. Objetivo y diseño

El propósito fue aumentar el corpus, reducir el desbalance de las cinco categorías gruesas de daño y, posteriormente, reforzar de forma específica `AMENAZA_DIRECTA`. La muestra es deliberadamente enriquecida para entrenamiento y validación; no representa la prevalencia natural del contenido de YouTube. En la preparación de cada lote se preservó el test histórico y los videos nuevos se asignaron a train o validation. Al materializar el experimento completo, 04_2 genera una partición nueva 70/15/15 del dataset integrado, siempre agrupada por `video_id`; por tanto ningún video cruza entre train, validation y test.

Se procesaron dos cohortes consecutivas:

1. **Ampliación general dirigida:** canales con rendimiento histórico de daño y búsquedas temáticas.
2. **Ampliación específica de amenazas:** 16 consultas de búsqueda, sin canales prefijados, hasta superar el objetivo aproximado de 5.300 chunks.

Antes de cada selección se excluyeron todos los `video_id` procesados o reservados en el corpus histórico y en cualquier ampliación previa. Después del chunking se deduplicó globalmente por ID y hash normalizado del texto.

## 2. Flujo de decisión Flash → Pro → humano

Cada chunk sigue una ruta jerárquica:

1. `deepseek-v4-flash` asigna etiquetas finas, flags transversales, `score_confianza` y `needs_review` usando el prompt operativo versionado.
2. Se llama a `deepseek-v4-pro` si Flash detecta cualquier daño, declara `needs_review=True`, tiene `score_confianza < 0.90` o el chunk pertenece al control aleatorio reproducible del 10% de seguros confiables.
3. Si Pro conserva `needs_review=True`, el chunk queda fuera del entrenamiento y se anexa al servidor humano.
4. El humano puede aceptar la decisión Pro, modificar las categorías gruesas/flags o rechazar y excluir el chunk.

Las etiquetas finas se conservan solo como fundamento y referencia; **el entrenamiento usa únicamente las cinco categorías gruesas de daño**. `SEGURO` es el negativo compartido. Los flags `ironia_ambigua`, `humor_encubridor` y `contexto_necesario` permanecen como variables transversales de enrutamiento y no se convierten en categorías base.

## 3. Resultados por lote

| Indicador | Ampliación general | Amenazas específicas | Total nuevo |
|---|---:|---:|---:|
| Videos seleccionados | 600 | 360 | 960 |
| Videos con subtítulos útiles | 548 | 353 | 901 |
| Videos con chunks | 547 | 351 | 898 |
| Chunks nuevos deduplicados | 19.971 | 5.451 | 25.422 |
| Chunks enviados a Pro | 4.105 | 921 | 5.026 |
| Resueltos automáticamente por Pro | 3.017 | 813 | 3.830 |
| Enviados a humano | 1.088 | 108 | 1.196 |
| Chunks utilizables sin esperar a humano | 18.883 | 5.343 | 24.226 |
| Costo Flash (USD) | 0,94* | 0,31 | 1,25* |
| Costo Pro (USD) | 1,27 | 0,27 | 1,55 |
| Costo API total (USD) | 2,22* | 0,58 | 2,79* |

\* La primera sesión Flash del lote general terminó después de escribir 18.505 etiquetas y antes de consolidar su telemetría de tokens. Se conservaron las etiquetas y el costo faltante se calibró con el costo por fila de la corrida completa anterior; por ello USD 0,94 y los totales derivados son estimaciones, no facturación exacta. El manifiesto registra 1.466 filas con telemetría directa, 18.505 calibradas y el método utilizado.

Sobre los **25.422 chunks nuevos**, 20.396 (80,23%) quedaron con Flash sin requerir Pro, 3.830 (15,07%) fueron resueltos por Pro y 1.196 (4,70%) requieren humano. Por tanto, el 95,30% del incremento es utilizable sin esperar la adjudicación humana.

## 4. Incremento de categorías gruesas

| Categoría de daño | Antes del lote 2 | Agregado lote 2 | Agregado lote 3 | Total final utilizable | Cambio |
|---|---:|---:|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 1.098 | 24 | 5 | 1.127 | +2,6% |
| `ACOSO_GENERO_IDENTIDAD` | 1.027 | 90 | 22 | 1.139 | +10,9% |
| `ACOSO_PERSONAL` | 1.515 | 227 | 150 | 1.892 | +24,9% |
| `AMENAZA_DIRECTA` | 300 | 100 | 128 | 528 | +76,0% |
| `CONTENIDO_SEXUAL` | 1.054 | 127 | 3 | 1.184 | +12,3% |

Cuatro de las cinco categorías superan 1.000 incidencias limpias. `AMENAZA_DIRECTA` continúa siendo la categoría limitante, pero aumentó de 300 a 528. La razón entre la categoría de daño mayor y la menor se redujo de 5,05:1 después del corpus previo a 3,58:1 al cierre.

En el lote específico se obtuvieron 128 amenazas Pro resueltas en 351 videos con chunks: **0,365 amenazas por video**. Un bootstrap no paramétrico por video con 10.000 remuestras dio IC 95% [0,265; 0,476]. Manteniendo la mezcla de consultas, cubrir las 472 restantes requeriría aproximadamente 1.295 videos en la estimación puntual o 1.782 usando el límite inferior como planificación conservadora. Esta extrapolación no es una garantía y depende de disponibilidad de subtítulos y estabilidad del rendimiento de búsqueda.

## 5. Dataset integrado disponible para 04_2

El cuaderno `04_2_entrenamiento_transformers_gruesos.ipynb` descubre automáticamente todos los archivos `datos/ampliacion/*/processed/dataset_etiquetado_utilizable.jsonl`. Antes de aceptar cada lote verifica manifiesto, SHA-256, esquema, unicidad de `chunk_id` y ausencia de `needs_review=True`.

| Estadística después de unificar | Valor |
|---|---:|
| Chunks crudos en las cuatro campañas | 117.266 |
| Chunks utilizables únicos | 114.200 |
| Videos únicos utilizables | 3.226 |
| Chunks `SEGURO` | 109.605 |
| Chunks con al menos un daño | 4.595 |
| Incidencias de etiquetas de daño | 5.870 |
| Chunks con más de un daño | 1.187 |
| Muestra después del submuestreo 4:1 de `SEGURO` | 22.975 |
| Train / validation / test | 16.076 / 3.449 / 3.450 |
| Solapamiento de videos entre particiones | 0 |

La estadística descriptiva y el gráfico reproducible se añadieron inmediatamente después de la integración en 04_2. La figura vigente es `resultados/figuras/transformer_grueso/descriptiva_dataset_unificado_categorias_gruesas.png`.

## 6. Servidor de validación humana

El servidor está activo en `http://127.0.0.1:8765` y descubre automáticamente todas las colas `pendientes_revision_humana.jsonl`. La actualización es append-only: conserva primero el orden de cohortes ya publicado y anexa los lotes nuevos al final.

| Cohorte humana | Casos | Resueltos | Pendientes |
|---|---:|---:|---:|
| Corrida original | 139 | 51 | 88 |
| Primera ampliación | 1.779 | 0 | 1.779 |
| Ampliación general del 27/07 | 1.088 | 0 | 1.088 |
| Amenazas específicas | 108 | 0 | 108 |
| **Total** | **3.114** | **51** | **3.063** |

Las 51 decisiones previas se conservaron con el mismo SHA-256 del archivo de progreso (`cd05878518ba36e1763462aa12f6b65bdedfceddb07aba06104b26f16d2a4a9b`), revisión 161 y cero IDs ausentes en la nueva campaña. La cola humana completa equivale al 2,66% de los 117.266 chunks crudos acumulados. Esta proporción describe enrutamiento operativo, no una tasa de error del modelo.

Mostrar la propuesta Pro antes de la decisión acelera la tarea, pero puede inducir anclaje. Estas decisiones son apropiadas para depurar el entrenamiento; una estimación independiente del error de Pro requiere una submuestra ciega y, preferentemente, doble anotación para calcular acuerdo entre codificadores.

## 7. Reproducibilidad

Lote general:

```powershell
$env:AMPLIACION_BATCH_ID='ampliacion_dano_20260727_lote2'
$env:AMPLIACION_SEED='27072026'
$env:AMPLIACION_MAX_VIDEOS='600'
$env:AMPLIACION_MIN_VIDEOS='600'
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage discover
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage transcribe
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage chunk
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage flash
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage pro
python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage prepare
```

Lote de amenazas:

```powershell
$env:AMPLIACION_BATCH_ID='ampliacion_amenaza_20260727_lote3'
$env:AMPLIACION_SEED='27072027'
$env:AMPLIACION_MAX_VIDEOS='360'
$env:AMPLIACION_MIN_VIDEOS='360'
$env:AMPLIACION_MAX_SEARCH_RESULTS='100'
$env:AMPLIACION_DISCOVERY_MODE='threat_search'
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage discover
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage transcribe
python -m scripts_auxiliares.ampliacion_dirigida_dano --stage chunk
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage flash
python -m scripts_auxiliares.etiquetar_ampliacion_dano --stage pro
python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage prepare
```

No se ejecutó `--stage train`. El siguiente paso autorizado es correr manualmente 04_2 desde sus primeras celdas, que volverá a materializar la integración antes del entrenamiento.

Informes y manifiestos detallados:

- `resultados/INFORME_AMPLIACION_DANO_20260727_LOTE2.md`.
- `resultados/INFORME_AMPLIACION_AMENAZA_20260727_LOTE3.md`.
- `resultados/INFORME_ADJUDICACION_HUMANA_COMBINADA_1918.md` (nombre histórico conservado para compatibilidad; contenido dinámico de 3.114 casos).
- `datos/ampliacion/ampliacion_dano_20260727_lote2/processed/dataset_etiquetado_utilizable.manifest.json`.
- `datos/ampliacion/ampliacion_amenaza_20260727_lote3/processed/dataset_etiquetado_utilizable.manifest.json`.
- `datos/model_ready/transformer_grueso/dataset_integrado_todas_pasadas.manifest.json`.

## 8. Limitaciones

- El muestreo dirigido modifica la distribución de clases y fuentes; no permite inferir prevalencia poblacional.
- Los casos difíciles se excluyen hasta adjudicación humana, lo que mejora pureza pero reduce temporalmente la cobertura de ejemplos fronterizos.
- El control aleatorio seguro del 10% permite auditar el enrutamiento, pero no sustituye un diseño ciego probabilístico independiente.
- Los conteos multietiqueta son incidencias: un chunk puede pertenecer a más de una categoría de daño.
- El test histórico se mantiene sin nuevos videos, aunque ya fue consultado en experimentos anteriores; para una afirmación final de producción conviene un holdout humano nuevo.

## 9. Referencias (APA 7)

Artstein, R., & Poesio, M. (2008). Inter-coder agreement for computational linguistics. *Computational Linguistics, 34*(4), 555–596. https://doi.org/10.1162/coli.07-034-R2

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Fairstein, Y., Kalinsky, O., Karnin, Z., Kushilevitz, G., Libov, A., & Tolmach, S. (2024). Class balancing for efficient active learning in imbalanced datasets. In *Proceedings of the 18th Linguistic Annotation Workshop* (pp. 77–86). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.law-1.8

Fithian, W., & Hastie, T. (2014). Local case-control sampling: Efficient subsampling in imbalanced data sets. *The Annals of Statistics, 42*(5), 1693–1724. https://doi.org/10.1214/14-AOS1220

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Scikit-learn developers. (2026). *GroupShuffleSplit*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
