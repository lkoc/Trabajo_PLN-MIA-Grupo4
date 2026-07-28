# Informe del entrenamiento incremental con ampliación de videos

Fecha: 2026-07-27T05:54:02-05:00  
Lote: `ampliacion_dano_20260726`  
Comparación que aísla el incremento de videos: `ampliado_sin_aeda`  
Modelo exportado: `ampliado_con_aeda`

## 1. Pregunta y diseño experimental

Se evaluó cuánto cambia el desempeño del moderador grueso al añadir videos adquiridos de forma dirigida hacia categorías minoritarias. La comparación mantiene el mismo algoritmo principal, la misma taxonomía gruesa y el mismo test histórico agrupado por video. El baseline y las variantes ampliadas difieren en la incorporación de los nuevos chunks; una variante añade además AEDA solo al entrenamiento.

La selección de configuración se realiza con validación histórica. El test congelado se usa para una comparación de ingeniería y no para ajustar hiperparámetros. Las etiquetas finas no se entrenan; los flags transversales siguen separados de las cinco categorías de daño y `SEGURO`.

## 2. Instantánea humana e inclusión de datos

La validación humana no tuvo que terminar. Se congeló `datos\etiquetado\humano\snapshots_entrenamiento\revision_humana_r161_cd05878518ba.json` (revisión 161, SHA-256 `cd05878518ba36e1763462aa12f6b65bdedfceddb07aba06104b26f16d2a4a9b`). Contenía 51 decisiones cerradas: 48 incluidas y 3 excluidas.

- Decisiones anteriores preservadas: 26.
- Propuestas LLM aceptadas: 18.
- Propuestas modificadas por humano: 4.
- Rechazos excluidos: 3.
- Todo caso todavía abierto se excluyó de esta corrida; continuar validando no altera retrospectivamente la instantánea.

## 3. Incremento del corpus

La adquisición produjo 21,991 chunks de 476 videos nuevos. Fueron utilizables 20,212: 16,808 pseudoetiquetas Flash y 3,404 decisiones Pro cerradas. Los 1,779 casos Pro aún dudosos se excluyeron.

Partición nueva por `video_id`: 16,168 train, 4,044 validation y 0 test. El test histórico conserva 10.293 filas.

| Categoría de daño | Antes | Nuevos utilizables | Después |
|---|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 1,054 | 44 | 1,098 |
| `ACOSO_GENERO_IDENTIDAD` | 985 | 42 | 1,027 |
| `ACOSO_PERSONAL` | 1,238 | 277 | 1,515 |
| `AMENAZA_DIRECTA` | 221 | 79 | 300 |
| `CONTENIDO_SEXUAL` | 928 | 126 | 1,054 |

## 4. Configuraciones entrenadas

1. `baseline_reproducido`: SVM lineal word+character con corpus histórico depurado.
2. `ampliado_sin_aeda`: mismo modelo más los chunks nuevos utilizables.
3. `ampliado_con_aeda`: corpus ampliado y una copia AEDA de cada ejemplo de daño, con peso reducido.

Se ajustan umbrales con validación; no se describen épocas porque una SVM lineal converge por optimización, no por pasadas neuronales o epochs.

## 5. Resultados cuantitativos

Comparación principal para atribución causal operativa: baseline frente a `ampliado_sin_aeda` (sin cambiar augmentation). La configuración `ampliado_con_aeda` se seleccionó por PR-AUC de daño en validación histórica y se informa por separado en la tabla completa.

| Métrica | Baseline | Ampliado | Δ absoluto | Δ relativo |
|---|---:|---:|---:|---:|
| PR-AUC macro de daño · validación histórica | 0.2264 | 0.2261 | -0.0003 | -0.15% |
| F1 macro de daño · validación histórica | 0.2928 | 0.2906 | -0.0023 | -0.77% |
| PR-AUC macro de daño · test congelado | 0.2265 | 0.2211 | -0.0054 | -2.37% |
| F1 macro de daño · test congelado | 0.2564 | 0.2444 | -0.0120 | -4.66% |
| Recall micro de daño · test congelado | 0.2693 | 0.2409 | -0.0283 | -10.53% |
| Exact match · test congelado | 0.9291 | 0.9336 | +0.0046 | +0.49% |
| PR-AUC macro de daño · validación nueva | 0.0903 | 0.1059 | +0.0157 | +17.37% |
| F1 macro de daño · validación nueva | 0.1040 | 0.1411 | +0.0371 | +35.69% |
| Recall micro de daño · validación nueva | 0.2105 | 0.2719 | +0.0614 | +29.17% |

Resultados completos de los tres modelos:

| Modelo | PR-AUC daño validación | F1 daño test | PR-AUC daño test | Recall daño test | Filas train |
|---|---:|---:|---:|---:|---:|
| `ampliado_con_aeda` | 0.2319 | 0.2487 | 0.2202 | 0.2457 | 67,900 |
| `baseline_reproducido` | 0.2264 | 0.2564 | 0.2265 | 0.2693 | 48,854 |
| `ampliado_sin_aeda` | 0.2261 | 0.2444 | 0.2211 | 0.2409 | 65,022 |

### Cambio cualitativo por categoría

| Categoría | F1 baseline | F1 ampliado | Δ |
|---|---:|---:|---:|
| `CONTENIDO_SEXUAL` | 0.3383 | 0.3538 | +0.0155 |
| `ACOSO_GENERO_IDENTIDAD` | 0.3301 | 0.3319 | +0.0018 |
| `AMENAZA_DIRECTA` | 0.0690 | 0.0690 | +0.0000 |
| `RACISMO_DISCRIMINACION` | 0.2969 | 0.2628 | -0.0342 |
| `ACOSO_PERSONAL` | 0.2475 | 0.2046 | -0.0429 |

## 6. Interpretación

La ampliación no mejora de forma consistente validación, PR-AUC y F1 del test; el resultado debe describirse como mixto y no como una mejora general demostrada.

En concreto, el efecto aislado de añadir videos es -0.0003 en PR-AUC macro de daño de validación histórica, -0.0054 en PR-AUC macro de daño de test y -0.0120 en F1 macro de daño de test. En la validación del dominio nuevo, la PR-AUC de daño cambia +0.0157 y el F1 macro de daño +0.0371.

La variante seleccionada por validación histórica (`ampliado_con_aeda`) alcanzó PR-AUC de daño 0.2319 en validación y 0.2202 en test. Se conserva esa selección para no escoger retrospectivamente con el test, pero el test no respalda una mejora general.

La PR-AUC se prioriza porque las clases de daño son poco frecuentes. Una ganancia en conteo o accuracy global puede deberse a `SEGURO` y no prueba mejor detección de daño. Tampoco se interpreta esta comparación como suficiencia para moderación autónoma sin un holdout humano ciego y contemporáneo.


## 7. Aceptabilidad operativa

Sobre 10,293 chunks de test había 494 con al menos una categoría de daño. El modelo predijo algún daño en 513 casos y marcó `needs_review` en 653 (6.34%). La unión de predicción de daño o revisión interviene sobre 895 chunks (8.70% del test).

Esa política cubre solo 48.58% de los daños y deja 254/494 (51.42%) como seguros sin revisión. La precisión de predecir algún daño es 34.89%. Aunque el valor predictivo negativo de los auto-seguros es 97.30%, ese valor está dominado por la prevalencia baja de daño y oculta el gran número absoluto de falsos negativos.

| Categoría | Soporte | Recall de categoría | Cobertura por daño o revisión | Omitidos como auto-seguros |
|---|---:|---:|---:|---:|
| `RACISMO_DISCRIMINACION` | 141 | 29.08% | 50.35% | 70 |
| `ACOSO_GENERO_IDENTIDAD` | 137 | 39.42% | 58.39% | 57 |
| `ACOSO_PERSONAL` | 180 | 16.11% | 37.22% | 113 |
| `AMENAZA_DIRECTA` | 55 | 5.45% | 34.55% | 36 |
| `CONTENIDO_SEXUAL` | 122 | 35.25% | 64.75% | 43 |

**Decisión:** no es aceptable para moderación autónoma ni para aprobar automáticamente todo caso sin `needs_review`. Puede conservarse como baseline experimental, herramienta de priorización o asistente cuya decisión final siga siendo humana.

## 8. Limitaciones

- La adquisición fue dirigida; mejora representación para entrenamiento, pero no estima prevalencia natural.
- Los casos humanos abiertos fueron excluidos, por lo que ejemplos fronterizos permanecen subrepresentados.
- El test histórico ya fue observado en experimentos anteriores; la comparación es de ingeniería y puede tener sesgo adaptativo.
- Una sola partición no cuantifica variabilidad entre videos; futuras conclusiones fuertes requieren bootstrap por video o validación humana externa.

## 9. Reproducibilidad y artefactos

- Modelo: `modelos\moderador_grueso_ampliado\moderador_grueso_ampliado.joblib`; SHA-256 `5d56568f13343a5f0ed56bcca7f8f8be8ff427da67728e1d395a2398655522d9`.
- Dataset ampliado: `datos\ampliacion\ampliacion_dano_20260726\processed\dataset_etiquetado_utilizable.jsonl`; SHA-256 `6f05735c93c16b165707240946156651affa90d21a5a72578c66ce6a578e4b41`.
- Comparación: `resultados\metricas\ampliacion_dano\comparacion_entrenamiento_ampliado.csv`.
- Manifiesto: `datos\ampliacion\ampliacion_dano_20260726\processed\dataset_etiquetado_utilizable.manifest.json`.
- Figura: `resultados\figuras\ampliacion_dano\comparacion_entrenamiento.png`.
- Aceptabilidad operativa: `resultados\metricas\ampliacion_dano\aceptabilidad_operativa.json`.

Comando:

```powershell
python -m scripts_auxiliares.preparar_entrenamiento_ampliado --stage train
```

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Karimi, A., Rossi, L., & Prati, A. (2021). AEDA: An easier data augmentation technique for text classification. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 2748–2754). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-emnlp.234

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
