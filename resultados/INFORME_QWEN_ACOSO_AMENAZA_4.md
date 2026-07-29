# Qwen3-0.6B LoRA con ACOSO_AMENAZA

Fecha: 2026-07-29T14:34:58+00:00

## Diseño

Se ajustó `Qwen/Qwen3-0.6B-Base` con LoRA para cuatro objetivos operativos de daño: RACISMO_DISCRIMINACION, ACOSO_GENERO_IDENTIDAD, ACOSO_AMENAZA, CONTENIDO_SEXUAL. `ACOSO_AMENAZA` es la unión reproducible de `ACOSO_PERSONAL` y `AMENAZA_DIRECTA`; `SEGURO` se deriva cuando ninguna salida supera su umbral.

Como regularización multitararea se añadieron cabezas auxiliares para 14 etiquetas finas y 3 flags transversales, con pesos de pérdida 0.20 y 0.15. Las etiquetas finas ausentes se enmascararon: no se interpretaron como negativos. Estas cabezas no agregan categorías de moderación ni se usan como entradas o reglas en inferencia; las únicas salidas operativas continúan siendo los cuatro daños primarios y `SEGURO` derivada.

El prompt operativo se conservó únicamente como procedencia de la taxonomía y no se introdujo en el texto de entrenamiento. Esto evita depender en producción de información que no estará disponible junto con cada chunk.

El dataset y splits 4:1 son los congelados por `04_2` (`df2ac01183271e44b6dcfb9cb4850bd6b1ef1cd11d9fc51c881be944670ef20f`). Se guardó un checkpoint reanudable cada 250 pasos de optimizador, alternando dos slots verificables, además de `last_adapter` y `best_adapter` por época.

Las probabilidades finales se calibraron por etiqueta mediante regresión sigmoide sobre logits de validación. Los umbrales también se fijaron en validación; test no intervino en entrenamiento, selección, calibración ni umbrales.

## Resultado en test

- PR-AUC macro de daño: 0.5520.
- F1 macro de daño: 0.5346.
- Precisión de cualquier daño: 0.6125.
- Recall de cualquier daño: 0.6849.
- Daños clasificados como seguro: 328.

| Categoría | Recall test |
|---|---:|
| RACISMO_DISCRIMINACION | 0.5977 |
| ACOSO_GENERO_IDENTIDAD | 0.6160 |
| ACOSO_AMENAZA | 0.5054 |
| CONTENIDO_SEXUAL | 0.5642 |

## Comparación sobre el mismo test

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
| Qwen3-0.6B LoRA · entrenado directamente con 4 etiquetas | 0.5520 | 0.5346 | 0.6849 | 328 |
| SVM plano 04_2 · unión post hoc a 4 etiquetas | 0.4605 | 0.4783 | 0.6254 | 390 |

## Operación selectiva

La alerta calibrada para 95% de recall en validación obtuvo en test recall 0.9721, tasa de revisión 0.6336, VPN 0.9850 y 29 falsos negativos automáticos. Alerta respaldada por la puerta declarada: **no**.

No se autoriza autonomía sin gold standard humano independiente, prevalencia natural y piloto prospectivo.

## Referencias (APA 7)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Qwen Team. (2025). Qwen3 technical report. *arXiv*. https://doi.org/10.48550/arXiv.2505.09388

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv*. https://doi.org/10.48550/arXiv.1706.05098
