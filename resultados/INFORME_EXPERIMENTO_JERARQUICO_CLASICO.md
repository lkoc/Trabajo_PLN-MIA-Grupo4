# Experimento jerárquico clásico con SEGURO ampliado

> **Informe histórico del contrato de cinco etiquetas.** La comparación activa está en `INFORME_EXPERIMENTO_JERARQUICO_CLASICO_4.md` y en la matriz de cuatro daños.

Fecha: 2026-07-27T13:41:32-05:00

## Diseño y datos

El mapa `train/validation/test` de `04_2` se propagó por `video_id` al dataset integrado completo. Se utilizaron **116,313 chunks**, incluidos **109,250 SEGURO** y **7,063 con daño**. Train contiene 75,943 seguros y 4,944 daños. Se excluyeron 931 seguros de 248 videos sin asignación; no se perdió ningún daño.

Se reutilizaron los hiperparámetros ganadores de `04_2`: SVM lineal palabra+carácter (`C=0,25`, `min_df=1`, 50.000 features) y regresión logística (`C=2`, `min_df=2`, 50.000 features). Cada familia compara, sobre el mismo TF-IDF y datos, un modelo plano, una cascada y una jerarquía de cabezas compartidas. fastText se reentrenó como referencia plana adicional.

Los márgenes de SVM y logística se calibraron con regresión sigmoide sobre predicciones out-of-fold de tres `GroupKFold` por video. Épocas no aplican a estos optimizadores convexos: cada ajuste converge según su tolerancia o `max_iter`. Umbrales, selección del ganador y abstención usan sólo validación.

## Resultados en test ampliado

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
| SVM lineal calibrado palabra+carácter · Cascada binaria → multietiqueta | 0.2352 | 0.2910 | 0.3612 | 665 |
| SVM lineal calibrado palabra+carácter · Plano | 0.2306 | 0.2929 | 0.4544 | 568 |
| Regresión logística calibrada · Cascada binaria → multietiqueta | 0.2303 | 0.2690 | 0.3631 | 663 |
| SVM lineal calibrado palabra+carácter · Jerárquico clásico con TF-IDF compartido | 0.2287 | 0.2883 | 0.3900 | 635 |
| Regresión logística calibrada · Plano | 0.2282 | 0.2914 | 0.4179 | 606 |
| Regresión logística calibrada · Jerárquico clásico con TF-IDF compartido | 0.2225 | 0.2691 | 0.3650 | 661 |
| fastText supervisado OVA · Plano | 0.1672 | 0.2339 | 0.2757 | 754 |

## Diferencias pareadas frente al plano de la misma familia

| Candidato | Métrica | Δ jerárquico − plano | IC 95 % por video |
|---|---|---:|---:|
| svm__cascade | pr_auc_macro | +0.0046 | [-0.0061, +0.0152] |
| svm__cascade | f1_macro | -0.0019 | [-0.0198, +0.0150] |
| svm__cascade | any_damage_recall | -0.0932 | [-0.1175, -0.0698] |
| svm__cascade | false_negative_rate | +0.0932 | [+0.0698, +0.1175] |
| svm__shared_hierarchy | pr_auc_macro | -0.0019 | [-0.0110, +0.0090] |
| svm__shared_hierarchy | f1_macro | -0.0046 | [-0.0186, +0.0093] |
| svm__shared_hierarchy | any_damage_recall | -0.0644 | [-0.0813, -0.0451] |
| svm__shared_hierarchy | false_negative_rate | +0.0644 | [+0.0451, +0.0813] |
| logistic__cascade | pr_auc_macro | +0.0022 | [-0.0092, +0.0109] |
| logistic__cascade | f1_macro | -0.0224 | [-0.0394, -0.0071] |
| logistic__cascade | any_damage_recall | -0.0548 | [-0.0786, -0.0308] |
| logistic__cascade | false_negative_rate | +0.0548 | [+0.0308, +0.0786] |
| logistic__shared_hierarchy | pr_auc_macro | -0.0057 | [-0.0167, +0.0058] |
| logistic__shared_hierarchy | f1_macro | -0.0223 | [-0.0369, -0.0077] |
| logistic__shared_hierarchy | any_damage_recall | -0.0528 | [-0.0728, -0.0331] |
| logistic__shared_hierarchy | false_negative_rate | +0.0528 | [+0.0331, +0.0728] |

El ganador se fijó con validación: **SVM lineal calibrado palabra+carácter · Cascada binaria → multietiqueta**. Decisión frente a su plano pareado: **`diferencia_inconclusa_con_este_test`**. Comparación secundaria frente al SVM histórico de `04_2` sobre el mismo test 4:1: **`modelo_plano_superior_en_pr_auc_macro`**.

No se autoriza moderación autónoma: el test sigue siendo retrospectivo, con etiquetas mayormente asistidas por LLM y sin prevalencia prospectiva de producción.

## Artefactos

- Resultado: `resultados\metricas\jerarquico_clasico\resultado.json`
- Comparación: `resultados\metricas\jerarquico_clasico\comparacion_modelos.csv`
- Bootstrap: `resultados\metricas\jerarquico_clasico\bootstrap_pareado_por_video.csv`
- Modelos: `modelos\jerarquico_clasico`
- Figuras: `resultados\figuras\jerarquico_clasico`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers* (pp. 427–431). Association for Computational Linguistics. https://aclanthology.org/E17-2068/

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
