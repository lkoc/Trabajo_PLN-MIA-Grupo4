# Modelos clásicos planos y jerárquicos con cuatro daños

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha: 2026-07-28T08:21:46-05:00

## Datos y diseño

Todos los experimentos proceden del dataset integrado `datos\model_ready\transformer_grueso\dataset_integrado_todas_pasadas.jsonl` con hash `3f01b76d285d4cdd2a0922df1d2437a7f01abc4e05e2699c218b1f5faaba2069`. Contiene 110,181 chunks SEGURO y 7,063 con daño. Train utiliza 76,874 SEGURO y 4,944 daños; ningún video de validation/test entra en train.

Se entrenan SVM lineal y regresión logística, cada uno como modelo plano, cascada binaria y jerarquía probabilística compartida. Las cuatro etiquetas son RACISMO_DISCRIMINACION, ACOSO_GENERO_IDENTIDAD, ACOSO_AMENAZA, CONTENIDO_SEXUAL. Los hiperparámetros provienen de la búsqueda de `04_2`; la calibración se realiza out-of-fold por video y los umbrales se fijan en validation. fastText queda fuera de esta variante porque su artefacto histórico codifica cinco salidas y no admite transferencia exacta de la cabeza.

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
| SVM lineal calibrado palabra+carácter · Cascada binaria → multietiqueta | 0.2631 | 0.3158 | 0.3506 | 676 |
| SVM lineal calibrado palabra+carácter · Plano | 0.2608 | 0.3200 | 0.4275 | 596 |
| SVM lineal calibrado palabra+carácter · Jerárquico clásico con TF-IDF compartido | 0.2584 | 0.3185 | 0.3881 | 637 |
| Regresión logística calibrada · Cascada binaria → multietiqueta | 0.2529 | 0.2966 | 0.3718 | 654 |
| Regresión logística calibrada · Plano | 0.2513 | 0.3047 | 0.4150 | 609 |
| Regresión logística calibrada · Jerárquico clásico con TF-IDF compartido | 0.2474 | 0.2883 | 0.3324 | 695 |

## Comparación cruzada sobre el test 4:1 común

| Modelo | PR-AUC macro | F1 macro | Recall daño |
|---|---:|---:|---:|
| SVM lineal calibrado palabra+carácter · Plano | 0.4508 | 0.4578 | 0.5889 |
| SVM lineal calibrado palabra+carácter · Cascada binaria → multietiqueta | 0.4437 | 0.4523 | 0.5379 |
| Regresión logística calibrada · Plano | 0.4370 | 0.4607 | 0.6206 |
| SVM lineal calibrado palabra+carácter · Jerárquico clásico con TF-IDF compartido | 0.4285 | 0.4219 | 0.4323 |
| Regresión logística calibrada · Cascada binaria → multietiqueta | 0.4274 | 0.4270 | 0.5860 |
| Regresión logística calibrada · Jerárquico clásico con TF-IDF compartido | 0.4081 | 0.4132 | 0.4476 |

Ganador por validation: **SVM lineal calibrado palabra+carácter · Cascada binaria → multietiqueta**. Decisión pareada frente a su plano: `diferencia_inconclusa_con_este_test`. No se autoriza moderación autónoma sin gold standard humano independiente y piloto prospectivo.

## Artefactos

- Resultado: `resultados/metricas/jerarquico_clasico_4/resultado.json`
- Modelos: `modelos/jerarquico_clasico_4`
- Comparación: `resultados/metricas/jerarquico_clasico_4/comparacion_modelos.csv`
- Bootstrap: `resultados/metricas/jerarquico_clasico_4/bootstrap_pareado_por_video.csv`

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
