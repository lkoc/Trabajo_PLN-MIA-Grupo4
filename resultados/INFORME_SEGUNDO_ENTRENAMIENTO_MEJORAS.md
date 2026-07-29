# Informe del segundo intento: mejoras del clasificador grueso

> **Informe histórico del contrato de cinco etiquetas.** No corresponde a los cuadernos activos `04_201`–`04_208`.

**Fecha de ejecución:** 26 de julio de 2026  
**Cuaderno:** `Cuadernos/04_1_mejoras_entrenamiento_moderador.ipynb`  
**Baseline de comparación:** `resultados/INFORME_PRIMER_ENTRENAMIENTO_MODELOS_GRUESOS.md`

## 1. Restricción del objetivo

El segundo experimento mantuvo exactamente el mismo problema del baseline: cinco categorías gruesas de daño o `SEGURO`. No se entrenaron las 14 etiquetas finas ni los tres flags. Las etiquetas finas solo derivaron los objetivos gruesos; los flags solo se usaron para evaluar el enrutamiento de casos ambiguos.

## 2. Mejoras evaluadas

Se aplicaron y evaluaron las siguientes propuestas:

1. contexto formado por título, chunk anterior, chunk actual y chunk siguiente;
2. ajuste de la regularización `C` de SVM y regresión logística;
3. comparación de pesos base 0.50 y 0.25 para pseudoetiquetas Flash;
4. aumentación AEDA mediante inserción de puntuación únicamente en ejemplos de daño del entrenamiento;
5. umbrales de alta sensibilidad que maximizan precisión bajo la restricción de recall de validación ≥ 0.80;
6. minería de seguros Flash difíciles para revisión posterior;
7. especificación, sin ejecución, de una futura fase BETO en GPU.

La partición del baseline se reprodujo exactamente: 48,927 chunks de entrenamiento, 10,633 de validación y 10,293 de prueba, agrupados por video. El test fue el mismo para permitir comparación directa. Como ya había sido observado durante el baseline, esta fase es una comparación de ingeniería y no sustituye un nuevo holdout humano ciego.

## 3. Resultados de la búsqueda

La configuración seleccionada por PR-AUC macro de daño en validación fue la SVM baseline de texto actual, `C=1`, peso Flash 0.50, con una copia AEDA de cada ejemplo de daño. El contexto no fue seleccionado.

| Configuración | PR-AUC daño validación | F1 daño validación | PR-AUC daño prueba | F1 daño prueba | Recall micro daño prueba |
|---|---:|---:|---:|---:|---:|
| **SVM texto + AEDA** | **0.1860** | 0.2377 | 0.2283 | **0.2774** | 0.2709 |
| SVM texto baseline | 0.1815 | **0.2424** | **0.2317** | 0.2752 | 0.2787 |
| Logística contexto, `C=1`, Flash 0.50 | 0.1533 | 0.2343 | 0.1991 | 0.2441 | 0.3370 |
| Logística contexto, `C=1`, Flash 0.25 | 0.1509 | 0.2287 | 0.1957 | 0.2470 | **0.3575** |
| SVM contexto, `C=0.3`, Flash 0.25 | 0.1500 | 0.2241 | 0.2083 | 0.2338 | 0.3055 |
| SVM contexto, `C=0.3`, Flash 0.50 | 0.1487 | 0.2223 | 0.2070 | 0.2418 | 0.3512 |

Otras configuraciones contextuales obtuvieron resultados inferiores y se conservan en `resultados/metricas/moderador_grueso_mejorado/comparacion_mejoras_test.csv`.

### 3.1 Interpretación

AEDA mejoró el criterio de selección en validación, pero el cambio no se reprodujo de forma consistente en prueba:

- F1 macro de daño: 0.2752 → 0.2774, incremento absoluto de 0.0022.
- PR-AUC macro de daño: 0.2317 → 0.2283, disminución absoluta de 0.0034.
- Recall micro de daño: 0.2787 → 0.2709, disminución absoluta de 0.0078.

Por tanto, no existe evidencia práctica de una mejora general del ranking o la detección. El pequeño cambio de F1 puede corresponder a variación de selección y no justifica afirmar que AEDA resolvió el problema.

El contexto tampoco mejoró el resultado con TF-IDF: las mejores variantes contextuales quedaron por debajo del baseline y las SVM contextuales necesitaron aproximadamente 116–120 segundos, frente a 65 segundos del texto actual. Esto no demuestra que el contexto sea inútil; muestra que concatenarlo directamente a una representación TF-IDF agrega ruido en este corpus. Un modelo contextual neuronal podría aprovecharlo de otra manera.

Reducir el peso Flash a 0.25 elevó el recall en algunas configuraciones, pero redujo precisión y PR-AUC. El resultado es consistente con una frontera más agresiva que deriva más seguros hacia daño, no con una mejor separación intrínseca.

## 4. Política de alta sensibilidad

Se eligió por categoría el umbral de validación con mayor precisión sujeto a recall ≥ 0.80. En prueba se obtuvo:

| Categoría | Precisión | Recall | F1 |
|---|---:|---:|---:|
| Racismo/discriminación | 0.0548 | 0.7801 | 0.1024 |
| Acoso por género/identidad | 0.0550 | 0.8029 | 0.1029 |
| Acoso personal | 0.0417 | 0.9000 | 0.0796 |
| Amenaza directa | 0.0096 | 0.9636 | 0.0191 |
| Contenido sexual | 0.0661 | 0.8443 | 0.1226 |

El objetivo de recall se generalizó aproximadamente, aunque racismo quedó ligeramente por debajo de 0.80. La precisión es demasiado baja: solo entre 1% y 6.6% de las alertas de cada categoría serían verdaderas según las pseudoetiquetas de prueba.

Al combinar daño predicho e incertidumbre, el margen mínimo evaluado deriva 82.3% de los chunks a revisión para capturar 98.6% del daño y 98.5% de los flags. Esta política no es operacionalmente eficiente y no convierte al clasificador en un moderador autónomo.

## 5. Minería de casos difíciles

Se generó `datos/processed/flash_seguros_dificiles_para_revision.csv` con 2,000 chunks que Flash dejó como seguros pero que el modelo considera próximos a alguna de las cinco categorías gruesas de daño. La selección cubre 905 videos y limita cada video a tres chunks. Los scores máximos de daño se encuentran aproximadamente entre 0.342 y 0.758.

Este archivo es un manifiesto de revisión, no un nuevo conjunto etiquetado. Ninguna etiqueta fue modificada y el archivo de prueba fue excluido. Su revisión independiente por Pro o humanos es el siguiente experimento con mayor probabilidad de mejorar datos y revelar falsos negativos sistemáticos.

## 6. Transformer español

BETO es una extensión metodológicamente plausible porque fue preentrenado exclusivamente con datos en español (Cañete et al., 2020). No se ejecutó en esta fase: el entorno verificado tiene PyTorch 2.6 solo para CPU, no dispone de CUDA y no tenía `transformers` instalado. Reportar un resultado Transformer sin ejecutar el ajuste completo sería incorrecto.

Una futura ejecución con GPU debe mantener únicamente las cinco categorías gruesas o `SEGURO`, usar pérdida BCE ponderada o focal, comenzar con tres épocas, máximo cinco, y parada temprana sobre PR-AUC macro de daño. Las etiquetas finas y los flags no deben convertirse en salidas.

## 7. Conclusión

Las mejoras clásicas ejecutadas no elevaron el desempeño a un nivel suficiente. La variante AEDA fue seleccionada en validación, pero su ventaja en prueba fue mínima e inconsistente entre F1, PR-AUC y recall. El contexto concatenado empeoró el desempeño y aumentó el costo. Los umbrales de alto recall convierten el modelo en un filtro excesivamente amplio que requiere revisar más de ocho de cada diez chunks.

El sistema sigue siendo un priorizador experimental. La oportunidad de mejora principal ya no es aumentar iteraciones: es revisar los 2,000 negativos difíciles, obtener un holdout humano independiente y, con GPU, evaluar un encoder contextual en español. El baseline original debe conservarse como referencia principal hasta que una mejora supere de forma consistente F1, PR-AUC y recall en un conjunto humano ciego.

## 8. Referencias (APA 7)

Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). *Spanish pre-trained BERT model and evaluation data*. PML4DC at ICLR 2020. https://users.dcc.uchile.cl/~jperez/papers/pml4dc2020.pdf

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Feng, S. Y., Gangal, V., Wei, J., Chandar, S., Vosoughi, S., Mitamura, T., & Hovy, E. (2021). A survey of data augmentation approaches for NLP. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021* (pp. 968–988). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-acl.84

Karimi, A., Rossi, L., & Prati, A. (2021). AEDA: An easier data augmentation technique for text classification. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 2748–2754). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-emnlp.234

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2980–2988). https://openaccess.thecvf.com/content_ICCV_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Song, X., Petrak, J., & Roberts, A. (2018). A deep neural network sentence level classification method with context information. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 900–904). Association for Computational Linguistics. https://doi.org/10.18653/v1/D18-1107
