# Informe del fine-tuning Transformer para categorías gruesas

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


> **Informe histórico del contrato de cinco etiquetas.** Se conserva para reproducibilidad y no representa la selección activa de cuatro daños.

Fecha de ejecución: 2026-07-27T15:49:08-05:00  
Instantánea humana: `datos\etiquetado\humano\snapshots_entrenamiento\revision_humana_final_r3226_db88b85c6d3a.json`  
SHA-256 de la instantánea: `db88b85c6d3a5c4da1d1e4e6fb32adceeec61b04683c3ac4a80efb6b539a9815`

## Resumen

Primero se reentrenaron seis baselines clásicos y después se compararon dos encoders compactos mediante fine-tuning completo. El ganador Transformer se fijó exclusivamente por PR-AUC macro de daño en validación: **Multilingual E5-small (linaje MiniLM)**. El test no intervino en el ajuste de modelos, umbrales ni selección.

El Transformer seleccionado obtiene una PR-AUC puntual mayor, pero el intervalo bootstrap no permite afirmar todavía una mejora estable frente al mejor clásico.

Esta evaluación no autoriza moderación autónoma. La aceptabilidad operativa depende también de recall, falsos negativos, calibración y revisión humana por categoría.

## Datos y objetivos

- Unión útil antes de balancear: 117,244 chunks; 7,063 con daño (6.02 %) y 110,181 seguros.
- Muestra balanceada antes de dividir: 35,315 chunks; 7,063 con daño y 28,252 seguros.
- Entrenamiento: 24,701 chunks y 2,086 videos.
- Validación: 5,324 chunks y 448 videos.
- Test: 5,290 chunks y 448 videos.
- Objetivos: RACISMO_DISCRIMINACION, ACOSO_GENERO_IDENTIDAD, ACOSO_PERSONAL, AMENAZA_DIRECTA, CONTENIDO_SEXUAL. `SEGURO` se deriva si no se activa daño.
- Etiquetas finas entrenadas: no. Flags transversales entrenados como categorías: no.
- Fuga de videos entre particiones: 0.

## Balanceo reproducible

Se conservaron los 7,063 chunks únicos con daño y se seleccionaron por SHA-256 4 seguros por cada chunk con daño. Solo después se aplicó la partición aleatoria agrupada por video 70/15/15. La muestra efectiva contiene 35,315 filas: 20 % con algún daño y 80 % `SEGURO` en el conjunto global. Al haber balanceado antes de dividir, validación y test miden comparación controlada bajo esa prevalencia; no estiman directamente el valor predictivo en la prevalencia natural de producción.

Para cada encoder, un linear probe sobre la validación eligió entre BCE normal y una ponderación positiva moderada `sqrt(N_neg/N_pos)`. Este paso no consultó el test.

## Modelos clásicos antes del fine-tuning

Los cinco modelos de los cuadernos 04/04_1 y **fastText supervisado OVA** se ejecutaron primero con configuraciones iniciales fijadas para un screening sin acceso a test. Las tres mejores familias no triviales por PR-AUC macro de daño en validación pasaron a una búsqueda acotada de ocho configuraciones cada una. Cada configuración se comparó con 3 folds de `GroupKFold` dentro de train, agrupando por `video_id`; por tanto ningún video estuvo a ambos lados de un fold.

Después del CV, cada familia seleccionada se reentrenó con los 24,701 chunks completos de entrenamiento. Los umbrales se calibraron en validación, el ganador se congeló por PR-AUC macro de daño y recién entonces se evaluó test. Los mejores parámetros fueron `{'linear_svm_word_char': {'C': 0.25, 'min_df': 1, 'max_features': 50000}, 'logistic_regression': {'C': 2.0, 'min_df': 2, 'max_features': 50000}, 'fasttext_supervised_ova': {'lr': 0.5, 'epoch': 15, 'wordNgrams': 2, 'bucket': 200000, 'dim': 50, 'loss': 'ova'}}`. fastText procede de `PLN_clases/clase4/Cuadernos/nlp_sesion4_1_FastText_Intro.ipynb` y de la receta OVA oficial; a diferencia de los modelos scikit-learn, no admite los pesos por observación, limitación conservada en la comparación.

| Modelo clásico | PR-AUC validación | PR-AUC test | F1 macro test | Recall micro test |
|---|---:|---:|---:|---:|
| SVM lineal palabra+carácter | 0.4612 | 0.4174 | 0.4517 | 0.4850 |

| Regresión logística | 0.4292 | 0.3994 | 0.4159 | 0.4700 |

| fastText supervisado OVA (sesión 4) | 0.3419 | 0.3240 | 0.3444 | 0.3400 |

El mejor clásico seleccionado en validación fue **SVM lineal palabra+carácter** (`tuned__linear_svm_word_char`).

## Configuración

- Longitud máxima común: 128 tokens.
- Batch de entrenamiento: 8; batch de evaluación: 32.
- Optimizador: AdamW; learning rate 2e-05; weight decay 0.01.
- Máximo: 3 épocas; parada temprana con paciencia 1.
- Criterio: PR-AUC macro de las cinco categorías de daño en validación.
- Semilla: 20260727.
- Hardware de esta ejecución: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD, PyTorch 2.6.0+cpu, dispositivo CPU.

## Resultados

| Modelo | Mejor época | PR-AUC validación | PR-AUC test | F1 macro test | Recall micro test | Precisión algún daño | Recall algún daño |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paraphrase Multilingual MiniLM-L12 | 2 | 0.5024 | 0.4576 | 0.4600 | 0.4993 | 0.5792 | 0.6427 |

| Multilingual E5-small (linaje MiniLM) | 3 | 0.5082 | 0.4399 | 0.4730 | 0.5107 | 0.5969 | 0.6359 |

El ganador Transformer por validación fue `e5_small`. Su diferencia de PR-AUC macro de daño frente al mejor clásico en test fue +0.0225. El bootstrap pareado de 1,000 réplicas, remuestreando los 448 videos como conglomerados, produjo IC 95 % percentil [-0.0102, +0.0546]. Este intervalo cuantifica variación muestral entre videos, no variación entre semillas de entrenamiento.

## Interpretación

La reducción de `SEGURO` disminuye el tiempo de entrenamiento y expone con más frecuencia los positivos, pero cambia la prevalencia de los tres subconjuntos. Un mejor F1 acompañado de pérdida fuerte de precisión no se interpreta automáticamente como una mejora operativa. Antes de desplegar se requiere una evaluación adicional con prevalencia natural y revisión humana de falsos negativos.

La comparación usa una sola semilla por modelo. Para reporte académico definitivo se recomienda repetir ambos fine-tunings con al menos tres semillas y reportar media, desviación e intervalos; el bootstrap actual no reemplaza esa estimación de variabilidad de optimización.

## Figuras y artefactos

- `resultados/figuras/transformer_grueso/balance_y_comparacion_transformers.png`.
- `resultados/figuras/transformer_grueso/curvas_validacion_transformers.png`.
- `resultados/figuras/transformer_grueso/comparacion_modelos_clasicos_antes_transformers.png`.
- `resultados/metricas/transformer_grueso/` contiene auditoría, curvas, scores y reportes por clase.
- `modelos/moderador_transformer_grueso/` contiene checkpoints y tokenizadores.
- `resultados/logs/transformer_grueso/progreso.jsonl` conserva el progreso temporal.

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Huang, Y., Giledereli, B., Köksal, A., Özgür, A., & Ozkirimli, E. (2021). Balancing methods for multi-label text classification with long-tailed class distribution. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 8153–8161). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.643

Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). Bag of tricks for efficient text classification. In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers* (pp. 427–431). Association for Computational Linguistics. https://aclanthology.org/E17-2068/

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. *arXiv*. https://doi.org/10.48550/arXiv.2402.05672

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained Transformers. In *Advances in Neural Information Processing Systems* (Vol. 33). https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
