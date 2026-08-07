# Robustez neuronal de la longitud de chunks

**Fecha de corte:** 7 de agosto de 2026  
**Estado:** ejecución completa  
**Decisión primaria congelada:** conservar 30 s  
**Ámbito:** análisis confirmatorio de sensibilidad sobre `validation` y cierre
complementario fuera de pliegue sobre `train`; `test` no participa

El contrato evaluado tiene cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; MiniLM y Ollama agregan sus métricas de daño sobre las otras cuatro categorías multietiqueta.

Este informe documenta el perfil activado por `RUN_NEURAL_ROBUST_TEST` y el
contraste 20 s–30 s activado por
`RUN_MINILM_20_30_NONINFERIORITY_TEST`. La
selección principal procede del perfil clásico y permanece congelada antes de
consultar MiniLM u Ollama. Esta separación evita reutilizar la evaluación final
para favorecer una configuración [1].

Los artefactos de evidencia son el
[manifiesto del panel](../resultados/pilotos/chunk_length/neural_robust/paired_validation_panel_manifest.json),
el [resultado MiniLM](../resultados/pilotos/chunk_length/neural_robust/minilm/minilm_robust_comparison.json),
el [resultado Ollama](../resultados/pilotos/chunk_length/neural_robust/ollama/ollama_robust_comparison.json),
la [síntesis jerárquica](../resultados/pilotos/chunk_length/neural_robust/hierarchical_synthesis.json),
el [resumen consolidado](../resultados/pilotos/chunk_length/neural_robust/neural_robust_comparison.json),
el [manifiesto cross-fit](../resultados/pilotos/chunk_length/neural_robust/minilm_20_30_noninferiority/paired_train_crossfit_panel_manifest.json)
y el [resultado de no inferioridad](../resultados/pilotos/chunk_length/neural_robust/minilm_20_30_noninferiority/minilm_20_30_noninferiority.json).
Las predicciones individuales permanecen como checkpoints locales
reconstruibles y no se presentan como resultados consolidados.

## Pregunta y jerarquía predeclarada

El análisis examina si la elección clásica de 30 s se mantiene cuando cambia la
familia de representación o cuando un modelo generativo aplica el contrato
semántico. Se comparan, sin excepciones, las longitudes 15, 20, 25, 30 y 35 s.

La regla es una decisión metodológica local:

1. el perfil robusto clásico selecciona o conserva la longitud principal;
2. MiniLM evalúa sensibilidad con representaciones neuronales y scores
   continuos;
3. Ollama evalúa sensibilidad semántica y viabilidad de salida estructurada con
   etiquetas duras;
4. las métricas de las tres familias no se promedian;
5. si alguna familia contradice la referencia, se informa el conflicto y se
   conservan 30 s hasta una validación humana independiente.

MiniLM deriva de una arquitectura comprimida mediante destilación [3] y el
modelo multilingüe usa transferencia entre idiomas [4]. La revisión exacta del
checkpoint se fija mediante su tarjeta oficial [5]. Ollama ejecuta localmente
`gemma3:4b` [6] y solicita una salida ajustada a JSON Schema, capacidad descrita
por su documentación oficial [7]. Estas fuentes justifican las familias y sus
implementaciones; el panel, las cuotas, los márgenes y la jerarquía pertenecen
al proyecto.

## Diseño experimental

### Panel pareado

La unidad de observación es una ventana textual centrada en un mismo punto
temporal; la unidad de remuestreo es el video completo. El panel se construye
exclusivamente desde `validation` y contiene 100 anclas de 93 videos. Se permite
un máximo de dos anclas por video y se exige un mínimo de 20 anclas para cada
daño. La composición ejecutada es:

| Etiqueta almacenada | Anclas |
|---|---:|
| `SEGURO` | 36 |
| `RACISMO_DISCRIMINACION` | 20 |
| `ATAQUE_POR_GENERO_IDENTIDAD` | 22 |
| `ACOSO_AMENAZA` | 41 |
| `CONTENIDO_SEXUAL` | 20 |

Los conteos no suman 100 porque el contrato admite múltiples daños por ancla.
Los videos se reparten en cinco cohortes de reporte disjuntas, cada una con 20
anclas. La muestra está enriquecida para cubrir daños y no estima prevalencia
en YouTube, en el Perú ni en producción.

Para cada ancla se generan cinco ventanas centradas, una por longitud, y se
conserva el mismo conjunto de etiquetas de referencia. Este pareamiento reduce
la variación debida a escoger otro instante, aunque no reproduce todas las
fronteras consecutivas del troceado productivo. Las etiquetas proceden del
contrato vigente; no constituyen una nueva anotación humana ciega.

### MiniLM robusto

Se mantiene congelado
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Para cada una de
cinco cohortes clásicas y cada longitud se codifican hasta 1 000 filas de
entrenamiento y se ajusta una cabeza logística multietiqueta. El diseño ejecuta
25 ajustes. En el panel pareado se promedian los scores de las cinco cabezas de
cada longitud antes de calcular la métrica primaria.

La métrica primaria es AP macro de los cuatro daños. AP resume el ordenamiento
a través de umbrales y resulta informativa ante desbalance [2]. Se realizan
2 000 réplicas bootstrap percentil del 95%, remuestreando `video_id` con todas
sus anclas. El mismo remuestreo se aplica a las cinco longitudes, de modo que las
diferencias frente a 30 s permanecen pareadas. El bootstrap proporciona una
aproximación de incertidumbre [8] y la agrupación evita tratar observaciones del
mismo video como independientes [9]. El protocolo pareado y la hipótesis previa
siguen recomendaciones de evaluación estadística para PLN [10].

El margen local de no inferioridad es 0.01 AP. El límite inferior del intervalo
de la diferencia debe ser al menos −0.01 para declarar una alternativa no
inferior a 30 s.

### Contraste complementario MiniLM 20 s–30 s

El panel robusto anterior se conserva como piloto de cinco longitudes. Para
resolver específicamente su incertidumbre se predeclara
`ΔAP = AP_20s − AP_30s`, con `H0: ΔAP ≤ −0.01` y
`H1: ΔAP > −0.01`, formulación direccional propia de no inferioridad [13]. El
seguimiento no vuelve a escoger entre cinco candidatos: compara únicamente
20 s y la referencia de 30 s.

Como `validation` solo contiene 181 videos, el seguimiento emplea 750 videos
distintos de `train` mediante *cross-fitting* de cinco pliegues por `video_id`.
Se selecciona una ancla por video, se generan las dos ventanas centradas y cada
video recibe predicciones exclusivamente de una cabeza que no observó ninguna
fila de ese video. Tres repeticiones por pliegue y longitud producen 30 ajustes.
Los scores fuera de pliegue se promedian entre repeticiones antes de 5 000
réplicas bootstrap pareadas por video [8], [9]. `test` permanece cerrado.

### Ollama robusto

Ollama procesa las mismas 100 anclas y cinco longitudes: 500 respuestas
solicitadas. Se usa temperatura cero, semilla fija, timeout configurado de 90 s,
un reintento correctivo y el prompt vigente
[`config/prompt_operacional_ollama_v2.md`](../config/prompt_operacional_ollama_v2.md).
Las duraciones rotan entre anclas para que una detención temporal no favorezca
siempre a la misma longitud.

La métrica primaria predeclarada es F1 macro de los cuatro daños con principio
de intención de evaluar: una salida que no cumple el esquema cuenta como
predicción vacía. F1 equilibra precisión y recobrado en un umbral [11]. Se
reportan además precisión y recobrado macro de daño, coincidencia exacta,
pérdida de Hamming, tasa de esquema válido y un análisis secundario de casos
completos. La compuerta operativa exige al menos 0.95 de salidas válidas en cada
longitud. El margen local de no inferioridad es 0.02 F1 y la incertidumbre usa
2 000 réplicas bootstrap agrupadas por video.

La confianza autodeclarada por el LLM no se interpreta como probabilidad
calibrada. Sus propuestas tampoco se consideran referencia humana; las tareas
subjetivas asistidas por LLM requieren controles de influencia y revisión [12].

## Resultados MiniLM

MiniLM completó los 25 ajustes en 5.7 min. La estimación puntual más alta
correspondió a 20 s, pero su intervalo pareado de diferencia frente a 30 s
incluyó cero. Por ello el resultado es inconcluso respecto de una contradicción
de la referencia. Solo 35 s quedó por debajo de 30 s con un intervalo de
diferencia completamente negativo.

| Longitud | AP macro daño | IC bootstrap 95% | Δ frente a 30 s | IC 95% de Δ | No inferior, margen 0.01 |
|---:|---:|---:|---:|---:|:---:|
| 15 s | 0.57 | [0.49, 0.67] | 0.018 | [−0.074, 0.11] | No |
| 20 s | 0.59 | [0.50, 0.68] | 0.029 | [−0.046, 0.10] | No |
| 25 s | 0.54 | [0.47, 0.64] | −0.014 | [−0.072, 0.060] | No |
| **30 s** | **0.56** | **[0.47, 0.66]** | **0.00** | **[0.00, 0.00]** | **Sí** |
| 35 s | 0.44 | [0.36, 0.54] | −0.12 | [−0.17, −0.062] | No |

Estos valores describen el panel enriquecido. No deben compararse en valor
absoluto con la AP clásica, porque cambian la representación, el entrenamiento
y el universo de evaluación.

### Resultado del contraste complementario MiniLM 20 s–30 s

La ejecución utilizó 750 anclas de 750 videos, cinco pliegues de 150 y soportes
de 95/122/263/94 para racismo, género/identidad, acoso/amenaza y contenido
sexual. Se excluyeron antes del muestreo 50 referencias cuyo intervalo ya no
contenía texto en la transcripción canónica. Los 30 ajustes registraron cero
videos compartidos entre entrenamiento y el pliegue evaluado. La corrida tardó
765.5 s —12.8 min— en CPU.

| Longitud | AP macro daño OOF | IC bootstrap 95% | Δ frente a 30 s | IC 95% de Δ | No inferior, margen 0.01 |
|---:|---:|---:|---:|---:|:---:|
| **20 s** | **0.492** | **[0.445, 0.543]** | **0.024** | **[−0.0090, 0.059]** | **Sí** |
| 30 s | 0.468 | [0.424, 0.517] | 0.000 | [0.000, 0.000] | Sí, referencia |

El límite inferior de `ΔAP` fue `−0.0090`, mayor que `−0.01`; se estableció
**no inferioridad de 20 s frente a 30 s**. No se estableció superioridad porque
el intervalo incluyó cero. La probabilidad bootstrap descriptiva de no
inferioridad fue 0.978. El artefacto reportable es
[`minilm_20_30_noninferiority.json`](../resultados/pilotos/chunk_length/neural_robust/minilm_20_30_noninferiority/minilm_20_30_noninferiority.json).

Este cierre es interno: el panel procede de `train`, aunque todas sus
predicciones sean fuera de pliegue. No equivale a una réplica externa ni a una
nueva anotación humana y, por la jerarquía del estudio, no modifica la selección
clásica.

## Resultados Ollama

Ollama intentó las 500 combinaciones y produjo 474 salidas válidas y 26 fallos
después del reintento correctivo. La tasa global fue 0.95 al redondear, pero la
compuerta se evalúa por longitud: 15, 20 y 25 s obtuvieron respectivamente
0.93, 0.93 y 0.94, por debajo del mínimo predeclarado de 0.95. Por ello la
familia se clasifica como operativamente inconclusa. Las longitudes de 30 y 35 s
sí alcanzaron 0.97.

El análisis primario conserva los 100 casos de cada longitud y cuenta una salida
inválida como predicción vacía. La estimación puntual más alta correspondió a
30 s. Las ventanas de 15, 20 y 25 s quedaron por debajo de 30 s con intervalos
pareados completamente negativos; 35 s presentó una diferencia pequeña cuyo
intervalo incluyó cero. Ninguna alternativa fue significativamente superior a
la referencia.

| Longitud | Válidas | Tasa de esquema | F1 macro daño | IC bootstrap 95% | Δ frente a 30 s | IC 95% de Δ | Exact match | Hamming |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 s | 93/100 | 0.93 | 0.25 | [0.17, 0.34] | −0.17 | [−0.27, −0.057] | 0.20 | 0.27 |
| 20 s | 93/100 | 0.93 | 0.25 | [0.18, 0.33] | −0.16 | [−0.24, −0.081] | 0.18 | 0.28 |
| 25 s | 94/100 | 0.94 | 0.30 | [0.21, 0.38] | −0.12 | [−0.22, −0.034] | 0.22 | 0.27 |
| **30 s** | **97/100** | **0.97** | **0.42** | **[0.33, 0.49]** | **0.00** | **[0.00, 0.00]** | **0.22** | **0.26** |
| 35 s | 97/100 | 0.97 | 0.40 | [0.30, 0.48] | −0.023 | [−0.11, 0.066] | 0.21 | 0.26 |

El análisis secundario de casos completos retuvo 82 anclas con respuesta válida
en las cinco longitudes. También situó 30 s en el mayor F1 puntual, 0.38, frente
a 0.35 para 35 s. Este análisis no reemplaza el primario porque excluir los
fallos puede favorecer un subconjunto más sencillo.

Las respuestas consumieron 91 min de tiempo observado, repartidas en dos
invocaciones reanudables por el corte configurado. Este tiempo es una medición
local, no un límite de complejidad del modelo.

## Síntesis jerárquica

El panel MiniLM de cinco longitudes fue inconcluso, pero el seguimiento
específico estableció que 20 s es no inferior —no superior— a 30 s dentro de la
evaluación interna fuera de pliegue. Ollama situó 30 s en la mayor F1 puntual y
no encontró una alternativa superior, aunque falló la compuerta operativa de
esquema. En ninguna familia se demostró que otra longitud fuera mejor que 30 s.
La síntesis conserva **30 s** porque el perfil clásico es decisorio y el cierre
MiniLM es complementario. No se promedian métricas, no se consulta `test` y no
se activa ningún cambio de datos.

La distancia entre la decisión clásica y su corroboración neuronal se reduce,
pero no se elimina: MiniLM admite alternativas inciertas y Ollama no alcanza la
calidad estructural exigida en tres longitudes. Una modificación futura de 30 s
requiere validación humana independiente, no la elección retrospectiva del
mayor punto neuronal.

## Amenazas de validez

- **Constructo:** las etiquetas son transferencias temporales del contrato
  vigente; una anotación humana ciega de cada longitud podría modificar las
  diferencias.
- **Interna:** las ventanas centradas aíslan longitud, pero no representan todas
  las fronteras secuenciales del troceador de producción.
- **Estadística:** el bootstrap agrupa por video, aunque las cohortes MiniLM de
  entrenamiento pueden solaparse y no son cinco estudios independientes.
- **Seguimiento 20/30:** el cross-fit elimina fuga directa del video evaluado,
  pero procede de `train` y no constituye una muestra externa independiente.
- **Externa:** el panel enriquecido permite comparar sensibilidad, no estimar
  prevalencia o desempeño productivo.
- **Medición:** AP continua de MiniLM y F1 dura de Ollama responden preguntas
  distintas; combinarlas en una puntuación borraría esa diferencia.
- **Reproducibilidad:** los resultados dependen de los checkpoints locales
  exactos y del proveedor Ollama; los manifiestos registran versiones y firmas.

Una decisión distinta requeriría una muestra humana independiente, construida
sin reutilizar las etiquetas transferidas y manteniendo el agrupamiento por
video. `test` solo debe consultarse después de congelar la longitud y el modelo.

## Reproducción y lectura en el cuaderno

El protocolo está implementado en
[`01_02_optimizacion_longitud_chunks.ipynb`](../flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb).
Con el perfil clásico ya materializado, los controles principales son:

```python
RUN_CHUNK_LENGTH_ROBUST_TEST=False
RUN_NEURAL_ROBUST_TEST=True
RUN_MINILM_20_30_NONINFERIORITY_TEST=True
FORCE_NEURAL_ROBUST_RECOMPUTE=False
FORCE_MINILM_20_30_RECOMPUTE=False
CANDIDATE_SECONDS=(15,20,25,30,35)
USE_ROBUST_RECOMMENDATION=True
APPLY_CHUNK_SELECTION=False
```

Con ambos controles `FORCE_...=False`, el cuaderno comprueba primero los JSON
consolidados y muestra las tablas sin llamar MiniLM, Ollama ni bootstrap. Los
controles `RUN_...=True` significan ejecutar únicamente si falta el artefacto.
Active un `FORCE_...=True` solo para reconstruir deliberadamente esa etapa.

Los conteos y la correspondencia entre afirmaciones, fuentes externas y
artefactos internos se registran en
[`AUDITORIA_CITAS_01_02.md`](AUDITORIA_CITAS_01_02.md).

## Referencias

[1] G. C. Cawley and N. L. C. Talbot, “On Over-Fitting in Model Selection and
Subsequent Selection Bias in Performance Evaluation,” *Journal of Machine
Learning Research*, vol. 11, pp. 2079–2107, 2010. [En línea]. Disponible:
https://www.jmlr.org/papers/v11/cawley10a.html

[2] T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,”
*PLOS ONE*, vol. 10, no. 3, e0118432, 2015. doi:
[10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).

[3] W. Wang, F. Wei, L. Dong, *et al*., “MiniLM: Deep Self-Attention
Distillation for Task-Agnostic Compression of Pre-Trained Transformers,” in
*Advances in Neural Information Processing Systems*, vol. 33, 2020. [En línea].
Disponible:
https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[4] N. Reimers and I. Gurevych, “Making Monolingual Sentence Embeddings
Multilingual Using Knowledge Distillation,” in *Proceedings of EMNLP*,
pp. 4512–4525, 2020. doi:
[10.18653/v1/2020.emnlp-main.365](https://doi.org/10.18653/v1/2020.emnlp-main.365).

[5] Sentence Transformers, “Model Card:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,” Hugging Face Hub,
revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. [En línea]. Disponible:
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42

[6] Ollama, “Model Card: gemma3:4b,” Ollama Model Library, 2026. [En línea].
Disponible: https://ollama.com/library/gemma3:4b

[7] Ollama, “Structured Outputs,” Ollama Documentation, 2026. [En línea].
Disponible: https://docs.ollama.com/capabilities/structured-outputs

[8] B. Efron, “Bootstrap Methods: Another Look at the Jackknife,” *The Annals
of Statistics*, vol. 7, no. 1, pp. 1–26, 1979. doi:
[10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).

[9] C. A. Field and A. H. Welsh, “Bootstrapping Clustered Data,” *Journal of
the Royal Statistical Society: Series B*, vol. 69, no. 3, pp. 369–390, 2007.
doi:
[10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x).

[10] R. Dror, G. Baumer, S. Shlomov, *et al*., “The Hitchhiker's Guide to
Testing Statistical Significance in Natural Language Processing,” in
*Proceedings of ACL*, pp. 1383–1392, 2018. doi:
[10.18653/v1/P18-1128](https://doi.org/10.18653/v1/P18-1128).

[11] M. Sokolova and G. Lapalme, “A Systematic Analysis of Performance Measures
for Classification Tasks,” *Information Processing & Management*, vol. 45,
no. 4, pp. 427–437, 2009. doi:
[10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002).

[12] H. Schroeder, D. Roy, and J. Kabbara, “Just Put a Human in the Loop?
Investigating LLM-Assisted Annotation for Subjective Tasks,” in *Findings of
ACL*, pp. 25771–25795, 2025. doi:
[10.18653/v1/2025.findings-acl.1323](https://doi.org/10.18653/v1/2025.findings-acl.1323).

[13] W. C. Blackwelder, “Proving the Null Hypothesis in Clinical Trials,”
*Controlled Clinical Trials*, vol. 3, no. 4, pp. 345–353, 1982. doi:
[10.1016/0197-2456(82)90024-1](https://doi.org/10.1016/0197-2456(82)90024-1).
