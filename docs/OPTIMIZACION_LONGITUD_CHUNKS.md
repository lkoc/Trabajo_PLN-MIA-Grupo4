# Informe de selección de la longitud de chunks

**Fecha de ejecución:** 6 de agosto de 2026  
**Decisión:** conservar ventanas de **30 segundos**  
**Ámbito:** prueba confirmatoria local, corta y reproducible para fundamentar el
preprocesamiento del paper y la presentación  
**Artefacto de resultados:**
[`confirmatory_comparison.json`](../resultados/pilotos/chunk_length_expanded/confirmatory_comparison.json)

## Resumen ejecutivo

Se compararon ventanas de 15, 20, 25, 30 y 35 segundos mediante tres cohortes
pareadas. Cada cohorte incluyó 200 videos de entrenamiento, 80 de validación y
80 de test. Para cada longitud se volvieron a generar los chunks y se entrenaron
desde cero tres modelos CPU: Complement Naive Bayes, regresión logística y SGD
con pérdida logística. Luego se calibraron los umbrales en `validation` y se
realizó inferencia sobre `validation` y `test` usando chunks de la misma longitud
que los empleados durante el ajuste.

La longitud de 30 s obtuvo la mayor AP macro de daño en las tres repeticiones y
alcanzó `0.1142 ± 0.0050` en validación. La segunda alternativa fue 20 s con
`0.0895 ± 0.0075`. La diferencia pareada media entre 30 y 20 s fue `0.0248` AP;
el intervalo t descriptivo del 95% para las tres repeticiones fue
`[0.0127, 0.0368]`. Test no participó en la selección.

30 s también requirió menos filas de entrenamiento que 15, 20 y 25 s. Solo
35 s fue más barato, pero perdió `0.0588` AP frente a 30 s, muy por encima de la
tolerancia absoluta predefinida de `0.01`. Por ello, la evidencia ampliada
justifica mantener 30 s como compromiso entre contexto, desempeño y costo.

## Fundamento metodológico del smoke test

En este informe, *smoke test* no designa una prueba estadística ni una versión
reducida capaz de demostrar generalización. Designa una comprobación barata de
integración que atraviesa el pipeline completo: carga de datos, nueva
segmentación, transferencia de etiquetas, entrenamiento, calibración,
inferencia, métricas y persistencia de artefactos. Esta función es coherente con
la advertencia de que los sistemas ML acumulan riesgos en dependencias de datos,
configuración y código de integración, no solo en el estimador [1], y con las
rúbricas de madurez que recomiendan probar datos, infraestructura y modelos como
un sistema [2]. La elección concreta de 72 videos, dos modelos y diez ajustes en
el smoke test inicial fue una decisión operativa local; no procede de esas
fuentes.

El ejercicio tuvo tres niveles deliberadamente distintos:

| Nivel | Propósito válido | Lo que no permite concluir |
|---|---|---|
| Smoke test rápido | detectar errores de contrato y comprobar que cada longitud puede completar el flujo | elegir por sí solo una longitud ni reportar desempeño final |
| Confirmación local ampliada | comparar el orden relativo de cinco longitudes con entrenamiento real y cohortes pareadas | afirmar significancia fuerte o generalización productiva |
| Evaluación productiva posterior | entrenar las familias finales y reportar el modelo congelado | reutilizar test para volver a escoger longitud o modelo |

El smoke test rápido sugirió 35 s, mientras la confirmación ampliada favoreció
30 s. Esta divergencia ilustra por qué el primer nivel se usa para depuración y
dimensionamiento, no para congelar una decisión metodológica. El cambio solo se
consideró justificable después de ampliar videos, modelos y repeticiones, y se
mantuvo `APPLY_CHUNK_SELECTION=False` durante todo el ejercicio.

## Secuencia completa del ejercicio

El análisis no comenzó directamente con la corrida ampliada. Se ejecutaron tres
escalas sucesivas para aumentar evidencia solo después de comprobar que el
pipeline funcionaba. Esta progresión conserva el propósito de bajo costo de las
pruebas de integración [1], [2] y evita gastar minutos adicionales mientras
existan errores básicos de datos, entrenamiento o persistencia.

### 1. Smoke test rápido

La primera ejecución usó una cohorte de 40/16/16 videos de
train/validation/test, ComplementNB y SGD, 12 000 rasgos y transferencia desde
el chunk histórico con mayor solapamiento, exigiendo 50% de cobertura. Fueron
`5 × 2 = 10` ajustes y terminaron en menos de un minuto.

| Longitud | AP daño validation | Proxy de costo |
|---:|---:|---:|
| 15 s | 0.0554 | 7 118 |
| 20 s | 0.0613 | 5 600 |
| 25 s | 0.0644 | 4 614 |
| 30 s | 0.0593 | 4 004 |
| **35 s** | **0.0706** | **3 454** |

La salida técnica sugirió 35 s, pero las AP eran muy bajas, solo existía una
cohorte y la etiqueta dominante podía ocultar desacuerdos en límites. Este
resultado validó el circuito de extremo a extremo, no la longitud.

### 2. Confirmación corta inicial

La segunda ejecución aumentó a 100/40/40 videos, incorporó regresión logística,
usó tres semillas y exigió acuerdo de todas las referencias temporales con 80%
de cobertura. Fueron 45 ajustes.

| Longitud | AP daño validation | Victorias |
|---:|---:|---:|
| 15 s | 0.0689 ± 0.0158 | 0/3 |
| 20 s | 0.0727 ± 0.0067 | 0/3 |
| 25 s | 0.0557 ± 0.0036 | 0/3 |
| **30 s** | **0.0954 ± 0.0121** | **3/3** |
| 35 s | 0.0516 ± 0.0128 | 0/3 |

Esta corrida invirtió la sugerencia del smoke test y mostró que 30 s era más
estable, pero se decidió duplicar train y validation/test antes de redactar la
justificación académica.

### 3. Confirmación ampliada definitiva

La tercera ejecución usó 200/80/80 videos por cohorte y mantuvo tres modelos,
tres semillas, acuerdo temporal y 20 000 rasgos. También completó 45 ajustes y
terminó en aproximadamente 5.6 minutos. Sus resultados son la base de la
decisión vigente y se desarrollan en las secciones siguientes.

## Pregunta y criterio de decisión

La pregunta fue: ¿qué longitud entre 15 y 35 s conserva el mejor desempeño de
clasificación con el menor costo computacional razonable?

La longitud se trató como un hiperparámetro del pipeline completo, no como una
propiedad aislada del texto. La regla se fijó antes de consultar test:

1. calcular para cada longitud la media de
   `average_precision_macro_damage` de los tres modelos en `validation`;
2. identificar la mayor media;
3. admitir longitudes cuya pérdida absoluta no exceda `0.01` AP;
4. entre las admisibles, elegir el menor proxy determinista
   `filas_train × número_de_modelos`;
5. usar mayor AP y luego mayor longitud únicamente como desempates;
6. informar test después de seleccionar, sin usarlo para cambiar la decisión.

La separación entre selección y test reduce el sobreajuste del criterio y el
sesgo posterior de evaluación discutidos por Cawley y Talbot [3]. AP se eligió
por el fuerte desbalance entre `SEGURO` y los cuatro daños, contexto en el que
las curvas precisión–recall son más informativas que ROC [4]. La tolerancia
`0.01`, el proxy de costo y sus desempates son decisiones locales, no valores
recomendados por esas fuentes.

## Insumos y trazabilidad

| Insumo | Filas/alcance | SHA-256 |
|---|---:|---|
| `datos/raw/transcripts_raw.jsonl` | transcripciones canónicas | `7bae24f829f06979155c85d8602f485d0cf37a17758da97ec6b5620147f27df3` |
| `datos/processed/chunks_v2.jsonl` | 47 449 chunks temporales | `bbf20426e65f6e231dd8b337d71cade121c6a936d5a1bcc999698983bf8c92d0` |
| `datos/model_ready/v2/dataset_5_salidas.jsonl` | 117 244 filas | `daea22bcf5fb0bdff080db0ecabc5d69ad4c097b57b5d5bab3ea92dc40af1b2e` |
| `confirmatory_comparison.json` | resultados y cohortes | `7802c10b9117b65cef9f1884ebb024880dd3f635be8e31373b71753f2780e02b` |

El cruce entre chunks temporales y el snapshot histórico no confió en los
`chunk_id` secuenciales antiguos. Se unieron las filas mediante
`(video_id, SHA-256(texto_normalizado_NFKC))`. NFKC corresponde a una forma
estándar de normalización Unicode [5] y SHA-256 al estándar SHS de NIST [6]. Se
obtuvieron 36 087 referencias
temporales de 1 336 videos y no se observaron claves con etiquetas o splits
conflictivos.

## Cohortes pareadas

Se usaron las semillas deterministas `20260805`, `20260817` y `20260829`. Cada
repetición seleccionó 360 videos:

- 200 videos de `train`;
- 80 videos de `validation`;
- 80 videos de `test`.

La selección aplicó round-robin por las cinco salidas para enriquecer la
cobertura de daños sin mover videos entre splits. Dentro de una repetición, las
cinco longitudes utilizaron exactamente los mismos videos. Por ello, las
diferencias por longitud son pareadas y no se deben a que una alternativa haya
recibido otra cohorte.

No se trata de tres folds disjuntos de validación cruzada. Son tres submuestras
pareadas repetidas dentro de los splits congelados. Esta denominación evita
sugerir que un video cambió de train a validation o test.

| Split | Videos por repetición | Videos únicos en las tres | Presentes en las tres | Jaccard pareado, rango |
|---|---:|---:|---:|---:|
| train | 200 | 385 | 45 | 0.262–0.286 |
| validation | 80 | 115 | 57 | 0.600–0.616 |
| test | 80 | 112 | 59 | 0.600–0.667 |

El mayor solapamiento de validation y test se debe a que estos splits contienen
menos videos elegibles. La repetición evalúa sensibilidad al muestreo, pero no
constituye tres estimaciones completamente independientes; esta dependencia se
declara como limitación. Nadeau y Bengio muestran que ignorar la variabilidad de
las muestras de entrenamiento y la correlación entre repeticiones puede
subestimar la varianza de comparaciones de generalización [7]. Por ello aquí no
se aplica un test t ingenuo ni se afirma que las tres cohortes sean tres estudios
independientes.

## Transferencia temporal de etiquetas

Cada transcripción seleccionada se volvió a trocear por separado a 15, 20, 25,
30 y 35 s. Las etiquetas del dataset existente se transfirieron con una regla
conservadora de acuerdo temporal:

1. identificar todos los chunks históricos que solapan la nueva ventana;
2. exigir que sus conjuntos de etiquetas sean idénticos;
3. exigir una cobertura temporal acumulada mínima del 80%;
4. descartar la nueva ventana ante desacuerdo o cobertura insuficiente.

Esta política evita elegir arbitrariamente una etiqueta cuando una ventana
cruza dos regiones con decisiones diferentes. Reduce el número de ejemplos,
pero mejora la comparabilidad. Sigue siendo una etiqueta proxy: no reemplaza una
nueva anotación humana independiente de cada segmentación.

## Modelos y entrenamiento

Los modelos se eligieron para cubrir tres sesgos inductivos clásicos sin recurrir
a Transformers ni entrenamientos de horas. Todos emplearon una representación
TF-IDF, ponderación clásica para recuperación y clasificación textual [8]:

| Modelo | Configuración esencial | Papel en la comparación |
|---|---|---|
| ComplementNB | `alpha=1.0` | variante de Naive Bayes para clasificación textual [9] |
| Regresión logística | `class_weight="balanced"`, `max_iter=2000` | clasificador lineal discriminativo basado en el modelo logístico [10] |
| SGD incremental | `loss="log_loss"`, semilla `20260805` | optimización estocástica de bajo costo [11] |

Cada modelo se envolvió en `OneVsRestClassifier` para las cinco salidas y usó
TF-IDF de unigramas y bigramas, con `min_df=2`, transformación sublineal y máximo de
20 000 rasgos. Se reutilizó `train_classical_experiments`, la misma ruta de
entrenamiento de `03_01`, implementada con scikit-learn [12]. Las opciones
concretas de vectorización, pesos, iteraciones y semilla son parámetros locales.

Para cada combinación longitud–modelo–cohorte se ejecutó:

`fit en train → calibración de cinco umbrales en validation → inferencia en validation → inferencia descriptiva en test`.

En total se realizaron `5 longitudes × 3 modelos × 3 cohortes = 45` ajustes. La
corrida local ampliada terminó en aproximadamente 5.6 minutos. No usó red, GPU,
Transformers ni checkpoints entrenados con otra longitud.

## Resultados agregados

La tabla presenta media y desviación estándar muestral entre las tres cohortes.
El proxy de costo es el promedio de `filas_train × 3 modelos`.

| Longitud | AP daño validation | Victorias | Proxy de costo | AP test descriptiva |
|---:|---:|---:|---:|---:|
| 15 s | 0.0784 ± 0.0035 | 0/3 | 52 106 | 0.0895 ± 0.0031 |
| 20 s | 0.0895 ± 0.0075 | 0/3 | 39 855 | 0.0822 ± 0.0103 |
| 25 s | 0.0688 ± 0.0056 | 0/3 | 31 881 | 0.0743 ± 0.0108 |
| **30 s** | **0.1142 ± 0.0050** | **3/3** | **30 181** | **0.1246 ± 0.0024** |
| 35 s | 0.0554 ± 0.0057 | 0/3 | 22 809 | 0.0604 ± 0.0092 |

## Resultados pareados por cohorte

| Semilla | 15 s | 20 s | 25 s | **30 s** | 35 s |
|---:|---:|---:|---:|---:|---:|
| 20260805 | 0.0758 | 0.0808 | 0.0685 | **0.1100** | 0.0616 |
| 20260817 | 0.0824 | 0.0943 | 0.0746 | **0.1198** | 0.0542 |
| 20260829 | 0.0771 | 0.0933 | 0.0634 | **0.1129** | 0.0504 |

30 s superó a las otras cuatro longitudes en cada repetición. Las diferencias
pareadas medias y sus intervalos t descriptivos del 95% fueron:

| Comparación | Diferencia media a favor de 30 s | IC descriptivo 95% |
|---|---:|---:|
| 30 s – 15 s | 0.0358 | [0.0318, 0.0398] |
| 30 s – 20 s | 0.0248 | [0.0127, 0.0368] |
| 30 s – 25 s | 0.0454 | [0.0353, 0.0555] |
| 30 s – 35 s | 0.0588 | [0.0360, 0.0817] |

Los intervalos se incluyen como descripción de estabilidad, no como inferencia
confirmatoria fuerte: solo existen tres repeticiones y sus cohortes se solapan.
No se empleó la corrección de Nadeau–Bengio ni se ejecutó un contraste formal;
su trabajo se usa para justificar precisamente esta cautela [7].

## Resultados por modelo en validation

| Longitud | ComplementNB | Regresión logística | SGD incremental |
|---:|---:|---:|---:|
| 15 s | 0.0337 ± 0.0007 | 0.0985 ± 0.0051 | 0.1031 ± 0.0048 |
| 20 s | 0.0267 ± 0.0005 | 0.1227 ± 0.0110 | 0.1190 ± 0.0123 |
| 25 s | 0.0218 ± 0.0007 | 0.0926 ± 0.0100 | 0.0921 ± 0.0077 |
| **30 s** | **0.0367 ± 0.0008** | **0.1535 ± 0.0054** | **0.1525 ± 0.0092** |
| 35 s | 0.0138 ± 0.0007 | 0.0844 ± 0.0134 | 0.0680 ± 0.0041 |

30 s fue la mejor longitud para cada una de las tres familias, no solo para el
promedio agregado. La evidencia a favor de 30 s, por tanto, no depende de un
único clasificador.

## Soporte a 30 segundos

Las tres cohortes mantuvieron ejemplos de las cinco salidas en todos los splits.
Los siguientes rangos corresponden al número de chunks etiquetados por
repetición; una fila multietiqueta puede contribuir a más de un daño.

| Split | Filas | SEGURO | Racismo | Género/identidad | Acoso/amenaza | Sexual |
|---|---:|---:|---:|---:|---:|---:|
| train | 9 769–10 332 | 8 847–9 388 | 163–198 | 270–281 | 404–437 | 387–426 |
| validation | 3 104–3 150 | 2 842–2 887 | 68 | 60 | 118–119 | 101 |
| test | 3 620–4 072 | 3 285–3 735 | 68 | 112 | 140–142 | 150 |

La prevalencia de `SEGURO` explica las AP absolutas modestas y confirma la
necesidad de una métrica precisión–recall. Estas cifras no estiman prevalencia
en YouTube: el corpus y las cohortes fueron enriquecidos deliberadamente.

## Justificación de la elección de 30 segundos

La recomendación se apoya en cinco observaciones concordantes:

1. 30 s ganó en las tres cohortes pareadas.
2. Obtuvo la mayor media para los tres modelos por separado.
3. Superó a la segunda alternativa, 20 s, por 0.0248 AP pareada.
4. Fue más barato que 15, 20 y 25 s según el proxy de filas entrenadas.
5. Aunque 35 s redujo el proxy de costo en 24.4%, perdió 0.0588 AP, casi seis
   veces la tolerancia máxima admitida de 0.01.

El resultado es coherente con un compromiso de contexto: las ventanas más
cortas fragmentan evidencia lingüística, mientras 35 s aumenta la probabilidad
de cruzar regiones con etiquetas distintas y pierde ejemplos bajo la regla de
acuerdo temporal. Esta explicación es plausible y no debe presentarse como un
mecanismo causal demostrado.

## Texto recomendado para el paper

> La longitud de los fragmentos se seleccionó mediante tres submuestras
> pareadas de 200/80/80 videos para entrenamiento, validación y test. Para cada
> alternativa de 15, 20, 25, 30 y 35 s se regeneraron los fragmentos y se
> reentrenaron ComplementNB, regresión logística y SGD con TF-IDF. La selección
> utilizó únicamente AP macro de los cuatro daños en validación. Las ventanas de
> 30 s alcanzaron 0.1142 ± 0.0050 y ganaron las tres repeticiones, frente a
> 0.0895 ± 0.0075 para la siguiente alternativa, 20 s. En consecuencia, se
> mantuvieron 30 s; test se reservó para reporte descriptivo.

Al incorporar este texto debe citarse la fuente de AP y de los modelos, y el
informe debe figurar como artefacto reproducible del proyecto. Las métricas deben
redondearse en la publicación, pero los valores completos permanecen en JSON.

## Contenido recomendado para la presentación

Una diapositiva puede usar tres mensajes:

- **Diseño:** 5 longitudes × 3 modelos × 3 cohortes pareadas = 45 ajustes CPU.
- **Resultado:** 30 s ganó 3/3; AP validation 0.114 ± 0.005.
- **Decisión:** se conserva 30 s; 35 s es 24% más barato, pero pierde 0.059 AP.

La tabla agregada o un gráfico de AP media con barras de desviación estándar es
suficiente. Test debe aparecer como resultado descriptivo, no como criterio de
selección.

## Limitaciones y amenazas de validez

- Las etiquetas de las segmentaciones alternativas son transferencias
  temporales, no nuevas anotaciones humanas ciegas.
- Las tres cohortes se solapan y no son folds independientes; por ello sus
  intervalos son descriptivos y no pruebas de significancia [7].
- Se compararon modelos clásicos ligeros; otra arquitectura puede responder de
  forma diferente a la longitud.
- La métrica absoluta no representa un modelo productivo ni sustituye el
  reentrenamiento completo del contrato de cinco salidas.
- La regla de acuerdo descarta ventanas ambiguas y puede favorecer longitudes
  cercanas a los chunks históricos de 30 s. Esta dependencia es una limitación
  central y debe declararse en el paper.
- La conclusión válida es acotada: con el corpus, etiquetas y modelos ligeros
  disponibles no existe evidencia para cambiar el contrato vigente de 30 s.

Una validación futura más fuerte requeriría anotar independientemente una
muestra de ventanas construidas con varias longitudes o etiquetar unidades
temporales más finas antes de agregarlas.

## Reproducción en el cuaderno

El protocolo está implementado en
[`01_02_optimizacion_longitud_chunks.ipynb`](../flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb).
Los controles relevantes son:

- `RUN_CHUNK_LENGTH_CONFIRMATORY_TEST=True`;
- `CONFIRMATORY_VIDEO_LIMITS={'train':200,'validation':80,'test':80}`;
- `CONFIRMATORY_SEEDS=(20260805,20260817,20260829)`;
- `CONFIRMATORY_MODELS=('complement_nb','logistic_regression','sgd_incremental')`;
- `USE_CONFIRMATORY_RECOMMENDATION=True` para previsualizar 30 s;
- `APPLY_CHUNK_SELECTION=False` para no mover el dataset durante la prueba.

La corrida es reanudable por firma. El cambio de longitud solo ocurre si se
activa explícitamente `APPLY_CHUNK_SELECTION=True`. La configuración actual
permanece en 30 s y ningún dataset activo fue movido durante estos experimentos.

## Archivo reversible

Si en el futuro se activa otra longitud, los chunks, etiquetas, snapshots,
modelos, resultados y bundle de la firma vigente se mueven —no se borran— a
`archivo/chunking_configurations/<firma>/state/`. El manifiesto registra tamaño
y SHA-256. Antes de restaurar una firma anterior se validan todos sus archivos;
si uno está alterado, la transición se rechaza antes de mover el estado activo.

Las transcripciones raw, candidatos y caché de adquisición no forman parte de
esta transición. Los estados grandes permanecen locales; el JSON confirmatorio
de 65 KB se sincroniza para conservar la evidencia metodológica en GitHub.

## Referencias

[1] D. Sculley, G. Holt, D. Golovin, *et al*., “Hidden Technical Debt in
Machine Learning Systems,” in *Advances in Neural Information Processing
Systems*, vol. 28, pp. 2503–2511, 2015.
[En línea](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html).

[2] E. Breck, S. Cai, E. Nielsen, *et al*., “The ML Test Score: A Rubric for ML
Production Readiness and Technical Debt Reduction,” in *2017 IEEE International
Conference on Big Data*, pp. 1123–1132, 2017,
doi: [10.1109/BigData.2017.8258038](https://doi.org/10.1109/BigData.2017.8258038).

[3] G. C. Cawley and N. L. C. Talbot, “On Over-Fitting in Model Selection and
Subsequent Selection Bias in Performance Evaluation,” *Journal of Machine
Learning Research*, vol. 11, pp. 2079–2107, 2010.
[En línea](https://www.jmlr.org/papers/v11/cawley10a.html).

[4] T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative
than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,”
*PLOS ONE*, vol. 10, no. 3, e0118432, 2015,
doi: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).

[5] Unicode Consortium, “Unicode Normalization Forms,” Unicode Standard Annex
No. 15, rev. 57, Unicode 17.0.0, 2025.
[En línea](https://www.unicode.org/reports/tr15/tr15-57.html).

[6] National Institute of Standards and Technology, *Secure Hash Standard
(SHS)*, FIPS PUB 180-4, 2015,
doi: [10.6028/NIST.FIPS.180-4](https://doi.org/10.6028/NIST.FIPS.180-4).

[7] C. Nadeau and Y. Bengio, “Inference for the Generalization Error,” *Machine
Learning*, vol. 52, no. 3, pp. 239–281, 2003,
doi: [10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366).

[8] G. Salton and C. Buckley, “Term-Weighting Approaches in Automatic Text
Retrieval,” *Information Processing & Management*, vol. 24, no. 5,
pp. 513–523, 1988,
doi: [10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0).

[9] J. D. M. Rennie, L. Shih, J. Teevan, and D. R. Karger, “Tackling the Poor
Assumptions of Naive Bayes Text Classifiers,” in *Proceedings of the 20th
International Conference on Machine Learning*, pp. 616–623, 2003.
[En línea](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf).

[10] D. R. Cox, “The Regression Analysis of Binary Sequences,” *Journal of the
Royal Statistical Society: Series B*, vol. 20, no. 2, pp. 215–232, 1958,
doi: [10.1111/j.2517-6161.1958.tb00292.x](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x).

[11] L. Bottou, “Large-Scale Machine Learning with Stochastic Gradient
Descent,” in *Proceedings of COMPSTAT 2010*, pp. 177–186, 2010,
doi: [10.1007/978-3-7908-2604-3_16](https://doi.org/10.1007/978-3-7908-2604-3_16).

[12] F. Pedregosa, G. Varoquaux, A. Gramfort, *et al*., “Scikit-Learn: Machine
Learning in Python,” *Journal of Machine Learning Research*, vol. 12,
pp. 2825–2830, 2011.
[En línea](https://www.jmlr.org/papers/v12/pedregosa11a.html).
