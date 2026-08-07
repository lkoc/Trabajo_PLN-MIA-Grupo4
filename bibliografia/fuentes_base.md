# Fuentes base y trazabilidad bibliográfica

Última verificación bibliográfica: **2026-07-29**. Revisión de contrato y rutas: **2026-08-05**.

Las fuentes de esta matriz orientan definiciones y límites; no “validan” por sí solas la taxonomía. El contrato v2 aprende cinco salidas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`; conserva 14 etiquetas finas y tres flags, y mantiene separadas la evidencia académica general, la evidencia peruana/institucional, la política de plataforma y las decisiones locales. `SEGURO` es excluyente, mientras los cuatro daños son multietiqueta. Sus métricas permanecen pendientes; las cifras publicadas pertenecen al baseline archivado.

La auditoría v2.1 vincula de forma explícita las cuatro salidas `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL` con los artículos adjuntos. El detalle de afirmación, fuente, alcance peruano y límite está en [`docs/MATRIZ_EVIDENCIA_TAXONOMIA.md`](../docs/MATRIZ_EVIDENCIA_TAXONOMIA.md). `ATAQUE_POR_GENERO_IDENTIDAD` sustituye el nombre anterior porque explicita daño sin reducir todos los casos a acoso ni exigir intención de odio.

La base BibTeX canónica del artículo es
[`Documento_final_paper/referencias.bib`](../Documento_final_paper/referencias.bib).
Este catálogo explica qué afirmación puede respaldar cada fuente y, sobre todo,
qué no debe inferirse de ella.

## Reglas de uso

1. El conjunto utilizado por el proyecto es un corpus propio de subtítulos de
   YouTube. HatEval, OffendES, DETOXIS, EXIST, ALYT, HateXplain y el corpus de
   ataques personales de Wikipedia son **antecedentes**, no datos de
   entrenamiento ni de prueba del proyecto.
2. Las etiquetas generadas con DeepSeek son preanotaciones asistidas. La
   documentación del proveedor acredita el identificador y la versión del
   modelo, pero no convierte sus respuestas en verdad de referencia. La validez
   depende del protocolo de revisión y adjudicación humana.
3. Los artículos fundacionales explican una familia de algoritmos. Las mejoras
   observadas en este proyecto deben apoyarse en sus propias tablas, intervalos
   y pruebas, no en los resultados del artículo fundacional.
4. En el código, la métrica principal se obtiene con
   `sklearn.metrics.average_precision_score`. Debe nombrarse **average
   precision (AP)**. La etiqueta histórica `PR-AUC` puede conservarse en nombres
   de campos, pero AP no es el área trapezoidal bajo la curva precisión--recall.
5. La selección de modelos, umbrales y calibradores se hace con entrenamiento y
   validación. El conjunto de prueba permanece bloqueado hasta la evaluación
   final.
6. La regla de consenso 2-de-3 y el objetivo operativo de 95 % de recall son
   decisiones del artefacto. Las fuentes de ensambles y clasificación selectiva
   las contextualizan, pero no prueban que sean óptimas para este corpus.
7. Para fuentes web y tarjetas de modelo se debe registrar fecha de consulta y,
   cuando sea posible, revisión o `commit` exacto.

## Guías normativas de escritura

Estas fuentes orientan la preparación del manuscrito; normalmente no se incluyen
en la bibliografía científica del paper.

| Fuente oficial | Uso |
|---|---|
| [IEEE: Structure Your Paper](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/) | Estructura de un artículo de conferencia y resumen autónomo de un solo párrafo y hasta 250 palabras. |
| [IEEE Article Templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/ieee-article-templates/) | Plantilla oficial; el paper usa `IEEEtran`. |
| [IEEE Editorial Style Manual](https://journals.ieeeauthorcenter.ieee.org/your-role-in-article-production/ieee-editorial-style-manual/) | Estilo editorial, títulos, tablas, figuras y convenciones. |
| [IEEE Reference Guide](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf) | Formato de referencias y citas numéricas. |
| [IEEE Author Ethics](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/ethical-requirements/) | Citar paráfrasis, ideas, datos, procesos y figuras adaptadas; evitar plagio y referencias irrelevantes. |
| [Beamer en CTAN](https://ctan.org/pkg/beamer/) | Clase oficial para la presentación. No existe una plantilla Beamer universal de IEEE. |

## Design Science Research

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `hevner2004dsr` | Diseño y evaluación rigurosa de un artefacto como contribución de conocimiento. | No define por sí solo la taxonomía ni los tres niveles de problema del proyecto. | [AIS / MIS Quarterly](https://aisel.aisnet.org/misq/vol28/iss1/6/) |
| `peffers2007dsrm` | Ciclo DSRM: problema, objetivos, diseño, demostración, evaluación y comunicación. | El mapeo de cuadernos e iteraciones al ciclo es una interpretación de los autores. | [DOI 10.2753/MIS0742-1222240302](https://doi.org/10.2753/MIS0742-1222240302) |

## Corpus propio, muestreo y depuración

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `grupo4dataset2026` | Nombre, esquema 2.0, fecha de corte y manifiesto del corpus propio. | Es un artefacto interno sin DOI ni licencia autónoma de redistribución de los textos. | [Manifiesto del corpus](https://github.com/lkoc/Trabajo_PLN-MIA-Grupo4/blob/main/datos/model_ready/transformer_grueso/dataset_integrado_todas_pasadas.manifest.json) |
| `brodley1999mislabeled` | Contexto para buscar rótulos potencialmente erróneos mediante desacuerdo o dificultad. | La minería de 2 000 negativos es una heurística del proyecto, no una réplica de su algoritmo. | [JAIR / DOI](https://doi.org/10.1613/jair.606) |
| `settles2009active` | Fundamento de selección informativa para anotación. | El muestreo dirigido local no implementa una función de adquisición publicada. | [University of Wisconsin](https://minds.wisconsin.edu/handle/1793/60660) |
| `fairstein2024balancing` | Selección activa bajo desbalance. | No valida las cuotas ni la prevalencia del corpus local. | [ACL Anthology](https://aclanthology.org/2024.law-1.8/) |
| `huang2021balancing` | Riesgos y métodos de balance en clasificación multietiqueta de cola larga. | El muestreo 4:1 es una decisión local y no una réplica del método. | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.643/) |
| `tonneau2024naijahate` | Importancia de separar muestras representativas y enriquecidas en evaluación de odio. | NaijaHate no forma parte del corpus peruano ni estima su prevalencia. | [ACL Anthology](https://aclanthology.org/2024.acl-long.488/) |

## Moderación, anotación y antecedentes

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `gorwa2020moderation` | Escala, dependencia contextual y desafíos técnicos y políticos de la moderación algorítmica. | No demuestra que el artefacto local sea justo, seguro o apto para producción. | [SAGE](https://journals.sagepub.com/doi/10.1177/2053951719897945) |
| `vidgen2020directions` | Calidad de datos, heterogeneidad de definiciones, muestreo, anotación y desbalance en lenguaje abusivo. | Revisión del campo; no describe el corpus peruano. | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243300) |
| `bender2018datastatements` | Declarar población, idioma, recolección, anotadores y límites de generalización. | Es una guía de documentación, no una validación del corpus. | [ACL Anthology](https://aclanthology.org/Q18-1041/) |
| `artstein2008agreement` | Condiciones y métricas para estudiar acuerdo entre codificadores en lingüística computacional. | Una sola adjudicación con propuesta visible no permite calcular acuerdo interanotador. | [Computational Linguistics / DOI](https://doi.org/10.1162/coli.07-034-R2) |
| `bertaglia2021youtube` | Antecedente de anotación de abuso asociado a YouTube y necesidad de guías explícitas. | ALYT contiene comentarios en inglés, no subtítulos peruanos en español. | [ACL Anthology](https://aclanthology.org/2021.woah-1.20/) |
| `rottger2021hatecheck` | Las métricas agregadas pueden ocultar fallas funcionales concretas. | HateCheck no fue usado para entrenar ni evaluar los modelos del proyecto. | [ACL Anthology](https://aclanthology.org/2021.acl-long.4/) |
| `sap2019racialbias` | Riesgo de asociar variedades lingüísticas o marcadores identitarios con toxicidad. | El artículo estudia otro contexto lingüístico; motiva auditoría, no cuantifica sesgo peruano. | [ACL Anthology](https://aclanthology.org/P19-1163/) |
| `wulczyn2017exmachina` | Antecedente de anotación y detección de ataques personales a gran escala. | Su corpus de Wikipedia no forma parte de los datos locales. | [ACM](https://doi.org/10.1145/3038912.3052591) |
| `mathew2021hatexplain` | Antecedente de objetivos del abuso, racionales humanos, explicabilidad y sesgo. | HateXplain no es un conjunto de entrenamiento del proyecto. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17745) |
| `deepseek2026v4` | Procedencia de los modelos `deepseek-v4-flash` y `deepseek-v4-pro` empleados en preanotación/revisión. | Es documentación del proveedor; no prueba calidad, acuerdo ni validez de las etiquetas. | [DeepSeek API Docs](https://api-docs.deepseek.com/news/news260424/) |

### Taxonomía de daño y señales de revisión

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `waseem2017abuse` | Dos ejes generales: blanco individual/entidad frente a grupo, y abuso explícito frente a implícito. | No propone las 14 etiquetas finas, la unión local de acoso/amenaza ni `SEGURO` como salida aprendida. | [ACL Anthology](https://aclanthology.org/W17-3012/) |
| `banko2020taxonomy` | Criterios y excepciones para ataque por identidad, insulto, doxeo, agresión sexual y amenaza de violencia. | La extensión local de amenaza a daño legal/económico no procede de esta fuente. | [ACL Anthology](https://aclanthology.org/2020.alw-1.16/) |
| `elsherief2021implicit` | Odio implícito expresado mediante lenguaje codificado o indirecto. | Corpus en inglés; no valida el flag local ni su umbral. | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.29/) |
| `ilic2018irony` | Sarcasmo e ironía como tarea de interpretación automática. | No define `ironia_ambigua` ni prueba intención del hablante. | [ACL Anthology](https://aclanthology.org/W18-6202/) |
| `bourgeade2024context` | El contexto conversacional puede ser necesario para anotar lenguaje abusivo. | Estudia conversaciones en inglés/francés; el flag y la regla de revisión son locales. | [ACL Anthology](https://aclanthology.org/2024.lrec-main.740/) |
| `zeinert2021misogyny` | Código de anotación de misoginia, incluidos acoso, descrédito, estereotipo y cosificación. | Corpus danés; orienta definiciones, no valida datos peruanos. | [ACL Anthology](https://aclanthology.org/2021.acl-long.247/) |
| `chakravarthi2024homotrans` | Detección de homofobia y transfobia en comentarios de YouTube. | No estudia Perú ni subtítulos en español. | [Springer](https://doi.org/10.1007/s41060-023-00400-0) |
| `youtube2026sexualpolicy` | Frontera operativa de plataforma: contenido explícito, sexualización no consentida y excepciones contextuales. | Es política de YouTube, no definición académica ni calificación legal. | [Ayuda de YouTube](https://support.google.com/youtube/answer/2802002?hl=es-419) |

### Antecedentes en español

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `basile2019hateval` | Tarea bilingüe de odio contra inmigrantes y mujeres, con nivel binario y atributos finos. | Datos de Twitter; no fueron usados por el proyecto. | [ACL Anthology](https://aclanthology.org/S19-2007/) |
| `plaza2021offendes` | Corpus de 47 128 comentarios españoles de Twitter, Instagram y YouTube. | Comentarios, no subtítulos; antecedente únicamente. | [ACL Anthology](https://aclanthology.org/2021.ranlp-1.123/) |
| `taule2021detoxis` | NewsCom-TOX y evaluación de toxicidad en comentarios españoles. | Dominio de noticias y foros; no datos del proyecto. | [SEPLN](https://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6390) |
| `rodriguez2021exist` | Identificación y categorización de sexismo en español e inglés. | Tweets y publicaciones de Gab; antecedente únicamente. | [SEPLN](https://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6389) |

## Modelos clásicos ejecutados

| Clave BibTeX | Componente local | Qué respalda | Fuente primaria |
|---|---|---|---|
| `salton1988tfidf` | TF--IDF de palabras y caracteres. | Fundamento del esquema de ponderación, no la configuración concreta de n-gramas. | [Elsevier](https://www.sciencedirect.com/science/article/pii/0306457388900210) |
| `cox1958logistic` | Regresión logística. | Fundamento del modelo binario; la extensión multietiqueta es parte de la implementación. | [JRSS B](https://academic.oup.com/jrsssb/article/20/2/215/7027376) |
| `cortes1995svm` | SVM lineal. | Fundamento de máquinas de soporte vectorial. | [Springer](https://doi.org/10.1007/BF00994018) |
| `rennie2003cnb` | `ComplementNB`. | Variante de Naive Bayes diseñada para corregir supuestos pobres en clasificación textual. | [MIT CSAIL](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) |
| `bottou2010sgd` | `SGDClassifier`. | Fundamento del aprendizaje a gran escala con descenso de gradiente estocástico; no fija la pérdida ni los hiperparámetros locales. | [Springer](https://doi.org/10.1007/978-3-7908-2604-3_16) |
| `deerwester1990lsa` | `TruncatedSVD` previo al modelo tabular. | Fundamento de análisis semántico latente; no prueba que una dimensión local sea óptima. | [Wiley](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9) |
| `friedman2001gbm` | Gradient boosting sobre características reducidas. | Familia de boosting; la implementación concreta es `HistGradientBoostingClassifier`. | [Project Euclid](https://doi.org/10.1214/aos/1013203451) |
| `joulin2017fasttext` | fastText supervisado OVA. | Clasificación textual eficiente con representaciones promediadas. | [ACL Anthology](https://aclanthology.org/E17-2068/) |

No se encontró `RandomForestClassifier` en el flujo activo. No debe presentarse
como experimento ejecutado.

## Multietiqueta, jerarquía y multitarea

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `tsoumakas2007multilabel` | Definición y familias clásicas de clasificación multietiqueta. | No describe la taxonomía propia ni sus umbrales. | [DOI 10.4018/jdwm.2007070101](https://doi.org/10.4018/jdwm.2007070101) |
| `zhang2014multilabel` | Revisión de transformaciones de problema, adaptación algorítmica y correlación entre etiquetas. | No demuestra ventaja del diseño jerárquico local. | [IEEE](https://doi.org/10.1109/TKDE.2013.39) |
| `silla2011hierarchical` | Taxonomía general de clasificación jerárquica. | Las cascadas y cabezas locales son diseños del proyecto. | [Springer](https://doi.org/10.1007/s10618-010-0175-9) |
| `caruana1997multitask` | Aprendizaje compartido entre varias tareas. | No garantiza transferencia positiva entre las cuatro categorías de daño. | [Springer](https://doi.org/10.1023/A:1007379606734) |
| `zhou2020hiagm` | Modelado global consciente de jerarquías para clasificación textual. | La cabeza local solo toma la consistencia jerárquica como motivación; no implementa HiAGM. | [ACL Anthology](https://aclanthology.org/2020.acl-main.104/) |

## Transformers y ajuste eficiente

| Clave BibTeX | Componente local | Qué respalda | Fuente primaria |
|---|---|---|---|
| `vaswani2017attention` | Arquitectura Transformer. | Fundamento general. | [NeurIPS](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need) |
| `devlin2019bert` | Preentrenamiento bidireccional y adaptación a tareas. | Antecedente arquitectónico, no checkpoint ejecutado. | [ACL Anthology](https://aclanthology.org/N19-1423/) |
| `reimers2019sbert` | Codificación de oraciones. | Fundamento de Sentence-BERT. | [ACL Anthology](https://aclanthology.org/D19-1410/) |
| `wang2020minilm` | Linaje de MiniLM. | Sustenta destilación de autoatención; no por sí solo la variante multilingüe. | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| `reimers2020multilingual` | `paraphrase-multilingual-MiniLM-L12-v2`. | Sustenta la estrategia de destilación multilingüe; el checkpoint exacto requiere tarjeta/revisión. | [ACL Anthology](https://aclanthology.org/2020.emnlp-main.365/) |
| `wang2024e5` | `intfloat/multilingual-e5-small`. | Sustenta el preentrenamiento y las representaciones E5 multilingües. El uso como clasificador es una adaptación local. | [arXiv](https://arxiv.org/abs/2402.05672) |
| `hu2022lora` | Ajuste LoRA de Qwen. | Sustenta matrices entrenables de bajo rango con base congelada. | [OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9) |
| `qwen2025qwen3` | Familia Qwen3 y escala 0.6B. | La calidad en moderación peruana solo puede derivarse de los experimentos locales. | [arXiv](https://arxiv.org/abs/2505.09388) |
| `loshchilov2019adamw` | Optimizador AdamW. | Sustenta desacoplar weight decay; no la tasa de aprendizaje elegida. | [OpenReview](https://openreview.net/forum?id=Bkg6RiCqY7) |
| `karimi2021aeda` | Aumento histórico por inserción de puntuación. | Citar solo en la iteración donde AEDA fue realmente aplicada. | [ACL Anthology](https://aclanthology.org/2021.findings-emnlp.234/) |

BETO y Whisper aparecían en documentos preliminares, pero no corresponden al
flujo activo auditado. Deben conservarse únicamente si una tabla histórica
identifica una ejecución reproducible concreta.

## Calibración, selección y operación

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `niculescu2005probabilities` | Comparación y calibración de probabilidades, incluida calibración sigmoidal. | Los calibradores por etiqueta y sus parámetros son del proyecto. | [ACM](https://doi.org/10.1145/1102351.1102430) |
| `platt1999probabilistic` | Fundamento del escalamiento sigmoidal de scores. | La regresión logística univariada por etiqueta y su partición son decisiones locales. | *Advances in Large Margin Classifiers*, MIT Press, pp. 61--74. |
| `guo2017calibration` | Las redes modernas pueden estar descalibradas. | Su método destacado es temperature scaling; el proyecto usa calibración sigmoidal por etiqueta. | [PMLR](https://proceedings.mlr.press/v70/guo17a.html) |
| `saito2015pr` | Conveniencia de precisión--recall ante desbalance. | No define el cálculo exacto de `average_precision_score`. | [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) |
| `davis2006pr` | Relación matemática entre curvas ROC y PR. | No convierte AP en área trapezoidal. | [ACM](https://doi.org/10.1145/1143844.1143874) |
| `efron1979bootstrap` | Fundamento del bootstrap. | El remuestreo agrupado por video es una adaptación local que debe explicarse. | [Project Euclid](https://doi.org/10.1214/aos/1176344552) |
| `cawley2010selection` | Sesgo por selección y necesidad de separar ajuste de evaluación final. | No sustituye la descripción concreta de la partición por video. | [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html) |
| `nadeau2003inference` | La varianza de comparaciones repetidas depende tanto de las muestras de entrenamiento como de evaluación. | No convierte tres cohortes solapadas en repeticiones independientes ni valida por sí solo un test t ingenuo. | [DOI](https://doi.org/10.1023/A:1024068626366) |
| `chow1970reject` | Fundamento de la opción de rechazo/abstención. | No fija el umbral ni el objetivo de recall local. | [IEEE](https://doi.org/10.1109/TIT.1970.1054406) |
| `geifman2017selective` | Relación riesgo--cobertura en clasificación selectiva. | El artículo no garantiza el riesgo del despliegue local. | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |
| `dietterich2000ensemble` | Contexto general para combinar clasificadores. | La mayoría 2-de-3 debe evaluarse con los resultados del proyecto. | [Springer](https://doi.org/10.1007/3-540-45014-9_1) |
| `wilson1927probable` | Intervalo binomial de Wilson usado para el recobrado selectivo. | No corrige sesgo de muestreo, selección de modelo ni dependencia entre chunks. | [DOI](https://doi.org/10.1080/01621459.1927.10502953) |
| `sklearn2026averageprecision` | Definición exacta de `average_precision_score` empleada por los artefactos. | Documenta el cálculo; no es evidencia de desempeño. | [scikit-learn API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) |

### Precisión promedio frente a área PR

La documentación oficial de
[`average_precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
define

\[
AP = \sum_n (R_n-R_{n-1})P_n.
\]

Es una suma ponderada de precisiones por incrementos de recall y no usa
interpolación lineal. Por ello es distinta de `auc(recall, precision)`, que
calcula un área trapezoidal. En el paper se recomienda escribir una vez:
“average precision (AP; denominada `PR-AUC` en los artefactos históricos)” y
usar luego AP.

## Software científico

| Clave BibTeX | Uso | Fuente primaria |
|---|---|---|
| `pedregosa2011sklearn` | Modelos clásicos, métricas, calibración y utilidades de selección. | [JMLR](https://www.jmlr.org/papers/v12/pedregosa11a.html) |
| `sculley2015technicaldebt` | Riesgos sistémicos de dependencias de datos y configuración en pipelines de ML. | Motiva probar el flujo completo; no define la longitud ni el diseño experimental local. | [NeurIPS](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) |
| `breck2017mltestscore` | Pruebas de datos, infraestructura, modelos y monitoreo para madurez de sistemas ML. | Sustenta controles ligeros de integración; el nombre y alcance exacto del smoke test son decisiones locales. | [Google Research](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) |
| `wolf2020transformers` | Carga y entrenamiento de Transformers y Qwen. | [ACL Anthology](https://aclanthology.org/2020.emnlp-demos.6/) |
| `paszke2019pytorch` | Entrenamiento e inferencia acelerados, incluida GPU. | [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| `ytdlp2026` | Recuperación de metadatos y pistas VTT sin descargar el video. | La revisión exacta instalada no quedó persistida en los cuadernos del corpus. | [Repositorio oficial](https://github.com/yt-dlp/yt-dlp) |
| `depoix2026transcript` | Fallback para recuperar transcripciones/subtítulos públicos. | La revisión exacta instalada no quedó persistida; solo aportó 33 chunks finales. | [Repositorio oficial](https://github.com/jdepoix/youtube-transcript-api) |

## Ontología y trazabilidad semántica

| Clave BibTeX | Qué respalda | Límite de la cita | Fuente primaria |
|---|---|---|---|
| `gruber1993ontology` | Una ontología como especificación explícita y portable de una conceptualización. | No implica que la taxonomía local sea ontología formal si no se expresan relaciones y restricciones. | [DOI 10.1006/knac.1993.1008](https://doi.org/10.1006/knac.1993.1008) |
| `w3c2012owl2` | Estándar OWL 2 para representar ontologías en la Web. | Es una recomendación técnica, no una evaluación empírica. | [W3C Recommendation](https://www.w3.org/TR/owl2-overview/) |
| `w3c2014turtle` | Sintaxis RDF 1.1 Turtle usada por el archivo de ontología. | Define serialización, no demuestra completitud semántica. | [W3C Recommendation](https://www.w3.org/TR/turtle/) |
| `w3c2009skos` | Propiedad `skos:definition` y vocabulario controlado. | SKOS no sustituye axiomas OWL ni validación con formas. | [W3C Recommendation](https://www.w3.org/TR/skos-reference/) |
| `wilkinson2016fair` | Principios FAIR para que artefactos y metadatos sean encontrables, accesibles, interoperables y reutilizables. | FAIR no equivale necesariamente a datos abiertos; deben respetarse licencias y privacidad. | [Scientific Data](https://www.nature.com/articles/sdata201618) |

## Contexto sociolingüístico peruano

| Clave BibTeX | Uso | Corrección o alcance | Fuente trazable |
|---|---|---|---|
| `zavala2007discurso` | Racismo cultural y discurso en el Perú contemporáneo. | Capítulo, pp. 333--370; editor correcto: Teun A. van Dijk. | [Catálogo y metadatos](https://dialnet.unirioja.es/servlet/libro?codigo=270993) |
| `zavala2017racismo` | Relación entre lenguaje, racialización y prácticas sociales peruanas. | Libro editado, no conjunto de datos. | [Repositorio PUCP](https://repositorio.pucp.edu.pe/index/handle/123456789/170315) |
| `almeida2022motoso` | Motoseo, terruqueo y racialización en política y redes peruanas. | DOI corregido a `10.18800/lexis.202202.002`; orden correcto: Zavala y Almeida. | [Lexis / PUCP](https://revistas.pucp.edu.pe/index.php/lexis/article/view/26332) |
| `branez2012amixer` | Identidades “amixer” y racismo cultural en espacio virtual peruano. | Es tesis de licenciatura, no artículo de revista. | [Repositorio de Tesis PUCP](https://tesis.pucp.edu.pe/repositorio/handle/20.500.12404/1618) |
| `callirgos1993racismo` | Marco peruano clásico sobre alteridad y racismo. | Fuente monográfica; usar para contexto, no para inferir prevalencias actuales. | Registro bibliográfico en la edición DESCO de 1993. |
| `portocarrero2009racismo` | Racismo, mestizaje y jerarquías sociales en Perú. | La entrada corresponde a la reimpresión de 2009 de la edición de 2007. | [Catálogo bibliográfico](https://biblioteca.unasam.edu.pe/bib/17017) |
| `vich2018dinamicas` | Síntesis de las dinámicas culturales de racismo propuestas por Portocarrero. | DOI y páginas corregidos: `10.18800/debatesensociologia.201802.008`, pp. 219--232. | [Debates en Sociología / PUCP](https://revistas.pucp.edu.pe/index.php/debatesensociologia/article/view/22090) |
| `thakur2025quechua` | Moderación y desigualdad lingüística en plataformas para usuarios de quechua. | Informe institucional con metodología propia; no estudia directamente el clasificador local. | [Center for Democracy & Technology](https://cdt.org/wp-content/uploads/2025/06/2025-Quechua-Report-Spanish-final-1.pdf) |
| `monge2023violencia` | Violencia de género e insultos feminizantes en conversación digital. | Se corrigieron tercer autor, páginas y DOI. | [Orkopata](https://revistas.inudi.edu.pe/ro/es/article/view/425) |
| `salem2016amixer` | Insulto, parodia, ironía y humor en la construcción racializada de `amixer`. | Estudio de caso; no convierte todo humor en daño. | [Revista Chilena de Antropología Visual](https://www.antropologiavisual.cl/amixer-esta-en-facebook-una-investigacion-de-la-choledad-virtual) |
| `albornoz2018conocer` | Modalidades de violencia de género en línea, denuncia y resistencia en Perú. | Informe institucional; no reemplaza una adjudicación experta del corpus. | [Hiperderecho](https://hiperderecho.org/tecnoresistencias/reporte/) |
| `rottenbacher2012homofobia` | Homofobia y prejuicio hacia grupos transgénero en una muestra universitaria de Lima. | Muestra no probabilística y de alcance limitado; no mide lenguaje de YouTube. | [Redalyc / Pensamiento Psicológico](https://www.redalyc.org/pdf/801/80124028002.pdf) |
| `lovon2022lesbofobia` | Léxico lesbofóbico e insulto por orientación sexual en foros peruanos. | Estudio cualitativo de siete lexemas; no cubre toda homofobia/transfobia. | [Whatever](https://whatever.cirque.unipi.it/index.php/journal/article/view/156) |
| `defensoria2021violenciaenlinea` | Amenazas, doxeo, difusión íntima no consentida y violencia de género en línea en el marco peruano. | Fuente institucional; no define por sí sola las salidas del modelo. | [Defensoría del Pueblo](https://www.defensoria.gob.pe/wp-content/uploads/2021/08/Documento-de-trabajo-01-Violencia-de-g%C3%A9nero-contra-las-mujeres-en-l%C3%ADnea.pdf) |

## Documentación operativa dinámica

Estas fuentes sirven para reproducibilidad y procedencia de artefactos. No deben
usarse como sustituto de evidencia científica revisada por pares.

| Artefacto | Fuente oficial | Registro necesario en el paper o manifiesto |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | [Tarjeta del modelo](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Identificador, revisión resuelta, licencia y fecha de descarga. |
| `intfloat/multilingual-e5-small` | [Tarjeta del modelo](https://huggingface.co/intfloat/multilingual-e5-small) | Identificador, revisión resuelta y adaptación a clasificación. |
| `Qwen/Qwen3-0.6B-Base` | [Tarjeta del modelo](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Checkpoint base, revisión, licencia, configuración LoRA y cabeza de clasificación. |
| `Qwen/Qwen3-4B` | [Tarjeta del modelo](https://huggingface.co/Qwen/Qwen3-4B) | Revisión `1cfa9a7208912126459214e8b04321603b3df60c`, licencia y rol de preanotación local en Colab. |
| `qwen3.5:4b` en Ollama | [Tarjeta del modelo](https://ollama.com/library/qwen3.5:4b) | Nombre, digest resuelto por `/api/tags`, cuantización y fecha de ejecución; la tarjeta no valida etiquetas. |
| Salida estructurada de Ollama | [Documentación oficial](https://docs.ollama.com/capabilities/structured-outputs) | JSON Schema enviado, validación Pydantic, temperatura y reintentos del adaptador local. |
| PEFT 0.18.0 | [Referencia LoRA](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora) | Versión, rango, `target_modules`, `lora_alpha` y dropout; los valores concretos son locales. |
| Google Colab para VS Code | [Wiki oficial](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds) | Versión de la extensión, soporte de `drive.mount`, hardware observado y SHA-256 del bundle. |
| Google Colab y Drive montado | [Preguntas frecuentes oficiales](https://research.google.com/colaboratory/faq.html) | Autorización de `drive.mount`, carácter efímero de la VM, `bundle_id`, SHA-256 verificados y ruta publicada. |
| DeepSeek V4 Flash/Pro | [Lanzamiento](https://api-docs.deepseek.com/news/news260424/) y [modelos](https://api-docs.deepseek.com/quick_start/pricing/) | IDs exactos, fecha de acceso, rol de preanotación/revisión y protocolo humano posterior. |
| yt-dlp | [Repositorio oficial](https://github.com/yt-dlp/yt-dlp) | Versión usada, opciones de subtítulos e idiomas solicitados. |
| youtube-transcript-api | [Repositorio oficial](https://github.com/jdepoix/youtube-transcript-api) | Versión usada y condición de fallback. |

## Declaración mínima del corpus propio

El paper debe incluir, como mínimo:

- nombre y versión del corpus, fecha de corte y hash del manifiesto;
- criterio de selección de videos y advertencia de enriquecimiento intencional;
- idiomas aceptados (`es-PE`, `es-419`, `es`) y distinción entre subtítulos
  manuales y automáticos;
- exclusión de videos sin subtítulos y sesgo de selección resultante;
- reglas de limpieza, fragmentación temporal, deduplicación y control de fuga;
- protocolo de preanotación, segunda revisión y adjudicación humana;
- separación por video entre entrenamiento, validación y prueba;
- taxonomía histórica de cinco etiquetas y transformación a cuatro categorías
  operativas;
- licencias, privacidad, condiciones de acceso y límites de redistribución;
- advertencia de que el corpus enriquecido no estima prevalencia natural en
  YouTube peruano.

## Mantenimiento

Al agregar una fuente:

1. preferir DOI, editorial, repositorio institucional, ACL Anthology, PMLR,
   NeurIPS, OpenReview o documentación oficial;
2. confirmar autores, título, año, volumen, número y páginas en la fuente
   primaria;
3. usar una clave BibTeX estable `apellidoAñoConcepto`;
4. registrar aquí la afirmación respaldada y su límite;
5. compilar BibTeX y revisar advertencias antes de citarla;
6. no añadir una referencia solo para aumentar el número de citas.
