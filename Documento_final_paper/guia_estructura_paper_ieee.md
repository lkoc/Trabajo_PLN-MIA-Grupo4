# Estructura del paper IEEE con metodologia DSR

> **Enmienda de contrato v2 (2026-08-05).** El paper debe distinguir el baseline ejecutado de cuatro daños con `SEGURO` derivado del flujo activo de cinco salidas aprendidas. Las métricas v2 permanecen pendientes hasta reentrenar y recalibrar.

Esta guia define la arquitectura narrativa del articulo final. Debe conservar el formato `IEEEtran` en hoja A4 mediante la opción de clase `a4paper` y presentar evidencia del artefacto construido, sus iteraciones y sus limites. La extension final depende de las reglas de la convocatoria o del curso; nunca debe reducirse la tipografia para forzar el numero de paginas. La auditoría final debe comprobar que el PDF mide físicamente 210 × 297 mm; no basta con que el documento declare A4 en el fuente.

## Hilo conductor

La revisión manual no escala al volumen de fragmentos de videos; el lenguaje peruano y la ambigüedad dificultan reconocer daño. El aporte central es un artefacto semiautomático y auditable que, con modelos compactos y recursos asequibles, prioriza casos, entrega un diagnóstico temporal y mantiene la decisión humana. DSR organiza su construcción y evaluación iterativa.

## Secciones obligatorias

### Titulo, autores e institucion

El título debe identificar el problema o fenómeno, el artefacto o enfoque, el objeto/contexto y el alcance. Incluya la metodología solo si es central, diferencia el aporte o la exige la convocatoria; no convierta esos componentes en una lista que vuelva ilegible el título. Audítelo además por fidelidad con lo ejecutado, términos útiles para recuperación bibliográfica, ausencia de siglas oscuras, concisión y prudencia.

El título actual —«Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural»— cumple los criterios principales: identifica el fenómeno y contexto de moderación y engloba las familias clásica y neuronal mediante la denominación completa de procesamiento del lenguaje natural. «Semiautomática» ya diferencia el artefacto de una moderación autónoma; la intervención humana se precisa en el resumen y el método sin alargar el título. No nombra DSR, pero esa omisión no es un defecto: la metodología se desarrolla en el cuerpo. La auditabilidad permanece como propiedad secundaria del artefacto.

Incluya los nombres oficiales de los cuatro autores, programa, institución, curso y periodo. Correos, ORCID y facultad solo se agregan cuando hayan sido confirmados.

### Resumen y palabras clave

Escriba un unico parrafo autocontenido de aproximadamente 150 a 250 palabras, sin citas ni siglas sin definir. Debe contener:

1. situacion actual y brecha;
2. objetivo del trabajo;
3. datos, metodologia DSR y familias comparadas;
4. resultado cuantitativo principal;
5. conclusion de uso y limite operacional.

No use “resultado esperado”. El trabajo ya tiene resultados. Las palabras clave deben facilitar recuperacion bibliografica y no repetir literalmente todo el titulo.

### Introduccion

Explique la totalidad del trabajo sin sustituir las secciones posteriores. Presente la brecha entre situacion actual y deseada en tres niveles:

- **problema real:** el volumen de contenido hace lenta y costosa la revision manual;
- **problema subyacente:** el daño depende de contexto, ironia, modismos y formas locales de discriminacion, mientras las etiquetas son escasas y desbalanceadas;
- **problema tecnologico:** falta un artefacto reproducible que segmente subtitulos, compare clasificadores, calibre la derivacion a revision y entregue evidencia trazable.

Cierre con aquello que el estudio examina, el objetivo general, las contribuciones demostradas y un mapa breve del articulo. La indagación puede formularse aparte, pero no debe sustituir la redacción declarativa del problema.

### Bases teoricas y antecedentes

Separe conceptos de antecedentes experimentales. Cubra moderacion de contenido y lenguaje abusivo, clasificacion multietiqueta, representaciones TF-IDF, modelos lineales, Transformers, ajuste fino, LoRA, jerarquias, calibracion, precision promedio (AP), clasificacion selectiva y participacion humana. Incluya literatura sobre español y contexto peruano.

Cada algoritmo, arquitectura o mejora tomada de la literatura debe citar su fuente fundacional o tecnica primaria. Las afirmaciones sobre ventajas, sesgos o limitaciones tambien requieren respaldo. El corpus construido y presentado por primera vez en el mismo artículo se documenta narrativamente como resultado propio mediante tablas, manifiestos, anexos y rutas de artefactos; no se crea una autorreferencia bibliográfica para citar a los propios autores. Sí deben citarse los corpus externos reutilizados y cualquier publicación previa e independiente del corpus propio.

La taxonomía de daño debe tener antecedentes lingüísticos, sociales y de moderación pertinentes al español y al contexto peruano. Esos antecedentes orientan la definición, pero no convierten las etiquetas del proyecto en tipos jurídicos. Si se formula una afirmación legal, debe apoyarse en una norma vigente o en una fuente jurídica especializada y debe distinguirse de la decisión operacional usada para entrenar el clasificador.

#### Aparato de citas y procedencia

El artículo debe permitir reconocer qué proviene de la literatura, qué se observó en los artefactos y qué decidió el equipo. Aplique estas reglas en todas las secciones:

- cite toda idea, definición, afirmación teórica, explicación del estado del arte, ventaja, riesgo o limitación que proceda de otra fuente;
- cite el aporte fundacional o la fuente técnica primaria de cada algoritmo, arquitectura, pérdida, calibrador, métrica o procedimiento publicado que se haya usado;
- para un modelo preentrenado, cite tanto el trabajo que presenta la familia como la tarjeta o repositorio del checkpoint exacto, con identificador y revisión inmutable cuando esté disponible;
- para estándares y funciones cuyo comportamiento depende de una especificación o implementación, cite el estándar, la documentación oficial versionada o el artículo de software correspondiente;
- presente cifras, configuraciones y resultados propios con su tabla, manifiesto, cuaderno, CSV o JSON de origen; una cita externa no sustituye esa evidencia;
- identifique como “decisión local”, “heurística de este trabajo” o “regla operacional del prototipo” toda elección propia que no reproduzca un método publicado;
- parafrasee con redacción propia y cite la fuente; una cita no justifica copiar la formulación de un resumen, una tarjeta de modelo o una documentación.

Mantenga durante la edición una matriz de trazabilidad con la forma `afirmación → fuente externa pertinente → artefacto interno → ubicación en el artículo`. No es necesario publicar la matriz completa, pero cada afirmación importante debe poder reconstruirse con esa ruta.

### Problemas y objetivos

Denomine «problema general» y «problemas específicos» a las situaciones que separan el estado actual del deseado. Escríbalos como enunciados declarativos y verificables, no como preguntas. Si se explicita una indagación, manténgala separada de los problemas y asegure que pueda responderse con la evidencia disponible. El objetivo general debe ser diseñar y evaluar el artefacto de moderación semiautomática: el modelo alerta y diagnostica; el supervisor decide.

Los objetivos especificos deben corresponder a las etapas realizadas:

1. construir y depurar un corpus trazable;
2. definir y revisar la taxonomia de daño;
3. entrenar y comparar familias clasicas, Transformer y Qwen en diseños planos y jerarquicos;
4. seleccionar y calibrar sin usar test para decidir;
5. integrar los ganadores en un prototipo con revision humana y datos reutilizables.

Incluya una tabla o figura que relacione cada problema especifico con su objetivo, actividad DSR, evidencia y seccion de resultados.

### Metodologia DSR

Describa las iteraciones de construccion y evaluacion, no solo el pipeline final:

1. identificacion del problema y requisitos del moderador;
2. corpus integrado, segmentacion y esquema de etiquetado;
3. pseudoetiquetado Flash/Pro y revisión final asistida con procedencia documentada;
4. baselines historicos de cinco daños y consolidacion del corpus;
5. Transformer y diseños jerarquicos de cinco daños;
6. baseline reproducible de cuatro daños mediante `ACOSO_AMENAZA`;
7. transición al contrato v2 de cinco salidas, con `SEGURO` aprendido y exclusivo;
8. modelos en `flujo/03_entrenamiento/` y demostración en `flujo/04_produccion/` con retroalimentación humana.

Para cada iteracion indique entrada, intervencion, artefacto, criterio de evaluacion, resultado y aprendizaje que motivo la siguiente iteracion.

#### Datos y etiquetas

Documente videos y chunks, deduplicación, procedencia de etiquetas, precedencia revisión humana final > Pro > Flash, exclusión de casos sin resolver, balanceo y partición por `video_id`. Describa la revisión humana por su función y cantidad, sin identificadores internos. Distinga:

- corpus integrado completo;
- muestra comparativa 4:1;
- train, validation y test;
- historial de contratos de daño y contrato activo v2 de cinco salidas, incluida `SEGURO`;
- etiquetas operativas, etiquetas finas auxiliares y flags transversales.

En el cuerpo y en las tablas principales, reporte el tamaño del corpus utilizado como un único total integrado. Describa cualitativamente las estrategias de adquisición y depuración necesarias para la reproducibilidad, pero no divida el tamaño por campañas o rondas salvo que comparar esas rondas sea una pregunta explícita del estudio. Aplique la misma regla a los conteos por categoría.

No presente el test como gold standard humano ni la muestra 4:1 como prevalencia natural de YouTube.

Explique para cada categoría su definición operacional, inclusión, exclusión, relación con las demás etiquetas y antecedente bibliográfico. Mantenga separados el vocabulario descriptivo, la política de moderación y las categorías legales. Documente también quién o qué produjo cada rótulo, qué sugerencia era visible durante la revisión y qué limitaciones impiden tratarlo como anotación humana independiente.

La reconstrucción histórica de la taxonomía debe comenzar en `archivo/taxonomia_v1_3/para_equiquetado_LLM/`, sin tomar un solo archivo como explicación suficiente. El contrato vigente se reconstruye desde `config/taxonomia_v2.json` y `docs/TAXONOMIA_V2.md`. Revise y relacione:

- `taxonomia_moderacion.csv`: vocabulario fino y agrupación didáctica; su columna `categoria=ACOSO` no define por sí sola el contrato preliminar de cinco salidas ni el activo de cuatro;
- `clasificacion_moderacion_peru.md` v1.3: criterios, ejemplos y bibliografía que vio el etiquetador;
- `PROMPT_ETIQUETADO_LLM.md`: esquema de salida e invariantes de labels, flags, confianza y revisión;
- `chunks_para_etiquetar.json`: objetos NDJSON pese a la extensión `.json`;
- `cgt_labeled_chunks_parte_0001`–`0003.jsonl`: prueba consecutiva de 60 filas, no muestra representativa ni referencia `gold`;
- regla de derivación: el mapeo fino produce cuatro daños y estados seguros explícitos; acoso personal y amenaza se unen mediante OR/máximo, y `SEGURO` constituye la quinta salida exclusiva. Los nombres de scripts pertenecen a la guía técnica, no al cuerpo del artículo.

Las copias archivadas de CSV y guía dentro de `archivo/taxonomia_v1_3/para_equiquetado_LLM/` son snapshots históricos. Para describir el contrato v2, dé prioridad a `config/taxonomia_v2.json`, `src/moderacion_peru/` y sus manifiestos; use el archivo solo para reconstruir qué instrucciones recibió el anotador original.

Reporte la estructura exacta como **12 fenómenos finos de daño + 2 estados seguros + 3 flags transversales**. Los doce fenómenos se agrupan 5/2/2/3 en `RACISMO_DISCRIMINACION`, `ACOSO_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es un estado derivado; los flags activan revisión y no son daños. La unión de acoso personal y amenaza, los umbrales 0,65/0,70 y la exigencia de acompañar cada flag con daño son reglas locales.

Construya una matriz `salida activa → etiqueta fina → definición académica → inclusión → exclusión → fuente general → fuente peruana/institucional → decisión local`. Use “taxonomía operativa informada por literatura y fuentes institucionales”; no use “taxonomía validada por expertos” mientras no exista adjudicación experta peruana documentada. Exponga las contradicciones históricas de la guía —amenaza explícita/implícita, ataque aislado/acoso repetido, flag sin daño y uso/mención de discurso reportado— en la auditoría o limitaciones; no las convierta en definiciones académicas.

Conserve la guía v1.3 como evidencia histórica. La versión v2 y su hash viven aparte; no se sobrescribe el prompt que originó anotaciones. El flujo activo crea campañas inmutables y no contiene la celda destructiva del cuaderno histórico 03.

#### Experimentos y arquitecturas

Una matriz debe inventariar al menos:

- Dummy, Complement Naive Bayes, regresion logistica, SVM, Gradient Boosting con SVD y fastText;
- Paraphrase Multilingual MiniLM y Multilingual E5-small;
- Qwen3-0.6B con LoRA y supervision auxiliar;
- modelos planos, cascadas binarias, jerarquias compartidas y cabezas multitarea;
- reutilización de encoders compatibles, cabeza nueva de cinco salidas y recalibración de todos los umbrales.

Explique por que se probo cada familia y que hipotesis o necesidad de ingenieria atendia.

Registre el nombre completo del modelo, repositorio, checkpoint, revisión, licencia declarada y linaje de inicialización. Las citas a MiniLM, E5 o Qwen como familias no reemplazan la referencia al checkpoint exacto. Si una versión de biblioteca no quedó persistida, indíquelo como limitación en vez de asignar una versión retrospectivamente.

#### Protocolo de evaluacion

Defina las metricas exactamente y aclare que el campo historico `PR-AUC` se calculo como precision promedio (AP), no como area trapezoidal. AP macro de daño es el criterio primario de seleccion; F1, precision, recall, falsos negativos y tasa de revision complementan la interpretacion. Explique:

- agrupacion por video y ausencia de fuga entre splits;
- ajuste, seleccion y calibracion con entradas de validation;
- cualquier exposicion previa a test, aunque la regla automatica no use sus metricas;
- bootstrap pareado por video cuando corresponda;
- diferencia entre recall micro de etiquetas y recall binario de cualquier daño.

### Resultados

Organice los resultados por preguntas, no por orden de ejecucion de celdas:

1. resultado de construcción del corpus y embudo de etiquetado/revisión final;
2. herramientas y hardware: CPU local, Colab/CUDA, alcance de la bitácora L4 y compatibilidad de inferencia con CPU;
3. aprendizaje de la iteración preliminar de cinco categorías;
4. comparación común de cuatro daños por familia;
5. efecto de cascadas y jerarquías frente a sus planos;
6. selección de la época Qwen y política de recall objetivo;
7. auditoría por etiquetas finas y flags;
8. prototipo operativo y trazabilidad de la revisión;
9. estado pendiente del reentrenamiento y evaluación del contrato v2.

La tabla principal debe derivarse de `comparacion_todos_modelos_4.csv`. Reporte nombres completos de metricas, tamaño y split; no mezcle resultados de test ampliado, test historico y test comun 4:1.

En el texto, las tablas, las figuras y el resumen, presente métricas y magnitudes estimadas con dos cifras significativas. Use los valores completos del artefacto para calcular, ordenar y seleccionar, y redondee solo la salida editorial. Mantenga exactos los conteos, el tamaño muestral, los años, las versiones, los identificadores, los hashes y los parámetros fijados por protocolo. Las constantes físicas pueden conservar la precisión que requiera su uso.

### Discusion

Interprete por que Qwen plano obtuvo la mayor estimacion puntual por el criterio fijado, por que E5 puede conservar mayor recall binario y por que los diseños jerarquicos no justificaron reemplazar sus referencias planas. Separe evidencia observada de explicaciones plausibles. Compare con antecedentes sin afirmar superioridad estadistica ni fuera del corpus evaluado.

### Limitaciones y trabajo futuro

Declare, como minimo:

- etiquetas mayormente asistidas por LLM y ausencia de holdout humano ciego;
- adjudicacion sugerida susceptible a anclaje y falta de doble anotacion/kappa;
- prevalencia artificial 4:1 y muestreo dirigido;
- reutilizacion historica del test entre iteraciones;
- variacion entre semillas neuronales no cuantificada;
- alcance textual, dependencia de subtitulos y sesgo por canales;
- validez externa, privacidad, deriva y desempeño por subgrupos pendientes.

Proponga un holdout prospectivo de prevalencia natural con doble anotacion, pruebas conductuales/adversariales, nuevas semillas, calibracion por subgrupo, monitoreo de deriva y reentrenamiento controlado con revisiones de produccion.

### Conclusiones

Verifique en una matriz interna que la conclusión responde al objetivo general y a cada objetivo específico, pero no escriba «objetivo», `O1`/`O2` ni fórmulas de cumplimiento en la prosa publicada. Integre las respuestas en una narración natural. Enuncie primero la decisión operacional positiva: se logró un asistente semiautomático para priorización, navegación temporal y diagnóstico preliminar con decisión humana. Luego indique, para cada distancia entre la situación inicial y la deseada, si se eliminó, se redujo o permanece. Cierre cada situación pendiente con una recomendación, una acción o una línea concreta de trabajo futuro, y delimite que bloqueo o sanción sin revisión no se evaluaron.

### Disponibilidad, etica y referencias

En el cuerpo publicado, indique únicamente qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorio se encuentran; incluya los enlaces a GitHub y al Drive de acceso controlado. Reserve SHA, commits y hashes concretos para manifiestos técnicos externos al cuerpo. Explique por separado las restricciones sobre subtítulos, datos personales y contenido potencialmente sensible. Denomine «corpus» de manera consistente al producto de datos del proyecto y remita sus cifras a tablas y artefactos internos, sin autocita bibliográfica. Cite mediante `\cite{clave}` las fuentes externas pertinentes. No use canales de YouTube como sustituto de literatura científica.

Documente por separado la licencia del código, de cada modelo base, de los adaptadores, de los datos y del frontend. No infiera que una licencia de modelo autoriza redistribuir subtítulos ni que el acceso público equivale a permiso de redistribución. Cuando no exista una licencia confirmada, limite el repositorio técnico a manifiestos, identificadores, hashes, estadísticas o instrucciones de reconstrucción que sí puedan compartirse, sin trasladar esas huellas a la declaración del artículo. Registre también las condiciones de uso de las herramientas de adquisición, el control de acceso a Drive, los riesgos para personas mencionadas y el procedimiento para retirar o corregir material.

## Figuras prioritarias

- Brecha en los tres niveles y problemas/objetivos.
- Ciclo DSR con las iteraciones ejecutadas.
- Pipeline de datos y embudo Flash → Pro → revisión final asistida.
- Evolución del baseline de cuatro daños al contrato v2 de cinco salidas aprendidas.
- Matriz de familias y estructuras experimentales.
- Separacion train/validation/test y regla contra fuga.
- Comparacion cuantitativa final y calibracion Qwen.
- Arquitectura 05: texto/YouTube, tres modelos, consenso 2 de 3, revision y almacenamiento.

Cada figura debe ser vectorial o de resolucion suficiente, citarse antes o inmediatamente despues de aparecer y conservar una fuente reproducible dentro de la carpeta final.

Reserve espacio entre nodos, grupos, flechas y bordes. En diagramas de flujo, ontologías y arquitecturas use rutas ortogonales —segmentos verticales y horizontales— cuando mejoren la lectura; evite diagonales arbitrarias. Ninguna línea debe atravesar cajas, texto, leyendas u otras etiquetas, y los cruces inevitables deben resolverse cambiando la disposición antes de añadir más trazos. Las etiquetas deben conservar un tamaño legible en dos columnas y en la proyección del Beamer, sin depender del zoom.

En cada diagrama de cajas, ajuste en conjunto la fuente, el ancho, el alto y la separación de los nodos; recalcule después las rutas ortogonales. No acepte cajas solapadas, texto fuera de sus límites ni flechas que invadan nodos. Para gráficos que comparan pocos modelos o métricas, pruebe una versión de una columna, reduzca los espacios entre grupos y mantenga visibles las etiquetas y los valores.

Después de compilar, inspeccione visualmente todas las páginas del PDF y todas las diapositivas del Beamer a tamaño de lectura normal. Revise recortes, solapamientos, flechas ambiguas, texto fuera de cajas, contraste, escala de grises, leyendas, numeración y correspondencia entre caption y figura. La compilación sin errores no sustituye esta inspección.

## Tablas prioritarias

- Problemas, objetivos, actividades DSR y evidencia.
- Corpus, fuentes de etiqueta y particiones.
- Iteraciones y experimentos de cinco/cuatro daños.
- Configuracion resumida de las tres familias.
- Resultados finales con definiciones de metricas.
- Limitaciones y controles de validez.

Si una tabla apaisada o muy ancha exige letra pequeña, reduzca columnas, parta encabezados y disponga cada observación en dos renglones —identificación y comparación principal primero; campos complementarios después—. Divida la tabla si todavía no puede leerse al tamaño final. Nunca compacte a costa de la legibilidad.

## Apéndices

En el orden editorial usual de IEEE, ubique los apéndices después del cuerpo y antes de los agradecimientos y las referencias, salvo que la publicación de destino indique otra secuencia. Inicie cada apéndice en una página nueva y titúlelo de manera que el lector sepa exactamente qué contiene. Cite cada apéndice desde el cuerpo, en el lugar donde amplía datos, método, resultados, trazabilidad o disponibilidad; una tabla o figura etiquetada dentro del apéndice no reemplaza esa referencia entrante. Puede cambiar temporalmente a una columna para prompts, ejemplos o tablas que lo necesiten, pero debe restaurar el formato principal antes de las referencias. Tras compilar, compruebe saltos, páginas vacías, numeración y referencias cruzadas.

En la bibliografía final, aplique la convención de mostrar los tres primeros autores y «et al.» cuando una obra tenga más de tres autores. Mantenga todos los nombres en `referencias.bib` y configure la abreviación en el estilo bibliográfico; no recorte manualmente los metadatos.
