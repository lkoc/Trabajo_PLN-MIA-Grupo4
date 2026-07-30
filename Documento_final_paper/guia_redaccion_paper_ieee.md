# Guia de redaccion y control de evidencia

El articulo debe leerse como investigacion aplicada terminada: directo, humano, verificable y prudente. Esta guia complementa la estructura IEEE/DSR y evita que resultados historicos, propuestas o supuestos aparezcan como hechos. La meta editorial es una prosa propia y natural; no se debe prometer que el texto superará detectores de IA ni usar esos detectores como prueba de autoría o ausencia de plagio.

## Principios de redaccion

- Use lenguaje simple para explicar el problema y lenguaje tecnico solo donde aporta precision.
- Prefiera oraciones con una idea principal, verbos concretos y sujetos identificables. Divida una frase cuando acumule antecedentes, método, resultado y conclusión.
- Evite fórmulas grandilocuentes, adjetivos promocionales, metáforas innecesarias, repeticiones de cierre y transiciones que no añaden información.
- Escriba en pasado para actividades ejecutadas, presente para hechos que muestran los artefactos y futuro solo para trabajo pendiente.
- Distinga observacion, interpretacion y recomendacion.
- Defina toda sigla y todo termino especializado en su primera aparicion.
- Use “fragmento (chunk)” al introducir la unidad de analisis; despues mantenga un termino consistente.
- Escriba con tildes y revise los nombres oficiales de personas e institucion.
- Evite tono publicitario, causalidad no demostrada y generalizaciones fuera del corpus.
- Presente métricas y demás magnitudes estimadas con dos cifras significativas. Mantenga exactos conteos, tamaño muestral, años, versiones, identificadores, hashes y parámetros fijados por protocolo; las constantes físicas pueden conservar la precisión requerida.

## Título y auditoría editorial

Evalúe el título con nueve preguntas:

1. ¿identifica el problema o fenómeno?;
2. ¿hace visible el artefacto o enfoque?;
3. ¿menciona la metodología cuando es distintiva o exigida?;
4. ¿delimita objeto, población o contexto?;
5. ¿expresa diseño, evaluación, comparación u otro propósito cuando aporta claridad?;
6. ¿mantiene un alcance prudente?;
7. ¿incluye términos útiles para búsqueda?;
8. ¿es conciso y evita siglas oscuras?;
9. ¿coincide con lo realmente ejecutado y concluido?

No es obligatorio incluir literalmente todos los componentes. Para este trabajo, el título actual satisface problema/fenómeno, amplitud metodológica mediante modelos clásicos y neuronales de procesamiento del lenguaje natural, contexto, operación supervisada, recuperación y prudencia. Los subtítulos se explican en el resumen y el método; no es necesario enumerarlos en el título. La ausencia de «DSR» es aceptable porque no reduce su identificación, y la auditabilidad se conserva como propiedad secundaria en el cuerpo.

## Contribuciones demostrables

La introduccion puede cerrar con contribuciones como las siguientes, siempre ajustadas a la evidencia final:

1. un corpus trazable de subtítulos públicos de YouTube peruano con pseudoetiquetado escalonado y revisión final asistida;
2. una taxonomia multietiqueta que evoluciono de cinco a cuatro daños y conserva señales finas/transversales para supervision o auditoria;
3. una comparacion comun de modelos clasicos, Transformers compactos y Qwen con alternativas planas y jerarquicas;
4. una politica de seleccion/calibracion basada en validation y una evaluacion final separada en test;
5. un artefacto reproducible para texto o videos con subtitulos, comparacion/consenso, revision humana y estadisticas reutilizables.

No describa como contribucion algo que solo aparece planificado.

## Trazabilidad de afirmaciones

Cada afirmación importante, no solo cada cifra, debe poder anotarse internamente con esta ficha:

```text
afirmación | tipo | fuente externa | artefacto interno | campo/tabla | split | fecha/hash | criterio
```

El campo `tipo` distingue al menos `idea externa`, `definición`, `método publicado`, `decisión local`, `dato del corpus`, `resultado` e `interpretación`. La ruta mínima es `afirmación → fuente → artefacto`; cuando la afirmación sea puramente empírica, la fuente puede ser el artefacto interno, y cuando sea teórica debe incluir una fuente externa pertinente.

Reglas:

- La comparacion principal de cuatro daños procede de `resultados/metricas/comparacion_final_4/comparacion_todos_modelos_4.csv`.
- La seleccion del modelo o checkpoint debe atribuirse a validation; test solo describe el desempeño congelado.
- Indique siempre si la cifra pertenece al test historico, ampliado o comun 4:1.
- No denomine “accuracy” a `exact_match`, ni “recall” sin aclarar si es por etiqueta, micro o binario de cualquier daño.
- Use dos cifras significativas para métricas y magnitudes estimadas en texto, tablas y gráficos. Calcule y compare con la precisión completa del artefacto fuente; redondee solo la presentación.
- No redondee conteos exactos, tamaño muestral, años, versiones, identificadores, hashes ni parámetros fijados por protocolo. Las constantes físicas pueden conservar la precisión que exija su uso.
- No infiera significancia si el intervalo incluye cero ni convierta una comparacion descriptiva en prueba causal.
- Una referencia debe respaldar la afirmación exacta junto a la que aparece. No acumule al final de un párrafo citas que podrían corresponder de manera ambigua a varias ideas.
- Compruebe claves, autores, título, año, DOI/URL y revisión del checkpoint contra la fuente primaria antes de cerrar la bibliografía.
- En la lista final, los trabajos con más de tres autores deben mostrar los tres primeros seguidos de «et al.». Configure `IEEEtranBSTCTL` o el gestor equivalente y conserve la lista completa de autores en el archivo bibliográfico.

## Datos y etiquetado

Describa con precision la procedencia:

- la recoleccion de subtitulos publicos puede haberse realizado sin la API oficial de YouTube;
- el etiquetado incluye modelos Flash/Pro, reglas de confianza y revisión final asistida;
- la revisión humana asistida debe describirse por su función, cantidad y procedencia, sin trasladar identificadores internos de revisores al artículo;
- una propuesta Pro aceptada con la sugerencia visible no equivale a anotación humana ciega independiente;
- el test actual no es un gold standard humano;
- la muestra 4:1 se diseño para comparacion controlada y no estima prevalencia productiva.

No afirme que se uso Whisper, transcripcion ASR o comentarios si la version final de datos y sus manifiestos no lo demuestran. El artefacto 05 rechaza enlaces sin subtitulos descargables y no transcribe audio.

Cada etiqueta debe enlazarse con antecedentes pertinentes y con una definición operacional propia que indique inclusión, exclusión y relación con otras categorías. No presente `RACISMO_DISCRIMINACION`, `ACOSO_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` o `CONTENIDO_SEXUAL` como tipos legales. Una taxonomía de moderación sirve al contrato del artefacto; una calificación jurídica exige fuentes normativas o jurídicas y un análisis separado.

Use siempre el conteo completo: “14 etiquetas finas: 12 fenómenos de daño y dos estados seguros; además, tres flags transversales”. No escriba “14 fenómenos de daño”. Explique que `SEGURO` significa que ninguna salida cubierta se activó; no prueba ausencia universal de daño, pues el contrato no cubre, entre otros, fraude, autolesión, extremismo, desinformación o violencia gráfica.

Al describir la elección de categorías, separe cuatro capas:

1. **Tipología académica general:** blanco individual frente a grupo, daño explícito frente a implícito, identidad, insulto, doxeo, amenaza y agresión sexual.
2. **Evidencia peruana o institucional:** racialización por lengua, origen, clase y criterios culturales; violencia de género en línea; prejuicio hacia personas homosexuales y transgénero en Lima.
3. **Política de plataforma:** límites sobre contenido sexual explícito, sexualización no consentida y excepciones informativas/documentales.
4. **Decisión local:** nombres exactos, fusión 5→4, OR/máximo, umbrales, flags y ruta de revisión.

No llame “expertos peruanos” a todas las fuentes. Por ejemplo, el estudio de Monge-Olivarría y coautores procede de la Universidad Autónoma de Sinaloa y no acredita específicamente Twitter peruano. Una fuente sobre moderación en quechua respalda carencias de datos/contexto para esa lengua, no cada regla del español peruano. Una política de YouTube es autoridad de plataforma, no literatura académica.

Las tres salidas CGT de `para_equiquetado_LLM/` cubren solo 60 filas consecutivas, 51 seguras y una fracción de las etiquetas; no son una muestra aleatoria, una validación de la taxonomía ni un `gold standard`. El nombre `GPT-5 (Codex)` tampoco identifica por sí solo un checkpoint/API reproducible. Presente estos archivos como prueba de formato y flujo, no como evidencia de calidad de anotación.

No copie como hechos las expectativas por canal de `clasificacion_moderacion_peru.md`: son priors heurísticos sin validación documentada y pueden anclar al anotador. Tampoco repita una justificación académica generada por el LLM como si fuera una cita comprobada. La fuente debe verificarse en la bibliografía primaria y la definición final debe redactarse de nuevo.

Incluya las decisiones éticas que afectan la evidencia: uso de subtítulos públicos, minimización de datos, exposición a contenido sensible, riesgo de anclaje en la revisión asistida, control de acceso y límites de redistribución. Distinga las licencias del código, modelos base, adaptadores, corpus y recursos visuales. No deduzca una licencia ausente ni extienda la licencia de un componente a otro.

## Modelos, mejoras y citas

Use `\cite{clave}` y el estilo numerico IEEE. Cite fuentes primarias o fundacionales para los algoritmos que aparecen en el metodo: TF-IDF, regresion logistica, SVM, fastText, Transformer, MiniLM/Sentence-BERT, E5, Qwen3, LoRA, calibracion, precision promedio (AP), bootstrap y DSR. Cite tambien cada arquitectura, aumento o mejora tomada de un trabajo previo.

- No copie frases de abstracts, model cards o documentacion. Parafrasee y cite.
- No invente DOI, año, autores, revision de checkpoint ni licencia.
- Prefiera DOI, editorial, ACL Anthology, actas oficiales o repositorio oficial del modelo frente a agregadores.
- Si el corpus se presenta por primera vez como producto del artículo, documéntelo mediante la narrativa, las tablas, el manifiesto y los anexos internos; no cree una entrada bibliográfica para autocitar a los autores. Cite normalmente cualquier corpus externo o publicación previa e independiente.
- Si el corpus final integra varias rondas de adquisición, reporte en el artículo el total realmente utilizado y los totales finales por categoría. Explique la estrategia de adquisición sin desglosar tamaños por campaña, salvo que las campañas sean una variable analizada.
- Las URLs de canales describen la muestra; no constituyen fundamento cientifico.
- Si una fuente no respalda exactamente una afirmacion fuerte, atenue la afirmacion o busque otra fuente.
- Una idea o definición externa requiere cita aunque no incluya una fórmula ni el nombre de un autor.
- Una cita general de la familia no identifica el modelo usado: añada la tarjeta o repositorio del checkpoint exacto, su revisión inmutable y la licencia declarada.
- Cite estándares como Unicode o SHA y documentación oficial cuando una propiedad del procedimiento dependa de ellos. Para bibliotecas centrales, use el artículo de software y, si se describe una API concreta, su documentación versionada.
- No atribuya a HiAGM, BCE u otro antecedente una fórmula o ponderación creada por el equipo. Marque coeficientes, umbrales, reglas 2-de-3 y heurísticas de muestreo como decisiones locales cuando corresponda.

## Prevención de plagio y procedencia intelectual

La prevención se basa en trazabilidad y redacción propia, no en intentar satisfacer un detector automático:

1. lea la fuente completa pertinente y registre qué idea concreta se usará;
2. cierre la fuente y explique la idea con palabras acordes al argumento del artículo;
3. contraste la paráfrasis con el original para evitar conservar su estructura o secuencia distintiva;
4. coloque la cita junto a la idea y verifique que la fuente realmente la respalde;
5. use comillas y página solo cuando una formulación textual breve sea imprescindible;
6. conserve notas de procedencia para figuras adaptadas, taxonomías, fórmulas y decisiones de diseño.

No encadene sin atribución definiciones tomadas de varias fuentes. No traduzca literalmente un abstract para presentarlo como paráfrasis. Una baja similitud textual tampoco demuestra originalidad si la idea carece de cita, y una cita no vuelve legítima una copia extensa.

## Metodo

Cada procedimiento debe indicar entrada, proceso, salida y control contra sesgo o fuga. Ejemplo:

```text
El particionado agrupado recibio chunks con identificador de video, asigno cada video a una sola particion y produjo conjuntos sin videos compartidos. Los umbrales y la seleccion se fijaron en validation antes de evaluar test.
```

Explique por que se ejecuto cada familia, no solo sus hiperparametros. En la transicion cinco → cuatro daños, describa la union operativa, el warm start de cabezas cuando corresponda, el reentrenamiento y la recalibracion; no presente una union post hoc como equivalente a un nuevo ajuste ni la fusion como equivalencia conceptual o legal.

## Resultados y discusion

- Abra cada bloque con la pregunta u objetivo que responde.
- Presente primero la magnitud y luego su interpretacion.
- Compare modelos sobre el mismo split y contrato de etiquetas.
- Distinga AP (llamada `PR-AUC` en artefactos historicos) de F1 y recall; una mejora en una metrica no implica mejora total.
- Reporte tambien carga humana, tasa de revision y falsos negativos cuando la conclusion sea operacional.
- Explique que la epoca 2 de Qwen maximiza AP de validation, mientras la epoca 3 fue la eleccion operacional entre los mejores checkpoints al mismo objetivo de recall; declare que ya existia una evaluacion de test de la epoca 2.
- Declare que Qwen plano obtuvo mejores estimaciones puntuales que sus variantes jerarquicas en las metricas principales; no infiera significancia ni generalice a toda arquitectura jerarquica.
- No describa el frontend como validacion de eficacia. Es una demostracion de integracion y trazabilidad.

## Conclusiones

Enlace internamente las conclusiones con el objetivo general y cada objetivo específico, pero no numere esa correspondencia ni use «objetivo», `O1`/`O2` o «se cumplió» en el cierre publicado. Responda mediante una narración fluida. Una conclusión válida combina logro, evidencia y límite, e indica para cada distancia entre la situación inicial y la deseada si se eliminó, se redujo o permanece. Toda situación pendiente debe conducir a una recomendación, una acción o una línea concreta de trabajo futuro. Por ejemplo:

```text
La comparación común dio a Qwen plano la mayor estimación puntual de ranking de daño por el criterio fijado. La política selectiva materializa el propósito semiautomático al capturar 0,97 de los daños para alerta o revisión y entregar evidencia al supervisor. Después de ese logro deben declararse la carga de revisión del 60 %, la exposición previa a test y la ausencia de un gold standard humano ciego.
```

En la declaración de disponibilidad del artículo, enumere solo qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorio se encuentran. No incluya SHA o commits concretos en el cuerpo; consérvelos en manifiestos técnicos externos.

## Lenguaje recomendado

Preferir:

- “el sistema prioriza casos, entrega evidencia temporal y un diagnóstico preliminar al supervisor”;
- “el modelo fue seleccionado en validation y evaluado despues en test”;
- “el resultado es descriptivo bajo una prevalencia 4:1”;
- “la evidencia no mostro una mejora concluyente”;
- “el fragmento y sus scores permiten auditar la sugerencia”;
- “el uso defendible es un piloto en modo sombra”.

Evitar:

- “el sistema modera o elimina contenido automaticamente”;
- “el modelo entiende el contexto peruano”;
- “el algoritmo detecta la verdad”;
- “las etiquetas son completamente humanas”;
- “se demostró validez para producción autónoma”;
- “se uso Whisper” sin artefactos que lo prueben;
- “dos anotadores obtuvieron Cohen kappa” sin doble anotacion real;
- “sin APIs” como propiedad de todas las fases.

## Figuras y tablas

- Cite toda figura o tabla en el texto y explique el hallazgo que aporta.
- El caption debe ser autocontenido e indicar split o universo cuando corresponda.
- No use color como unico canal semantico; compruebe escala de grises y legibilidad en dos columnas.
- Evite capturas de notebooks. Genere figuras desde CSV/JSON o use TikZ/PGFPlots.
- Copie o regenere dentro del entregable los recursos necesarios para que compile sin depender de archivos ignorados por Git.
- No sature tablas con hiperparametros secundarios; remitalos a un repositorio o anexo reproducible.
- Si una tabla apaisada o muy ancha obliga a encoger la letra, reduzca columnas, use encabezados en dos líneas y distribuya cada observación en dos renglones antes de escalar. Divídala si sigue siendo ilegible.
- Para gráficos de pocos modelos o métricas, pruebe el ancho de una columna, reduzca espacios entre grupos y preserve etiquetas y valores legibles.
- Deje espacio suficiente entre cajas, flechas, grupos, etiquetas y márgenes; una figura compacta no debe convertirse en una figura ilegible.
- Prefiera rutas ortogonales verticales y horizontales en flujos, ontologías y arquitecturas. Ajuste fuente, ancho, alto y separación de las cajas; reordene los nodos para impedir cualquier solapamiento o que una línea cruce una caja, texto o etiqueta. No tape un cruce con color o grosor.
- Mantenga etiquetas breves, inequívocas y legibles al tamaño final de dos columnas y al proyectar el Beamer. Explique abreviaturas en el caption o la leyenda.
- Inspeccione visualmente el PDF y cada diapositiva del Beamer después de compilar. Revise recortes, solapamientos, contraste, texto fuera de cajas, flechas ambiguas y consistencia con los datos; una compilación exitosa no basta.

En el orden editorial usual de IEEE, los apéndices se colocan después del cuerpo y antes de los agradecimientos y las referencias, salvo instrucción distinta de la publicación. Cada apéndice debe comenzar en página nueva, llevar un título descriptivo de su contenido y recibir una referencia explícita desde el pasaje pertinente del cuerpo. Si ningún pasaje necesita el apéndice, intégrelo en el cuerpo o elimínelo. Si se cambia temporalmente a una columna para mejorar la lectura, restaure el formato principal antes de las referencias y revise que no aparezcan páginas vacías accidentales.

## Dos pases obligatorios

### Pase 1: validez cientifica

- rastrear todas las cifras y nombres de modelos;
- rastrear las ideas externas, definiciones, algoritmos, checkpoints, estándares y decisiones locales;
- verificar taxonomia y split;
- comprobar que la taxonomía tiene antecedentes y se presenta como contrato operacional, no como clasificación legal;
- comprobar que test no decide modelos ni umbrales;
- revisar citas, paráfrasis y procedencia intelectual sin afirmar que un detector automático certifica autoría o ausencia de plagio;
- comprobar la ruta afirmación → fuente → artefacto y las licencias/restricciones de cada componente;
- declarar pseudoetiquetado, muestreo dirigido y limitaciones humanas;
- comprobar que resultados y conclusiones responden a los objetivos;
- verificar en una matriz interna que el objetivo general y cada objetivo específico se responden, pero retirar del cierre publicado la palabra «objetivo», los códigos `O1`/`O2` y las listas de cumplimiento;
- comprobar que cada distancia inicial se clasifica como eliminada, reducida o pendiente, y que cada pendiente tiene una acción posterior;
- comprobar que el logro semiautomático aparece antes que los límites sobre producción autónoma; eliminar afirmaciones no sustentadas sobre Whisper y kappa.

### Pase 2: calidad editorial y visual

- corregir ortografia, tildes, consistencia terminologica y nombres propios;
- revisar titulo, resumen, palabras clave y orden de secciones;
- comprobar todas las referencias cruzadas, captions y bibliografia;
- comprobar que cada apéndice tenga al menos una referencia entrante desde el cuerpo y que las obras con más de tres autores aparezcan con los tres primeros seguidos de «et al.»;
- compilar con `latexmk` y resolver citas indefinidas y cajas `Overfull`;
- verificar que la clase use `a4paper` y que el PDF resultante mida 210 × 297 mm;
- leer el PDF a tamaño normal y en escala de grises;
- inspeccionar rutas, cruces, cajas, etiquetas, contraste y recortes de todos los gráficos;
- verificar que el Beamer resume exactamente esta version y no un MVP anterior, y revisar visualmente cada diapositiva en modo presentación.

### Registro de auditoría de citas y estilo

No cierre los pases solo con una lectura informal. Conserve una tabla de hallazgos con ubicación, tipo, severidad, evidencia, corrección y estado, y compruebe:

- cobertura: toda idea externa, definición, afirmación teórica, algoritmo, arquitectura, dataset, checkpoint, estándar y política tiene la fuente pertinente;
- correspondencia: la referencia sostiene la proposición exacta y su alcance no se exagera;
- metadatos: autores, título, año, sede, DOI/URL y versión coinciden con la fuente primaria;
- integridad: no hay claves citadas sin BibTeX, entradas huérfanas, duplicados ni paráfrasis demasiado cercanas;
- evidencia interna: cada cifra conserva artefacto, campo, split, versión y criterio;
- precisión editorial: métricas y magnitudes estimadas usan dos cifras significativas, mientras conteos, tamaño muestral, años, versiones, identificadores, hashes y parámetros fijados por protocolo permanecen exactos;
- decisiones locales: umbrales, fusiones, reglas y heurísticas se identifican como propias;
- estilo: lenguaje directo, una idea principal por oración, términos consistentes, siglas definidas, tiempos verbales correctos, ortografía revisada y ausencia de tono promocional;
- cierre visual: captions autocontenidos, referencias cruzadas correctas, texto legible y líneas que no atraviesan cajas o rótulos.

Clasifique como crítica toda fuente inventada, afirmación central sin respaldo o resultado sin origen; como alta, una fuente real que no sostiene la afirmación; como media, un metadato o alcance ambiguo; y como baja, una inconsistencia formal. El informe debe dar conteos y excepciones, no limitarse a afirmar que las citas y el estilo fueron revisados.
