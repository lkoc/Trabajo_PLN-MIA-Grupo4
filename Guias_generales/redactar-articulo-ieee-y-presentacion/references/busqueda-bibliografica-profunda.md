# Búsqueda bibliográfica profunda y recursiva

## Propósito

Construir cadenas de búsqueda —search strings— reproducibles a partir de unas pocas palabras, ampliarlas mediante vocabulario encontrado en fuentes pertinentes y alinearlas con cada componente del artículo. El objetivo no es obtener muchas referencias, sino recuperar evidencia suficiente, relevante y trazable para las afirmaciones del manuscrito.

Esta guía sirve para antecedentes de un artículo y para revisiones estructuradas. Una revisión sistemática exige además un protocolo propio, criterios de selección y el estándar de reporte que corresponda.

## 1. Traducir el artículo a preguntas de búsqueda

Extraer conceptos desde:

- problemática y situación deseada;
- problema real, causa subyacente y brecha tecnológica;
- pregunta general y preguntas específicas;
- objetivo general y objetivos específicos;
- artefacto, intervención o fenómeno;
- metodología y alternativas;
- datos, población, territorio, idioma o contexto;
- variables, categorías, métricas y resultados;
- ontología, términos canónicos y relaciones;
- ética, regulación, riesgos y limitaciones;
- palabras clave propuestas para el título y el resumen.

Crear una matriz:

| Componente del artículo | Pregunta bibliográfica | Conceptos | Tipo de fuente buscada | Sección destino |
|---|---|---|---|---|
| Problemática | ¿Qué evidencia demuestra la brecha? | fenómeno, actor, contexto | estudios empíricos, informes primarios | Introducción |
| Fundamentos | ¿Cómo se define cada término? | concepto, sinónimo, término superior | fuentes fundacionales, revisiones | Bases teóricas |
| Artefacto | ¿Qué soluciones comparables existen? | sistema, intervención, arquitectura | papers de método, benchmarks | Antecedentes |
| Metodología | ¿Qué diseño permite responder la pregunta? | método, protocolo, métrica | guías metodológicas, papers primarios | Método |
| Resultados | ¿Con qué trabajos es válido comparar? | tarea, datos, métrica, condiciones | estudios comparables | Discusión |
| Limitaciones | ¿Qué sesgos o riesgos se conocen? | amenaza, subgrupo, validez | auditorías, estudios críticos | Limitaciones |

No intentar resolver todas las preguntas con una única cadena gigantesca. Crear búsquedas por función y después integrar la evidencia.

## 2. Formar bloques conceptuales

Usar OR dentro de cada concepto y AND entre conceptos:

    (sinónimo_A1 OR sinónimo_A2 OR "frase A3")
    AND
    (sinónimo_B1 OR acrónimo_B2 OR término_B3*)
    AND
    (contexto_C1 OR contexto_C2)

Bloques frecuentes:

| Bloque | Contenido |
|---|---|
| P: problema/fenómeno | nombres, variantes, causas y manifestaciones |
| A: artefacto/intervención | sistema, herramienta, programa o enfoque |
| M: método/tecnología | algoritmo, diseño, arquitectura o procedimiento |
| C: contexto/población | país, idioma, sector, población o fuente de datos |
| O: resultado | métrica, efecto o capacidad buscada |
| X: exclusión | sentidos homónimos claramente ajenos |

No todos los bloques deben aparecer siempre. Un bloque de resultado o contexto demasiado restrictivo puede eliminar estudios relevantes. Empezar con dos o tres conceptos centrales y agregar límites después de inspeccionar los resultados.

## 3. Operadores y agrupación

### OR

Amplía un concepto con sinónimos, acrónimos, variantes ortográficas, términos históricos y traducciones:

    ("término principal" OR sinónimo OR acronym OR variant*)

### AND

Exige la intersección entre conceptos distintos:

    (problema OR sinónimo) AND (artefacto OR enfoque)

### NOT o AND NOT

Excluye un sentido ajeno. Usarlo con cautela y solo después de comprobar que no elimina artículos válidos:

    (concepto principal) AND NOT (homónimo ajeno)

Antes de conservar una exclusión, probarla sobre artículos centinela conocidos y revisar una muestra de los registros eliminados.

### Paréntesis

Agrupar siempre los sinónimos y hacer explícita la lógica. La precedencia varía entre plataformas:

    (A1 OR A2) AND (B1 OR B2) AND NOT (X1 OR X2)

No confiar en que el motor evaluará de izquierda a derecha ni en que todas las bases usan la misma prioridad.

### Frases, truncamiento y proximidad

- Usar comillas para frases exactas cuando la base lo permita.
- Usar truncamiento solo después de comprobar qué variantes recupera.
- Evitar raíces tan cortas que introduzcan ruido.
- Usar proximidad cuando dos términos deben aparecer cerca, no necesariamente como frase fija.
- Verificar símbolos, comodines, campos y límites en la ayuda oficial de cada base.

## 4. Crear la versión cero

Partir de:

1. tres a cinco términos del problema;
2. tres a cinco términos del artefacto o método;
3. nombres alternativos en inglés y en el idioma del contexto;
4. uno o dos artículos centinela ya conocidos, si existen;
5. descriptores de tesauros, ontologías o vocabularios controlados.

Ejemplo abstracto:

    ("problema principal" OR "nombre alternativo" OR problem*)
    AND
    ("artefacto principal" OR system* OR framework*)
    AND
    (evaluation OR validation OR benchmark*)

La cadena es una hipótesis de recuperación. No debe congelarse hasta probar qué incluye y qué omite.

## 5. Ampliación recursiva

Ejecutar ciclos numerados.

### Ciclo 0: semillas

- formular las primeras cadenas;
- elegir dos o más bases complementarias;
- registrar fecha, base, campos y número de resultados;
- guardar artículos centinela.

### Ciclo 1: cosecha de vocabulario

Revisar títulos, resúmenes, palabras clave e índices de una muestra relevante. Extraer:

- sinónimos y antónimos útiles;
- términos superiores e inferiores;
- acrónimos y nombres anteriores;
- variantes regionales, lingüísticas y ortográficas;
- términos de métodos, datasets, métricas y tareas;
- descriptores de vocabularios controlados;
- autores, grupos, venues y proyectos recurrentes.

No aceptar automáticamente términos sugeridos por un LLM. Verificarlos en fuentes reales, tesauros o resultados de la base.

### Ciclo 2: refactorización

- añadir términos que recuperan evidencia nueva;
- retirar términos que solo producen ruido;
- dividir cadenas por subpregunta;
- usar campos de título/resumen/palabras clave;
- probar proximidad o frases;
- comparar resultados con y sin cada bloque.

Conservar versiones. No reemplazar la cadena anterior sin registrar el cambio y su efecto.

### Ciclo 3: expansión por citas

Para cada estudio central:

- revisar referencias hacia atrás;
- buscar trabajos que lo citan;
- localizar versiones, datasets, protocolos y erratas;
- revisar artículos relacionados y trabajos del mismo grupo;
- buscar el aporte fundacional del método citado;
- comprobar si existe una revisión o réplica posterior.

La expansión por citas complementa la búsqueda booleana; no la sustituye.

### Ciclo 4: búsqueda por huecos

Cruzar las fuentes reunidas con la matriz del artículo. Formular consultas específicas para celdas vacías:

- definición sin fuente;
- afirmación contextual sin evidencia;
- algoritmo sin paper fundacional;
- categoría sin antecedente;
- métrica sin definición;
- comparación sin protocolo equivalente;
- riesgo o limitación sin literatura;
- afirmación actual que requiere una fuente vigente.

### Ciclo 5: actualización

Repetir la búsqueda al cerrar el manuscrito. Las bases incorporan registros y los campos pueden cambiar. Registrar una nueva fecha de corte y revisar si la evidencia nueva modifica el argumento.

## 6. Ajustar recall y precisión

Si hay pocos resultados pertinentes:

- agregar sinónimos y traducciones;
- quitar un bloque restrictivo;
- buscar en título/resumen antes que solo en título;
- reducir frases exactas;
- usar términos superiores del vocabulario;
- explorar citas de artículos centinela;
- añadir otra base disciplinaria.

Si hay demasiado ruido:

- añadir el contexto o artefacto;
- exigir campos de título, resumen o palabras clave;
- usar proximidad;
- reemplazar raíces amplias;
- añadir un tercer bloque;
- aplicar filtros justificados de fecha o tipo documental;
- usar NOT solo después de una prueba de pérdida.

Medir al menos:

- si recupera los artículos centinela;
- proporción de pertinentes en una muestra;
- conceptos nuevos por ciclo;
- duplicados entre bases;
- huecos de evidencia que permanecen.

## 7. Traducir la cadena a cada base

No copiar la misma sintaxis literalmente.

| Base | Adaptación típica |
|---|---|
| IEEE Xplore | AND, OR y NOT en mayúsculas; paréntesis; frases; NEAR/ONEAR en interfaces compatibles; comprobar campos y límites actuales |
| Scopus | campos como TITLE-ABS-KEY; OR, AND y AND NOT; proximidad W/n o PRE/n; usar paréntesis explícitos |
| Web of Science | campos como TS; AND, OR, NOT, NEAR y SAME; comprobar reglas de la colección elegida |
| PubMed | términos libres y MeSH; etiquetas de campo; AND, OR y NOT; paréntesis porque procesa operadores según sus reglas propias |
| Google Scholar | frases y exclusiones simples; usarlo como complemento para rastreo, no como sustituto de una cadena reproducible en índices estructurados |

Guardar la traducción exacta que se ejecutó, no solo una cadena «maestra».

### Fuentes oficiales de sintaxis

- [IEEE Xplore Search Tips](https://ieeexplore.ieee.org/Xplorehelp/searching-ieee-xplore/search-tips).
- [IEEE Xplore: Searching and Saving Searches](https://ieeexplore.ieee.org/Xplorehelp/downloads/user-guides/IEEE_Xplore_Searching_and_Saving_Searches.pdf).
- [Scopus Advanced Search](https://service.elsevier.com/app/answers/detail/a_id/11365/supporthub/scopus/).
- [Web of Science Search Operators](https://webofscience.help.clarivate.com/Content/search-operators.html).
- [PubMed Help](https://pubmed.ncbi.nlm.nih.gov/help/).

Consultar estas páginas al ejecutar la búsqueda: sintaxis, interfaces y límites pueden cambiar.

## 8. Registro reproducible

Guardar una fila por ejecución:

| ID | Fecha/hora/zona | Base | Cobertura | Cadena exacta | Filtros | Resultados | Exportación | Cambio y motivo |
|---|---|---|---|---|---|---:|---|---|

Mantener:

- protocolo y preguntas;
- diccionario de conceptos;
- cadenas maestras y traducciones;
- historial de versiones;
- exportaciones originales;
- regla y herramienta de deduplicación;
- criterios de inclusión/exclusión;
- decisiones de cribado y motivo;
- matriz fuente → afirmación/sección;
- fecha de actualización;
- limitaciones de acceso.

No deduplicar solo por título. Combinar DOI, identificadores, título normalizado, autores y año, conservando el registro más completo y la procedencia de todas las bases.

## 9. Selección y evaluación

Definir antes del cribado:

- periodo y justificación;
- idiomas;
- tipos documentales;
- población o contexto;
- diseños elegibles;
- criterios de calidad;
- tratamiento de preprints, literatura gris y duplicados;
- procedimiento para desacuerdos;
- datos que se extraerán.

Separar relevancia temática de calidad metodológica. Una fuente puede definir un concepto sin ser evidencia empírica suficiente para una afirmación causal.

Para búsquedas sistemáticas, usar [PRISMA-S](https://pmc.ncbi.nlm.nih.gov/articles/PMC8270366/) para reportar fuentes, estrategias y actualizaciones, y considerar [PRESS 2015](https://pubmed.ncbi.nlm.nih.gov/27005575/) para revisar la traducción de la pregunta, operadores, encabezamientos, términos libres, sintaxis y filtros. PRISMA-S es una guía de reporte; no reemplaza el protocolo de conducta de la revisión.

## 10. Criterios de parada

Una búsqueda narrativa profunda puede cerrarse cuando:

- recupera los artículos centinela;
- cada afirmación sustantiva tiene fuente adecuada;
- la matriz del artículo no conserva huecos críticos;
- dos ciclos consecutivos no añaden conceptos o líneas de evidencia relevantes;
- se cubrieron índices y fuentes complementarias justificadas;
- se revisaron referencias hacia atrás y citas hacia adelante;
- quedó fijada una fecha de actualización.

Esto no demuestra exhaustividad. En una revisión sistemática, seguir el protocolo y documentar todas las fuentes y estrategias; no detenerse solo por saturación percibida.

## 11. Auditoría de la búsqueda

Comprobar:

- la pregunta se tradujo a conceptos correctos;
- OR une sinónimos y AND separa conceptos;
- los paréntesis expresan la intención;
- vocabulario controlado y texto libre se complementan;
- las frases y raíces no son demasiado restrictivas o amplias;
- NOT no elimina estudios válidos;
- cada base recibió una traducción sintáctica válida;
- artículos centinela aparecen;
- filtros están justificados;
- fecha, cadena, resultados y exportación quedaron registrados;
- las fuentes elegidas sostienen la afirmación exacta;
- la búsqueda se actualizó antes del cierre.

Si es una síntesis de evidencia de alto impacto, pedir revisión a una persona con experiencia en búsqueda bibliográfica o aplicar PRESS.

## Entregables mínimos

1. mapa de conceptos y sinónimos;
2. matriz componente del artículo → pregunta bibliográfica;
3. cadena maestra por subpregunta;
4. traducción por base;
5. historial de iteraciones;
6. exportaciones y deduplicación;
7. matriz de evidencia;
8. registro de cribado y exclusiones;
9. informe de actualización y límites.

## Prompt operativo para cualquier LLM

Proporcionar:

> Tema y brecha: [describir]. Preguntas y objetivos: [listar]. Artefacto o fenómeno: [describir]. Metodología prevista: [describir]. Contexto, población e idiomas: [delimitar]. Bases disponibles: [listar]. Artículos centinela: [identificar]. Criterios y fechas: [indicar].
>
> Construye la matriz artículo → pregunta bibliográfica → bloques conceptuales; propón cadenas maestras con AND, OR, paréntesis y NOT solo cuando una exclusión esté justificada y probada; tradúcelas a cada base; explica cómo ampliar vocabulario y citas en ciclos; define criterios de inclusión, registro, deduplicación y parada; y marca toda fuente o dato que requiera verificación.

Si el LLM no puede consultar una base, debe entregar una estrategia para ejecutar, no afirmar que la ejecutó ni inventar conteos, artículos o DOI. Después de cada ciclo real, devolverle la exportación o una muestra para que sugiera nuevos términos; verificar cada sugerencia antes de incorporarla.
