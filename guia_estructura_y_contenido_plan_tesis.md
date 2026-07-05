# Guía de estructura y contenido para elaborar un plan de tesis de especialización

Esta guía consolida la estructura institucional, la lógica de contenido y el flujo de trabajo aplicado en el proyecto `plan_tesis_cables_pinn.tex`. Su propósito es permitir que una persona o un LLM puedan reproducir el enfoque para elaborar otro plan de tesis de especialización a partir de un tema y una bibliografía organizada en Zotero.

Esta guía debe usarse junto con `Plan/guia_redaccion_plan_tesis.md`. La guía de redacción define el estilo, el tono, la longitud de oraciones, la relación entre párrafos, el uso de fórmulas, figuras y citas. Esta guía define la estructura, los componentes, la función lógica de cada parte y el proceso de construcción del plan.

## 1. Alcance de la guía

Esta guía sirve para:

1. Replicar la estructura de un plan de tesis de maestría de especialización o profesionalizante.
2. Cumplir los requerimientos formales de la `GUÍA N°01 Plan.pdf` de la UNI para planes de tesis de posgrado.
3. Convertir una bibliografía de Zotero en problema, objetivos, metodología, marco teórico y anexos trazables.
4. Guiar a un LLM mediante loops, roles de agente, entradas, salidas y criterios de aceptación.
5. Evitar que el plan sea solo una revisión bibliográfica sin propuesta, o solo una propuesta técnica sin fundamento.

No reemplaza el criterio del asesor ni las normas vigentes de la unidad de posgrado. Si existe conflicto entre esta guía y una indicación institucional posterior, debe prevalecer la indicación institucional.

## 2. Principio central

Un plan de tesis de especialización debe demostrar que existe un problema real o técnico, que el problema puede abordarse mediante una propuesta de solución, y que la propuesta puede evaluarse con evidencia suficiente.

En este proyecto, esa lógica se expresó así:

- objeto de estudio: sistema cable--instalación--entorno térmico;
- brecha: diferencia entre entorno térmico heterogéneo y representación homogénea;
- propuesta: artefacto PINN 2D verificable;
- metodología: ciencia del diseño, con diseño, demostración y evaluación;
- evidencia: casos analíticos, bibliográficos, computacionales y referencias FEM;
- trazabilidad: matrices, diseño factorial, revisión documental y repositorio.

Para otro tema, deben cambiarse el objeto de estudio, el dominio técnico, las fuentes y el artefacto, pero debe conservarse la misma lógica: problema, evidencia, propuesta, evaluación y trazabilidad.

## 3. Requerimientos de la Guía N.° 01 para tesis de especialización

La `GUÍA N°01 Plan.pdf` distingue los planes de investigación académica y los planes de especialización o profesionalizantes. Para una maestría de especialización, el plan debe orientarse a la solución de problemas reales mediante metodologías y técnicas aplicadas.

### 3.1 Partes obligatorias del documento

El plan debe contener tres bloques:

1. Cuerpo preliminar.
2. Contenidos.
3. Referencias y anexos.

El cuerpo preliminar incluye carátula y datos generales. Los contenidos se numeran en arábigos. Las páginas preliminares usan numeración romana.

### 3.2 Datos generales

Los datos generales deben incluir:

- título del plan de tesis;
- nombre del autor o autores;
- nombre del asesor o asesores;
- área involucrada o unidad de posgrado;
- lugar o institución donde se desarrolla el proyecto;
- duración estimada del proyecto en meses;
- fecha de inicio y fecha de culminación, cuando corresponda.

En el proyecto actual se agregó también una descripción breve del artefacto de investigación. Ese agregado es útil cuando el plan es tecnológico, porque aclara qué se diseña o evalúa.

### 3.3 Estructura institucional del plan de tesis de especialización

La estructura base indicada por la Guía N.° 01 es:

```text
CARÁTULA
DATOS GENERALES

CAPÍTULO I. PLANTEAMIENTO DEL PROBLEMA
1.1 Diagnóstico
1.2 Identificación y descripción del problema de estudio
1.2.1 Antecedentes bibliográficos
1.2.2 Formulación del problema
1.2.2.1 Formulación del problema general
1.2.2.2 Formulación de los problemas específicos
1.2.3 Justificación y alcances
1.2.3.1 Justificación
1.2.3.2 Alcances

CAPÍTULO II. OBJETIVOS
2.1 Objetivo general
2.2 Objetivos específicos

CAPÍTULO III. MARCO TEÓRICO
3.1 Bases teóricas
3.2 Definición de términos

CAPÍTULO IV. METODOLOGÍA
4.1 Tipo de investigación
4.2 Nivel de investigación
4.3 Métodos de trabajo
4.4 Población y muestra, cuando corresponda
4.5 Tipo de diseño
4.6 Técnicas e instrumentos de recolección de datos
4.7 Técnicas e instrumentos de análisis y procesamiento de datos
4.8 Etapas de intervención del estudio

CAPÍTULO V. ADMINISTRACIÓN DEL PLAN DE TESIS
5.1 Cronograma
5.2 Presupuesto
5.3 Financiamiento

REFERENCIAS
ANEXOS
```

El proyecto actual reorganizó algunos contenidos para reforzar trazabilidad, matrices y anexos. Sin embargo, si se replica el enfoque en un nuevo plan, debe verificarse que la estructura institucional anterior esté cubierta o que cualquier adaptación esté aprobada por la unidad de posgrado.

### 3.4 Naturaleza profesionalizante

La Guía N.° 01 indica que la tesis de especialización se orienta a resolver problemas reales y actuales. Por tanto:

- el diagnóstico debe identificar un problema práctico o técnico;
- la propuesta debe ofrecer una solución determinada;
- la metodología debe demostrar viabilidad, sostenibilidad o utilidad;
- el análisis puede apoyarse en técnicas cuantitativas y estadística descriptiva;
- no debe presentarse como una investigación básica si el grado es profesionalizante.

En este proyecto, la solución se formuló como un artefacto computacional. Para otro tema, el artefacto puede ser un modelo, procedimiento, prototipo, metodología, arquitectura, sistema, protocolo, herramienta o plan de mejora.

### 3.5 Formato formal

La Guía N.° 01 establece los siguientes criterios formales:

- papel A4;
- margen izquierdo y superior de 3,0 cm;
- margen derecho e inferior de 2,5 cm;
- interlineado 1,5;
- fuente Arial 11 para el cuerpo;
- uso de cursiva solo para palabras extranjeras o ajenas al español;
- numeración romana minúscula para preliminares;
- numeración arábiga para el texto;
- numeración en la esquina superior derecha;
- no usar la palabra `página` antes del número;
- las páginas con figuras y tablas también se numeran;
- estilo APA vigente para citas y referencias;
- máximo institucional de 130 páginas para tesis de maestría, incluyendo gráficos, tablas, referencias y anexos.

En este proyecto se implementó además:

- títulos en mayúscula;
- tres niveles numéricos y cuarto nivel con letras;
- `ANEXO` en mayúscula;
- anexos grandes en hoja apaisada cuando mejora la legibilidad;
- fórmulas con número matemático a la derecha y caption inferior.

Los detalles de redacción y formato fino se controlan con `Plan/guia_redaccion_plan_tesis.md`.

## 4. Adaptación aplicada en este proyecto

El proyecto actual usó la estructura institucional, pero la ajustó a una tesis tecnológica basada en ciencia del diseño. La adaptación conserva la lógica de especialización porque construye y evalúa un artefacto aplicado.

### 4.1 Capítulos usados

El documento `plan_tesis_cables_pinn.tex` se organizó en:

1. Datos generales.
2. Planteamiento del problema.
3. Objetivos de la investigación.
4. Metodología de la investigación.
5. Marco teórico.
6. Referencias.
7. Anexos.

Para replicar el enfoque en otro plan, se recomienda usar la estructura institucional completa y ubicar los contenidos aplicados de este proyecto dentro de ella. Si la unidad de posgrado exige el Capítulo V de administración, se debe incluir cronograma, presupuesto y financiamiento antes de referencias y anexos.

### 4.2 Componentes añadidos por trazabilidad

El proyecto incorporó elementos no siempre explícitos en la estructura mínima, pero útiles para planes técnicos:

- matriz de consistencia;
- matriz de operacionalización de variables;
- diseño factorial o diseño de escenarios;
- estrategia de revisión documental;
- repositorio digital del informe;
- listado de fórmulas;
- figuras explicativas del hilo conductor;
- referencias cruzadas internas entre capítulos y anexos.

Estos elementos no deben agregarse como adornos. Deben resolver una función: hacer visible la relación entre problema, objetivos, metodología, evidencia y límites.

## 5. Función de cada componente

### 5.1 Carátula

La carátula identifica la institución, la unidad de posgrado, el título, el propósito de graduación, el autor o autores, el asesor, el lugar y el año.

El título debe:

- ser claro, conciso y específico;
- reflejar problema, objeto de estudio, variable o método principal;
- indicar alcance cuando sea necesario;
- escribirse en mayúsculas;
- evitar exceso de palabras;
- usar cursiva solo para términos extranjeros.

### 5.2 Datos generales

Los datos generales ubican el proyecto en su contexto académico y operativo.

Debe evitarse que esta sección sea solo administrativa. En proyectos aplicados, conviene incluir una línea breve sobre el artefacto, producto o solución, porque ayuda a orientar la lectura desde el inicio.

### 5.3 Capítulo I: Planteamiento del problema

Este capítulo debe construir el problema. Su función no es presentar todo el tema, sino demostrar que existe una dificultad que justifica una propuesta aplicada.

Debe incluir:

- contexto general;
- diagnóstico;
- identificación del problema;
- antecedentes bibliográficos;
- formulación del problema general;
- problemas específicos;
- justificación;
- alcances y limitaciones.

La secuencia recomendada es:

1. Presentar el objeto de estudio.
2. Mostrar la situación problemática.
3. Sustentar la dificultad con evidencia bibliográfica o técnica.
4. Formular la brecha.
5. Presentar la propuesta como respuesta posible.
6. Delimitar qué se hará y qué no se hará.

En un plan profesionalizante, el problema debe tener conexión con necesidades reales. En un plan computacional o documental, debe evitarse afirmar que se harán mediciones directas si solo se usarán datos bibliográficos, casos analíticos o simulaciones.

### 5.4 Capítulo II: Objetivos

Este capítulo transforma el problema en dirección de trabajo.

El objetivo general debe contener:

- acción principal;
- objeto de estudio;
- artefacto o propuesta;
- resultado esperado;
- alcance de evaluación.

Los objetivos específicos deben funcionar como pasos verificables. Cada objetivo debe vincularse con:

- una actividad metodológica;
- un producto;
- una evidencia;
- una sección o anexo.

En este proyecto se añadieron hipótesis y dimensiones operativas. Para otros planes, pueden usarse hipótesis, proposiciones de diseño o supuestos verificables si ayudan a organizar la evaluación. En una tesis de especialización no deben presentarse como contrastación estadística obligatoria, salvo que el diseño lo exija.

### 5.5 Hipótesis, proposiciones o criterios de diseño

Cuando se incluyan hipótesis o proposiciones, deben:

- derivarse del problema y de los objetivos;
- ser evaluables con la metodología;
- relacionarse con variables, factores o resultados;
- indicar condiciones de control;
- evitar prometer efectos no sustentados.

En un estudio DSR, las hipótesis pueden reformularse como proposiciones de diseño o criterios de utilidad. Por ejemplo: si el artefacto representa una variable crítica, entonces debería mejorar la trazabilidad o la exactitud frente a una referencia.

### 5.6 Dimensiones operativas

Esta sección convierte conceptos en variables, factores, resultados y controles.

Debe indicar:

- variables independientes o factores;
- variables dependientes o resultados;
- variables controladas;
- variables no controladas que se registrarán;
- unidades o indicadores;
- relación con la matriz de operacionalización.

La matriz de consistencia organiza ideas. La matriz de operacionalización vuelve esas ideas observables.

### 5.7 Capítulo III: Marco teórico

El marco teórico explica la teoría fundamental del problema abstracto. No debe ser una lista de autores.

Debe incluir:

- antecedentes relevantes;
- bases teóricas;
- conceptos clave;
- modelos, ecuaciones o relaciones necesarias;
- técnica o método propuesto;
- definición de términos.

Debe responder:

- ¿Qué teoría explica el problema?
- ¿Qué conceptos se usan después en la metodología?
- ¿Qué métodos existen y por qué el plan elige uno?
- ¿Qué límites reporta la literatura?
- ¿Qué términos deben definirse para evitar ambigüedad?

En este proyecto, el marco teórico explicó conducción de calor, PDE, régimen transitorio, régimen estacionario, métodos analíticos, FEM, FDM, FVM, PINN, función de pérdida, DSR y heterogeneidad térmica.

Para otro tema, el marco debe explicar el equivalente teórico del problema: no solo el contexto, sino el mecanismo o modelo que lo sostiene.

### 5.8 Capítulo IV: Metodología

La metodología explica cómo se abordará el problema y cómo se evaluará la solución.

Debe incluir:

- tipo de investigación;
- nivel de investigación;
- métodos de trabajo;
- población y muestra, si corresponde;
- unidad de análisis, si no hay población estadística;
- tipo de diseño;
- técnicas e instrumentos de recolección o registro de datos;
- técnicas de análisis y procesamiento;
- etapas de intervención;
- criterios de verificación o evaluación.

En proyectos de diseño, la metodología debe separar:

- objeto de estudio;
- artefacto o propuesta;
- entrada del artefacto;
- salida del artefacto;
- evidencia de evaluación;
- criterios de aceptación;
- límites de validez.

En este proyecto se usó DSR. Para otro tema, puede usarse DSR cuando el trabajo construye y evalúa un artefacto. Si el trabajo no construye un artefacto, debe elegirse una metodología más pertinente.

### 5.9 Artefacto, propuesta o solución

La propuesta debe describirse antes de la implementación detallada.

Debe indicar:

- qué se diseña o construye;
- qué problema atiende;
- qué entradas requiere;
- qué salidas entrega;
- qué componentes tiene;
- cómo se verifica;
- qué límites tiene.

La propuesta no debe confundirse con el objeto de estudio. En este proyecto, el objeto de estudio fue el sistema físico, y el artefacto fue la PINN usada para estudiarlo.

### 5.10 Capítulo V: Administración del plan

Cuando se siga estrictamente la Guía N.° 01, se debe incluir:

- cronograma;
- presupuesto;
- financiamiento.

El cronograma debe mostrar actividades y tiempos. El presupuesto debe mostrar recursos necesarios. El financiamiento debe indicar si el proyecto será autofinanciado, financiado por fondos concursables o financiado por otra fuente.

La administración también puede apoyarse en dos anexos de gestión: ciclo del proyecto y línea de tiempo. El ciclo del proyecto muestra la secuencia desde tema, plan de tesis, tesis y post tesis; la línea de tiempo muestra hitos, revisiones y cierres dentro del plazo disponible.

Si el formato del asesor o la unidad no exige este capítulo dentro del cuerpo principal, debe quedar cubierto en anexos o documentación complementaria.

### 5.11 Referencias

Las referencias deben seguir APA vigente y corresponder a fuentes realmente consultadas.

Para un flujo basado en Zotero:

- usar la biblioteca local como fuente primaria;
- verificar DOI, ISBN o URL cuando corresponda;
- no inventar metadatos;
- no citar fuentes de estilo de escritura si no forman parte de la tesis;
- mantener `.bib` y Zotero coherentes;
- usar páginas, figuras o tablas cuando la afirmación sea específica.

### 5.12 Anexos

Los anexos deben sostener trazabilidad. No deben ser depósitos de información suelta.

En este enfoque se recomiendan:

- ANEXO 1: matriz de consistencia;
- ANEXO 2: matriz de operacionalización de variables;
- ANEXO 3: diseño factorial o diseño de escenarios;
- ANEXO 4: estrategia de revisión documental;
- ANEXO 5: repositorio digital, cronograma, presupuesto, financiamiento, ciclo del proyecto, línea de tiempo o protocolo de reproducción, según necesidad.

Si las matrices son extensas, deben ponerse en orientación apaisada para legibilidad. La palabra `ANEXO` debe mantenerse en mayúscula.

## 6. Criterios de contenido aplicados en el proyecto

### 6.1 Hilo conductor

El plan debe poder leerse como una cadena:

```text
Problema real -> brecha -> propuesta -> metodología -> evidencia -> evaluación -> contribución
```

En este proyecto, la cadena fue:

```text
heterogeneidad térmica -> campo de temperatura -> temperatura máxima -> ampacidad -> artefacto PINN -> verificación DSR
```

Para otro tema, se debe definir una cadena equivalente y repetirla con moderación en capítulos, figuras, tablas y anexos.

### 6.2 Referencias cruzadas internas

Las referencias cruzadas internas deben usarse cuando conectan ideas. No deben usarse por decoración.

Conviene referenciar:

- la matriz de consistencia cuando se formula el problema;
- la matriz de operacionalización cuando se explican variables;
- el diseño de escenarios cuando se describen factores;
- la metodología cuando se explica el marco teórico;
- fórmulas, figuras y tablas cuando ayudan a interpretar el argumento.

### 6.3 Figuras, tablas y fórmulas

Toda figura, tabla o fórmula debe:

- estar anunciada antes de aparecer;
- tener caption;
- tener fuente o indicar elaboración propia;
- ser citada en el texto;
- ser interpretada después o en el párrafo cercano;
- aportar claridad o trazabilidad.

Las reglas detalladas están en la guía de redacción. Esta guía solo exige que cada objeto visual o matemático tenga función estructural.

### 6.4 Lenguaje sobre datos y mediciones

Si el estudio no levanta datos de campo, no debe decir que medirá directamente.

Usar:

- `calcular`;
- `estimar`;
- `registrar`;
- `comparar`;
- `usar valores bibliográficos`;
- `usar casos analíticos`;
- `usar referencias computacionales`.

Reservar `medición` para ensayos, instrumentación, sensores, campo o laboratorio.

### 6.5 Evidencia y citas

Cada afirmación técnica relevante debe tener apoyo. La evidencia puede provenir de:

- artículos;
- libros;
- normas;
- reportes técnicos;
- casos publicados;
- resultados computacionales;
- matrices o anexos del propio plan.

Las afirmaciones de organización interna, apertura o cierre pueden no tener cita si solo explican la estructura del documento.

## 7. Uso de Zotero y bibliografía local

La bibliografía local permite construir el plan sin inventar fuentes.

### 7.1 Clasificación mínima de fuentes

Antes de redactar, clasificar las fuentes en:

- estado del arte;
- aporte metodológico o teórico;
- aplicación o caso;
- norma o estándar;
- libro de fundamento;
- fuente institucional o contextual;
- fuente de verificación o benchmark.

### 7.2 Extracción de evidencia

Para cada fuente relevante, registrar:

- tema;
- aporte;
- método;
- variable o concepto usado;
- dato cuantitativo, si existe;
- página, tabla o figura;
- DOI, ISBN o URL;
- uso dentro del plan.

### 7.3 Regla contra citas fantasma

No se deben inventar autores, DOI, ISBN, páginas, figuras ni resultados.

Si una fuente no está en Zotero:

1. buscarla en la biblioteca local;
2. revisar PDFs existentes;
3. consultar bases o sitios oficiales si es necesario;
4. agregarla a Zotero y al `.bib` solo después de verificar su existencia.

## 8. Matrices y organizadores

### 8.1 Matriz de consistencia

La matriz de consistencia debe alinear:

- problema general;
- problemas específicos;
- objetivo general;
- objetivos específicos;
- hipótesis o proposiciones;
- variables o factores;
- evidencia;
- productos verificables;
- controles.

Debe leerse como el esqueleto lógico del plan. Si una fila no puede conectarse con la metodología, debe corregirse el problema, el objetivo o la evidencia.

### 8.2 Matriz de operacionalización

La matriz de operacionalización debe transformar conceptos en elementos observables.

Debe contener:

- categoría;
- variable o factor;
- definición conceptual;
- definición operacional;
- indicador;
- fuente o instrumento;
- unidad o criterio;
- control o límite.

En proyectos computacionales, `instrumento` puede significar protocolo, script, artefacto, repositorio, matriz de resultados o benchmark. No debe llamarse instrumento de medición si no hay medición directa.

### 8.3 Diseño factorial o diseño de escenarios

El diseño factorial debe indicar:

- factores;
- niveles;
- razón de cada factor;
- combinaciones posibles;
- restricciones;
- controles;
- salida esperada.

Debe aclararse si se trata de diseño factorial de escenarios y no de análisis factorial estadístico.

## 9. Proceso reproducible para crear un plan nuevo

### Paso 1. Preparar insumos

Entradas mínimas:

- tema inicial;
- programa y modalidad de tesis;
- Guía N.° 01 o norma institucional vigente;
- bibliografía en Zotero;
- archivo `.bib`;
- guía de redacción;
- restricciones de alcance y tiempo.

Salida:

- ficha inicial del proyecto.

### Paso 2. Definir objeto de estudio y problema

Identificar:

- sistema o fenómeno;
- actores o contexto;
- brecha;
- consecuencia del problema;
- evidencia inicial.

Salida:

- enunciado preliminar del problema.

### Paso 3. Construir mapa bibliográfico

Clasificar fuentes y extraer evidencia.

Salida:

- tabla de fuentes por función;
- lista de conceptos clave;
- fuentes base para marco teórico;
- fuentes base para metodología.

### Paso 4. Diseñar la matriz de consistencia

Antes de redactar capítulos completos, construir la matriz.

Salida:

- problema general;
- problemas específicos;
- objetivos;
- hipótesis o proposiciones;
- variables o factores;
- evidencia;
- controles.

### Paso 5. Diseñar metodología

Definir:

- enfoque;
- tipo y nivel de investigación;
- método de trabajo;
- unidad de análisis;
- artefacto o propuesta;
- datos o casos;
- métricas;
- criterios de aceptación;
- etapas.

Salida:

- capítulo metodológico esquematizado.

### Paso 6. Redactar capítulos

Orden recomendado:

1. Planteamiento del problema.
2. Objetivos e hipótesis/proposiciones.
3. Metodología.
4. Marco teórico.
5. Administración del plan.
6. Anexos.
7. Resumen o síntesis final, si se requiere.

El marco teórico puede redactarse después de la metodología porque debe explicar los conceptos que realmente se usan.

### Paso 7. Integrar anexos

Agregar matrices, diseño de escenarios, estrategia documental y repositorio.

Salida:

- anexos citados al menos una vez en el texto;
- anexos con función clara;
- tablas legibles.

### Paso 8. Verificar consistencia

Aplicar la lista de control:

- cada problema tiene objetivo asociado;
- cada objetivo tiene actividad y producto;
- cada hipótesis o proposición tiene evidencia;
- cada variable tiene indicador;
- cada figura, tabla y fórmula está citada;
- cada fuente existe;
- cada anexo se cita en el texto;
- el documento respeta la Guía N.° 01;
- el estilo respeta la guía de redacción.

## 10. Enfoque operacional por loops para LLMs

El uso por LLM debe organizarse en ciclos. Cada loop tiene entradas, acciones, salidas y criterios de aceptación.

### Loop 1. Ingesta institucional

Entrada:

- Guía N.° 01;
- indicaciones del asesor;
- plantilla o plan previo.

Acciones:

- extraer estructura obligatoria;
- identificar formato formal;
- separar requisitos institucionales de criterios propios del proyecto.

Salida:

- checklist institucional.

Criterio de aceptación:

- ninguna parte obligatoria queda sin ubicación.

### Loop 2. Ingesta bibliográfica

Entrada:

- Zotero;
- `.bib`;
- PDFs locales;
- reportes bibliográficos.

Acciones:

- clasificar fuentes;
- extraer evidencia;
- detectar vacíos;
- verificar DOI, ISBN o URL.

Salida:

- mapa de fuentes por función.

Criterio de aceptación:

- cada afirmación técnica principal tiene al menos una fuente real.

### Loop 3. Modelado del problema

Entrada:

- mapa de fuentes;
- objeto de estudio;
- contexto aplicado.

Acciones:

- formular brecha;
- separar problema general y específicos;
- identificar consecuencias;
- evitar afirmaciones genéricas.

Salida:

- planteamiento del problema y matriz inicial.

Criterio de aceptación:

- el problema no es solo un tema; expresa una dificultad evaluable.

### Loop 4. Diseño metodológico

Entrada:

- problema;
- objetivos;
- tipo de propuesta;
- disponibilidad de datos.

Acciones:

- elegir metodología;
- definir unidad de análisis;
- definir artefacto o solución;
- definir métricas y criterios;
- identificar límites.

Salida:

- metodología estructurada y verificable.

Criterio de aceptación:

- cada objetivo específico tiene método, producto y evidencia.

### Loop 5. Marco teórico funcional

Entrada:

- metodología;
- variables;
- fuentes base.

Acciones:

- explicar teoría necesaria;
- definir términos;
- comparar métodos disponibles;
- conectar conceptos con el problema.

Salida:

- marco teórico breve, suficiente y conectado.

Criterio de aceptación:

- el marco explica lo que luego se usa; no contiene teoría decorativa.

### Loop 6. Redacción y estilo

Entrada:

- capítulos esquematizados;
- guía de redacción.

Acciones:

- redactar en tercera persona;
- usar conectores;
- mantener párrafos proporcionados;
- citar fuentes;
- citar figuras, tablas, fórmulas y anexos.

Salida:

- borrador discursivo.

Criterio de aceptación:

- el texto puede leerse como argumento continuo.

### Loop 7. Auditoría de trazabilidad

Entrada:

- borrador;
- matrices;
- Zotero;
- `.bib`.

Acciones:

- revisar citas;
- revisar referencias cruzadas;
- revisar origen de fórmulas;
- revisar fuente de figuras;
- revisar anexos citados.

Salida:

- borrador corregido.

Criterio de aceptación:

- no hay citas fantasma, anexos sin uso, figuras sin fuente ni fórmulas sin interpretación.

### Loop 8. Cierre formal

Entrada:

- documento completo;
- checklist institucional;
- guía de redacción.

Acciones:

- verificar títulos, numeración y TOC;
- verificar formato;
- compilar PDF;
- revisar advertencias;
- revisar legibilidad de anexos.

Salida:

- plan listo para revisión.

Criterio de aceptación:

- compila sin referencias indefinidas y cumple la estructura requerida.

## 11. Roles de agente recomendados

Un LLM puede aplicar esta guía como un sistema de agentes internos. No es necesario ejecutar agentes separados; basta con aplicar estos roles de manera secuencial.

### 11.1 Agente institucional

Verifica cumplimiento de Guía N.° 01, carátula, datos generales, capítulos, referencias, anexos, formato y paginación.

### 11.2 Agente bibliográfico

Lee Zotero, clasifica fuentes, verifica metadatos y asigna fuentes a problema, marco, metodología y anexos.

### 11.3 Agente de problema

Transforma el tema en problema, diferencia contexto de brecha, formula problema general y específicos.

### 11.4 Agente metodológico

Define enfoque, tipo, nivel, método, unidad de análisis, diseño, instrumentos, métricas y etapas.

### 11.5 Agente de artefacto o solución

Define la propuesta aplicada, sus entradas, salidas, componentes, criterios de utilidad y límites.

### 11.6 Agente de marco teórico

Selecciona teoría necesaria, compara métodos, define términos y evita teoría irrelevante.

### 11.7 Agente de matrices

Construye y audita matriz de consistencia, operacionalización y diseño de escenarios.

### 11.8 Agente de trazabilidad

Verifica que cada afirmación importante tenga fuente, supuesto, fórmula, figura, tabla, anexo o decisión metodológica que la sostenga.

### 11.9 Agente de redacción

Aplica `guia_redaccion_plan_tesis.md`, sin cambiar el contenido técnico ni inventar fuentes.

### 11.10 Agente de compilación y control final

Revisa LaTeX, índice, listados, referencias cruzadas, bibliografía, anexos apaisados y PDF final.

## 12. Skills necesarios para humanos y LLMs

Para reproducir este enfoque se requieren las siguientes capacidades:

1. Delimitación de problema.
2. Lectura bibliográfica con extracción de evidencia.
3. Clasificación de fuentes por función.
4. Construcción de matriz de consistencia.
5. Operacionalización de variables o factores.
6. Diseño de metodología aplicada.
7. Diseño de artefactos o propuestas verificables.
8. Redacción de marco teórico funcional.
9. Uso responsable de Zotero y BibLaTeX.
10. Trazabilidad de citas, figuras, tablas, fórmulas y anexos.
11. Revisión de cumplimiento institucional.
12. Integración discursiva con estilo académico.

## 13. Checklist final

Antes de cerrar un plan, verificar:

- La estructura institucional de especialización está cubierta.
- Los datos generales están completos.
- El problema tiene diagnóstico, antecedentes, formulación, justificación y alcances.
- Los objetivos derivan del problema.
- Las hipótesis, proposiciones o criterios de diseño son evaluables.
- El marco teórico explica la teoría fundamental y los métodos usados.
- La metodología define tipo, nivel, método, unidad de análisis, diseño, instrumentos, análisis y etapas.
- La propuesta o artefacto está claramente separado del objeto de estudio.
- El cronograma, presupuesto y financiamiento están incluidos cuando la unidad lo exige.
- Las referencias siguen APA vigente.
- Las fuentes existen y están en Zotero o en el `.bib`.
- Los anexos están citados en el texto.
- Las matrices son legibles.
- Las figuras tienen fuente en el caption.
- Las fórmulas tienen origen conceptual, numeración, caption inferior e interpretación.
- No se afirma medición directa si solo hay análisis documental, computacional o bibliográfico.
- El PDF compila sin referencias indefinidas.

## 14. Forma recomendada de uso

Para crear un plan nuevo, usar esta guía como mapa de contenido y `guia_redaccion_plan_tesis.md` como control de estilo.

Orden práctico:

1. Leer la Guía N.° 01 y confirmar modalidad.
2. Preparar Zotero y el archivo `.bib`.
3. Construir matriz de consistencia preliminar.
4. Redactar problema y objetivos.
5. Diseñar metodología y artefacto.
6. Redactar marco teórico funcional.
7. Construir anexos.
8. Aplicar guía de redacción.
9. Auditar trazabilidad.
10. Compilar y revisar PDF.

El resultado esperado no es solo un documento correcto en forma. Debe ser un plan que permita entender qué problema se resolverá, por qué importa, cómo se abordará, con qué evidencia se evaluará y qué límites tendrá la propuesta.
