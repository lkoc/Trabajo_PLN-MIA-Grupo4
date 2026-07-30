---
name: redactar-articulo-ieee-y-presentacion
description: Planifica, redacta, revisa y valida artículos académicos en estilo IEEE y presentaciones derivadas, con búsquedas bibliográficas profundas y trazabilidad entre la situación investigada, las interrogantes, los objetivos, los métodos, la evidencia, los resultados, las conclusiones, las citas y los recursos visuales. Usar cuando se deba crear o corregir un manuscrito científico, convertir un proyecto en un paper, construir cadenas booleanas de búsqueda, preparar su Beamer o presentación, auditar referencias, evitar afirmaciones sin respaldo o cerrar un paquete reproducible para entrega o publicación.
---

# Redactar un artículo IEEE y su presentación

Producir un manuscrito verificable y una presentación fiel al mismo. Tratar las reglas de la convocatoria, la evidencia disponible y los artefactos ejecutados como restricciones, no como detalles editoriales.

## Principios no negociables

- No inventar cifras, fuentes, autores, DOI, versiones, licencias, experimentos ni resultados.
- Separar siempre: conocimiento externo, evidencia interna, decisión del equipo, inferencia y trabajo pendiente.
- Distinguir resultados preliminares de la comparación final. No llamarlos «históricos» cuando solo representan iteraciones tempranas del mismo estudio. Resumirlos en prosa si su única función es explicar una decisión posterior y retirar tablas que no aportan una comparación válida.
- Citar toda idea ajena y respaldar cada resultado propio con su artefacto de origen.
- No crear una autorreferencia bibliográfica para un corpus, modelo, código u otro producto que se presenta por primera vez dentro del mismo artículo. Explicarlo como resultado propio y remitir a sus tablas, anexos, manifiestos o repositorios; citarlo solo si existe una publicación previa, independiente y pertinente.
- No usar el conjunto final de prueba para elegir modelos, umbrales, hipótesis o redacción favorable.
- No presentar una propuesta, una salida de demostración o una expectativa como resultado ejecutado.
- No forzar Design Science Research (DSR) ni otra metodología: elegirla solo cuando corresponda al tipo de contribución.
- Verificar las instrucciones vigentes del evento o revista antes de fijar plantilla, extensión, anonimización o secciones.
- Redactar con lenguaje directo y propio. La integridad se demuestra con trazabilidad y atribución, no intentando engañar detectores automáticos.
- Mantener una voz editorial estable. Salvo exigencia distinta de la publicación, redactar en tercera persona, tiempo presente y tono neutro; usar construcciones impersonales solo cuando aclaren el sujeto. Reservar el pasado para hechos históricos cuya fecha sea parte del argumento y no alternar tiempos por costumbre entre método, resultados y conclusiones.
- Elegir un término canónico para cada nivel de la taxonomía. Por ejemplo, usar «categoría de moderación» para una salida operativa, «etiqueta» para el rótulo almacenado y «daño» para el fenómeno o estado semántico; no intercambiarlos como sinónimos.
- Exigir una función comunicativa a cada frase: plantear la necesidad, explicar una decisión, describir el método, presentar evidencia, interpretar un resultado, delimitar un uso o proponer una acción. Eliminar arqueología interna, nombres de scripts, identificadores personales, cautelas obvias y frases que solo dicen que una fuente o herramienta «no demuestra» algo que el artículo nunca le atribuye. Si al retirar una oración no cambia el argumento, la reproducibilidad, la interpretación ni una decisión, suprimirla o integrarla en una oración cercana.
- Auditar la redundancia por idea, no solo por coincidencia literal. Conservar una repetición únicamente cuando cumple una función distinta —por ejemplo, síntesis en el resumen, evidencia en resultados e implicación en conclusiones—; en los demás casos, eliminarla o sustituirla por una referencia cruzada al lugar que la desarrolla.
- Mantener en el cuerpo solo cifras y parámetros que cambian la validez, la comparación o la interpretación. Trasladar semillas, conteos exactos de cada partición, volúmenes auxiliares de entrenamiento y configuraciones exhaustivas a tablas, anexos o artefactos reproducibles, salvo que sean el resultado estudiado. Una partición puede describirse por su regla y ausencia de fuga sin enumerar todos sus conteos en cada sección.
- Nombrar formatos técnicos —por ejemplo, JSON, CSV o una extensión concreta— solo cuando se explique cómo se almacenan, intercambian o reutilizan los datos. En la narración científica, preferir el objeto y su función: «corpus», «resultados de validación», «registro de decisiones» o «manifiesto». Reservar nombres de archivos y rutas para guías técnicas, manifiestos o material suplementario.
- Presentar primero la capacidad lograda y formular después las limitaciones que cambian su interpretación o uso. Una limitación debe ser específica y conducir a una decisión, control o trabajo futuro; no debe desacreditar de forma genérica el aporte ni acumular reservas sin consecuencia práctica.
- Mantener separados el diagnóstico y la indagación: describir la situación de forma declarativa; formular después lo que el estudio examina. No convertir una carencia, dificultad o condición observable en una oración interrogativa.
- Conservar la denominación académica de «problema» en encabezados, matrices y relaciones de trazabilidad, por ejemplo «Problemas y objetivos» o «Problema específico». Redactar cada problema como una situación declarativa que exponga el estado actual, el deseado, las condiciones que dificultan alcanzarlo y la capacidad requerida; no sustituirlo por una pregunta. En la prosa continua, evitar fórmulas metadiscursivas repetitivas como «el problema es» o «la pregunta es» cuando el enunciado pueda presentarse directamente.
- Reportar métricas y demás magnitudes estimadas con dos cifras significativas. Conservar sin redondear los conteos exactos, el tamaño muestral, los años, las versiones, los identificadores, los hashes y los parámetros fijados por protocolo; las constantes físicas pueden mantener la precisión que exija su uso. Calcular y comparar siempre con la precisión completa del artefacto fuente y redondear solo al presentar.
- Reservar SHA, hashes y commits concretos para manifiestos, registros o anexos técnicos externos al cuerpo. En la declaración publicada de disponibilidad, indicar únicamente qué datos, cuadernos, scripts y artefactos se ofrecen y en qué repositorio se encuentran; no convertirla en un inventario de huellas técnicas.
- Priorizar legibilidad sobre compactación. En el orden editorial usual de IEEE, colocar los apéndices después del cuerpo y antes de los agradecimientos y las referencias, salvo instrucción distinta de la publicación. Iniciar cada apéndice en una página nueva y titularlo por su contenido; si hace falta, usar temporalmente una columna y restaurar el formato principal antes de las referencias. Reestructurar tablas, gráficos y diagramas antes de reducir la tipografía hasta un tamaño incómodo.
- Citar cada apéndice explícitamente desde el cuerpo en el pasaje donde su contenido amplía el argumento. Un apéndice sin referencia entrante debe integrarse, justificarse y citarse, o eliminarse.
- Mencionar y explicar brevemente en la prosa cada figura y tabla, incluidos los elementos de los apéndices. El caption y la mera presencia del recurso no cuentan como referencia entrante; si no aporta una idea distinta, integrarlo con otro recurso o retirarlo.
- Compilar y revisar visualmente los entregables; una compilación sin errores no garantiza calidad.
- Evitar páginas con grandes vacíos causados por barreras de flotantes. Permitir que el texto continúe entre tablas y figuras; usar una columna temporal solo cuando un elemento no sea legible a dos columnas y comprobar que la transición no produzca páginas parciales innecesarias.

## Elegir la ruta

1. **Artículo nuevo con resultados:** recorrer todo el flujo desde el contrato editorial hasta la auditoría.
2. **Revisión de manuscrito:** inventariar afirmaciones y evidencia, detectar brechas y corregir por prioridad.
3. **Presentación:** congelar primero la versión del artículo que será la fuente de verdad y después sintetizarla.
4. **Auditoría final:** no reescribir por intuición; ejecutar las comprobaciones científicas, bibliográficas, editoriales, visuales y reproducibles.
5. **Protocolo o trabajo en curso:** redactar la situación investigada, las interrogantes, los antecedentes, el método y el plan de evaluación; etiquetar como pendientes los resultados y reservar las conclusiones empíricas hasta ejecutar el estudio. No usar resultados esperados como evidencia.

El flujo siguiente define controles obligatorios, no una extensión fija. En un bosquejo se puede abreviar la forma, pero no omitir el origen de la evidencia, el estado real del trabajo ni las brechas pendientes.

## Flujo obligatorio

### 1. Establecer el contrato

Confirmar tema, situación investigada, interrogante, contribución, audiencia, idioma, modalidad, plantilla, tamaño físico de hoja, límite de páginas, política de anonimato, fecha, disponibilidad de datos y estado real del trabajo. Si falta información, señalar el supuesto sin fabricar contenido.

Leer [contrato-y-trazabilidad.md](references/contrato-y-trazabilidad.md) para construir el inventario de fuentes de verdad y las matrices de control.

### 2. Auditar la evidencia antes de escribir

Localizar artículos, datos, cuadernos, código, configuraciones, registros, figuras, modelos y resultados. Determinar cuál versión manda cuando hay conflicto y registrar fecha, versión o hash cuando sea útil.

Crear al menos:

- matriz situación/interrogante → objetivo → método → evidencia → resultado → conclusión;
- registro afirmación → tipo → fuente externa → artefacto interno → ubicación;
- inventario de experimentos comparables y de sus diferencias.

### 3. Diseñar el argumento

Describir el problema como la diferencia entre la situación actual y la deseada mediante enunciados verificables. En investigación aplicada, separar: 1) la condición observable; 2) los factores que la sostienen; y 3) la capacidad técnica o de diseño que falta construir o evaluar. Denominar estos enunciados «problema general» y «problemas específicos» cuando la estructura académica lo requiera, pero redactarlos siempre como situaciones, nunca como interrogaciones. Después, formular de manera separada aquello que la investigación examina, en forma directa o indirecta; preferir «el estudio examina cómo...» a «la pregunta general es...». Definir objetivos respondibles por la evidencia y contribuciones ya demostradas.

Leer [estructura-y-redaccion-ieee.md](references/estructura-y-redaccion-ieee.md). Auditar el título por necesidad o fenómeno, artefacto o enfoque, método cuando sea distintivo, contexto, propósito, alcance y concisión. Usar la estructura de la convocatoria; aplicar la organización IEEE propuesta allí como base adaptable, no como regla universal.

### 4. Explicar método y evaluación

Describir para cada etapa la entrada, intervención o procedimiento, salida, control de calidad y vínculo con un objetivo. Justificar alternativas, datos, variables, algoritmos, arquitectura experimental, métricas y recursos computacionales. Para cada familia de modelos, explicar brevemente su mecanismo, la razón de selección, una ventaja, una limitación y el aporte fundacional que la respalda. Incluir en el cuerpo solo los hiperparámetros o detalles internos que cambian la interpretación, la comparabilidad o la reproducción esencial; trasladar configuraciones completas a tablas suplementarias, manifiestos o artefactos. Para cada indicador, presentar definición o fórmula, dirección de mejora, agregación, dependencia del umbral y utilidad para la decisión del estudio, con una fuente metodológica pertinente.

Leer [metodologia-resultados-y-validez.md](references/metodologia-resultados-y-validez.md). Si se construyó y evaluó un artefacto mediante iteraciones, documentar DSR; si no, escoger el diseño apropiado y explicitarlo.

### 5. Construir el aparato de citas

Buscar fuentes primarias o fundacionales, comprobar que cada una sostenga la afirmación exacta y registrar la referencia completa. Añadir fuentes exactas para conjuntos de datos, checkpoints, software, estándares y políticas cuando sean parte del método.

Cuando la búsqueda deba empezar con pocas palabras o cubrir el campo en profundidad, leer y ejecutar [busqueda-bibliografica-profunda.md](references/busqueda-bibliografica-profunda.md). Construir bloques de sinónimos con OR, unir conceptos con AND, usar NOT solo tras comprobar pérdidas, traducir la sintaxis a cada base y ampliar recursivamente mediante vocabularios, artículos centinela y redes de citas.

Leer [evidencia-citas-y-bibliografia.md](references/evidencia-citas-y-bibliografia.md) antes de cerrar antecedentes, teoría, método y bibliografía. Cuando el contrato editorial establezca abreviación por autoría, configurarla en el gestor bibliográfico sin borrar autores de los metadatos; para la convención de esta guía, una obra con más de tres autores muestra los tres primeros seguidos de «et al.».

### 6. Redactar resultados, discusión y cierre

Organizar resultados por objetivo o aspecto evaluado, no por orden de archivos o celdas. Comparar solo universos equivalentes. Separar magnitud observada, interpretación, explicación plausible y recomendación. Aplicar dos cifras significativas a métricas y magnitudes estimadas sin alterar la precisión almacenada ni redondear conteos o identificadores exactos. Si se informa un ganador global o por categoría, elegirlo con validación o con la partición de selección declarada, congelar la decisión y usar test únicamente para estimar el desempeño resultante; el caption y la prosa deben nombrar la métrica y la partición de selección.

Trazar internamente el objetivo general y cada objetivo específico hasta el cierre, pero redactar las conclusiones publicadas como una narración fluida: no usar «objetivo», códigos como `O1`/`O2` ni fórmulas del tipo «se cumplió el objetivo». Para cada distancia entre la situación inicial y la deseada, indicar con evidencia si se eliminó, se redujo o permanece. Cerrar con acciones, recomendaciones o trabajo futuro vinculados de manera explícita con lo que siga pendiente. Declarar limitaciones y amenazas de validez sin borrar primero el aporte efectivamente demostrado.

### 7. Diseñar tablas, figuras y ontología

Usar una visualización solo si hace más clara una relación, secuencia, comparación, jerarquía o trazabilidad. Generarla desde datos o fuentes reproducibles siempre que sea posible.

Antes de cerrar, inventariar para cada tabla y figura: pregunta que responde, idea nueva, ubicación, referencia entrante y explicación en la prosa. Comparar pares visuales que compartan entidades, flechas o resultados; si comunican la misma relación, integrar la información útil en uno solo. Mantener tabla y gráfico sobre los mismos datos únicamente cuando una ofrece valores exactos y el otro revela un patrón que la prosa no muestra con igual claridad.

Leer [figuras-tablas-y-ontologias.md](references/figuras-tablas-y-ontologias.md). Verificar que ninguna línea atraviese cajas o texto, que los conectores salgan y entren por el lado lógico de cada nodo, que los rótulos sean legibles al tamaño final y que color no sea el único canal semántico. Ante una tabla demasiado ancha, reducir columnas, partir encabezados, agrupar bloques con separadores discretos y distribuir cada observación en dos renglones antes de encoger la letra. Usar el ancho natural de una tabla cuando el contenido no necesita ocupar toda la columna y aprovechar el ancho disponible antes de reducir su tipografía. Probar una columna para gráficos comparativos con pocas series y ajustar fuente, cajas, separación y rutas ortogonales en diagramas. Numerar figuras y tablas en el orden de su primera mención sustantiva, ubicar cada flotante cerca de esa mención y usar barreras solo cuando sean necesarias y se comprueben en el PDF. No adelantar en teoría una referencia directa que rompa ese orden; cuando corresponda, remitir allí al apéndice y explicar el recurso concreto en la prosa del propio apéndice. Elegir orientación apaisada solo cuando el ancho mejora de forma material una tabla, captura o grafo y no basta una columna o un rediseño; cada página apaisada debe conservar márgenes, número de página y letra legible. No aceptar figuras, tablas o apéndices huérfanos: cada uno debe tener referencia y explicación natural.

### 8. Derivar la presentación

Crear la presentación solo desde el manuscrito auditado y sus datos canónicos. Mantener una idea principal por diapositiva, reducir texto, mostrar el argumento mediante evidencia visual y conservar citas breves.

Si el artículo y la presentación se solicitan a la vez antes de terminar el estudio, crear primero un bosquejo fuente y presentar la exposición como protocolo o trabajo en curso. No reservar cajas que aparenten resultados ni formular conclusiones empíricas.

Leer [presentacion-academica.md](references/presentacion-academica.md) para definir narrativa, selección de contenido, diseño y ensayo.

### 9. Ejecutar el cierre

Aplicar al menos dos pasadas independientes por propósito: una de validez científica y otra de calidad editorial/visual. Puede realizarlas la misma persona o agente, pero como lecturas separadas; un segundo revisor mejora el control cuando está disponible. Ejecutar además la auditoría de citas y, cuando el riesgo lo justifique, una revisión separada de reproducibilidad. Comprobar título, resumen, aparato bibliográfico, estilo, figuras y coherencia con la presentación.

Leer y ejecutar [auditoria-y-entrega.md](references/auditoria-y-entrega.md). Añadir una pasada frase por frase para comprobar aporte, necesidad y tono. No declarar el trabajo terminado mientras queden referencias indefinidas, cifras sin origen, contradicciones entre artículo y presentación o fallos visuales materiales.

## Formato de respuesta al usar esta skill

Entregar, según el encargo:

1. diagnóstico breve del estado y de los aspectos pendientes;
2. decisiones y supuestos explícitos;
3. artefactos creados o modificados;
4. evidencia y fuentes usadas;
5. validaciones ejecutadas y sus resultados;
6. pendientes reales que requieren evidencia o decisión humana.

No ocultar incertidumbre bajo prosa fluida. Si una afirmación no puede verificarse, atenuarla, marcarla o retirarla.

## Referencias de la skill

- [contrato-y-trazabilidad.md](references/contrato-y-trazabilidad.md): alcance, fuentes de verdad y matrices.
- [estructura-y-redaccion-ieee.md](references/estructura-y-redaccion-ieee.md): arquitectura del manuscrito y estilo.
- [metodologia-resultados-y-validez.md](references/metodologia-resultados-y-validez.md): diseño, experimentos, resultados y límites.
- [busqueda-bibliografica-profunda.md](references/busqueda-bibliografica-profunda.md): cadenas booleanas, expansión recursiva y registro reproducible.
- [evidencia-citas-y-bibliografia.md](references/evidencia-citas-y-bibliografia.md): búsqueda, atribución y auditoría bibliográfica.
- [figuras-tablas-y-ontologias.md](references/figuras-tablas-y-ontologias.md): comunicación visual reproducible.
- [presentacion-academica.md](references/presentacion-academica.md): síntesis para exposición.
- [auditoria-y-entrega.md](references/auditoria-y-entrega.md): controles finales.
