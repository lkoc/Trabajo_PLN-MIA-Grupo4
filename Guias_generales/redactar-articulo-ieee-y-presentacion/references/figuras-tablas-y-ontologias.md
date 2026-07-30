# Figuras, tablas y ontologías

## Elegir el recurso correcto

Usar una visualización cuando reduzca carga cognitiva frente a la prosa:

| Necesidad | Recurso recomendado |
|---|---|
| Comparar valores exactos | Tabla |
| Comparar magnitudes o tendencias | Gráfico |
| Mostrar pasos o estados | Flujo o línea temporal |
| Mostrar componentes y conexiones | Arquitectura |
| Mostrar jerarquía | Árbol |
| Mostrar entidades y relaciones semánticas | Ontología |
| Mostrar reducción de datos | Embudo |
| Relacionar problemas, objetivos y evidencia | Matriz |

No convertir toda lista en figura. Una figura debe responder una pregunta identificable.

## Inventario y redundancia visual

Antes de maquetar, registrar para cada figura y tabla:

| Campo | Comprobación |
|---|---|
| Pregunta | Qué relación, secuencia, jerarquía o comparación aclara |
| Aporte | Qué idea nueva deja frente a la prosa y otros recursos |
| Evidencia | De qué fuente o artefacto proceden sus datos |
| Referencia entrante | En qué oración del cuerpo se menciona |
| Interpretación | Qué observa el texto sin repetir todo el caption |

Comparar los recursos por pares. Si dos diagramas contienen casi las mismas etapas, flechas o salidas, integrar en uno la información necesaria y retirar el otro. Si una tabla y un gráfico usan los mismos valores, conservar ambos solo cuando la tabla permita consulta exacta y el gráfico haga visible un patrón relevante. No mantener una figura por razones decorativas, cronológicas o porque ya estaba producida.

## Procedencia y reproducibilidad

- Generar gráficos desde CSV, JSON, base de datos o fuente canónica.
- Conservar el código o fuente vectorial junto al entregable.
- Registrar filtros, orden, regla de cifras significativas y fecha de corte.
- No editar valores manualmente en la imagen final.
- Indicar si la figura es propia, reproducida o adaptada.
- Verificar licencia o permiso cuando no sea elaboración propia.
- Añadir texto alternativo cuando el formato lo permita.

## Tablas

Una tabla debe:

- tener título o caption autocontenido;
- declarar universo, partición, unidad y métrica;
- presentar métricas y magnitudes estimadas con dos cifras significativas y conservar exactos los conteos, tamaños muestrales, años, versiones, identificadores, hashes y parámetros fijados por protocolo;
- resaltar solo la comparación válida;
- explicar abreviaturas y símbolos;
- distinguir no disponible de cero;
- evitar demasiadas columnas estrechas;
- poder leerse sin buscar definiciones esenciales en otra página.
- usar solo el ancho que exige su contenido; una tabla corta puede centrarse con ancho natural en vez de estirar columnas y crear espacio vacío.

Cuando el ancho adicional permita recuperar una tipografía legible sin rebasar márgenes, aprovechar primero el ancho disponible de la columna o de la página. No reducir una tabla por debajo de un tamaño cómodo si todavía queda espacio horizontal útil.

No mezclar resultados de protocolos distintos. Si una celda no es comparable, separarla o marcarla explícitamente.

Redondear solo la representación final: los cálculos y la selección deben usar los valores completos del artefacto canónico. Las constantes físicas pueden conservar la precisión que requiera su uso. Cuando un cero final sea significativo, mostrarlo explícitamente; por ejemplo, `0,40` y `3,0` comunican dos cifras significativas.

Si una tabla apaisada o muy ancha obliga a usar letra pequeña, rediseñarla antes de escalarla:

1. eliminar o trasladar columnas secundarias;
2. dividir encabezados largos en dos líneas;
3. representar cada observación en dos renglones relacionados —el primero con su identificación y comparación principal, el segundo con los campos complementarios—;
4. agrupar métricas compatibles o dividir la tabla en paneles;
5. aumentar de nuevo la tipografía y comprobarla al tamaño final.

Cuando una tabla reúne bloques conceptuales, puede separarlos con reglas horizontales finas o discontinuas y un encabezado interno. Las filas subordinadas deben llevar una indentación consistente. El separador organiza la lectura, no reemplaza los encabezados ni debe producir una cuadrícula pesada.

Si una letra mayor aumenta el número de renglones pero conserva la tabla dentro de la página, se prefiere esa solución. Revisar después el alto total, los cortes de página y la alineación vertical. Una tabla de doble columna que no puede dividirse debe rediseñarse antes de desbordar la página.

No priorizar el menor número de páginas, filas o tablas sobre la legibilidad. Una tabla compacta que exige zoom debe reestructurarse.

## Gráficos cuantitativos

- Elegir ejes, escala y cero de acuerdo con el fenómeno.
- Mostrar incertidumbre cuando sea relevante.
- No usar área o volumen para representar una magnitud lineal.
- Evitar gráficos tridimensionales decorativos.
- Usar una paleta accesible y distinguir series también por forma o patrón.
- Mantener el mismo color semántico entre paper y presentación.
- Etiquetar las unidades.
- No ocultar valores desfavorables mediante recorte de ejes.
- Validar cada valor contra su fuente.

En comparaciones con pocos modelos o métricas, probar primero una figura de una columna. Reducir el espacio vacío entre grupos y barras, ajustar márgenes y leyenda, y mantener nombres, valores y unidades legibles sin depender de una ampliación. Usar dos columnas solo cuando la comparación realmente necesite ese ancho.

## Diagramas de flujo y arquitectura

Diseñar primero la topología y después el estilo:

1. identificar entradas, procesos, decisiones, almacenes y salidas;
2. ordenar la lectura de izquierda a derecha o de arriba abajo;
3. reservar espacio entre grupos;
4. usar rutas ortogonales, con segmentos verticales y horizontales;
5. asignar carriles externos a relaciones que cruzan niveles;
6. reordenar nodos antes de aceptar un cruce;
7. colocar rótulos fuera de las trayectorias;
8. limitar el texto dentro de cajas;
9. ajustar conjuntamente tamaño de fuente, ancho, alto y espaciado de cada caja;
10. ampliar la separación horizontal o vertical cuando el texto, las flechas o los rótulos compitan por el mismo espacio.

Anclar cada conector de forma explícita al lado lógico del nodo: `east` hacia un elemento situado a la derecha, `west` hacia la izquierda y `north`/`south` entre niveles. En una bifurcación, usar un tronco ortogonal y una unión visible antes de abrir las ramas. Dejar un tramo libre entre caja, punta de flecha y rótulo; comprobar que la ruta no parezca entrar por una esquina incorrecta ni terminar sobre el borde equivocado.

Ninguna caja debe solaparse con otra ni invadir un rótulo. Ninguna línea debe atravesar una caja, palabra, leyenda o cifra. No «resolver» un cruce haciéndolo más fino o transparente. Si varias flechas convergen, usar un nodo de unión claro o separar etapas. Recalcular ancho, alto, fuente, separación y rutas ortogonales después de cada cambio de texto.

## Ontologías y mapas conceptuales

Usar una ontología cuando se necesite mantener consistencia entre conceptos, datos, método y artefactos. Definir:

- clases o tipos de entidad;
- propiedades o relaciones;
- dirección y cardinalidad cuando aporten;
- términos canónicos;
- equivalencias y exclusiones;
- fuente académica o condición de decisión local;
- ejemplo de instancia sin datos sensibles.

La ontología gráfica debe ser una vista legible, no el volcado completo de un archivo formal. Puede existir además una representación formal o un glosario tabular en los materiales de reutilización.

### Trazabilidad conceptual

Cuando un concepto proviene de la literatura y una categoría se adapta localmente, mostrar la secuencia:

> concepto externo → definición operacional → variable o etiqueta → medición → resultado → decisión.

No presentar el nombre operativo como categoría jurídica o consenso experto sin evidencia.

## Captions y referencias cruzadas

Mencionar cada figura o tabla en el texto antes o inmediatamente después de su aparición. El texto debe explicar el hallazgo, no repetir todos los elementos visuales. Aplicar la misma regla a los recursos de los apéndices: citarlos desde el cuerpo en el pasaje donde amplían el argumento y explicitar qué aporta cada uno. El caption, una lista de figuras o una referencia genérica al apéndice no sustituyen la mención del elemento.

Asignar la numeración según el orden de la primera mención sustantiva. Antes de compilar, recorrer las referencias en el orden del texto y comprobar una secuencia monótona independiente para figuras y tablas. No citar anticipadamente una figura posterior solo como avance teórico. Si el recurso está en un apéndice y la numeración es continua, el cuerpo puede remitir al apéndice en el punto pertinente y la prosa del apéndice debe citar y explicar cada figura o tabla antes de presentarla; si la sede exige cita directa desde el cuerpo, considerar numeración propia del apéndice.

La posición lógica también debe cumplirse en el PDF. Si una figura cierra una subsección, colocarla en la fuente después del texto correspondiente y usar las barreras o controles de flotantes de la plantilla para impedir que aparezca antes del encabezado o detrás de la subsección siguiente. En figuras de doble columna, aceptar un salto a la página siguiente cuando sea necesario para preservar ese orden; nunca prometer una posición exacta sin inspeccionar el PDF compilado.

Un caption adecuado responde:

- ¿qué muestra?;
- ¿sobre qué universo?;
- ¿cómo se calculó o agrupó?;
- ¿qué abreviaturas usa?;
- ¿cuál es su procedencia?;
- ¿qué límite evita una lectura equivocada?

## Adaptación a dos columnas y proyección

La misma figura puede requerir dos variantes:

- paper: legible al ancho real de una o dos columnas;
- presentación: menos detalle, tipografía mayor y foco en un hallazgo.

No escalar una figura compleja hasta volver ilegibles sus rótulos. Simplificar o dividir. Comprobar impresión en escala de grises y proyección con contraste moderado.

### Orientación apaisada en apéndices

Usar una página apaisada cuando el recurso sea intrínsecamente ancho y el giro produzca una mejora visible de lectura, por ejemplo:

- una captura de interfaz cuya organización lateral sea parte de la evidencia;
- una tabla con varias columnas esenciales que no admite una división clara;
- un grafo u ontología con flujo horizontal y múltiples ramas;
- una figura cuya reducción en vertical vuelva ilegibles rótulos o conectores.

No usarla para evitar un rediseño simple ni para una tabla corta. Antes de girar, probar reducción de columnas, encabezados en dos líneas, disposición en dos paneles o una sola columna. Después del giro, usar el ancho útil sin tocar márgenes, mantener el número de página visible y orientado de manera consistente con la plantilla, empezar el apéndice en página nueva y comprobar que el documento vuelve a su orientación y número de columnas normales.

## Auditoría visual

Inspeccionar el PDF final a tamaño normal, no solo la fuente:

- texto cortado o fuera de cajas;
- flechas que atraviesan nodos;
- cruces ambiguos;
- leyendas superpuestas;
- rótulos demasiado pequeños;
- diferencias entre caption y datos;
- colores indistinguibles;
- recursos rasterizados o borrosos;
- tablas que salen del margen;
- tablas cuya letra obliga a ampliar la página;
- gráficos comparativos con separación excesiva o valores ilegibles;
- cajas superpuestas o demasiado ajustadas al texto;
- figuras citadas pero ausentes;
- figuras, tablas o apéndices presentes pero nunca mencionados y explicados desde el cuerpo;
- pares de figuras que repiten la misma secuencia, jerarquía o comparación sin aportar una lectura distinta;
- numeración o referencias cruzadas incorrectas.
- orden de numeración distinto del orden de primera mención;
- páginas sin número o números perdidos/rotados de forma incoherente al cambiar de orientación;

Repetir la inspección después de cualquier cambio de texto, escala o diseño. La compilación exitosa no sustituye esta revisión.
