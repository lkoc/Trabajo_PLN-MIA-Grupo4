# Auditoría y entrega

## Objetivo

Cerrar el trabajo mediante comprobaciones reproducibles. Ejecutar las auditorías por separado para evitar que la corrección de estilo oculte un problema científico.

Registrar antes si el documento presenta resultados terminados, resultados parciales o un protocolo. En un protocolo, marcar resultado, comparación empírica y conclusión de desempeño como «no aplica/pendiente», y auditar en su lugar la coherencia del plan de evaluación. Esa condición no autoriza a presentar expectativas como evidencia.

## Pase 1: validez científica

### Problema y contribución

- ¿La brecha describe una diferencia verificable entre situación actual y deseada?
- ¿Las preguntas pueden responderse con el diseño?
- ¿Cada objetivo tiene método, evidencia y conclusión?
- ¿Las contribuciones fueron ejecutadas y evaluadas?
- ¿Se distingue artefacto, actividad y conocimiento aportado?

### Método

- ¿Datos, muestra, filtros y particiones son reconciliables?
- ¿Las unidades de observación y análisis están claras?
- ¿Se explica la procedencia de etiquetas o mediciones?
- ¿Cada alternativa experimental responde a una razón?
- ¿Las métricas están definidas?
- ¿Entrenamiento, validación, calibración y test están separados?
- ¿Se declara cualquier exposición previa a test?
- ¿Software, hardware, versión y configuración relevantes están documentados?

### Resultados y conclusiones

- Si es un protocolo, ¿los campos no ejecutados aparecen como «no aplica/pendiente» y se evita toda cifra anticipada?
- ¿Toda cifra tiene artefacto, campo, universo y versión?
- ¿Las métricas y magnitudes estimadas se presentan con dos cifras significativas y se calcularon con la precisión completa del artefacto?
- ¿Se conservaron exactos los conteos, tamaños muestrales, años, versiones, identificadores, hashes y parámetros fijados por protocolo, así como la precisión necesaria de las constantes físicas?
- ¿Las comparaciones usan el mismo protocolo?
- ¿La incertidumbre se trata de forma adecuada?
- ¿La discusión separa observación de explicación?
- ¿Cada conclusión responde a un objetivo?
- ¿La matriz interna confirma respuesta al objetivo general y a cada objetivo específico, aunque esos rótulos no aparezcan en el cierre publicado?
- ¿La prosa de conclusión evita «objetivo», `O1`/`O2` y fórmulas de cumplimiento, y responde de manera narrativa y natural?
- ¿Para cada distancia entre la situación inicial y la deseada se indica si fue eliminada, reducida o permanece?
- ¿Cada situación pendiente conduce a una acción, recomendación o trabajo futuro concreto?
- ¿La conclusión empieza por el aporte demostrado y después presenta el límite?
- ¿No se afirma causalidad, generalización o autonomía no evaluada?

## Pase 2: auditoría de citas

Aplicar la lista completa de [evidencia-citas-y-bibliografia.md](evidencia-citas-y-bibliografia.md).

Comprobar de forma explícita:

1. toda idea externa, definición y afirmación teórica;
2. todo aspecto sustantivo del estado del arte;
3. paper fundacional de cada algoritmo, arquitectura o método;
4. fuente exacta de datasets, checkpoints, software y estándares;
5. correspondencia semántica entre afirmación y fuente;
6. metadatos, DOI, URL y claves bibliográficas;
7. citas sin entrada, entradas sin uso y duplicados;
8. paráfrasis propia y citas textuales delimitadas;
9. procedencia y licencia de figuras adaptadas;
10. identificación de decisiones locales.

Generar un reporte con cantidades y excepciones. No usar «todo citado» sin una comprobación trazable.

## Pase 3: auditoría de título, resumen y estilo

### Título

Evaluar el título con esta rúbrica:

| Criterio | Pregunta |
|---|---|
| Problema o fenómeno | ¿Se entiende qué necesidad o fenómeno estudia? |
| Artefacto o enfoque | ¿Indica qué se diseña, evalúa o propone? |
| Método | ¿La metodología es central y diferencia el aporte? |
| Objeto y contexto | ¿Delimita datos, población, dominio o escenario cuando importa? |
| Propósito | ¿El verbo o formulación refleja diseño, evaluación, comparación o estimación? |
| Alcance prudente | ¿Evita prometer causalidad, autonomía o universalidad? |
| Recuperación | ¿Incluye términos que un lector usaría para buscar el trabajo? |
| Concisión | ¿Elimina palabras redundantes y siglas oscuras? |
| Fidelidad | ¿Coincide con lo realmente ejecutado y concluido? |

No es obligatorio incluir literalmente todos los elementos. Mencionar la metodología cuando sea distintiva, exigida o central; omitirla si vuelve el título largo sin mejorar identificación. El título debe funcionar como descripción, no como resumen completo.

### Resumen

- ¿Incluye brecha, objetivo, método, resultado y conclusión, o la alternativa explícita de protocolo?
- ¿Es autocontenido?
- ¿Usa resultados reales?
- ¿Define siglas?
- ¿Evita citas y detalle secundario?
- ¿Coincide con las cifras y límites del cuerpo?

### Estilo

- ¿Cada oración contiene una idea principal?
- ¿Los sujetos y verbos son concretos?
- ¿Los términos se usan de forma consistente?
- ¿Las siglas se definen una sola vez?
- ¿La voz y el tiempo verbal siguen el contrato editorial de forma uniforme? En la opción predeterminada, ¿todo el manuscrito usa tercera persona y presente, salvo hechos históricos justificados y trabajo futuro?
- ¿Se eliminó tono promocional, grandilocuencia y repetición?
- ¿Se distingue observación, interpretación y recomendación?
- ¿Cada frase cumple al menos una función necesaria: necesidad, decisión, método, evidencia, interpretación, alcance o acción futura?
- Si se elimina la frase, ¿cambia el argumento, la reproducibilidad, la lectura de la evidencia o una decisión? Si no cambia, ¿puede suprimirse o fusionarse sin pérdida?
- ¿Cada idea repetida cumple una función editorial distinta? Si no, ¿se conservó un solo desarrollo y se usó una referencia cruzada?
- ¿Las semillas, conteos detallados de particiones, volúmenes auxiliares y configuraciones internas permanecen en el cuerpo solo cuando cambian validez, comparabilidad o interpretación?
- ¿Se eliminaron rutas de scripts, identificadores internos y detalles de arqueología del proyecto que no ayudan a entender ni reutilizar el artefacto?
- ¿Cada limitación cambia la interpretación o el uso y conduce a un control o trabajo futuro? ¿Se retiraron cautelas obvias y fórmulas genéricas que desacreditan el aporte sin añadir información?
- ¿Nombres, tildes, unidades y símbolos son correctos, y el redondeo aplica dos cifras significativas solo a estimaciones y magnitudes medidas?
- ¿La redacción es propia y natural, sin intentar imitar o evadir detectores?

## Pase 4: auditoría visual y técnica

- Compilar desde cero cuando el formato lo permita.
- Comprobar el tamaño físico de página exigido; para A4 debe ser 210 × 297 mm, además de estar declarada la opción correspondiente en la clase o plantilla.
- Resolver errores, citas o referencias indefinidas.
- Revisar cajas fuera de margen y avisos relevantes.
- Inspeccionar cada página o diapositiva a tamaño normal.
- Comprobar tablas, captions, numeración y referencias cruzadas.
- Recorrer las primeras menciones de figuras y tablas y confirmar que ambas secuencias sean monótonas; revisar que cada recurso aparezca cerca de la oración que lo introduce.
- Inventariar todas las figuras, tablas y apéndices; comprobar que cada uno tenga una referencia entrante y una explicación breve y natural en el cuerpo. El caption no cuenta como explicación.
- Comparar los recursos visuales por pares y fusionar o retirar los que presenten la misma secuencia, jerarquía o comparación sin una función distinta.
- Verificar que ninguna línea atraviese texto o cajas.
- Confirmar el orden editorial exigido; en el esquema usual de IEEE, cuerpo, apéndices, agradecimientos si existen y referencias. Verificar que cada apéndice empieza en página nueva, tiene un título descriptivo, recibe al menos una referencia explícita desde el lugar pertinente del cuerpo y que cualquier cambio temporal a una columna vuelve al formato principal antes de las referencias.
- Confirmar que la orientación apaisada se reserve para recursos intrínsecamente anchos, mejore su legibilidad frente a la alternativa vertical y conserve márgenes y numeración de página.
- Revisar la forma de autoría en la bibliografía: bajo la convención de esta guía, más de tres autores se presentan como los tres primeros seguidos de «et al.», sin eliminar nombres del metadato bibliográfico fuente.
- Rechazar tablas anchas que dependan de letra pequeña: comprobar reducción de columnas, encabezados partidos y observaciones distribuidas en dos renglones cuando sea necesario.
- Comprobar que las tablas cortas no se estiren sin necesidad; que sus bloques, separadores e indentaciones sean consistentes; y que un aumento razonable de renglones permita recuperar una tipografía legible.
- Verificar que cada límite reportado se identifique como configurado u observado y que la unidad coincida con la implementada.
- Confirmar en el PDF que las barreras de flotantes mantienen cada figura después del texto que la introduce y antes de la subsección siguiente cuando esa relación sea necesaria.
- Probar gráficos comparativos de pocos modelos o métricas a una columna, con espacios entre grupos reducidos y etiquetas y valores legibles.
- Revisar en diagramas el ajuste conjunto de fuente, ancho, alto, separación y rutas ortogonales; no aceptar cajas o trayectorias superpuestas.
- Revisar escala de grises, contraste y tipografía.
- Confirmar que todas las imágenes y fuentes viajan con el paquete.
- Comparar fecha del PDF con sus fuentes.
- Verificar que todas las páginas, incluida la primera y las apaisadas, tengan el número visible cuando el formato de entrega lo requiera.

Un registro limpio no reemplaza la inspección visual.

## Pase 5: reproducibilidad, ética y disponibilidad

- ¿Existe una instrucción mínima para reconstruir tablas, figuras y resultados?
- ¿Se identifican versión, entorno, semillas y dependencias relevantes?
- ¿Los enlaces y rutas funcionan?
- ¿Datos, código, modelos y recursos tienen licencia o restricción separada?
- ¿Se protegen datos sensibles?
- ¿Se declara aprobación ética o razón de no aplicabilidad según el caso?
- ¿Los archivos compartidos corresponden a la versión descrita?
- ¿Los materiales históricos están diferenciados de los activos?
- ¿La declaración publicada se limita a decir qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorios, dejando SHA, commits y hashes concretos en manifiestos técnicos fuera del cuerpo?

## Consistencia artículo–presentación

Comparar automáticamente o manualmente:

| Elemento | Debe coincidir |
|---|---|
| Título y contribución | Formulación y alcance |
| Datos | Tamaño, particiones y procedencia |
| Método | Configuración esencial |
| Resultados | Cifras, métrica y universo |
| Decisión | Criterio de selección |
| Conclusiones | Respuestas narrativas, estado de cada distancia, logro, límite y acciones pendientes |
| Disponibilidad | Artefactos ofrecidos y repositorios, sin huellas técnicas en el cuerpo |

La presentación se corrige después del paper, no al revés, salvo que el cambio revele un error en la fuente.

## Paquete de entrega

Según la convocatoria, incluir:

- fuente principal y PDF;
- bibliografía;
- figuras y tablas reproducibles;
- material suplementario;
- declaración de datos, código, ética y conflictos;
- carta o formulario;
- presentación y recursos;
- instrucciones de compilación;
- manifiesto o lista de archivos.

Excluir datos sensibles, credenciales, archivos temporales y artefactos obsoletos que puedan confundirse con la versión final.

## Criterios de bloqueo

No entregar mientras exista alguno:

- resultado central sin origen;
- cita inventada o que no respalda la afirmación;
- comparación incompatible presentada como directa;
- conclusión que usa test para justificar selección;
- autoría o afiliación sin confirmar;
- recurso visual ilegible o engañoso;
- referencia o figura ausente;
- licencia o restricción crítica ignorada;
- contradicción material entre paper y presentación.

## Informe de cierre

Registrar:

- versión y fecha;
- archivos auditados;
- validaciones y comandos;
- número de citas y entradas;
- errores y advertencias pendientes;
- revisión visual realizada;
- limitaciones no corregibles;
- responsable o decisión necesaria.

Hacer una segunda lectura integral después de todas las correcciones. Los cambios de cierre pueden introducir contradicciones nuevas.
