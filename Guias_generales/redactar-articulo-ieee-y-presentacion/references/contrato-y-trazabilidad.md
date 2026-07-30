# Contrato editorial y trazabilidad

## Propósito

Definir qué se puede afirmar antes de empezar a redactar. Esta etapa evita que una narración elegante mezcle versiones, atribuya resultados a una ejecución distinta o complete vacíos con suposiciones.

## 1. Ficha de inicio

Registrar como mínimo:

| Campo | Pregunta de control |
|---|---|
| Destino | ¿Revista, conferencia, curso, tesis o preprint? |
| Normas | ¿Plantilla, tamaño físico de hoja, extensión, anonimato, idioma, secciones y formato de referencias? |
| Audiencia | ¿Qué sabe el lector y qué debe definirse? |
| Estado | ¿Trabajo propuesto, en curso o terminado? |
| Contribución | ¿Conocimiento, método, conjunto de datos, artefacto, estudio o combinación? |
| Pregunta | ¿Qué brecha concreta intenta cerrar? |
| Evidencia | ¿Qué datos, ejecuciones y fuentes están realmente disponibles? |
| Restricciones | ¿Privacidad, ética, licencias, confidencialidad o acceso? |
| Entregables | ¿Artículo, anexos, datos, código, presentación y material suplementario? |
| Fecha de corte | ¿Qué versión y qué momento representan los resultados? |

Verificar en una fuente oficial las reglas actuales del destino. Si no se dispone de acceso, pedir la plantilla o marcar la comprobación como pendiente.

## 2. Jerarquía de fuentes de verdad

Inventariar los artefactos y asignarles autoridad. Una jerarquía frecuente es:

1. datos y resultados canónicos generados por la ejecución final;
2. manifiestos, configuraciones y registros vinculados con esa ejecución;
3. código o cuadernos que reproducen el proceso;
4. informe técnico contemporáneo a los resultados;
5. salidas incrustadas, borradores y documentos históricos;
6. memoria de los autores.

Esta jerarquía debe adaptarse. Un archivo más nuevo no siempre es más autoritativo: puede ser una copia incompleta. Para resolver un conflicto, comparar:

- hora con zona horaria explícita;
- hash o tamaño;
- versión del esquema;
- identificador de ejecución;
- dependencias y entradas;
- relación con el manifiesto;
- completitud del contenido.

No sobrescribir silenciosamente una discrepancia. Registrar la decisión y conservar la alternativa cuando tenga valor probatorio.

## 3. Inventario de evidencia

Crear una tabla con:

| ID | Artefacto | Tipo | Versión/fecha/hash | Alcance | Autoridad | Restricciones | Uso previsto |
|---|---|---|---|---|---|---|---|
| E-01 | ruta o referencia | datos, código, resultado, fuente | identificador | qué contiene | alta/media/baja | licencia o acceso | sección o afirmación |

Clasificar además los elementos ausentes. La ausencia de un registro de hardware, una semilla o una licencia no autoriza a reconstruirlos retrospectivamente.

Esta tabla y sus hashes pertenecen al control técnico interno o a un manifiesto reproducible. No trasladar SHA o commits concretos al cuerpo publicado por rutina. La declaración de disponibilidad debe limitarse a identificar qué datos, cuadernos, scripts y artefactos están disponibles y el repositorio que aloja cada grupo.

## 4. Matriz maestra del argumento

Usar una fila por pregunta específica:

| Pregunta | Objetivo | Método/actividad | Entrada | Evidencia producida | Resultado | Estado de la distancia | Conclusión | Sección |
|---|---|---|---|---|---|---|---|---|

Reglas:

- toda pregunta debe tener un objetivo y una respuesta;
- todo objetivo debe tener método y evidencia;
- toda conclusión debe apuntar a un resultado;
- cada distancia entre la situación inicial y la deseada debe clasificarse como eliminada, reducida o pendiente;
- una actividad sin objetivo puede ser secundaria o debe justificarse;
- un objetivo sin evidencia debe reformularse como trabajo futuro o retirarse.

Conservar esta trazabilidad en la matriz. En la prosa publicada del cierre, responder de forma natural a todo lo que el estudio se propuso lograr sin escribir «objetivo», códigos `O1`/`O2` ni encabezados de cumplimiento.

En investigación aplicada, añadir requisito del usuario, iteración de diseño y cambio del artefacto. En DSR, esta matriz puede incorporar etapa, evaluación y aprendizaje de cada ciclo.

## 5. Registro de afirmaciones

Mantener durante la redacción:

| Afirmación | Tipo | Fuente externa | Artefacto interno | Campo/split | Versión | Ubicación | Estado |
|---|---|---|---|---|---|---|---|

Tipos mínimos:

- idea o definición externa;
- método publicado;
- dato del dominio;
- decisión local;
- dato del corpus o muestra;
- resultado experimental;
- interpretación;
- recomendación;
- limitación.

Estados útiles: verificada, atenuar, citar, recalcular, aclarar o retirar.

### Prueba de trazabilidad

Para cada frase sustantiva preguntar:

1. ¿Es conocimiento ajeno? Entonces necesita una fuente pertinente.
2. ¿Es un dato propio? Entonces necesita un artefacto reproducible.
3. ¿Es una decisión? Entonces debe declararse como tal y justificarse.
4. ¿Es una interpretación? Entonces debe distinguirse del resultado observado.
5. ¿Es una expectativa? Entonces pertenece a trabajo futuro, no a resultados.

Para límites de longitud, tiempo, capacidad o tamaño, distinguir siempre tres campos: parámetro configurado, regla real de cierre y máximo observado en el artefacto final. No convertir un máximo observado en hiperparámetro ni expresar en otra unidad un límite que el código no fija. Si se informa una conversión o conteo derivado, documentar su regla —por ejemplo, palabras separadas por espacios— y el archivo canónico del que procede.

## 6. Inventario de experimentos

Antes de comparar, registrar:

| Experimento | Datos/split | Variable objetivo | Método | Inicialización | Ajuste | Criterio de selección | Métricas | Estado |
|---|---|---|---|---|---|---|---|---|

Comparar directamente solo filas con contrato compatible. Si cambian datos, etiqueta, métrica, preprocesamiento o protocolo, presentar la relación como histórica o exploratoria.

Separar:

- entrenamiento;
- selección y ajuste de hiperparámetros;
- calibración o fijación de umbrales;
- evaluación final;
- análisis posterior.

Registrar cualquier consulta previa al conjunto final de prueba, aunque el código de selección no lo utilice.

## 7. Ontología o glosario de conceptos

Crear un glosario cuando el trabajo usa términos cercanos, capas de etiquetas, componentes o entidades que pueden confundirse. Para cada concepto registrar:

| Término canónico | Definición | Sinónimos permitidos | No confundir con | Fuente/decisión | Relaciones |
|---|---|---|---|---|---|

Una ontología formal o gráfica es útil si hay al menos tres clases de entidades y relaciones repetidas. Debe mostrar, por ejemplo, cómo un dato produce una observación, cómo una observación alimenta un modelo y cómo una predicción es revisada. No inventar consenso académico: marcar qué conceptos son externos y cuáles son decisiones operativas.

## 8. Control de cambios

- Fijar una fecha o versión de corte para artículo y presentación.
- Regenerar tablas y figuras cuando cambien sus datos.
- Actualizar artículo antes que presentación.
- Mantener historial o copia cuando una corrección altera evidencia ya usada.
- No editar manualmente una salida derivada si existe una fuente reproducible.
- Registrar las zonas horarias al comparar archivos de entornos distintos.

## Criterio de salida

No pasar a redacción completa hasta poder responder:

- qué versión de cada resultado se usará;
- qué preguntas pueden responderse;
- qué afirmaciones requieren bibliografía;
- qué límites no pueden resolverse con los artefactos actuales;
- qué documentos o resultados son históricos y no comparables.
