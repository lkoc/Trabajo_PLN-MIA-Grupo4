# Metodología, resultados y validez

## Elegir el diseño

La metodología depende de la pregunta y de la contribución:

| Contribución principal | Diseño posible |
|---|---|
| Construcción y evaluación de un artefacto | Design Science Research, ingeniería experimental o estudio de diseño |
| Comparación de métodos | Experimento controlado o benchmark |
| Estimación en una población | Estudio observacional con muestreo apropiado |
| Comprensión de experiencias | Diseño cualitativo |
| Síntesis de conocimiento | Revisión sistemática o mapeo |
| Intervención con usuarios | Experimento, estudio de campo o método mixto |

No elegir DSR solo porque existe software. DSR resulta apropiada cuando el conocimiento se produce al diseñar, demostrar y evaluar un artefacto frente a una necesidad. Citar el marco metodológico usado.

## Descripción mínima de cada etapa

Para cada procedimiento indicar:

| Elemento | Pregunta |
|---|---|
| Entrada | ¿Qué datos, requisitos o artefactos recibe? |
| Procedimiento | ¿Qué transformación se aplica y por qué? |
| Salida | ¿Qué producto verificable genera? |
| Control | ¿Cómo se reduce sesgo, fuga o error? |
| Evaluación | ¿Con qué criterio se juzga? |
| Aprendizaje | ¿Qué decisión habilita? |
| Objetivo | ¿Qué objetivo específico atiende? |

## DSR cuando corresponda

Documentar iteraciones, no solo el sistema final:

1. identificar el problema y los actores;
2. derivar objetivos o requisitos de la solución;
3. diseñar y construir el artefacto;
4. demostrarlo en un contexto pertinente;
5. evaluarlo con criterios explícitos;
6. comunicar resultados y aprendizaje.

Para cada iteración registrar problema observado, cambio, evidencia, resultado y motivo de la siguiente decisión. Diferenciar evaluación técnica, utilidad con usuarios y despliegue real.

## Datos y muestra

Explicar:

- fuente, periodo y criterios de inclusión/exclusión;
- unidad de observación y unidad de análisis;
- tamaño antes y después de filtros;
- deduplicación, limpieza y transformaciones;
- faltantes y descartes;
- muestreo y balance;
- particiones y agrupación;
- procedencia de etiquetas o mediciones;
- restricciones éticas y de licencia.

Una tabla o embudo debe permitir reconciliar los conteos. No presentar una muestra enriquecida o balanceada como prevalencia natural.

### Etiquetado o medición

Si existen anotaciones humanas o asistidas:

- definir categorías, inclusión, exclusión y casos ambiguos;
- identificar quién o qué produjo cada rótulo;
- indicar instrucciones visibles, entrenamiento y adjudicación;
- distinguir anotación ciega de revisión con sugerencias;
- reportar acuerdo solo si el diseño permite calcularlo;
- no llamar gold standard a un conjunto sin fundamento suficiente;
- versionar guía, esquema, prompt o instrumento.

Cuando una taxonomía combine literatura y reglas propias, separar definición académica, evidencia contextual, política institucional y decisión operacional.

## Métodos y alternativas

No listar algoritmos sin explicar su función experimental. Para cada familia indicar:

- hipótesis o necesidad que atiende;
- representación de entrada;
- arquitectura o procedimiento;
- configuración que afecta resultados;
- inicialización o transferencia;
- recursos de entrenamiento e inferencia;
- criterio de selección;
- fuente fundacional y versión implementada.

Incluir baselines simples. Una mejora solo se atribuye a un componente si el diseño permite aislarlo; de lo contrario, hablar de diferencia entre configuraciones.

## Protocolo contra fuga

Separar con claridad:

- entrenamiento: ajusta parámetros;
- validación: selecciona arquitectura, época, hiperparámetros y umbrales;
- test: estima una vez el desempeño congelado;
- producción o estudio prospectivo: observa comportamiento externo.

Agrupar la partición por entidad cuando varias observaciones comparten origen, sujeto, sesión, documento o tiempo. Prevenir que duplicados o transformaciones de la misma unidad crucen particiones.

Documentar toda exposición histórica al test. No recuperar cegamiento borrando una salida.

La misma separación se aplica cuando se declara un ganador distinto por salida o categoría. Seleccionar cada ganador mediante una métrica predefinida en validación, congelar modelo y umbral, y usar test solo para reportar el resultado final. Nombrar en el título o caption de la tabla la métrica y la partición que gobiernan la elección; nunca construir una tabla de ganadores escogiendo el mayor valor observado en test.

## Métricas y análisis

Definir cada métrica matemáticamente o mediante fuente pertinente y explicar por qué responde al objetivo. Indicar:

- dirección de mejora;
- nivel de agregación;
- promedio macro, micro o ponderado;
- umbral;
- unidad o denominador;
- tratamiento de clases ausentes;
- intervalo de incertidumbre;
- coste operacional asociado.

Cuando se comparan clasificadores, explicar en lenguaje simple qué mide cada indicador principal. Por ejemplo, la precisión responde cuántas alertas son correctas, el recobrado cuántos positivos se recuperan, F1 equilibra ambas cantidades en un umbral y la precisión promedio resume el ordenamiento a lo largo de umbrales. Citar una fuente metodológica y distinguir con claridad métricas de clasificación, ranking, calibración y operación.

## Descripción de las familias de modelos

Para cada familia utilizada, incluir solo lo necesario para comprender la elección y el resultado:

- representación de entrada y mecanismo de decisión;
- aporte fundacional y cita primaria;
- variante o checkpoint concreto;
- razón de selección en el estudio;
- ventaja esperada y limitación pertinente;
- forma de entrenamiento, calibración o adaptación que cambia la comparación.

Agrupar variantes que comparten mecanismo para evitar repetir teoría. Añadir un esquema de funcionamiento únicamente cuando ayude a comparar tres o más rutas o haga visible una diferencia estructural que la prosa no muestra con igual claridad.

Evitar inventarios internos en el cuerpo: rangos de capas, nombres de tensores, dimensiones intermedias, fórmulas auxiliares completas, conteos extensos de parámetros y barridos de configuración solo se incluyen si explican una diferencia experimental. Conservar el detalle íntegro en manifiestos, material suplementario o artefactos reproducibles y resumir en el artículo la decisión que afecta el resultado.

No intercambiar nombres parecidos. Una métrica histórica mal nombrada debe explicarse y conservarse solo para trazabilidad.

### Precisión de las cifras publicadas

- Presentar métricas, porcentajes, medias, tiempos medidos, tamaños de efecto, límites de intervalos y otras magnitudes estimadas con dos cifras significativas.
- Hacer cálculos, ordenamientos, selección y pruebas con la precisión completa; aplicar el redondeo únicamente a la salida destinada al lector.
- No redondear conteos exactos, tamaño muestral, años, versiones, identificadores, hashes ni parámetros fijados por protocolo. Conservar también la precisión requerida de las constantes físicas.
- Usar ceros finales cuando comuniquen precisión: `0,40` tiene dos cifras significativas, mientras `0,4` comunica una. Emplear notación científica cuando evite ambigüedad.
- Aplicar el mismo criterio en texto, tablas, figuras, resumen y presentación; una cifra repetida debe verse igual en todos los entregables.

Cuando sea pertinente, reportar:

- estimación puntual;
- intervalo de confianza;
- variación entre semillas;
- comparación pareada;
- tamaño del efecto;
- análisis de errores;
- carga humana, latencia, memoria o costo.

No afirmar significancia si el análisis no la demuestra.

## Calibración y decisión operacional

Si un score conduce a una acción:

1. fijar el objetivo operacional antes de consultar test;
2. calibrar o seleccionar umbrales con validación;
3. comparar opciones al mismo objetivo cuando corresponda;
4. reportar precisión, recall, falsos negativos, falsos positivos y tasa de derivación;
5. separar una sugerencia del modelo de la decisión humana;
6. congelar la política antes de evaluar.

Una alta métrica de ranking no define por sí sola el mejor punto de operación.

## Herramientas y hardware

Reportar solo lo que afecta reproducibilidad, costo o interpretación:

- sistema operativo o entorno;
- CPU, GPU y memoria relevantes;
- entorno local, nube o clúster;
- bibliotecas y versiones centrales;
- precisión numérica;
- tiempo o consumo si fue medido;
- diferencias de hardware entre familias.

No inferir el acelerador a partir del nombre de una plataforma. Si falta una bitácora, declararlo. Distinguir hardware de entrenamiento, evaluación e inferencia.

## Presentar resultados

Organizar cada bloque:

1. recordar la pregunta;
2. identificar el universo y protocolo;
3. mostrar tabla o figura;
4. describir la magnitud principal;
5. señalar incertidumbre o excepción;
6. cerrar con una respuesta descriptiva.

Comparar alternativas solo con igual dato, split, objetivo y métrica. Colocar resultados históricos en una tabla separada si explican el proceso, sin mezclarlos con la comparación principal.

Incluir resultados negativos o configuraciones que no superan la referencia si forman parte relevante del razonamiento. Denominar «preliminares» a las iteraciones tempranas del mismo estudio; reservar «históricos» para antecedentes externos o series temporales. Si esos resultados solo justifican una decisión de diseño —por ejemplo, fusionar salidas con soporte escaso—, resumir la evidencia mínima y la decisión en prosa, sin mantener una tabla extensa de protocolos no comparables.

Cuando una taxonomía cambia entre iteraciones, declarar: categorías antes y después, regla de mapeo, soporte que motiva el cambio, información que se conserva y aquello que la fusión no implica. Mantener términos distintos para categoría operativa, etiqueta almacenada y fenómeno de daño.

## Discusión

Separar:

- **observación:** qué muestra el resultado;
- **interpretación:** qué significa para la pregunta;
- **explicación plausible:** por qué pudo ocurrir;
- **comparación externa:** relación con estudios previos;
- **implicación:** qué decisión permite;
- **límite:** dónde deja de sostenerse.

Evitar convertir una comparación descriptiva en superioridad universal. No extrapolar a poblaciones, idiomas, equipos o condiciones no evaluadas.

## Amenazas de validez

Revisar:

| Dimensión | Pregunta |
|---|---|
| Constructo | ¿La variable o métrica representa lo que se afirma? |
| Interna | ¿Otra causa explica la diferencia? |
| Estadística | ¿Hay tamaño, incertidumbre y análisis suficientes? |
| Externa | ¿A qué población, contexto y periodo se generaliza? |
| Reproducibilidad | ¿Otra persona puede reconstruir entradas y decisiones? |
| Ética | ¿El procedimiento genera daño, sesgo o exposición? |

No usar «limitación» como lista ceremonial. Explicar dirección y posible magnitud del efecto cuando se conozcan, junto con el control aplicado.

## Conclusiones vinculadas

Preparar una tabla de cierre:

| Objetivo interno | Evidencia principal | Respuesta | Distancia eliminada/reducida/pendiente | Límite | Próximo paso |
|---|---|---|---|---|---|

Usar esta tabla para verificar que el objetivo general y cada objetivo específico quedaron respondidos. No copiar sus rótulos al cierre publicado: narrar las respuestas sin la palabra «objetivo», sin códigos `O1`/`O2` y sin frases de cumplimiento. Explicar para cada distancia entre la situación inicial y la deseada si se eliminó, se redujo o permanece, y asociar lo pendiente con una acción, recomendación o línea concreta de trabajo futuro.

Una conclusión no debe prometer más autonomía, validez o causalidad que la evaluación realizada. El aporte puede ser útil y publicable dentro de un alcance prudente.
