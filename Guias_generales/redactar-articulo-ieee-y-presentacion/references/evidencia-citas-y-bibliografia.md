# Evidencia, citas y bibliografía

## Objetivo

Hacer visible qué ideas proceden de otras personas, qué resultados proceden del estudio y qué elecciones pertenecen al equipo. Una bibliografía extensa no sustituye esta trazabilidad.

## Qué debe citarse

Citar, en su primera explicación pertinente:

- teorías, definiciones, taxonomías y marcos conceptuales;
- afirmaciones sobre el dominio, prevalencia, efectos, riesgos o contexto;
- estado del arte y comparaciones con trabajos previos;
- algoritmos, arquitecturas, pérdidas, métricas, procedimientos y mejoras publicadas;
- conjuntos de datos, instrumentos, protocolos y escalas;
- modelos preentrenados y checkpoints exactos;
- estándares, normas, políticas y documentación cuyo comportamiento se describe;
- software central cuando su publicación o documentación es relevante;
- figuras, tablas, fórmulas o esquemas adaptados.

Una afirmación propia basada directamente en los resultados no necesita una fuente externa para ser «validada», pero sí una ruta al artefacto que la produce. Si un corpus, modelo, código u otro producto se presenta por primera vez en el mismo artículo, no se crea una entrada bibliográfica para que los autores se citen a sí mismos: se explica narrativamente y se remite a tablas, anexos, manifiestos o repositorios internos. Una publicación previa e independiente sí puede citarse cuando sea pertinente. Una decisión local puede citar antecedentes que la motivan, aunque debe seguir identificándose como decisión local.

## Fuente apropiada para cada uso

| Uso | Fuente preferida |
|---|---|
| Aporte de un algoritmo | Paper fundacional o publicación primaria |
| Estado actual de un método | Revisión sistemática reciente más estudios primarios relevantes |
| Modelo o checkpoint | Paper de la familia y tarjeta/repositorio oficial de la versión exacta |
| Conjunto de datos | Paper, ficha o repositorio oficial de la versión usada |
| API o comportamiento de software | Documentación oficial y versionada |
| Norma o política | Texto oficial vigente |
| Hecho institucional | Fuente primaria de la institución |
| Resultado propio | Archivo canónico, manifiesto, tabla o registro de ejecución |

Preferir DOI, editor, actas oficiales, repositorios institucionales o sitios oficiales. Usar una fuente secundaria cuando sintetiza un campo o cuando la primaria no es accesible, pero no atribuirle la autoría del aporte original.

## Estrategia de búsqueda

Para construir cadenas booleanas, expandir vocabulario y documentar ciclos completos, aplicar [busqueda-bibliografica-profunda.md](busqueda-bibliografica-profunda.md).

1. Convertir cada bloque del artículo en preguntas de evidencia.
2. Buscar primero revisiones, vocabularios del campo y trabajos seminales.
3. Seguir referencias hacia la fuente primaria.
4. Buscar trabajos recientes que prueben, cuestionen o extiendan el método.
5. Verificar título, autores, año, sede, DOI o URL en la página canónica.
6. Registrar la afirmación concreta que cada fuente puede respaldar.
7. Descartar la fuente si solo coincide por palabras clave.

Para afirmaciones sensibles o actuales, comprobar vigencia. Si no hay acceso a búsqueda, declarar la limitación y no crear metadatos plausibles.

## Ficha de lectura

Registrar por fuente:

| Campo | Contenido |
|---|---|
| Clave | Identificador bibliográfico |
| Tipo | Paper, estándar, dataset, software, política, libro |
| Pregunta | Qué necesidad del artículo cubre |
| Aporte verificable | Qué sostiene realmente |
| Alcance | Población, idioma, datos, condiciones |
| Límite | Qué no permite afirmar |
| Ubicación | Página, sección, figura o tabla |
| Uso previsto | Frase o apartado del manuscrito |
| Procedencia | DOI/URL/editor y copia legal disponible |

Leer el pasaje pertinente completo. No construir una afirmación a partir del fragmento de un buscador.

## Colocación de citas

- Situar la cita junto a la afirmación que respalda.
- Separar oraciones cuando una cita no cubre todas sus cláusulas.
- Evitar un racimo de referencias al final de un párrafo con funciones ambiguas.
- Citar de nuevo si la fuente reaparece lejos y la atribución podría perderse.
- Distinguir claramente una serie de antecedentes de la decisión adoptada por el trabajo.
- No citar una fuente general para una cifra, fórmula o propiedad que no contiene.

En estilo IEEE, usar numeración según la plantilla. El orden final de referencias suele seguir la primera aparición; no modificarlo manualmente si el gestor bibliográfico lo resuelve. Cuando se adopte la convención de abreviar obras con más de tres autores, mostrar los tres primeros seguidos de «et al.» en la lista final. Configurar el estilo o el gestor bibliográfico para conservar la autoría completa en el archivo fuente; no recortar manualmente el campo de autores.

## Algoritmos, modelos y mejoras

Para cada componente usado registrar:

| Componente | Fuente fundacional | Implementación | Versión/checkpoint | Licencia | Decisión local |
|---|---|---|---|---|---|

La cita a una familia de modelos no identifica el checkpoint. La documentación de una biblioteca no reemplaza el paper de un algoritmo, y el paper no reemplaza la versión exacta de la implementación.

Marcar como local:

- combinaciones de etiquetas;
- ponderaciones y umbrales;
- reglas de consenso;
- estrategia de muestreo;
- criterios de parada;
- arquitectura modificada;
- procedimiento de selección no tomado literalmente de una fuente.

## Paráfrasis e integridad

1. Leer y comprender la idea en contexto.
2. Registrar su función en el argumento.
3. Cerrar la fuente y explicar la idea con estructura y vocabulario propios.
4. Comparar con el original para eliminar una secuencia demasiado cercana.
5. Añadir la cita junto a la idea.
6. Usar comillas y página si una formulación textual breve es imprescindible.

No traducir literalmente un resumen, encadenar definiciones sin atribución ni usar una cita para justificar copia extensa. La similitud baja no demuestra originalidad; la atribución correcta tampoco permite reproducir grandes pasajes protegidos.

## Datos, código y licencias

Documentar por separado:

- procedencia y versión de datos;
- consentimiento, términos o base de uso;
- transformaciones y restricciones de redistribución;
- licencia del código;
- licencia de modelos, pesos o adaptadores;
- licencia de imágenes y otros recursos;
- ubicación y condiciones de acceso.

No deducir que un recurso público tiene licencia abierta. Si no puede redistribuirse, compartir en el repositorio o manifiesto técnico los metadatos, hashes, identificadores, estadísticas o el procedimiento legal de reconstrucción que sí estén permitidos.

En el texto publicado, la declaración de disponibilidad no necesita SHA ni commits concretos. Indicar de forma breve qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorio se localizan. Mantener las huellas, revisiones exactas y comprobaciones de integridad en manifiestos técnicos fuera del cuerpo.

## Auditoría de citas

### Cobertura

- ¿Toda definición externa tiene cita?
- ¿Cada afirmación teórica o del estado del arte tiene respaldo?
- ¿Cada algoritmo o modelo usado conserva su fuente fundacional?
- ¿Cada versión exacta de dataset, checkpoint, estándar o software tiene procedencia?
- ¿Las cifras propias apuntan a artefactos internos?
- ¿Las decisiones locales están identificadas como tales?

### Correspondencia

- ¿La fuente sostiene exactamente la proposición vecina?
- ¿Su población, idioma, fecha y alcance permiten ese uso?
- ¿Se distingue correlación, causalidad y explicación plausible?
- ¿Una política, noticia o documentación se presenta con su autoridad correcta?
- ¿Las citas múltiples cumplen funciones claras y no decorativas?

### Metadatos

- ¿Coinciden autores, título, año, sede, volumen, páginas y DOI/URL?
- ¿El DOI resuelve y corresponde al trabajo citado?
- ¿La clave usada existe en la bibliografía?
- ¿Toda entrada bibliográfica es citada y toda cita tiene entrada?
- ¿No hay duplicados con claves distintas?
- ¿Las URLs y fechas de consulta se incluyen cuando el estilo las exige?

### Procedencia intelectual

- ¿La paráfrasis usa redacción propia?
- ¿Las citas textuales son breves, necesarias y correctamente marcadas?
- ¿Las figuras adaptadas indican fuente y permiso/licencia?
- ¿Las definiciones combinadas mantienen atribución por componente?
- ¿Se preservan notas que permitan reconstruir la ruta afirmación → fuente?

### Resultado de auditoría

Clasificar cada problema:

| Severidad | Ejemplo | Acción |
|---|---|---|
| Crítica | Fuente inventada o afirmación central sin respaldo | Retirar o verificar antes de entregar |
| Alta | Fuente real que no sostiene la afirmación | Sustituir, atenuar o reescribir |
| Media | Metadato incompleto o cita ambigua | Corregir y volver a comprobar |
| Baja | Inconsistencia menor de formato | Normalizar en el cierre editorial |

El informe final debe declarar conteos de citas, entradas, claves ausentes, duplicados y fuentes pendientes; no basta con decir «bibliografía revisada».
