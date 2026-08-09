# Auditoría del etiquetado peruano: muestra estratificada del 10 % y búsquedas dirigidas de corpus completo

**Fecha de corte:** 9 de agosto de 2026  
**Supervisor de referencia:** `CODEX`  
**Modelo de auditoría:** `CODEX–Sol` (`GPT-5.6 Sol`), razonamiento `xhigh` o
“extra high”, velocidad estándar  
**Contrato:** `moderacion_peru_5_salidas_v2`, taxonomía `2.1.0`  
**Carácter de las decisiones:** referencial, trazable y human-in-the-loop; una
decisión humana posterior puede prevalecer sobre cualquier modelo.

Las cantidades de este documento se calcularon a partir de
`datos/etiquetado/consolidado/anotaciones_v2.jsonl`, la última decisión por
`chunk_id` de `datos/etiquetado/humano/labeling_events_v2.jsonl` y los
manifiestos de `datos/etiquetado/cascada_deepseek_v4/`. Son resultados del
proyecto, no afirmaciones tomadas de fuentes externas. La documentación del
dataset y del contexto de producción se explicita porque favorece la
reproducibilidad y el análisis de sesgos [1], [2].

## 1. Estadísticas descriptivas del dataset

### 1.1. Tamaño y depuración de alcance

| Magnitud | Antes del filtro | Dataset peruano elegible | Variación |
|---|---:|---:|---:|
| Chunks | 166.940 | 157.719 | −9.221 (−5,524 %) |
| Videos | 4.992 | 4.513 | −479 (−9,595 %) |
| Canales con al menos un chunk | 322 | 276 | −46 (−14,286 %) |

El filtro encontró 78 canales de origen extranjero con país identificable. No
se excluyó un canal solo por ser extranjero: se conservaron los videos cuyo
título o texto completo contenía evidencia temática explícita sobre el Perú.
De ese modo se retuvieron 70 videos extranjeros pertinentes, equivalentes a
3.890 chunks. Se excluyeron 9.221 chunks de 479 videos, pertenecientes a 72
canales afectados; 46 canales quedaron completamente fuera y los restantes
conservaron al menos un video sobre el Perú.

La exclusión es lógica y reversible: cada chunk recibió un evento `reject`, no
fue borrado físicamente. Los canales con más material retirado fueron Infobae
(818 chunks), Instituto Nacional de Formación Política de Morena (796), 24
Horas–TVN Chile (746), La Negra Candela Oficial (723) y W Radio Colombia (677).
La contaminación extranjera ajena al Perú queda, por tanto, resuelta en el
estado efectivo usado por esta auditoría.

### 1.2. Distribución por canal y video

En los 157.719 chunks elegibles:

| Unidad | Número | Media de chunks | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| Canal | 276 | 571,45 | 25,5 | 1 | 30.888 |
| Video | 4.513 | 34,95 | 19 | 1 | 767 |

Los diez canales con más chunks son:

| Canal | Chunks |
|---|---:|
| Hablando Huevadas | 30.888 |
| Arde Troya con Juliana Oxenford | 11.596 |
| Goblinciano | 6.525 |
| Nada Espacial | 6.224 |
| Sin Guion con Rosa María Palacios | 5.623 |
| Todo Good | 5.071 |
| Nunca MÁS | 4.716 |
| ATV Noticias | 4.504 |
| RPP Noticias | 4.348 |
| DÍA D | 4.129 |

Como resumen exploratorio —no como metadato editorial certificado— se aplicó
una heurística reproducible sobre el nombre del canal:

| Tipo heurístico | Canales | Chunks | Porcentaje de chunks |
|---|---:|---:|---:|
| Noticias/comentario | 63 | 58.823 | 37,30 % |
| Entretenimiento/comedia | 9 | 58.586 | 37,15 % |
| Otros/mixto | 204 | 40.310 | 25,56 % |

Esta concentración importa: un mismo error sistemático en pocos canales de
gran tamaño puede mover más el dataset que errores dispersos en muchos canales.

### 1.3. Estado final de las etiquetas

`SEGURO` es excluyente y las cuatro categorías de daño son multietiqueta; por
eso la suma de asignaciones de daño puede superar el número de chunks dañinos.

| Estado o etiqueta | Chunks/asignaciones | % de los 157.719 chunks |
|---|---:|---:|
| `SEGURO` | 105.751 | 67,050 % |
| Sin decisión final / contexto pendiente | 40.902 | 25,933 % |
| Al menos una categoría de daño | 11.066 | 7,016 % |
| `ACOSO_AMENAZA` | 6.337 | 4,018 % |
| `CONTENIDO_SEXUAL` | 3.093 | 1,961 % |
| `RACISMO_DISCRIMINACION` | 2.151 | 1,364 % |
| `ATAQUE_POR_GENERO_IDENTIDAD` | 2.027 | 1,285 % |

Las categorías menos representadas son ataque por género/identidad y
racismo/discriminación. Sobre las cuatro cuentas de daño, la razón
máximo/mínimo es **3,126**, el coeficiente de variación es **0,513** y la
entropía de Shannon normalizada es **0,915**. La entropía relativamente alta
indica que las cuatro categorías están presentes; la razón y el coeficiente de
variación muestran, sin embargo, una concentración clara en acoso/amenaza. La
taxonomía distingue blanco, modalidad y alcance porque “lenguaje abusivo” no es
una clase semántica única [3], [4].

## 2. Diseño metodológico

### 2.1. Jerarquía de revisión

La auditoría combinó una fase probabilística general y varias fases dirigidas.
La separación es crucial: la muestra estima una tasa de desacuerdo sin buscar
palabras específicas; la búsqueda dirigida inspecciona todo el universo
elegible para errores raros o sistemáticos. La literatura muestra que el
contexto conversacional puede cambiar la clasificación de lenguaje abusivo
[5], por lo que cada decisión consideró hablante, blanco, atribución, sentido
local y postura, no una coincidencia léxica.

```mermaid
flowchart TD
    A[166.940 chunks originales] --> B{Filtro de alcance por canal, video y tema Perú}
    B -->|9.221 ajenos al Perú| X[Eventos reject reversibles]
    B -->|157.719 elegibles| C[Estado efectivo: última decisión por chunk]
    C --> D[Muestra SHA-256 de 16.694 chunks en 35 estratos]
    D --> E[Lectura semántica CODEX–Sol xhigh]
    H[Interacción humana: criterios, excepciones y prioridades] --> E
    E --> F[62 correcciones de alta confianza]
    C --> G[Búsqueda dirigida sobre los 157.719 chunks]
    H --> G
    G --> I[Sexualidad, identidad, clase, región, amenazas, citas y usos amistosos]
    I --> J[469 chunks únicos adicionales corregidos]
    F --> K[531 correcciones semánticas únicas]
    J --> K
    K --> L[Eventos append-only con revisor CODEX]
    L --> M[Métricas, prompt operativo v3 e informe]
```

### 2.2. Intervención humana en el ciclo

La revisión no fue una inferencia autónoma aislada. Hubo interacción humana en
cuatro puntos:

1. se estableció que una decisión de `CODEX` era referencial y que, si no había
   evidencia suficiente para cambiarla, prevalecía la decisión Pro;
2. se incorporaron criterios propuestos durante la auditoría: significado
   sexual peruano de *cachar*, ambigüedad de *tirar/coger/chupar*,
   condescendencia, diminutivos, clasismo educativo, citas informativas y
   exclusión temática de canales extranjeros;
3. se pidió ampliar el análisis desde una muestra del 10 % hacia búsquedas
   dirigidas de corpus completo;
4. una persona conserva la capacidad de intervenir y prevalecer en cualquier
   decisión posterior.

`CODEX–Sol` aplicó esos criterios como supervisor de referencia y registró solo
cambios de confianza alta. El proceso sigue el principio práctico de diferir
casos inciertos a un experto, sin confundir deferencia con una etiqueta de
daño [15].

### 2.3. Muestra estratificada del 10 %

Se fijó un tamaño exacto de **16.694 chunks**, igual al 10 % del dataset
original redondeado y al 10,585 % del universo peruano elegible. La selección
se congeló antes de corregir etiquetas:

- semilla textual: `CODEX-AUDIT-20260809-CLEAN`;
- clave pseudoaleatoria: SHA-256 de `semilla|chunk_id`;
- 35 estratos observados, definidos por la combinación de etiqueta gruesa
  efectiva, modelo fuente y estado de revisión;
- asignación proporcional, mínimo de un caso por estrato y distribución de los
  residuos por restos mayores;
- sin reemplazo y sin chunks extranjeros fuera de alcance.

La muestra contenía 11.186 `SEGURO`, 4.353 casos sin decisión final y 1.155
chunks con al menos un daño; 12.341 estaban resueltos y eran comparables de
forma directa con la decisión de referencia.

### 2.4. Búsqueda dirigida de corpus completo

Después de la muestra se recorrieron **los 157.719 chunks elegibles completos**.
El procedimiento tuvo dos niveles:

1. recuperación amplia por familias léxicas, flexiones, errores de transcripción
   y patrones de atribución;
2. criba semántica por ventana y lectura del chunk: blanco, relación entre
   hablantes, literalidad, atribución, postura y plausibilidad del daño.

Las familias incluyeron:

- sexualidad: *cachar, tirar, coger, chupar/cupar, manosear, chapar, agarrar,
  comer/comerse, polvo, leche, venirse, acabar, meter, calato, arrecho, concha,
  poto, pinga/pichula, huevos, culear/culiar, encamar, fornicar, pajear, mamar,
  felación, penetrar, violar, mamacita* y *chibola/chibolo*;
- racialización, clase y región: *terruco, motoso, amixer, pituco, chusma,
  huachafo, conero, provinciano, charapa, paisano, marrón, color puerta, llama,
  auquénido, veneco, chamo, gringo/gringa, comemote, llorcho, characato,
  cholo/cholito, chino/chinito, serrano, indio, chuncho* y *puneño*;
- género e identidad: *cabro, maricón, rosquete, traba, mostacero, marimacha,
  machona, marica, loca, machito, mandilón, pisado, sacolargo, feminazi, hembra,
  histérica, zorra, perra, puta* y *mantenida*;
- amenazas: *enfriar, dar piso, meter plomo, reventar, cuadrar, cogotear,
  bajar, hacer la vuelta, sacar la mierda, plomear, chifar, ajustar, marcar,
  pepear, sembrar, levantar, desaparecer, romperte, sacar el ancho* y “sabemos
  dónde vives”;
- usos amistosos o figurativos: *causa, pata, brother, huevón, gordo, negro,
  chino, cholo, matarse de risa, romperla, está criminal, tirar la toalla* y
  *chuparse el dedo*;
- jerarquías educativas: universidad privada, PUCP, analfabetismo, lectura y
  escritura, y colegio estatal.

Los estudios peruanos justifican que *motoso* y *terruco* pueden operar dentro
de procesos de racialización lingüística y política [6]; *amixer* también se ha
estudiado como identidad y marcación social en espacios digitales peruanos
[7]. La asociación entre “falta de educación”, superioridad y discurso racista
tiene antecedentes específicos en el Perú [8], dentro de dinámicas sociales
que no se reducen al insulto racial explícito [9]. Estas fuentes orientan el
contexto; la etiqueta concreta de cada chunk se decidió con el contrato local.

La palabra *cachar* recibió atención especial: se revisaron los 347 chunks
elegibles que contenían flexiones relevantes y aún no tenían
`CONTENIDO_SEXUAL`; 230 mostraron sentido sexual inequívoco y fueron corregidos.
En cambio, *llama*, *meter el dedo*, *venirse encima*, *hacer la vuelta* y otros
patrones generaron muchos homónimos, verbos literales o modismos seguros. Esto
confirma que el léxico sirve para recuperar candidatos, no para etiquetar.

Las amenazas, el doxeo y la difusión íntima no consentida tienen relevancia
documentada en la violencia de género en línea en el Perú [10], pero una noticia
que las denuncia no se convirtió automáticamente en daño del narrador.

La síntesis operacional de las expresiones adicionales es:

| Familia | Riesgo de daño | Lectura segura que debe comprobarse |
|---|---|---|
| *terruco, motoso, comemote, llorcho* | criminalización o inferiorización andina/lingüística | explicación, cita condenatoria o referencia histórica |
| *pituco, conero, amixer, chusma, huachafo* | clasismo racializado | descripción social, autorreferencia o broma afiliativa |
| *gringo, chamo, veneco, charapa, puneño* | degradación por nacionalidad o región | gentilicio/vocativo neutral; *chamo* suele ser afiliativo |
| *cabro, marica, rosquete, loca, machito, pisado* | homofobia o vigilancia de expresión de género | animal, estado mental descrito, situación “loca” o confianza recíproca |
| *perra, zorra, puta, mantenida, mamita* | misoginia, humillación o cosificación | animal literal, interjección, parentesco o vocativo afectuoso |
| *cachar, tirar, coger, chupar, culear* | actividad sexual explícita | lanzar, tomar, beber o sentido no sexual |
| *mamacita, chibola/chibolo* | cosificación o sexualización por edad | parentesco, juventud o vocativo sin contenido sexual |
| *desaparecer, plomear, dar piso, sacar el ancho* | amenaza plausible | noticia atribuida, consecuencia figurativa o esfuerzo económico |
| *caviar, rojo, zurdo* | ataque político; puede combinarse con terruqueo | posición política no protegida por sí sola |

## 3. Resultados de calidad

### 3.1. Auditoría muestral congelada

En los 12.341 chunks resueltos de la muestra hubo 62 desacuerdos de alta
confianza:

- 61 daños se corrigieron a `SEGURO` porque eran menciones, citas o denuncias
  claramente atribuidas sin respaldo del narrador;
- 1 caso cambió de ataque por género/identidad a acoso por apariencia, porque
  había humillación personal pero no motivación de género.

| Métrica | Resultado |
|---|---:|
| Acuerdo exacto modelo–`CODEX` en resueltos | 99,498 % |
| Desacuerdo exacto en resueltos | 0,502 % |
| Cambio sobre toda la muestra | 0,371 % |
| Error observado dentro de los 1.155 chunks de daño | 5,368 % |
| Falsos positivos de daño dentro de la muestra de daño | 5,281 % |
| Precisión observada de daño, aproximación | 94,632 % |
| IC Wilson 95 % del error en daño | [4,210 %, 6,822 %] |
| IC complementario de precisión observada | [93,178 %, 95,790 %] |

Se usó el intervalo de Wilson por su comportamiento para proporciones
binomiales [11]. Esta “precisión observada” es una estimación de alta precisión
sobre los daños presentes en la muestra; no mide exhaustivamente falsos
negativos entre los 4.353 casos sin decisión ni sustituye una matriz de verdad
de referencia independiente. El acuerdo tampoco equivale por sí solo a validez
semántica [12].

En 7.083 chunks con decisión tanto de Flash como de Pro, el acuerdo exacto entre
modelos fue 56,346 % y el acuerdo binario daño/no-daño 83,778 %. Es una medida
de consistencia entre preanotadores, no de exactitud.

### 3.2. Rendimiento de las búsquedas dirigidas

Las colas se superponen y por eso no deben sumarse como universos
independientes.

| Cola dirigida | Candidatos adjudicados | Cambios | Rendimiento de cambio |
|---|---:|---:|---:|
| *cachar* sin etiqueta sexual | 347 | 230 | 66,28 % |
| *coger* sin etiqueta sexual | 283 | 20 | 7,07 % |
| *chupar/cupar*, contexto sexual fuerte | 52 | 37 | 71,15 % |
| *tirar*, contexto sexual fuerte | 48 | 21 | 43,75 % |
| Cita/denuncia y condena, cola de alta precisión | 55 | 46 | 83,64 % |
| Amenazas locales fuertes sin `ACOSO_AMENAZA` | 41 | 25 reclasificados a `SEGURO` | 60,98 % |
| Patrones sexuales ambiguos fuertes | 20 | 7 decisiones correctivas | 35,00 % |
| Léxico adicional: 4 colas semánticas | 223 evaluaciones de familia | 11 | 4,93 % |

En las amenazas, “cambio” significa principalmente resolver como `SEGURO` una
cita periodística, amenaza a un animal, broma no plausible o giro figurativo;
no significa ignorar la gravedad del hecho reportado. La distinción es entre
el contenido referido y la postura del narrador.

Las cinco fases de esta auditoría dejaron:

| Fase | Eventos | Chunks únicos |
|---|---:|---:|
| Exclusión de alcance extranjero | 9.221 | 9.221 |
| Muestra estratificada | 62 | 62 |
| Búsqueda dirigida *cachar* | 230 | 230 |
| Léxico peruano y educación, v2 | 121 | 121 |
| Léxico ampliado, amenazas y citas, v3 | 108 | 108 |
| Léxico adicional, v4 | 11 | 11 |
| **Correcciones semánticas únicas** | **532 eventos** | **531 chunks** |

Un chunk de *cachar* recibió después una ampliación por género y explica la
diferencia entre eventos y chunks únicos.

### 3.3. Cuánto cambió el dataset

Tomando como línea base el estado posterior a la cola prioritaria Flash/Pro y
anterior a esta auditoría:

- **531 de 157.719 chunks elegibles** cambiaron semánticamente: **0,337 %**;
- **9.221 de 166.940 chunks** cambiaron de alcance: **5,524 %**;
- alcance y semántica juntos afectaron **9.752 chunks**, **5,842 %** del dataset
  original, sin solapamiento entre los excluidos y los corregidos;
- la cobertura resuelta pasó de 116.586 a 116.817 chunks, de **73,920 %** a
  **74,067 %**, un aumento de **0,146 puntos porcentuales**.

Dentro de los 531 cambios semánticos únicos:

- 127 chunks con daño pasaron a `SEGURO`;
- 55 chunks sin decisión pasaron a `SEGURO`;
- se añadieron 315 asignaciones sexuales que antes faltaban;
- el resto corrigió o completó racismo, género/identidad y acoso.

El saldo de asignaciones antes→después fue:

| Etiqueta | Antes | Después | Cambio neto |
|---|---:|---:|---:|
| `SEGURO` | 105.682 | 105.751 | +69 |
| `ACOSO_AMENAZA` | 6.400 | 6.337 | −63 |
| `CONTENIDO_SEXUAL` | 2.791 | 3.093 | +302 |
| `RACISMO_DISCRIMINACION` | 2.189 | 2.151 | −38 |
| `ATAQUE_POR_GENERO_IDENTIDAD` | 2.039 | 2.027 | −12 |

El cambio sexual alto no contradice la tasa pequeña de error muestral: fue el
resultado de una búsqueda deliberadamente enriquecida por términos locales,
especialmente *cachar*. Del mismo modo, la reducción de racismo, género y acoso
proviene sobre todo del sesgo sistemático de atribuir al narrador una cita que
este reportaba o condenaba. La detección de datos mal etiquetados y la revisión
enriquecida de candidatos son estrategias conocidas para mejorar conjuntos de
entrenamiento [13], [14].

### 3.4. Evaluación cualitativa final

La calidad global encontrada es **alta en consistencia gruesa, pero desigual
por fenómeno lingüístico**. El 99,498 % de acuerdo en los casos resueltos de la
muestra respalda una buena estabilidad general. No obstante, dos errores
sistemáticos son materialmente importantes:

1. **sobreetiquetado de uso/mención:** noticias, denuncias y explicaciones se
   marcaron como si el narrador respaldara el daño;
2. **subetiquetado de sexualidad peruana:** *cachar* y, en menor medida,
   *tirar/coger/chupar*, se interpretaron literalmente o quedaron sin resolver
   aun cuando el contexto era sexual.

También aparecieron errores menos frecuentes al separar clasismo racializado
de acoso personal, homofobia coloquial de confianza entre amistades, y amenaza
plausible de hipérbole o lenguaje figurado. Los diminutivos (*cholito,
chinito, mamita*) no fueron considerados dañinos por defecto: el daño dependió
de infantilización, desigualdad de relación y efecto degradante.

La calidad final es mejor que la línea base en los sesgos específicamente
auditados, pero no se presenta una “exactitud final” independiente porque las
correcciones y su evaluación fueron realizadas por el mismo supervisor de
referencia. La métrica más honesta es el conjunto de indicadores anteriores:
tasa de desacuerdo congelada, intervalo, rendimiento de colas dirigidas,
cambio neto y cobertura.

### 3.5. ¿Hace falta revisar más del 10 %?

**Sí, pero de manera dirigida, no mediante otra muestra uniforme ciega de todo
el corpus.** Los rendimientos de 66,28 % para *cachar*, 71,15 % para
*chupar/cupar* y 83,64 % en la cola estricta de citas muestran errores
sistemáticos concentrados. Ya se recorrió el 100 % de los 157.719 chunks con
los patrones dirigidos definidos en esta auditoría. Para una siguiente ronda se
recomienda:

1. ejecutar el prompt operativo v3 sobre los candidatos residuales de
   sexualidad polisémica y uso/mención;
2. auditar de forma dirigida todos los casos nuevos con baja confianza, choque
   Flash/Pro o palabras locales de alta ambigüedad;
3. extraer después un **holdout adicional del 2–3 %**, estratificado por canal y
   categoría, que no haya participado en las reglas actuales, para estimar la
   mejora sin circularidad;
4. priorizar los canales grandes de comedia y noticias, pues concentran los dos
   sesgos opuestos. Las palabras *chibolo, loca, gringo* y *desaparecer* deben
   continuar como disparadores contextuales, no como reglas automáticas.

Esta recomendación es consistente con aprendizaje activo: invertir revisión
en ejemplos informativos o inciertos suele ser más eficiente que aumentar una
muestra aleatoria sin criterio [14].

## 4. Trazabilidad y decisiones previas

Antes de la auditoría muestral se procesó una cola prioritaria de 2.705 chunks
que Pro no pudo resolver con suficiente confianza o que mantenían etiquetas
problemáticas. `CODEX` registró 2.688 decisiones y 17 ya tenían una decisión
previa. Esta cola no se mezcló con la estimación congelada del 10 %: el estado
posterior a ella fue la línea base de la auditoría descrita aquí.

Para todo evento nuevo se usó:

- revisor visible `CODEX`, seudonimizado por el servidor como
  `reviewer-879d60dc246ead2d`;
- acción `modify` o `reject`;
- etiquetas propuestas y finales;
- modelo fuente, evento fuente cuando existía, nota metodológica y fecha;
- persistencia append-only y precedencia de la última decisión por `chunk_id`.

Si `CODEX` no cambió un caso, prevaleció la decisión Pro o la última decisión
efectiva existente. No se generaron eventos de aceptación redundantes para cada
coincidencia correcta.

## 5. Reporte técnico de modelos, tiempo, hardware y costo

### 5.1. Modelos y ejecución de preetiquetado

DeepSeek Flash realizó la primera pasada remota y DeepSeek Pro la revisión
dirigida. Ambos operaron con salida JSON, cinco registros por solicitud,
concurrencia de hasta 32 solicitudes y *thinking* desactivado. La identidad de
la familia V4 procede de la documentación del proveedor [16] y los costos se
reconstruyen con sus precios documentados [17]. Los manifiestos locales son la
fuente de las cifras de consumo real o proyectado.

| Componente | Volumen activo | Tiempo neto | Costo USD |
|---|---:|---:|---:|
| Calibración Flash | 1.000 | 97,25 s | 0,0733 |
| Calibración Pro | 1.000 | 122,59 s | 0,2692 |
| Flash, primera pasada nueva | 114.696 aprox. | 132 min aprox. | 7,24 aprox. |
| Pro, segmento previo | 14.079 | 31,1 min | 4,7275 |
| Pro, continuación final | 40.704, 1 error | 79,0 min | 13,4555 |
| **Total de procesamiento remoto activo** | — | **245,8 min ≈ 4,10 h** | **25,77 aprox.** |

Se recuperaron además 52.244 anotaciones Flash y 9.912 Pro mediante
coincidencia exacta y única de `video_id` más texto normalizado. Esa reutilización
no tuvo costo API activo y su tiempo histórico no está disponible. El
checkpoint final de Flash sobrescribió contadores acumulativos; por eso su
tiempo y costo se reportan como proyección reconstruida y no como telemetría
exacta.

`Qwen/Qwen3-1.7B` estaba disponible como fallback local, con revisión fijada en
la tarjeta del modelo [18], pero no se mezcló con las métricas de la cascada
remota final que aquí se evalúa.

### 5.2. Hardware

El equipo local fue un GMKtec NucBox K8 Plus con AMD Ryzen 7 8845HS, 8 núcleos
y 16 hilos, 28,8 GB de RAM disponible y Radeon 780M integrada reportada con 3 GB
de memoria gráfica. Este equipo realizó preparación, enrutamiento, validación,
servidor, búsquedas léxicas y persistencia. El etiquetado DeepSeek se ejecutó en
infraestructura remota del proveedor; el tipo exacto de acelerador remoto no se
expone en los artefactos del proyecto.

### 5.3. Costo y tiempo de `CODEX`

La auditoría semántica, adjudicación de colas y elaboración del reporte utilizó
`CODEX–Sol` / `GPT-5.6 Sol`, razonamiento `xhigh`, velocidad estándar. No se usó
`Terra` ni `Luna` para las decisiones reportadas. La documentación oficial
describe Sol como el modelo de frontera de la familia [19]; el cálculo
equivalente usa el precio oficial de USD 5 por millón de tokens de entrada,
USD 0,50 por millón de entrada en caché y USD 30 por millón de salida [20].

La interfaz no expone telemetría exacta de tokens, costo real de suscripción ni
porcentaje semanal consumido. Por ello se registra un intervalo auditable, no
una falsa precisión:

| Recurso `CODEX` | Estimación |
|---|---:|
| Tiempo activo dedicado al etiquetado/auditoría | 2–3 h |
| Costo equivalente API, escenario | USD 2–6 |
| Consumo semanal orientativo comunicado | 3–7 %, no telemetría |

El intervalo excluye la construcción previa de las interfaces y cuenta la
lectura de candidatos, decisiones, criterios, cálculo de métricas y redacción
metodológica de esta auditoría.

## 6. Conclusiones y recomendaciones técnicas

1. El dataset efectivo queda en 157.719 chunks peruanos o temáticamente
   vinculados al Perú; la reducción por alcance es 5,524 %.
2. La calidad muestral es alta: 99,498 % de acuerdo exacto en los casos
   resueltos, con un error observado de 5,368 % dentro de los daños de la
   muestra. La principal debilidad fue la precisión contextual, no la
   estabilidad del formato.
3. La búsqueda de corpus completo confirmó dos sesgos sistemáticos: citas y
   denuncias sobreetiquetadas, y sexualidad peruana subetiquetada. Las 531
   correcciones semánticas únicas afectaron 0,337 % del corpus elegible.
4. Para futuros flujos semiautomáticos conviene usar el prompt v3, separar
   recuperación léxica de adjudicación semántica, almacenar hablante/postura y
   mantener colas específicas por sesgo.
5. El próximo control debe ser dirigido y seguido de un holdout pequeño no
   usado para construir reglas. No se recomienda repetir sin más otra muestra
   uniforme del 10 %.
6. En un modelo fine-tuned, los ejemplos fronterizos deben aparecer como pares
   contrastivos: misma palabra en uso sexual/no sexual, amistoso/degradante y
   cita/respaldo. La revisión por incertidumbre y por desacuerdo entre modelos
   debe mantenerse como ruta de operación.

El prompt mejorado quedó en
[`config/prompt_operacional_ollama_v3.md`](../config/prompt_operacional_ollama_v3.md),
sin alterar
[`config/prompt_operacional_ollama_v2.md`](../config/prompt_operacional_ollama_v2.md).

## Anexo A. Prompt de auditoría de etiquetado

El siguiente prompt reconstruye las instrucciones efectivamente aplicadas. Su
objetivo es reproducir la auditoría, no sustituir el prompt operativo de
producción. Los valores de fecha, semilla y tamaño congelan esta ejecución; para
una réplica nueva deben cambiarse y documentarse.

```text
PROMPT DE AUDITORÍA DE ETIQUETADO — CODEX v1.0

ROL
Actúa como supervisor de referencia CODEX para una auditoría human-in-the-loop
de subtítulos de YouTube relacionados con el Perú. Tu decisión es referencial:
puede ser revisada y reemplazada por una persona. No cambies una decisión
Flash/Pro salvo que tengas alta confianza en un error o una inconsistencia. Si
no cambias el caso, prevalece la última decisión efectiva, en particular Pro.

AUTORIDAD Y ENTRADAS
1. Autoridad normativa: config/taxonomia_v2.json, contrato
   moderacion_peru_5_salidas_v2, versión 2.1.0.
2. Prompt vigente en la corrida auditada:
   config/prompt_operacional_ollama_v2.md.
3. Base: datos/etiquetado/consolidado/anotaciones_v2.jsonl.
4. Eventos: datos/etiquetado/humano/labeling_events_v2.jsonl.
5. Para cada chunk usa la última decisión: accept/modify reemplaza etiquetas;
   reject lo saca del universo; defer no inventa etiqueta.
6. Revisor para nuevos eventos: CODEX. Conserva proposed_labels, final_labels,
   modelo y evento fuente, nota y fecha. La persistencia es append-only.

ALCANCE
1. Identifica canales extranjeros mediante metadatos de canal y video.
2. No excluyas por nacionalidad solamente. Retén un video extranjero si título
   o texto completo habla explícitamente del Perú, una entidad peruana, una
   región peruana o un acontecimiento peruano.
3. Excluye mediante reject los videos extranjeros sin vínculo temático con el
   Perú. Cuantifica chunks, videos y canales retirados y retenidos.
4. No uses los chunks rechazados en la muestra ni en el análisis semántico.

FASE A — MUESTRA GENERAL CONGELADA
1. Universo elegible esperado en esta ejecución: 157.719 chunks.
2. Tamaño exacto: 16.694, igual al 10 % del dataset original de 166.940.
3. Semilla: CODEX-AUDIT-20260809-CLEAN.
4. Clave de selección: SHA-256("semilla|chunk_id").
5. Estratifica por combinación de etiqueta gruesa efectiva, modelo fuente y
   estado resuelto/necesita revisión. Usa asignación proporcional, mínimo uno
   por estrato y restos mayores. Selecciona por hash ascendente sin reemplazo.
6. Congela la muestra y sus etiquetas antes de registrar correcciones.
7. Lee cada caso con esta jerarquía:
   a) evaluabilidad y referente;
   b) hablante y blanco;
   c) uso, mención, cita o denuncia;
   d) postura del narrador;
   e) sentido peruano y relación entre hablantes;
   f) daño sustentado y multietiqueta.
8. No fuerces SEGURO si falta contexto. No heredes la etiqueta del vecino.

FASE B — BÚSQUEDA DIRIGIDA DE CORPUS COMPLETO
Recorre los 157.719 chunks elegibles, no solo la muestra. La palabra recupera
candidatos, pero nunca decide una etiqueta. Haz criba semántica de:

- sexual: cachar, tirar, coger, chupar/cupar, manosear, chapar, agarrar,
  comer/comerse, polvo, leche, venirse, acabar, meter, calato, arrecho, concha,
  poto, pinga/pichula, huevos, culear/culiar, encamar, fornicar, pajear, mamar,
  felación, penetrar, violar, mamacita, chibola/chibolo;
- raza/clase/región: terruco, motoso, amixer, pituco, chusma, huachafo, conero,
  provinciano, charapa, paisano, marrón, color puerta, llama, auquénido, veneco,
  chamo, gringo/gringa, comemote, llorcho, characato, cholo/cholito,
  chino/chinito, serrano, indio, chuncho, puneño;
- género/identidad: cabro, maricón, rosquete, traba, mostacero, marimacha,
  machona, marica, loca, machito, mandilón, pisado, sacolargo, feminazi, hembra,
  histérica, zorra, perra, puta, mantenida;
- amenaza: enfriar, dar piso, meter plomo, reventar, cuadrar, cogotear, bajar,
  hacer la vuelta, sacar la mierda, plomear, chifar, ajustar, marcar, pepear,
  sembrar, levantar, desaparecer, romperte, sacar el ancho, sabemos dónde vives;
- amistoso/figurado: causa, pata, brother, huevón, gordo, negro, chino, cholo,
  matarse de risa, romperla, está criminal, tirar la toalla, chuparse el dedo;
- educación/clase: universidad privada, PUCP, colegio estatal, analfabeto, no
  saber leer o escribir, y cualquier uso de superioridad educativa.

Para cada coincidencia determina: sentido literal/local, blanco, vínculo entre
hablantes, atribución, postura, consentimiento cuando corresponda y daño
plausible. Cachar es sexual en el Perú cuando el contexto semántico habla de
personas y actividad sexual. Un diminutivo puede ser afectuoso o
condescendiente. Una universidad o nivel de alfabetización no implica daño sin
superioridad, exclusión o racialización. Las citas informativas y condenatorias
no heredan el daño; una descripción sexual gráfica puede ser la excepción.

UMBRAL DE CAMBIO
1. Modifica solo si la evidencia contextual es clara y el cambio puede
   justificarse con blanco, atribución y criterio del contrato.
2. Si dos lecturas razonables permanecen abiertas, conserva Pro o difiere.
3. Humor no borra daño, pero amistad cercana y reciprocidad pueden convertir un
   vocativo en seguro. No supongas amistad sin evidencia.
4. Registra todos los daños concurrentes y respeta que SEGURO es excluyente.

MÉTRICAS OBLIGATORIAS
1. Estadísticas del dataset, canales, videos y chunks por canal.
2. Conteos finales por categoría, categoría menos representada, razón
   máximo/mínimo, coeficiente de variación y entropía normalizada.
3. En la muestra congelada: acuerdo exacto, desacuerdo, cambio total, error en
   chunks de daño e intervalo Wilson 95 %.
4. En la búsqueda dirigida: candidatos, cambios y rendimiento por familia.
5. Cambio semántico único sobre elegibles; exclusión de alcance sobre original;
   cobertura resuelta antes y después.
6. Separa acuerdo Flash/Pro de exactitud frente a CODEX.
7. No declares exactitud final independiente usando los mismos casos con los
   que construiste las reglas.

REPORTE
Genera un Markdown en docs. Empieza con estadísticas descriptivas; incluye un
diagrama Mermaid, metodología human-in-the-loop, resultados cuantitativos y
cualitativos, jerga peruana, sesgos, modelos, hardware, tiempo y costo. Cita en
IEEE toda idea externa y agrega referencias. Distingue resultados locales de
afirmaciones externas. Incluye este prompt como anexo para reproducibilidad.
Concluye si corresponde ampliar la auditoría aleatoria o la revisión dirigida.
```

## Referencias

[1] E. M. Bender and B. Friedman, “Data Statements for Natural Language
Processing: Toward Mitigating System Bias and Enabling Better Science,”
*Transactions of the Association for Computational Linguistics*, vol. 6, pp.
587–604, 2018, doi: [10.1162/tacl_a_00041](https://doi.org/10.1162/tacl_a_00041).

[2] B. Vidgen and L. Derczynski, “Directions in Abusive Language Training Data,
a Systematic Review: Garbage In, Garbage Out,” *PLOS ONE*, vol. 15, no. 12,
e0243300, 2020, doi:
[10.1371/journal.pone.0243300](https://doi.org/10.1371/journal.pone.0243300).

[3] Z. Waseem, T. Davidson, D. Warmsley, and I. Weber, “Understanding Abuse: A
Typology of Abusive Language Detection Subtasks,” in *Proc. First Workshop on
Abusive Language Online*, 2017, pp. 78–84, doi:
[10.18653/v1/W17-3012](https://doi.org/10.18653/v1/W17-3012).

[4] M. Banko, B. MacKeen, and L. Ray, “A Unified Taxonomy of Harmful Content,”
in *Proc. Fourth Workshop on Online Abuse and Harms*, 2020, pp. 125–137, doi:
[10.18653/v1/2020.alw-1.16](https://doi.org/10.18653/v1/2020.alw-1.16).

[5] T. Bourgeade, Z. Li, F. Benamara, V. Moriceau, J. Su, and A. Sun, “Humans
Need Context, What about Machines? Investigating Conversational Context in
Abusive Language Detection,” in *Proc. LREC-COLING*, 2024, pp. 8438–8452.
[ACL Anthology](https://aclanthology.org/2024.lrec-main.740/).

[6] V. Zavala and C. Almeida, “«Motoso y terruco»: ideologías lingüísticas y
racialización en la política peruana,” *Lexis*, vol. 46, no. 2, pp. 481–521,
2022, doi:
[10.18800/lexis.202202.002](https://doi.org/10.18800/lexis.202202.002).

[7] V. Salem, “Amixer está en Facebook: una investigación de la choledad
virtual,” *Revista Chilena de Antropología Visual*, no. 27, pp. 69–88, 2016.
[Texto](https://www.antropologiavisual.cl/sites/default/files/2016_27_art04_salem.pdf).

[8] V. Zavala and R. Zariquiey, “«Yo te segrego a ti porque tu falta de
educación me ofende»: una aproximación al discurso racista en el Perú
contemporáneo,” in *Racismo y discurso en América Latina*, T. A. van Dijk, Ed.
Barcelona, España: Gedisa, 2007, pp. 333–370.

[9] J. C. Callirgos, *El racismo: la cuestión del otro (y de uno)*. Lima, Perú:
DESCO, 1993.

[10] Defensoría del Pueblo del Perú, *Violencia de género contra las mujeres en
línea*, Documento de Trabajo n.º 001-2021-DP/ADM, ago. 2021.
[PDF](https://www.defensoria.gob.pe/wp-content/uploads/2021/08/Documento-de-trabajo-01-Violencia-de-g%C3%A9nero-contra-las-mujeres-en-l%C3%ADnea.pdf).

[11] E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical
Inference,” *Journal of the American Statistical Association*, vol. 22, no.
158, pp. 209–212, 1927, doi:
[10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).

[12] R. Artstein and M. Poesio, “Inter-Coder Agreement for Computational
Linguistics,” *Computational Linguistics*, vol. 34, no. 4, pp. 555–596, 2008,
doi: [10.1162/coli.07-034-R2](https://doi.org/10.1162/coli.07-034-R2).

[13] C. E. Brodley and M. A. Friedl, “Identifying Mislabeled Training Data,”
*Journal of Artificial Intelligence Research*, vol. 11, pp. 131–167, 1999,
doi: [10.1613/jair.606](https://doi.org/10.1613/jair.606).

[14] B. Settles, *Active Learning Literature Survey*, Computer Sciences
Technical Report 1648, University of Wisconsin–Madison, 2009.
[Repositorio](https://minds.wisconsin.edu/handle/1793/60660).

[15] H. Mozannar and D. Sontag, “Consistent Estimators for Learning to Defer to
an Expert,” in *Proc. 37th International Conference on Machine Learning*, vol.
119, 2020, pp. 7076–7087.
[PMLR](https://proceedings.mlr.press/v119/mozannar20b.html).

[16] DeepSeek, “DeepSeek V4 Preview Release,” *DeepSeek API Documentation*, abr.
2026. Accedido: 29-jul-2026.
[En línea](https://api-docs.deepseek.com/news/news260424/).

[17] DeepSeek, “Models and Pricing,” *DeepSeek API Documentation*, 2026.
Accedido: 7-ago-2026.
[En línea](https://api-docs.deepseek.com/quick_start/pricing).

[18] Qwen Team, “Model Card: Qwen/Qwen3-1.7B,” revisión
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Hugging Face Hub, 2025.
Accedido: 7-ago-2026.
[En línea](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e).

[19] OpenAI, “GPT-5.6 Sol,” *OpenAI Developers*, 2026. Accedido: 9-ago-2026.
[En línea](https://developers.openai.com/api/docs/models/gpt-5.6).

[20] OpenAI, “Compare models,” *OpenAI Developers*, 2026. Accedido:
9-ago-2026.
[En línea](https://developers.openai.com/api/docs/models/compare).

## Auditoría de citas y antiplagio

- Estilo aplicado: IEEE numérico, citas en el cuerpo y lista final.
- Citas en el cuerpo: 21 apariciones; 20 referencias únicas.
- Entradas en la lista de referencias: 20.
- Afirmaciones cuantitativas del proyecto: derivadas de artefactos locales
  identificados al inicio; no requieren adjudicarse a una fuente externa.
- Ideas, tipologías, contexto lingüístico, métodos estadísticos, modelos y
  precios externos: parafraseados y citados.
- Citas textuales extensas de fuentes externas: 0.
- Referencias duplicadas: 0.
- Claves faltantes o citas sin referencia: 0.
- Referencias no utilizadas: 0.
- Fuentes pendientes: ninguna para las afirmaciones externas incluidas; los
  tiempos/costos aproximados están marcados explícitamente como estimaciones.
