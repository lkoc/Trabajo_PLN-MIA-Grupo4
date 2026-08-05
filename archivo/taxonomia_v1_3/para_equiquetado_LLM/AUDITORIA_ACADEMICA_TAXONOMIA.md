# Auditoría académica y operativa de la taxonomía

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha de revisión: 2026-07-29

Alcance: todos los archivos de `para_equiquetado_LLM/`, sus copias canónicas y el código que deriva las salidas gruesas.
Propósito: documentar qué significan las etiquetas, qué fuentes las orientan y qué decisiones pertenecen al proyecto.

La carpeta se conserva como evidencia histórica. Esta auditoría no modifica retrospectivamente la guía v1.3 ni las salidas que la usaron.

## Conclusión principal

La formulación correcta es:

> Taxonomía operativa multietiqueta, informada por literatura sobre lenguaje abusivo y por estudios sociolingüísticos e institucionales pertinentes al Perú, destinada a priorizar revisión humana.

No corresponde llamarla taxonomía legal, política completa de YouTube ni taxonomía validada por expertos. No se encontró una adjudicación documentada por un panel experto peruano.

El contrato contiene:

- 12 fenómenos finos de daño;
- 2 estados seguros;
- 3 flags transversales de revisión;
- 5 salidas históricas de daño;
- 4 salidas activas, luego de unir ataque personal y amenaza mediante OR/máximo.

Por tanto, la frase «14 fenómenos de daño» es incorrecta. Debe decir «14 etiquetas finas: 12 fenómenos de daño y dos estados seguros, más tres flags transversales».

## Inventario completo

| Archivo | Contenido comprobado | Papel probatorio |
|---|---|---|
| `taxonomia_moderacion.csv` | 14 etiquetas no-FLAG y 3 flags | Vocabulario fino y agrupación documental |
| `clasificacion_moderacion_peru.md` | Guía semántica v1.3, siete pasos, ejemplos y bibliografía | Instrucción histórica para humano/LLM |
| `PROMPT_ETIQUETADO_LLM.md` | Entrada, salida, invariantes, confianza y revisión | Protocolo operativo del paquete |
| `chunks_para_etiquetar.json` | 69 853 objetos NDJSON sin anotar | Entrada transportada; la extensión es incorrecta |
| `ejemplo_formato_salida.json` | Un objeto de 12 campos | Ejemplo estructural, no dato evaluado |
| `cgt_labeled_chunks_parte_0001.jsonl` | Primeras 20 decisiones | Salida LLM no adjudicada |
| `cgt_labeled_chunks_parte_0002.jsonl` | Siguientes 20 decisiones | Salida LLM no adjudicada |
| `cgt_labeled_chunks_parte_0003.jsonl` | Siguientes 20 decisiones | Salida LLM no adjudicada |

La entrada tiene 69 853 `chunk_id` y `text_hash` únicos, 1 856 videos, 26 canales y ningún texto vacío. `channel_id` y `published_at` son nulos en las 69 853 filas. Todas las filas llegan con `labels=[]`, `flags=[]` y `needs_review=true`.

El archivo termina en `.json`, pero contiene un objeto JSON independiente por línea. Su SHA-256 es:

```text
EB90DEBF66D5E16AF72C41C17C3701197E42BDCC78B81E0F914C6A49C56F8AB4
```

Ese hash coincide con `datos/processed/chunks_para_etiquetar.jsonl`.

## Jerarquía de autoridad

La carpeta auditada es un paquete de transporte, no la única fuente ejecutable. La trazabilidad correcta es:

1. `datos/processed/taxonomia_moderacion.csv`: vocabulario que carga el flujo híbrido;
2. `modelos/skills/clasificacion_moderacion_peru.md`: guía semántica que cargan los cuadernos;
3. `para_equiquetado_LLM/PROMPT_ETIQUETADO_LLM.md`: reglas adicionales del paquete;
4. `scripts_auxiliares/modelos_gruesos_moderador.py`: mapeo fino a cinco daños;
5. `scripts_auxiliares/entrenar_qwen_acoso_amenaza.py`: unión de cinco a cuatro daños;
6. ejemplos y salidas CGT: ilustración del flujo, no autoridad académica.

Las copias auditadas del CSV y la guía coinciden por hash con sus rutas canónicas:

```text
taxonomia_moderacion.csv
763C62F3D51706F0260636CDF26FE29E7FE23A9596DF17C0D9FA2649E88BC8B4

clasificacion_moderacion_peru.md
45F9D3231A92453835EE6DFCBB8CFFF0B682718CAA4111F4BCA3E841573A0EFB
```

La columna `categoria=ACOSO` del CSV agrupa cuatro etiquetas para la guía. No representa las cinco salidas históricas ni las cuatro activas.

## Marco académico usado para revisar las definiciones

La taxonomía general se apoya en:

- [Waseem et al. (2017)](https://aclanthology.org/W17-3012/): diferencia blanco individual o entidad frente a grupo generalizado, y abuso explícito frente a implícito.
- [Banko et al. (2020)](https://aclanthology.org/2020.alw-1.16/): criterios y excepciones para ataque por identidad, insulto, doxeo, agresión sexual y amenaza de violencia; recomienda categorías finas y no necesariamente excluyentes.
- [Wulczyn et al. (2017)](https://doi.org/10.1145/3038912.3052591): ataques personales a escala.
- [ElSherief et al. (2021)](https://aclanthology.org/2021.emnlp-main.29/): odio implícito expresado mediante lenguaje codificado o indirecto.
- [Ilić et al. (2018)](https://aclanthology.org/W18-6202/): ironía y sarcasmo como problema de interpretación automática.
- [Bourgeade et al. (2024)](https://aclanthology.org/2024.lrec-main.740/): el contexto conversacional puede ser necesario para anotar lenguaje abusivo.
- [Zeinert et al. (2021)](https://aclanthology.org/2021.acl-long.247/): taxonomía y código de anotación de misoginia, con distinción de blanco, acoso, descrédito, estereotipo y cosificación.

La adaptación peruana se relaciona con:

- Callirgos (1993), Portocarrero (2009) y [Vich (2018)](https://revistas.pucp.edu.pe/index.php/debatesensociologia/article/view/22090): racismo, mestizaje y formas encubiertas.
- Zavala y Zariquiey (2007), [Zavala y Back (2017)](https://repositorio.pucp.edu.pe/index/handle/123456789/170315), Brañez (2012) y [Zavala y Almeida (2022)](https://revistas.pucp.edu.pe/index.php/lexis/article/view/26332): racialización mediante educación, cultura, clase, territorio, habla y escritura.
- [Salem (2016)](https://www.antropologiavisual.cl/amixer-esta-en-facebook-una-investigacion-de-la-choledad-virtual): insulto, parodia, ironía y humor en la construcción del estereotipo `amixer`.
- [Albornoz y Flores (2018)](https://hiperderecho.org/tecnoresistencias/reporte/) y [Defensoría del Pueblo (2021)](https://www.defensoria.gob.pe/wp-content/uploads/2021/08/Documento-de-trabajo-01-Violencia-de-g%C3%A9nero-contra-las-mujeres-en-l%C3%ADnea.pdf): violencia de género en línea, amenazas, doxeo y difusión íntima sin consentimiento en Perú.
- [Rottenbacher (2012)](https://www.redalyc.org/pdf/801/80124028002.pdf): homofobia y prejuicio hacia grupos transgénero en una muestra de Lima.
- [Lovón-Cueva y Lovón-Cueva (2022)](https://whatever.cirque.unipi.it/index.php/journal/article/view/156): léxico lesbofóbico en foros peruanos.
- [Thakur (2025)](https://cdt.org/wp-content/uploads/2025/06/2025-Quechua-Report-Spanish-final-1.pdf): límites de moderación para quechua; no valida por sí sola reglas del español peruano.

La [política oficial de YouTube sobre contenido sexual](https://support.google.com/youtube/answer/2802002?hl=es-419) orienta el alcance de la plataforma y sus excepciones educativas, documentales, científicas o artísticas. Es una fuente normativa de plataforma, no una publicación académica.

## Matriz de etiquetas

| Etiqueta | Definición defendible para el artículo | Inclusión | Exclusión o límite | Sustento principal |
|---|---|---|---|---|
| `seguro` | Estado sin daño cubierto activado | Información, descripción o humor sin ataque | No significa ausencia universal de daño | Regla local |
| `seguro_ironia_marcada` | Ironía cuyo blanco no recibe ataque cubierto | Crítica de institución, política o situación | Debe excluir ataques a personas y grupos | Regla local; Ilić para dificultad de ironía |
| `racismo_etnico_explicito` | Ataque explícito por identidad étnico-racial | Término étnico usado para degradar o excluir | Mención, cita, reapropiación o descripción no bastan | Waseem; Callirgos; Zavala y Back |
| `racismo_linguistico` | Racialización mediante burla de habla o escritura | Burla de motoseo, acento andino o escritura asociada a migración | Español andino o quechua mezclado no son daños | Zavala y Almeida; Zavala y Back |
| `clasismo_racial` | Inferiorización de clase articulada con racialización | Clase, consumo o apariencia usados para construir inferioridad étnica | No todo insulto de clase tiene componente racial | Callirgos; Brañez; Zavala y Back |
| `discriminacion_regional` | Ataque por origen regional | Inferiorización de personas de provincia, sierra u otra región | La mención geográfica neutral no activa | Zavala y Zariquiey; estudios peruanos de racialización |
| `racismo_encubierto` | Racialización indirecta tras criterios supuestamente neutrales | Educación, cultura o civismo aplicados para inferiorizar a un grupo racializado | Una crítica concreta de conducta no basta | Portocarrero; Vich; ElSherief |
| `misoginia_acoso_genero` | Ataque por ser mujer o degradación sexualizada/de género | Insulto sexualizado, descrédito, estereotipo o cosificación de género | Insulto genérico a una mujer no es automáticamente misoginia | EXIST; Zeinert; Defensoría; Albornoz y Flores |
| `homofobia_transfobia` | Ataque por orientación sexual o identidad/expresión de género | Insulto, rechazo, ridiculización o amenaza por esa identidad | Mención neutral, apoyo, testimonio o reapropiación no activan | Chakravarthi; Rottenbacher; Lovón-Cueva y Lovón-Cueva |
| `acoso_personal` | Ataque dirigido/acoso personal operativo | Ataque a persona identificable, doxeo o llamado a atacarla | Crítica no degradante; la repetición no puede inferirse de un solo chunk | Waseem; Wulczyn; Banko; Defensoría |
| `amenaza_directa` | Expresión explícita de intención de daño | Amenaza física; el protocolo local añade daño legal/económico | Crítica, advertencia legítima, noticia, ficción o «que se vaya» no bastan | Banko para amenaza de violencia; extensión local para legal/económico |
| `sexual_explicito` | Descripción sexual gráfica sin finalidad informativa | Texto explícito cuyo propósito no es informar o documentar | Salud, educación, periodismo, ciencia o arte requieren contexto | Banko; política de YouTube |
| `sexual_cosificacion` | Reducción de una persona a cuerpo, función o valor sexual | Comentario que niega agencia o valor no sexual de la persona | Atracción, mención corporal o broma consentida no bastan sin degradación | Zeinert; EXIST; Banko |
| `sexual_no_consensual` | Producción, exposición o circulación íntima sin consentimiento | Filtración, grabación o distribución no consentida | Una noticia o condena del hecho debe distinguir uso de mención | Defensoría; Albornoz y Flores; YouTube |
| `ironia_ambigua` | Flag de sentido incierto entre crítica y reproducción del daño | Hay daño plausible, pero el blanco o postura no se resuelve | No es daño ni probabilidad calibrada | Ilić; Waseem; ElSherief |
| `humor_encubridor` | Flag cuando el humor minimiza o niega un daño ya identificado | «Es broma» u otra defensa explícita del ataque | Risa o tono jocoso por sí solos no bastan | Salem; ElSherief; regla local |
| `contexto_necesario` | Flag de evidencia insuficiente en el chunk | Referencia anafórica, cita, término local o frase cortada que cambia la decisión | No debe forzar una etiqueta positiva inventada | Bourgeade; Thakur; regla local |

## De etiquetas finas a salidas operativas

El código histórico deriva cinco daños:

```text
RACISMO_DISCRIMINACION
  <- 5 etiquetas raciales

ACOSO_GENERO_IDENTIDAD
  <- misoginia_acoso_genero OR homofobia_transfobia

ACOSO_PERSONAL
  <- acoso_personal

AMENAZA_DIRECTA
  <- amenaza_directa

CONTENIDO_SEXUAL
  <- 3 etiquetas sexuales
```

El contrato activo aplica:

```text
ACOSO_AMENAZA = max(ACOSO_PERSONAL, AMENAZA_DIRECTA)
```

La unión aumentó soporte de entrenamiento. No afirma que ataque personal y amenaza sean iguales en sentido, gravedad o tratamiento jurídico.

`SEGURO` se deriva cuando ninguna de las cuatro salidas está activa. No es una quinta salida de daño ni una certificación de seguridad universal.

## Contradicciones encontradas

1. La guía indica usar `contexto_necesario` si un término no se comprende, pero el prompt elimina todo flag que no acompañe una categoría de daño. El esquema no puede expresar «indeterminado sin daño todavía sustentado».
2. El CSV exige intención explícita para `amenaza_directa`; la guía permite amenaza implícita. El ejemplo R-1 trata «que se vaya» como amenaza aunque no expresa daño físico, legal o económico.
3. La guía pregunta si el ataque personal es sistemático, pero otros ejemplos aplican `acoso_personal` a una sola descalificación. En el artículo se usa «ataque dirigido/acoso personal operativo».
4. La tabla de `seguro_ironia_marcada` solo excluye como blanco a un grupo humano, aunque la regla general también debe excluir el ataque a una persona.
5. La tabla de `humor_encubridor` exige minimización explícita; el procedimiento posterior lo amplía a cualquier `jajaja` o tono jocoso.
6. La regla de discurso citado supone daño si no aparece oposición explícita. Falta distinguir uso, mención, noticia, condena, ficción y contrargumentación.
7. La tabla de canales asigna riesgos y frecuencias a medios concretos sin evidencia documentada. Es un prior heurístico que puede anclar al anotador y no debe presentarse como parte académica de la taxonomía.
8. `score_confianza` es una autoevaluación del LLM, no una probabilidad calibrada. Los cortes 0,65 y 0,70 son reglas locales de flujo.

## Errores bibliográficos de la guía v1.3

- Zavala y Almeida: DOI correcto `10.18800/lexis.202202.002`, no `.004`.
- Vich: páginas 219–232 y DOI `10.18800/debatesensociologia.201802.008`, no 127–145 y `.006`.
- Zavala y Zariquiey: capítulo pp. 333–370; el libro está editado por Teun A. van Dijk.
- Monge-Olivarría, Guerra-Corrales y Bringas-Castro trabajan desde la Universidad Autónoma de Sinaloa. El artículo no acredita un corpus de Twitter peruano.
- Brañez se conserva en la bibliografía del paper como tesis de licenciatura PUCP. No debe citarse simultáneamente como si fuera un artículo distinto sin verificar la versión.
- La guía atribuye a Brañez una regla amplia sobre humor como cobertura. Salem ofrece evidencia peruana más directa sobre insulto, parodia, ironía y humor alrededor de `amixer`.

## Auditoría de las 60 salidas CGT

Las tres partes contienen exactamente los primeros 60 `chunk_id`, en el mismo orden, todos del primer video. Representan 0,086 % de la entrada y no son una muestra aleatoria.

Distribución observada:

| Etiqueta/flag | Conteo |
|---|---:|
| `seguro` | 51 |
| `homofobia_transfobia` | 4 |
| `sexual_cosificacion` | 4 |
| `acoso_personal` | 3 |
| `misoginia_acoso_genero` | 1 |
| `ironia_ambigua` | 2 |

Las etiquetas son multietiqueta, por lo que los conteos no suman 60. Nueve de las 14 etiquetas finas y dos de los tres flags no aparecen. Los archivos cumplen mecánicamente el esquema y las reglas de coexistencia, pero:

- no son `gold` humano;
- no prueban validez de contenido;
- ninguna justificación contiene una referencia bibliográfica formal, pese a que la guía la solicita;
- `GPT-5 (Codex)` no identifica una revisión de modelo/API suficientemente exacta para reproducir la anotación.

## Riesgo de reproducibilidad del cuaderno 03

La celda de empaquetado de `Cuadernos/03_frontend_etiquetado_humano_html.ipynb` elimina el contenido de `para_equiquetado_LLM/` y lo vuelve a crear. En su estado actual:

- borraría las tres salidas CGT;
- cambiaría los nombres de entrada y ejemplo a `.jsonl`;
- restauraría una versión anterior del prompt, sin las reglas actuales de flags;
- exigiría una lista fija de cinco archivos.

No se debe ejecutar esa celda para regenerar el paquete actual hasta versionarla y hacer una copia no destructiva. Una solución futura debe escribir en una carpeta nueva, incluir un manifiesto con hashes y preservar toda salida existente.

## Texto recomendado para paper o memoria

> Se definió una taxonomía operativa para priorizar revisión humana, no para establecer delitos ni infracciones de plataforma. El contrato conserva 14 etiquetas finas: 12 fenómenos de daño y dos estados seguros. La versión histórica derivó cinco salidas de daño; la activa une ataque personal y amenaza mediante OR para aumentar el soporte de entrenamiento. Esta unión es una decisión estadística y no implica equivalencia semántica o jurídica. Tres flags transversales registran ambigüedad, humor que minimiza un daño ya identificado o necesidad adicional de contexto.

## Requisitos para una versión 2

1. Conservar v1.3 con hash y crear una guía v2 separada.
2. Resolver las ocho contradicciones anteriores.
3. Añadir definición, inclusión, exclusión, contraejemplo y fuente por etiqueta.
4. Separar crítica, cita, noticia, condena, ficción y reproducción del daño.
5. Permitir el estado «indeterminado/contexto requerido» sin inventar daño ni declararlo seguro.
6. Eliminar priors por canal o validarlos y tratarlos como variable de auditoría, nunca como atajo de decisión.
7. Someter el código de anotación a especialistas peruanos pertinentes: sociolingüística/racismo, violencia de género, diversidad sexual, moderación y ética.
8. Ejecutar doble anotación ciega, adjudicación y acuerdo por etiqueta sobre una muestra estratificada.
9. Crear contrastes funcionales y casos mínimos por uso/mención, ironía y contexto.
10. Versionar guía, CSV, prompt, modelo anotador y resultados con hashes.
