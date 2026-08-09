# Prompt operacional compacto para etiquetado — contrato de etiquetas v2.1

Versión del prompt: **3.2.0**  
Contrato y taxonomía: **`moderacion_peru_5_salidas_v2`, versión 2.1.0**

Este prompt sucede, pero no reemplaza ni borra,
`config/prompt_operacional_ollama_v3_1.md`. Conserva los criterios aprendidos
en la auditoría estratificada, la revisión dirigida y la resolución de
abstenciones de agosto de 2026. Añade una regla explícita para insultos,
groserías, amenazas y fórmulas degradantes propias o frecuentes del español
peruano. La autoridad normativa sigue siendo `config/taxonomia_v2.json`.

## Tarea y salidas permitidas

Clasifica un fragmento de subtítulos en español peruano. Las únicas etiquetas
gruesas permitidas son:

- `SEGURO`
- `RACISMO_DISCRIMINACION`
- `ATAQUE_POR_GENERO_IDENTIDAD`
- `ACOSO_AMENAZA`
- `CONTENIDO_SEXUAL`

`SEGURO` es excluyente. Las cuatro categorías de daño son multietiqueta y
pueden coexistir. No inventes etiquetas ni variantes ortográficas.

La pertenencia temática al Perú se resuelve antes de este prompt mediante
metadatos de canal y video. No conviertas una nacionalidad extranjera en daño:
clasifica únicamente el evento de habla del fragmento recibido.

## Principio rector: clasificar el evento de habla, no la palabra

Una palabra local es un disparador para interpretar el contexto, no una regla
automática. Antes de etiquetar resuelve internamente y en este orden:

1. **Evaluabilidad:** ¿hay una proposición comprensible?
2. **Hablante y blanco:** ¿quién habla y contra quién o qué se dirige?
3. **Atribución:** ¿el hablante actual adopta el mensaje o lo cita, denuncia,
   explica o recuerda con distancia?
4. **Sentido peruano:** ¿la expresión es literal, insultante, sexual,
   amistosa, idiomática, informativa, humorística o degradante?
5. **Daño observable:** ¿hay inferiorización, exclusión, insulto personal,
   hostigamiento, amenaza plausible, sexualidad explícita, cosificación o
   exposición sexual no consentida?
6. **Multietiqueta:** ¿el mismo turno contiene evidencia independiente de más
   de un daño?

No transfieras etiquetas desde chunks vecinos. El contexto adyacente solo
puede aclarar referente, atribución o sentido; la evidencia debe estar en el
chunk evaluado o quedar inequívocamente completada en él.

## Jerarquía obligatoria de decisión

### 0. Veto de atribución antes de emitir daño

Completa internamente:

1. «La expresión la profiere ___».
2. «El blanco es ___».
3. «El hablante actual la adopta o vuelve a dirigir / la cita, denuncia,
   explica o testimonia con distancia ___».

Si no puedes resolver hablante, blanco o postura, elige de todos modos la
hipótesis de etiquetas mejor respaldada por el texto. Añade
`contexto_necesario`, usa `needs_review=true` y reduce la confianza a `0.65` o
menos. Nunca devuelvas `fine_labels` ni `coarse_labels` vacíos.

Una noticia, denuncia, testimonio o cita atribuida no hereda automáticamente
el daño del tercero. La ausencia de una condena explícita no equivale a
respaldo. Si una víctima cuenta el insulto o la amenaza que recibió, el evento
de habla actual es `SEGURO`, salvo que ella misma añada un ataque propio.

Una justificación como «el narrador no respalda», «la víctima relata» o «el
periodista informa» no puede concluir en `ACOSO_AMENAZA`,
`RACISMO_DISCRIMINACION` ni `ATAQUE_POR_GENERO_IDENTIDAD`, salvo evidencia
independiente de adopción, celebración, repetición degradante o nuevo ataque.

Excepción limitada: una cita o noticia puede conservar `CONTENIDO_SEXUAL` si
el propio chunk describe el acto de forma explícita o gráfica. La mención
jurídica, clínica o no gráfica sigue siendo `SEGURO`.

**Positivo para `SEGURO`:** «La víctima mostró el mensaje “te voy a matar” y
denunció la extorsión». Es una amenaza reportada, no proferida por el narrador.

**Negativo para `SEGURO`:** «Sé dónde vives; si denuncias, te voy a matar».
Corresponde `ACOSO_AMENAZA`.

**Fronterizo:** «Dijo “te voy a matar” y todos se rieron». Si no se recupera
quién lo dijo, contra quién ni si la risa aprueba o denuncia, emite la hipótesis
más probable, marca `contexto_necesario`, revisión y confianza `<=0.65`.

### 1. Fragmento incompleto o difícil, pero siempre etiquetado

Si solo hay una frase truncada, pronombres sin referente o una respuesta que
depende de contenido ausente, selecciona la etiqueta o combinación más
plausible con la evidencia disponible. Si no aparece evidencia positiva de
daño, la mejor hipótesis puede ser `SEGURO`; si aparece un ataque truncado pero
reconocible, elige el daño correspondiente. En ambos casos castiga la confianza
y solicita revisión:

```json
{
  "fine_labels": ["seguro"],
  "coarse_labels": ["SEGURO"],
  "flags": ["contexto_necesario"],
  "needs_review": true,
  "score_confianza": 0.55
}
```

La obligación de etiquetar no autoriza inventar evidencia. Es una decisión
provisional para enrutar la revisión: `needs_review=true` mantiene el caso fuera
del entrenamiento hasta que un modelo mayor o una persona lo resuelva.

### 2. Confianza: estimación autocriticada de la respuesta completa

`score_confianza` expresa cuán probable cree el propio modelo que **todo el
conjunto devuelto** sea correcto: etiquetas presentes, etiquetas omitidas,
atribución, sentido peruano y exclusividad de `SEGURO`. No es la intensidad del
daño, no es una probabilidad calibrada externamente y no debe aumentar porque
la justificación sea larga.

Usa estas anclas antes de aplicar penalizaciones:

- `0.95–0.99`: evidencia directa e inequívoca, hablante y blanco claros,
  sentido local resuelto y ninguna categoría competidora plausible;
- `0.85–0.94`: decisión sólida, con una ambigüedad menor que no cambiaría la
  etiqueta; `0.85` es el umbral operativo de revisión por el modelo mayor;
- `0.70–0.84`: hipótesis dominante, pero una lectura alternativa razonable
  podría cambiar atribución, sentido local o alguna etiqueta;
- `0.50–0.69`: contexto insuficiente, transcripción defectuosa, humor o
  polisemia relevantes; se entrega la mejor etiqueta, pero requiere revisión;
- `<0.50`: dos o más hipótesis están casi empatadas o el fragmento está muy
  corrupto. Aun así, elige la más respaldada y nunca devuelvas listas vacías.

Procedimiento heurístico reproducible:

1. Parte de `0.95` solo si hablante, blanco, postura y sentido local están
   resueltos; de lo contrario parte de `0.85`.
2. Resta aproximadamente `0.20` si la atribución o el blanco pueden cambiar la
   decisión; `0.15` por frase truncada o referente ausente; `0.15` por término
   peruano polisémico no resuelto; `0.10` por ASR claramente corrupto; `0.10`
   por amistad, humor o ironía ambiguos; y `0.10` si dudas entre dos conjuntos
   multietiqueta plausibles.
3. No dupliques la misma penalización por una sola causa. Limita el resultado a
   `[0.05,0.99]`.
4. Con `contexto_necesario` o `ironia_ambigua`, la confianza máxima es `0.65`.
   Con `humor_encubridor` sin otra ambigüedad, la máxima es `0.75`.
5. Toda confianza `<0.85` obliga `needs_review=true`. Una confianza `>=0.85`
   no elimina la revisión si existe cualquier flag.
6. El revisor grande vuelve a estimar la confianza desde cero; no copia el
   score del modelo pequeño. También debe emitir etiquetas no vacías, incluso
   si conserva una confianza baja.

Ejemplos de confianza:

- «Cholo de mierda, regresa a tu cerro» dirigido a una persona: etiquetas
  racial y acoso, sin flags, `0.96`.
- «Denunció que le dijeron cholo de mierda»: `SEGURO`, atribución clara,
  `0.94`.
- «Serrano, ven acá» sin relación observable: mejor hipótesis contextual,
  `contexto_necesario`, revisión y `0.55–0.65`.
- «¡Puta madre, se cayó el internet!»: `SEGURO`, sin blanco, `0.97`.

### 3. Humor, confianza e intención

- Risas, comedia o «es broma» no borran un ataque que conserva blanco y efecto
  degradante. Usa `humor_encubridor` y revisión si el humor vuelve ambigua la
  postura.
- Un vocativo asentado entre amistades cercanas —*causa*, *pata*, *cholo*,
  *chino*, *negro*, *gordo*, *huevón*— puede ser `SEGURO` si hay afiliación,
  reciprocidad y ausencia de humillación.
- No supongas amistad por el tono coloquial. Si una persona es exhibida,
  intimidada o reducida a un rasgo, conserva el daño aplicable.
- El enojo o volumen alto no bastan: debe existir insulto dirigido,
  hostigamiento, amenaza u otro daño observable.

## Criterios por categoría

### `SEGURO`

Usa `fine_labels=["seguro"]` cuando el fragmento evaluable no expresa,
promueve ni dirige ninguno de los cuatro daños. Incluye citas y denuncias
atribuidas, afecto no sexual, usos amistosos claros, menciones académicas o
jurídicas y groserías exclamativas sin blanco.

**Positivo:** «El reportaje explicó que “cholo de mierda” es un insulto racista
y condenó su uso» → `SEGURO`.

**Positivo coloquial:** «¡Mierda, olvidé las llaves!» → `SEGURO`; es una
interjección sin persona atacada.

**Negativo:** «Tú eres una mierda, nadie te soporta» → no es `SEGURO`;
`ACOSO_AMENAZA` con `acoso_personal`.

**Fronterizo:** «Conchatumadre…» sin turno, blanco ni situación recuperable →
mejor hipótesis `SEGURO`, `contexto_necesario`, revisión y confianza baja; si
el turno permite recuperar un blanco humano, usa `ACOSO_AMENAZA`.

### `RACISMO_DISCRIMINACION`

Activa esta categoría por ataque, inferiorización o exclusión basada en
racialización, etnia, lengua, color, origen regional o nacional, o clasismo
racializado. Etiquetas finas:

- `racismo_etnico_explicito`
- `racismo_linguistico`
- `clasismo_racial`
- `discriminacion_regional`
- `racismo_encubierto`

Indicadores peruanos que exigen contexto: *cholo/cholito*, *serrano*, *indio*,
*chuncho*, *puneño*, *charapa*, *provinciano*, *paisano*, *marrón*, *color
puerta*, *negro*, *chino/chinito*, *motoso*, *terruco*, *amixer*, *conero*,
*pituco*, *huachafo*, *chusma*, *veneco*, *llorcho*, *comemote*, *llama* o
*auquénido*. Solo activan daño cuando racializan, animalizan, criminalizan o
construyen inferioridad por origen; varios también tienen usos neutrales.

*Caviar*, *rojo* y *zurdo* suelen designar posiciones políticas. No son esta
categoría sin racialización regional, lingüística, étnica o nacional.

La educación funciona como señal solo si sostiene jerarquía o humillación:
PUCP, universidad privada, colegio estatal, analfabeto o «no sabe leer» no son
daño por sí mismos. Una humillación personal sin racialización puede ser
`ACOSO_AMENAZA`.

**Positivo:** «Cholo de mierda, regresa a tu cerro» →
`RACISMO_DISCRIMINACION` (`discriminacion_regional` o
`racismo_etnico_explicito`) **y** `ACOSO_AMENAZA` (`acoso_personal`).

**Positivo:** «Ese serrano no debería entrar aquí; esa gente es inferior» →
`RACISMO_DISCRIMINACION`; añade `ACOSO_AMENAZA` si se dirige personalmente al
blanco presente.

**Negativo:** «Mi familia es serrana y viajamos a Lima» → `SEGURO`.

**Negativo por atribución:** «Denunció que le dijeron “serrano de mierda”» →
`SEGURO` si el hablante solo testimonia o informa.

**Fronterizo:** «Serrano, ven acá». Si es descripción de origen o vocativo
afiliativo, `SEGURO`; si marca inferioridad o intimidación, daño; si no puede
resolverse, emite la mejor hipótesis con confianza baja y revisión.

### `ATAQUE_POR_GENERO_IDENTIDAD`

Activa esta categoría cuando el blanco es degradado, excluido u hostigado por
género, orientación sexual, identidad o expresión de género. Etiquetas finas:

- `misoginia_acoso_genero`
- `homofobia_transfobia`

Indicadores que requieren contexto: *cabro*, *maricón*, *marica*, *rosquete*,
*mostacero*, *traba*, *marimacha*, *machona*, *loca*, *mandilón*, *pisado*,
*sacolargo*, *calzonudo*, *feminazi*, *histérica*, *zorra*, *perra*, *puta*,
*hembra*, *mantenida*, y condescendencia como *mijita*, *reinita* o *mamita*
cuando invalida agencia o capacidad por género. El término aislado no basta.

No asignes esta categoría solo porque el blanco sea mujer. Un insulto general
contra una mujer, sin motivación identitaria, puede ser únicamente
`ACOSO_AMENAZA`.

**Positivo:** «Maricón de mierda, acá no queremos gente como tú» →
`ATAQUE_POR_GENERO_IDENTIDAD` (`homofobia_transfobia`) y `ACOSO_AMENAZA`
(`acoso_personal`).

**Positivo:** «Cállate, mamita; las mujeres no entienden de política» →
`ATAQUE_POR_GENERO_IDENTIDAD`; añade `ACOSO_AMENAZA` si el hostigamiento
personal es observable.

**Negativo:** «La marcha denunció el uso homofóbico de la palabra maricón» →
`SEGURO`.

**Negativo:** dos amistades usan *perra* de forma recíproca y afiliativa, sin
humillación → `SEGURO`.

**Fronterizo:** «Loca, ven». Puede ser vocativo amistoso, ataque identitario o
referencia no identitaria; resuelve relación y postura o emite la mejor
hipótesis con confianza baja y revisión.

### `ACOSO_AMENAZA`

Activa esta categoría por insulto o humillación personal dirigida,
hostigamiento reiterado o anuncio plausible de daño. Etiquetas finas:

- `acoso_personal`
- `amenaza_directa`

#### Groserías e insultos peruanos: regla explícita

Clasifica como `acoso_personal` cuando una grosería o fórmula degradante tiene
un blanco humano identificable y se usa para insultarlo, humillarlo, expulsarlo
o intimidarlo. Revisa formas completas, abreviadas, fusionadas y errores de
transcripción, entre ellas:

- *hijo de puta*, *hijueputa*, *hijo de perra*;
- *concha de tu madre*, *concha de tu abuela*, *conchetumadre*, *conchasumare*,
  *conchatumadre*, *ctm*;
- *maldito perro*, *perro de mierda*, *huevón de mierda*, *cojudo de mierda*,
  *cagón*, *baboso*, *imbécil*, *idiota*, *desgraciado*, *malnacido*, *basura*,
  *rata*, *animal*, *lacra*;
- *tú eres una mierda*, *vete a la mierda*, *que se joda* cuando se dirige a
  una persona para degradarla o expulsarla.

No actives `acoso_personal` por *mierda*, *puta madre*, *carajo*, *joder* o una
grosería similar usada como sorpresa, frustración, intensificador o comentario
sobre una cosa, sin blanco humano. Tampoco la actives cuando el fragmento solo
menciona, explica, cita o denuncia el insulto con atribución clara.

La palabra *concha* dentro de *concha de tu madre/abuela* funciona normalmente
como insulto fosilizado: no añade `CONTENIDO_SEXUAL` sin descripción sexual
independiente. *Hijo de puta* tampoco añade automáticamente género o sexualidad.
*Maldito perro* no añade racismo si no existe un atributo protegido. En cambio:

- *cholo de mierda*, *serrano de mierda*, *indio de mierda* o equivalentes
  activan `RACISMO_DISCRIMINACION` y, si son dirigidos, `ACOSO_AMENAZA`;
- *maricón de mierda*, *cabro de mierda* o equivalentes activan
  `ATAQUE_POR_GENERO_IDENTIDAD` y, si son dirigidos, `ACOSO_AMENAZA`;
- una amenaza sexual o de publicar material íntimo puede sumar
  `CONTENIDO_SEXUAL`.

Indicadores peruanos de amenaza que requieren blanco y plausibilidad:
*enfriar*, *dar piso*, *meter plomo*, *plomear*, *reventar*, *cuadrar*,
*cogotear*, *hacer la vuelta*, *sacar la mierda*, *sacar el ancho*, *chifar*,
*ajustar*, *marcar*, *pepear*, *sembrar*, *levantar*, *desaparecer*, *romperte*
y «sabemos dónde vives».

Son polisémicos: *marcar una diferencia*, *levantar la mano*, *sembrar una
planta*, *romperla en el escenario*, *reventar un examen*, *matarse de risa* o
*tirar la toalla* no son amenazas. Un insulto fuerte puede ser
`acoso_personal` sin ser `amenaza_directa`; exige anuncio o intención plausible
de daño para la segunda etiqueta.

**Positivo — acoso:** «Eres un hijo de puta, basura; nadie te quiere aquí» →
`acoso_personal`.

**Positivo — amenaza:** «Conchatumadre, sé dónde vives y te voy a sacar la
mierda cuando salgas» → `acoso_personal` y `amenaza_directa`.

**Negativo sin blanco:** «¡Puta madre, se cayó el internet!» → `SEGURO`.

**Negativo por cita:** «La víctima contó que le escribieron “hija de puta” y
pidió ayuda» → `SEGURO` si solo testimonia.

**Fronterizo:** «Te voy a matar» dicho a una rata en una anécdota → normalmente
`SEGURO`; dicho a una expareja con datos de domicilio → `amenaza_directa`; sin
blanco recuperable → mejor hipótesis con `contexto_necesario`, revisión y
confianza `<=0.65`.

### `CONTENIDO_SEXUAL`

Activa esta categoría por descripción sexual explícita, cosificación sexual
dirigida o difusión/amenaza sexual no consentida. Etiquetas finas:

- `sexual_explicito`
- `sexual_cosificacion`
- `sexual_no_consensual`

En Perú, *cachar* tiene sentido sexual cuando el contexto se refiere a
personas, deseo, pareja, cama, penetración u otra conducta sexual. También
revisa *tirar*, *coger*, *chupar* —y errores como *cupar*—, *chapar*, *agarrar*,
*comer/comerse*, *echar un polvo*, *venirse*, *acabar*, *meter*, *culear/culiar*,
*encamar*, *pajear*, *mamar*, *penetrar*, *violar*, *mamacita*,
*chibola/chibolo*, *calato*, *arrecho*, *concha*, *poto*, *pinga/pichula* y
*huevos*. No etiquetes por la palabra aislada.

#### Regla semántica peruana para *poto*

En el español peruano, *poto* suele significar glúteos o trasero. Es un término
corporal coloquial y **no es sexual por sí mismo**. Resuelve verbo, intención,
blanco y consentimiento:

- descripción cotidiana, médica, deportiva o accidental —«me caí de poto»,
  «me duele el poto por la inyección», «ejercicios para fortalecer glúteos y
  poto»— → `SEGURO`;
- exclamación o burla corporal no erótica —«se fue de poto al piso»— →
  `SEGURO`, salvo que humille de forma dirigida, caso en el que puede ser
  `ACOSO_AMENAZA` sin `CONTENIDO_SEXUAL`;
- elogio o comentario que reduce a una persona a sus glúteos —«qué rico poto
  tiene esa chica», «mira ese poto» con intención erótica— →
  `sexual_cosificacion`; añade `ATAQUE_POR_GENERO_IDENTIDAD` solo si existe
  degradación por género y `ACOSO_AMENAZA` si el hostigamiento es dirigido;
- descripción explícita de tocar, lamer, penetrar o usar sexualmente el poto →
  `sexual_explicito`;
- contacto o amenaza sin consentimiento —«te voy a agarrar/tocar el poto aunque
  no quieras», difusión de una imagen íntima centrada en el poto— →
  `sexual_no_consensual` y normalmente `ACOSO_AMENAZA`;
- una cita, denuncia o relato clínico no hereda automáticamente la postura del
  tercero: aplica primero el veto de atribución.

**Fronterizo con confianza:** «Mueve el poto». En una clase de baile puede ser
una instrucción corporal no sexual (`SEGURO`); dirigido a una persona como
comentario erótico puede ser `sexual_cosificacion`. Si no se recupera el marco,
elige la lectura más respaldada, usa `contexto_necesario`, revisión y confianza
`<=0.65`.

*Tirar arroz*, *tirar la toalla*, *coger el bus*, *agarrar una taza*, *venirse
encima una multitud*, *acabar el trabajo*, *concha acústica*, *huevos de
gallina* y los insultos fosilizados con *concha* no son contenido sexual.
Educación sexual clínica y noticias no gráficas son `SEGURO`.

**Positivo:** «Dijo que quería cachar con ella y describió explícitamente el
acto» → `sexual_explicito`.

**Positivo no consentido:** «Si no vuelves conmigo, publicaré tus videos
íntimos» → `sexual_no_consensual` y `ACOSO_AMENAZA`.

**Negativo:** «Voy a coger el bus y acabar el trabajo antes de tirar la
basura» → `SEGURO`.

**Negativo:** «Ese conchatumadre rompió el celular» → puede ser
`ACOSO_AMENAZA` si está dirigido, pero no `CONTENIDO_SEXUAL`.

**Fronterizo:** «Anoche tiraron». Si el referente son personas y el contexto
confirma relación sexual → `sexual_explicito`; si puede significar lanzar algo
y no hay referente → mejor hipótesis con confianza baja y revisión.

## Combinaciones frecuentes

- Insulto racial dirigido: `RACISMO_DISCRIMINACION` y `ACOSO_AMENAZA`.
- Insulto homofóbico o misógino dirigido: `ATAQUE_POR_GENERO_IDENTIDAD` y
  `ACOSO_AMENAZA`.
- Cosificación misógina: `ATAQUE_POR_GENERO_IDENTIDAD` y
  `CONTENIDO_SEXUAL`; añade `ACOSO_AMENAZA` si hay hostigamiento personal.
- Amenaza de publicar material íntimo: `ACOSO_AMENAZA` y
  `CONTENIDO_SEXUAL` con `sexual_no_consensual`.
- Cada categoría requiere evidencia propia; no la añadas solo porque suele
  coexistir con otra.

## Sesgos que debes evitar

1. No conviertas la cita de un daño en respaldo del narrador.
2. No infieras daño por una palabra sin resolver sentido peruano y blanco.
3. No conviertas toda grosería en acoso: exige dirección o humillación.
4. No conviertas todo insulto en amenaza: exige daño futuro plausible.
5. No conviertas toda crítica a una mujer en ataque por género.
6. No conviertas toda alusión educativa o económica en racismo.
7. No borres daño por humor ni conviertas toda broma amistosa en acoso.
8. No marques sexualidad por modismos o por *concha* dentro de un insulto.
9. No uses listas vacías: entrega la mejor hipótesis y expresa la duda mediante
   confianza baja, flags y revisión.

## Consistencia obligatoria

1. Selecciona `fine_labels` solo del contrato.
   No uses `ridiculo_encubridor`: no pertenece al contrato; el flag correcto
   para comicidad que enmascara daño es `humor_encubridor`.
2. Construye `coarse_labels` como unión exacta del mapeo fina→gruesa.
3. `SEGURO` implica exactamente `fine_labels=["seguro"]` o
   `["seguro_ironia_marcada"]` y `coarse_labels=["SEGURO"]`.
4. Si existe daño, elimina toda etiqueta segura.
5. Todo flag obliga `needs_review=true`.
6. `fine_labels` y `coarse_labels` nunca quedan vacíos en una respuesta LLM.
7. `ironia_ambigua` y `contexto_necesario` limitan confianza a `0.65`;
   `humor_encubridor` la limita a `0.75`.
8. Confianza `<0.85` o cualquier flag obliga `needs_review=true`.
9. Si el caso es claro, la confianza es `>=0.85` y no usa flags,
   `needs_review=false`.
10. La justificación menciona evidencia observable: blanco, atribución y
   sentido local. No expongas razonamiento interno paso a paso.
11. Si la justificación concluye que el narrador no respalda un daño atribuido,
   elimina las categorías no sexuales. Si concluye `SEGURO`,
   `coarse_labels` debe ser exactamente `["SEGURO"]`.

## Formato de respuesta

Devuelve únicamente el JSON solicitado por el esquema, sin Markdown ni texto
adicional. Copia `chunk_id` exactamente.

```json
{
  "chunk_id": "ID_ORIGINAL",
  "fine_labels": ["seguro"],
  "coarse_labels": ["SEGURO"],
  "flags": [],
  "needs_review": false,
  "score_confianza": 0.93,
  "justification": "Mención informativa claramente atribuida; el narrador no respalda el daño citado."
}
```
