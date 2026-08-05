# Matriz de evidencia de las categorías de daño

Versión auditada: `2.1.0`
Fecha de revisión de los adjuntos locales: 2026-08-05

Esta matriz justifica las cuatro salidas de daño del contrato activo. La bibliografía orienta definiciones, fenómenos y límites, pero no valida automáticamente la taxonomía, sus prevalencias ni su desempeño en subtítulos peruanos. La decisión final sigue siendo operacional y deberá someterse a doble anotación independiente, adjudicación y evaluación por fenómeno.

## Resumen de cobertura

| Salida | Pregunta operacional | Fenómenos finos | Base general | Evidencia peruana adjunta | Decisión y límite local |
|---|---|---|---|---|---|
| `RACISMO_DISCRIMINACION` | ¿Se inferioriza, excluye o ataca por racialización, etnia, lengua, procedencia o clase racializada? | cinco | ataque por identidad; blanco grupal/individual; abuso explícito/implícito (`banko2020taxonomy`, `waseem2017abuse`) | Vich; Zavala y Almeida; Brañez; Salem, además de Callirgos, Portocarrero y Zavala en la bibliografía | La realidad peruana exige observar jerarquías, motoseo/terruqueo, ortografía, “amixer”, humor y formas encubiertas. Mencionar una identidad o variedad lingüística no basta. |
| `ATAQUE_POR_GENERO_IDENTIDAD` | ¿Hay daño dirigido por ser mujer, por género, orientación sexual, identidad o expresión de género? | dos | ataque por identidad, misoginia, homofobia y transfobia (`banko2020taxonomy`, `zeinert2021misogyny`, `chakravarthi2024homotrans`) | Albornoz y Flores; Defensoría del Pueblo; Lovón-Cueva y Lovón-Cueva; Rottenbacher | `ATAQUE_POR` explicita el daño sin afirmar que todo caso sea acoso o que requiera intención de odio. Puede coexistir con `ACOSO_AMENAZA`. |
| `ACOSO_AMENAZA` | ¿Existe ataque interpersonal, hostigamiento o anuncio plausible de daño contra una persona? | dos | ataque dirigido, insulto y amenaza (`waseem2017abuse`, `wulczyn2017exmachina`, `banko2020taxonomy`) | Albornoz y Flores; Defensoría del Pueblo | La unión mejora el soporte de entrenamiento; no equipara acoso y amenaza. La reiteración se registra cuando se observa y la amenaza implícita exige blanco, daño plausible y contexto suficiente. |
| `CONTENIDO_SEXUAL` | ¿Existe sexualidad explícita cubierta por la política, cosificación o material sexual no consentido? | tres | agresión sexual, cosificación y contenido sexual (`banko2020taxonomy`, `zeinert2021misogyny`) | Albornoz y Flores; Defensoría del Pueblo | La evidencia peruana es fuerte para sexualización dañina y material íntimo no consentido. La inclusión de sexual explícito es una frontera de moderación de plataforma, no la afirmación de que toda sexualidad sea daño. |

## 1. `RACISMO_DISCRIMINACION`

La distinción general entre blanco individual/grupal y abuso explícito/implícito procede de Waseem et al.; Banko et al. separan el ataque basado en identidad de otras modalidades. Los adjuntos peruanos muestran por qué el contrato no puede depender solo de insultos raciales literales:

- [Zavala y Almeida](../referencias_y_descargas/almeida2022motoso__zavala_2022_motoso-terruco.pdf) analizan en redes peruanas cómo “motoso” y “terruco” funcionan como recursos de racialización, reubicación y silenciamiento, y cómo el habla atribuida puede inferiorizar al hablante.
- [Brañez](../referencias_y_descargas/branez2012amixer__branez_2012_identidades-amixer.pdf) documenta la construcción digital del “amixer” mediante procedencia, clase, estética, migración y ortografía convertida en herramienta de jerarquización.
- [Salem](../referencias_y_descargas/salem2016amixer__salem_2016_amixer-facebook.pdf) muestra que insulto, parodia, ironía y humor reproducen estereotipos y ataque simbólico en una página peruana de Facebook.
- [Vich](../referencias_y_descargas/vich2018dinamicas__vich_2018_dinamicas-racismo-peru.pdf) describe el racismo peruano como jerarquización, inferiorización y exclusión que cambia de forma y puede actuar sin una declaración racial literal.

Estas fuentes justifican conservar `racismo_linguistico`, `clasismo_racial` y `racismo_encubierto`, además del insulto étnico explícito. `discriminacion_regional` queda respaldada de manera contextual por la asociación entre procedencia, migración y jerarquía, pero requiere una auditoría funcional propia: la bibliografía no ofrece una prevalencia nacional ni una lista cerrada de regionalismos dañinos.

## 2. `ATAQUE_POR_GENERO_IDENTIDAD`

El nombre expresa una relación de daño: una persona o grupo es degradado, excluido, sexualizado o atacado por género, orientación, identidad o expresión. No se usa `GENERO_IDENTIDAD` solo porque sería neutral; tampoco se usa “odio” como condición necesaria, ya que dejaría fuera misoginia, machismo hostil, estereotipo degradante y discriminación sin intención de odio demostrable.

- [Albornoz y Flores](../referencias_y_descargas/albornoz2018conocer__albornoz-flores_2018_violencia-genero-linea-peru.pdf) estudian violencia de género en línea en el Perú y describen ataques contra mujeres, personas LGBTIQ+ y quienes desafían normas machistas, patriarcales o heteronormativas; también registran lenguaje agresivo, hostigamiento, amenazas y deslegitimación de orientaciones e identidades.
- La [Defensoría del Pueblo](../referencias_y_descargas/defensoria2021violenciaenlinea__defensoria_2021_violencia-genero-en-linea.pdf) sitúa el daño en un continuo estructural, simbólico, psicológico o sexual mediado por tecnología y señala afectación desproporcionada a mujeres y personas discriminadas por género, incluidas identidades LGBTI.
- [Lovón-Cueva y Lovón-Cueva](../referencias_y_descargas/lovon2022lesbofobia__lovon-cueva_2022_lexico-lesbofobico.pdf) identifican léxico lesbofóbico en ciberforos peruanos y lo analizan como violencia simbólica y actos de lenguaje de odio por orientación sexual.
- [Rottenbacher](../referencias_y_descargas/rottenbacher2012homofobia__rottenbacher_2012_homofobia-transgenero.pdf) aporta evidencia localizada en una muestra universitaria de Lima sobre homofobia y prejuicio hacia grupos transgénero; su alcance no se extrapola a toda la población peruana ni a YouTube.

Las categorías generales de ataque por identidad, misoginia, homofobia y transfobia se contrastan con `banko2020taxonomy`, `zeinert2021misogyny`, `rodriguez2021exist` y `chakravarthi2024homotrans`. Estas últimas no son estudios del Perú: aportan definición computacional y criterios, no vocabulario peruano ni prevalencia local.

## 3. `ACOSO_AMENAZA`

Banko et al. distinguen insulto y amenaza de violencia; Waseem et al. separan abuso dirigido y generalizado, y Wulczyn et al. aportan el antecedente computacional de ataque personal. En el contexto peruano:

- Albornoz y Flores distinguen lenguaje agresivo, hostigamiento/acoso y amenazas/extorsión. Su encuesta dirigida registró esas modalidades, pero no fue una muestra probabilística; sus porcentajes no se usan como prevalencia nacional ni de YouTube.
- La Defensoría enumera lenguaje agresivo, hostigamiento, amenazas, ataques coordinados y difusión de datos como modalidades que pueden coexistir y escalar.

La salida agrupa `acoso_personal` y `amenaza_directa` para aumentar soporte estadístico, pero mantiene ambos fenómenos finos y reporta sus métricas por separado. Un ataque aislado grave puede activar la salida sin afirmar un patrón repetido. Una amenaza puede ser inequívocamente implícita solo si hay blanco y daño plausibles; de lo contrario se marca `contexto_necesario` y se difiere. Esa ampliación es una regla local informada por la distinción general explícito/implícito, no una definición tomada literalmente de una fuente peruana ni una calificación jurídica.

## 4. `CONTENIDO_SEXUAL`

La categoría reúne tres riesgos distintos y los conserva como etiquetas finas:

- `sexual_explicito`: frontera operativa de moderación textual. Se excluyen educación sexual, salud, información periodística y otros usos contextualmente legítimos. Su sustento principal es la política de plataforma y la tipología general; no se presenta como particularidad peruana.
- `sexual_cosificacion`: Zeinert et al. incluyen estereotipo y cosificación en taxonomías de misoginia; la Defensoría describe lenguaje que asigna a las mujeres un papel únicamente reproductivo o sexualizado.
- `sexual_no_consensual`: Albornoz y Flores y la Defensoría documentan en el Perú almacenamiento o difusión de material íntimo sin consentimiento, acoso sexual y explotación sexual facilitada por tecnología.

El modelo analiza subtítulos, no imágenes. Por tanto, solo puede detectar evidencia textual del contenido o del acto narrado; no debe inferir desnudez visual, consentimiento ni delito sin evidencia adicional. `CONTENIDO_SEXUAL` es una categoría de riesgo de moderación y no convierte toda expresión sexual en daño ni en infracción legal.

## Fronteras multietiqueta e interseccionalidad

Las fuentes peruanas describen daños superpuestos. El contrato preserva esa realidad mediante multietiqueta:

- un insulto lesbofóbico a una persona puede ser `ATAQUE_POR_GENERO_IDENTIDAD` y `ACOSO_AMENAZA`;
- una amenaza sexualizada contra una mujer puede activar `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`;
- un ataque que combina origen andino y misoginia puede activar `RACISMO_DISCRIMINACION` y `ATAQUE_POR_GENERO_IDENTIDAD`.

No se asigna una segunda categoría por asociación automática: cada salida necesita evidencia propia en el fragmento o contexto documentado. La mención neutral, la cita condenatoria y la recuperación de una palabra identitaria no constituyen daño por sí solas.

## Resultado de la auditoría

Las cuatro categorías gruesas son defendibles para un moderador peruano si se mantienen estos límites:

1. se presentan como categorías operativas, no jurídicas ni exhaustivas;
2. las fuentes peruanas contextualizan lenguaje, blancos y modalidades, pero no validan el modelo;
3. las decisiones de fusión, nombres, umbrales y reglas de contexto se declaran como locales;
4. el rendimiento nuevo se mide por las cinco salidas entrenadas y por los 12 fenómenos finos, con atención a falsos seguros e intersecciones;
5. futuras referencias adjuntas se incorporan mediante la matriz de trazabilidad, sin cambiar silenciosamente el contrato.

La correspondencia BibTeX y los límites de cada cita se mantienen en [fuentes_base.md](../bibliografia/fuentes_base.md) y en [referencias.bib](../Documento_final_paper/referencias.bib).
