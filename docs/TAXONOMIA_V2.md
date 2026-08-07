# Taxonomía operativa v2

Versión: `2.1.0`  
Contrato: `moderacion_peru_5_salidas_v2`

La taxonomía es una decisión de moderación informada por literatura y fuentes institucionales. No constituye una clasificación jurídica ni una taxonomía validada por expertos peruanos.

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

## Salidas entrenadas

| Salida | Definición breve | Fenómenos finos |
|---|---|---|
| `SEGURO` | Fragmento evaluable sin ninguno de los cuatro daños cubiertos | `seguro`, `seguro_ironia_marcada` |
| `RACISMO_DISCRIMINACION` | Ataque o exclusión por racialización, etnia, lengua, origen o clasismo racializado | cinco fenómenos |
| `ATAQUE_POR_GENERO_IDENTIDAD` | Daño dirigido por género, orientación, identidad o expresión | misoginia/acoso de género; homofobia/transfobia |
| `ACOSO_AMENAZA` | Ataque personal o anuncio plausible de daño | acoso personal; amenaza directa/implícita plausible |
| `CONTENIDO_SEXUAL` | Sexual explícito, cosificación o sexual no consentido | tres fenómenos |

`RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL` son multietiqueta. `SEGURO` es positivo y supervisado, pero excluyente. Un texto sin contexto suficiente queda indeterminado y no se fuerza a `SEGURO`.

## Correcciones respecto de v1.3

- Una amenaza puede ser implícita si existen blanco y daño plausibles; la etiqueta fina histórica se conserva por trazabilidad.
- Un ataque aislado grave puede entrar en `ACOSO_AMENAZA`; la repetición se registra cuando el contexto la demuestra, no se presume.
- Discurso citado, reportado, condenado, educativo o ficticio se evalúa por atribución y respaldo, no por palabras aisladas.
- `contexto_necesario` puede acompañar una abstención sin inventar daño.
- Ironía y humor solo activan flags cuando alteran materialmente la interpretación; “jajaja” por sí solo no basta.
- Se eliminan priors por canal del prompt activo para evitar anclaje.

La especificación ejecutable, inclusiones, exclusiones y contraejemplos están en `config/taxonomia_v2.json`. La guía v1.3 permanece intacta en `archivo/taxonomia_v1_3/`.

## Justificación de `ATAQUE_POR_GENERO_IDENTIDAD`

La categoría se define por el atributo contra el que se dirige el daño —género, orientación sexual, identidad o expresión— y no por una única modalidad de conducta. Incluye misoginia o machismo hostil, degradación, exclusión, acoso por género, homofobia y transfobia. El término `ATAQUE_POR` hace explícito el daño, mientras que `GENERO_IDENTIDAD` por sí solo sería neutral. Tampoco se usa «odio», porque exigiría una intención más estrecha y dejaría fuera ataques degradantes o discriminatorios que el contrato sí cubre.

`ACOSO_AMENAZA` describe otra dimensión: ataque interpersonal, hostigamiento o amenaza. Como los daños son multietiqueta, un ataque misógino dirigido a una persona puede recibir ambas categorías. El nombre heredado se acepta únicamente como alias de entrada para migrar datos; toda salida nueva usa `ATAQUE_POR_GENERO_IDENTIDAD`.
