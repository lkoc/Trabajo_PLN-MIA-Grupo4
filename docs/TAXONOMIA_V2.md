# Taxonomía operativa v2

Versión: `2.0.0`  
Contrato: `moderacion_peru_5_salidas_v2`

La taxonomía es una decisión de moderación informada por literatura y fuentes institucionales. No constituye una clasificación jurídica ni una taxonomía validada por expertos peruanos.

## Salidas entrenadas

| Salida | Definición breve | Fenómenos finos |
|---|---|---|
| `SEGURO` | Fragmento evaluable sin ninguno de los cuatro daños cubiertos | `seguro`, `seguro_ironia_marcada` |
| `RACISMO_DISCRIMINACION` | Ataque o exclusión por racialización, etnia, lengua, origen o clasismo racializado | cinco fenómenos |
| `ACOSO_GENERO_IDENTIDAD` | Ataque por género, orientación, identidad o expresión | misoginia/acoso de género; homofobia/transfobia |
| `ACOSO_AMENAZA` | Ataque personal o anuncio plausible de daño | acoso personal; amenaza directa/implícita plausible |
| `CONTENIDO_SEXUAL` | Sexual explícito, cosificación o sexual no consentido | tres fenómenos |

Los daños son multietiqueta. `SEGURO` es positivo y supervisado, pero excluyente. Un texto sin contexto suficiente queda indeterminado y no se fuerza a `SEGURO`.

## Correcciones respecto de v1.3

- Una amenaza puede ser implícita si existen blanco y daño plausibles; la etiqueta fina histórica se conserva por trazabilidad.
- Un ataque aislado grave puede entrar en `ACOSO_AMENAZA`; la repetición se registra cuando el contexto la demuestra, no se presume.
- Discurso citado, reportado, condenado, educativo o ficticio se evalúa por atribución y respaldo, no por palabras aisladas.
- `contexto_necesario` puede acompañar una abstención sin inventar daño.
- Ironía y humor solo activan flags cuando alteran materialmente la interpretación; “jajaja” por sí solo no basta.
- Se eliminan priors por canal del prompt activo para evitar anclaje.

La especificación ejecutable, inclusiones, exclusiones y contraejemplos están en `config/taxonomia_v2.json`. La guía v1.3 permanece intacta en `archivo/taxonomia_v1_3/`.

