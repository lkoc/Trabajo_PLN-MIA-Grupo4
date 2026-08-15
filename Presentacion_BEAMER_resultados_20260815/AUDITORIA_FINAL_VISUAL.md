# Auditoría final visual y gráfica

**Fecha:** 15 de agosto de 2026  
**Artefacto auditado:** `presentacion_resultados_finales.pdf`  
**Estado:** aprobada para exposición

## Alcance

Se revisaron las 46 páginas renderizadas de la presentación, incluidos portada, seis separadores de capítulo, 32 diapositivas temáticas y siete páginas de referencias. La revisión se hizo sobre imágenes rasterizadas del PDF final, no únicamente sobre el código LaTeX.

## Comprobaciones realizadas

| Criterio | Resultado |
|---|---|
| Formato | 16:9 uniforme, 453,543 × 255,118 pt |
| Jerarquía | Título, subtítulo sobrio, contenido, síntesis y fuente diferenciados |
| Separación por capítulos | Seis separadores a página completa, consistentes en color y numeración |
| Gráficos y tablas | Ejes, leyendas, cifras y categorías visibles; sin recortes laterales |
| Capturas de frontend | Ampliadas con proporción conservada y condición histórica declarada |
| Taxonomía | Diagrama legible y tabla bibliográfica complementaria en la página siguiente |
| Ensemble | Miembros, regla de promedio y alternativas comparadas en páginas consecutivas |
| Referencias | Redistribuidas de nueve a siete páginas; se eliminó una página final casi vacía |
| Compilación | 0 cajas horizontales desbordadas, 0 errores fatales y 0 citas indefinidas |

## Hallazgos y correcciones

1. La página final de referencias contenía una sola entrada. Se ajustó únicamente el tamaño tipográfico de la bibliografía a 6 pt y las 52 referencias quedaron distribuidas en siete páginas equilibradas.
2. Se redujeron espacios verticales en “Resultados principales del sistema” y “Limitaciones y trabajo futuro”; la información principal ganó aire y no se pierde contenido.
3. Se verificaron en vista individual las diapositivas de taxonomía y fundamento bibliográfico. La primera preserva el mapa visual de las cinco salidas y la segunda utiliza todo el ancho para documentar autores por dimensión.
4. El archivo de compilación conserva dos avisos internos de caja vertical menores a 2,2 pt. La inspección del PDF confirma que no producen texto, gráficos ni pies cortados; no constituyen un defecto visual observable.

## Evidencia de revisión

Las páginas rasterizadas y hojas de contacto quedan en `.audit_final_visual/`. Los grupos `contacto_01_08.png` a `contacto_41_46.png` permiten repetir la revisión panorámica; `final-03.png` y `final-37.png` registran las dos páginas ajustadas al cierre.

## Dictamen

La presentación mantiene una estética uniforme, aprovecha adecuadamente el área útil, prioriza los resultados vigentes y distingue de manera visible entre evidencia externa y resultados propios. No se identificaron recortes, deformaciones de imágenes, elementos superpuestos ni páginas temáticas con espacio desaprovechado.
