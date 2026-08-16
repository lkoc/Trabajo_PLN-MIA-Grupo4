# Auditoría final visual y gráfica

**Fecha:** 15 de agosto de 2026  
**Artefacto auditado:** `presentacion_resultados_finales.pdf`  
**Estado:** aprobada para exposición

## Alcance

Se revisaron las 58 páginas renderizadas de la presentación, incluidos portada, ocho separadores de capítulo, 42 diapositivas temáticas y siete páginas de referencias. La revisión se hizo sobre imágenes rasterizadas del PDF final, no únicamente sobre el código LaTeX.

## Comprobaciones realizadas

| Criterio | Resultado |
|---|---|
| Formato | 16:9 uniforme, 453,543 × 255,118 pt |
| Jerarquía | Título, subtítulo sobrio, contenido, síntesis y fuente diferenciados |
| Índice | Siete capítulos de contenido visibles en una sola página, con jerarquía y colores coherentes |
| Separación por capítulos | Ocho separadores a página completa, consistentes en color y numeración |
| Gráficos y tablas | Ejes, leyendas, cifras y categorías visibles; sin recortes laterales |
| Estadística del dataset | Alcance inicial y universo efectivo diferenciados; canales principales, concentración, distribución y dificultades visibles |
| Métricas | Definiciones, dirección, lectura de valores y motivos de exclusión de métricas alternativas antes del ranking |
| Estado del arte | Capítulo metodológico inicial y capítulo posterior de comparación externa; métodos, papers, gráfica, muestreo y análisis crítico legibles |
| Capturas de frontend | Ampliadas con proporción conservada y condición histórica declarada |
| Taxonomía | Diagrama legible y tabla bibliográfica complementaria en la página siguiente |
| Ensemble | Miembros, regla de promedio y alternativas comparadas en páginas consecutivas |
| Referencias | Redistribuidas de nueve a siete páginas; se eliminó una página final casi vacía |
| Enlaces operativos | Botones visibles en conclusiones; etiquetado y producción respondieron HTTP 200 |
| Compilación | 0 cajas desbordadas, 0 errores fatales, 0 citas indefinidas y 58 páginas generadas |

## Hallazgos y correcciones

1. Se incorporó un índice inmediatamente después de la portada y se retiró de esa página cualquier bloque inferior que invadiera el pie.
2. Las gráficas horizontales ocultaban sus etiquetas al desactivar el eje vertical. Se restituyeron los nombres de clases, canales, tipos de modelo y papers, y se redujo el ancho de los ejes para conservar los márgenes.
3. La radiografía del corpus distingue 182 461 chunks/5 385 videos/322 canales del alcance inicial frente a 173 240/4 906/276 efectivos; no se mezclan universos.
4. El estado del arte se adelantó: la página 8 abre el capítulo y las páginas 9--12 explican métodos, benchmarks, explicabilidad y muestreo antes de presentar los datos propios. Las páginas 40--43 quedan reservadas para el contraste posterior con resultados.
5. Las diapositivas nuevas de métricas y estado del arte se revisaron individualmente: fórmulas, tablas, subtítulos y notas permanecen dentro del área útil.
6. Las conclusiones muestran dos controles clicables sin desplazar la tabla ni el bloque de síntesis.
7. Las 52 referencias continúan distribuidas en siete páginas y no se añadió una página final residual.

## Evidencia de revisión

Las páginas rasterizadas y hojas de contacto quedan en `.audit_final_visual/`. Los grupos `contacto_01_08.png` a `contacto_57_58.png` permiten repetir la revisión panorámica; los archivos `slide-01.png` a `slide-58.png` permiten inspección individual.

## Dictamen

La presentación mantiene una estética uniforme, aprovecha adecuadamente el área útil, prioriza los resultados vigentes y distingue de manera visible entre evidencia externa y resultados propios. No se identificaron recortes, deformaciones de imágenes, elementos superpuestos ni páginas temáticas con espacio desaprovechado.
