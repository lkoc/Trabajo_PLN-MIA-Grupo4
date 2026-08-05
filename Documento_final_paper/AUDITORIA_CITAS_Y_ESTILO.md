# Auditoría final de citas, redacción y gráficos

> **Enmienda 2026-08-05.** El cierre descrito abajo acredita la versión publicada del baseline de cuatro daños con `SEGURO` derivado. El contrato activo v2 entrena cinco salidas y conserva ese baseline en `archivo/`; la compilación, la inspección visual y los conteos posteriores a esta enmienda se registran al final de este documento.

Fecha de cierre: 2026-07-29
Alcance: artículo IEEE, Beamer, bibliografía, figuras, guías editoriales, contrato v2 y paquete histórico `archivo/taxonomia_v1_3/para_equiquetado_LLM/`.

## Resultado verificable

- 117 claves citadas y 117 entradas en `referencias.bib`.
- 0 claves citadas sin BibTeX y 0 entradas BibTeX sin uso.
- 81 PDF oficiales o de acceso abierto validados (76,21 MiB) y 36 fuentes sin PDF local, todas explicadas en `../referencias_y_descargas/indice_referencias.csv`.
- Paper compilado en 22 páginas con `IEEEtran`, modo conferencia y hoja A4; Beamer compilado en 23 diapositivas 16:9.
- Tamaño físico del paper verificado con `pdfinfo`: 595,276 × 841,89 pt, equivalente a 210 × 297 mm.
- 0 errores LaTeX, 0 citas o referencias indefinidas y 0 cajas `Overfull` en los dos registros finales.
- Bibliografía final equilibrada mediante `\IEEEtriggeratref{106}`, sin aplicar `\balance` a las cuatro páginas completas de referencias.
- Las obras con más de tres autores se abrevian mediante `IEEEtranBSTCTL`: tres autores y «et al.», sin recortar la autoría almacenada en `referencias.bib`.
- Los apéndices A--F reciben una referencia explícita desde el pasaje correspondiente del cuerpo.

Los avisos `Underfull` que permanecen provienen sobre todo de identificadores monoespaciados, URL y celdas estrechas. Se inspeccionó el PDF para comprobar que no producen texto cortado ni elementos fuera de página.

## Criterio de citas

La revisión aplicó cuatro reglas:

1. cada algoritmo o familia usada se enlaza con su aporte fundacional o documentación primaria;
2. cada definición teórica relevante se apoya en literatura pertinente;
3. las decisiones propias —chunking, unión 5→4, umbrales, flags y consenso 2 de 3— se declaran como decisiones del proyecto y no se atribuyen a las fuentes;
4. una fuente de plataforma, una tarjeta de modelo y un estudio académico se distinguen por su función y no se presentan como autoridades intercambiables.

El inventario reproducible está en `../referencias_y_descargas/`. No se descargaron copias de procedencia dudosa ni se eludieron barreras de pago.

## Auditoría del título

El título final es «Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural». Se revisó por necesidad o fenómeno, artefacto/enfoque, metodología cuando aporta identificación, objeto y contexto, propósito, alcance prudente, recuperación bibliográfica, concisión y fidelidad. «Semiautomática» ya expresa que el proceso no es enteramente autónomo; la supervisión humana se mantiene explícita en el resumen, el método, los resultados y el alcance operativo.

Cumple los criterios principales: delimita la moderación de videos peruanos de YouTube, engloba aprendizaje automático clásico, modelos Transformer y ajuste fino bajo «modelos clásicos y neuronales de procesamiento del lenguaje natural», y conserva la supervisión humana. La entrada basada en subtítulos se precisa en el resumen y el método. DSR no aparece porque el título ya comunica la amplitud técnica relevante. «Auditable» se retiró al no ser el foco principal, aunque la trazabilidad se mantiene como propiedad del artefacto en el cuerpo.

La forma final se sincronizó en el fuente y los metadatos PDF del paper, la portada y los metadatos del Beamer, los README y las guías que auditan el título. Después de compilar se inspeccionaron visualmente ambas portadas: el texto permanece dentro de sus márgenes, sin recortes ni solapamientos.

## Procedimiento reproducible de auditoría

La auditoría de citas y estilo debe producir una tabla con ubicación, hallazgo, severidad, evidencia, corrección y estado, y cubrir:

1. ideas, definiciones, estado del arte, algoritmos y arquitecturas;
2. fuentes exactas de datasets, checkpoints, software, estándares y políticas;
3. correspondencia entre cada proposición y la fuente vecina;
4. autores, título, año, sede, DOI/URL, claves ausentes y duplicados;
5. paráfrasis propia, citas textuales y procedencia de figuras adaptadas;
6. ruta de cada cifra al artefacto, campo, split, versión y criterio;
7. identificación expresa de decisiones locales;
8. lenguaje directo, consistencia terminológica, siglas, tiempos verbales, ortografía y prudencia;
9. captions, referencias cruzadas, legibilidad y ausencia de líneas sobre cajas o texto.

La auditoría numérica usa dos cifras significativas para métricas y magnitudes estimadas en el paper y el Beamer. Los cálculos y comparaciones conservan la precisión completa del artefacto fuente. No se redondean conteos exactos, tamaño muestral, años, versiones, identificadores, hashes ni parámetros fijados por protocolo; las constantes físicas pueden mantener la precisión que requiera su uso.

La auditoría del cierre conserva una matriz interna que vincula el objetivo general y cada objetivo específico con su evidencia, pero exige una conclusión publicada narrativa: sin la palabra «objetivo», sin códigos `O1`/`O2` y sin listas de cumplimiento. Para cada distancia entre la situación inicial y la deseada debe indicarse si se eliminó, se redujo o permanece; lo pendiente debe enlazarse con una acción, recomendación o trabajo futuro concreto.

La disponibilidad publicada solo identifica qué datos, cuadernos, scripts y artefactos se ofrecen y el repositorio donde se encuentran. Los SHA, commits y hashes concretos se conservan en manifiestos técnicos fuera del cuerpo.

Se considera crítico inventar una fuente, dejar una afirmación central sin respaldo o publicar un resultado sin origen; alto, citar una fuente que no sostiene la afirmación; medio, mantener metadatos o alcances ambiguos; y bajo, conservar una inconsistencia formal. El cierre debe reportar conteos y excepciones verificables.

## Taxonomía y paquete de etiquetado

La revisión histórica completa está en `../archivo/taxonomia_v1_3/para_equiquetado_LLM/AUDITORIA_ACADEMICA_TAXONOMIA.md`; el contrato activo está en `../docs/TAXONOMIA_V2.md`. La formulación defendible es **taxonomía operativa multietiqueta informada por literatura y fuentes institucionales pertinentes al Perú**, no taxonomía legal ni validada por un panel experto.

El contrato conserva 14 etiquetas finas: 12 fenómenos de daño y dos estados seguros; además, tres flags transversales. La implementación v2 aprende `SEGURO` y cuatro daños como cinco salidas; `ACOSO_AMENAZA` une ataque personal y amenaza directa para aumentar soporte sin tratarlas como equivalentes semántica o jurídicamente.

La auditoría también documenta diferencias entre versiones sobre amenaza explícita o implícita, ataque aislado o acoso repetido, humor encubridor, citas de discurso dañino y el caso indeterminado con `contexto_necesario`. El artículo conserva esas reglas como procedencia y las separa de las definiciones académicas.

## Revisión visual

Se inspeccionaron las diez figuras del paper y las diapositivas que las reutilizan. Se corrigieron:

- el solapamiento entre estado seguro y flags en la taxonomía;
- los rótulos sobre cajas en el embudo de construcción del corpus;
- las relaciones que atravesaban nodos en la ontología;
- las diagonales y abanicos innecesarios en familias, despliegue y flujo de datos;
- el recorte del contador del Beamer.

Las versiones finales usan rutas horizontales/verticales y carriles externos. No se observaron líneas encima de cajas, texto o leyendas en los PDF compilados.

## Límites que deben conservarse al publicar

- No existe todavía una adjudicación documentada por un panel experto peruano ni acuerdo interanotador calculable.
- Las 60 salidas CGT de la carpeta de etiquetado son una prueba consecutiva de formato, no una muestra aleatoria ni un conjunto gold.
- El test enriquecido 4:1 no estima prevalencia de producción y ya se consulta en iteraciones preliminares.
- Deben cerrarse licencia del corpus, revisión ética, versiones exactas de adquisición y evaluación prospectiva antes de una publicación externa de datos.
- La celda destructiva del cuaderno histórico quedó aislada en `archivo/`; el frontend activo carga campañas versionadas y no borra el paquete de etiquetado.

Esta fue una auditoría estructural, bibliográfica y visual. No se usó un detector externo de plagio ni de texto generado por IA; por tanto, no se promete el resultado de esos detectores. La protección aplicable es la trazabilidad de afirmaciones, la paráfrasis propia, la atribución explícita y la declaración de límites.

## Adenda de contrato v2.1 — 2026-08-05

Se auditó el nombre y el sustento de las cuatro categorías de daño contra los PDF locales. La salida de género e identidad pasa a denominarse `ATAQUE_POR_GENERO_IDENTIDAD`: el prefijo expresa daño sin reducir el fenómeno a acoso ni exigir intención de odio. Se verificaron como base peruana los trabajos adjuntos de Albornoz y Flores, la Defensoría del Pueblo, Lovón-Cueva y Lovón-Cueva y Rottenbacher; para racismo y lenguaje se contrastaron Zavala y Almeida, Brañez, Salem y Vich. `ACOSO_AMENAZA` conserva la fusión únicamente como decisión de soporte y `CONTENIDO_SEXUAL` distingue evidencia peruana de no consentimiento/cosificación frente a la frontera de contenido explícito tomada de política de plataforma. La matriz ampliada está en `../docs/MATRIZ_EVIDENCIA_TAXONOMIA.md`.

Tras el cambio se recompilaron desde cero el paper A4 (28 páginas) y el Beamer (24 diapositivas). Los logs no contienen citas o referencias indefinidas ni cajas desbordadas; se revisaron visualmente la figura taxonómica y la transición de contrato.
