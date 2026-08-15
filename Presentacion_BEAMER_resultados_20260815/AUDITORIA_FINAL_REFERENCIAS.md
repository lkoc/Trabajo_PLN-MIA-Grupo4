# Auditoría final de autores, referencias y citado

**Fecha:** 15 de agosto de 2026  
**Artefactos auditados:** `presentacion_resultados_finales.tex`, `referencias.bib`, `presentacion_resultados_finales.bbl` y PDF compilado  
**Estado:** aprobada

## Resultado cuantitativo

| Control | Resultado |
|---|---:|
| Llamadas `\cite{...}` | 77 |
| Claves bibliográficas únicas citadas | 52 |
| Entradas incluidas en la bibliografía final | 52 |
| Claves citadas ausentes de `referencias.bib` | 0 |
| Citas indefinidas en LaTeX | 0 |
| Advertencias de BibTeX | 0 |

## Taxonomía

La diapositiva “Taxonomía de clasificación” ahora identifica de forma explícita a los autores del marco general y de la adaptación peruana. La diapositiva complementaria “Fundamento bibliográfico de la taxonomía” documenta 18 fuentes únicas y separa el sustento de cada dimensión:

| Dimensión | Autores o fuentes declarados |
|---|---|
| Marco general | Waseem et al.; Banko et al. |
| Racismo / discriminación | Callirgos; Zavala y Zariquiey; Zavala y Back; Brañez; Almeida y Zavala; Salem; Vich |
| Género / identidad | Rodríguez-Sánchez et al.; Zeinert, Inie y Derczynski; Albornoz y Flores; Defensoría del Pueblo; Chakravarthi et al.; Rottenbacher; Lovón-Cueva y Lovón-Cueva |
| Acoso / amenaza | Waseem et al.; Wulczyn, Thain y Dixon; Banko et al.; Defensoría del Pueblo |
| Contenido sexual | Banko et al.; Zeinert, Inie y Derczynski; Albornoz y Flores; Defensoría del Pueblo; política de YouTube |

Se incorporó una precisión metodológica necesaria: esas fuentes sustentan las dimensiones de daño, pero los nombres, fusiones, exclusiones y cinco salidas finales son decisiones operativas del proyecto. Por tanto, no se atribuye a un autor externo una taxonomía exacta que no formuló.

## Correspondencia entre afirmaciones y fuentes

- Las afirmaciones de estado del arte se presentan como comparación contextual y conservan autor, corpus y métrica; no se plantea equivalencia directa entre particiones con distinta prevalencia.
- Los métodos de modelos clásicos, Transformers, Qwen, LoRA, ensembles, calibración, bootstrap y clasificación selectiva tienen citas próximas a la afirmación correspondiente.
- Las cifras del corpus, validación, test natural y política de revisión se rotulan como “Resultado propio” y remiten a los artefactos del run 03_07/03_07a.
- Las capturas de los frontends se identifican como evidencia interna y, cuando corresponde, como capturas históricas; no se usan como respaldo bibliográfico externo.
- Las conclusiones cuantitativas reproducen los valores mostrados en las diapositivas de resultados: BA 0,840 y macro-AUPRC 0,555 en validación; BA 0,846 en test natural; BA 0,940 con 65,2 % de cobertura automática.

## Controles nominales

Se contrastaron los nombres mostrados en la nueva tabla taxonómica con las entradas de `referencias.bib` y con las secciones `02_bases.tex` y `04_metodologia_datos.tex` del artículo. Se verificaron en particular las autorías compuestas susceptibles de abreviarse o confundirse: Zavala–Zariquiey, Zavala–Back, Almeida–Zavala, Lovón-Cueva–Lovón-Cueva y Zeinert–Inie–Derczynski.

## Dictamen

La presentación tiene cobertura bibliográfica suficiente para sus afirmaciones externas, diferencia fuentes académicas, institucionales e internas, y compila sin citas pendientes. La taxonomía queda ahora trazable por categoría y con una atribución prudente de la contribución propia.
