# Procedencia de las guías generalizadas

## Propósito

Este archivo mantiene trazabilidad hacia los documentos que originaron la skill. Las referencias enlazadas contienen decisiones y cifras de un proyecto concreto; deben consultarse como antecedente metodológico, no copiarse como hechos en otro artículo. Se mantiene fuera de la carpeta de la skill para que esta pueda instalarse o copiarse de forma independiente.

## Mapa de procedencia

| Guía del proyecto | Aporte abstraído | Recurso general |
|---|---|---|
| [guia_estructura_paper_ieee.md](../Documento_final_paper/guia_estructura_paper_ieee.md) | Hilo narrativo, relación problema–objetivo–método–resultado, estructura IEEE, DSR, visualizaciones y revisión final | [estructura-y-redaccion-ieee.md](redactar-articulo-ieee-y-presentacion/references/estructura-y-redaccion-ieee.md), [metodologia-resultados-y-validez.md](redactar-articulo-ieee-y-presentacion/references/metodologia-resultados-y-validez.md) |
| [guia_redaccion_paper_ieee.md](../Documento_final_paper/guia_redaccion_paper_ieee.md) | Prosa directa, trazabilidad de afirmaciones, fuentes primarias, prevención de plagio, resultados prudentes y dos pases | [evidencia-citas-y-bibliografia.md](redactar-articulo-ieee-y-presentacion/references/evidencia-citas-y-bibliografia.md), [auditoria-y-entrega.md](redactar-articulo-ieee-y-presentacion/references/auditoria-y-entrega.md) |
| [AUDITORIA_CITAS_Y_ESTILO.md](../Documento_final_paper/AUDITORIA_CITAS_Y_ESTILO.md) | Reporte verificable de cobertura bibliográfica, compilación y revisión visual | [auditoria-y-entrega.md](redactar-articulo-ieee-y-presentacion/references/auditoria-y-entrega.md) |
| [README del paper](../Documento_final_paper/README.md) | Fuentes de verdad, compilación, afirmaciones prohibidas y alineación con presentación | [contrato-y-trazabilidad.md](redactar-articulo-ieee-y-presentacion/references/contrato-y-trazabilidad.md), [auditoria-y-entrega.md](redactar-articulo-ieee-y-presentacion/references/auditoria-y-entrega.md) |
| [README de figuras](../Documento_final_paper/figuras/README.md) | Recursos reproducibles, rutas ortogonales y revisión de solapamientos | [figuras-tablas-y-ontologias.md](redactar-articulo-ieee-y-presentacion/references/figuras-tablas-y-ontologias.md) |
| [README del Beamer](../Presentación_BEAMER/README.md) | Presentación derivada del paper, narrativa visual y revisión científica/editorial | [presentacion-academica.md](redactar-articulo-ieee-y-presentacion/references/presentacion-academica.md) |
| [AUDITORIA_ACADEMICA_TAXONOMIA.md](../para_equiquetado_LLM/AUDITORIA_ACADEMICA_TAXONOMIA.md) | Separación entre concepto académico, evidencia contextual, política y regla local; auditoría de definiciones | [contrato-y-trazabilidad.md](redactar-articulo-ieee-y-presentacion/references/contrato-y-trazabilidad.md), [figuras-tablas-y-ontologias.md](redactar-articulo-ieee-y-presentacion/references/figuras-tablas-y-ontologias.md) |

## Generalizaciones realizadas

- DSR pasó de ser una obligación del caso a una opción condicionada al tipo de contribución.
- Las familias de modelos concretas se sustituyeron por una matriz general de alternativas y evidencia.
- La taxonomía de moderación se convirtió en un método general para auditar vocabularios operativos.
- Los conteos, métricas, categorías y decisiones del proyecto se excluyeron de las reglas reutilizables.
- El control de tiempos entre almacenamiento local y nube se generalizó como comparación de versiones con zona horaria explícita.
- La revisión de paper y Beamer se convirtió en un control aplicable a cualquier manuscrito y presentación.

## Ampliación sobre búsqueda bibliográfica

La guía [busqueda-bibliografica-profunda.md](redactar-articulo-ieee-y-presentacion/references/busqueda-bibliografica-profunda.md) se añadió como una capacidad general posterior. Integra la solicitud de construir search strings con AND, OR, NOT y paréntesis, y la apoya en documentación oficial de [IEEE Xplore](https://ieeexplore.ieee.org/Xplorehelp/searching-ieee-xplore/search-tips), [Scopus](https://service.elsevier.com/app/answers/detail/a_id/11365/supporthub/scopus/), [Web of Science](https://webofscience.help.clarivate.com/Content/search-operators.html) y [PubMed](https://pubmed.ncbi.nlm.nih.gov/help/). Para búsquedas sistemáticas incorpora los controles de reporte de [PRISMA-S](https://pmc.ncbi.nlm.nih.gov/articles/PMC8270366/) y la revisión de estrategias de [PRESS 2015](https://pubmed.ncbi.nlm.nih.gov/27005575/).

## Elementos deliberadamente no transferidos

No usar como plantilla factual:

- nombres de autores, institución o correos;
- tamaño o procedencia del corpus;
- categorías de moderación;
- modelos, checkpoints o hiperparámetros;
- métricas y resultados;
- enlaces de despliegue;
- conclusiones sobre producción;
- licencias particulares.

Cada nuevo proyecto debe reconstruir esos elementos desde sus propios artefactos y fuentes.

## Evaluación del título actual del proyecto de origen

Título auditado:

> Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

Aplicando la rúbrica general:

| Criterio | Evaluación |
|---|---|
| Problema o fenómeno | Sí: identifica moderación de videos |
| Artefacto o enfoque | Sí: moderación semiautomática mediante modelos de PLN |
| Método o tecnología | Sí: engloba modelos clásicos, deep learning, Transformers y ajuste fino como modelos clásicos y neuronales de procesamiento del lenguaje natural |
| Objeto y contexto | Sí: videos peruanos de YouTube |
| Propósito y operación | Sí: clasificación para apoyar una decisión supervisada |
| Alcance prudente | Sí: «semiautomática» evita prometer decisión autónoma |
| Recuperación | Sí: contiene moderación, YouTube, procesamiento del lenguaje natural y supervisión humana |
| Concisión | Es largo, pero evita el acrónimo PLN y omite detalles que pertenecen al método |
| Fidelidad | Sí, según el manuscrito y el artefacto descritos |

Conclusión: el título cumple los criterios principales y hace explícita la amplitud técnica mediante una categoría metodológica correcta: «modelos clásicos y neuronales de procesamiento del lenguaje natural». No necesita enumerar machine learning, deep learning, Transformers y ajuste fino como si fueran niveles equivalentes, ni añadir DSR. Los subtítulos y la auditabilidad se conservan en el resumen y el cuerpo como delimitación de entrada y propiedad secundaria, respectivamente.
