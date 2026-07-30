# Guías generales para artículos IEEE y presentaciones

Este directorio convierte las guías editoriales desarrolladas en el proyecto en un recurso general, reutilizable con cualquier LLM y aplicable a otros temas. La versión operativa está empaquetada como una skill; los hechos, cifras, taxonomías, modelos y conclusiones del proyecto original no forman parte de sus reglas generales.

## Punto de entrada

Use [SKILL.md](redactar-articulo-ieee-y-presentacion/SKILL.md) como instrucción principal. La skill dirige al LLM hacia referencias breves según la tarea y cubre:

- definición del contrato editorial y de la fuente de verdad;
- formulación del problema, preguntas, objetivos y contribuciones;
- estructura y redacción de un artículo IEEE;
- metodología, experimentos, resultados, discusión y validez;
- búsqueda, verificación y trazabilidad de citas;
- búsquedas bibliográficas profundas con cadenas booleanas y expansión recursiva;
- tablas, figuras, diagramas y ontologías;
- preparación de una presentación académica;
- auditoría científica, de título, citas, estilo, recursos visuales y reproducibilidad.

La carpeta sigue una estructura compatible con sistemas de skills:

    Guias_generales/
    ├── INDICE_GUIAS.md
    └── redactar-articulo-ieee-y-presentacion/
        ├── SKILL.md
        ├── agents/openai.yaml
        └── references/

## Uso con cualquier LLM

Si la plataforma admite skills, copie o registre la carpeta [redactar-articulo-ieee-y-presentacion](redactar-articulo-ieee-y-presentacion/). Si solo admite contexto o archivos adjuntos:

1. entregue primero [SKILL.md](redactar-articulo-ieee-y-presentacion/SKILL.md);
2. adjunte las referencias que la tarea requiera;
3. proporcione la plantilla de la convocatoria, el manuscrito y los artefactos verificables;
4. pida al modelo que declare evidencia faltante en vez de completarla por inferencia.

Prompt mínimo:

> Aplica la guía redactar-articulo-ieee-y-presentacion. Trabaja únicamente con evidencia verificable, separa fuentes externas, artefactos internos y decisiones locales, y entrega diagnóstico, cambios, validaciones y pendientes.

Ejemplos de encargo:

- «Convierte este proyecto terminado en un artículo IEEE de ocho páginas».
- «Audita si cada resultado y afirmación teórica de este manuscrito tiene respaldo».
- «Crea una presentación de doce minutos basada únicamente en la versión final del paper».
- «Revisa todos los diagramas para evitar cruces, texto ilegible y conclusiones no sustentadas».

## Referencias generalizadas

| Recurso | Cuándo usarlo |
|---|---|
| [Contrato y trazabilidad](redactar-articulo-ieee-y-presentacion/references/contrato-y-trazabilidad.md) | Al iniciar, resolver conflictos de versiones o construir matrices de evidencia |
| [Estructura y redacción IEEE](redactar-articulo-ieee-y-presentacion/references/estructura-y-redaccion-ieee.md) | Al planificar o corregir el manuscrito |
| [Metodología, resultados y validez](redactar-articulo-ieee-y-presentacion/references/metodologia-resultados-y-validez.md) | Al documentar diseño, datos, experimentos, resultados y límites |
| [Búsqueda bibliográfica profunda](redactar-articulo-ieee-y-presentacion/references/busqueda-bibliografica-profunda.md) | Al construir search strings, ampliar vocabulario y cerrar huecos de evidencia |
| [Evidencia, citas y bibliografía](redactar-articulo-ieee-y-presentacion/references/evidencia-citas-y-bibliografia.md) | Al buscar, citar, parafrasear o auditar fuentes |
| [Figuras, tablas y ontologías](redactar-articulo-ieee-y-presentacion/references/figuras-tablas-y-ontologias.md) | Al convertir relaciones complejas en recursos visuales |
| [Presentación académica](redactar-articulo-ieee-y-presentacion/references/presentacion-academica.md) | Al derivar Beamer, diapositivas o guion oral |
| [Auditoría y entrega](redactar-articulo-ieee-y-presentacion/references/auditoria-y-entrega.md) | Antes de someter, publicar o exponer |
| [Procedencia](PROCEDENCIA_GUIAS_PROYECTO.md) | Para rastrear qué documentos del proyecto originaron cada regla |

La [auditoría y entrega](redactar-articulo-ieee-y-presentacion/references/auditoria-y-entrega.md) contiene una rúbrica de título y controles separados para validez científica, citas, estilo, recursos visuales y reproducibilidad. La [procedencia](PROCEDENCIA_GUIAS_PROYECTO.md) aplica esa rúbrica al título del paper de este proyecto.

## Guías originales del proyecto

Las reglas se abstrajeron principalmente de:

- [Guía de estructura del paper IEEE](../Documento_final_paper/guia_estructura_paper_ieee.md).
- [Guía de redacción y control de evidencia](../Documento_final_paper/guia_redaccion_paper_ieee.md).
- [Auditoría final de citas, redacción y gráficos](../Documento_final_paper/AUDITORIA_CITAS_Y_ESTILO.md).
- [Guía de la presentación Beamer](../Presentación_BEAMER/README.md).
- [Guía de figuras reproducibles](../Documento_final_paper/figuras/README.md).
- [Auditoría académica de una taxonomía de dominio](../para_equiquetado_LLM/AUDITORIA_ACADEMICA_TAXONOMIA.md), usada solo para generalizar el método de trazabilidad conceptual.

Consulte [PROCEDENCIA_GUIAS_PROYECTO.md](PROCEDENCIA_GUIAS_PROYECTO.md) para ver qué se conservó, qué se generalizó y qué se excluyó por ser específico del caso. Este archivo queda fuera de la skill para que el paquete pueda copiarse a otro proyecto sin enlaces externos ni ejemplos que condicionen al LLM.

## Regla de adaptación

Las instrucciones vigentes de la revista, conferencia, institución o curso tienen prioridad sobre esta guía. IEEE no es una sola convocatoria: se debe comprobar la plantilla, el límite de páginas, el proceso de revisión, el anonimato, el estilo de referencias, los anexos y las políticas sobre datos, código y uso de IA para el destino concreto.
