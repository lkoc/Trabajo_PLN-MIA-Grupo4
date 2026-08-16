# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Presentación de resultados finales

Nueva versión independiente de la presentación Beamer. No modifica la carpeta Presentación_BEAMER original.

## Entregables

- presentacion_resultados_finales.pdf: presentación compilada.
- presentacion_resultados_finales.tex: fuente Beamer.
- referencias.bib: referencias citadas en las diapositivas.
- assets/: figuras de resultados y capturas usadas.
- datos_fuente/: tablas CSV copiadas desde los últimos resultados de 03_07a.
- FUENTES_Y_TRAZABILIDAD.md: correspondencia entre afirmaciones, cifras y fuentes.
- AUDITORIA_FINAL_VISUAL.md: revisión visual y gráfica de las 59 páginas renderizadas.
- AUDITORIA_FINAL_REFERENCIAS.md: control de autores, referencias y citas de la versión final.

## Compilación

Desde esta carpeta:

    latexmk -pdf -interaction=nonstopmode -halt-on-error presentacion_resultados_finales.tex

La presentación usa resultados de 03_07 y 03_07a disponibles localmente el 15 de agosto de 2026. Las capturas documentan los entornos de etiquetado y producción; la de producción refleja el selector v2.1 con clásico, Transformer, Qwen y ensemble, pero no se usa como evidencia de desempeño.

La versión compilada contiene 59 páginas: portada, ocho separadores de capítulo, 43 diapositivas temáticas —incluidos el índice, un capítulo inicial de estado del arte y enfoques, la estadística descriptiva del corpus, la explicación diferenciada de los entrenamientos Qwen, la explicación de métricas, la comparación externa y los enlaces operativos— y siete páginas de referencias.

Los botones de conclusiones abren los servicios locales activos en
<http://127.0.0.1:8765/> (etiquetado) y <http://127.0.0.1:8876/> (producción).
