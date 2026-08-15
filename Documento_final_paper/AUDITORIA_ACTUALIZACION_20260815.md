# Auditoría de actualización del artículo · 15 de agosto de 2026

## Alcance

El artículo y sus anexos fueron reescritos para el contrato v2.1 y los artefactos vigentes de `03_07`/`03_07a`. La línea histórica se conserva solo como antecedente breve de la mejora; ninguna métrica antigua se presenta como resultado actual.

## Trazabilidad cuantitativa

- Corpus y particiones: `docs/artefactos/auditoria_estado_final_182461.json`.
- Comparación en validation: `resultados/modelos/comparacion_individual_ensemble_validation.json`.
- Selección, miembros, calibradores y umbrales: `resultados/modelos/seleccion_congelada.json`.
- Test natural y vista secundaria 4:1: `resultados/modelos/test_final_abierto_una_vez.json`.
- Síntesis y tablas: `resultados/modelos/resumen_03_07a.json` y `resultados/modelos/tablas_03_07a/`.

Se verificaron en el cuerpo y anexos los totales del corpus, los cuatro ganadores por tipo, las cinco estrategias de ensemble, las métricas por categoría, el bootstrap pareado, el estado inferencial inconcluso y la política `NEEDS_REVIEW`.

## Métodos y ensemble

El método describe los 28 candidatos individuales y cinco ensembles. Detalla la regresión logística TF--IDF, la cascada E5 v2 y Qwen3--0.6B con LoRA. El ensemble ganador se especifica mediante sus tres identificadores, promedio de scores crudos, cinco calibradores sigmoidales, cinco umbrales, compuerta binaria y razones de revisión.

## Citas y referencias

- 118 comandos de cita en las fuentes incluidas.
- 99 referencias efectivamente numeradas en el PDF.
- 0 citas indefinidas.
- 0 referencias cruzadas indefinidas.
- La tabla de taxonomía atribuye explícitamente las fuentes generales y las fuentes por categoría.
- La comparación con el estado del arte declara que las cifras externas no son un ranking directo porque cambian corpus, prevalencia y taxonomía.

## Compilación y revisión visual

- PDF A4 de 19 páginas.
- 0 errores fatales.
- 0 cajas `Overfull`.
- Inspección visual de las 19 páginas rasterizadas: tablas, fórmulas, diagramas, gráficos, notas y referencias permanecen dentro de los márgenes.
- Las capturas de etiquetado y producción son legibles. La captura de producción documenta el selector vigente y los cuatro sistemas; es una vista previa del frontend, no una inferencia usada para medir desempeño.

El PDF resultante es `paper_moderador_contenido_youtube_ieee.pdf`.
