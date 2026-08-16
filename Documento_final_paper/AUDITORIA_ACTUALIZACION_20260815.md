# Auditoría de actualización del artículo · 15 de agosto de 2026

## Alcance

El artículo y sus anexos quedaron actualizados al contrato v2.1 y a los artefactos vigentes de `03_07`/`03_07a`. Los intentos anteriores aparecen solo como antecedente breve de la mejora metodológica; sus métricas no se mezclan con la comparación actual.

La auditoría integral contra las guías generales y específicas, con matriz de hallazgos, está en `AUDITORIA_CITAS_Y_ESTILO.md`.

## Trazabilidad cuantitativa

- Corpus y particiones: `docs/artefactos/auditoria_estado_final_182461.json`.
- Selección de la ventana temporal: `docs/OPTIMIZACION_LONGITUD_CHUNKS.md` y `docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md`.
- Comparación en validation: `resultados/modelos/comparacion_individual_ensemble_validation.json`.
- Selección, miembros, calibradores y umbrales: `resultados/modelos/seleccion_congelada.json`.
- Test natural y vista secundaria 4:1: `resultados/modelos/test_final_abierto_una_vez.json`.
- Síntesis y tablas: `resultados/modelos/resumen_03_07a.json` y `resultados/modelos/tablas_03_07a/`.

Se verificaron el alcance inicial (182 461 chunks, 5 385 videos y 322 canales), el corpus elegible, la concentración por canal, los cuatro ganadores por tipo, las cinco estrategias de ensemble, las métricas por categoría, el bootstrap pareado, el estado inferencial inconcluso y la política `NEEDS_REVIEW`.

## Métodos, ensemble e integración

El método describe los 28 candidatos individuales y cinco ensembles. Detalla la regresión logística TF--IDF, la cascada E5 v2 y Qwen3--0.6B con LoRA. El ensemble ganador se especifica mediante sus tres identificadores, promedio de scores crudos, cinco calibradores sigmoidales, cinco umbrales, compuerta binaria y razones de revisión.

Las rutas Qwen se separan técnicamente: LoRA base a 128 tokens, continuación LoRA a 256, tres continuaciones LoRA con penalización estructural y una corrida histórica de ajuste completo sin LoRA. Se documentan inicialización, épocas, tasas, rango, matrices adaptadas y resultados. El LoRA base es el mejor Qwen y el miembro congelado del ensemble; la continuación a 256 solo lidera categorías específicas. También se declaran la escala exacta de Qwen y MiniLM, la ausencia de cuantización y el uso de BF16 mixto en A100.

La BA de `ANY_DAMAGE` OOF a cobertura completa queda identificada como métrica primaria; macro-AUPRC de daños actúa como salvaguarda y F1 describe decisiones ya umbralizadas. El texto explica por qué exactitud ordinaria, micro-F1, ROC-AUC o una suma de indicadores no gobiernan el ranking. El desempate $R_{0,67}$ se define como $0,67\,\mathrm{FNR}+0,33\,\mathrm{FPR}$, con menor valor preferible y un peso aproximado 2,03 veces mayor para la omisión de daño.

La integración ya no es solo una propuesta: clásico, Transformer, Qwen y ensemble respondieron a solicitudes reales de la API local sobre CPU. El despliegue permanece en modo sombra y no autoriza sanciones automáticas.

## Citas y referencias

- 125 comandos de cita y 208 apariciones de claves en las 18 fuentes TeX compiladas.
- 100 referencias numeradas en el PDF.
- 0 citas indefinidas, 0 claves duplicadas y 0 referencias cruzadas indefinidas.
- Las 33 etiquetas de figuras, tablas y apéndices tienen una referencia entrante.
- La taxonomía atribuye fuentes generales y fuentes por categoría.
- La comparación con el estado del arte declara que las cifras externas no forman un ranking directo.
- DETOXIS, HatEval, OffendES, EXIST, HateXplain y NaijaHate se describen por ámbito, datos, tarea, resultado y diferencia con el proyecto.

## Compilación y revisión visual

- PDF A4 de 23 páginas.
- 0 errores fatales y 0 cajas `Overfull`.
- Inspección visual de las 23 páginas rasterizadas: 17 tablas, 11 figuras, fórmulas, diagramas, gráficos, capturas y referencias permanecen dentro de los márgenes.
- Los apéndices A--E comienzan en página nueva; se eliminó una página casi vacía en la transición desde las conclusiones.
- La captura de producción documenta el selector vigente y los cuatro sistemas; la inferencia funcional se verificó además contra la API real.

El entregable es `paper_moderador_contenido_youtube_ieee.pdf`.
