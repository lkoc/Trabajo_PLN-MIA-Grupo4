# Auditoría de etiquetas finas y transversales

Fecha: 2026-07-29T13:34:15-05:00

Se analizaron 13 modelos sobre el mismo test de 5,290 chunks. Las etiquetas finas se usaron para desagregar recall o especificidad de `SEGURO`; los flags transversales midieron cuánto captura una revisión ordenada por cercanía a los umbrales.

Las etiquetas finas y flags **no se usaron como predictores gold**. Qwen `04_205` sí las usa como supervisión auxiliar multitararea y `04_206` consume sus logits auxiliares; los demás son controles `coarse-only`. Por tanto, una diferencia entre esos regímenes no debe atribuirse exclusivamente a la arquitectura sin una ablación específica.

La referencia Qwen plana corresponde a la época operativa **3**, elegida en validation sin consultar test.

- Tabla fina: `resultados/metricas/auditoria_auxiliar_modelos_4/desempeno_por_etiqueta_fina.csv`
- Tabla de flags: `resultados/metricas/auditoria_auxiliar_modelos_4/captura_flags_por_incertidumbre.csv`
- Resultados pendientes: ninguno.

## Conclusión

Esta auditoría sirve para detectar debilidades por fenómeno fino y para estudiar qué casos capturaría una cola de revisión. Sus diferencias son descriptivas: no sustituyen la comparación principal de cuatro daños, no prueban causalmente el beneficio de la supervisión auxiliar y no autorizan despliegue autónomo.
