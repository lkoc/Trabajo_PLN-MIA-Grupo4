# Fuentes y trazabilidad de la presentación

## Regla de interpretación

Las métricas propias provienen de los artefactos más recientes de 03_07 y 03_07a. Las comparaciones con otros autores son contextuales: los corpus, idiomas, plataformas, prevalencias, categorías y particiones no son idénticos.

| Afirmación o cifra | Fuente primaria | Uso en la presentación |
|---|---|---|
| Alcance inicial: 182 461 chunks, 5 385 videos y 322 canales; dataset efectivo: 173 240 chunks, 4 906 videos y 276 canales | docs/artefactos/auditoria_estado_final_182461.json | Diferencia entre universo observado, exclusiones y dataset final |
| 159 077 chunks SEGURO; 14 163 con daño; 17 052 asignaciones; 2 709 chunks multietiqueta | docs/artefactos/auditoria_estado_final_182461.json | Desbalance y carácter multietiqueta |
| Canales principales, media/mediana, máximos y concentración top-1/top-5/top-10 | docs/artefactos/auditoria_estado_final_182461.json | Estadística descriptiva y riesgo de concentración por fuente |
| Comparación de 15/20/25/30/35 s y decisión de conservar 30 s | docs/OPTIMIZACION_LONGITUD_CHUNKS.md y docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md | Procedimiento breve de optimización del tamaño de chunk |
| 55 966 eventos trazables; panel congelado de 16 694; cobertura combinada de 10,18 % | docs/artefactos/auditoria_estado_final_182461.json | Metodología de etiquetado y auditoría |
| 28 candidatos individuales y 5 ensembles; ganador ensemble_soft_mean | resultados/modelos/comparacion_individual_ensemble_validation.json | Ranking y selección global |
| Composición y métricas de promedio simple, promedio ponderado, unión, mayoría e intersección | resultados/modelos/comparacion_individual_ensemble_validation.json y src/moderacion_peru/ensemble_evaluation.py | Diapositiva dedicada a estrategias de ensemble; copia tabular en datos_fuente/ensembles_validation.csv |
| Mejores representantes clásico, Transformer, Qwen y ensemble | resultados/modelos/mejores_por_tipo_validation.csv | Comparación por tipo de modelo |
| Ganadores de AUPRC y F1 por categoría | resultados/modelos/ganadores_por_categoria_validation.csv | Comparación por categoría |
| Empate estadístico, 2 000 réplicas por video, intervalo y valor p ajustado | resultados/modelos/comparacion_individual_ensemble_validation.json | Frontera de Pareto e incertidumbre |
| Definición, dirección e interpretación de BA, macro-AUPRC, macro-F1, R_0.67 y carga de revisión | docs/CRITERIO_SELECCION_MODELOS_03.md; Brodersen et al.; Saito y Rehmsmeier; Davis y Goadrich | Dos diapositivas previas al ranking y protocolo de selección |
| Test natural de 22 684 chunks; BA 0,846; sensibilidad 0,906; AP 0,530 | resultados/modelos/test_final_abierto_una_vez.json | Evaluación final a cobertura completa |
| Cobertura automática 65,2 %; revisión 34,8 %; BA selectiva 0,940 | resultados/modelos/test_final_abierto_una_vez.json | Política humano–IA |
| Resultados por daño en test natural | resultados/modelos/metricas_por_categoria_test.csv | Gráfico por categoría |
| HatEval 0,730; OffendES 0,7839; DETOXIS 0,6461; EXIST 0,7944; HateXplain 0,687 | Artículos primarios citados en referencias.bib y copias en referencias_y_descargas | Capítulo propio: aplicación, ámbito, diseño, semejanzas, diferencias y comparación contextual de F1 |
| NaijaHate: AP 0,34 representativo y 0,83–0,90 enriquecido | Tonneau et al., citado en referencias.bib y estado del arte en docs | Efecto del diseño muestral |
| Definiciones y exclusiones de las cinco salidas | docs/TAXONOMIA_V2.md y docs/METODOLOGIA_ETIQUETADO_CASCADA.md | Taxonomía y etiquetado |
| Paridad de los frontends activos y carácter histórico de capturas | docs/PARIDAD_FRONTENDS_ACTIVOS.md | Diapositivas de interfaces |
| Servicios activos de etiquetado y producción | http://127.0.0.1:8765/ y http://127.0.0.1:8876/ | Botones clicables en conclusiones; verificación HTTP 200 del 15-08-2026 |

## Notas de comunicación

- Exactitud balanceada (BA), precisión promedio (AP), área bajo la curva precisión–sensibilidad (AUPRC), predicción fuera de muestra (OOF), frecuencia de término–frecuencia inversa de documento (TF–IDF) y adaptación de bajo rango (LoRA) se definen la primera vez que aparecen.
- El valor 0,940 corresponde solamente a la ruta automática, que cubre 65,2 % del test. No se presenta como resultado a cobertura completa.
- El estado del ganador en validación es empate estadístico o evidencia inconclusa. La selección del ensemble es operativa y reproducible, no una afirmación de superioridad estadística.
- Las capturas de etiquetado y producción son históricas; la paridad de la versión activa se sustenta documentalmente.
- La calificación crítica es deliberadamente doble: competitividad contextual y alineamiento con prácticas del estado del arte aplicado, sin afirmar un récord común que requeriría un benchmark externo compartido.
