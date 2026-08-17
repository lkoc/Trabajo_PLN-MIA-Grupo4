# Fuentes y trazabilidad de la presentación

## Regla de interpretación

Las métricas propias provienen de los artefactos más recientes de 03_07b, integrados en el reporte único 03_07. Las comparaciones con otros autores son contextuales: los corpus, idiomas, plataformas, prevalencias, categorías y particiones no son idénticos.

| Afirmación o cifra | Fuente primaria | Uso en la presentación |
|---|---|---|
| Alcance inicial: 182 461 chunks, 5 385 videos y 322 canales; dataset efectivo: 173 240 chunks, 4 906 videos y 276 canales | docs/artefactos/auditoria_estado_final_182461.json | Diferencia entre universo observado, exclusiones y dataset final |
| 159 077 chunks SEGURO; 14 163 con daño; 17 052 asignaciones; 2 709 chunks multietiqueta | docs/artefactos/auditoria_estado_final_182461.json | Desbalance y carácter multietiqueta |
| Canales principales, media/mediana, máximos y concentración top-1/top-5/top-10 | docs/artefactos/auditoria_estado_final_182461.json | Estadística descriptiva y riesgo de concentración por fuente |
| Comparación de 15/20/25/30/35 s y decisión de conservar 30 s | docs/OPTIMIZACION_LONGITUD_CHUNKS.md y docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md | Procedimiento breve de optimización del tamaño de chunk |
| 55 966 eventos trazables; panel congelado de 16 694; cobertura combinada de 10,18 % | docs/artefactos/auditoria_estado_final_182461.json | Metodología de etiquetado y auditoría |
| 28 individuos, 5 reglas base y 2 variantes optimizadas; único ganador ensemble_soft_optimized | comparación integrada + optimizacion_ensembles_validation.json | Ranking y selección global |
| Fórmulas, pesos, calibradores y métricas de las siete variantes | optimizacion_ensembles_validation.json, ensemble_optimization.py y cuaderno 03_07b ejecutado | Dos diapositivas de metodología/resultados; copia en datos_fuente/ensembles_validation.csv |
| Mejores representantes clásico, Transformer, Qwen y ensemble | resultados/modelos/mejores_por_tipo_validation.csv | Comparación por tipo de modelo |
| Arquitectura y precisión numérica de Qwen3--0.6B y MiniLM; 596 049 920 y 117 662 230 parámetros, respectivamente; LoRA estándar sin cuantización y cálculo mixto BF16 | Tarjetas versionadas de Hugging Face; configuraciones y tensores locales; `docs/ARQUITECTURAS_MODELOS_03.md` | Diapositiva de escala, ajuste, cuantización y precisión numérica |
| LoRA base 128, continuación LoRA 256, tres brazos LoRA estructurados y ajuste completo histórico sin LoRA | Cuadernos `03_05`/`03_06`, estados de entrenador, `adapter_config.json` y ranking vigente de `03_07` | Diapositiva de diferencias entre entrenamientos Qwen; identificación del LoRA base como ganador y miembro del ensemble |
| Ganadores de AUPRC y F1 por categoría | resultados/modelos/ganadores_por_categoria_validation.csv | Comparación por categoría |
| Ventaja pareada inconclusa, 2 000 réplicas por video, IC95 % y p sin corrección | optimizacion_ensembles_validation.json | Frontera de Pareto e incertidumbre |
| Definición, dirección e interpretación de BA, macro-AUPRC, macro-F1, $R_{0,67}=0,67\,\mathrm{FNR}+0,33\,\mathrm{FPR}$ y carga de revisión | docs/CRITERIO_SELECCION_MODELOS_03.md; Brodersen et al.; Saito y Rehmsmeier; Davis y Goadrich | Dos diapositivas previas al ranking y protocolo de selección; ejemplo 0,149 frente a 0,152 |
| Test natural de 22 684 chunks; BA 0,84594; sensibilidad 0,89401; AP 0,53121 | resultados/modelos/test_final_abierto_una_vez.json | Evaluación final a cobertura completa |
| Cobertura automática 72,7 %; revisión 27,3 %; BA selectiva 0,92489 | resultados/modelos/test_final_abierto_una_vez.json | Política humano–IA |
| Resultados por daño en test natural | resultados/modelos/metricas_por_categoria_test.csv | Gráfico por categoría |
| HatEval 0,730; OffendES 0,7839; DETOXIS 0,6461; EXIST 0,7944; HateXplain 0,687 | Artículos primarios citados en referencias.bib y copias en referencias_y_descargas | Capítulo propio: aplicación, ámbito, diseño, semejanzas, diferencias y comparación contextual de F1 |
| NaijaHate: AP 0,34 representativo y 0,83–0,90 enriquecido | Tonneau et al., citado en referencias.bib y estado del arte en docs | Efecto del diseño muestral |
| Definiciones y exclusiones de las cinco salidas | docs/TAXONOMIA_V2.md y docs/METODOLOGIA_ETIQUETADO_CASCADA.md | Taxonomía y etiquetado |
| Paridad de los frontends activos y carácter histórico de capturas | docs/PARIDAD_FRONTENDS_ACTIVOS.md | Diapositivas de interfaces |
| Servicios activos de etiquetado y producción | http://127.0.0.1:8765/ y http://127.0.0.1:8876/ | Botones clicables en conclusiones; verificación HTTP 200 del 15-08-2026 |

## Notas de comunicación

- Exactitud balanceada (BA), precisión promedio (AP), área bajo la curva precisión–sensibilidad (AUPRC), predicción fuera de muestra (OOF), frecuencia de término–frecuencia inversa de documento (TF–IDF) y adaptación de bajo rango (LoRA) se definen la primera vez que aparecen.
- El valor 0,925 corresponde solamente a la ruta automática, que cubre 72,7 % del test. No se presenta como resultado a cobertura completa.
- `ensemble_soft_optimized` es el único seleccionado por la regla; su ventaja pareada es inconclusa y no se presenta como superioridad estadística demostrada.
- Las capturas de etiquetado y producción son históricas; la paridad de la versión activa se sustenta documentalmente.
- La calificación crítica es deliberadamente doble: competitividad contextual y alineamiento con prácticas del estado del arte aplicado, sin afirmar un récord común que requeriría un benchmark externo compartido.
- Los entrenamientos Qwen se describen solo cuando existe un artefacto ejecutado y verificable; propuestas no ejecutadas no se presentan como método ni resultado del estudio.
