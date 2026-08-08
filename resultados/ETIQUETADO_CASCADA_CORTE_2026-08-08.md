# Corte cuantitativo de la campaña de etiquetado Flash→Pro

**Corte documentado:** 2026-08-08 13:15:01 (UTC−05), checkpoint
periódico completo seleccionado de la primera pasada Flash. La campaña seguía en
ejecución después de ese instante y el checkpoint operativo se sobrescribe. Las
cifras siguientes preservan la lectura de ese instante, pero este documento no
reemplaza los `*.result.json` finales.

## Estado del corpus y recuperación histórica

El corpus activo contiene **166 940 chunks**. Antes de hacer llamadas nuevas,
`02_01` reindexó exclusivamente coincidencias unívocas por
`(video_id, texto_normalizado)`; una coincidencia aproximada o una segmentación
distinta permanece pendiente.

| Fuente | Filas históricas | Recuperadas exactamente | Reutilización | No reutilizables |
|---|---:|---:|---:|---:|
| Flash | 69 853 | 52 244 | 74.79 % | 17 609 |
| Pro, tres fuentes | 13 421 | 9 912 | 73.85 % | 3 509 |

La recuperación dejó **114 696 pendientes para la primera pasada Flash**. Las
9 912 filas Pro recuperadas no significan que los restantes 157 028 chunks
deban pasar por Pro: la cola Pro se materializa después de Flash y contiene
solo daño, abstención, baja confianza y una muestra segura de control.

## Calibración activa de 1 000 pares

Flash y Pro recibieron el mismo panel, con el mismo prompt operacional, contrato
JSON y modo `thinking=disabled`. No hubo errores de esquema.

| Etapa medida | Chunks | Errores | Tiempo | Velocidad | Caché de entrada | Costo estimado por respuestas |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 1 000 | 0 | 97.250 s | 616.969 chunks/min | 64.22 % | US$0.073349 |
| Pro | 1 000 | 0 | 122.593 s | 489.424 chunks/min | 53.72 % | US$0.269223 |
| Secuencia completa | 2 000 | 0 | 219.843 s | 545.844 chunks/min efectiva | — | US$0.342572 |

Sin caché, el mismo tráfico habría costado aproximadamente US$0.134656 en
Flash y US$0.431785 en Pro. Los ahorros observados fueron **45.53 %** y
**37.65 %**, respectivamente. Son ahorros medidos sobre tokens facturados, no
una garantía para lotes posteriores.

### Eficacia operacional, no exactitud humana

La calibración usa Pro como referencia más fuerte, no como *gold standard*.
El umbral conservador 0.95 produjo:

| Indicador a umbral 0.95 | Resultado |
|---|---:|
| autoaceptaciones evaluables | 434/1 000 |
| cobertura operacional | 43.40 % |
| acuerdo exacto Flash–Pro | 80.41 % |
| límite inferior Wilson unilateral 95 %, acuerdo exacto | 77.10 % |
| acuerdo binario daño/seguro | 99.77 % |
| límite inferior Wilson unilateral 95 %, acuerdo binario | 98.97 % |
| bootstrap agrupado por video, acuerdo exacto | [76.50 %, 84.10 %] |
| bootstrap agrupado por video, acuerdo binario | [99.31 %, 100.00 %] |

El estado es `inconclusive_conservative_threshold`: se superó el objetivo
binario, pero no el mínimo predeclarado de 90 % para el límite inferior de
acuerdo exacto. La confianza autodeclarada tampoco quedó bien calibrada respecto
del acuerdo exacto (`MAE=0.365`). En consecuencia, **43.40 % no debe describirse
como exactitud ni como cobertura humana validada**; 0.95 es el umbral de
enrutamiento conservador disponible mientras continúa la revisión dirigida y
la adjudicación humana.

## Primera pasada Flash en curso

El checkpoint atómico seleccionado registra:

| Indicador | Valor |
|---|---:|
| ya recuperados antes de la API | 52 244 |
| pendientes seleccionados | 114 696 |
| etiquetados nuevos en el checkpoint | 14 399 (12.55 % de los pendientes) |
| errores rechazados por validación | 1 |
| progreso total recuperado+nuevo | 66 643/166 940 (39.92 %) |
| tiempo de primera pasada | 996.219 s |
| velocidad | 867.279 chunks/min |
| tasa de caché Flash acumulada | 75.17 % |
| costo Flash acumulado, incluida su calibración | US$0.982130 |

Como el proveedor Flash se reutiliza entre calibración y primera pasada, sus
contadores son acumulativos. Restando la calibración Flash, los primeros 14 399
pendientes válidos costaron aproximadamente **US$0.908781**, o **US$0.06311 por
1 000 chunks**. Si velocidad, longitud de salida y caché permanecieran
constantes, la primera pasada completa sobre 114 696 pendientes tardaría unas
**2 h 12 min** y costaría alrededor de **US$7.24**; desde el checkpoint
restaban unas **1 h 56 min**. Ambas cifras son proyecciones tempranas, no
resultados finales. El error corresponde a un `chunk_id` u orden alterado por el
proveedor: la fila se rechazó y no se contabilizó como etiquetada.

Sumando el contador acumulado Flash del checkpoint y el resultado Pro de la
calibración, la campaña había registrado **US$1.251353** frente a
**US$2.552732** en un escenario contrafactual sin caché: ahorro estimado de
**50.98 %**, con tasa de caché ponderada de **73.89 %**. La última consulta de
saldo documentada antes de este checkpoint informó **US$19.10 disponibles**.
El saldo de cuenta y el costo reconstruido pueden diferir por redondeo y
actualización del proveedor; ambos se conservan como mediciones distintas.

## Reanudación Pro presupuestada

Después de detener manualmente la revisión Pro, el checkpoint más reciente
conservó **29 270 revisiones únicas**: 15 191 ya estaban presentes al iniciar el
tramo y se añadieron 14 079 válidas. Ese tramo consumió 13 949 465 tokens de
entrada y 2 236 040 de salida, con 54.61 % de caché, a un costo estimado de
**US$4.727523** y una velocidad de **452.415 chunks/min**. La consulta de saldo
sin corpus del mismo día informó **US$5.75**.

La cola anterior seleccionaba 81 610 chunks y aún dejaba 58 086 sin Pro; al
costo unitario observado, completarla costaría aproximadamente US$19.50. Incluso
el conjunto formado por todo daño y toda abstención dejaría 51 673 pendientes y
costaría aproximadamente US$17.35. Por ello no se afirma que US$15 alcancen para
la regla conservadora anterior.

La regla presupuestada conserva todas las revisiones existentes y selecciona:

| Prioridad | Total todavía pendiente |
|---|---:|
| todo daño Flash | 12 613 |
| 36 000 abstenciones de menor confianza, descontando las ya revisadas | 27 078 |
| `SEGURO` con confianza Flash `<0.85` | 145 |
| control seguro aleatorio reproducible del 1 % | 859 |
| **total nuevo** | **40 695** |

Usando el costo y la velocidad observados, el punto de planificación es
**US$13.66** y **90.0 minutos**. `MAX_REVIEW_COST_USD=14.50` deja margen frente
a un saldo aproximado de US$15.75 después de una recarga de US$10. La proyección
no garantiza el precio final: longitud de respuesta, caché y superlotes
concurrentes pueden variar. Si el tope se alcanza, el checkpoint detiene nuevos
grupos y conserva lo completado.

## Fuentes y regla de actualización

Las cifras provienen de:

- `datos/etiquetado/cascada_deepseek_v4/primary_flash.jsonl.recovery.json`;
- `datos/etiquetado/cascada_deepseek_v4/review_pro.jsonl.recovery.json`;
- `datos/etiquetado/cascada_deepseek_v4/calibration_flash.result.json`;
- `datos/etiquetado/cascada_deepseek_v4/calibration_pro.result.json`;
- `datos/etiquetado/cascada_deepseek_v4/calibration_flash_vs_pro.json`;
- `datos/etiquetado/cascada_deepseek_v4/primary_flash.jsonl.checkpoint.json`.

El siguiente corte debe sustituir las proyecciones por `primary_flash.result.json`,
`routing_summary.json`, `review_pro.result.json` y, al cerrar la etapa humana,
por las métricas del snapshot de `02_05`. Hasta entonces no existe una medida
de exactitud contra referencia humana independiente.
