# Comparación de modelos 03_07 — actualización con ensembles optimizados

Fecha de actualización: 2026-08-17  
Ganador vigente: `ensemble_soft_optimized`  
Firma: `79f173133c3d3441d6241e671ac7af10ceb18253fcad3fe9aba43c653b0a58d5`

Este es el reporte único de la comparación 03_07. La actualización conserva la
pantalla de 28 modelos individuales y cinco reglas base, y añade una comparación
anidada y simétrica de todos los ensembles con grados de libertad: mezcla suave
y mezcla dura. No es una segunda pasada editorial ni escoge con test.

## Flujo reproducible

1. Se alinean 10 600 predicciones de validation (716 vídeos) de logística
   `C=0.5`, cascada E5 v2 y Qwen3–LoRA.
2. En cinco pliegues externos `GroupKFold` por vídeo se estima desempeño. Dentro
   de cada entrenamiento externo, otros cinco pliegues ajustan calibradores,
   umbrales y pesos.
3. Los pesos no negativos se buscan en el simplex, suma uno y paso 0,025: 861
   ternas por mezcla. El orden es BA descendente, riesgo 0,67 ascendente y
   macro-AUPRC descendente; un empate exacto favorece pesos cercanos a iguales.
4. Tras seleccionar solo con validation, se ajusta la fórmula final y se aplica
   a las tres matrices verificadas de la apertura original de test. No se ejecuta
   nueva inferencia ni se reajusta con test.

La combinación aprendida se apoya en *stacked generalization*/Super Learner
(Wolpert, 1992; van der Laan, Polley y Hubbard, 2007). El objetivo lexicográfico,
la ponderación proporcional a AUPRC y las reglas máximo/mínimo son decisiones
propias del trabajo. Unión e intersección no poseen coeficientes optimizables.

Entradas verificadas:

| Entrada | SHA-256 |
|---|---|
| Transformer `transformer_03_03b_run_outputs.tar.gz` | `fff9f75ae381ec0123b57850afc72528bd27e4d6ae75b8e3a6aedf150bbab290` |
| Qwen `run_outputs-b.tar` | `4a6242fa9a9c5c6e5182ebcf695dc44bdf32e224a7b93dd8b70cb7cf6eb1bf7b` |
| checkpoints de test `run_outputs-a.tar` | `6ac7ac71d4173819a07d266001eb2eec9711fef5d4edbce41062d28a069bc7e2` |

## Comparación actualizada de ensembles

Todas las cifras son OOF de los pliegues externos anidados. AP y F1 son macro
sobre los cuatro daños.

| Rango | Ensemble | BA | FNR | FPR | Riesgo 0,67 | Macro-AP | Macro-F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **suave optimizado** | **0,8366** | 0,1495 | 0,1774 | **0,1587** | 0,5506 | 0,5618 |
| 2 | ponderado por AUPRC | 0,8360 | 0,1557 | 0,1724 | 0,1612 | **0,5614** | 0,5675 |
| 3 | unión / máximo | 0,8334 | 0,1717 | 0,1616 | 0,1684 | 0,5369 | 0,5621 |
| 4 | promedio suave | 0,8328 | 0,1675 | 0,1670 | 0,1673 | 0,5604 | **0,5703** |
| 5 | duro optimizado | 0,8272 | 0,1736 | 0,1721 | 0,1731 | 0,4547 | 0,5584 |
| 6 | mayoría dura | 0,8272 | 0,1736 | 0,1721 | 0,1731 | 0,4415 | 0,5638 |
| 7 | intersección / mínimo | 0,8149 | 0,1958 | 0,1744 | 0,1887 | 0,5131 | 0,5373 |

El ganador actual es únicamente `ensemble_soft_optimized`, porque ocupa el
primer lugar bajo la regla declarada. Esto no implica superioridad universal:
frente al ponderado heurístico, ΔBA es +0,00059, IC95 %
[-0,00367; 0,00509], p=0,804 sin corrección (2 000 réplicas pareadas por vídeo).

## Fórmulas y coeficientes

Sea `p_mk = sigmoid(a_mk s_mk + b_mk)` la probabilidad calibrada del miembro
`m` para la salida `k`, y `D_mk = 1[p_mk >= t_mk]`.

| Caso | Fórmula | Pesos clásico / Transformer / Qwen |
|---|---|---|
| suave optimizado | `Σ w_m p_mk` | **0,100 / 0,650 / 0,250** |
| ponderado por AUPRC | `Σ w_m p_mk` | 0,3317 / 0,3110 / 0,3573 |
| promedio suave | `Σ p_mk / 3` | 1/3 / 1/3 / 1/3 |
| duro optimizado | `Σ w_m D_mk` | 0,300 / 0,325 / 0,375 |
| mayoría dura | `Σ D_mk / 3` | 1/3 / 1/3 / 1/3 |
| unión | `max_m p_mk` | no aplica |
| intersección | `min_m p_mk` | no aplica |

Pesos suaves externos: `[.225,.200,.575]`, `[.250,.150,.600]`,
`[.225,.200,.575]`, `[.075,.775,.150]` y `[.250,.175,.575]`. La variación,
especialmente del Transformer, es una limitación. El ajuste final
`[.100,.650,.250]` es el único congelado para inferencia.

Umbrales finales: `SEGURO=.46`, racismo `.17`, género `.18`, acoso `.39`,
sexual `.47`; compuerta cualquier daño `.072919`; revisión `delta=.03`.
Los 15 pares `(a,b)` de Platt están en el anexo del artículo y en
`optimizacion_ensembles_validation.json`.

## Auditoría de la pantalla base

Antes de la capa común y la optimización se obtuvieron: promedio suave
(BA/AP/F1 `0,8400/0,5549/0,5683`), ponderado
(`0,8383/0,5560/0,5694`), unión (`0,8328/0,5236/0,5648`), mayoría
(`0,8288/0,4606/0,5723`) e intersección (`0,8242/0,5165/0,5381`). Se conservan
para trazabilidad, no como una segunda comparación vigente.

## Test natural de la fórmula seleccionada

La fórmula fue congelada con validation antes de leer los checkpoints de test.
El TAR contiene las tres matrices de forma `(22684, 5)` y manifiestos con hashes
individuales. Reanalizarlas no constituye otra inferencia (`new_inference_passes=0`).

| Vista | Filas | BA | Sens. | Espec. | FNR | FPR | Macro-AP daño | Macro-F1 daño |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 22 684 | 0,84594 | 0,89401 | 0,79786 | 0,10599 | 0,20214 | 0,41194 | 0,42349 |
| 4:1 secundaria | 9 010 | 0,84635 | 0,89401 | 0,79870 | 0,10599 | 0,20130 | 0,56354 | 0,54521 |

Conteos naturales: TP=1 611, FN=191, FP=4 221, TN=16 661. Frente al promedio
simple anterior, BA cambia +0,00004: bajan los falsos positivos, pero suben los
falsos negativos. El test no revierte la selección porque no participa en ella.

Con `delta=.03`, se revisa 27,26 % y queda 72,74 % automático. En esa ruta:
BA=0,92489, sensibilidad=0,90224, especificidad=0,94755 y FN=96.

## Artefactos

- Cuaderno ejecutado: `flujo/03_entrenamiento/03_07b_optimizacion_ensembles.ipynb`.
- Resultado completo: `resultados/modelos/optimizacion_ensembles/optimizacion_ensembles_validation.json`.
- Comparación integrada: `comparacion_individual_ensemble_validation.json`.
- Congelación vigente: `seleccion_congelada.json`.
- Test vigente: `test_final_abierto_una_vez.json`.

La corrida ejecutada conserva números de ejecución y salidas sin errores en el
cuaderno. Requiere los dos TAR de `Downloads` con los hashes indicados.
