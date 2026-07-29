# Matriz activa de entrenamiento con cuatro categorías

Esta carpeta conserva únicamente cuadernos `04_*` activos cuya taxonomía operativa es:

1. `RACISMO_DISCRIMINACION`
2. `ACOSO_GENERO_IDENTIDAD`
3. `ACOSO_AMENAZA = ACOSO_PERSONAL ∪ AMENAZA_DIRECTA`
4. `CONTENIDO_SEXUAL`

`SEGURO` se deriva cuando ninguna categoría de daño se activa.

## Contrato común de datos

- Dataset unificado: `datos/model_ready/transformer_grueso/dataset_integrado_todas_pasadas.jsonl`
- SHA-256 unificado: `3f01b76d285d4cdd2a0922df1d2437a7f01abc4e05e2699c218b1f5faaba2069`
- Filas unificadas: 117.244
- `SEGURO`: 110.181
- Con algún daño: 7.063
- Dataset 4:1 común para comparaciones planas: `datos/model_ready/transformer_grueso/dataset_balanceado_4a1_particionado.jsonl`
- SHA-256 4:1: `df2ac01183271e44b6dcfb9cb4850bd6b1ef1cd11d9fc51c881be944670ef20f`
- Manifiesto de splits SHA-256: `11f87f57c81b217267773fdbd971fe613a473e652c9c64ef2ef2f726e7364bd6`
- Splits 4:1: train 24.701, validation 5.324 y test 5.290.

Los modelos planos usan el 4:1 para permitir comparación directa. Las puertas binarias pueden usar el dataset integrado, pero excluyen todos los videos de validation/test. Por esa restricción académica, train puede utilizar 76.874 de los 110.181 chunks `SEGURO`; usar los 110.181 en train produciría fuga por video.

## Cuadernos activos

| Orden | Cuaderno | Familia | Estructura | Datos temáticos | `SEGURO` adicional | Finas/transversales |
|---:|---|---|---|---|---|---|
| 1 | `04_201_clasicos_planos_y_jerarquicos_4_etiquetas.ipynb` | SVM y regresión logística | Plano, cascada y jerarquía compartida | Dataset integrado con split por video | Train: 76.874 seguros | No en entrenamiento; auditoría común en `04_208` |
| 2 | `04_202_transformers_planos_4_etiquetas.ipynb` | MiniLM y E5 | Plano | Train 4:1 | No | No en entrenamiento; auditoría común en `04_208` |
| 3 | `04_203_transformer_cascada_4_etiquetas.ipynb` | E5/MiniLM seleccionado | Puerta binaria + segunda etapa | Segunda etapa 4:1 | Puerta: 76.874 seguros sin fuga | No en entrenamiento; auditoría común en `04_208` |
| 4 | `04_204_transformer_jerarquico_multitarea_4_etiquetas.ipynb` | E5/MiniLM seleccionado | Encoder compartido, puerta + cuatro daños | Pérdida temática 4:1 | Cabeza binaria: 76.874; pérdida temática enmascarada en extras | No en entrenamiento; auditoría común en `04_208` |
| 5 | `04_205_finetuning_qwen_acoso_amenaza.ipynb` | Qwen3-0.6B LoRA | Plano, cuatro daños; cuatro épocas | Train 4:1 | No | Supervisión auxiliar multitarea: el modelo las predice |
| 6 | `04_206_qwen_cascada_y_jerarquico_4_etiquetas.ipynb` | Qwen `04_205` época operativa 3 congelada + cabezas | Cascada y multitarea | Cabezas temáticas 4:1 | Cabeza binaria: 76.874; ejecutar después de `04_205` | Usa logits auxiliares predichos por Qwen; no usa anotación gold en inferencia |
| 7 | `04_207_comparacion_final_modelos_4_etiquetas.ipynb` | Todas | Comparación final sobre test común | Solo lectura de resultados | No aplica | Compara las cuatro etiquetas gruesas |
| 8 | `04_208_auditoria_finas_transversales_modelos_4.ipynb` | Todas | Auditoría por subgrupos e incertidumbre | Mismo test 4:1 | No aplica | Audita etiquetas finas y flags en todos los modelos terminados |

## Orden sugerido

El orden completo está en `04_200_ORDEN_EJECUCION.md`. Los cuadernos `04_201`–`04_205` pueden ejecutarse independientemente, aunque conviene respetar el orden pedagógico. `04_206` depende estrictamente de que `04_205` haya completado cuatro épocas y generado `finetuning.json`, `seleccion_operativa_validacion.json`, `evaluacion_test_modelo_seleccionado.json` y el adaptador de la época seleccionada. `04_207` y `04_208` van al final.

La ejecución Qwen iniciada bajo el nombre anterior `04_7_finetuning_qwen_acoso_amenaza.ipynb` equivale a `04_205`. Se conserva temporalmente para no interrumpir el kernel y no debe ejecutarse una segunda vez desde el nombre nuevo.

Todos los resultados se seleccionan con validation; test queda reservado para evaluación final. Las comparaciones jerárquicas usan bootstrap pareado por `video_id`.

## Estado actual de la selección Qwen

`best_adapter` corresponde a la época 2 porque maximiza PR-AUC de validación. La política operativa fijada antes de test seleccionó la época 3 al comparar los dos mejores checkpoints a 95 % de recall: mantuvo el objetivo con mejor precisión y menor tasa de revisión. Los cuadernos `04_206`–`04_208` deben usar la época 3, no el alias `best_adapter`.

La comparación recalculada indica que Qwen plano sigue siendo superior a sus dos variantes jerárquicas. En test, Qwen plano obtiene PR-AUC 0,5488, F1 0,5247 y recall 0,7003; el ganador entre los jerárquicos obtiene 0,5320, 0,5170 y 0,6398, además de 63 daños adicionales clasificados como seguros. No se reemplaza el modelo plano.

`04_207` publica para `05` un registro desplegable con los ganadores por familia: SVM plano, E5-small plano y Qwen plano época operativa 3. `04_208` verifica sus hashes y mantiene la auditoría de los 13 modelos disponibles; el servidor no escoge modelos usando test.

## Regla contra fuga de información

Las etiquetas finas y transversales gold no se entregan como columnas predictoras: un chunk nuevo no las tendría antes de ser moderado. Qwen `04_205` puede usarlas legítimamente como objetivos auxiliares durante entrenamiento porque aprende a predecirlas desde el texto. `04_206` reutiliza únicamente sus logits predichos. `04_208` aplica las anotaciones gold solo después de la inferencia para desagregar resultados y estudiar la priorización de revisión humana.

Por ello, `04_205`/`04_206` frente a los modelos `coarse-only` no constituyen por sí solos una ablación causal de las etiquetas auxiliares. La comparación sirve para selección operacional; atribuir la diferencia exclusivamente a esas pérdidas exigiría entrenar la misma arquitectura y semilla con y sin supervisión auxiliar.
