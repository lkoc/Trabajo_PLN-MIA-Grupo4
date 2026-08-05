# Criterios y selección vigente de Qwen3-0.6B + LoRA

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Actualización: 29 de julio de 2026
Cuaderno: `Cuadernos/04_205_finetuning_qwen_acoso_amenaza.ipynb`

## Configuración

Se ajusta `Qwen/Qwen3-0.6B-Base`, revisión `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, mediante LoRA para cuatro daños. `ACOSO_AMENAZA` une `ACOSO_PERSONAL` y `AMENAZA_DIRECTA`; `SEGURO` se deriva. Las 14 etiquetas finas y tres banderas transversales se usan como objetivos auxiliares enmascarados, nunca como variables gold de entrada.

| Parámetro | Valor |
|---|---:|
| Dataset 4:1 | 24.701 train / 5.324 validation / 5.290 test |
| Longitud | 128 tokens |
| Batch físico | 2 |
| Acumulación | 4 |
| Batch efectivo | 8 |
| Épocas completadas | 4 |
| Learning rate | 1e-4 |
| LoRA rank / alpha / dropout | 8 / 16 / 0,05 |
| Semilla | 20260727 |

El entrenamiento se ejecuta en GPU CUDA cuando está disponible y guarda adaptador, optimizador, scheduler y estados aleatorios para reanudación. El aviso `score.weight MISSING` al cargar el checkpoint base es esperado: la cabeza de clasificación no existe en el modelo base y debe aprenderse durante el fine-tuning.

## Dos conceptos de “mejor época”

`best_adapter` conserva la época con mayor PR-AUC macro de validación. En la corrida final es la época 2. Para la política operativa se tomaron las dos mejores épocas por esa métrica, se calibró cada una en validation y se compararon al mismo objetivo de 95 % de recall:

| Época | PR-AUC validación | Recall de revisión | Precisión | Tasa de revisión | Falsos negativos |
|---:|---:|---:|---:|---:|---:|
| 2 | 0,5432 | 0,9508 | 0,3016 | 63,82 % | 53 |
| 3 | 0,5425 | 0,9508 | 0,3173 | 60,67 % | 53 |

La regla fijada antes de test seleccionó la **época 3** porque mantuvo el mismo recall y los mismos falsos negativos con mejor precisión y menor carga de revisión. Por eso `04_206`–`04_208` usan `modelos/qwen3_06b_lora_acoso_amenaza_4/epoch_adapters/epoch_03`, no `best_adapter`.

## Evaluación final

Test se abrió después de congelar época, calibradores, umbrales y corte de revisión. La época 3 obtiene PR-AUC macro 0,5488, F1 macro 0,5247, precisión de cualquier daño 0,5932 y recall ordinario 0,7003. Clasifica 312 chunks con daño como seguros.

La política selectiva alcanza recall 0,9731, precisión 0,3171, VPN 0,9866 y 28 falsos negativos, pero requiere revisar 60,40 % del test. No supera la puerta predefinida de tasa de revisión máxima de 60 %.

## Decisión operativa

El modelo no está listo para moderación autónoma, bloqueo ni sanción. Puede utilizarse sólo fuera de línea o en un piloto controlado en modo sombra, con revisión humana. La autorización futura requiere gold standard humano independiente, prevalencia natural, validación prospectiva y controles de latencia, coste, capacidad de revisión, deriva y subgrupos.

## Trazabilidad

- Selección: `resultados/metricas/qwen3_06b_lora_acoso_amenaza_4/seleccion_operativa_validacion.json`.
- Test del modelo fijado: `resultados/metricas/qwen3_06b_lora_acoso_amenaza_4/evaluacion_test_modelo_seleccionado.json`.
- Informe: `resultados/INFORME_QWEN_ACOSO_AMENAZA_4.md`.
- Los JSON registran `test_used_for_selection = false`; los pesos y estados se verifican por SHA-256.

## Referencias (APA 7)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Qwen Team. (2025). *Qwen3 technical report*. arXiv. https://doi.org/10.48550/arXiv.2505.09388
