# Qwen3-0.6B LoRA con cuatro daños

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha: 2026-07-29T13:06:28-05:00

## Diseño y selección

Se entrenó `Qwen/Qwen3-0.6B-Base` con LoRA durante 4 épocas para cuatro daños: RACISMO_DISCRIMINACION, ACOSO_GENERO_IDENTIDAD, ACOSO_AMENAZA, CONTENIDO_SEXUAL. `SEGURO` se deriva cuando ninguna categoría supera su umbral. Las 14 etiquetas finas y tres banderas transversales son supervisión auxiliar; no amplían las salidas operativas.

`best_adapter` conserva la época **2**, que maximiza el PR-AUC macro de validación. Para la política de alto recall se compararon los dos mejores checkpoints al mismo objetivo de 95%, exclusivamente en validación:

| Época | PR-AUC validación | Recall de revisión | Precisión de revisión | Tasa de revisión | Falsos negativos |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.5432 | 0.9508 | 0.3016 | 63.82% | 53 |
| 3 | 0.5425 | 0.9508 | 0.3173 | 60.67% | 53 |

La regla predefinida seleccionó la época **3** porque redujo la tasa de revisión manteniendo el objetivo de recall. La selección quedó fijada antes de consultar test (`test_used_for_selection = false`). El adaptador operativo es `modelos/qwen3_06b_lora_acoso_amenaza_4/epoch_adapters/epoch_03`.

## Resultado final en test

- PR-AUC macro de daño: 0.5488.
- F1 macro de daño: 0.5247.
- Precisión de cualquier daño: 0.5932.
- Recall ordinario de cualquier daño: 0.7003.
- Daños clasificados como seguros: 312.

| Categoría | Recall test |
|---|---:|
| RACISMO_DISCRIMINACION | 0.5898 |
| ACOSO_GENERO_IDENTIDAD | 0.5209 |
| ACOSO_AMENAZA | 0.5702 |
| CONTENIDO_SEXUAL | 0.5910 |

## Operación selectiva al objetivo de recall

Con el corte fijado en validación, test alcanza recall 0.9731, precisión 0.3171, tasa de revisión 60.40%, VPN 0.9866 y 28 falsos negativos automáticos. La puerta operativa resulta **no aprobada**; controles incumplidos: review_rate_at_most_0_60.

## Conclusión sobre desempeño y producción

El modelo muestra capacidad útil para priorizar revisión, pero su clasificación ordinaria aún es moderada y la política de alto recall requiere revisar 60.40% de los textos. Por ello, **no está listo para moderación autónoma, bloqueo ni sanción en producción**. Puede usarse en experimentación fuera de línea o en un piloto controlado en modo sombra, con revisión humana y sin afectar usuarios. Antes de desplegarlo se necesita un gold standard humano independiente, prevalencia real, validación prospectiva y controles de capacidad, latencia, coste, deriva y desempeño por subgrupos.

## Referencias (APA 7)

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Qwen Team. (2025). Qwen3 technical report. *arXiv*. https://doi.org/10.48550/arXiv.2505.09388
