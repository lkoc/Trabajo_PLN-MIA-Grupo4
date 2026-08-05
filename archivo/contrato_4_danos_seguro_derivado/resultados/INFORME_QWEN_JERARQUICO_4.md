# Qwen plano, cascada y jerárquico multitarea con cuatro daños

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha: 2026-07-29T13:05:12-05:00

El adaptador operativo de la época **3**, elegido en validación antes de consultar test, permanece congelado y produce 21 logits: cuatro operativos y 17 auxiliares. Sobre esas mismas representaciones se entrenan una cascada logística y una cabeza neuronal multitarea. La supervisión binaria usa 76,874 SEGURO de train; la pérdida temática de la cabeza conjunta se enmascara en los negativos adicionales. Validation/test son exactamente los de `04_205`, y tanto las cabezas jerárquicas como la referencia plana corresponden al mismo checkpoint operativo.

| Modelo | PR-AUC macro | F1 macro | Recall daño | Daños como seguro |
|---|---:|---:|---:|---:|
| Qwen congelado + cascada calibrada | 0.5379 | 0.5184 | 0.6302 | 385 |
| Qwen congelado + cabeza jerárquica multitarea | 0.5320 | 0.5170 | 0.6398 | 375 |
| Qwen 04_205 plano · época operativa 3 | 0.5488 | 0.5247 | 0.7003 | 312 |

Ganador entre los dos diseños jerárquicos por validation: **Qwen congelado + cabeza jerárquica multitarea**. Las decisiones pareadas frente a Qwen plano están en `resultado.json`. Esta variante es Qwen congelado más cabezas jerárquicas; no es un segundo fine-tuning end-to-end del LLM.

## Conclusión sobre el esquema jerárquico

El ganador jerárquico obtiene en test PR-AUC macro 0.5320, F1 macro 0.5170, recall de daño 0.6398 y deja 375 daños como seguros. La referencia Qwen plana operativa obtiene respectivamente 0.5488, 0.5247, 0.7003 y 312. Las diferencias del ganador jerárquico frente al plano son -0.0168 en PR-AUC, -0.0077 en F1, -0.0605 en recall y +63 falsos negativos de daño.

Por tanto, **estos esquemas jerárquicos no resultan mejores que Qwen plano y no deben reemplazarlo** bajo el criterio pareado predefinido. Este resultado tampoco autoriza autonomía: cualquier uso operativo requiere validación humana independiente y un piloto prospectivo.

## Referencias (APA 7)

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.

Zhou, J., Ma, C., Long, D., Xu, G., Ding, N., Zhang, H., Xie, P., & Liu, G. (2020). Hierarchy-aware global model for hierarchical text classification. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1106–1117). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.104
