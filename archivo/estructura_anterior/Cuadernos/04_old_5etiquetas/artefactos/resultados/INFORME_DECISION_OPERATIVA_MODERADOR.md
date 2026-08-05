# Decisión operativa del moderador de contenido

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha: 2026-07-27T15:49:08-05:00

## Método

La evaluación distingue dos usos. Para autonomía se exigen simultáneamente precisión y recall de daño ≥ 0,90, límites inferiores Wilson 95 % ≥ 0,85, recall mínimo por categoría ≥ 0,80 y F1 macro de daño ≥ 0,75. Además se exige evidencia externa: test humano independiente, prevalencia natural y piloto prospectivo.

Para asistencia humana se calibra **sólo en validación** el mayor umbral de margen `max(score - umbral_de_categoría)` que capture al menos 95% del daño. En test se exige recall ≥ 90%, límite inferior Wilson ≥ 85%, VPN ≥ 95% y revisión ≤ 60%. Esto formaliza el intercambio cobertura–riesgo de la clasificación selectiva (Geifman & El-Yaniv, 2017) y el intercambio costo–recall de moderación humana (Tonneau et al., 2024).

Los puntos de corte son criterios del proyecto, declarados para hacer auditable la decisión; no son estándares universales.

## Resultados

| Modelo | PR-AUC val. | PR-AUC test | Precisión autónoma | Recall autónomo | Autonomía | Tasa revisión | Recall alerta | VPN auto-paso | Alerta respaldada |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|
| SVM lineal palabra+carácter | 0.4612 | 0.4174 | 0.5390 | 0.6302 | no | 0.6376 | 0.9520 | 0.9739 | no |

| Paraphrase Multilingual MiniLM-L12 | 0.5024 | 0.4576 | 0.5792 | 0.6427 | no | 0.7180 | 0.9539 | 0.9678 | no |

| Multilingual E5-small (linaje MiniLM) | 0.5082 | 0.4399 | 0.5969 | 0.6359 | no | 0.6537 | 0.9433 | 0.9678 | no |

## Decisión

- Modelo seleccionado: **ninguno**.
- Modo: `none`.
- Estado: `no_model_is_ready_for_production`.
- Moderación autónoma respaldada: **no**.
- Alerta con revisión humana respaldada: **no**.

Aunque un modelo supere la puerta numérica, el test actual fue construido con prevalencia 4:1 y está etiquetado mayormente por LLM. Por eso no autoriza sanciones, eliminación o bloqueo autónomos. La salida defendible, si supera la puerta de alerta, es un piloto controlado: el modelo prioriza casos y una persona toma la decisión final. Antes de producción se requiere un test aleatorio de prevalencia natural con gold standard humano, análisis por subgrupo y monitoreo de deriva.

## Referencias (APA 7)

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

Tonneau, M., Quinta de Castro, P. V., Lasri, K., Farouq, I., Subramanian, L., Orozco-Olvera, V., & Fraiberger, S. P. (2024). NAIJAHATE: Evaluating hate speech detection on Nigerian Twitter using representative data. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 9020–9040). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.488/
