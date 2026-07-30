# Orden de ejecución de los experimentos 04

La numeración `04_20X` expresa el orden recomendado de ejecución manual. Todos los cuadernos usan la taxonomía activa de cuatro daños y `SEGURO` derivado.

| Paso | Cuaderno | Requisito | Resultado principal |
|---:|---|---|---|
| 1 | `04_201_clasicos_planos_y_jerarquicos_4_etiquetas.ipynb` | Dataset unificado | Baselines rápidos SVM/logística: plano, cascada y jerárquico |
| 2 | `04_202_transformers_planos_4_etiquetas.ipynb` | Dataset 4:1 y checkpoints históricos de encoders | MiniLM y E5 planos con cuatro salidas |
| 3 | `04_203_transformer_cascada_4_etiquetas.ipynb` | Dataset 4:1; lógicamente después del baseline Transformer | Puerta binaria y segunda etapa temática |
| 4 | `04_204_transformer_jerarquico_multitarea_4_etiquetas.ipynb` | Dataset 4:1; lógicamente después del baseline Transformer | Encoder compartido, puerta y cuatro daños |
| 5 | `04_205_finetuning_qwen_acoso_amenaza.ipynb` | Dataset 4:1 | Qwen3-0.6B LoRA plano, cuatro épocas y selección operativa en validación |
| 6 | `04_206_qwen_cascada_y_jerarquico_4_etiquetas.ipynb` | **`04_205` completo, calibrado y con adaptador operativo** | Cascada y cabeza multitarea sobre logits de la época 3 congelada |
| 7 | `04_207_comparacion_final_modelos_4_etiquetas.ipynb` | Resultados de `04_201`–`04_206` | Comparación final sobre el mismo test común |
| 8 | `04_208_auditoria_finas_transversales_modelos_4.ipynb` | Resultados finales disponibles | Auditoría por etiquetas finas, flags e incertidumbre |

## Dependencias y ejecución actual

`04_201`–`04_205` pueden entrenarse de manera independiente, pero el orden anterior facilita interpretar primero los baselines y después los modelos más costosos. `04_206` tiene una dependencia estricta de la selección operativa de `04_205`; `04_207` y `04_208` deben dejarse para el final si se desea una comparación completa. Estos tres últimos cuadernos reutilizan los artefactos existentes y no vuelven a entrenar Qwen.

La sesión Qwen iniciada con el nombre anterior `04_7_finetuning_qwen_acoso_amenaza.ipynb` corresponde exactamente al paso `04_205`. Ese archivo se conserva temporalmente mientras el kernel está activo. **No se debe volver a ejecutar `04_205` en paralelo.** Ambos nombres apuntan al mismo procedimiento y a los mismos artefactos externos; al terminar la sesión, el nombre antiguo puede archivarse.

## Kernel local o Colab

`04_202`–`04_206` incluyen arranque híbrido para kernel local y Colab con persistencia mínima en Google Drive. Antes de usarlos en Colab, ejecute `scripts_auxiliares/sincronizar_04_20x_google_drive.ps1`. `04_206`, `04_207` y `04_208` verifican los resultados locales frente al bundle de Drive. En `04_208`, un conflicto se resuelve conservando el artefacto más reciente, que puede provenir de Drive o del workspace; la decisión y las fechas quedan visibles. La preparación, persistencia y recuperación se describen en [`04_20X_COLAB_HIBRIDO.md`](04_20X_COLAB_HIBRIDO.md).

Después de `04_207`/`04_208`, `05_frontend_produccion.ipynb` consume `resultados/metricas/comparacion_final_4/registro_modelos_desplegables.json`. Ese paso no reentrena: carga los tres ganadores por familia y abre el flujo de operación documentado en `05_MODO_OPERACION.md`.
