# Modelos

> **Contrato activo:** `moderacion_peru_5_salidas_v2`.

El registro activo esperado es `registro_modelos_5_salidas.json`. Debe declarar las cinco salidas en orden, umbrales, checkpoint, SHA-256, hardware, linaje y estado de despliegue.

No se copiará un clasificador de cuatro daños al registro v2. Los encoders o backbones pueden reutilizarse como inicialización, pero la nueva cabeza y sus umbrales requieren entrenamiento y validación. Los artefactos históricos permanecen en `archivo/`.

## Documentación de los modelos anteriores

Carpeta de checkpoints, adaptadores, calibradores, vectorizadores y cabezas entrenadas por los cuadernos `04_201`–`04_206`.

Para Qwen, `qwen3_06b_lora_acoso_amenaza_4/best_adapter` conserva el máximo PR-AUC de validación (época 2), mientras que `epoch_adapters/epoch_03` es el checkpoint operativo elegido a igual objetivo de recall. Los consumidores no deben inferir el checkpoint a partir del nombre de una carpeta: deben leer `resultados/metricas/qwen3_06b_lora_acoso_amenaza_4/seleccion_operativa_validacion.json` y verificar los hashes del estado y de los pesos.

`qwen_jerarquico_4/` contiene sólo las cabezas entrenadas sobre representaciones Qwen congeladas; no es un segundo fine-tuning del LLM.
