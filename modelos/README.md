# Modelos

Carpeta de checkpoints, adaptadores, calibradores, vectorizadores y cabezas entrenadas por los cuadernos `04_201`–`04_206`.

Para Qwen, `qwen3_06b_lora_acoso_amenaza_4/best_adapter` conserva el máximo PR-AUC de validación (época 2), mientras que `epoch_adapters/epoch_03` es el checkpoint operativo elegido a igual objetivo de recall. Los consumidores no deben inferir el checkpoint a partir del nombre de una carpeta: deben leer `resultados/metricas/qwen3_06b_lora_acoso_amenaza_4/seleccion_operativa_validacion.json` y verificar los hashes del estado y de los pesos.

`qwen_jerarquico_4/` contiene sólo las cabezas entrenadas sobre representaciones Qwen congeladas; no es un segundo fine-tuning del LLM.
