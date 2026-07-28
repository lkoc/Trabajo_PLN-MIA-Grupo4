# Criterios previos para Qwen3-0.6B + LoRA

Fecha de decisión: 27 de julio de 2026  
Cuaderno: `Cuadernos/04_3_finetuning_qwen3_lora.ipynb`

## Decisión

El tercer modelo será `Qwen/Qwen3-0.6B-Base`, revisión inmutable `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, mediante LoRA para clasificación multietiqueta. No se entrenarán etiquetas finas ni flags transversales.

Qwen3-0.6B-Base fue preferido porque:

- tiene licencia Apache-2.0 y pesos descargables;
- es el menor checkpoint base de la familia Qwen3, con 0,6B parámetros;
- su ficha declara preentrenamiento sobre 36 billones de tokens y 119 idiomas;
- Transformers implementa `Qwen3ForSequenceClassification`;
- LoRA permite congelar el modelo base y entrenar una fracción pequeña de parámetros.

GPT-3 fue ofrecido como modelo de API y no como checkpoint abierto para ajuste local. La alternativa abierta actual de OpenAI es `gpt-oss-20b`, pero la guía oficial de fine-tuning está diseñada para una H100 de 80 GB; por tanto no es una tercera corrida razonable en esta máquina. DeepSeek-V2-Lite tiene 16B parámetros y los checkpoints generales de texto Kimi/Moonlight publicados por Moonshot parten de 16B o son mucho mayores. QLoRA podría reducir memoria en una GPU adecuada, pero no elimina el coste de recorrer esos modelos ni hace reproducible su ajuste en este CPU o en una sesión gratuita no garantizada. Qwen3-0.6B ofrece una comparación de lenguaje generativo mucho más proporcionada al hardware disponible.

## Datos y dependencia del 04.2

El cuaderno no crea otra muestra. Consume el dataset 4:1 de 20.040 chunks generado en `04_2`:

- train: 14.064;
- validación: 2.992;
- test: 2.984;
- objetivos: cinco categorías gruesas de daño;
- separación agrupada por video, sin fuga.

Antes de entrenar se exige `modelos/moderador_transformer_grueso/registro_modelos_comparables.json`. El registro incluye SHA-256 del dataset y las rutas verificables del clásico ganador, MiniLM y E5. Qwen se compara con esos resultados sin repetir sus entrenamientos.

## Configuración fijada

| Parámetro | Valor |
|---|---:|
| Longitud | 128 tokens |
| Batch físico | 2 |
| Acumulación | 4 |
| Batch efectivo | 8 |
| Épocas máximas | 2 |
| Learning rate | 1e-4 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0,05 |
| Módulos LoRA | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Semilla | 20260727 |

La selección de época usa PR-AUC macro de las cinco categorías en validación. El test no selecciona época, umbral ni modelo.

## Cómputo

La ejecución detecta CUDA automáticamente. En CPU local el modelo cabe razonablemente en 28,83 GB de RAM porque el checkpoint tiene 0,6B parámetros y el optimizador sólo mantiene estados para los adaptadores, pero el recorrido hacia atrás por las 28 capas puede tomar muchas horas. Por ello se recomienda una GPU CUDA de Colab o un servidor Jupyter remoto. La barra `tqdm` informa lotes, pérdida, learning rate y ETA observada.

Para una ejecución remota debe copiarse o montarse el proyecto completo, incluidos dataset, manifiesto, checkpoints y registro del `04_2`. Esto evita comparaciones contra artefactos distintos.

Colab alojado se usa desde su interfaz web. Como alternativa, VS Code admite una URL de servidor Jupyter remoto autenticado; no se presupone una conexión directa de VS Code al kernel alojado de Colab. El cuaderno admite ambos escenarios mediante `PROJECT_ROOT_OVERRIDE` y conserva rutas relativas dentro del proyecto.

## Decisión operativa

Qwen se somete a las mismas puertas que los modelos del `04_2`:

1. autonomía: precisión y recall ≥ 0,90, límites inferiores Wilson ≥ 0,85, recall mínimo por categoría ≥ 0,80 y F1 macro ≥ 0,75;
2. alerta humana: umbral calibrado en validación para recall ≥ 0,95 y confirmado en test con recall ≥ 0,90, límite inferior Wilson ≥ 0,85, VPN ≥ 0,95 y revisión ≤ 0,60;
3. evidencia externa: un test humano independiente, prevalencia natural y piloto prospectivo son obligatorios para autorizar autonomía.

El test actual es balanceado y mayormente pseudoetiquetado. Por ello, aun con buen desempeño, la filosofía defendible es como máximo un piloto de alerta con decisión final humana.

## Trazabilidad

El cuaderno generará:

- `resultados/metricas/transformer_grueso/finetuning_qwen3_06b_lora.json`;
- `resultados/metricas/transformer_grueso/evaluacion_qwen3_06b_lora.json`;
- `resultados/metricas/transformer_grueso/operacion_qwen3_06b_lora.json`;
- `resultados/INFORME_FINETUNING_QWEN3_LORA.md`;
- `modelos/moderador_transformer_grueso/qwen3_06b_lora/best_adapter/`;
- registro actualizado de todos los modelos comparables.

No se generan CSV específicos para Qwen.

El registro contiene ruta, tamaño y SHA-256 de cada checkpoint/adaptador y de su JSON de evaluación. `registered_artifact_frame()` los enumera y `load_registered_model(model_key)` vuelve a cargar joblib, checkpoints PyTorch o el adaptador Qwen; para fastText devuelve la ruta del binario compatible con el CLI incluido.

## Referencias (APA 7)

DeepSeek-AI. (2024). *DeepSeek-V2-Lite* [Modelo de lenguaje]. Hugging Face. https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite

Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*. https://openreview.net/forum?id=nZeVKeeFYf9

Google. (n.d.). *Colaboratory: Local runtimes*. https://research.google.com/colaboratory/local-runtimes.html

Microsoft. (n.d.). *Jupyter Notebooks in Visual Studio Code*. https://code.visualstudio.com/docs/datascience/jupyter-notebooks

Moonshot AI. (n.d.). *Modelos publicados por Moonshot AI* [Colección de modelos]. Hugging Face. https://huggingface.co/moonshotai/models

OpenAI. (n.d.). *Fine-tuning a multilingual reasoner with Hugging Face*. https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers

Qwen Team. (2025). *Qwen3 technical report*. arXiv. https://doi.org/10.48550/arXiv.2505.09388

Qwen Team. (2025). *Qwen3-0.6B-Base* [Modelo de lenguaje]. Hugging Face. https://huggingface.co/Qwen/Qwen3-0.6B-Base

Tonneau, M., Quinta de Castro, P. V., Lasri, K., Farouq, I., Subramanian, L., Orozco-Olvera, V., & Fraiberger, S. P. (2024). NAIJAHATE: Evaluating hate speech detection on Nigerian Twitter using representative data. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 9020–9040). Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.488/
