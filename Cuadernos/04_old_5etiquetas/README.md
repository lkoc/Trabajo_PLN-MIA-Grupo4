# Archivo de experimentos históricos con cinco etiquetas

Esta carpeta contiene los cuadernos `04` anteriores cuya salida temática separaba `ACOSO_PERSONAL` y `AMENAZA_DIRECTA`. Se conservaron para reproducibilidad y eventual reutilización de encoders o resultados; no forman parte de la matriz activa de cuatro categorías.

## Cuadernos archivados

- `04_entrenamiento_moderador.ipynb`
- `04_1_mejoras_entrenamiento_moderador.ipynb`
- `04_2_entrenamiento_transformers_gruesos.ipynb`
- `04_3_finetuning_qwen3_lora.ipynb`
- `04_4_cascada_jerarquica_moderacion.ipynb`
- `04_5_transformer_jerarquico_multitarea.ipynb`
- `04_6_jerarquico_clasico.ipynb`

## Resultados y modelos

`artefactos/` contiene una copia de los modelos, métricas, figuras e informes históricos disponibles. Las copias originales de algunos artefactos se mantienen temporalmente en sus rutas antiguas porque:

1. La sesión iniciada como `04_7` —ahora enumerada `04_205`— está ejecutándose y puede leer scores clásicos al finalizar.
2. Los cuadernos activos de cuatro etiquetas usan los encoders de `04_2` como warm start verificable.

Por tanto, el archivo es una copia de preservación, no una segunda fuente de verdad para los nuevos entrenamientos. `artefactos/ARCHIVO_COMPLETO.json` aparece al finalizar y verificar la copia por número de archivos y tamaño.

La transferencia a cuatro etiquetas nunca reutiliza los umbrales históricos. Se copia el encoder y, cuando corresponde, las filas compatibles de la cabeza; `ACOSO_AMENAZA` se inicializa con el promedio de las filas anteriores de acoso y amenaza y luego se reentrena.
