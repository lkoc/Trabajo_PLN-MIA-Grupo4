# Resultados

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Carpeta de métricas, figuras, informes y registros reproducibles.

- `metricas/`: JSON, CSV y scores por modelo y partición.
- `figuras/`: visualizaciones regenerables.
- `logs/sincronizacion_colab_04_20x/`: copias Drive → `D:` verificadas por SHA-256.
- `INFORME_QWEN_ACOSO_AMENAZA_4.md`: desempeño de Qwen plano con selección operativa de época 3.
- `INFORME_QWEN_JERARQUICO_4.md`: comparación consistente del Qwen plano y sus dos variantes jerárquicas.

La comparación global vigente se genera en `metricas/comparacion_final_4/comparacion_todos_modelos_4.csv`. Todo ranking usa validation; test describe el resultado final y no puede cambiar retrospectivamente la selección. Los informes históricos de cinco etiquetas se conservan por reproducibilidad, pero no pertenecen al contrato activo.

`operacion_05/` contiene el SQLite de inferencias, la bitácora de revisiones humanas y el JSONL deduplicado para un reentrenamiento futuro. No debe incorporarse automáticamente al train ni al test histórico.
