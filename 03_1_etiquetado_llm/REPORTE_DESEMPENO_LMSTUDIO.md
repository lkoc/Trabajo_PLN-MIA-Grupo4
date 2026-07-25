# Prueba de desempeño de LM Studio

Fecha de ejecución: 2026-07-19 (America/Lima)

## Objetivo

Comparar configuraciones de procesamiento secuencial y concurrente para el etiquetado local, manteniendo constantes el modelo, el prompt, la muestra y las reglas de validación.

## Entorno evaluado

- CPU: AMD Ryzen 7 8845HS, 8 núcleos y 16 hilos.
- RAM: 28.8 GiB.
- GPU instaladas: Radeon RX 570 y Radeon 780M.
- GPU detectadas por el runtime activo de LM Studio: ninguna.
- Runtime: `llama.cpp-win-x86_64-avx2@2.24.0`.
- Modelo: `qwen/qwen3.5-9b`, cuantización local de 6.55 GB.
- Identificador API: `qwen-local-primary`.
- Contexto: 16 384 tokens.
- Temperatura: 0.
- Muestra medida: los mismos 4 chunks del manifiesto piloto, con referencia CGT.

El modelo funcionó íntegramente en CPU. Se ejecutó un calentamiento antes de cada familia de pruebas y su tiempo no se incluyó en el rendimiento medido.

## Criterios de comparación

1. **Rendimiento:** chunks procesados por minuto; un valor mayor es mejor.
2. **Tiempo total:** segundos necesarios para procesar los mismos cuatro chunks; un valor menor es mejor.
3. **Eficiencia del prompt:** tokens de entrada consumidos para la muestra completa; menos repeticiones del prompt son preferibles.
4. **Estabilidad:** finalización y validación correcta del JSON estructurado.
5. **Control de calidad:** coincidencia exacta del conjunto de etiquetas con la referencia CGT.

## Resultados cuantitativos

| Escenario | Parallel | Trabajadores | Chunks por solicitud | Tiempo (s) | Chunks/min | Tokens de prompt | Tokens de salida | Coincidencia exacta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Secuencial, lote 1 | 1 | 1 | 1 | 105.75 | 2.270 | 12 556 | 735 | 3/4 (75%) |
| Secuencial, lote 2 | 1 | 1 | 2 | 87.64 | 2.739 | 6 664 | 610 | 4/4 (100%) |
| Secuencial, lote 4 | 1 | 1 | 4 | 82.50 | 2.909 | 3 718 | 578 | 4/4 (100%) |
| Concurrente, lote 1 | 2 | 2 | 1 | 84.07 | 2.855 | 12 556 | 751 | 3/4 (75%) |
| Concurrente, lote 2 | 2 | 2 | 2 | **67.67** | **3.547** | 6 664 | 606 | **4/4 (100%)** |

Todos los escenarios finalizaron con JSON válido. No se observaron reintentos durante las mediciones.

## Comparación

- El lote secuencial de 4 fue 28.1% más rápido que el lote secuencial unitario.
- Dos trabajadores con lotes de 2 fueron 56.3% más rápidos que el lote secuencial unitario.
- La mejor configuración concurrente fue 21.9% más rápida que el mejor resultado secuencial.
- Agrupar comentarios redujo considerablemente la repetición del prompt: de 12 556 tokens con lotes unitarios a 6 664 con lotes de 2 y 3 718 con lote de 4.
- El calentamiento tomó 33.41 s con `parallel=1`. La inicialización simultánea de las dos ranuras con `parallel=2` tomó 186.15 s; este coste se amortiza en ejecuciones largas.

## Proyección orientativa

Usando 3.547 chunks/min y sin contar calentamiento, reintentos ni variación de longitud:

- Piloto de 300 chunks: aproximadamente 84.6 minutos.
- Conjunto completo de 69 853 chunks: aproximadamente 328.2 horas o 13.7 días de ejecución continua.

Estas proyecciones se basan en una muestra pequeña y no sustituyen una prueba sostenida de 20–50 chunks.

## Recomendación

La configuración más rápida y estable de esta prueba fue:

```text
Modelo: qwen/qwen3.5-9b
GPU: off
Contexto: 16384
LM Studio parallel: 2
Trabajadores del cliente: 2
Chunks por solicitud: 2
```

Para una implementación sencilla sin concurrencia, la mejor alternativa fue `BATCH_SIZE=4` con `parallel=1`, a 2.909 chunks/min y 4/4 coincidencias.

Antes de procesar los 69 853 chunks se recomienda repetir el benchmark con al menos 20 registros. La muestra de cuatro permite orientar la configuración, pero es insuficiente para estimar con precisión la tasa de errores en textos largos o difíciles.

## Reproducción

Desde la raíz del proyecto:

```powershell
python .\03_1_etiquetado_llm\benchmark_lmstudio.py
```

El script vuelve a ejecutar los escenarios, genera este reporte y restaura el modelo con `parallel=1` al finalizar. Puede ampliarse la muestra con:

```powershell
python .\03_1_etiquetado_llm\benchmark_lmstudio.py --sample-size 20
```
