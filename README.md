# Moderador de contenido en videos de YouTube

Trabajo final del curso de Procesamiento de Lenguaje Natural, Maestría en Inteligencia Artificial, Universidad Nacional de Ingeniería. Grupo 4, semestre 2026-1.

## Objetivo

Construir un flujo reproducible para recolectar transcripciones públicas, crear fragmentos auditables, etiquetarlos, entrenar clasificadores multietiqueta y comparar su utilidad para moderación asistida. La taxonomía activa contiene cuatro daños: `RACISMO_DISCRIMINACION`, `ACOSO_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`; `SEGURO` se deriva cuando ninguno se activa.

## Flujo activo

1. `Cuadernos/01*`: descubrimiento y recolección de transcripciones públicas.
2. `Cuadernos/02*`: limpieza, segmentación y deduplicación.
3. `Cuadernos/03*`: etiquetado, consolidación y revisión humana.
4. `Cuadernos/04_201`–`04_208`: modelos clásicos, Transformers, Qwen, comparación final y auditoría.
5. `Cuadernos/05_frontend_produccion.ipynb`: servidor local, texto/YouTube, comparación/consenso, revisión humana y estadísticas reentrenables. Consulte `Cuadernos/05_MODO_OPERACION.md`.

El contrato de entrenamiento vigente está en [Cuadernos/04_MATRIZ_ENTRENAMIENTO_4_ETIQUETAS.md](Cuadernos/04_MATRIZ_ENTRENAMIENTO_4_ETIQUETAS.md) y el orden reproducible en [Cuadernos/04_200_ORDEN_EJECUCION.md](Cuadernos/04_200_ORDEN_EJECUCION.md). Los experimentos históricos de cinco etiquetas se conservan en `Cuadernos/04_old_5etiquetas/` y no deben mezclarse con el flujo activo.

## Estado de Qwen

`04_205` completó cuatro épocas. La época 2 sigue siendo el `best_adapter` formal por PR-AUC de validación, mientras que la época 3 es el checkpoint operativo: fue elegida entre los dos mejores modelos al mismo objetivo de 95 % de recall, con menor tasa de revisión y sin consultar test. `04_206`, `04_207` y `04_208` consumen explícitamente esa época operativa y verifican sus hashes.

La comparación actual conserva Qwen plano como mejor modelo global por validación. Los esquemas Qwen jerárquicos no lo superan y no deben reemplazarlo. Ningún resultado autoriza moderación autónoma; el uso defendible es experimentación o piloto en modo sombra con revisión humana.

## Directorios

- `datos/`: insumos, datos intermedios y datasets congelados.
- `modelos/`: checkpoints, adaptadores, calibradores y cabezas entrenadas.
- `resultados/`: métricas, figuras, informes y registros de sincronización.
- `05_frontend_despliegue/`: bundle Docker autocontenido generado por el
  cuaderno 05; se ignora en Git por su tamaño y contiene su propia guía.
- `scripts_auxiliares/`: entrenamiento reproducible, evaluación, auditoría y sincronización.
- `Documento_final_paper/` y `Presentación_BEAMER/`: entregables académicos.

## Criterios del proyecto

- Mantener evidencia trazable por video, chunk, etiqueta, score y artefacto.
- Separar entrenamiento, selección/calibración en validación y evaluación final en test.
- Usar revisión humana para decisiones ambiguas y para cualquier piloto operativo.
- No almacenar credenciales. Las llamadas opcionales a servicios externos para etiquetado deben estar autorizadas, documentadas y separadas del clasificador de producción.
- No usar APIs para el clasificador operativo ni para sustituir una evaluación humana independiente.
