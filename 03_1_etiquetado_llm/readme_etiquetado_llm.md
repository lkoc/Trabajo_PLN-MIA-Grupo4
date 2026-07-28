# Etiquetado local con LLM

Este módulo automatiza el etiquetado multi-etiqueta de los chunks del proyecto sin usar APIs comerciales ni enviar datos fuera de la máquina. Usa la API local de LM Studio, impone un esquema JSON estricto, valida las reglas de la taxonomía y guarda cada lote de forma incremental para poder reanudar una corrida interrumpida.

## Decisión técnica

La máquina disponible tiene un Ryzen 7 8845HS, cerca de 29 GB de RAM, una Radeon RX 570 de 4 GB y una Radeon 780M integrada. No hay CUDA. Por ello, el límite práctico no es almacenar un modelo cuantizado de 9–12B, sino la velocidad de inferencia y el soporte parcial de aceleración AMD/Vulkan.

El corpus canónico contiene 69,853 chunks y aproximadamente 9.5 millones de tokens de texto antes de sumar el prompt y las respuestas. Ejecutar un modelo grande una sola vez sobre todo el corpus, sin medirlo antes, sería lento y difícil de auditar. El cuaderno implementa este orden:

1. Verificar archivos, taxonomía, API y modelo cargado.
2. Construir un piloto reproducible y balanceado por canal.
3. Comparar modelos con el mismo prompt, muestra y parámetros.
4. Ejecutar una sola fase por corrida mediante `ETIQUETADO_RUN_MODE`.
5. Guardar por lotes, validar cada objeto y reanudar por `chunk_id`.
6. Enviar a una segunda pasada solo los casos con flags, confianza baja, daño o una muestra de controles seguros.

### Modelos candidatos

La recomendación original era Qwen3 8B Q4 como modelo equilibrado y Gemma 4 12B Q4 como revisor. Al reparar LM Studio, el CLI actual mostró que ya resuelve Qwen3.5 9B; por ello conviene incluirlo en el piloto como reemplazo actualizado de Qwen3 8B, si está disponible en formato GGUF.

| Prioridad | Modelo | Papel sugerido | Motivo |
|---|---|---|---|
| 1 | Qwen3.5 9B Instruct GGUF Q4 | Primera pasada | Equilibrio entre español, seguimiento de instrucciones, tamaño y velocidad |
| 2 | Gemma 4 12B Instruct GGUF Q4 | Revisor o primera pasada si gana el piloto | Mayor capacidad potencial, pero inferencia más lenta |
| 3 | Qwen3 8B Instruct GGUF Q4 | Alternativa estable | Mantiene la recomendación inicial si Qwen3.5 no está disponible |
| 4 | Modelo 4B | Pruebas rápidas | No usar como única fuente de etiquetas finales |

El modelo definitivo no se elige por un benchmark genérico: debe ganar en esta taxonomía por F1 macro, acuerdo exacto, validez del JSON y velocidad. Los 60 registros previamente clasificados sirven como referencia inicial, pero no deben presentarse como *gold standard* humano.

## Autoridad normativa

El cuaderno valida y registra por hash los siguientes archivos de autoridad:

- `modelos/skills/clasificacion_moderacion_peru.md`
- `para_equiquetado_LLM/PROMPT_ETIQUETADO_LLM.md`
- `datos/processed/taxonomia_moderacion.csv`

El modelo no puede inventar etiquetas ni sustituir esas reglas por una política genérica de toxicidad. Las palabras clave se usan únicamente para muestreo o priorización; nunca asignan una etiqueta.

Una prueba real mostró que enviar los documentos completos en cada solicitud tardó más de cuatro minutos por chunk sobre CPU y agotó el timeout. Por ello `PROMPT_MODE='compact'` usa `prompt_operacional_compacto.md`, una compilación que conserva todas las etiquetas, flags, exclusiones, siete pasos, términos peruanos y ejemplos decisivos. `PROMPT_MODE='full'` permanece disponible para auditorías muy pequeñas. Ambos modos registran los hashes de las fuentes para detectar cambios.

El benchmark de esta máquina seleccionó `BATCH_SIZE=2`, `MAX_WORKERS=2` y `MODEL_PARALLEL=2`. El cuaderno mantiene siempre ocupadas las dos ranuras de inferencia aunque un lote anterior tarde más, pero escribe los resultados en el orden reproducible de la selección. También corrige localmente las invariantes mecánicas de confianza/revisión y, si una fila todavía es inválida, vuelve a solicitar solo esa fila en vez de repetir todo el lote.

### Rendimiento medido en esta máquina

La prueba integral con Qwen3.5 9B Q4, CPU AVX2, contexto 16K, razonamiento desactivado y prompt compacto obtuvo:

- primera solicitud: 100 segundos, incluyendo precalentamiento del prefijo;
- segunda solicitud: 26 segundos gracias a reutilización del prefijo;
- aproximadamente 3,150 tokens de entrada y 170 de salida por chunk;
- JSON Schema válido y cero tokens de razonamiento.

A 26 segundos por chunk, procesar 69,853 registros uno a uno tomaría alrededor de 504 horas, unas tres semanas continuas. No inicies producción con esa extrapolación: mide lotes de 2 y 4 en el piloto y considera un modelo 4B o el clasificador del Cuaderno 04 si la tasa sigue siendo insuficiente.

## Archivos

- `03_1_etiquetado_llm.ipynb`: pipeline ejecutable.
- `readme_etiquetado_llm.md`: razonamiento, metodología y operación.
- `INSTRUCTIVO_LM_STUDIO.md`: instalación, modelos y diagnóstico de LM Studio.
- `prompt_operacional_compacto.md`: compilación ejecutable y trazable de las reglas completas.
- `requirements.txt`: dependencias Python del cuaderno.

El cuaderno lee el texto desde `datos/processed/chunks_para_etiquetar.jsonl`. Sus salidas son ligeras y se guardan en `datos/etiquetado/llm_local/`; no duplican `text`, títulos, canal ni tiempos.

## Uso recomendado

1. Sigue `INSTRUCTIVO_LM_STUDIO.md` y carga un modelo con un identificador estable.
2. Abre el cuaderno desde la raíz del proyecto o desde esta carpeta.
3. Ejecuta instalación, configuración y preflight.
4. Define `PRIMARY_MODEL_ID` con el identificador que devuelve `lms ps` o `/v1/models`.
5. Edita el bloque visible al comienzo de la primera celda de código: `RUN_MODE`, `PILOT_SAMPLE_SIZE`, `PRODUCTION_SAMPLE_SIZE`, `REVIEW_SAMPLE_SIZE` y `SAMPLE_SEED`. `PRODUCTION_SAMPLE_SIZE=None` procesa todo el corpus.
6. Usa una única fase por ejecución: `pilot` (predeterminado), `production`, `review` o `validate`. También puedes sobrescribir el modo desde PowerShell con `$env:ETIQUETADO_RUN_MODE='production'`.
7. Compara al menos dos modelos. Selecciona el ganador por calidad y rendimiento medidos.
8. Solo entonces ejecuta el cuaderno con `ETIQUETADO_RUN_MODE=production`.
9. No combines directamente las salidas de dos anotadores: conserva una fila por `(chunk_id, annotator_id)` y consolida en el Cuaderno 03.

## Reanudación e integridad

Cada salida válida se agrega al JSONL y se fuerza a disco al terminar el lote. En una nueva ejecución se leen los `chunk_id` ya presentes y no se vuelven a solicitar. Antes de aceptar una salida se comprueba:

- campos exactos y tipos correctos;
- ID perteneciente al lote y sin duplicados;
- etiquetas y flags presentes en la taxonomía;
- al menos una etiqueta;
- ninguna etiqueta segura junto con daño;
- cualquier flag o confianza menor de 0.70 activa `needs_review`;
- `ironia_ambigua` y `contexto_necesario` limitan la confianza a 0.65;
- ausencia de texto y metadatos fuente en la salida.

Las fases son excluyentes, por lo que no se puede iniciar accidentalmente la revisión sin haber elegido `review`. Esta fase exige tanto la primera salida de producción como `LMSTUDIO_REVIEW_MODEL`.

La producción baraja de forma determinista el corpus completo con `SAMPLE_SEED` y toma los primeros `PRODUCTION_SAMPLE_SIZE` registros. Así la membresía no depende del orden del JSONL. La misma semilla y cantidad producen exactamente los mismos IDs; si aumentas la cantidad, la selección anterior es un prefijo y solo se procesan los IDs nuevos. Cambiar la semilla crea otro archivo de salida. No reduzcas la cantidad sobre un archivo que ya contiene más filas.

Si una fila falla después de los reintentos, la corrida se detiene. Las filas válidas del mismo lote se conservan en memoria y solo se reenvían las inválidas; la escritura sigue el orden reproducible de la selección. Al reanudar, comienza en el primer ID pendiente.

## Métricas de avance en vivo

Después de cada lote el cuaderno actualiza, sin imprimir una tabla nueva cada vez, un panel con:

- chunks completados, pendientes y porcentaje total;
- chunks seguros y chunks con al menos una etiqueta de daño;
- conteo acumulado de cada etiqueta;
- conteo acumulado de cada flag;
- casos `needs_review=true` y su porcentaje;
- confianza media acumulada.

La barra de `tqdm` muestra además completados, daño, revisión y confianza como resumen compacto. Los conteos se reconstruyen desde las filas válidas del JSONL al reanudar, por lo que no empiezan de cero.

Después de cada lote también se actualiza atómicamente un archivo lateral con el mismo nombre base y sufijo `.metrics.json`. Por ejemplo:

```text
qwen-local-primary_labeled_chunks_seed42.jsonl
qwen-local-primary_labeled_chunks_seed42.metrics.json
qwen-local-primary_labeled_chunks_seed42.manifest.json
```

Este archivo permite consultar el avance sin abrir ni recorrer el JSONL completo mientras la corrida está activa.

## Evaluación

El cuaderno puede medir contra una referencia disponible:

- coincidencia exacta del conjunto de etiquetas;
- Jaccard medio multi-etiqueta;
- precisión, recall y F1 por etiqueta;
- F1 macro;
- porcentaje de JSON válido;
- chunks por minuto;
- proporción enviada a revisión.

Una evaluación académicamente más sólida requiere 300–500 casos revisados por humanos y estratificados por categoría, canal, ironía y ambigüedad. A mediano plazo, 1,000–2,000 etiquetas humanas permitirían entrenar el clasificador del Cuaderno 04 y reservar el LLM local para casos inciertos.

## Salidas y compatibilidad

El formato final contiene exactamente:

```text
chunk_id, labels, flags, needs_review, notes, annotator_type,
annotator_id, annotator_model, skill_file, score_confianza,
justificacion, annotated_at
```

`chunk_id` permite recuperar el texto desde el JSONL canónico. Este contrato es compatible con la consolidación del Cuaderno 03 y con el cruce validado por ID del Cuaderno 04.
