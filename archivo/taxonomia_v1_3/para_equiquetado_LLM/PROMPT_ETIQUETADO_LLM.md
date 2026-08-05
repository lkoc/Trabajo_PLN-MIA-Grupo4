# Prompt operativo: etiquetado del corpus peruano

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


## Archivos de autoridad
Antes de clasificar, lee completos `clasificacion_moderacion_peru.md` y
`taxonomia_moderacion.csv`. Esos dos archivos son la autoridad normativa para esta tarea.
No sustituyas sus criterios por categorías aprendidas previamente, políticas genéricas de
toxicidad ni inferencias propias. Si algo no está sustentado por esos criterios, no inventes
una etiqueta: aplica `contexto_necesario` cuando corresponda.

El archivo de entrada es `chunks_para_etiquetar.jsonl`. Cada línea es un objeto independiente.
Usa el texto y el contexto del registro para aplicar, en orden, los siete pasos del skill.
La clasificación es multi-etiqueta y debe reflejar todas las categorías aplicables.

## Tarea
Etiqueta cada chunk sin omitir, duplicar ni reordenar registros. Conserva exactamente su
`chunk_id`. Evalúa cada caso con los criterios del skill; no reemplaces la evaluación por una
búsqueda simple de palabras clave ni por un clasificador heurístico. Nunca dejes `labels` vacío:
si no existe daño usa `seguro` o, únicamente cuando corresponda, `seguro_ironia_marcada`.
`seguro` no puede coexistir con una etiqueta de daño.

Respeta estrictamente la separación entre categorías principales y flags transversales:
`ironia_ambigua`, `humor_encubridor` y `contexto_necesario` van exclusivamente en `flags`,
nunca en `labels`. Un flag no sustituye la categoría principal. En particular,
`humor_encubridor` debe registrarse junto con la categoría o categorías de daño que el humor
encubre; por ejemplo, `labels=["racismo_encubierto"]` y
`flags=["humor_encubridor"]`.
Si el análisis produce únicamente flags pero ninguna categoría de daño, no dejes `labels`
vacío: elimina esos flags y usa `labels=["seguro"]`. Los flags solo se conservan cuando existe
al menos una categoría de daño sustentada por el texto.

Usa un identificador constante de tres caracteres para todo el archivo: `CGT` para ChatGPT,
`GEM` para Gemini o `DSK` para DeepSeek. Registra en `annotator_model` el nombre exacto del
modelo utilizado y en `skill_file` el valor `clasificacion_moderacion_peru.md`.

## Salida obligatoria
Genera un archivo descargable `<id>_labeled_chunks.jsonl`, con un objeto JSON por línea y sin
bloques Markdown. Cada objeto debe contener exactamente estos campos:

- `chunk_id`
- `labels`
- `flags`
- `needs_review`
- `notes`
- `annotator_type` = `llm`
- `annotator_id`
- `annotator_model`
- `skill_file`
- `score_confianza`
- `justificacion`
- `annotated_at` en ISO 8601

No copies `text`, `video_id`, títulos, canal, tiempos, `text_hash` ni ningún otro campo del
chunk original. `ejemplo_formato_salida.jsonl` es únicamente una referencia estructural.

## Autoverificación antes de entregar
1. La cantidad de salidas coincide con la cantidad de chunks efectivamente procesados.
2. No hay `chunk_id` vacío, inventado o duplicado.
3. Todos los valores de `labels` y `flags` existen en la taxonomía.
   Las filas cuya categoría sea `FLAG` aparecen solo en `flags`; nunca en `labels`.
4. `labels` nunca está vacío y `seguro` no coexiste con daño.
5. Cualquier flag, o confianza menor que 0.70, activa `needs_review=true`.
6. `ironia_ambigua` o `contexto_necesario` limitan `score_confianza` a 0.65.
7. La salida no contiene el texto ni otros campos fuente.
8. Cada justificación se basa en el criterio concreto del skill, no en categorías externas.

El corpus es grande. Si la plataforma no puede completarlo en una sola ejecución, trabaja en
partes consecutivas y entrega cada parte como JSONL, indicando con precisión el primer y último
`chunk_id` procesados. No afirmes que terminaste filas que no evaluaste.
