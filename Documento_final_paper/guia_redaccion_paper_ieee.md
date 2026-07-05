# Guia de redaccion para el paper IEEE

Esta guia controla el estilo de redaccion del articulo final. El texto debe ser directo, tecnico y verificable.

## Tono

- Escribir como articulo cientifico aplicado.
- Evitar tono administrativo o de informe de avance.
- Usar frases breves y conectadas.
- No prometer resultados que aun no fueron medidos.
- Distinguir claramente entre implementacion, experimento y propuesta futura.

## Contribuciones

La introduccion debe cerrar con contribuciones concretas:

1. Un flujo local sin APIs para recolectar y preparar texto de videos publicos.
2. Una guia de categorias para moderacion textual en contenido peruano.
3. Un prototipo de etiquetado humano y entrenamiento local.
4. Un mecanismo de salida con evidencia textual revisable.

## Redaccion de metodo

Cada paso metodologico debe indicar entrada, proceso y salida.

```text
La limpieza recibe subtitulos o transcripciones crudas, normaliza marcas no textuales y genera fragmentos de longitud controlada. La salida es un archivo JSONL con identificador de video, posicion del fragmento y texto auditable.
```

## Citas

- Usar `\cite{clave}` con estilo IEEE.
- Citar una fuente cuando se mencione un modelo, herramienta o metodo publicado.
- No citar URLs de canales como evidencia cientifica; los canales son fuentes de datos y deben documentarse en la seccion de corpus.
- No inventar DOI, autores ni anios.

## Figuras

- Cada figura debe ser citada en el texto antes o inmediatamente despues de aparecer.
- El caption debe explicar la funcion de la figura.
- Las figuras del pipeline deben mostrar entradas, procesamiento, salida y revision humana.

## Tablas

- Usar tablas compactas.
- No llenar tablas con texto largo.
- Las categorias de moderacion deben incluir definicion breve y ejemplo de decision.

## Resultados

Cuando existan metricas, reportarlas con precision: dataset usado, numero de fragmentos, distribucion por clase, split, metrica principal y errores frecuentes.

Si aun no hay metricas finales, escribir "resultado esperado" o "criterio de evaluacion", no "resultado".

## Lenguaje recomendado

Preferir:

- "el sistema prioriza revision humana";
- "el clasificador sugiere una categoria";
- "el fragmento activador permite auditar la salida";
- "el baseline establece una referencia reproducible".

Evitar:

- "el sistema elimina contenido";
- "el modelo decide si un video es ilegal";
- "el algoritmo detecta la verdad";
- "se garantiza precision".
