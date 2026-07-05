# Guia de estructura para el paper IEEE

Esta guia define la estructura del articulo final sobre un moderador textual local para videos peruanos de YouTube. El documento debe escribirse como paper tecnico IEEE y no como tesis.

## Estructura recomendada

1. Titulo: debe mencionar moderacion de contenido, videos de YouTube y enfoque local o auditable.
2. Resumen: problema, brecha, metodo, resultado esperado y alcance.
3. Palabras clave: PLN, moderacion de contenido, clasificacion de texto, YouTube y corpus peruano.
4. Introduccion: contexto, dificultad local, necesidad de trazabilidad y contribuciones.
5. Trabajo relacionado: modelos de lenguaje, transcripcion local, evaluacion y moderacion textual.
6. Corpus y recoleccion: canales candidatos, criterios de descarte y restriccion sin APIs.
7. Metodo: limpieza, chunks, etiquetado humano, categorias, baseline y extension avanzada.
8. Evaluacion: acuerdo humano, split, F1 macro, precision, recall y analisis de errores.
9. Resultados: metricas del baseline, matriz de confusion y revision de fragmentos.
10. Discusion y limitaciones: sesgo por canal, ironia, sarcasmo, citas y alcance textual.
11. Conclusiones: contribucion, utilidad y trabajo futuro.

## Figuras minimas

- Pipeline general del sistema.
- Flujo de etiquetado humano.
- Agregacion de riesgo por video.

## Tablas minimas

- Categorias de moderacion.
- Fuentes candidatas por tipo de canal.
- Metricas del baseline.
- Casos de error frecuentes.

## Hilo conductor

Lenguaje local en videos peruanos -> fragmentos textuales auditables -> etiquetado humano -> clasificador local -> evidencia para revision humana.
