# Datos

Carpeta de trabajo para insumos y datasets generados por los cuadernos.

## Subcarpetas

- `raw/`: metadatos, subtitulos y transcripciones originales.
- `interim/`: texto normalizado y archivos intermedios.
- `processed/`: chunks listos para etiquetar y datasets finales. `chunks_para_etiquetar.jsonl` es la fuente canónica del texto.
- `etiquetado/`: exportaciones ligeras del frontend; contienen `chunk_id`, etiquetas y metadatos, pero no duplican el texto.

Las anotaciones se relacionan con el texto mediante `chunk_id`. El cuaderno 03 valida y consolida estas referencias; el cuaderno 04 realiza el cruce con `processed/chunks_para_etiquetar.jsonl` antes de entrenar.

No subir datos sensibles, innecesarios o pesados. Conservar solo lo requerido para reproducibilidad academica.
