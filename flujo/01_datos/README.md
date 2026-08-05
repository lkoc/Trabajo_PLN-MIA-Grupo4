# Etapa 01 · Scraping, limpieza y troceado

## Orden

1. `01_01_scraping_incremental.ipynb`
2. `01_02_limpieza_troceado_incremental.ipynb`
3. `01_03_ampliacion_dirigida.ipynb`, solo cuando se incorporen nuevas fuentes.

Entrada: candidatos con `video_id` y URL, transcripciones canónicas y caché local.  
Salida: transcripciones JSONL y chunks v2 con tiempos, hash de transcripción y versión del troceador.  
Control: nunca se descarga audio o video; un `video_id` canónico se omite y un caché válido se reutiliza antes de consultar la red.

Para ampliar la muestra, agregue filas a `datos/raw/video_candidates.jsonl` y vuelva a ejecutar `01_01`. El corpus previo permanece intacto.

