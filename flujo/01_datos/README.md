# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 01 · Scraping, limpieza y troceado

## Orden

1. `01_01_scraping_incremental.ipynb`
2. `01_02_limpieza_troceado_incremental.ipynb`
3. `01_03_ampliacion_dirigida.ipynb`, solo cuando se incorporen nuevas fuentes.

Entrada: candidatos con `video_id` y URL, snapshots históricos, transcripciones canónicas y caché local.  
Salida: transcripciones JSONL y chunks v2 con tiempos, hash de transcripción y versión del troceador.  
Control: nunca se descarga audio o video; primero se consolidan sin modificarlos los `transcripts_raw.jsonl` ya existentes, después se reutiliza el caché y solo al final se consulta la red para un `video_id` nuevo. La limpieza conserva la eliminación de hasta 12 palabras solapadas en subtítulos rodantes, el cierre a 30 segundos/600 caracteres y el mínimo de 90 caracteres.

Para ampliar la muestra, agregue filas a `datos/raw/video_candidates.jsonl` o `datos/raw/videos_candidatos.csv` y vuelva a ejecutar `01_01`. El corpus previo permanece intacto; `FETCH_NEW=False` permite comprobar la reutilización sin usar red.
