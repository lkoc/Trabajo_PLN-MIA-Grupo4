# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## VTT consolidados por video

Esta carpeta es el checkpoint sincronizable de las pistas WebVTT. Los archivos
mantienen el nombre `video_id.idioma.vtt`; si existen varias pistas se conservan
todas sin sobrescribir bytes diferentes.

`index.json` registra `video_id`, idioma, procedencia, canal según el mismo
descriptor usado por `transcripts_by_channel`, tamaño, validez y SHA-256.
`missing_vtt.jsonl` contiene únicamente las transcripciones canónicas que aún no
tienen un VTT válido y es la cola de reanudación de `01_01`.

Los archivos `*.transcript-api.vtt` son una serialización explícitamente marcada
de segmentos obtenidos mediante `youtube-transcript-api`; no son archivos VTT
originales descargados por `yt-dlp`.
