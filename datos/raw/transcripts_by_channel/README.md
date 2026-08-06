# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Transcripciones particionadas por canal

Esta carpeta es el checkpoint sincronizable del corpus de subtítulos. Cada JSONL contiene videos de un solo canal; los canales grandes usan partes `part-0001`, `part-0002`, etc., de hasta 25 MiB. `index.json` registra canal, parte, cantidad de videos, tamaño y SHA-256 de cada archivo.

`datos/raw/transcripts_raw.jsonl` se conserva localmente como vista canónica y no se borra ni se versiona. El cuaderno `01_01` materializa inicialmente estas particiones y, después, añade cada transcripción nueva directamente al archivo de su canal. La escritura es idempotente por `video_id`.

Después de clonar el repositorio en otra máquina, ejecute:

```bash
python tools/restore_synced_checkpoints.py
```

El comando verifica hashes, recompone el canónico sin sobrescribir sus filas existentes y restaura las entradas comprimidas del bundle. `transcripts_cache/` continúa siendo una optimización local y no se sincroniza.
