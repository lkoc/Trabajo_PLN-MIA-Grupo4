# Orden reproducible de ejecución

| Paso | Cuaderno | Entrada principal | Salida principal | Reanudación |
|---:|---|---|---|---|
| 1 | `01_01_scraping_incremental` | candidatos + corpus/caché existente | transcripciones JSONL | `video_id` |
| 2 | `01_02_limpieza_troceado_incremental` | transcripciones | chunks v2 | `chunk_id` y hash |
| 3 | `01_03_ampliacion_dirigida` | déficits de datos | candidatos adicionales | solo videos nuevos |
| 4 | `02_01_etiquetado_local_ollama` | chunks pendientes | anotaciones Ollama | `chunk_id` |
| 5 | `02_02_etiquetado_remoto` | muestra opcional | anotaciones remotas | `chunk_id` |
| 6 | `02_03_revision_llm_dirigida` | duda/desacuerdo | cola priorizada | selección determinista |
| 7 | `02_04_consolidacion_validacion_humana` | campañas y eventos | anotación consolidada | eventos append-only |
| 8–15 | `03_01`–`03_08` | snapshot v2 congelado | modelos, métricas y registro | checkpoint/run ID |
| 16 | `04_01_frontend_produccion` | registro v2 | demostrador local | SQLite/JSONL append-only |

Reglas globales:

- ningún notebook instala dependencias;
- ninguna etapa sobrescribe su fuente;
- los nuevos datos se añaden por ID y hash;
- los manifiestos registran contrato, entradas, salidas, código y hardware;
- validation selecciona y calibra; test informa después de congelar.

