# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 02 · Etiquetado semiautomático

## Orden

1. `02_01_etiquetado_local_ollama.ipynb` — Ollama local oficial o Hugging Face/Qwen sobre Colab L4 como backend opcional.
2. `02_02_etiquetado_remoto.ipynb` — opcional y con activación explícita.
3. `02_03_revision_llm_dirigida.ipynb` — discrepancias y baja confianza.
4. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.
5. `02_05_cierre_humano_snapshot.ipynb` — reaplica el último evento humano, recupera `video_id` desde el chunk fuente y congela el snapshot entrenable.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. El proceso reanuda por `chunk_id` y no vuelve a pagar ni recalcular filas completas.

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La sugerencia LLM permanece oculta hasta que el revisor decide mostrarla. La interfaz impide combinar `SEGURO` con daño y admite aceptar, modificar, diferir o excluir. También recupera de la versión histórica contexto vecino, enlace temporal al video, guía, progreso, filtros, borradores, atajos y exportación, sin incrustar la campaña en el HTML.

Los eventos humanos no alteran el archivo LLM. `02_05` construye una vista derivada por precedencia y después un snapshot inmutable. Si no existen eventos humanos —la revisión es opcional— conserva las decisiones automáticas resueltas; nunca convierte una abstención en `SEGURO`.

La alternativa Colab no ejecuta Ollama: usa el adaptador Hugging Face bajo el mismo contrato. Solo sincroniza `chunks_v2.jsonl.gz`; las anotaciones se escriben en `/content` y se publican a Drive como un run reanudable. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
