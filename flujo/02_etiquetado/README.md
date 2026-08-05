# Etapa 02 · Etiquetado semiautomático

## Orden

1. `02_01_etiquetado_local_ollama.ipynb` — Ollama local oficial o Hugging Face/Qwen sobre Colab L4 como backend opcional.
2. `02_02_etiquetado_remoto.ipynb` — opcional y con activación explícita.
3. `02_03_revision_llm_dirigida.ipynb` — discrepancias y baja confianza.
4. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. El proceso reanuda por `chunk_id` y no vuelve a pagar ni recalcular filas completas.

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La sugerencia LLM permanece oculta hasta que el revisor decide mostrarla. La interfaz impide combinar `SEGURO` con daño y admite diferir casos.

La alternativa Colab no ejecuta Ollama: usa el adaptador Hugging Face bajo el mismo contrato. Solo sincroniza `chunks_v2.jsonl.gz`; las anotaciones se escriben en `/content` y se publican a Drive como un run reanudable. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
