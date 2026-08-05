# Etapa 02 · Etiquetado semiautomático

## Orden

1. `02_01_etiquetado_local_ollama.ipynb` — ruta local oficial.
2. `02_02_etiquetado_remoto.ipynb` — opcional y con activación explícita.
3. `02_03_revision_llm_dirigida.ipynb` — discrepancias y baja confianza.
4. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. El proceso reanuda por `chunk_id` y no vuelve a pagar ni recalcular filas completas.

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La sugerencia LLM permanece oculta hasta que el revisor decide mostrarla. La interfaz impide combinar `SEGURO` con daño y admite diferir casos.

