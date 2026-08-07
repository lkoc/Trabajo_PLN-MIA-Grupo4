# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 02 · Etiquetado semiautomático

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

## Orden

1. `02_00_preparacion_bundle_colab.ipynb` — con kernel local, publica en Drive la versión inmutable requerida por Colab; ejecútelo antes de `02_01` y nuevamente después de `02_05` antes de entrenar en Colab.
2. `02_01_etiquetado_local_ollama.ipynb` — Ollama local oficial o Hugging Face/Qwen sobre Colab L4 como backend opcional.
3. `02_02_etiquetado_remoto.ipynb` — opcional y con activación explícita.
4. `02_03_revision_llm_dirigida.ipynb` — discrepancias y baja confianza.
5. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.
6. `02_05_cierre_humano_snapshot.ipynb` — reaplica el último evento humano, recupera `video_id` desde el chunk fuente y congela el snapshot entrenable.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. El proceso reanuda por `chunk_id` y no vuelve a pagar ni recalcular filas completas.

Los seis cuadernos muestran barras `tqdm` en las operaciones potencialmente largas. `02_00` informa construcción, compresión y copia; `02_01` y `02_02` informan etiquetados y errores; `02_03` muestra el recorrido que construye la cola dirigida; `02_04` separa lectura, carga y consolidación; `02_05` separa eventos humanos, reconciliación, deduplicación y validación de splits. Las etapas instantáneas de preflight no crean barras artificiales.

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La sugerencia LLM permanece oculta hasta que el revisor decide mostrarla. La interfaz impide combinar `SEGURO` con daño y admite aceptar, modificar, diferir o excluir. También recupera de la versión histórica contexto vecino, enlace temporal al video, guía, progreso, filtros, borradores, atajos y exportación, sin incrustar la campaña en el HTML.

Los eventos humanos no alteran el archivo LLM. `02_05` construye una vista derivada por precedencia y después un snapshot inmutable. Si no existen eventos humanos —la revisión es opcional— conserva las decisiones automáticas resueltas; nunca convierte una abstención en `SEGURO`.

La alternativa Colab no ejecuta Ollama: usa el adaptador Hugging Face bajo el mismo contrato. El transporte es exclusivamente Google Drive: `02_00` publica `bundle_releases/<bundle_id>` y actualiza `bundle_releases/latest.json`; el bootstrap lee ese puntero y activa la versión después de verificar todos sus SHA-256. Las anotaciones se escriben en `/content` y se publican a Drive como un run reanudable. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md).
