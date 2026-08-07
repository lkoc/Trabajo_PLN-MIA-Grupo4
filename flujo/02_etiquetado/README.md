# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 02 · Etiquetado semiautomático

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

## Orden

1. `02_00_preparacion_bundle_colab.ipynb` — se ejecuta en Colab, descarga el bundle sincronizado de GitHub o recibe los cuatro archivos locales mediante el navegador, verifica identidad y SHA-256 y publica la versión inmutable en Drive. Ejecútelo antes de `02_01` y nuevamente después de `02_05`.
2. `02_01_etiquetado_local_ollama.ipynb` — nombre conservado por compatibilidad; implementa la cascada calibrada `deepseek-v4-flash`→`deepseek-v4-pro`, con panel pareado, bootstrap por video, lotes 5×32, presupuesto, cuarentena y reanudación.
3. `02_02_etiquetado_remoto.ipynb` — fallback local independiente `Qwen/Qwen3-1.7B`; no se mezcla con la campaña principal.
4. `02_03_revision_llm_dirigida.ipynb` — recupera y presenta calibración, cobertura y revisión Pro sin repetir API.
5. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.
6. `02_05_cierre_humano_snapshot.ipynb` — reaplica el último evento humano, recupera `video_id` desde el chunk fuente y congela el snapshot entrenable.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. El proceso reanuda por `chunk_id` y no vuelve a pagar ni recalcular filas completas.

Los seis cuadernos muestran barras `tqdm` en las operaciones potencialmente largas. `02_00` informa descarga y copia a Drive; `02_01` cuenta chunks, errores, velocidad y costo real en calibración, primera pasada y revisión; `02_02` muestra el fallback local; `02_03` informa la lectura de artefactos; `02_04` separa lectura, carga y consolidación; `02_05` separa eventos humanos, reconciliación, deduplicación y validación de splits. El preflight `/models` no envía corpus.

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La sugerencia LLM permanece oculta hasta que el revisor decide mostrarla. La interfaz impide combinar `SEGURO` con daño y admite aceptar, modificar, diferir o excluir. También recupera de la versión histórica contexto vecino, enlace temporal al video, guía, progreso, filtros, borradores, atajos y exportación, sin incrustar la campaña en el HTML.

Los eventos humanos no alteran el archivo LLM. `02_05` construye una vista derivada por precedencia y después un snapshot inmutable. Si no existen eventos humanos —la revisión es opcional— conserva las decisiones automáticas resueltas; nunca convierte una abstención en `SEGURO`.

La cascada puede ejecutarse localmente o en Colab; la API no necesita GPU. En Colab, `DEEPSEEK_API_KEY` se obtiene del secreto homónimo y nunca se versiona. `02_00` usa la autorización integrada para publicar `bundle_releases/<bundle_id>` y actualizar `latest.json`; no requiere Google Cloud Console ni Drive Desktop. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md) y [`docs/METODOLOGIA_ETIQUETADO_CASCADA.md`](../../docs/METODOLOGIA_ETIQUETADO_CASCADA.md).
