# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 02 · Etiquetado semiautomático

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

## Orden

1. `02_00_preparacion_bundle_colab.ipynb` — se ejecuta en Colab, descarga el bundle sincronizado de GitHub o recibe los nueve archivos locales mediante el navegador, verifica identidad y SHA-256 y publica la versión inmutable en Drive. Ejecútelo antes de `02_01` y nuevamente después de `02_05`.
2. `02_01_etiquetado_local_ollama.ipynb` — nombre conservado por compatibilidad; recupera únicamente equivalencias históricas exactas 1:1 y ejecuta la cascada `deepseek-v4-flash`→`deepseek-v4-pro` sobre pendientes, con `thinking=disabled`, contrato JSON validado, caché, saldo, persistencia por grupos de 5, checkpoints atómicos, presupuesto, cuarentena y reanudación.
3. `02_02_etiquetado_remoto.ipynb` — fallback local independiente `Qwen/Qwen3-1.7B`; no se mezcla con la campaña principal.
4. `02_03_revision_llm_dirigida.ipynb` — recupera y presenta calibración, cobertura y revisión Pro sin repetir API.
5. `02_04_consolidacion_validacion_humana.ipynb` — precedencia y frontend.
6. `02_05_cierre_humano_snapshot.ipynb` — reaplica el último evento humano, recupera `video_id` desde el chunk fuente y congela el snapshot entrenable.

Cada salida conserva modelo, prompt, taxonomía, confianza, flags y estado de revisión. Las etiquetas de entrada se interpretan sin distinguir mayúsculas de minúsculas y se guardan con la forma canónica en mayúsculas. El proceso reanuda por `chunk_id`, migra manifiestos anteriores compatibles y no vuelve a pagar ni recalcular filas completas. Ante `Ctrl+C`, termina y guarda las solicitudes ya iniciadas antes de publicar el checkpoint.

Los seis cuadernos muestran barras `tqdm` en las operaciones potencialmente largas. `02_00` informa descarga y copia a Drive; `02_01` cuenta chunks, errores, velocidad, caché, costo estimado y saldo periódico en calibración, primera pasada y revisión; `02_02` muestra el fallback local; `02_03` informa la lectura de artefactos; `02_04` separa lectura, carga y consolidación; `02_05` separa eventos humanos, reconciliación, deduplicación y validación de splits. El preflight `/models` y la consulta de saldo no envían corpus.

## Metodología vigente

1. Recuperar solo coincidencias históricas unívocas por
   `(video_id, texto_normalizado)`; nunca copiar por posición o similitud.
2. Validar credencial, modelos Flash/Pro, modo no razonador y JSON antes de
   transmitir texto.
3. Calibrar ambos modelos sobre el mismo panel balanceado de 1 000 chunks.
4. Procesar con Flash solo los `chunk_id` pendientes y persistir cada grupo de
   cinco con `fsync`; se usan hasta 32 solicitudes concurrentes.
5. Enviar a Pro todo daño, las 36 000 abstenciones Flash de menor confianza y
   los seguros con confianza `< 0.85`, con contexto vecino. La reanudación
   presupuestada usa un control seguro aleatorio reproducible del 1 %, conserva
   toda revisión Pro existente y limita el gasto nuevo a US$14.50; el 0.95 permanece como
   resultado de calibración diagnóstico.
6. Consolidar con precedencia Pro→Flash y cerrar mediante decisiones humanas
   append-only; una abstención nunca se convierte automáticamente en `SEGURO`.

### Reanudación manual exclusivamente Pro

Después de recargar, reinicie el kernel y ejecute las celdas en orden hasta
**“Enrutamiento y revisión dirigida con Pro”**. Los parámetros activos dejan
`RUN_CALIBRATION=False`, `RUN_PRIMARY=False` y `RUN_DIRECTED_REVIEW=True`, por
lo que no se repiten calibración ni llamadas Flash. La última celda reconstruye
la cola, descuenta los `chunk_id` ya presentes en `review_pro.jsonl` y muestra
la previsión antes de transmitir corpus. Si el saldo es menor que US$15.00 o la
proyección supera US$14.50, se detiene antes de iniciar Pro.

El prompt operacional y la taxonomía se colocan al principio de cada solicitud
para que la caché automática de prefijo pueda reutilizarlos. Flash y Pro reciben
el mismo contrato de raíz `annotations`; se comprueba cantidad, orden de
`chunk_id`, tipos, exclusividad de `SEGURO` y longitud de `notes` antes de
guardar. Una fila inválida se reenvía individualmente y no obliga a repetir las
filas válidas del grupo.

## Corte cuantitativo disponible

El checkpoint documentado del **2026-08-08 13:15:01 (UTC−05)** registró:

| Indicador medido | Resultado |
|---|---:|
| recuperación Flash histórica | 52 244/69 853 (74.79 %) |
| recuperación Pro histórica | 9 912/13 421 (73.85 %) |
| calibración Flash | 1 000, 0 errores, 97.250 s, 616.969 chunks/min, US$0.073349 |
| calibración Pro | 1 000, 0 errores, 122.593 s, 489.424 chunks/min, US$0.269223 |
| caché de entrada Flash / Pro | 64.22 % / 53.72 % |
| ahorro medido frente a no usar caché | 45.53 % / 37.65 % |
| acuerdo exacto Flash–Pro a 0.95 | 80.41 %; Wilson inferior 77.10 % |
| acuerdo binario daño/seguro a 0.95 | 99.77 %; Wilson inferior 98.97 % |
| primera pasada nueva | 14 399/114 696 válidos, 1 error rechazado, 867.279 chunks/min |
| costo incremental del tramo | US$0.908781; US$0.06311 por 1 000 válidos |
| caché Flash acumulada | 75.17 % |
| revisiones Pro ya persistidas | 29 270; nunca se eliminan al reconstruir la cola |
| cola crítica y control 1 % | 40 695 pendientes; US$13.66 y 90.0 min proyectados |
| saldo verificado antes de recarga | US$5.75; con US$10 serían aproximadamente US$15.75 |

El acuerdo exacto no alcanzó el criterio predeclarado y el estado de
calibración es `inconclusive_conservative_threshold`. Estas son comparaciones
entre modelos, **no exactitud humana**. Manteniendo las condiciones iniciales,
la primera pasada pendiente se proyectaba en unas 2 h 12 min y US$7.24; tiempo
y costo finales todavía no estaban cerrados. Consulte el
[corte cuantitativo completo](../../resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md)
y la [metodología central](../../docs/METODOLOGIA_ETIQUETADO_CASCADA.md).

```powershell
modperu serve-labeling --campaign datos/etiquetado/consolidado/anotaciones_v2.jsonl
```

La barra superior identifica el proyecto y ofrece cuatro vistas calculadas desde
la campaña: **Urgentes** (2.705 conflictos de consolidación), **Prioritarios
Pro** (44.617 salidas Pro con `needs_review=true` o alguna categoría de daño,
sin duplicar su intersección), **Todos** (166.940 chunks) y **Excluidos**
(decisión vigente `reject` o estado de origen `excluded`). Esta última vista
permite auditar y revertir una exclusión mediante una nueva decisión. Los
conteos no están fijados en el HTML y se actualizan desde la API.

La sugerencia LLM se muestra desde el inicio junto al chunk para acelerar la revisión, sin presentarla como verdad humana. La vista de escritorio cabe en una sola altura: contexto anterior/posterior, detalle de propuesta, notas, guía e información del proyecto se abren en diálogos. La interfaz impide combinar `SEGURO` con daño y solo habilita flags cuando existe al menos una categoría de daño. Admite aceptar, modificar, diferir o excluir; un diferido se conserva, pero sigue pendiente hasta recibir una decisión resolutiva. Las acciones masivas pueden aceptar la propuesta propia de cada chunk, aplicar una misma clasificación humana (`SEGURO` o una combinación de daños y flags) o excluir por video/canal. Muestran el número afectado y exigen confirmación; “Clasificar todo igual” incluye inicialmente los ya resueltos para abarcar todo el alcance, aunque puede limitarse a pendientes. El diálogo de filtros combina cola, cohorte, categorías, flags y estado de etiquetado; entre bloques usa AND y permite exigir cualquiera o todas las selecciones dentro de categorías y flags. También conserva enlace temporal al video, progreso, borradores, atajos `A`/`R`/`Ctrl+Enter`/flechas y exportación, sin incrustar la campaña en el HTML. Estas invariantes también se validan en el servidor, por lo que no dependen únicamente del navegador. Consulte la [matriz de paridad de los frontends](../../docs/PARIDAD_FRONTENDS_ACTIVOS.md).

El filtro de completitud añade **Sin etiqueta** y **Con etiqueta**. El primero
selecciona chunks sin ninguna categoría efectiva y no debe confundirse con
**Pendiente**, que puede conservar una sugerencia automática.

Los botones distinguen cuatro efectos: **Aceptar sugerencia** confirma exactamente la propuesta automática; **Guardar mi decisión** guarda las categorías y flags seleccionados por la persona revisora cuando discrepa del modelo; **Revisar después (diferir)** deja el chunk pendiente y fuera del entrenamiento hasta una decisión final; **Excluir del dataset** descarta el chunk del conjunto entrenable sin convertirlo en `SEGURO`. La misma explicación está disponible en el diálogo “¿Qué hace cada botón?”.

Los eventos humanos no alteran el archivo LLM. `02_05` construye una vista derivada por precedencia y después un snapshot inmutable. Si no existen eventos humanos —la revisión es opcional— conserva las decisiones automáticas resueltas; nunca convierte una abstención en `SEGURO`.

La cascada puede ejecutarse localmente o en Colab; la API no necesita GPU. En Colab, `DEEPSEEK_API_KEY` se obtiene del secreto homónimo y nunca se versiona. `02_00` usa la autorización integrada para publicar `bundle_releases/<bundle_id>` y actualizar `latest.json`; no requiere Google Cloud Console ni Drive Desktop. Consulte [`docs/COLAB_L4.md`](../../docs/COLAB_L4.md) y [`docs/METODOLOGIA_ETIQUETADO_CASCADA.md`](../../docs/METODOLOGIA_ETIQUETADO_CASCADA.md).
