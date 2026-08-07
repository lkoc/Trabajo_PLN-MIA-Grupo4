# Metodología de etiquetado en cascada Flash→Pro

Última actualización: **2026-08-07**.

## Decisión vigente

El flujo activo reproduce y fortalece la campaña histórica: `deepseek-v4-flash`
realiza la primera pasada económica y `deepseek-v4-pro` revisa de manera
dirigida. El fallback local `Qwen/Qwen3-1.7B` queda como diagnóstico separado;
no se mezclan sus métricas ni sus salidas con la campaña principal.

El contrato v2.1 aprende `SEGURO`, `RACISMO_DISCRIMINACION`,
`ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO`
es excluyente; los cuatro daños pueden coexistir. Una abstención se difiere y no
entra al entrenamiento.

## Evidencia histórica recuperada

El [instructivo histórico de la API](../archivo/contrato_4_danos_seguro_derivado/03_2_etiquetado_llm_api/INSTRUCTIVO_API.md)
y su cuaderno `03_2_etiquetado_llm_api` emplearon razonamiento desactivado,
32 solicitudes concurrentes y cinco chunks por solicitud. La primera pasada
Flash produjo **69 853** propuestas y Pro revisó **10 000**. Bajo la regla
conservadora histórica —revisar `needs_review` o confianza menor que 0.90— la
cobertura automática fue **91.24 %**, el acuerdo exacto Flash–Pro **93.19 %** y
el acuerdo binario **96.55 %**. Estas cifras describen acuerdo entre modelos, no
exactitud frente a verdad humana.

La migración activa conserva `deepseek-v4-flash`→`deepseek-v4-pro`, el preflight
`/models`, 32 solicitudes, cinco chunks por solicitud, piloto Flash de 300,
piloto Pro de 500, contexto vecino, reanudación, `notes` de 160 caracteres,
rescate de flags y reenvío individual de filas inválidas. Amplía el esquema con
un panel pareado de 1 000, balance por canal/video, Wilson, bootstrap agrupado y
topes de presupuesto. No modifica el instructivo archivado porque este describe
una corrida y taxonomía históricas.

El consumo observado fue aproximadamente **8.28 millones de tokens de entrada**
y **0.724 millones de salida por cada 5 000 chunks**. Es el fundamento de la
proyección previa; el cuaderno registra después el consumo y costo reales.

## Selección económica a la fecha de corte

DeepSeek publica para V4 Flash US$0.0028/M tokens de entrada en caché,
US$0.14/M sin caché y US$0.28/M de salida
([tabla oficial](https://api-docs.deepseek.com/quick_start/pricing)). Para los
**166 940** chunks actuales, el consumo histórico escalado proyecta 276.45 M de
entrada y 24.17 M de salida:

| Escenario Flash | Costo proyectado |
|---|---:|
| toda la entrada como cache miss | US$45.47 |
| 90 % de entrada en caché | US$11.34 |

Como contraste, Groq publica US$0.075/M de entrada y US$0.30/M de salida para
`openai/gpt-oss-20b`; su Batch API aplica 50 % de descuento
([modelo](https://console.groq.com/docs/model/openai/gpt-oss-20b),
[Batch](https://console.groq.com/docs/batch)). Con el mismo consumo, serían
aproximadamente US$27.98 en servicio estándar o US$13.99 en batch. Flash resulta
más barato que Groq estándar desde aproximadamente **46 %** de cache hit y que
Groq batch desde aproximadamente **83 %**. El prefijo operacional repetido hace
plausible una tasa alta, pero la decisión se audita con el contador real.
El resultado de cada lote incluye `cache_hit_rate` y el costo que habría tenido
el mismo volumen en Groq batch; si el piloto queda sostenidamente por debajo de
83 % de caché, conviene ejecutar un panel Groq antes de comprometer el corpus,
sin mezclar sus etiquetas con la campaña validada.

Mistral publica clasificadores de 3B a US$0.10/M por dirección y de 8B a
US$0.04/M después de fine-tuning
([precios oficiales](https://mistral.ai/pricing/api/)). Es una alternativa futura
cuando exista suficiente referencia humana, no un sustituto directo de la
preanotación generativa condicionada por el prompt. Gemini Flash-Lite y OpenAI
GPT-5.6 Luna tienen precios de salida mayores para esta carga
([Google](https://ai.google.dev/gemini-api/docs/pricing),
[OpenAI](https://openai.com/api/pricing/)).

Se conserva DeepSeek V4 Flash porque combina costo esperado bajo, JSON,
concurrencia alta y evidencia previa en este contrato. Groq puede compararse en
un panel independiente, pero no reemplaza la campaña sin calibración propia.

## Protocolo activo

1. **Preflight.** `/models` valida credencial e identificadores sin enviar texto
   del corpus.
2. **Calibración pareada.** Se seleccionan 1 000 chunks balanceados por canal y
   video, máximo uno por video. Flash y Pro reciben exactamente el mismo panel.
3. **Riesgo–cobertura.** Se evalúan umbrales 0.70–0.95. Solo puede aceptarse
   automáticamente un caso `SEGURO`, no diferido y sobre el umbral. La selección
   exige límites inferiores Wilson unilaterales de 0.90 para acuerdo exacto y
   0.95 para acuerdo binario, con al menos 200 aceptaciones.
4. **Incertidumbre.** Se generan 1 000 réplicas bootstrap agrupadas por
   `video_id`; no se remuestrean chunks como si fueran independientes.
5. **Primera pasada.** Flash procesa en grupos de cinco, hasta 32 solicitudes en
   paralelo. La barra cuenta chunks, errores, velocidad y costo acumulado.
6. **Revisión dirigida.** Pro recibe todo daño, `needs_review`, confianza bajo el
   umbral calibrado y una muestra reproducible de 10 % de controles seguros.
   Incluye los chunks vecino anterior y posterior.
7. **Humano final.** La salida Pro precede a Flash solo donde existe revisión.
   La persona revisora puede aceptar, modificar, diferir o rechazar. Las
   abstenciones no se transforman en `SEGURO`.

La confianza es autodeclarada por el LLM y no una probabilidad estadística. La
calibración contra Pro es operacional; no reemplaza validación humana
independiente, precaución consistente con la evidencia sobre anotación asistida
en tareas subjetivas ([Schroeder et al., 2025](https://aclanthology.org/2025.findings-acl.1323/)).

## Robustez operativa

- Cada salida tiene una firma de modelo, prompt, taxonomía y agrupación.
- La reanudación omite `chunk_id` válidos ya guardados.
- Filas incompatibles se respaldan en `*.quarantine-<UTC>.jsonl`, se retiran del
  progreso y vuelven a quedar pendientes.
- `notes=null` se normaliza a cadena vacía y se limita a 160 caracteres.
- Un flag transversal mal ubicado se mueve a `flags`.
- Si una sola fila de un lote es inválida, solo esa fila se reenvía de forma
  individual.
- `MAX_PRIMARY_COST_USD` y `MAX_REVIEW_COST_USD` detienen nuevos grupos cuando
  se alcanza el tope; por concurrencia puede existir un sobrepaso acotado al
  superlote ya iniciado.
- En Colab, los checkpoints se publican como un archivo verificable en Drive.

## Artefactos y reporte

`02_01` escribe en `datos/etiquetado/cascada_deepseek_v4/`:

| Archivo | Contenido |
|---|---|
| `calibration_panel.jsonl` | cohorte pareada reproducible |
| `calibration_flash.jsonl` / `calibration_pro.jsonl` | propuestas del panel |
| `calibration_flash_vs_pro.json` | umbral, riesgo–cobertura y bootstrap |
| `primary_flash.jsonl` | primera pasada incremental |
| `directed_review_queue.jsonl` | cola con razón y contexto |
| `routing_summary.json` | cobertura por motivo de revisión |
| `review_pro.jsonl` | segunda opinión dirigida |
| `*.result.json` | tiempo, velocidad, errores, tokens y costo |

La última celda de `02_01` y todo `02_03` leen estos archivos y muestran las
tablas sin repetir llamadas. Los resultados cuantitativos del protocolo nuevo
permanecen **pendientes de ejecución**; no se sustituyen por las cifras
históricas. Al finalizar, deben reportarse tamaño del panel, umbral, cobertura,
acuerdos con intervalos, composición de la cola, errores de esquema, tokens,
cache hit, costo y decisiones humanas finales.
