# Metodología de etiquetado en cascada Flash→Pro

Última actualización: **2026-08-08**.

## Decisión vigente

El flujo activo reproduce y fortalece la campaña histórica: `deepseek-v4-flash`
realiza la primera pasada económica y `deepseek-v4-pro` revisa de manera
dirigida. El fallback local `Qwen/Qwen3-1.7B` queda como diagnóstico separado;
no se mezclan sus métricas ni sus salidas con la campaña principal.

La decisión operativa presupuestada vigente desde el **2026-08-08** fija
`REVIEW_CONFIDENCE_THRESHOLD=0.85`, `SAFE_CONTROL_RATE=0.01` y
`MAX_NEEDS_REVIEW_FOR_PRO=36_000`. Pro conserva todo daño, los 36 000 casos
`needs_review` de menor confianza —con desempate SHA-256 reproducible— y los
casos resueltos como `SEGURO` cuya confianza Flash sea menor que 0.85. Se añade
un control seguro aleatorio reproducible del 1 %. El umbral 0.95 permanece como
resultado de calibración diagnóstico y no como regla de esta reanudación.

La reducción se decidió después de detener manualmente Pro y verificar el
saldo, sin eliminar ninguna de las **29 270** revisiones ya persistidas. La cola
conservadora completa dejaba 58 086 pendientes; al costo observado no cabía en
unos US$15. La regla presupuestada deja **40 695** revisiones nuevas: 12 613
daños, 27 078 abstenciones priorizadas, 145 seguros de baja confianza y 859
controles seguros. Su costo puntual proyectado es **US$13.66** y el tope es
`MAX_REVIEW_COST_USD=14.50`.

El contrato v2.1 aprende `SEGURO`, `RACISMO_DISCRIMINACION`,
`ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO`
es excluyente; los cuatro daños pueden coexistir. Una abstención se difiere y no
entra al entrenamiento. El validador interpreta nombres de etiqueta sin
distinguir mayúsculas de minúsculas y persiste siempre la forma canónica en
mayúsculas.

## Evidencia histórica recuperada

El [instructivo histórico de la API](../archivo/contrato_4_danos_seguro_derivado/03_2_etiquetado_llm_api/INSTRUCTIVO_API.md)
y su cuaderno `03_2_etiquetado_llm_api` emplearon razonamiento desactivado,
32 solicitudes concurrentes y cinco chunks por solicitud. La primera pasada
Flash produjo **69 853** propuestas y Pro revisó **10 000**. Bajo la regla
conservadora histórica —revisar `needs_review` o confianza menor que 0.90— la
cobertura automática fue **91.24 %**, el acuerdo exacto Flash–Pro **93.19 %** y
el acuerdo binario **96.55 %**. Estas cifras describen acuerdo entre modelos, no
exactitud frente a verdad humana.

La migración activa conserva `deepseek-v4-flash`→`deepseek-v4-pro`, 32
solicitudes concurrentes y cinco chunks por solicitud. Añade el preflight
`/models`, un panel pareado de 1 000, balance por canal/video, Wilson, bootstrap
agrupado, contexto vecino, reanudación, `notes` de 160 caracteres, rescate de
flags, reenvío individual de filas inválidas y topes de presupuesto. Los
límites `PRIMARY_LIMIT` y `REVIEW_LIMIT` están en `None` durante la campaña
completa; los valores 300/500 pertenecían a la fase de prueba. No se modifica
el instructivo archivado porque describe una corrida y taxonomía históricas.

El consumo observado fue aproximadamente **8.28 millones de tokens de entrada**
y **0.724 millones de salida por cada 5 000 chunks**. Es el fundamento de la
proyección previa; el cuaderno registra después el consumo y costo reales.

## Selección económica a la fecha de corte

DeepSeek publica para V4 Flash US$0.0028/M tokens de entrada en caché,
US$0.14/M sin caché y US$0.28/M de salida
([tabla oficial](https://api-docs.deepseek.com/quick_start/pricing)). Para los
**166 940** chunks actuales, el consumo histórico escalado proyectaba 276.45 M
de entrada y 24.17 M de salida si se reetiquetaba todo:

| Escenario Flash | Costo proyectado |
|---|---:|
| toda la entrada como cache miss | US$45.47 |
| 90 % de entrada en caché | US$11.34 |

La recuperación exacta redujo la primera pasada pagada a **114 696 chunks**.
Escalando el consumo histórico a ese remanente, la referencia previa es
US$31.24 sin caché o US$10.77 con el 78.56 % de *cache hit* histórico. El
checkpoint activo descrito más abajo ofrece ya una proyección observada menor;
ninguna de estas proyecciones sustituye el resultado final del proveedor.

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

## Resultados cuantitativos disponibles

El corte documentado del **2026-08-08 13:15:01 (UTC−05)** usa un checkpoint
periódico completo de Flash; la campaña continuaba ejecutándose. La
recuperación por coincidencia unívoca `(video_id, texto_normalizado)` reutilizó
52 244 de 69 853 filas Flash históricas (**74.79 %**) y 9 912 de 13 421 filas
Pro (**73.85 %**), sin trasladar etiquetas por posición ni por similitud.

La calibración pareada completó 1 000 chunks por modelo sin errores de esquema:

| Medición | Flash | Pro |
|---|---:|---:|
| tiempo | 97.250 s | 122.593 s |
| velocidad | 616.969 chunks/min | 489.424 chunks/min |
| caché de entrada | 64.22 % | 53.72 % |
| costo estimado | US$0.073349 | US$0.269223 |
| ahorro frente al mismo tráfico sin caché | 45.53 % | 37.65 % |

Con umbral 0.95, Flash y Pro coincidieron exactamente en **80.41 %** de las 434
autoaceptaciones evaluables y en la decisión binaria daño/seguro en **99.77 %**.
Los límites inferiores Wilson fueron 77.10 % y 98.97 %, respectivamente. El
resultado es `inconclusive_conservative_threshold`: superó el objetivo binario,
pero no el criterio predeclarado de acuerdo exacto. Es acuerdo entre modelos,
no exactitud humana; el 43.40 % de cobertura tampoco es cobertura validada por
personas.

En el checkpoint, Flash había procesado 14 399 de los 114 696 pendientes
válidos a **867.279 chunks/min**; una respuesta con `chunk_id` u orden alterado
se rechazó y quedó como error pendiente. Restando su calibración, el tramo costó
**US$0.908781** (**US$0.06311 por 1 000 chunks**). La tasa de caché Flash
acumulada fue 75.17 %. Si esas condiciones se mantuvieran, la primera pasada
completa tardaría unas **2 h 12 min** y costaría aproximadamente **US$7.24**;
son proyecciones tempranas. El corte completo y sus fuentes están en
[`resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md`](../resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md).

## Protocolo activo

1. **Preflight.** `/models` valida credencial e identificadores sin enviar texto
   del corpus. Flash y Pro deben declarar `thinking=disabled`, aceptar
   `response_format=json_object` y devolver la raíz `annotations`; también se
   consulta el saldo sin transmitir chunks.
2. **Calibración pareada.** Se seleccionan 1 000 chunks balanceados por canal y
   video, máximo uno por video. Flash y Pro reciben exactamente el mismo panel.
3. **Riesgo–cobertura.** Se evalúan umbrales 0.70–0.95. Solo puede aceptarse
   automáticamente un caso `SEGURO`, no diferido y sobre el umbral. La selección
   exige límites inferiores Wilson unilaterales de 0.90 para acuerdo exacto y
   0.95 para acuerdo binario, con al menos 200 aceptaciones.
4. **Incertidumbre.** Se generan 1 000 réplicas bootstrap agrupadas por
   `video_id`; no se remuestrean chunks como si fueran independientes.
5. **Primera pasada.** Flash procesa en grupos de cinco, hasta 32 solicitudes en
   paralelo. La barra cuenta chunks, errores, velocidad, caché, costo acumulado
   y saldo periódico. Las respuestas se validan contra el esquema y el orden de
   `chunk_id` antes de persistirse.
6. **Revisión dirigida presupuestada.** Pro recibe todo daño, las 36 000
   abstenciones Flash de menor `score_confianza` y los casos resueltos como
   `SEGURO` con confianza menor que 0.85. Los empates se ordenan mediante SHA-256
   con semilla 42; además, se toma un control seguro aleatorio reproducible del
   1 %.
   Incluye los chunks vecino anterior y posterior. El 0.95 de la tabla de
   riesgo–cobertura se conserva como resultado diagnóstico.
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
- Cada grupo terminado de cinco respuestas se fuerza a disco con `fsync`; el
  checkpoint se renueva cada diez ventanas, al cerrar una fase y ante
  `Ctrl+C`. La recuperación vuelve a contar el JSONL y nunca confía solo en una
  barra de progreso.
- Filas incompatibles se respaldan en `*.quarantine-<UTC>.jsonl`, se retiran del
  progreso y vuelven a quedar pendientes.
- `notes=null` se normaliza a cadena vacía y se limita a 160 caracteres.
- Un flag transversal mal ubicado se mueve a `flags`.
- Si una sola fila de un lote es inválida, solo esa fila se reenvía de forma
  individual.
- `MAX_PRIMARY_COST_USD` y `MAX_REVIEW_COST_USD` detienen nuevos grupos cuando
  se alcanza el tope; por concurrencia puede existir un sobrepaso acotado al
  superlote ya iniciado. Para la reanudación Pro presupuestada, el segundo vale
  US$14.50.
- Antes de enviar el primer chunk de la reanudación, el cuaderno reconstruye la
  cola, descuenta los IDs ya revisados, muestra el costo proyectado y exige al
  menos US$15.00 de saldo. Con un saldo menor se detiene sin enviar corpus.
- El proveedor informa tokens de entrada con caché, sin caché y de salida. El
  cuaderno muestra saldo aproximadamente cada 60 segundos, advierte bajo
  US$2.00 y señala una tasa de caché inferior a 50 % después de 50 solicitudes.
- En Windows se prefiere la variable de proceso cuando es válida; si una sesión
  heredó una clave inválida, se recupera la variable persistida del usuario. La
  clave nunca se escribe en JSONL, checkpoints, Drive, Git ni salidas del
  cuaderno.
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
tablas sin repetir llamadas. Ya existen resultados parciales medidos de
recuperación, calibración y primera pasada; el corte documentado se mantiene en
[`resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md`](../resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md).
Al finalizar deben reemplazarse las proyecciones con el tiempo, velocidad,
tokens, caché, costo y errores de los `*.result.json`, además de la composición
de la cola Pro y las decisiones humanas finales.
