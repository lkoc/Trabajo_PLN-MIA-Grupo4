# Piloto prerregistrado de adjudicación con DeepSeek V4-Pro Reasoning

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


Fecha de prerregistro: 2026-07-26  
Estado inicial: **PRERREGISTRADO; SIN LLAMADAS DEL PILOTO AL MOMENTO DE CREAR ESTA SECCIÓN**

## 1. Pregunta y alcance

Se evaluará si `deepseek-v4-pro`, usado con `thinking=enabled` y `reasoning_effort=max`, puede reducir de forma trazable la revisión humana pendiente. El piloto no sustituye una verdad de terreno humana, no sobrescribe las 23 decisiones humanas existentes y no modifica el conjunto de entrenamiento.

La población elegible tiene dos cohortes:

- 139 casos difíciles de la campaña humana original; 23 ya cuentan con decisión humana y 116 siguen pendientes.
- 1.779 casos de la ampliación que conservaron `needs_review=True` después de Pro sin razonamiento.

## 2. Muestra fijada antes de consultar el modelo

El piloto contiene exactamente 200 chunks:

1. Los 139 casos de la campaña humana original, para cubrir íntegramente esa cohorte y permitir una comparación ciega con las 23 decisiones humanas disponibles.
2. Una muestra determinista de 61 entre las 1.779 dudas nuevas.

La muestra nueva usa semilla `26072027`. Los estratos son las combinaciones exactas ordenadas de `pro_coarse_labels`. Dentro de cada estrato los chunks se ordenan por SHA-256 de `26072027|chunk_id`; se toma un caso por estrato en rondas sucesivas hasta completar 61. Este diseño aumenta la cobertura de combinaciones minoritarias y no pretende estimar prevalencia poblacional.

## 3. Intervención ciega y reproducible

Cada chunk recibe dos adjudicaciones separadas del mismo modelo:

- **Pasada A, normativa:** aplicación directa y conservadora de las definiciones gruesas.
- **Pasada B, contraevidencia:** evaluación explícita de la mejor interpretación segura/contextual antes de decidir.

Configuración común:

- Modelo: `deepseek-v4-pro`.
- Fuentes de autoridad literales: `03_2_etiquetado_llm_api/prompt_operacional_compacto.md` (versión 1.1) y `datos/processed/taxonomia_moderacion.csv`, las mismas usadas por el flujo 03_2.
- El modelo devuelve las etiquetas finas mediante el contrato original (`labels`, `flags`, `needs_review`, `notes`, `score_confianza`, `justificacion`); el programa calcula determinísticamente la proyección gruesa. Solo esa proyección puede participar en el consenso.
- `thinking={"type":"enabled"}`.
- `reasoning_effort="max"`.
- Máximo: 2.000 tokens de salida por chunk y pasada, incluyendo razonamiento.
- Un chunk por solicitud; concurrencia máxima 16.
- Salida JSON validada contra un contrato estricto.
- Se incluyen el chunk anterior y posterior del mismo video cuando existen.
- No se muestran etiquetas Flash, etiquetas Pro, estados humanos ni la otra pasada.
- No se conserva el contenido de la cadena de razonamiento; solo sus contadores de tokens.

## 4. Regla de aceptación fijada

Un caso queda **resuelto provisionalmente por consenso reasoning** solo si:

1. Ambas pasadas devuelven exactamente el mismo conjunto de categorías gruesas.
2. Ambas tienen `needs_review=False`.
3. Ninguna usa el flag `contexto_necesario`.
4. La confianza declarada en ambas es al menos 0,80.

Todo lo demás conserva `requires_human=True`. Una aceptación del piloto es una pseudoetiqueta adjudicada por LLM, no una etiqueta humana.

## 5. Evaluación prerregistrada

Se informarán, sin cambiar umbrales después de observar resultados:

- éxito técnico y costo real por pasada;
- proporción aceptada y pendiente total y por cohorte;
- concordancia exacta entre pasadas, Jaccard multietiqueta y distribución por categoría;
- contra las 23 etiquetas humanas: exactitud de conjunto, Hamming loss, precisión/recall/F1 micro y macro por categoría, y un intervalo Wilson de 95% para la exactitud de conjunto;
- desempeño humano restringido a los casos aceptados por la regla, junto con su cobertura.

El piloto se considerará operacionalmente prometedor si alcanza al menos 98% de éxito técnico, resuelve al menos 50% de la muestra y la exactitud exacta puntual contra humanos es al menos 85% entre los casos aceptados. Por el tamaño humano pequeño, estos criterios no autorizan afirmar equivalencia con adjudicación humana.

## 6. Presupuesto y detención

El techo nominal de salida es 800.000 tokens (200 casos × 2 pasadas × 2.000). Con los precios vigentes de V4-Pro y el patrón de caché observado, el costo esperado es inferior a USD 1; se fija un margen operativo máximo de USD 1,50 incluyendo reintentos. Un error de esquema se reintenta como máximo dos veces; los fallos persistentes se registran y permanecen para humano.

## 7. Artefactos previstos

- Selección: `datos/etiquetado/reasoning/piloto_reasoning_200_seleccion.jsonl`.
- Pasada A: `datos/etiquetado/reasoning/piloto_reasoning_200_pasada_a.jsonl`.
- Pasada B: `datos/etiquetado/reasoning/piloto_reasoning_200_pasada_b.jsonl`.
- Consenso: `datos/etiquetado/reasoning/piloto_reasoning_200_consenso.jsonl`.
- Manifiesto: `datos/etiquetado/reasoning/piloto_reasoning_200.manifest.json`.
- Métricas: `resultados/metricas/piloto_reasoning_200_metricas.json`.
- Figura: `resultados/figuras/piloto_reasoning_200_resultados.png`.

## 8. Referencias

DeepSeek. (2026). *Models & pricing*. https://api-docs.deepseek.com/quick_start/pricing

DeepSeek. (2026). *Thinking mode*. https://api-docs.deepseek.com/guides/thinking_mode

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953

## 8.1 Enmienda técnica registrada antes de observar resultados semánticos

La ejecución inicial reveló que el límite prerregistrado de 2.000 tokens era técnicamente insuficiente para `reasoning_effort=max`. Se detuvo después de 48 solicitudes de la pasada A: 44 terminaron por truncamiento, cuatro produjeron JSON válido y el costo acumulado fue USD 0,3100. La decisión de detener y ajustar se tomó usando únicamente `finish_reason`, validez de esquema, tokens y costo; no se inspeccionaron ni usaron las categorías devueltas.

La corrida se preserva, pero se excluye del análisis sustantivo:

- Archivo: `datos/etiquetado/reasoning/piloto_reasoning_200_pasada_a_cap2000_fallida.jsonl`.
- SHA-256: `c6a4b9375f0e2d8d1532f3c05743c29a24919583af0a74b6998023c29771cb09`.
- Tokens de razonamiento facturados: 278.261.

La terminación inicial detuvo el proceso controlador, pero dejó vivo el proceso Python con solicitudes ya iniciadas; este continuó hasta 128 filas (9 válidas, 119 fallidas; USD 0,7436) antes de ser detenido explícitamente por PID. Esa continuación obsoleta también queda excluida: `datos/etiquetado/reasoning/piloto_reasoning_200_pasada_a_cap2000_continuacion_128_fallida.jsonl`, SHA-256 `9397679b64a8b9552ae5fbfa975059bac68ca5365a2fade18d902a5a7fde32d2`.

Antes de reiniciar se fija la siguiente enmienda:

1. Calibración puramente técnica con los primeros ocho IDs ya seleccionados, una pasada A, máximo 8.000 tokens y sin reintento.
2. Si al menos siete de ocho respuestas cierran con JSON semánticamente válido, las dos pasadas completas se ejecutarán desde cero con máximo 8.000 tokens.
3. Se permitirán como máximo dos intentos por solicitud y el nuevo techo global será USD 5,00.
4. Si la calibración no alcanza 7/8, el piloto se detendrá sin análisis semántico.
5. La muestra, las dos variantes de prompt, la regla de aceptación y las métricas prerregistradas no cambian.

Esta enmienda aumenta únicamente el presupuesto de cómputo necesario para obtener una respuesta completa; no se eligió a partir de concordancia con humanos ni de resultados de clasificación.

### Segunda corrección técnica del contrato

La calibración de ocho casos a 8.000 tokens obtuvo 2/8 respuestas válidas y costó USD 0,0322. Ninguna de las seis fallas restantes fue truncamiento: cinco devolvieron nombres/campos del contrato operativo original (`labels`, `notes`, `justification` o `confidence`) que el adaptador inicial rechazaba, y una produjo JSON mal formado. Esto confirmó que 8.000 tokens resuelven el problema de longitud, pero también reveló una incompatibilidad introducida por el adaptador del piloto.

- Archivo preservado: `datos/etiquetado/reasoning/piloto_reasoning_calibracion8_cap8000.jsonl`.
- SHA-256: `4073d7071a599362b322c5cb0445c2d84e6ef35824b29ef351dce908623ccc2b`.

Antes de nuevas llamadas se corrige el programa para aceptar **exactamente** el contrato del prompt operativo 03_2: `chunk_id`, `labels`, `flags`, `needs_review`, `notes`, `score_confianza` y `justificacion`. La proyección fina→gruesa deja de pedirse al modelo y se calcula en código mediante un mapa versionado. Se repetirá la misma calibración de ocho IDs, sin reintento, y se mantiene la exigencia 7/8 para iniciar el piloto completo. Esta corrección no cambia textos, muestra, etiquetas permitidas, reglas sustantivas ni criterios de aceptación.

La segunda calibración produjo 5/8 respuestas aceptadas por el parser y costó USD 0,0457. Dos de las tres restantes contenían exactamente los siete campos del contrato operativo, pero en la raíz del JSON en lugar del wrapper `annotation`; una respuesta se truncó a 8.000 tokens. El artefacto se preserva en `datos/etiquetado/reasoning/piloto_reasoning_calibracion8_cap8000_contrato03_2.jsonl`, SHA-256 `10092c3e4bb7033a94e043c9ba98cd0af6994252c63ce9ea30ad74728b61b7af`.

Se registra una última corrección de transporte: el parser envolverá una respuesta raíz solo cuando sus claves sean **exactamente** los siete campos permitidos; después aplicará el mismo JSON Schema y todas las reglas semánticas. No se aceptan alias ni campos extra. Se repite una última calibración 8/8 sin reintento y se conserva el umbral 7/8. Esta normalización no modifica ninguna etiqueta ni usa referencias humanas.

La calibración final obtuvo 6/8 respuestas estrictamente válidas. Una séptima contenía el contrato, etiquetas y decisión completos, pero `notes` superaba el máximo de 160 caracteres; el flujo 03_2 preexistente normaliza precisamente `notes` mediante recorte antes de validarlo. El octavo caso fue el único truncamiento real a 8.000 tokens. Al incorporar ese mismo normalizador de longitud —sin alterar etiquetas, flags, decisión ni confianza— el resultado técnico es 7/8 y satisface el umbral fijado. Artefacto: `datos/etiquetado/reasoning/piloto_reasoning_calibracion8_final.jsonl`; SHA-256 `d3e49357c07183a9ef93779c514b36d08eb97ba73dbe2ebab5e66923e8e1a602`.

Por tanto, se autoriza la ejecución completa desde cero con 8.000 tokens, dos intentos máximos y presupuesto global USD 5,00. Los tres artefactos de calibración quedan excluidos de las métricas sustantivas.

---

## 9. Ejecución y resultados

Ejecución cerrada: 2026-07-27T00:58:13-05:00  
Modelo: `deepseek-v4-pro` con `thinking=enabled` y `reasoning_effort=max`  
Costo de la corrida analítica: **USD 1.5434**  
Costo total del proceso, incluidas calibraciones y corrida técnica excluida: **USD 2.3996**

### 9.1 Integridad técnica y consumo

| Pasada | Éxitos | Errores | Entrada | Salida | Reasoning | Costo USD |
|---|---:|---:|---:|---:|---:|---:|
| A | 199 | 1 | 852,755 | 795,971 | 0 | 0.7363 |
| B | 200 | 0 | 877,422 | 868,737 | 0 | 0.8071 |

La API no desglosó `reasoning_tokens` en estas respuestas; el cero de la tabla significa «no reportado por separado», no ausencia de razonamiento. Los tokens de razonamiento están incluidos en `completion_tokens` y en el costo.

### 9.2 Consenso y reducción potencial de revisión humana

Las dos pasadas coincidieron exactamente en 118/200 casos (59.0%); Jaccard medio multietiqueta 0.678. La regla prerregistrada aceptó **69/200 (34.5%)** y conservó 131 para humano.

| Cohorte | N | Aceptados | Conservan humano | Tasa de aceptación |
|---|---:|---:|---:|---:|
| Original 139 | 139 | 53 | 86 | 38.1% |
| Ampliación 61 | 61 | 16 | 45 | 26.2% |

![Resultados del piloto reasoning](figuras/piloto_reasoning_200_resultados.png)

### 9.3 Comparación ciega con las 23 decisiones humanas congeladas

| Evaluación | N | Exactitud de conjunto | Hamming loss | F1 micro | F1 macro |
|---|---:|---:|---:|---:|---:|
| Pasada A | 23 | 17.4% | 0.297 | 0.281 | 0.236 |
| Pasada B | 23 | 30.4% | 0.217 | 0.516 | 0.437 |
| Consenso aceptado | 7 | 14.3% | 0.310 | 0.133 | 0.042 |

El consenso cubrió 30.4% de la referencia humana. Su IC Wilson 95% para exactitud exacta es [2.6%, 51.3%]; el intervalo amplio refleja que solo hay 23 casos humanos.

Indicador de seguridad: entre 6 casos aceptados que humano marcó como daño, el consenso reasoning clasificó 6 como `SEGURO` (100.0%). Este patrón de falsos seguros impide usar el consenso para absolver automáticamente casos.

### 9.4 Criterios prerregistrados y conclusión

- Éxito técnico ≥98%: **sí**.
- Aceptación ≥50%: **no**.
- Exactitud puntual ≥85% contra humano entre aceptados: **no**.

Conclusión prerregistrada: **el piloto no supera todos los criterios operativos**. Aunque los cumpla, las salidas siguen siendo pseudoetiquetas LLM y no reemplazan una validación humana académica. No se integró ningún resultado al entrenamiento y no se autoriza escalar este procedimiento a los 1.895 pendientes.

La referencia humana de 23 casos corresponde a los casos completados disponibles, no a una muestra aleatoria nueva. Por ello la comparación es un control de seguridad del piloto y no una estimación poblacional definitiva; el mal desempeño observado, sin embargo, es suficiente para rechazar la automatización propuesta.

### 9.5 Trazabilidad de salida

- Selección SHA-256: `c6d162f13906ecb823347c4cd1a4be4b5b6034724c31c832b24015f77b9d2f31`.
- Prompt operacional SHA-256: `52d4fec14ad433d35ec20de5f51a6954aad69dcedd1422059419dcecc2f9e778`.
- Taxonomía CSV SHA-256: `763c62f3d51706f0260636cdf26fe29e7fe23a9596df17c0d9fa2649e88bc8b4`.
- Referencia humana congelada SHA-256: `cef3b83a0b7e322b486e51c41787a9cea236b6879355c72c3409c73f54807ae2`.
- Pasada A SHA-256: `c76e34579d5bb32f656e5f7882d7ea5499399079b88947afd3512691de899409`.
- Pasada B SHA-256: `abd69080901df6c161dd7c27cf1927bf8ab0adc12cff67db405dae95ceb5bdd5`.
- Consenso SHA-256: `8a194670eebd43fd0ae5985e902a4ce12aff69564aca5475399aef0216ab9566`.
- Métricas SHA-256: `77268714d538ec360ad3dd47137e84aa560551a38dcb14b78cd619f9c89d74a6`.
