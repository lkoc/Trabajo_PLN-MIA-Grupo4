# Paridad funcional de los frontends activos

Fecha de revisión: **2026-08-09**.

## Criterio

Los HTML activos pueden cambiar de diseño, pero no deben perder funciones del
flujo histórico. La referencia de etiquetado es
`archivo/estructura_anterior/Cuadernos/frontend/revision_humana_sospechosos_139.html`;
la de producción es
`archivo/estructura_anterior/Cuadernos/frontend/produccion_moderador.html`, junto
con su servidor histórico. Las decisiones peligrosas de la versión anterior no
se copian literalmente: el contrato v2.1, la trazabilidad append-only y la
revisión obligatoria prevalecen.

## Etiquetado humano

| Función histórica mínima | Implementación activa | Estado |
|---|---|---|
| texto y contexto anterior/posterior | chunk principal fijo y contexto vecino en diálogos | implementado; evita desplazamiento de página |
| enlace al segundo del video | enlace temporal por `video_id` | implementado |
| propuesta y categorías/flags | propuesta LLM visible y controles derivados de la taxonomía | implementado; la visibilidad inicial es una decisión operativa con riesgo de anclaje documentado |
| aceptar, modificar, diferir y excluir | botones explícitos “Aceptar sugerencia”, “Guardar mi decisión”, “Revisar después” y “Excluir del dataset”; ayuda contextual; excluir exige confirmación | implementado |
| aceptar, clasificar o excluir por video/canal | diálogo con vista previa; permite conservar cada propuesta, asignar la misma combinación humana de `SEGURO`/daños/flags o excluir; exige confirmación | implementado; la clasificación común puede cubrir también los ya resueltos y cada evento registra alcance/lote |
| progreso, colas, filtros y navegación | botones directos Urgentes, Prioritarios Pro, Todos y Excluidos; filtros combinables de cohorte, categorías, flags, estado y completitud; navegación circular | implementado; incluye Sin etiqueta/Con etiqueta; los bloques se cruzan con AND y categorías/flags admiten OR o AND |
| guía, exportación y reanudación | diálogo, JSONL y lectura de eventos previos | implementado |
| borradores locales | `localStorage` por `chunk_id` | implementado |
| atajos A/R/Ctrl+Enter/flechas | se conservan; D añade diferimiento | implementado y cubierto por prueba estructural |
| flags condicionados a daño | los flags se desactivan y limpian sin categoría de daño; el servidor también rechaza esa combinación | implementado en interfaz y contrato |
| dashboard de seguimiento | enlace en cabecera y segunda página con estado efectivo, composición, categorías, colas, canales, actividad y auditoría inferencial | implementado; refresco automático cada 15 segundos y botón manual |

El HTML no contiene la campaña: el servidor pagina hasta 1 000 filas por
solicitud y recupera los eventos previos. Por ello una recarga no borra lo
revisado y una campaña grande no obliga a incrustar datos sensibles en el
frontend. Una decisión `defer` se conserva en el historial, pero no cuenta como
resuelta ni desaparece del conjunto pendiente.

El dashboard no mantiene una copia del dataset. `/api/dashboard` vuelve a
resumir la campaña cargada y el diccionario de últimas decisiones bajo el mismo
bloqueo de persistencia que usa la validación; por ello una acción recién
guardada se refleja en la próxima consulta. Separa explícitamente dos planos:
las estadísticas descriptivas vivas y las métricas inferenciales de la muestra
estratificada congelada. Estas últimas se leen de
`docs/artefactos/auditoria_16k_flash_pro_sol_eh_metrics.json` y conservan sus
intervalos, advertencia de referencia interna y alcance del panel pareado.

Las colas no son sinónimas. **Urgentes** es el subconjunto corto de propuestas
máximas incompatibles que la consolidación no resolvió automáticamente.
**Prioritarios Pro** reúne las salidas de DeepSeek Pro que quedaron
efectivamente pendientes o conservaron al menos una categoría de daño después
de aplicar la última decisión superior. En el corte vigente son 12.671 casos,
todos con daño efectivo, y cero pendientes finales. Los 35.385 registros Pro
con `needs_review=true` permanecen visibles como señal histórica intermedia,
pero todos están superados por una decisión CODEX–Sol-EH o humana y no inflan
la cola pendiente. La confianza autodeclarada no se usa como umbral adicional,
pues la calibración disponible fue inconclusa.
**Todos** conserva acceso a la campaña completa. **Excluidos** reúne los chunks
cuya decisión efectiva es `reject` —o cuyo estado base es `excluded` si no hay
revisión posterior—. No constituye una sexta categoría semántica: es un estado
de elegibilidad para entrenamiento y puede revertirse con una nueva decisión.

La acción masiva deriva el video o canal desde el chunk activo. Para canales sin
`channel_id`, usa coincidencia normalizada del título; el diálogo muestra el
alcance antes de habilitar la confirmación. Por defecto solo afecta pendientes y
diferidos. Incluir resueltos crea nuevos eventos de precedencia, nunca reescribe
ni elimina el historial anterior. Un reintento del mismo lote es idempotente.
La opción **Clasificar todo igual** activa inicialmente la inclusión de resueltos
para abarcar realmente todo el video o canal; la persona revisora puede
desmarcarla. La clasificación común admite `SEGURO` de forma excluyente o una
combinación de categorías de daño y flags válidos. El servidor vuelve a validar
estas invariantes antes de guardar.

## Producción supervisada

| Función histórica mínima | Implementación activa | Estado |
|---|---|---|
| entrada texto/YouTube y detección automática | entrada única, tipo automático o forzado | implementado |
| subtítulos, caché, troceado y reproductor | solo subtítulos; caché local y enlace temporal | implementado |
| mejor clásico, Transformer y Qwen | `03_07` publica el mejor de cada slot usando solo validation | implementado en código; modelos v2.1 pendientes de ejecutar |
| consulta individual | selector por los tres slots disponibles | implementado |
| comparación de respuestas | modo `compare` con tarjetas separadas | implementado |
| consenso mayoritario | voto 2-de-3; desacuerdo o ausencia de mayoría obliga revisión | implementado y probado |
| scores, umbrales, confianza y motivos | cinco scores por tarjeta; umbrales por miembro y promedio en consenso | implementado |
| aceptar, rechazar o modificar | revisión ligada al `event_id`; rechazar corrige a `SEGURO` | implementado |
| persistencia y exportación | inferencias y revisiones JSONL append-only | implementado |
| estadísticas por modelo/categoría | predicciones, revisión y decisión humana por slot | implementado |
| preparación para reentrenamiento | deduplicación, exclusión de conflictos, umbrales orientativos y exportación aparte | implementado y probado |
| acceso protegido al publicar fuera de loopback | Basic Auth mediante `MODERATOR_ACCESS_USER/PASSWORD` | implementado; sin contraseña solo se admite loopback |

La publicación produce un registro principal verificable y registros miembro
`.<slot>.json`, todos con SHA-256. `consensus` solo se ofrece cuando existen los
tres slots; `compare` requiere al menos dos. Si falta el registro v2.1 no se
reutiliza silenciosamente un modelo histórico. Si una entrada excede el máximo
de chunks configurado, la API la rechaza explícitamente en lugar de truncarla y
producir una decisión parcial silenciosa.

## Capturas

Las seis imágenes existentes en `Documento_final_paper/figuras/captura_*.png`
son **históricas**, no capturas de una ejecución v2.1. La presentación las
rotula como tales. No se generará una captura sintética que parezca un resultado
real.

- Captura actual de etiquetado: pendiente de materializar la campaña consolidada
  con `02_04` y abrir el servidor activo.
- Captura actual de producción: pendiente de entrenar las tres familias, publicar
  sus registros con `03_07` y ejecutar `04_01`.

El código de ambas interfaces está disponible ahora; lo pendiente es la
evidencia visual con artefactos actuales. Como comprobación técnica se abrieron
ambas interfaces en Chrome sin interfaz gráfica usando fixtures aislados: se
verificaron diferimiento y reanudación en etiquetado, los cinco modos de
producción, consenso, inferencia y estadísticas, sin errores de consola. Esas
capturas sintéticas no se versionan ni se presentan como resultados del corpus.
El diseño no necesita imitar píxel por píxel al histórico, pero la matriz
anterior constituye el mínimo funcional.

## Verificación automatizada

`tests/test_frontend_parity.py` comprueba la publicación de los tres miembros,
el consenso 2-de-3, la validación de flags y la materialización conflict-safe
para reentrenamiento.
`tests/test_structure.py::test_required_frontends_are_small_templates`
comprueba los controles esenciales de ambos HTML.
