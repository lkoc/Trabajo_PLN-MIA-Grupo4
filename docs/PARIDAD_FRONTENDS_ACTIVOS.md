# Paridad funcional de los frontends activos

Fecha de revisión: **2026-08-08**.

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
| texto y contexto anterior/posterior | panel de tres fragmentos | implementado |
| enlace al segundo del video | enlace temporal por `video_id` | implementado |
| propuesta y categorías/flags | propuesta LLM revelable y controles derivados de la taxonomía | implementado; se oculta inicialmente para reducir anclaje |
| aceptar, modificar, diferir y excluir | eventos `accept`, `modify`, `defer` y `reject`; excluir exige confirmación | implementado |
| progreso, cohorte y siguiente pendiente | barra, filtro y navegación circular; un diferido sigue pendiente hasta una decisión resolutiva | implementado y probado en navegador |
| guía, exportación y reanudación | diálogo, JSONL y lectura de eventos previos | implementado |
| borradores locales | `localStorage` por `chunk_id` | implementado |
| atajos A/R/Ctrl+Enter/flechas | se conservan; D añade diferimiento | implementado y cubierto por prueba estructural |
| flags condicionados a daño | los flags se desactivan y limpian sin categoría de daño; el servidor también rechaza esa combinación | implementado en interfaz y contrato |

El HTML no contiene la campaña: el servidor pagina hasta 1 000 filas por
solicitud y recupera los eventos previos. Por ello una recarga no borra lo
revisado y una campaña grande no obliga a incrustar datos sensibles en el
frontend. Una decisión `defer` se conserva en el historial, pero no cuenta como
resuelta ni desaparece del conjunto pendiente.

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
