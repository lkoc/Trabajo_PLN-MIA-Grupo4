# Revisión contextual de los cuadernos `02_etiquetado`

**Fecha:** 2026-08-08
**Alcance:** `02_00`–`02_05`, el contexto de traspaso sincronizado, la configuración Colab, el código de ejecución, el generador, los metadatos y las referencias.
**Dictamen:** la etapa 02 es coherente con el estado local sincronizado. De los dos cambios propuestos en la auditoría inicial, solo corresponde conservar el ajuste de la importación de credenciales; no corresponde parametrizar `COLAB_REQUIRE_L4`.

## Fuentes de autoridad locales contrastadas

1. `CONTEXTO_TRASPASO_OTRA_MAQUINA.md`: la campaña principal se ejecuta localmente sin GPU; Colab es opcional y usa CPU. Los cuadernos `03_02`–`03_06` sí requieren L4.
2. `config/colab_l4.json`: `02_01.requires_cuda=false` y `expected_gpu=null`; las cinco tareas neuronales declaran CUDA y L4.
3. `src/moderacion_peru/colab.py`: la validación de L4 está protegida por la conjunción `requires_cuda and require_l4`. Para `02_01`, `resolve_device` recibe `auto` y la comprobación de L4 no se ejecuta.
4. `tools/generate_workflow_notebooks.py`: el bootstrap común conserva un interruptor estricto para los consumidores que sí usan GPU.

## Reevaluación de los hallazgos iniciales

### R-02-01 — `COLAB_REQUIRE_L4=True` en `02_01`: no aplicar

La primera auditoría interpretó el valor común del bootstrap de manera aislada. Con el contexto completo, no es un defecto operativo ni reserva una GPU:

- la autoridad para decidir si hay CUDA es `config/colab_l4.json`;
- `prepare_colab_context` solicita `auto` cuando `requires_cuda=false`;
- tanto la exigencia de CUDA como la comprobación del modelo L4 dependen primero de `requires_cuda`;
- los metadatos y la instrucción visible de `02_01` ya dicen CPU y `expected_gpu=null`.

Parametrizar el texto del notebook duplicaría una decisión que ya pertenece a la configuración central y crearía dos fuentes susceptibles de divergir. Por ello se revierte ese cambio y se conserva `COLAB_REQUIRE_L4=True` como política estricta común: queda inerte en `02_01` y activa en los notebooks GPU.

### R-02-02 — importación de `google.colab.userdata`: aplicar

Sí corresponde conservar este ajuste. `IN_COLAB` se determina antes mediante `find_spec("google.colab")`; una vez confirmada esa rama, el import de `userdata` no debe quedar dentro de `except Exception`, porque eso ocultaría un fallo de entorno. El manejo de excepción queda limitado a leer el secreto, operación que puede fallar por ausencia o permisos sin impedir que el cuaderno muestre su advertencia de credencial.

## Controles aprobados con el contexto sincronizado

- Los seis cuadernos mantienen el contrato `moderacion_peru_5_salidas_v2` y la taxonomía `2.1.0`; el troceador `v2.2.0` no se confunde con una versión de etiquetas.
- Las rutas siguen la cadena sincronizada: `chunks_v2.jsonl` → cascada Flash–Pro → consolidado → eventos humanos append-only → snapshot de cinco salidas.
- `02_01` conserva los parámetros históricos 5×32, piloto Flash 300, piloto Pro 500, panel pareado 1 000, reanudación por `chunk_id`, presupuesto y cuarentena.
- `02_02` sigue siendo un fallback independiente y no se mezcla con la cascada principal.
- `02_00` y el bootstrap verifican identidad y SHA-256 del bundle antes de activarlo; este ajuste no modifica el core sincronizado porque el generador y el notebook no forman parte de `project_core.zip`.
- Las acciones con costo y la publicación en Drive permanecen desactivadas por defecto.
- Las celdas de código son sintácticamente válidas, no hay salidas ejecutadas obsoletas en etapa 02 y las referencias IEEE son consistentes con la bibliografía maestra.

## Límites del dictamen

No se inventan resultados de la calibración pendiente ni se consideran las cifras históricas como verdad humana. La aprobación es estructural y de coherencia local. Las llamadas reales a DeepSeek/Hugging Face y la publicación/restauración en Colab/Drive requieren credenciales e infraestructura externa; antes de procesar todo el corpus deben realizarse los pilotos documentados y comprobar costo, errores, caché y persistencia.
