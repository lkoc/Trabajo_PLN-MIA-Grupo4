# Contexto de traspaso a otra instancia o máquina

Actualizado: **2026-08-12**. Este documento no contiene credenciales ni secretos.

## Punto de partida

- Repositorio: `lkoc/Trabajo_PLN-MIA-Grupo4`.
- Rama base: `main`.
- Revisión local al actualizar este documento: `df15b94`. Compruebe la rama
  remota antes de asumir que este commit ya fue publicado.
- La raíz del proyecto debe localizarse por `pyproject.toml`; no deben fijarse
  rutas personales como `D:\trabajo_PLN\...` dentro del código.
- Los cuadernos activos están en `flujo/`. Las carpetas históricas permanecen
  bajo `archivo/` y no definen el flujo vigente.

Antes de modificar nada en otra máquina:

```powershell
git pull --ff-only origin main
python tools/restore_synced_checkpoints.py
python tools/audit_project.py
```

## Contratos vigentes

La taxonomía sigue en **v2.1.0**. Las cinco salidas entrenadas son:

1. `SEGURO`;
2. `RACISMO_DISCRIMINACION`;
3. `ATAQUE_POR_GENERO_IDENTIDAD`;
4. `ACOSO_AMENAZA`;
5. `CONTENIDO_SEXUAL`.

`SEGURO` es excluyente. Los cuatro daños son multietiqueta. Los casos
indeterminados se difieren y no entran al entrenamiento. No debe cambiarse la
taxonomía a v2.2: **v2.2.0 corresponde al troceador**, no al contrato de
etiquetas.

La guía generativa activa es
`config/prompt_operacional_ollama_v3_2.md`. Aunque conserva “ollama” en el nombre
por compatibilidad histórica, también es la autoridad compacta usada por
DeepSeek y Hugging Face.

## Estado cuantitativo de datos

El último manifiesto de materialización es
`datos/processed/chunk_materialization_manifest.json`:

- 5 002 videos con transcripción;
- 4 992 videos con chunks;
- 10 videos sin chunks por no cumplir la materialización;
- 166 940 chunks únicos;
- 1 451.17 horas acumuladas;
- troceador v2.2.0, objetivo 30 s, 600 caracteres, mínimo 90 caracteres y 12
  palabras de solapamiento;
- SHA-256 de `datos/processed/chunks_v2.jsonl`:
  `2506123ed7a9d78fcf466e1af8875d96a70651ae8b4c22a1e8e13ccd1c542828`.

Las transcripciones sincronizables se particionan por canal en
`datos/raw/transcripts_by_channel/`. Los VTT crudos se conservan por video en
`datos/raw/vtt_by_video/` y **nunca deben borrarse**. El manifiesto
`datos/raw/vtt_by_video/missing_vtt.jsonl` contiene actualmente 57 registros que
pueden retomarse desde `01_01_scraping_incremental.ipynb`.

## Revisión del etiquetado histórico

Se revisaron completamente:

- `archivo/contrato_4_danos_seguro_derivado/03_2_etiquetado_llm_api/INSTRUCTIVO_API.md`;
- `archivo/contrato_4_danos_seguro_derivado/03_2_etiquetado_llm_api/03_2_etiquetado_llm_api.ipynb`.

El diseño histórico usó:

- producción: `deepseek-v4-flash`, pensamiento desactivado;
- revisión: `deepseek-v4-pro`;
- 32 solicitudes concurrentes;
- cinco chunks por solicitud;
- piloto Flash de 300;
- reanudación por `chunk_id`;
- revisión de daño, baja confianza y controles seguros con contexto vecino.

Resultados históricos preservados, que no deben confundirse con la taxonomía
activa ni con verdad humana:

- 69 853 chunks Flash;
- 10 000 chunks Pro;
- regla conservadora: `needs_review OR score_confianza < 0.90`;
- cobertura automática: 91.24 %;
- acuerdo exacto Flash–Pro: 93.19 %;
- acuerdo binario daño/seguro: 96.55 %.

El consumo histórico fue aproximadamente 8.28 M tokens de entrada y 0.724 M de
salida por cada 5 000 chunks.

## Flujo de etiquetado implementado

El cuaderno principal es
`flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb`. El nombre describe
por compatibilidad, pero ahora implementa una cascada completa y reanudable:

1. preflight `/models` sin enviar corpus;
2. panel Flash–Pro de 1 000 chunks, balanceado por canal y video;
3. límites Wilson unilaterales y 1 000 réplicas bootstrap por `video_id`;
4. primera pasada Flash;
5. revisión Pro de daño, abstenciones, baja confianza y 10 % de controles
   seguros;
6. consolidación por precedencia y validación humana posterior.

La API trabaja con `/chat/completions`, no con Responses API. Por ello
`deepseek-v4-pro` funciona aunque la documentación todavía indique que Pro no
está soportado por Responses API.

Detalles operativos implementados:

- solicitudes agrupadas 5×32;
- barra por chunks durante carga, búsqueda de pendientes, API y enrutamiento;
- tokens, costo estimado y `cache_hit_rate` visibles;
- límite de presupuesto por campaña;
- firma de modelo, prompt y configuración;
- cuarentena recuperable `*.quarantine-<UTC>.jsonl` para progreso inválido;
- `notes=null` se normaliza a `""` y se limita a 160 caracteres;
- flags transversales mal ubicados se mueven a `flags`;
- solo la fila inválida de un lote se reenvía individualmente;
- salidas incrementales con `fsync` y reanudación O(n).

La corrida Ollama permanece separada. No se tocaron VTT, datos humanos ni
campañas históricas.

## Estado de la campaña Flash→Pro

`02_01` ya ejecutó recuperación y calibración, y la primera pasada Flash estaba
activa al corte. La recuperación exacta reutilizó 52 244/69 853 filas Flash
(74.79 %) y 9 912/13 421 filas Pro (73.85 %). Quedaron 114 696 pendientes para
Flash; el resto de los chunks no constituye automáticamente una cola Pro.

La calibración de 1 000 pares terminó sin errores. Flash registró 97.250 s,
616.969 chunks/min, 64.22 % de caché y US$0.073349; Pro, 122.593 s, 489.424
chunks/min, 53.72 % y US$0.269223. A umbral 0.95, el acuerdo exacto fue 80.41 %
y el binario daño/seguro 99.77 %, pero el límite inferior exacto no alcanzó el
criterio predeclarado. No es una estimación de exactitud humana.

El checkpoint atómico del 2026-08-08 13:15:01 (UTC−05) contenía 14 399
etiquetas nuevas válidas, un error rechazado y 867.279 chunks/min. Restando la
calibración Flash, ese tramo costó US$0.908781. A tasa estable, la primera
pasada completa se proyectaba en 2 h 12 min y US$7.24. Consulte
[`resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md`](resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md)
antes de citar cifras: la campaña continuó después de ese checkpoint.

## Cómo continuar localmente

El cuaderno se ejecuta localmente y llama a DeepSeek en la nube; no requiere GPU
ni Colab. Sí requiere internet y una variable de entorno. No copie la clave a
este documento, a Git ni a Drive.

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY='sk-...'
jupyter lab
```

En Windows también puede persistirse como variable del usuario. El proveedor
usa la variable del proceso cuando es válida y recupera la variable persistida
si el proceso heredó un valor inválido; reinicie el kernel después de cambiarla.
Nunca muestre ni guarde el valor completo en una celda.

Abra `flujo/02_etiquetado/02_01_etiquetado_deepseek_flash_pro.ipynb` y ejecute las
celdas en orden. Para una verificación sin corpus:

```python
RUN_API_PREFLIGHT=True
RUN_CALIBRATION=False
RUN_PRIMARY=False
RUN_DIRECTED_REVIEW=False
```

La calibración y los pilotos ya existen. Para continuar la campaña, conserve
`RECOVER_HISTORICAL=True`, use `RUN_PRIMARY=True` y `PRIMARY_LIMIT=None`. El
JSONL válido se reanuda por `chunk_id`; no borre ni renombre los artefactos. Al
terminar Flash, ponga `RUN_PRIMARY=False`, active `RUN_DIRECTED_REVIEW=True` y
use `REVIEW_LIMIT=None`. Cada fase que transmite chunks debe habilitarse
deliberadamente.

Las salidas locales se guardan en
`datos/etiquetado/cascada_deepseek_v4/`. La última celda de `02_01` y todo
`02_03_revision_llm_dirigida.ipynb` muestran los resultados guardados sin
repetir llamadas.

`02_02_etiquetado_hf_qwen_colab.ipynb` quedó como alternativa Colab independiente:
`Qwen/Qwen3-1.7B` cubre la primera pasada y `Qwen/Qwen3-4B` revisa daño,
confianza baja, abstenciones y controles. Los modelos se cargan secuencialmente
en la L4. La campaña no se mezcla ni promedia automáticamente con Flash–Pro.

## Precios y control de riesgo

Tarifas DeepSeek usadas para reconstruir el costo del corte:

- Flash: US$0.0028/M entrada cache hit, US$0.14/M cache miss y US$0.28/M salida;
- Pro: US$0.003625/M entrada cache hit, US$0.435/M cache miss y US$0.87/M salida.

Antes de iniciar otra campaña debe revisarse la
[tabla oficial](https://api-docs.deepseek.com/quick_start/pricing/). Tras
recuperar el histórico, la proyección previa para 114 696 pendientes era
US$31.24 sin caché o US$10.77 usando 78.56 % de *cache hit* histórico. El tramo
activo observado redujo la proyección temprana de primera pasada a US$7.24. El
límite interno es una estimación, no un tope contractual de la cuenta DeepSeek;
el cuaderno consulta y muestra el saldo por separado.

## Colab y bundle reproducible

Colab es opcional para `02_01`; la campaña API funciona con runtime CPU. Si se
usa Colab:

1. ejecute `02_00_preparacion_bundle_colab.ipynb`;
2. publique el bundle desde GitHub o mediante `local_upload`;
3. configure `DEEPSEEK_API_KEY` como secreto de Colab;
4. active deliberadamente el interruptor de entrenamiento. Cada época terminada
   se publica automáticamente como checkpoint verificable y una nueva corrida
   reanuda el último checkpoint válido; los resultados finales también se
   publican automáticamente. `PUBLISH_TO_DRIVE=True` queda solo como reintento
   manual.

Bundle actualmente verificado:

- `bundle_id`: `57820ed6c4b2453e53cefb1fde9b8c4675b22bdf21db0159eaec08c315506691`;
- core SHA-256: `cded349ce51da1421aab4a13f40de61bc471674151bf6a0b6178198b397ad1f2`;
- manifiesto SHA-256: `6ed332fbadb5b05c81f27390d0892886e80a9747c02d39de01f85f125ee73d96`.

Los cuadernos `03_02`–`03_06b` requieren CUDA en Colab (L4 o un perfil
BF16 de 40 GB para los Qwen). `03_02` publica MiniLM y E5 inmediatamente al
terminar cada uno; `03_06b` publica piloto y corrida completa por separado. La
publicación final usa dos ranuras redundantes y nunca reemplaza la última copia
verificada antes de comprobar tamaño y SHA-256 de la nueva.

## Documentos activos relevantes

- `README.md`;
- `flujo/02_etiquetado/README.md`;
- `docs/ORDEN_EJECUCION.md`;
- `docs/COLAB_L4.md`;
- `docs/METODOLOGIA_ETIQUETADO_CASCADA.md`;
- `docs/AUDITORIA_CITAS_CUADERNOS.md`;
- `docs/MATERIALIZACION_TROCEADO.md`;
- `resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md`.

La metodología de etiquetado separa cifras históricas, mediciones del checkpoint
y proyecciones. El acuerdo Flash–Pro no debe describirse como verdad humana.

## Verificación documental del corte

La actualización documental cerró con:

```text
python tools/audit_project.py
                              18/18 cuadernos con referencias
                              101/101 citas válidas
                              118 archivos Markdown revisados
                              0 incidencias
```

`01_03_limpieza_troceado_incremental.ipynb` conserva deliberadamente resultados
descriptivos ejecutados por el usuario. Los otros cuadernos activos no guardan
salidas obsoletas.

## Trabajo pendiente

1. Dejar terminar o reanudar Flash sobre los pendientes ya materializados, sin
   borrar el progreso.
2. Sustituir las proyecciones con `primary_flash.result.json` y revisar caché,
   errores, costo y saldo.
3. Ejecutar Pro dirigido y auditar sus tablas persistidas en `02_03`.
4. Consolidar en `02_04`, completar la revisión humana y congelar el snapshot en
   `02_05`.
5. Regenerar/publicar el bundle después del snapshot humano antes de entrenar la
   etapa 03.
6. Retomar los 57 VTT pendientes con `01_01` cuando se decida continuar la
   adquisición; no bloquean la campaña actual de los chunks ya materializados.

