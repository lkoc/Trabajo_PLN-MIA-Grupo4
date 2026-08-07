# Contexto de traspaso a otra instancia o máquina

Actualizado: **2026-08-07**. Este documento no contiene credenciales ni secretos.

## Punto de partida

- Repositorio: `lkoc/Trabajo_PLN-MIA-Grupo4`.
- Rama base: `main`.
- Revisión base sincronizada antes de crear este documento: `f4d7ead`.
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
`config/prompt_operacional_ollama_v2.md`. Aunque conserva “ollama” en el nombre
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
`flujo/02_etiquetado/02_01_etiquetado_local_ollama.ipynb`. El nombre se conserva
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

Las dos etiquetas y cuatro errores de la corrida parcial reciente de Ollama se
eliminaron deliberadamente. No se tocaron VTT, datos humanos ni campañas
históricas.

## Cómo continuar localmente

DeepSeek se ejecuta localmente; no requiere GPU ni Colab. Sí requiere internet y
una variable de entorno. No copie la clave a este documento ni a Git.

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY='sk-...'
jupyter lab
```

Abra `flujo/02_etiquetado/02_01_etiquetado_local_ollama.ipynb` y ejecute las
celdas en orden. Primera fase:

```python
RUN_API_PREFLIGHT=True
RUN_CALIBRATION=True
RUN_PRIMARY=False
RUN_DIRECTED_REVIEW=False
```

Después:

1. ponga `RUN_CALIBRATION=False`;
2. use `RUN_PRIMARY=True` y `PRIMARY_LIMIT=300`;
3. si el piloto es correcto, use `PRIMARY_LIMIT=None` para todos los pendientes;
4. ponga `RUN_PRIMARY=False` y `RUN_DIRECTED_REVIEW=True`;
5. use primero `REVIEW_LIMIT=500` y luego `REVIEW_LIMIT=None`.

Las salidas locales se guardan en
`datos/etiquetado/cascada_deepseek_v4/`. La última celda de `02_01` y todo
`02_03_revision_llm_dirigida.ipynb` muestran los resultados guardados sin
repetir llamadas.

`02_02_etiquetado_remoto.ipynb` quedó como fallback local independiente con
`Qwen/Qwen3-1.7B`. No debe mezclarse ni promediarse con la campaña Flash–Pro.

## Precios y control de riesgo

Tarifas DeepSeek verificadas el 2026-08-07:

- Flash: US$0.0028/M entrada cache hit, US$0.14/M cache miss y US$0.28/M salida;
- Pro: US$0.003625/M entrada cache hit, US$0.435/M cache miss y US$0.87/M salida.

DeepSeek anuncia un aumento próximo todavía no especificado. Antes de lanzar el
corpus completo debe revisarse la
[tabla oficial](https://api-docs.deepseek.com/quick_start/pricing/). Con las
tarifas actuales, la primera pasada de 166 940 chunks proyecta aproximadamente
US$45.47 sin caché o US$11.34 con 90 % de cache hit. El cuaderno compara además
el costo equivalente de Groq batch. El límite interno es una estimación, no un
tope contractual de la cuenta DeepSeek.

## Colab y bundle reproducible

Colab es opcional para `02_01`; la campaña API funciona con runtime CPU. Si se
usa Colab:

1. ejecute `02_00_preparacion_bundle_colab.ipynb`;
2. publique el bundle desde GitHub o mediante `local_upload`;
3. configure `DEEPSEEK_API_KEY` como secreto de Colab;
4. publique checkpoints coherentes con `PUBLISH_TO_DRIVE=True`.

Bundle actualmente verificado:

- `bundle_id`: `ce0fa584054b7b3a1afa7b09d4bb91507634b335ff25511b6a9f9f2a03f72b09`;
- core SHA-256: `a58c8cf341c73ceafa06089546eb13942a6b36e3ce69e79c929faa276d3ead1f`.

Los cuadernos `03_02`–`03_06` sí requieren NVIDIA L4.

## Documentos activos relevantes

- `README.md`;
- `flujo/02_etiquetado/README.md`;
- `docs/ORDEN_EJECUCION.md`;
- `docs/COLAB_L4.md`;
- `docs/METODOLOGIA_ETIQUETADO_CASCADA.md`;
- `docs/AUDITORIA_CITAS_CUADERNOS.md`;
- `docs/MATERIALIZACION_TROCEADO.md`.

La metodología de etiquetado documenta qué cifras son históricas y cuáles están
pendientes de ejecución. No deben inventarse resultados del nuevo panel.

## Verificación cerrada

La última actualización cerró con:

```text
pytest -q                    108 pruebas aprobadas
python tools/audit_project.py
                              18/18 cuadernos con referencias
                              101/101 citas válidas
                              0 incidencias
```

`01_03_limpieza_troceado_incremental.ipynb` conserva deliberadamente resultados
descriptivos ejecutados por el usuario. Los otros cuadernos activos no guardan
salidas obsoletas.

## Trabajo pendiente

1. Ejecutar la calibración Flash–Pro de 1 000 pares; todavía no existen
   resultados nuevos.
2. Revisar cache hit, errores, costo real e intervalos antes de aprobar el corpus
   completo.
3. Ejecutar Flash completo y después Pro dirigido.
4. Auditar las tablas persistidas en `02_03`.
5. Consolidar en `02_04`, completar la revisión humana y congelar el snapshot en
   `02_05`.
6. Regenerar/publicar el bundle después del snapshot humano antes de entrenar la
   etapa 03.
7. Retomar los 57 VTT pendientes con `01_01` cuando se decida continuar la
   adquisición; no bloquean la campaña actual de los chunks ya materializados.

