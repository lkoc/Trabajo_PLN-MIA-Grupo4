# Metodología de scraping y adquisición de subtítulos

## 1. Propósito y alcance

Este documento describe la metodología implementada por
[`01_01_scraping_incremental.ipynb`](../flujo/01_datos/01_01_scraping_incremental.ipynb)
y por [`acquisition.py`](../src/moderacion_peru/acquisition.py). En este proyecto,
*scraping* significa:

1. descubrir metadatos públicos de videos y canales;
2. localizar pistas públicas de subtítulos en español;
3. descargar la representación textual de esas pistas; y
4. conservar procedencia, fallos, cachés y manifiestos reproducibles.

El flujo no descarga audio ni video y no ejecuta reconocimiento automático del
habla. `yt-dlp` se utiliza para descubrir fuentes y localizar pistas de
subtítulos [1]. Los subtítulos automáticos son un insumo imperfecto: pueden
contener errores y sesgos asociados, entre otros factores, con dialecto y
género [2]. Por ello, una transcripción obtenida no se interpreta como verdad
textual ni como etiqueta de moderación.

La adquisición debe respetar los términos de la plataforma [3] y las
consideraciones contextuales de la investigación en Internet [4]. El proyecto
no usa este muestreo para estimar prevalencias de daño en YouTube o en el Perú.

## 2. Principios operativos

El diseño aplica los siguientes principios:

- **Incrementalidad:** un `video_id` canónico no vuelve a entrar a la cola de red.
- **Reanudación:** una transcripción presente en caché se reutiliza antes de
  intentar una solicitud nueva.
- **Cohorte explícita:** el modo dirigido trabaja sobre todos los candidatos
  inéditos de su selección vigente, no sobre el backlog general sin procedencia.
- **Separación entre selección y verdad:** `target_category` explica por qué se
  seleccionó un candidato; no es una etiqueta observada ni se copia al dataset
  entrenable.
- **Test congelado:** `test` no interviene en el cálculo de déficits, la elección
  de canales ni el ajuste de cuotas.
- **Tolerancia a fallos:** un video privado, futuro, sin subtítulos o retirado no
  detiene el lote.
- **Preservación:** el reinicio desde cero archiva los artefactos activos en vez
  de eliminarlos de forma irreversible.

La ponderación por déficit se inspira en la necesidad de balancear la selección
en aprendizaje activo desbalanceado [5] y en clasificación multietiqueta con
distribuciones de cola larga [6]. Las fórmulas, umbrales, cuotas y reglas de
reanudación descritas a continuación son decisiones locales del proyecto; las
fuentes citadas no prescriben esta implementación exacta.

## 3. Modos de descubrimiento

La variable `DISCOVERY_MODE` admite tres valores:

| Modo | Fuentes | Cola de adquisición |
|---|---|---|
| `seed` | Catálogo general de canales y consultas | Archivo acumulado de candidatos generales |
| `directed` | Canales estimados como productivos, consultas temáticas y canales expandidos | Solo `directed_candidates_latest.jsonl` |
| `both` | Unión del descubrimiento general y dirigido | Candidatos generales de la corrida actual más la cohorte dirigida vigente |

`DISCOVER_NEW=False` evita las consultas de descubrimiento. En modo `directed`
se reutiliza la última cohorte dirigida materializada. `FETCH_NEW=False` permite
inspeccionar el plan, los candidatos, el canónico y la caché sin solicitar
subtítulos nuevos.

## 4. Reutilización y filtro previo

Antes de cualquier descarga, el cuaderno:

1. define `datos/raw/transcripts_raw.jsonl` como vista canónica;
2. incorpora idempotentemente los snapshots históricos, restaura las
   particiones sincronizadas por canal y anexa los JSON válidos de caché que
   todavía falten, salvo que esté activo el reinicio desde cero;
3. vuelve a materializar las particiones por canal desde el canónico ya
   consolidado;
4. inventaría todo `video_id` con texto disponible en el canónico, la caché,
   los snapshots, los datasets model-ready y los chunks históricos;
5. carga y deduplica candidatos por `video_id`; y
6. elimina de la cohorte activa la unión global de videos conocidos, no solo
   los presentes en el canónico.

Los datasets y chunks históricos no se convierten artificialmente en una
transcripción raw: pueden haber perdido segmentos o tiempos. Sus `video_id` sí
se usan para evitar volver a solicitar a YouTube un texto que ya está disponible
como derivado. El resumen separa `transcripciones_canónicas_completas`,
`transcripciones_completas_disponibles`, `videos_solo_en_derivados` y
`videos_conocidos_globales`. Así, un cero en `ya_canónicos_omitidos` describe
solo el solapamiento de la cohorte actual y nunca el tamaño total del corpus.

La barra `Procesando pendientes` representa la cohorte posterior al inventario
global, no la suma histórica de candidatos. `ingest_incremental` repite
internamente la verificación del canónico y de la caché como segunda defensa.

Con `RANDOMIZE_DOWNLOAD_QUEUE=True`, los pendientes se ordenan mediante una
prioridad pseudoaleatoria derivada de `DOWNLOAD_RANDOM_SEED` y `video_id`. La
semilla hace reproducible el orden y estable frente a reanudaciones. Los videos
se agrupan por identidad de canal y se intercalan en round-robin: cada vuelta
toma como máximo un video por canal. Así se evita comenzar siempre por las
mismas fuentes sin eliminar candidatos ni alterar sus etiquetas de procedencia.

## 5. Cálculo dinámico del plan dirigido

### 5.1 Unidad de soporte

El soporte se mide por videos únicos y no por chunks. Esto evita que un programa
largo, dividido en muchos fragmentos, tenga más influencia que varios videos
cortos. Solo se leen filas pertenecientes a `train` o `validation`.

Para cada categoría de daño \(l\):

\[
S_l = \left|\{video\_id : l \in labels(video\_id)\}\right|
\]

El objetivo operativo de una corrida es el mayor soporte observado entre los
cuatro daños:

\[
T = \max_l S_l
\]

El déficit y el peso de adquisición son:

\[
D_l = \max(T-S_l, 0), \qquad
W_l = \frac{D_l}{\sum_j D_j}
\]

La categoría que ya alcanza \(T\) recibe peso cero en esa corrida. Los pesos se
recalculan cada vez que se ejecuta la celda, de modo que una ampliación integrada
puede modificar la siguiente prioridad.

### 5.2 Fallback sin datos previos

Si no existe el dataset model-ready, no hay filas etiquetadas utilizables, no
hay positivos de daño o todos los soportes son iguales, se activa
`fallback_equal`:

\[
W_l = 0.25
\]

para cada una de las cuatro categorías:

- `RACISMO_DISCRIMINACION`;
- `ATAQUE_POR_GENERO_IDENTIDAD`;
- `ACOSO_AMENAZA`; y
- `CONTENIDO_SEXUAL`.

Este fallback evita inventar una prioridad cuando no existe evidencia previa.

## 6. Preclasificación histórica y canales semilla

Las etiquetas existentes de `train+validation` se unen con las transcripciones
canónicas mediante `video_id`. Después se agrupan por `channel_id` para obtener,
por canal:

- videos etiquetados disponibles;
- videos positivos por daño; y
- tasa positiva histórica por daño.

Para reducir el efecto de canales observados una sola vez, el rendimiento por
categoría utiliza un suavizado:

\[
Y_{c,l} = \frac{P_{c,l}+0.5}{N_c+2}
\]

donde \(P_{c,l}\) es el número de videos positivos y \(N_c\) es el número de
videos etiquetados del canal. La confiabilidad por volumen es:

\[
R_c = \frac{N_c}{N_c+5}
\]

y la prioridad global del canal es:

\[
Score(c) = R_c \sum_l W_l Y_{c,l}
\]

Solo se consideran automáticamente canales con al menos tres videos históricos
y `channel_id` conocido. El resultado se combina con un catálogo curado de
canales semilla para conservar cobertura cuando el historial es pequeño. La
selección garantiza, cuando hay fuentes disponibles, al menos una semilla por
categoría con peso positivo y luego completa el cupo por score.

Esta etapa es una preclasificación para adquisición. Un canal no recibe una
etiqueta verdadera: sus tasas son estimaciones históricas y pueden cambiar.

## 7. Consultas temáticas y expansión de canales

El catálogo de consultas relaciona cada búsqueda con una o más categorías
objetivo. Solo se activan consultas asociadas a pesos positivos. Se garantiza
primero cobertura por categoría y después se ordenan las consultas restantes por
la suma de sus pesos.

Los videos devueltos por las consultas proporcionan `channel_id`. Esos resultados
se agrupan para proponer canales adicionales. La prioridad de expansión combina:

- el peso de las categorías que condujeron al canal;
- la cantidad de coincidencias temáticas, con bonificación acotada; y
- el mejor rango del canal en las búsquedas.

Los canales mejor puntuados se inspeccionan con una cuota pequeña de videos. El
parámetro `MAX_EXPANDED_CHANNELS` limita cuántos canales se incorporan y
`MAX_VIDEOS_PER_EXPANDED_CHANNEL` limita el piloto por canal. Esta expansión es
de un salto por corrida para contener el sesgo de muestreo en cadena.

## 8. Construcción de la cohorte dirigida

La cohorte se construye solo a partir de candidatos descubiertos en fuentes con
`sampling_mode="directed"`. Primero se excluyen todos los `video_id` canónicos.
Los candidatos se organizan por categoría objetivo y, dentro de cada categoría,
por fuente. Un round-robin interno impide que un solo canal o consulta monopolice
la categoría.

El round-robin ponderado externo acumula créditos según \(W_l\), selecciona la
categoría con mayor crédito y descuenta una unidad al incorporar un candidato.
Los videos multietiqueta pueden aparecer en varias colas, pero se materializan
una sola vez. Cada fila seleccionada recibe:

- `directed_priority_label`;
- `directed_selection_rank`;
- `target_category` como procedencia de selección; y
- metadatos de canal, consulta y rango de descubrimiento.

Con `MAX_DIRECTED_CANDIDATES=None`, la cohorte vigente conserva todos los
candidatos dirigidos inéditos. Un entero permite hacer un piloto deliberado sin
alterar el archivo acumulado. De forma análoga, `MAX_NEW_VIDEOS=None` incluye
toda la cola pendiente en la adquisición; un entero es solo un límite opcional
de prueba. El control ordinario de carga se realiza por lotes, no descartando
candidatos.

## 9. Descubrimiento técnico con `yt-dlp`

El descubrimiento usa `extract_flat="in_playlist"`, `skip_download=True` e
`ignoreerrors=True`. Para cada canal se normaliza la URL hacia `/videos`; para
cada consulta se usa `ytsearchN:`. La operación devuelve metadatos planos:

- `video_id` y URL de visualización;
- título;
- `channel_id` y nombre de canal;
- tipo, fuente y rango de descubrimiento; y
- motivo y categoría objetivo cuando la fuente es dirigida.

Los duplicados de una misma corrida se fusionan por `video_id`. Si un video fue
hallado por una fuente general y una dirigida, se conserva su condición dirigida
y se combinan las categorías objetivo. El archivo acumulado
`video_candidates.jsonl` conserva trazabilidad, pero el modo `directed` no lo usa
como cola indiscriminada.

La barra anuncia `started` antes de abrir cada fuente y muestra su nombre en
`fuente`; el contador avanza únicamente al terminar ese canal o consulta. Cada
resultado se escribe atómicamente en
`datos/raw/manifests/discovery_<modo>_checkpoint.json`. La identidad del
checkpoint incorpora URL o consulta, tipo, cuota y metadatos de selección: una
fuente sin cambios se reanuda sin red, mientras que un cambio relevante obliga a
descubrirla nuevamente. Los fallos también quedan registrados en el checkpoint,
pero se reintentan en una ejecución posterior; las fuentes exitosas anteriores
no se repiten.

`YT_SOCKET_TIMEOUT_SECONDS=45` limita cada operación HTTP de `yt-dlp`. Al
agotarse, se aplican los reintentos configurados y, si no hay recuperación, la
fuente se registra como `timeout`, se guarda su estado y el recorrido continúa
con la siguiente. El timeout no es un límite de duración total del canal: un
canal puede requerir varias operaciones HTTP válidas y pausas entre ellas.

## 10. Adquisición de subtítulos

La implementación activa recupera la técnica que funcionaba en el cuaderno
histórico. Para cada video realmente pendiente:

1. se crea un directorio temporal aislado;
2. `yt-dlp` se ejecuta con `skip_download=True`, `writesubtitles=True` y
   `writeautomaticsub=True`;
3. se solicitan `es-PE`, `es-419` y `es` en formato WebVTT;
4. `yt-dlp` gestiona la sesión HTTP, cabeceras, extracción, pausas y reintentos,
   y escribe únicamente las pistas VTT; no se descarga audio ni video;
5. el parser histórico materializa inicio, duración y texto, elimina marcas HTML
   y eventos vacíos, y selecciona el VTT con mayor texto útil;
6. si ningún VTT alcanza el umbral, `youtube-transcript-api` hace un último
   intento en memoria, primero manual y luego automático [7];
7. solo se acepta una transcripción con al menos
   `MIN_TRANSCRIPT_CHARACTERS=200`; y
8. el resultado se guarda atómicamente en la caché individual;
9. se anexa idempotentemente al JSONL pequeño de su canal bajo
   `datos/raw/transcripts_by_channel/`; un canal grande abre partes numeradas de
   hasta 25 MiB; y
10. después se incorpora una sola vez al canónico mediante `video_id`.

La versión anterior del flujo activo abría con `requests` la URL firmada de
`youtube.com/api/timedtext` que había localizado `yt-dlp`. Esa separación no
existía en el cuaderno histórico y podía perder parte del contexto HTTP que
administra `yt-dlp`; además generaba una secuencia de solicitudes directas. Esa
ruta fue retirada. Restaurar VTT reduce la regresión observada, aunque ninguna
técnica puede garantizar que YouTube nunca responda 429.

Cada registro canónico conserva `source_candidate`, `acquisition_status` y
`transcript_sha256`. El hash se calcula sobre los segmentos serializados de
forma estable y permite detectar cambios posteriores.

La escritura por canal ocurre antes que el append al canónico. Si el proceso se
interrumpe entre ambas operaciones, la reanudación vuelve a leer el checkpoint
por video, detecta que la partición ya contiene ese `video_id` y completa el
canónico sin duplicar. La materialización inicial recorre el canónico existente
sin borrarlo ni modificarlo. Cuando una fila histórica solo tiene nombre de
canal, se une a un `channel_id` únicamente si ese título identifica de forma
inequívoca un solo ID; en caso contrario conserva una partición estable basada
en el título.

## 11. Pausas, reintentos y aislamiento HTTP 429 por canal

La regulación ocurre en dos niveles:

- `yt-dlp` recibe `sleep_interval=5`, `max_sleep_interval=10`,
  `sleep_interval_requests=5` y `sleep_interval_subtitles=5`, además de tres
  reintentos para extracción y descarga y un timeout de socket de 45 segundos
  por operación HTTP;
- `ingest_incremental` procesa toda la cola en lotes de
  `NETWORK_BATCH_SIZE=10` y espera `NETWORK_BATCH_PAUSE_SECONDS=20` antes del
  lote siguiente.

El lote cuenta solo llamadas nuevas: reutilizar caché o filtrar un `video_id`
canónico no consume cupo ni provoca una pausa. La barra informa `lotes` y cambia
temporalmente a `Pausa entre lotes`. Con `MAX_NEW_VIDEOS=None`, terminado el
enfriamiento continúa con el siguiente lote hasta recorrer todos los candidatos.
Antes de cada pausa, las filas completas acumuladas se anexan al canónico y se
sincronizan a disco; la lista en memoria se vacía. Esto acota el uso de memoria y
convierte cada lote en otro punto de reanudación.

No existe un cortacircuito global. Con `EXCLUDE_CHANNEL_ON_429=True`, un HTTP 429:

1. registra únicamente el video actual como `rate_limited`;
2. identifica el canal por `channel_id`; si falta, usa título o URL del canal;
3. incorpora esa identidad al conjunto de canales excluidos de la corrida;
4. difiere sin red únicamente los candidatos posteriores del mismo canal; y
5. continúa normalmente con todos los demás canales.

La caché se consulta antes de esta exclusión: una transcripción ya descargada de
un canal afectado todavía se reutiliza. Si el candidato no contiene ninguna
identidad de canal, solo falla ese video y la cola general continúa. Los videos
diferidos permanecen fuera del canónico y se reintentan en una ejecución futura.
La barra separa `intentos_429`, `canales_429` y `pausa_429`; este último es el
número de videos diferidos de esos canales, no el número de respuestas 429.
La aleatorización no resuelve un bloqueo general de la IP: si muchos canales
distintos devuelven 429, debe detenerse la corrida y reanudarse cuando haya
terminado el enfriamiento impuesto por la plataforma.

## 12. Taxonomía de fallos

Los errores heterogéneos se reducen a motivos auditables:

| Motivo | Interpretación operativa |
|---|---|
| `stale_channel_or_no_videos_tab` | Canal obsoleto, 404 o sin pestaña de videos |
| `members_only` | Contenido exclusivo para miembros |
| `unavailable_or_private` | Video privado, retirado o no disponible |
| `no_spanish_subtitles` | No existe pista en los idiomas configurados |
| `subtitle_too_short` | Existe texto, pero no alcanza los 200 caracteres exigidos |
| `access_challenge` | YouTube solicita autenticación o verificación anti-bot |
| `rate_limited` | HTTP 429 o límite de solicitudes |
| `timeout` | Tiempo de espera agotado |
| `scheduled_or_upcoming` | Estreno o video todavía no disponible |
| `fetch_error` | Error no clasificado |

Los fallos de fuentes de la última corrida se escriben en JSON. Cada fallo de
video se deduplica mediante un `failure_id` derivado de `video_id` y motivo y se
persiste inmediatamente, sin esperar el cierre de una corrida larga. Cada
transcripción exitosa también se escribe de inmediato en su caché individual.
Si hay una interrupción, la próxima ejecución reutiliza los checkpoints de
fuentes exitosas, transcripciones y fallos por video, y solo vuelve a recorrer
lo que continúa pendiente. Las fuentes que terminaron con error se reintentan
sin repetir los canales o consultas exitosos.

## 13. Artefactos y trazabilidad

| Artefacto | Papel |
|---|---|
| `datos/raw/video_candidates.jsonl` | Archivo acumulado y append-only de candidatos descubiertos |
| `datos/raw/directed_candidates_latest.jsonl` | Cohorte dirigida vigente y ordenada |
| `datos/raw/manifests/directed_plan_latest.json` | Soportes, déficits, pesos, fuentes, expansión y tamaño de cohorte |
| `datos/raw/manifests/discovery_<modo>_checkpoint.json` | Resultado atómico y reanudable por canal o consulta |
| `datos/raw/fallos_descubrimiento_ultima_ejecucion.json` | Fallos de canales y consultas de la corrida |
| `datos/raw/fallos_adquisicion.jsonl` | Fallos deduplicados por video y motivo |
| `datos/raw/transcripts_cache/*.json` | Checkpoints atómicos por video |
| `datos/raw/transcripts_raw.jsonl` | Vista canónica de transcripciones |
| `datos/raw/transcripts_by_channel/*.jsonl` | Checkpoint sincronizable por canal, deduplicado por `video_id` |
| `datos/raw/transcripts_by_channel/index.json` | Inventario de canales, tamaños y SHA-256 de cada partición |

La cohorte vigente se sobrescribe atómicamente porque representa una selección
concreta. El archivo general de candidatos y el canónico son incrementales y se
deduplican por identificador.

### 13.1 Sincronización y continuidad entre máquinas

Las reglas de Git incluyen las particiones por canal, el índice, los candidatos
generales y dirigidos, los manifiestos de adquisición y todo
`resultados/colab_bundle/`. Permanecen fuera de Git:

- `datos/raw/transcripts_cache/`, porque solo acelera reintentos locales;
- `datos/raw/transcripts_raw.jsonl`, porque se recompone sin red desde las
  particiones por canal;
- `datos/processed/chunks_v2.jsonl` como artefacto de trabajo, porque el
  troceado es determinista y barato; su copia comprimida se conserva únicamente
  dentro del bundle requerido por Colab; y
- `datos/model_ready/v2/dataset_5_salidas.jsonl` sin comprimir, porque el bundle
  conserva una copia gzip verificable del snapshot, que no es barato recrear al
  contener decisiones humanas y pseudoetiquetado con procedencia.

Tras clonar, `python tools/restore_synced_checkpoints.py` verifica los hashes de
las particiones y del bundle, completa el canónico sin borrar filas y
descomprime atómicamente chunks y dataset. Los cuadernos `03_01`–`03_08`
ejecutan además una verificación previa del dataset: restauran solo cuando falta
y se detienen si la copia local existente tiene otro SHA-256.

El dataset comprimido actual ocupa cerca de 20,3 MiB frente a 104,2 MiB sin
comprimir. No se particiona mientras permanezca holgadamente por debajo de 50
MiB. Si crece hasta ese orden, la primera partición será por `split` y, si aún
fuera necesario, por partes numeradas; el manifiesto deberá seguir describiendo
el hash de cada archivo y el hash lógico del dataset completo.

## 14. Reconstrucción desde cero

La celda de configuración contiene una activación comentada:

```python
RESET_VIDEO_DATASET = ""
# RESET_VIDEO_DATASET = "ARCHIVAR_Y_REINICIAR_DATASET_VIDEOS"
```

Al descomentar la segunda asignación, el cuaderno exige la frase exacta y mueve
los artefactos activos conocidos a:

```text
archivo/reinicios_dataset_videos/<timestamp>/
```

Se archivan las vistas activas raw, processed, etiquetadas y model-ready, además
de modelos y resultados directamente derivados. No se eliminan el código, la
configuración, la documentación ni `datos/ampliacion/`. Después se crea
`datos/raw/manifests/rebuild_from_zero.json`, que impide reimportar
automáticamente los snapshots históricos.

El reinicio es idempotente: si el marcador ya existe, una nueva ejecución no
vuelve a mover datos. Para recuperar el estado anterior deben restaurarse las
rutas desde el directorio indicado por `archive_path` y retirarse el marcador de
forma deliberada.

## 15. Limitaciones y controles de sesgo

- La disponibilidad de subtítulos varía por canal, idioma y fecha; los videos
  adquiridos no representan todos los videos publicados.
- Las consultas temáticas enriquecen daños concretos y alteran deliberadamente
  la distribución observada.
- El rendimiento histórico de un canal puede cambiar y no demuestra que todo su
  contenido pertenezca a una categoría.
- Las categorías multietiqueta pueden compartir candidatos; deduplicar evita
  descargas repetidas, pero no elimina la dependencia entre etiquetas.
- Los errores de subtitulado automático pueden afectar el texto que luego se
  etiqueta [2].
- El modo dirigido mejora cobertura de entrenamiento; no produce una muestra
  probabilística ni una estimación poblacional.
- La etiqueta final debe proceder del flujo de anotación y validación humana. La
  procedencia de scraping nunca sustituye esa decisión.

## 16. Validación automatizada

Las pruebas cubren:

- fallback equitativo sin historia;
- exclusión de `test`;
- cálculo de déficits por videos;
- selección de consultas y expansión de canales;
- exclusión de videos ya procesados;
- round-robin de la cohorte dirigida;
- cobertura completa con pausas entre lotes;
- descarga VTT por la ruta integrada de `yt-dlp` y umbral mínimo de integridad;
- reinicio confirmado, acotado, recuperable e idempotente;
- aislamiento por canal ante HTTP 429, sin corte global; y
- presencia de los controles y llamadas relevantes en el cuaderno activo.

La auditoría estática también verifica que el cuaderno tenga código sintácticamente
válido, metadatos bibliográficos, citas numéricas IEEE y una celda final de
referencias coherente.

## Referencias

[1] yt-dlp contributors, "yt-dlp: A Feature-Rich Command-Line Audio/Video Downloader," GitHub repository, 2026. [Online]. Available: https://github.com/yt-dlp/yt-dlp. Accessed: Aug. 5, 2026.

[2] R. Tatman, "Gender and Dialect Bias in YouTube's Automatic Captions," in *Proc. 1st ACL Workshop Ethics NLP*, 2017, pp. 53–59, doi: 10.18653/v1/W17-1606.

[3] YouTube, "Terms of Service," Nov. 2023. [Online]. Available: https://www.youtube.com/t/terms. Accessed: Aug. 5, 2026.

[4] A. S. franzke, A. Bechmann, M. Zimmer, et al., "Internet Research: Ethical Guidelines 3.0," Association of Internet Researchers, 2020. [Online]. Available: https://aoir.org/reports/ethics3.pdf.

[5] Y. Fairstein, O. Kalinsky, Z. Karnin, et al., "Class Balancing for Efficient Active Learning in Imbalanced Datasets," in *Proc. 18th Linguistic Annotation Workshop*, 2024, pp. 77–86, doi: 10.18653/v1/2024.law-1.8.

[6] Y. Huang, B. Giledereli, A. Köksal, et al., "Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution," in *Proc. EMNLP*, 2021, pp. 8153–8161, doi: 10.18653/v1/2021.emnlp-main.643.

[7] J. Depoix and contributors, "YouTube Transcript API: Python API for Retrieving YouTube Transcripts and Subtitles," GitHub repository, 2026. [Online]. Available: https://github.com/jdepoix/youtube-transcript-api. Accessed: Aug. 6, 2026.
