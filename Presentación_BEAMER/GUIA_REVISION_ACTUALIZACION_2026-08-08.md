# Guía de revisión y actualización de la presentación Beamer

**Fecha de corte:** 2026-08-08  
**Fuente del encargo:** requisitos entregados por el usuario para actualizar
`presentacion_grupo4.tex`.  
**Estado:** lista de control autoritativa para esta adecuación.

## Propósito

La presentación debe explicar, de forma visual y autocontenida, por qué se
necesita un artefacto de moderación semiautomática para subtítulos de videos
peruanos de YouTube, cómo se construyó mediante *Design Science Research* (DSR),
qué evidencia existe en cada etapa y qué permanece pendiente. El sistema
prioriza y presenta evidencia temporal; un supervisor conserva la decisión.
No se afirmará moderación autónoma, validez jurídica ni eficacia productiva.

## Reglas generales y específicas aplicables

1. Seguir `Presentación_BEAMER/README.md`, las guías IEEE/DSR del paper y
   `Documento_final_paper/AUDITORIA_CITAS_Y_ESTILO.md`.
2. Mantener una idea principal por diapositiva, formato 16:9, tipografía
   proyectable, alto contraste y figuras preferentemente vectoriales.
3. Distinguir siempre:
   - **resultado actual:** artefacto del contrato v2.1 o de la campaña activa;
   - **resultado histórico:** evidencia ejecutada con el contrato anterior;
   - **proyección:** cálculo condicionado, nunca resultado final;
   - **decisión local:** regla creada para este artefacto;
   - **evidencia externa:** idea o método que exige referencia.
4. Mostrar métricas estimadas con dos cifras significativas. Mantener exactos
   conteos, tamaños de muestra, versiones, fechas y parámetros fijados.
5. No describir acuerdo Flash–Pro como exactitud humana ni el test histórico
   enriquecido 4:1 como prevalencia de YouTube.
6. No afirmar uso de Whisper/ASR, doble anotación, Cohen kappa, ausencia total
   de APIs o validación productiva.
7. Toda cifra debe conservar una ruta a su artefacto; toda idea, algoritmo,
   arquitectura, modelo base o limitación externa debe citar su fuente primaria
   o técnica pertinente.
   - Una referencia interna solo puede documentar un resultado propio ejecutado.
     No sustituye la fuente externa de una definición, categoría, arquitectura,
     algoritmo, métrica, herramienta o política.
   - La cita se coloca junto a la afirmación que respalda y su entrada completa
     se presenta en la bibliografía IEEE; no se admiten listas decorativas de
     autores sin correspondencia proposición--fuente.
8. La presentación debe compilar sin referencias indefinidas ni cajas
   `Overfull` y revisarse visualmente diapositiva por diapositiva.

## Etiquetas obligatorias de estado

Se usarán dos rótulos, con texto exactamente reconocible:

- **En progreso:** la etapa vigente está ejecutándose y ya posee resultados
  parciales medidos. Debe indicarse la fecha/hora del corte.
- **Por actualizar con la última versión:** la diapositiva solo dispone de
  resultados históricos o preliminares y será reemplazada cuando exista el
  artefacto final del contrato v2.1.

No se usará rótulo cuando el resultado actual ya esté cerrado y verificado.
Las etiquetas deben estar en una posición y color consistentes, sin ser el único
canal semántico.

## Estructura narrativa obligatoria

### 1. Introducción

- Portada.
- Propósito y alcance: texto de subtítulos, evidencia temporal, diagnóstico
  preliminar y decisión humana.
- Mapa de las siete secciones.

### 2. Problemática

- Describir por qué violencia, acoso, discriminación, ataques por identidad y
  contenido sexual requieren moderación en plataformas como YouTube.
- Separar tres niveles:
  - **problema real:** escala y costo de la revisión manual;
  - **problema subyacente:** ambigüedad, contexto, ironía, modismos y escasez o
    desbalance de etiquetas pertinentes al Perú;
  - **problema técnico/de modelamiento:** ausencia de un artefacto reproducible
    que segmente, compare, calibre, priorice y entregue evidencia trazable.
- Presentar el problema general como *statement*, nunca como pregunta: existe
  la necesidad de construir y evaluar el artefacto semiautomático.
- Presentar los problemas específicos como brechas necesarias para construir el
  artefacto; publicar o difundir el sistema es una tarea, no un problema.
- Explicar que DSR aborda la brecha técnica mediante construcción y evaluación
  iterativa del artefacto.
- Definir:
  - **problema de estudio:** moderación semiautomática de daño textual;
  - **objeto de estudio:** expresiones potencialmente dañinas en subtítulos de
    videos peruanos de YouTube;
  - **objeto modelado:** fragmento textual temporal con cinco salidas, etiquetas
    finas y flags;
  - **unidad de análisis:** fragmento (*chunk*) con `video_id`, texto e intervalo.
- Mostrar cómo literatura general y evidencia peruana/institucional informan la
  taxonomía. No denominarla validada por expertos peruanos ni taxonomía legal.
- Encadenar objetivos por entradas y salidas:

| Etapa/objetivo específico | Entradas | Salida que alimenta la siguiente etapa |
|---|---|---|
| construir y depurar el corpus | canales/candidatos, subtítulos, VTT, criterios de inclusión | transcripciones y chunks trazables |
| definir y aplicar la taxonomía | chunks, literatura, evidencia peruana, prompt y contrato | propuestas de cinco salidas con procedencia |
| consolidar etiquetas | Flash, Pro, flags y decisiones humanas | snapshot etiquetado inmutable |
| entrenar y comparar | snapshot, splits por video, familias y configuraciones | candidatos calibrados y comparables |
| integrar el artefacto | candidatos elegidos, umbrales, texto/URL | evidencia, alerta y decisión humana registrable |

El objetivo general debe expresar diseñar y evaluar el artefacto; la metodología
explica cómo se alcanzan estos productos. Las conclusiones responderán a esta
cadena en prosa, sin rótulos `O1`, `O2` ni frases de “cumplimiento”.

### 3. Descripción de los datos

- Describir el corpus integrado y no inflar la muestra con rondas superpuestas.
- Explicar tipos de canales y la búsqueda dirigida/reanudable: canales previos,
  consultas temáticas, expansión por canales y asignación *round-robin*
  ponderada por déficits/rendimiento histórico.
- Presentar la metodología de adquisición vigente: `yt-dlp`, subtítulos manuales
  o automáticos, `youtube-transcript-api` solo como respaldo, sin descargar
  audio/video y sin ASR.
- Incluir criterios de inclusión/exclusión, deduplicación, pausas, reintentos,
  aislamiento por canal ante 429 y checkpoints por VTT/video.
- Reportar el corte actual: 5 002 transcripciones, 4 992 videos con chunks,
  166 940 chunks y 1 451.17 horas; cualquier cifra de canales o etiquetas que
  todavía corresponda al snapshot histórico llevará el rótulo correspondiente.
- Describir tamaño por categoría solo cuando exista el snapshot actual cerrado;
  hasta entonces presentar la taxonomía y marcar la distribución como pendiente.

### 4. Preprocesamiento y etiquetado

- Limpieza: normalización Unicode/espacios, conservación de texto y tiempos,
  deduplicación y procedencia; no eliminar expresiones que contienen la señal.
- Troceado vigente v2.2.0: objetivo 30 s, máximo 600 caracteres, mínimo 90
  caracteres y solapamiento de 12 palabras.
- Explicar la selección de longitud 15/20/25/30/35 s, perfil clásico decisorio,
  confirmación MiniLM/Gemma y bootstrap agrupado por `video_id`. Resultado:
  conservar 30 s; el contraste 20/30 mostró no inferioridad de 20 s, no
  superioridad.
- Explicar la cascada vigente:
  1. recuperar solo equivalencias históricas exactas y unívocas;
  2. preflight sin corpus;
  3. Flash y Pro con `thinking=disabled`, prompt operacional y JSON validado;
  4. Flash solo sobre pendientes, 5 chunks por solicitud y hasta 32 solicitudes;
  5. Pro dirigido a daño, abstención, baja confianza y control seguro;
  6. precedencia humano > Pro > Flash y exclusión de diferidos/rechazados.
- Mostrar interrupción/reanudación: `fsync`, checkpoints, cuarentena y omisión
  de `chunk_id` válidos ya guardados.
- Incluir resultados disponibles del corte documentado del etiquetado: tasas de
  recuperación, calibración, velocidad, caché, costo y error rechazado. Marcar
  la campaña **En progreso**.
- Aclarar que 80.41 % de acuerdo exacto y 99.77 % binario son acuerdos
  Flash–Pro, no exactitud humana. La eficacia humana permanece pendiente.
- Mostrar el esquema semiautomático y una captura legible del frontend de
  validación humana; explicar aceptar, modificar, diferir o excluir y que la
  sugerencia puede permanecer oculta para reducir anclaje.
- El tamaño final del dataset etiquetado, las particiones y la distribución por
  salida se marcarán **En progreso** hasta cerrar `02_05`.

### 5. Modelamiento y evaluación

- Comparar las familias actuales implementadas:
  - clásicos TF–IDF: Dummy, ComplementNB, regresión logística, SVM lineal y SGD;
  - Transformers planos: MiniLM multilingüe y E5-small multilingüe;
  - cascada, multitarea, Qwen3-0.6B LoRA y Qwen estructurado.
- Explicar para cada familia fundamento, ventaja, desventaja y costo/recursos,
  citando TF–IDF, SVM, Transformer, Sentence-BERT/MiniLM, E5, Qwen3 y LoRA.
- Presentar split estable por `video_id`, selección y calibración en validation,
  test solo después de congelar, AP macro de daño como criterio principal y
  bootstrap agrupado por video cuando corresponda.
- Los resultados disponibles de entrenamiento pertenecen al baseline histórico
  de cuatro daños con `SEGURO` derivado. Toda diapositiva con esas cifras debe
  llevar **Por actualizar con la última versión**.
- No transferir AP/F1 históricos al contrato v2.1 de cinco salidas.
- Describir el ensamble operacional planificado: mejor clásico, mejor
  Transformer y mejor Qwen; mayoría 2 de 3 para alertar o remitir al supervisor.
  Hasta ejecutar `03_07` con el snapshot nuevo, marcarlo **En progreso** o como
  diseño del prototipo, no como eficacia demostrada.
- Mostrar una captura legible del frontend productivo y explicar texto/URL,
  cinco scores, umbrales, evidencia temporal, motivos de revisión y modo sombra.

### 6. Resultados

Organizar los resultados por producto, no por celdas:

1. scraping y corpus actual;
2. limpieza/troceado y elección de 30 s;
3. recuperación, calibración, velocidad, caché y costo del etiquetado activo;
4. snapshot etiquetado y distribución: **En progreso**;
5. entrenamiento/comparación v2.1: **Por actualizar con la última versión**;
6. frontend de etiquetado: demostración funcional, no validación de eficacia;
7. frontend productivo: demo/piloto en modo sombra, no despliegue autónomo.

Cada comparación debe mostrar universo, split, métrica y, si existe, intervalo
o prueba estadística. Un valor histórico se conserva para contexto y se rotula;
un valor inexistente se representa mediante el estado pendiente, no con cero.

### 7. Conclusiones y trabajos futuros

- Responder narrativamente a la cadena de objetivos: corpus, taxonomía,
  etiquetado, comparación y prototipo.
- Indicar si cada brecha fue eliminada, reducida o permanece.
- No usar `O1`, `O2`, “se cumplió el objetivo” ni prometer producción autónoma.
- Límites mínimos: pseudoetiquetado asistido, falta de gold humano ciego,
  riesgo de anclaje, falta de doble anotación/kappa, muestreo dirigido,
  subtítulos solamente, sesgo por canales, falta de semillas repetidas y
  evaluación prospectiva/productiva.
- Trabajo futuro: cerrar etiquetado y snapshot v2.1; entrenamiento y
  recalibración; holdout prospectivo de prevalencia natural con doble revisión;
  pruebas adversariales; semillas; calibración por subgrupo; deriva; latencia,
  privacidad, seguridad y capacidad del supervisor.

## Fuentes internas prioritarias

- Datos/scraping: `docs/METODOLOGIA_SCRAPING.md`,
  `docs/MATERIALIZACION_TROCEADO.md` y manifiestos de `datos/`.
- Longitud: `docs/OPTIMIZACION_LONGITUD_CHUNKS.md` y
  `docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md`.
- Taxonomía: `config/taxonomia_v2.json`, `docs/TAXONOMIA_V2.md` y
  `docs/MATRIZ_EVIDENCIA_TAXONOMIA.md`.
- Etiquetado: `docs/METODOLOGIA_ETIQUETADO_CASCADA.md` y
  `resultados/ETIQUETADO_CASCADA_CORTE_2026-08-08.md`.
- Entrenamiento: `flujo/03_entrenamiento/README.md` y artefactos v2 cuando
  existan; `archivo/` solo para resultados rotulados como históricos.
- Producción: `flujo/04_produccion/README.md`, frontend y registro verificable.
- Referencias: `Documento_final_paper/referencias.bib` y matriz de trazabilidad.

## Resultado de la revisión documental complementaria

Se inventariaron 85 documentos únicos dentro del alcance solicitado: los 17
`*.md` de `docs/`, 50 `*.md` bajo `archivo/` y los 26 `README.md`/`readme.md`
del repositorio, descontando intersecciones. La revisión condujo a estas
decisiones:

- `archivo/README.md` declara que todo `archivo/` conserva evidencia anterior y
  no es dependencia del flujo activo. Por ello, sus métricas de cuatro daños,
  sus capturas y su bundle 05 solo pueden aparecer como evidencia histórica.
- Los informes históricos siguen siendo útiles para explicar decisiones,
  límites y comparaciones ejecutadas, pero no definen rutas, salidas, umbrales
  ni desempeño del contrato v2.1.
- Los documentos actuales y los manifiestos coinciden en 5 002
  transcripciones, 4 992 videos con chunks y 166 940 chunks. El índice distingue
  336 canales o claves de procedencia y 339 archivos por canal; no se confunden
  ambos conteos.
- Se corrigió `datos/README.md`: la regulación vigente añade 15 segundos entre
  lotes de diez, además de pausas internas de 2.5--10 segundos; la campaña
  `04_201`--`04_208` es histórica y el entrenamiento activo vive en
  `flujo/03_entrenamiento/`.
- `Presentación_BEAMER/README.md` ahora coloca manifiestos, código y documentos
  vigentes por encima del paper y de `archivo/`, y adopta las siete secciones y
  los rótulos de estado de esta guía.

Ante una discrepancia residual se mantiene la regla de autoridad de esta guía:
artefacto ejecutable actual, manifiesto actual, documentación vigente y, por
último, evidencia histórica explícitamente rotulada.

## Capturas y elementos visuales

- Preferir TikZ/PGFPlots y tablas breves para procesos y cifras.
- Incluir capturas reales de los frontends de etiquetado y producción; no usar
  capturas de notebooks. Si no existe una captura verificable, usar un mockup
  explícitamente rotulado como interfaz implementada y reemplazarlo al generar
  la captura.
- Mantener flechas ortogonales y no permitir líneas sobre cajas o texto.
- Revisar contraste en escala de grises y legibilidad al proyectar.

## Criterios de cierre

- Las siete secciones aparecen en el orden solicitado.
- Problema general/específicos, objetivo general/específicos, entradas, salidas,
  objetos y unidad de análisis están explícitos.
- Scraping, preprocesamiento, etiquetado, entrenamiento y producción reflejan el
  flujo activo y nombran los modelos/herramientas usados.
- Todo resultado histórico o pendiente tiene el rótulo correcto.
- No hay cifras sin fuente, resultados inventados ni mezcla de contratos.
- Las conclusiones se enlazan narrativamente con la evidencia y los límites.
- El `.tex` compila, el log no contiene referencias indefinidas ni `Overfull`, y
  todas las diapositivas se inspeccionan visualmente.
