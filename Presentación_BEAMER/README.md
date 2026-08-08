# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Presentación Beamer actualizable

> **Estado de contrato (2026-08-08).** El flujo activo aprende `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente y los cuatro daños son multietiqueta; los casos indeterminados se difieren y no entran al entrenamiento. La presentación incorpora el corpus y el corte cuantitativo vigente del etiquetado Flash→Pro. Los resultados de entrenamiento disponibles pertenecen al baseline histórico de cuatro daños con `SEGURO` derivado y aparecen únicamente con el rótulo **Por actualizar con la última versión**.

Esta carpeta contiene la presentación académica del Grupo 4 para el artículo **Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural**. El Beamer resume visualmente el diseño IEEE/DSR y su aporte: moderación semiautomática con modelos compactos, alertas temporales y decisión de un supervisor. Las etapas activas pueden aparecer como **En progreso**; una proyección debe distinguirse siempre de un resultado medido.

## Archivos

- `presentacion_grupo4.tex`: fuente autoritativa en LaTeX Beamer.
- `presentacion_grupo4.pdf`: salida derivada; debe regenerarse despues de cualquier cambio.
- `GUIA_REVISION_ACTUALIZACION_2026-08-08.md`: requisitos, jerarquía de fuentes, estructura y lista de control de esta actualización.
- `Moderador_Contenido_YouTube_PLN.pptx`: formato alternativo no autoritativo. No debe entregarse como equivalente mientras no se regenere y compare diapositiva por diapositiva con el Beamer final.

## Fuentes de verdad

La presentación debe concordar, en este orden, con:

1. manifiestos y artefactos ejecutables vigentes de `../datos/`, `../resultados/`, `../config/` y `../src/`;
2. `../docs/`, el `README.md` raíz y los `README` de `../flujo/`;
3. `../Documento_final_paper/referencias.bib` para metadatos bibliográficos y el manuscrito para resultados que ya estén rotulados como históricos;
4. `../archivo/` únicamente como evidencia histórica preservada, nunca como definición del flujo activo.

No use el PPTX antiguo, salidas aisladas de notebooks ni resultados históricos sin rótulo como fuente de la conclusión final. Ante discrepancias documentales, prevalece el artefacto actual verificable; la cifra anterior se conserva solo si aporta contexto y se identifica como histórica.

## Compilacion reproducible

Desde la raiz del repositorio:

```powershell
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -file-line-error "Presentación_BEAMER/presentacion_grupo4.tex"
```

O desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error presentacion_grupo4.tex
```

Antes de entregar, revise `presentacion_grupo4.log`. No se aceptan citas/referencias indefinidas, elementos fuera de pagina ni cajas `Overfull`. El PDF debe usar fuentes vectoriales incrustadas y conservar la relacion 16:9.

## Estructura narrativa vigente

La presentacion debe dedicar una idea principal a cada diapositiva y priorizar diagramas, tablas breves y graficos legibles:

1. Introducción.
2. Problemática: daño en YouTube, tres niveles, brechas específicas, objetos, unidad de análisis, objetivos y DSR.
3. Descripción de los datos: canales, descubrimiento, scraping incremental y corpus.
4. Preprocesamiento y etiquetado: limpieza, selección de longitud, contrato v2.1, cascada Flash→Pro, costo, caché, calibración, frontend humano y snapshot.
5. Modelamiento y evaluación: seis ramas, fundamentos, protocolo por video, baseline histórico rotulado y ensamble supervisado.
6. Resultados: productos actuales, en progreso e históricos por actualizar.
7. Conclusiones, límites y trabajos futuros.

Puede distribuir el contenido en mas de una diapositiva cuando una figura lo requiera, pero debe eliminar listados extensos de canales, estructura del futuro paper y cronogramas ya concluidos.

## Reglas de contenido

- Mostrar resultados con nombre de metrica y split; no mezclar test historico, ampliado y comun 4:1.
- Reportar métricas y magnitudes estimadas con dos cifras significativas, usando el mismo redondeo que el paper. Conservar exactos los conteos, el tamaño muestral, los años, las versiones, los identificadores, los hashes y los parámetros fijados por protocolo; las constantes físicas pueden mantener la precisión necesaria. Calcular y comparar siempre con los valores completos del artefacto fuente.
- Explicar que la regla de seleccion uso validation, pero que ya existia una evaluacion de test de la epoca 2; no afirmar cegamiento completo.
- Presentar la epoca 3 de Qwen como seleccion operacional; la epoca 2 sigue siendo el maximo de AP de validation (campo historico `PR-AUC`).
- Mostrar que Qwen plano no fue superado por sus variantes jerarquicas.
- Indicar que el consenso operativo es mayoria 2 de 3, no unanimidad.
- Enunciar primero el logro: el sistema prioriza, muestra evidencia y entrega un diagnóstico preliminar a un supervisor usando recursos asequibles. Después delimitar que no se evaluó bloqueo o sanción sin revisión.
- Responder narrativamente a todo lo que el estudio se propuso lograr, sin usar «objetivo», `O1`/`O2` ni listas de cumplimiento en las conclusiones. Indicar para cada distancia entre la situación inicial y la deseada si se eliminó, se redujo o permanece, y ligar lo pendiente con acciones o trabajo futuro.
- En la diapositiva de disponibilidad, indicar qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorios; reservar SHA, commits y hashes concretos para manifiestos técnicos fuera del cuerpo.
- No afirmar uso de Whisper/ASR sin evidencia ejecutada.
- No afirmar que todo el proyecto fue “sin APIs”: el pseudoetiquetado uso servicios externos autorizados, aunque la inferencia desplegada es local.
- No afirmar doble anotacion ni Cohen kappa.
- No presentar la prevalencia enriquecida 4:1 como prevalencia de producción. La revisión humana asistida conserva la decisión y su procedencia.

Las referencias breves de las diapositivas deben usar las mismas claves y fuentes verificadas que el paper. Si la bibliografia no cabe, divida las referencias o deje la lista completa en el paper; nunca permita texto recortado.

## Revision en dos pases

1. **Cientifico:** contrastar cifras, etiquetas, modelos, seleccion y conclusiones con las fuentes de verdad; verificar que toda limitacion decisiva sea visible.
2. **Editorial y visual:** leer el PDF proyectado, reducir texto, revisar contraste/escala de grises, fuentes, referencias, ortografia y ausencia de desbordes. En gráficos de pocos modelos o métricas, reducir espacios entre grupos y mantener etiquetas y valores visibles; en diagramas de cajas, ajustar fuente, dimensiones, separación y rutas ortogonales hasta eliminar toda superposición.

El Beamer se considera técnicamente cerrado cuando compila sin cajas `Overfull` ni referencias indefinidas y todas sus diapositivas se revisan visualmente. Los rótulos permiten actualizar cifras y capturas sin confundir el cierre editorial de esta versión con el cierre futuro del etiquetado y entrenamiento.
