# Presentacion Beamer final

> **Estado de contrato (2026-08-05).** Los resultados presentados corresponden al baseline ejecutado de cuatro daños y `SEGURO` derivado. El flujo activo incorpora `SEGURO` como quinta salida aprendida; sus métricas quedan pendientes de reentrenamiento y no se infieren de las tablas históricas.

Esta carpeta contiene la presentación académica del Grupo 4 para el artículo **Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural**. El Beamer resume visualmente el paper IEEE/DSR y su aporte: moderación semiautomática con modelos compactos, alertas temporales y decisión de un supervisor. No debe conservar lenguaje de propuesta, plan de trabajo o MVP futuro.

## Archivos

- `presentacion_grupo4.tex`: fuente autoritativa en LaTeX Beamer.
- `presentacion_grupo4.pdf`: salida derivada; debe regenerarse despues de cualquier cambio.
- `Moderador_Contenido_YouTube_PLN.pptx`: formato alternativo no autoritativo. No debe entregarse como equivalente mientras no se regenere y compare diapositiva por diapositiva con el Beamer final.

## Fuentes de verdad

La presentacion debe concordar con:

1. `../Documento_final_paper/paper_moderador_contenido_youtube_ieee.tex` y `../Documento_final_paper/referencias.bib`;
2. `../config/taxonomia_v2.json`, `../docs/TAXONOMIA_V2.md` y `../flujo/` para el contrato activo;
3. `../archivo/estructura_anterior/Cuadernos/04_MATRIZ_ENTRENAMIENTO_4_ETIQUETAS.md` para el baseline;
4. `../archivo/contrato_4_danos_seguro_derivado/resultados/metricas/comparacion_final_4/comparacion_todos_modelos_4.csv`;
5. los informes en `../archivo/contrato_4_danos_seguro_derivado/resultados/`;
6. `../docs/MATRIZ_TRAZABILIDAD.md` para separar evidencia ejecutada de trabajo pendiente.

No use el PPTX antiguo, salidas aisladas de notebooks ni resultados preliminares de cinco categorías como fuente de la conclusión final.

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

## Narrativa recomendada

La presentacion debe dedicar una idea principal a cada diapositiva y priorizar diagramas, tablas breves y graficos legibles:

1. portada con proyecto, autores e institucion;
2. brecha real, subyacente y tecnologica;
3. problema general, objetivo y artefacto DSR;
4. iteraciones DSR y pipeline de datos;
5. corpus, embudo Flash--Pro--revisión final y fuentes de etiqueta;
6. herramientas y hardware: CPU local, Colab/CUDA y alcance del registro L4;
7. evolución del baseline de cuatro daños al contrato activo de cinco salidas, con `SEGURO` aprendido;
8. taxonomía: 12 fenómenos de daño, dos estados seguros y tres flags;
9. matriz de modelos y arquitecturas;
10. protocolo train/validation/test y selección sin fuga;
11. comparación final de clásico, E5 y Qwen;
12. Qwen: épocas, calibración y costo de revisión;
13. planos frente a jerárquicos;
14. frontend 05, consenso 2 de 3 y retroalimentación humana;
15. conclusiones narrativas que cubren todo lo que el estudio se propuso lograr;
16. limitaciones y trabajo futuro;
17. última diapositiva con referencias esenciales, GitHub público y Google Drive de acceso controlado.

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

El Beamer solo se cierra despues de compilar y revisar la version definitiva del paper.
