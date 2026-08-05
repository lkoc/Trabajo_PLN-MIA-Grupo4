# Figuras reproducibles del paper

Los diagramas del artículo son fuentes vectoriales TikZ/PGFPlots compiladas
desde LaTeX. Las capturas de interfaz son imágenes raster obtenidas de los
artefactos funcionales y se conservan junto con sus fuentes reproducibles.

- pipeline_moderacion.tex: adquisición, preanotación, adjudicación y entrenamiento.
- brechas_problema.tex: brecha real, subyacente y tecnológica.
- ciclo_dsr.tex: fases DSR y cronología de iteraciones.
- datos_taxonomia.tex: cinco salidas activas (SEGURO y cuatro daños), 12 fenómenos de daño, dos estados seguros finos y tres flags transversales en carriles separados.
- fuentes_dataset.tex: criterios de adquisición y volumen total del corpus integrado.
- familias_modelo.tex: familias y estructuras comparadas.
- resultados_finales.tex: métricas de los ganadores por familia.
- despliegue.tex: artefacto operativo, consenso 2 de 3 y retroalimentación.
- ontologia_trazabilidad.tex: grafo de conceptos y relaciones del anexo.
- captura_entorno_etiquetado.png: vista del entorno de revisión humana.
- captura_operacion_texto_qwen.png: chunk real de validación procesado con Qwen3--LoRA.
- captura_operacion_texto_consenso.png: el mismo chunk procesado con consenso 2 de 3.
- captura_operacion_youtube_qwen.png: URL de un video público con subtítulos procesada con Qwen3--LoRA.
- captura_operacion_youtube_consenso.png: la misma URL procesada con consenso 2 de 3.
- captura_panel_operacion.evidence.json: registro conjunto de las cuatro corridas, con estado del servidor, fuentes, modos, respuestas visibles y conteos antes/después.
- ../../archivo/implementacion_anterior/scripts_auxiliares/capturar_panel_operacion_ejecuciones_reales.py: automatización histórica que abre la interfaz, envía los cuatro formularios, espera las inferencias y genera recortes textuales sin inyectar resultados en el HTML.

Las inferencias temporales empleadas para estas capturas se eliminan de la base
del paquete después de conservar la evidencia; así, el despliegue reproducible
no se entrega con estadísticas de demostración mezcladas con datos operativos.

Los valores numéricos proceden de las fuentes canónicas documentadas en el artículo.
Los diagramas son elaboración propia y no requieren permiso de reproducción.
Las relaciones se trazan por carriles ortogonales externos a las cajas; el PDF del paper y el Beamer deben inspeccionarse después de cualquier cambio de texto o escala para comprobar que ninguna línea cubra nodos, rótulos o leyendas.

Para cada diagrama se ajustan conjuntamente fuente, ancho, alto y separación de cajas antes de recalcular las rutas ortogonales; no se aceptan cajas, textos o trayectorias superpuestos. Los gráficos con pocos modelos o métricas deben probarse a una columna, con separación reducida entre grupos y etiquetas y valores legibles. Si una tabla muy ancha exige letra pequeña, debe reducir columnas, partir encabezados y representar cada observación en dos renglones antes de considerar una reducción de escala.
