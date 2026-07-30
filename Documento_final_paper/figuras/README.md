# Figuras reproducibles del paper

Los diagramas del artículo son fuentes vectoriales TikZ/PGFPlots compiladas
desde LaTeX. Las dos capturas de interfaz son imágenes raster obtenidas de los
artefactos funcionales y se conservan junto con sus fuentes reproducibles.

- pipeline_moderacion.tex: adquisición, preanotación, adjudicación y entrenamiento.
- brechas_problema.tex: brecha real, subyacente y tecnológica.
- ciclo_dsr.tex: fases DSR y cronología de iteraciones.
- datos_taxonomia.tex: cuatro salidas activas, 12 fenómenos de daño, dos estados seguros y tres flags transversales en carriles separados.
- fuentes_dataset.tex: criterios de adquisición y volumen total del corpus integrado.
- familias_modelo.tex: familias y estructuras comparadas.
- resultados_finales.tex: métricas de los ganadores por familia.
- despliegue.tex: artefacto operativo, consenso 2 de 3 y retroalimentación.
- ontologia_trazabilidad.tex: grafo de conceptos y relaciones del anexo.
- captura_entorno_etiquetado.png: vista del entorno de revisión humana.
- captura_entorno_operacion.png: resultado de una inferencia real enviada por el formulario al servidor del paquete 05.
- captura_entorno_operacion.evidence.json: registro de la corrida usada en la captura, con estado del servidor, chunk de validación, modo, resultado visible y conteos antes/después.
- ../../scripts_auxiliares/capturar_figura9_ejecucion_real.py: automatización que abre la interfaz, envía el formulario, espera la inferencia y genera la captura sin inyectar resultados en el HTML.

Los valores numéricos proceden de las fuentes canónicas documentadas en el artículo.
Los diagramas son elaboración propia y no requieren permiso de reproducción.
Las relaciones se trazan por carriles ortogonales externos a las cajas; el PDF del paper y el Beamer deben inspeccionarse después de cualquier cambio de texto o escala para comprobar que ninguna línea cubra nodos, rótulos o leyendas.

Para cada diagrama se ajustan conjuntamente fuente, ancho, alto y separación de cajas antes de recalcular las rutas ortogonales; no se aceptan cajas, textos o trayectorias superpuestos. Los gráficos con pocos modelos o métricas deben probarse a una columna, con separación reducida entre grupos y etiquetas y valores legibles. Si una tabla muy ancha exige letra pequeña, debe reducir columnas, partir encabezados y representar cada observación en dos renglones antes de considerar una reducción de escala.
