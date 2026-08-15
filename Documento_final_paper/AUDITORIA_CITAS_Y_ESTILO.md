# Auditoría integral del artículo contra las guías aplicables

Fecha de cierre: 2026-08-15

Versión evaluada: contrato v2.1, resultados `03_07`/`03_07a`

Resultado: **aprobado con las cautelas estadísticas y operativas declaradas en el propio artículo**.

## 1. Normativa aplicada

La auditoría utilizó la jerarquía y el alcance definidos por:

- `../Guias_generales/INDICE_GUIAS.md` y `PROCEDENCIA_GUIAS_PROYECTO.md`;
- `../Guias_generales/redactar-articulo-ieee-y-presentacion/SKILL.md`;
- las referencias generales `contrato-y-trazabilidad.md`, `estructura-y-redaccion-ieee.md`, `metodologia-resultados-y-validez.md`, `evidencia-citas-y-bibliografia.md`, `figuras-tablas-y-ontologias.md` y `auditoria-y-entrega.md`;
- `guia_estructura_paper_ieee.md` y `guia_redaccion_paper_ieee.md`;
- el inventario bibliográfico `../referencias_y_descargas/indice_referencias.csv` y las instrucciones de `figuras/README.md`.

`busqueda-bibliografica-profunda.md` no se activó como protocolo completo porque este cierre no solicitó una nueva revisión sistemática; sí se volvieron a contrastar las fuentes primarias que sostienen el modelo Qwen y la comparación cuantitativa más visible. `presentacion-academica.md` se controla por separado en `../Presentacion_BEAMER_resultados_20260815/AUDITORIA_FINAL_VISUAL.md` y `AUDITORIA_FINAL_REFERENCIAS.md`.

Las enmiendas específicas del 5 de agosto describían correctamente un estado histórico en el que las métricas v2 estaban pendientes. Ambas guías recibieron una enmienda final v2.1 para dejar explícito que los artefactos canónicos actuales sustituyen esa cláusula en el manuscrito vigente.

## 2. Fuentes de verdad y trazabilidad

Se aplicó la prioridad `artefacto canónico del run > síntesis 03_07a > documentos históricos`. Las cifras del manuscrito se cotejaron con:

1. `../resultados/modelos/seleccion_congelada.json`;
2. `../resultados/modelos/comparacion_individual_ensemble_validation.json`;
3. `../resultados/modelos/test_final_abierto_una_vez.json`;
4. `../resultados/modelos/resumen_03_07a.json` y sus tablas;
5. `../docs/artefactos/auditoria_estado_final_182461.json`.

Controles reproducidos:

- corpus: 173 240 chunks elegibles, 4 906 videos y 14 163 chunks con daño;
- comparación: 28 candidatos individuales y cinco ensembles sobre 10 600 filas de validation;
- selección: `ensemble_soft_mean`, con clásico, cascada E5 v2 y Qwen3--LoRA;
- validation OOF: BA 0,84, macro-AUPRC de daños 0,55 y macro-F1 de daños 0,57;
- contraste: 2 000 réplicas pareadas por video; el IC95 % incluye cero y `winner_status=statistical_tie_or_inconclusive`;
- test: una apertura y una inferencia sobre 22 684 filas; la vista 4:1 reutiliza esas predicciones;
- compuerta congelada en test natural: 1 632 verdaderos positivos, 170 falsos negativos, 4 466 falsos positivos y 16 416 verdaderos negativos; BA 0,85 y sensibilidad 0,91;
- política selectiva: 35 % a revisión; cobertura automática 65 % y BA automática 0,94.

Los conteos, identificadores, hashes y parámetros de protocolo se conservan exactos. Las métricas y magnitudes estimadas se publican con dos cifras significativas; los calibradores, umbrales y pesos del apéndice conservan la precisión necesaria para reconstruir el ensemble.

## 3. Auditoría de citas y bibliografía

La compilación efectiva contiene 18 fuentes TeX, 118 comandos de cita y 193 apariciones de claves. El resultado bibliográfico es:

- 99 referencias numeradas en el PDF, más una entrada de control `IEEEtranBSTCTL`;
- 141 registros en `referencias.bib`;
- 41 registros inactivos que no se emiten en el PDF;
- 0 claves citadas sin entrada BibTeX;
- 0 claves BibTeX duplicadas;
- 0 citas o referencias cruzadas indefinidas;
- 31 etiquetas, sin duplicados, y 0 etiquetas sin referencia entrante.

El inventario de procedencia contiene 121 fuentes: 81 PDF de acceso abierto validados y presentes localmente, y 40 recursos sin PDF local con estado y justificación explícitos. No se interpreta la disponibilidad de un PDF como prueba de que toda tarea externa sea comparable con el corpus propio.

Se reabrieron las fuentes primarias de los números comparativos más visibles:

- EXIST 2021: F1 0,7944 del primer sistema en español;
- OffendES: macro-F1 78,39 %;
- HatEval: macro-F1 máximo 0,7300 en español;
- HateXplain: macro-F1 0,687 para BERT-HateXplain;
- DETOXIS: F-measure 0,6461 del primer sistema.

El texto redondea esas cifras a 0,79, 0,78, 0,73, 0,69 y 0,65, respectivamente. La comparación se mantiene como contextual: cambian corpus, idioma, unidad, taxonomía, partición y prevalencia. El informe técnico de Qwen3 y la tarjeta versionada de `Qwen/Qwen3-0.6B-Base` respaldan la familia y el checkpoint; el artículo distingue esos documentos de la evidencia empírica del proyecto.

## 4. Matriz de hallazgos y correcciones

| Ubicación | Hallazgo | Severidad | Evidencia o regla | Corrección | Estado |
|---|---|---:|---|---|---|
| Guías específicas | La enmienda histórica aún decía que v2 no tenía métricas. | Alta | Jerarquía de fuentes y artefactos `03_07`/`03_07a`. | Se añadió una enmienda final v2.1 sin borrar el antecedente. | Cerrado |
| Cuerpo y tablas | Varias métricas conservaban tres o más cifras aunque la guía exige dos significativas. | Media | Guía de redacción. | Se redondeó la presentación editorial; parámetros reproducibles siguen exactos. | Cerrado |
| Conclusiones | La redacción anterior podía leerse como lista de cumplimiento de objetivos. | Alta | Guía de cierre narrativo. | Se reescribió como síntesis de brechas reducidas, evidencia, límites y siguiente paso. | Cerrado |
| Figuras, tablas y anexos | Faltaban algunas llamadas explícitas desde el texto. | Media | Regla de referencia entrante. | Todas las 31 etiquetas reciben una referencia desde el discurso. | Cerrado |
| Anexos | Algunos anexos compartían página y se debilitaba la jerarquía editorial. | Media | Guía específica de anexos. | Los apéndices A--E comienzan en página nueva; D usa dos páginas por sus dos capturas. | Cerrado |
| Transición a anexos | Una frase aislada ocupaba casi toda una página. | Baja | Auditoría visual de todas las páginas. | Se compactó disponibilidad y ética sin retirar contenido ni citas. | Cerrado |
| Producción | La documentación previa aún marcaba el Transformer como pendiente. | Alta | Verificación funcional exigida para afirmar integración. | Se restauraron y verificaron los pesos; los cuatro modos respondieron por API local sobre CPU. | Cerrado |
| Estado del arte | Un primer lugar interno podía confundirse con superioridad universal. | Alta | Validez externa y bootstrap inconcluso. | Se mantiene “competitividad contextual” y se explicitan las diferencias experimentales. | Cerrado |
| Bibliografía maestra | 41 entradas no se citan en esta versión. | Baja | BibTeX solo emite las claves usadas. | Se conservan como catálogo compartido; ninguna aparece como referencia huérfana en el PDF. | Aceptado |
| Registro LaTeX | Persisten avisos `Underfull` por URL e identificadores largos. | Baja | No implican recorte. | Inspección visual confirmó márgenes y legibilidad; no hay `Overfull`. | Aceptado |

## 5. Revisión científica y editorial

- El resumen es autocontenido, declara método, muestra, resultados, conclusión de uso y límite operacional.
- El problema se expone en los niveles real, subyacente y tecnológico.
- Datos, particiones, pseudoetiquetado, revisión humana y límites de validez están separados de resultados.
- Los 28 modelos y cinco ensembles se describen con atención especial a los tres miembros ganadores y al promedio suave.
- La evaluación distingue validation OOF, test natural primario y vista 4:1 secundaria; test no modifica selección, calibradores ni umbrales.
- El contrato aprende las cinco salidas literales `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`; `SEGURO` es excluyente y no se presenta como mero complemento.
- La salida Qwen se describe correctamente como cabeza clasificadora de 22 logits; la taxonomía Markdown y el JSON estructurado pertenecen al proceso de anotación, no a esta inferencia.
- El término “ganador” se limita al orden de la regla predeclarada; la inferencia estadística permanece inconclusa.
- Las conclusiones son cualitativas y cuantitativas, responden a las brechas del trabajo y conservan límites sobre capacidad, doble anotación, deriva, privacidad y ausencia de sanción automática.

## 6. Revisión visual y compilación

- Documento `IEEEtran` en modo conferencia, papel A4 físico: 595,276 × 841,89 pt (210 × 297 mm).
- PDF final: 20 páginas.
- 0 errores fatales, 0 citas indefinidas, 0 referencias indefinidas y 0 cajas `Overfull`.
- Portada, texto a dos columnas, 15 tablas, 11 figuras, fórmulas, apéndices y 99 referencias fueron inspeccionados tras rasterizar las 20 páginas.
- Las tablas del ensemble y de resultados ampliados son legibles a tamaño normal.
- Las capturas de etiquetado y producción ocupan el ancho útil y conservan texto reconocible.
- Los apéndices y la bibliografía no contienen páginas vacías ni elementos cortados.

## 7. Cautelas que permanecen

No son defectos editoriales sino límites del estudio: `Natural` no equivale a una muestra probabilística de YouTube; falta una cohorte con doble anotación humana independiente; el margen de no inferioridad 0,10 es permisivo; la revisión de 35 % aún debe traducirse en horas y costo; y no se evaluaron imagen, audio, red ni deriva temporal. Estas cautelas aparecen en el artículo y evitan extrapolar el resultado más allá de la evidencia.
