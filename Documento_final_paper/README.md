# Documento final IEEE

Esta carpeta contiene el artículo final del proyecto **Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural**. El manuscrito presenta el artefacto ya construido y evaluado mediante Design Science Research (DSR): modelos compactos priorizan y diagnostican preliminarmente fragmentos problemáticos, y un supervisor humano conserva la decisión.

## Archivos

- `paper_moderador_contenido_youtube_ieee.tex`: fuente principal con clase `IEEEtran`, modo conferencia y hoja A4.
- `secciones/`: contenido modular del resumen, introduccion, bases, problema/objetivos, metodologia, resultados, discusion, conclusiones y anexos.
- `referencias.bib`: bibliografia comun en BibTeX y estilo numerico IEEE.
- `figuras/`: figuras publicables y sus fuentes reproducibles.
- `ontologia_moderacion.ttl`: vocabulario formal OWL/Turtle que enlaza video, subtitulo, chunk, anotacion, daño, prediccion, revision y artefacto.
- `guia_estructura_paper_ieee.md`: contenido obligatorio y relacion entre problemas, objetivos, metodo y resultados.
- `guia_redaccion_paper_ieee.md`: reglas de evidencia, citas y revision editorial.
- `AUDITORIA_CITAS_Y_ESTILO.md`: cierre comprobable de citas, compilación y revisión visual.
- `../para_equiquetado_LLM/AUDITORIA_ACADEMICA_TAXONOMIA.md`: inventario, definiciones, sustento y contradicciones del paquete de etiquetado.

## Datos de autoría

La forma canónica de autoría es: Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano. Todos comparten la afiliación Maestría en Inteligencia Artificial, Universidad Nacional de Ingeniería, Lima, Perú. La forma abreviada para una plantilla que admita encabezado corto es “L. Koc, A. Mancilla, H. Meléndez y D. Paitán”.

Los correos registrados son `luis.koc@gmail.com`, `amancillaa@uni.pe`, `hamg.94@gmail.com` y `dennis.paitan.c@uni.pe`. La clase `IEEEtran` los presenta en el bloque común de afiliación; no admite directamente los comandos `\shortauthors`, `\fnmark` y `\fntext` de otras plantillas. No se incorporan los valores `auid=000`, porque son marcadores sin un identificador de autor verificable.

El `.tex` y `referencias.bib` son las fuentes del entregable. El PDF se considera un producto derivado y debe regenerarse despues de cualquier cambio.

La revisión cerrada el 29 de julio de 2026 contiene 117 claves citadas y 117 entradas BibTeX, sin claves ausentes ni entradas sin uso. `../referencias_y_descargas/` conserva el índice de las fuentes y 81 PDF oficiales o de acceso abierto validados; las 36 fuentes restantes se documentan mediante su página canónica o la razón por la que no existe una copia local apropiada.

El detalle visual del corpus aparece en la metodología, los resultados y los anexos: total integrado, embudo Flash--Pro--revisión humana final, tabla por categoría de fuente, procedencia de etiquetas y diagrama de selección. El cuerpo reporta el corpus como una sola unidad y evita desglosar su tamaño por campañas. Las cifras se recalculan desde el JSONL integrado y sus manifiestos; no deben sustituirse por totales de informes intermedios. La revisión humana se describe por su función, cantidad y procedencia, sin trasladar identificadores internos al artículo.

## Fuentes de verdad

No se deben reconstruir cifras desde la memoria ni desde salidas antiguas incrustadas en cuadernos. Use, en este orden:

1. `README.md`, `Cuadernos/README.md` y `datos/README.md` para el alcance y el contrato activo.
2. `Cuadernos/04_MATRIZ_ENTRENAMIENTO_4_ETIQUETAS.md` y `Cuadernos/04_200_ORDEN_EJECUCION.md` para taxonomia, dependencias y separacion entre experimentos historicos y activos.
3. `resultados/metricas/comparacion_final_4/comparacion_todos_modelos_4.csv` para la comparacion comun final.
4. `resultados/metricas/comparacion_final_4/registro_modelos_desplegables.json` para los modelos publicados por familia y sus hashes.
5. Los informes `resultados/INFORME_*.md` y sus JSON/CSV asociados para configuracion, resultados por experimento y limitaciones.
6. `Cuadernos/05_MODO_OPERACION.md` para el artefacto desplegable, consenso 2 de 3, revision humana y persistencia.

Los documentos historicos de cinco daños deben identificarse como iteraciones anteriores. No pueden usarse para cambiar la seleccion activa de cuatro daños ni compararse como si hubieran usado el mismo conjunto de prueba.

## Compilacion reproducible

Desde la raiz del repositorio:

```powershell
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -file-line-error Documento_final_paper/paper_moderador_contenido_youtube_ieee.tex
```

O desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error paper_moderador_contenido_youtube_ieee.tex
```

`latexmk` ejecuta automaticamente las pasadas necesarias de LaTeX y BibTeX. Una compilacion final aceptable debe satisfacer:

- hoja física A4 de 210 × 297 mm, generada con la opción de clase `a4paper`;
- cero referencias o citas indefinidas;
- cero errores y cero cajas `Overfull` sin justificar;
- todos los graficos presentes, legibles y citados en el texto;
- bibliografia completa y numerada en orden de aparicion;
- PDF generado despues del `.tex`, `.bib` y figuras que lo originan.

Revise el archivo `paper_moderador_contenido_youtube_ieee.log`; los avisos `Underfull` requieren inspeccion visual, aunque no todos implican un defecto.

## Afirmaciones que deben evitarse

- No afirmar que se uso Whisper o ASR local si no existe una ejecucion y un artefacto verificables. El modo operativo 05 rechaza enlaces sin subtitulos descargables.
- No describir todo el flujo como “sin APIs”. Debe distinguirse la recoleccion sin la API oficial de YouTube del pseudoetiquetado autorizado mediante modelos externos y de la inferencia operativa local.
- No afirmar doble anotacion ni reportar Cohen kappa: la adjudicacion disponible no permite estimarlo.
- No llamar “verdad de terreno humana independiente” al test actual, compuesto principalmente por etiquetas asistidas por LLM.
- Presentar primero el logro validado: priorización semiautomática, diagnóstico preliminar y evidencia temporal para un supervisor con recursos asequibles. Después delimitar que el bloqueo o la sanción sin revisión pertenecen a otro alcance.

## Cierre en dos revisiones

1. **Revision cientifica y de trazabilidad:** comprobar cada cifra contra su artefacto; separar seleccion en validacion de evaluacion en test; declarar pseudoetiquetado, balance 4:1, reutilizacion historica del test y demas limitaciones; verificar internamente que el cierre responde al objetivo general y a cada objetivo específico, aunque la prosa publicada no use «objetivo» ni códigos `O1`/`O2`; clasificar cada distancia inicial como eliminada, reducida o pendiente y ligar cada pendiente con una acción posterior.
2. **Revision editorial y visual:** comprobar estilo IEEE, ortografia, nombres oficiales de autores e institucion, citas, captions, legibilidad en dos columnas y compilacion limpia. Leer tambien el PDF completo, no solo el fuente.

La declaración publicada de disponibilidad debe indicar qué datos, cuadernos, scripts y artefactos están disponibles y en qué repositorios. Los SHA, commits y hashes concretos pertenecen a manifiestos técnicos fuera del cuerpo.

El manuscrito solo se considera cerrado cuando supera ambas revisiones y el Beamer se vuelve a alinear con su version final.
