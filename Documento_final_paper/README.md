# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Artículo IEEE actualizado

El manuscrito presenta el contrato v2.1 y los resultados vigentes de `03_07`/`03_07a`. Las versiones antiguas se mencionan únicamente como antecedentes de la mejora metodológica; sus métricas no se mezclan con la comparación actual.

Las cinco salidas aprendidas son `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`.

## Resultado principal

- Corpus: 173 240 chunks elegibles de 4 906 videos; 14 163 contienen al menos un daño.
- Comparación: 28 modelos individuales y cinco ensembles sobre una validación común de 10 600 filas.
- Selección congelada: `ensemble_soft_mean`, compuesto por regresión logística TF--IDF, cascada E5 v2 y Qwen3--0.6B con LoRA.
- Validation OOF: BA 0,84; macro-AUPRC de daños 0,55; macro-F1 de daños 0,57.
- Test natural abierto una vez: 22 684 chunks; BA 0,85 y sensibilidad 0,91 para cualquier daño.
- Política selectiva: 35 % a revisión; sobre el 65 % automático, BA 0,94.

El estado inferencial sigue siendo `statistical_tie_or_inconclusive`: el ensemble ocupa el primer lugar por el criterio predeclarado, pero el contraste pareado no demuestra superioridad estadística frente al retador más cercano.

## Archivos principales

- `paper_moderador_contenido_youtube_ieee.tex`: fuente principal IEEE A4.
- `paper_moderador_contenido_youtube_ieee.pdf`: PDF compilado.
- `secciones/`: resumen, problema, datos, modelos, resultados, discusión, conclusiones y anexos.
- `referencias.bib`: bibliografía común.
- `ontologia_moderacion.ttl`: vocabulario formal de trazabilidad.
- `figuras/`: diagramas, gráficos de `03_07a` y capturas de interfaces.
- `AUDITORIA_CITAS_Y_ESTILO.md`: auditoría integral vigente contra las guías generales y específicas.
- `AUDITORIA_ACTUALIZACION_20260815.md`: síntesis del cierre cuantitativo, editorial y visual.

## Fuentes de verdad

1. `../resultados/modelos/seleccion_congelada.json` para miembros, calibradores, umbrales y política de revisión.
2. `../resultados/modelos/comparacion_individual_ensemble_validation.json` para ranking, Pareto y bootstrap.
3. `../resultados/modelos/test_final_abierto_una_vez.json` para test natural y vista secundaria 4:1.
4. `../resultados/modelos/resumen_03_07a.json` y sus tablas para la síntesis editorial.
5. `../docs/artefactos/auditoria_estado_final_182461.json` para corpus, particiones y auditoría del etiquetado.

## Compilación

Desde esta carpeta:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error paper_moderador_contenido_youtube_ieee.tex
```

El cierre exige cero errores, citas o referencias indefinidas y cajas `Overfull`, además de inspección visual de todas las páginas.
