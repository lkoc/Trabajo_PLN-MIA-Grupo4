# Auditoría de citas y referencias de `01_02`

**Fecha:** 7 de agosto de 2026  
**Estado del estudio:** ejecución completa  
**Resultado de auditoría:** aprobado sin incidencias abiertas

## Alcance

Se revisaron:

- `flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb`;
- `docs/OPTIMIZACION_LONGITUD_CHUNKS.md`;
- `docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md`;
- `README.md`, `flujo/01_datos/README.md` y
  `resultados/pilotos/README.md`;
- los cinco resúmenes JSON reportables del perfil neuronal y los dos artefactos
  reportables del contraste MiniLM 20 s–30 s;
- `tools/notebook_references.py` y
  `Documento_final_paper/referencias.bib` como fuentes bibliográficas maestras.

Los README cumplen una función operativa y enlazan los informes metodológicos
con aparato bibliográfico completo; no duplican una segunda lista de
referencias que pueda divergir.

## Conteos

| Entregable | Menciones de cita | Entradas | Citas sin entrada | Entradas sin uso | Duplicados |
|---|---:|---:|---:|---:|---:|
| Cuaderno `01_02` | 13 | 13 | 0 | 0 | 0 |
| Informe neuronal | 13 | 13 | 0 | 0 | 0 |
| Informe general de optimización | 26 | 15 | 0 | 0 | 0 |

El auditor integral contabiliza 99 menciones en los 17 cuadernos activos; los
17 terminan con una celda IEEE numerada y no conservan marcadores bibliográficos
sin resolver. La bibliografía maestra contiene todas las claves usadas por
`01_02`.

## Correspondencia afirmación–fuente

| Uso | Fuente comprobada | Estado |
|---|---|---|
| separación entre selección y evaluación | Cawley y Talbot, 2010 | pertinente |
| AP ante desbalance | Saito y Rehmsmeier, 2015 | pertinente |
| TF-IDF y estimadores clásicos | Salton y Buckley, 1988; Pedregosa *et al*., 2011 | pertinente |
| familia MiniLM y transferencia multilingüe | Wang *et al*., 2020; Reimers y Gurevych, 2020 | pertinente |
| checkpoint MiniLM exacto | tarjeta oficial de Hugging Face | pertinente |
| `gemma3:4b` y salida estructurada | biblioteca y documentación oficial de Ollama | pertinente |
| bootstrap y dependencia agrupada | Efron, 1979; Field y Welsh, 2007 | pertinente |
| hipótesis direccional de no inferioridad | Blackwelder, 1982 | pertinente |
| evaluación estadística pareada en PLN | Dror *et al*., 2018 | pertinente |
| F1 y métricas multietiqueta | Sokolova y Lapalme, 2009 | pertinente |
| cautela con anotación asistida por LLM | Schroeder, Roy y Kabbara, 2025 | pertinente |

La estrategia de muestreo, las cuotas, las semillas, los márgenes de no
inferioridad, la penalización de JSON inválido y la jerarquía entre familias se
identifican como decisiones locales; las fuentes externas no reciben autoría
sobre ellas.

El cálculo de 686 videos para 80% de potencia y 919 para 90% se conserva como
aproximación local de planificación basada en la diferencia y el error estándar
del piloto. No se atribuye a Blackwelder: esa fuente sustenta la dirección de
las hipótesis. La ejecución posterior usó 750 videos, predicción fuera de
pliegue y bootstrap pareado agrupado por `video_id`.

## Trazabilidad de resultados propios

| Afirmación | Artefacto y campo |
|---|---|
| panel de 100 anclas y 93 videos | `paired_validation_panel_manifest.json`: `anchors`, `distinct_videos` |
| 25 ajustes MiniLM | `minilm_robust_comparison.json`: `design.fits` |
| AP, intervalos y diferencias MiniLM | mismo artefacto: `bootstrap.comparisons` |
| 474/500 salidas Ollama válidas | `ollama_robust_comparison.json`: `duration_results` |
| F1, intervalos y diferencias Ollama | mismo artefacto: `bootstrap.comparisons` |
| fallo de la compuerta de esquema | mismo artefacto: `interpretation.schema_gate_passed` |
| conservación de 30 s | `hierarchical_synthesis.json`: `final_recommended_seconds` |
| ausencia de agregación entre familias | mismo artefacto: `metric_aggregation_across_families` |
| exclusión de `test` | resúmenes: `test_used` o `test_used_for_selection` |
| panel cross-fit de 750 videos | `paired_train_crossfit_panel_manifest.json`: `anchors`, `distinct_videos`, `anchors_per_crossfit_fold` |
| 30 ajustes sin fuga directa de video | resúmenes locales: `video_leakage_count`; resultado: `design.fits` |
| no inferioridad MiniLM 20 s–30 s | `minilm_20_30_noninferiority.json`: `interpretation` y `bootstrap.comparisons` |

Los conteos exactos, versiones, parámetros fijados y hashes permanecen sin
redondear en los artefactos. El informe presenta métricas estimadas con dos
cifras significativas y conserva la precisión completa para cálculos y
selección.

## Validaciones ejecutadas

```powershell
python -m pytest -q
python tools/audit_project.py
python tools/generate_workflow_notebooks.py --only flujo/01_datos/01_01_scraping_incremental.ipynb flujo/01_datos/01_02_optimizacion_longitud_chunks.ipynb
git diff --check
```

La prueba específica del informe exige las referencias `[1]`–`[13]` en orden,
sin claves ausentes, entradas no usadas ni duplicados. La prueba estructural del
cuaderno verifica además que la celda final de referencias coincida con las
claves IEEE registradas en sus metadatos.

## Excepciones y pendientes

No quedan citas críticas, altas, medias ni bajas pendientes dentro del alcance.
Los README incluidos en checkpoints de `modelos/` son metadatos generados por
bibliotecas externas y se excluyen de la auditoría de carátula; no son puntos de
entrada ni documentación académica del proyecto.
