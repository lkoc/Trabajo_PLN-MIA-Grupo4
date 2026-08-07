# Auditoría de citas de los cuadernos activos

Versión auditada: `2.1.0`  
Fecha: 2026-08-07

## Alcance y criterio

Se revisaron los 18 cuadernos activos de `flujo/`. Se aplicó la guía local
[`evidencia-citas-y-bibliografia.md`](../Guias_generales/redactar-articulo-ieee-y-presentacion/references/evidencia-citas-y-bibliografia.md):

- cita IEEE numérica junto a la afirmación que respalda;
- fuente primaria para algoritmos, métricas y estudios;
- documentación oficial para APIs, software y checkpoints exactos;
- separación explícita entre antecedentes externos y decisiones locales;
- celda final `Referencias` con solo las fuentes citadas en el cuaderno;
- tres autores seguidos de *et al.* cuando la obra tiene más de tres;
- ausencia de citas textuales y de referencias creadas para resultados propios.

Las claves fuente permanecen completas en
[`referencias.bib`](../Documento_final_paper/referencias.bib), aunque cada
cuaderno materializa una lista independiente numerada por primera aparición.

## Inventario

| Cuaderno | Entradas en la celda final |
|---|---:|
| `01_01_scraping_incremental.ipynb` | 7 |
| `01_02_optimizacion_longitud_chunks.ipynb` | 13 |
| `01_03_limpieza_troceado_incremental.ipynb` | 2 |
| `02_00_preparacion_bundle_colab.ipynb` | 2 |
| `02_01_etiquetado_local_ollama.ipynb` | 7 |
| `02_02_etiquetado_remoto.ipynb` | 2 |
| `02_03_revision_llm_dirigida.ipynb` | 4 |
| `02_04_consolidacion_validacion_humana.ipynb` | 3 |
| `02_05_cierre_humano_snapshot.ipynb` | 3 |
| `03_01_modelos_clasicos.ipynb` | 7 |
| `03_02_transformers_planos.ipynb` | 9 |
| `03_03_transformer_cascada.ipynb` | 5 |
| `03_04_transformer_multitarea.ipynb` | 5 |
| `03_05_qwen_lora.ipynb` | 6 |
| `03_06_qwen_estructurado.ipynb` | 6 |
| `03_07_comparacion_final.ipynb` | 5 |
| `03_08_auditoria_finas_flags.ipynb` | 10 |
| `04_01_frontend_produccion.ipynb` | 5 |
| **Total materializado** | **101** |

El total corresponde a 101 entradas materializadas en 18 bibliografías y 65
claves únicas de la bibliografía maestra. Una misma fuente puede reaparecer en
cuadernos distintos cuando sustenta componentes distintos del flujo.

## Correspondencia revisada

- Adquisición: `yt-dlp`, fallback de transcripciones, términos de plataforma,
  ética y sesgo del subtitulado automático.
- Preparación: Unicode NFKC y SHA-256; ventanas y deduplicación declaradas
  como configuración local. La elección de longitud separa validation de test,
  cita AP para desbalance y reutiliza la implementación clásica documentada.
- Etiquetado: documentación oficial de Ollama, DeepSeek, Qwen y Colab, junto
  con evidencia sobre anotación asistida, anclaje, contexto y acuerdo.
- Entrenamiento: fuente fundacional de cada familia realmente nombrada,
  tarjetas de los checkpoints exactos e implementaciones centrales.
- Evaluación: definición exacta de AP, precisión–recall, calibración y sesgo
  de selección; los falsos seguros y la carga de revisión se identifican como
  criterios propios.
- Taxonomía: antecedentes generales y evidencia peruana diferenciados de las
  fusiones, nombres, flags y umbrales locales.
- Producción: moderación semiautomática, rechazo, clasificación selectiva y
  deferencia; el modo sombra no se presenta como garantía de seguridad.

## Resultado automático

Comandos de cierre:

```powershell
python -m pytest
python tools/audit_project.py
```

Resultado: 102 pruebas aplicables aprobadas; 18/18 cuadernos con celda final; 101/101
entradas numeradas y citadas; cero claves ausentes en `referencias.bib`, cero
entradas finales sin uso, cero números discontinuos y cero incidencias del
auditor del proyecto. La única comprobación estructural no incluida en esa
corrida exige cuadernos sin salidas guardadas; `01_03` conserva deliberadamente
los resultados del usuario y no se limpió durante esta actualización.

El control automático verifica estructura y correspondencia de claves. La
correspondencia semántica afirmación–fuente se revisó manualmente y conserva los
límites documentados en
[`fuentes_base.md`](../bibliografia/fuentes_base.md) y
[`MATRIZ_EVIDENCIA_TAXONOMIA.md`](MATRIZ_EVIDENCIA_TAXONOMIA.md).
