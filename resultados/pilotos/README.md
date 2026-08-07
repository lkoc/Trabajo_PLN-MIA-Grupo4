# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Pilotos locales

Los pilotos de Ollama se guardan aquí. Una muestra menor de 200 casos es un smoke test técnico y no selecciona el modelo semántico. Las referencias humanas históricas estuvieron asistidas y no constituyen gold standard ciego.

El 5 de agosto de 2026 se verificaron instalados los tres modelos y se ejecutó el smoke test acotado documentado en `INFORME_PILOTO_OLLAMA_20260805.md`. Qwen 9B fue el único técnicamente válido en el caso común; el tamaño no permite selección ni métricas semánticas comparables. El resultado vigente termina en `_bounded.json` y dispone de manifiesto SHA-256; los archivos anteriores se conservan como diagnósticos históricos.

## Longitud de chunks

`01_02_optimizacion_longitud_chunks.ipynb` contiene perfiles CPU rápido,
confirmatorio y robusto. Todos
vuelven a trocear, entrenan desde cero, calibran en `validation` e infieren en
`validation` y `test` para cada longitud; ningún modelo entrenado a 30 s se usa
para evaluar otra longitud.

El perfil neuronal robusto posterior compara también 15, 20, 25, 30 y 35 s.
Usa el mismo panel pareado de 100 anclas de `validation` en las dos familias y
mantiene sus papeles separados: MiniLM congelado con 25 cabezas logísticas
evalúa sensibilidad de representación; `gemma3:4b` solicita hasta 500 salidas
estructuradas con el prompt operativo v2 y evalúa sensibilidad semántica y
factibilidad. Ambas familias calculan 2 000 réplicas bootstrap agrupadas por
video. Sus métricas no son intercambiables y no se promedian.

Los checkpoints voluminosos permanecen locales. Git conserva únicamente los
artefactos reportables: `paired_validation_panel_manifest.json`,
`minilm/minilm_robust_comparison.json`,
`ollama/ollama_robust_comparison.json`, `hierarchical_synthesis.json` y
`neural_robust_comparison.json`, todos bajo `chunk_length/neural_robust/`.
La interpretación con citas y las amenazas de validez están en
[`docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md`](../../docs/ROBUSTEZ_NEURONAL_LONGITUD_CHUNKS.md).
El antiguo `neural_smoke_comparison.json` queda como piloto preliminar y no se
mezcla con esta comparación final.

La corrida final completó 25 cabezas MiniLM y 500 intentos Ollama. MiniLM dejó
la comparación inconclusa: 20 s obtuvo la mayor AP puntual, 0.59, pero su IC de
diferencia frente a 30 s incluyó cero; solo 35 s quedó significativamente por
debajo. Ollama obtuvo 474 salidas válidas y 26 fallos; la compuerta de 0.95
falló para 15, 20 y 25 s. La mayor F1 puntual fue 0.42 para 30 s, seguida por
0.40 para 35 s. La síntesis no combina esas métricas y conserva 30 s.

El piloto rápido de una cohorte y dos modelos sugirió 35 s, pero su resultado
fue inestable y no se acepta como decisión. La confirmación ampliada usó tres
cohortes pareadas, 200/80/80 videos por split, tres modelos y transferencia de
etiqueta solo cuando todos los chunks temporales solapados concordaban.

| Segundos | AP daño validation, media ± DE | Victorias | Proxy de costo medio |
|---:|---:|---:|---:|
| 15 | 0.0784 ± 0.0035 | 0/3 | 52 106 |
| 20 | 0.0895 ± 0.0075 | 0/3 | 39 855 |
| 25 | 0.0688 ± 0.0056 | 0/3 | 31 881 |
| **30** | **0.1142 ± 0.0050** | **3/3** | **30 181** |
| 35 | 0.0554 ± 0.0057 | 0/3 | 22 809 |

La recomendación confirmatoria del 6 de agosto de 2026 es **conservar 30 s**.
35 s es más barato, pero su pérdida de 0.0588 AP excede la tolerancia absoluta
de 0.01. `test` no intervino en la selección. Estos valores siguen siendo
proxy y no sustituyen la evaluación productiva del flujo completo.

El perfil robusto del 7 de agosto completó cinco cohortes de 300/100/100 videos,
75 ajustes y 1 000 réplicas bootstrap agrupadas por `video_id` en 838.5 s. Con
30 s como referencia y margen de no inferioridad de 0.01 AP, obtuvo AP `0.1233`
e IC 95% `[0.1099, 0.1446]`; ninguna otra longitud fue no inferior. `test` no
participó en la selección. Los resultados están en
`chunk_length/robust_30min/robust_comparison.json` y
`robust_recommendation.json`.

El resultado completo, las semillas, cohortes y métricas por modelo se
sincronizan en `chunk_length_expanded/confirmatory_comparison.json`; su
interpretación para el paper y la presentación está en
`docs/OPTIMIZACION_LONGITUD_CHUNKS.md`.
