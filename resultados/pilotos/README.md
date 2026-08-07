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

El cuaderno también ofrece dos diagnósticos neuronales desactivados por defecto.
MiniLM multilingüe se usa como encoder congelado con *mean pooling* y una cabeza
logística pequeña sobre 120/40 filas. `gemma3:4b` procesa tres filas por longitud,
persiste cada respuesta para reanudar y se detiene al alcanzar diez minutos. Sus
etiquetas duras no son directamente comparables con la AP continua de MiniLM y
ninguno de estos diagnósticos modifica la recomendación confirmatoria.

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
