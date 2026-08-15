# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Resultados del contrato v2

Este directorio recibirá únicamente métricas, figuras e informes producidos con `moderacion_peru_5_salidas_v2`: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; los cuatro daños son multietiqueta. Los casos indeterminados se difieren y no entran al entrenamiento. Toda corrida debe guardar manifiesto, hashes, split, cinco umbrales y evidencia de que test no seleccionó el modelo.

Los resultados ejecutados anteriormente están en `archivo/contrato_4_danos_seguro_derivado/resultados` y no son comparables directamente con el nuevo contrato.

`colab_bundle/` contiene la copia local verificable de los nueve archivos del bundle reproducible que `02_00` puede publicar como una versión inmutable en la carpeta privada `ModeracionPeru_Colab/bundle_releases` de Google Drive. No contiene modelos ni resultados de entrenamiento.

[`ETIQUETADO_CASCADA_CORTE_2026-08-08.md`](ETIQUETADO_CASCADA_CORTE_2026-08-08.md)
documenta un checkpoint atómico de la campaña Flash→Pro: recuperación
histórica, calibración, tiempo, velocidad, caché, costo y límites de la medida de
acuerdo. Es un corte parcial documentado; los `*.result.json` finales deben
reemplazar sus proyecciones al concluir `02_01`.

`modelos/comparacion_individual_ensemble_validation.json` es el checkpoint firmado de `03_07`: documenta el ranking completo de *validation*, la frontera Pareto, salvaguardas y pruebas pareadas. `modelos/seleccion_congelada.json` conserva la decisión compatible. `03_07a` consulta Drive mediante OAuth de solo lectura, compara `published_at` y SHA-256 con `modelos/sincronizacion_google_drive_03_07.json`, y solo entonces actualiza el área temporal `sincronizados/03_07/`. Del TAR remoto extrae únicamente comparación, selección y test: no Drive Desktop, pesos ni checkpoints. Después genera `modelos/REPORTE_COMPARACION_MODELOS_03_07.md`, sus CSV bajo `modelos/tablas_03_07a/` y sus PNG bajo `modelos/figuras_03_07a/`. Test solo aparece cuando exista `test_final_abierto_una_vez.json` con la misma firma.

El checkpoint vigente compara 28 individuos y 5 *ensembles*, selecciona `ensemble_soft_mean` y conserva `winner_status=statistical_tie_or_inconclusive`; por tanto, el orden operacional no se presenta como superioridad estadística. `auditorias/auditoria_finas_flags_v2.json` registra cobertura auxiliar por SHA-256 sin atribuir métricas predictivas a finas/flags cuando no existen predicciones gold separadas.
