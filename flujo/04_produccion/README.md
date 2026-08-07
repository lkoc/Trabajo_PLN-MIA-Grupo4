# Moderación semiautomática de videos peruanos de YouTube mediante modelos clásicos y neuronales de procesamiento del lenguaje natural

**Trabajo final del curso de Procesamiento de Lenguaje Natural (PLN) de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI) — Semestre 2026-1**

**Grupo 4:** Luis Enrique Koc Góngora, Alex Felipe Mancilla Antay, Herbert Antonio Meléndez García y Dennis Jack Paitán Cano

---

## Etapa 04 · Producción supervisada

**Contrato de etiquetas v2.1:** cinco salidas entrenadas: `SEGURO`, `RACISMO_DISCRIMINACION`, `ATAQUE_POR_GENERO_IDENTIDAD`, `ACOSO_AMENAZA` y `CONTENIDO_SEXUAL`. `SEGURO` es excluyente; las cuatro categorías de daño son multietiqueta y pueden coexistir. Los casos indeterminados se difieren y no entran al entrenamiento. Esta combinación, sus umbrales y sus reglas de exclusividad son decisiones operativas locales.

`04_01_frontend_produccion.ipynb` comprueba el registro de cinco salidas e inicia un servidor exclusivamente local:

```powershell
modperu serve-production --host 127.0.0.1 --port 8765
```

La interfaz recupera de la versión histórica la entrada única para texto o YouTube, caché de subtítulos, troceado temporal, reproductor, resultados por chunk, revisión en línea, estadísticas y exportación. Añade los cinco scores y umbrales aprendidos, contrato/modelo activo y razones de revisión. Un enlace de YouTube descarga solo subtítulos; nunca audio ni video.

El frontend opera en modo sombra: no bloquea, elimina ni sanciona contenido. Cada inferencia y revisión queda local y append-only. Conflicto `SEGURO+daño`, ausencia de salida o cercanía a un umbral obliga revisión. Si falta `modelos/registro_modelos_5_salidas.json`, no sustituye el modelo con resultados históricos.
