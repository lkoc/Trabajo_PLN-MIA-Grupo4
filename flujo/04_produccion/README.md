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

La interfaz recupera de la versión histórica la entrada única para texto o YouTube, caché de subtítulos, troceado temporal, reproductor, resultados por chunk, revisión en línea, estadísticas y exportación. También conserva el selector del mejor clásico, mejor Transformer y mejor Qwen, la comparación de sus respuestas y el consenso mayoritario 2-de-3. Añade los cinco scores y umbrales aprendidos, contrato/modelo activo y razones de revisión. Un enlace de YouTube descarga solo subtítulos; nunca audio ni video. Si el texto o los subtítulos exceden el máximo de chunks, la solicitud se rechaza de forma explícita y no se trunca silenciosamente.

`03_07` selecciona exclusivamente con validation el mejor candidato de cada familia y publica un registro principal más tres registros miembro verificables. El modo `consensus` solo se habilita cuando los tres existen; cualquier desacuerdo, conflicto `SEGURO+daño`, ausencia de mayoría o cercanía a un umbral obliga revisión.

El frontend opera en modo sombra: no bloquea, elimina ni sanciona contenido. Cada inferencia y revisión queda local y append-only y se enlaza al evento exacto. Las estadísticas se separan por modelo y categoría; la retroalimentación concordante se deduplica en `datos/produccion/retraining_ready_v2.jsonl`, mientras los conflictos se excluyen. Si falta `modelos/registro_modelos_5_salidas.json`, no sustituye el modelo con resultados históricos.

Por defecto solo escucha en loopback. Para exponerlo deliberadamente en otra interfaz de red debe configurarse `MODERATOR_ACCESS_PASSWORD` y, opcionalmente, `MODERATOR_ACCESS_USER`; nunca se escriben esas credenciales en el repositorio. La [matriz de paridad](../../docs/PARIDAD_FRONTENDS_ACTIVOS.md) documenta cada función y el estado de las capturas actuales.
