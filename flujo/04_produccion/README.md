# Etapa 04 · Producción supervisada

`04_01_frontend_produccion.ipynb` comprueba el registro de cinco salidas e inicia un servidor exclusivamente local:

```powershell
modperu serve-production --host 127.0.0.1 --port 8765
```

La interfaz recupera de la versión histórica la entrada única para texto o YouTube, caché de subtítulos, troceado temporal, reproductor, resultados por chunk, revisión en línea, estadísticas y exportación. Añade los cinco scores y umbrales aprendidos, contrato/modelo activo y razones de revisión. Un enlace de YouTube descarga solo subtítulos; nunca audio ni video.

El frontend opera en modo sombra: no bloquea, elimina ni sanciona contenido. Cada inferencia y revisión queda local y append-only. Conflicto `SEGURO+daño`, ausencia de salida o cercanía a un umbral obliga revisión. Si falta `modelos/registro_modelos_5_salidas.json`, no sustituye el modelo con resultados históricos.
