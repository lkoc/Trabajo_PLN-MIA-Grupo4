# Migración y compatibilidad

## Datos

La orden siguiente crea otro archivo y su manifiesto; rechaza como error usar la misma ruta de entrada y salida:

```powershell
modperu migrate origen.jsonl datos/model_ready/v2/dataset_5_salidas.jsonl datos/model_ready/v2/migracion.manifest.json
```

Reglas:

- `ACOSO_GENERO_IDENTIDAD` se acepta solo como nombre histórico de entrada y se proyecta a la salida canónica `ATAQUE_POR_GENERO_IDENTIDAD`;
- `ACOSO_PERSONAL` y `AMENAZA_DIRECTA` se proyectan a `ACOSO_AMENAZA`;
- `SEGURO` solo procede de `SEGURO` histórico explícito;
- vacío, desconocido o conflicto se envía a revisión y se excluye del entrenamiento;
- se preserva `legacy_coarse_labels` y la fuente original.
- cada fila requiere `video_id` explícito; si el histórico no lo contiene, primero se cruza con la tabla de chunks. Nunca se deduce partiendo `chunk_id`, porque un ID de YouTube puede contener `_`.

El alias histórico nunca aparece en predicciones, campañas, modelos o exportaciones nuevas. El cambio de nombre no altera automáticamente los rótulos fuente: la migración crea un snapshot nuevo y conserva el original para auditoría.

## Modelos

Los modelos anteriores pueden proporcionar encoder o backbone. No se reutilizan sus umbrales ni se presentan como modelos v2. La primera cabeza de cinco salidas se entrena desde una inicialización nueva; después, las ampliaciones reanudan una interrupción o usan *warm start* desde un candidato v2 compatible y entrenan con el snapshot completo anterior+nuevo.

## Frontend y eventos

Los eventos históricos permanecen en su formato original. Los nuevos eventos usan `ReviewEvent` v2. Una migración posterior debe conservar el evento original y escribir otro registro normalizado; nunca editar la bitácora append-only.
