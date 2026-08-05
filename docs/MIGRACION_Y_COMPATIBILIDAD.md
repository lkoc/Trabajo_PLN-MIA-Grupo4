# Migración y compatibilidad

## Datos

La orden siguiente crea otro archivo y su manifiesto; rechaza como error usar la misma ruta de entrada y salida:

```powershell
modperu migrate origen.jsonl datos/model_ready/v2/dataset_5_salidas.jsonl datos/model_ready/v2/migracion.manifest.json
```

Reglas:

- `ACOSO_PERSONAL` y `AMENAZA_DIRECTA` se proyectan a `ACOSO_AMENAZA`;
- `SEGURO` solo procede de `SEGURO` histórico explícito;
- vacío, desconocido o conflicto se envía a revisión y se excluye del entrenamiento;
- se preserva `legacy_coarse_labels` y la fuente original.

## Modelos

Los modelos anteriores pueden proporcionar encoder o backbone. No se reutilizan sus umbrales ni se presentan como modelos v2. La primera cabeza de cinco salidas se entrena desde una inicialización nueva; después, las ampliaciones pueden reanudar desde el checkpoint v2 y mezclar el lote nuevo con datos anteriores.

## Frontend y eventos

Los eventos históricos permanecen en su formato original. Los nuevos eventos usan `ReviewEvent` v2. Una migración posterior debe conservar el evento original y escribir otro registro normalizado; nunca editar la bitácora append-only.

