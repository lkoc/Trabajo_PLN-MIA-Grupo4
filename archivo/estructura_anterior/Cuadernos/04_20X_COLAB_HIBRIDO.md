# Ejecución híbrida local/Colab de `04_202`–`04_206`

> **Documento histórico preservado.** Describe una estructura, taxonomía o corrida anterior a `moderacion_peru_5_salidas_v2`. Sus resultados siguen siendo evidencia, pero sus rutas y salidas no definen el flujo activo. Consulte el README raíz y `archivo/README.md`.


## Arquitectura

Los cuadernos detectan automáticamente el entorno:

- **Kernel local:** usan directamente `D:\trabajo_PLN\Trabajo_PLN-MIA-Grupo4`.
- **Kernel de Colab desde VS Code o Colab:** montan Google Drive, crean en `/content/Trabajo_PLN-MIA-Grupo4` un checkout disperso de sólo `scripts_auxiliares` y fijan el código al commit `ce33b5efd797eef0809e9ff8694c90220cd81e9d`.
- En Colab, `datos`, `modelos` y `resultados` son enlaces simbólicos Linux hacia `/content/drive/MyDrive/PLN_colab_04_artifacts`. Por ello, checkpoints, métricas, figuras e informes sobreviven al cierre del runtime.

No se crea un enlace permanente entre `D:` y `G:`. Esa separación evita que Google Drive sincronice archivos de checkpoints mientras todavía están siendo escritos.

## 1. Preparar el paquete mínimo en Drive

Con Google Drive para escritorio activo y la unidad `G:` disponible, ejecutar desde la raíz del proyecto:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\sincronizar_04_20x_google_drive.ps1
```

El script copia únicamente los datasets congelados, referencias y checkpoints que comparten `04_202`–`04_206`. Cada archivo se comprueba con SHA-256 y queda registrado en:

`G:\My Drive\PLN_colab_04_artifacts\MANIFIESTO_ARTEFACTOS_04_20X.json`

Si existe un checkpoint reanudable de Qwen creado por `04_7`/`04_205`, se copia sólo el slot publicado por su puntero atómico, el puntero y el tokenizer. Se vuelve a comprobar que el puntero no haya cambiado durante la copia. Así `04_205` puede reanudarlo sin copiar el slot inactivo ni un archivo a medio escribir.

Si `04_7` ya produjo `finetuning.json`, la misma sincronización añade `best_adapter` y los resultados finales disponibles. Esto permite que `04_205` reconozca el entrenamiento como terminado y que `04_206` lo consuma sin repetirlo.

## 2. Ejecutar los cuadernos

En VS Code se puede conservar el archivo `.ipynb` local y seleccionar un kernel de Colab. La primera celda:

1. monta Drive;
2. valida que estén todos los archivos declarados en el manifiesto;
3. prepara el checkout reproducible en `/content`;
4. instala solamente las dependencias ausentes;
5. muestra el dispositivo y la ruta persistente.

En `04_205`, `qwen4.resume_status()` muestra el checkpoint disponible y `qwen4.run_finetuning(..., resume=True, force_restart=False)` lo recupera automáticamente. No ejecutar `04_205` en Colab mientras la sesión original `04_7` siga entrenando localmente; sincronizar la instantánea no detiene ni modifica `04_7`.

## 3. Recuperar resultados en el workspace local

Los resultados de Colab se guardan en Drive y aparecen en Windows bajo `G:\My Drive\PLN_colab_04_artifacts`; **no se copian implícitamente a `D:`**. Para llevar al workspace los resultados Transformer:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1
```

Para incluir Qwen, después de cerrar cualquier entrenamiento local que escriba sobre el mismo experimento:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1 -IncludeQwen
```

También existen alcances precisos para evitar copiar familias no relacionadas:

```powershell
# Sólo fine-tuning, calibración y adaptadores de 04_205
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1 -Qwen04_205Only

# Sólo cabezas, scores e informe de 04_206
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1 -Qwen04_206Only

# Conjunto completo 04_205 + 04_206
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1 -Qwen04_20XOnly

# Artefactos mínimos que necesita el servidor 05 (E5 ganador + Qwen operativo)
powershell -ExecutionPolicy Bypass -File .\scripts_auxiliares\recuperar_resultados_colab_04_20x.ps1 -DeploymentOnly
```

La recuperación compara SHA-256. Si encuentra un archivo local distinto, se detiene para impedir una sobrescritura silenciosa. `-Force` sólo debe usarse después de revisar el conflicto. Cada recuperación genera un registro en `resultados/logs/sincronizacion_colab_04_20x/`.

Una vez recuperado, el mismo código local encuentra los checkpoints en sus rutas habituales. Esto permite continuar entrenamiento o posprocesar sin repetirlo. `best_adapter` y el adaptador operativo no son necesariamente el mismo: en la ejecución actual, `best_adapter` es la época 2 por PR-AUC de validación y la selección operativa es `epoch_adapters/epoch_03`, elegida a igual objetivo de recall antes de test. Los consumidores deben leer `seleccion_operativa_validacion.json`, verificar hashes y conservar su estado; nunca deben usar un slot transitorio de reanudación.

`04_206` verifica y sincroniza `04_205` al arrancar si la copia local falta o es una extensión consistente. `04_207` y `04_208` son comparación y auditoría, no fine-tuning. `04_208` usa `-PreferNewest`: ante hashes diferentes compara primero `completed_at`, `generated_at`, `created_at`, `finished_at` o `updated_at` en JSON. Sólo acepta timestamps con `Z` u offset explícito y los normaliza a UTC; una fecha sin zona se considera ambigua y se usa `LastWriteTimeUtc`. Se conserva el más reciente, sea Drive o `D:`, y el log registra los archivos copiados y los locales omitidos por ser más nuevos. Un empate exacto con hashes diferentes se detiene para revisión manual.
